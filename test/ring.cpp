/**
 *  @file test/ring.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Tests for @c basic_ring: capacity rules, first-in first-out order across wraparound,
 *      eviction and lifetimes.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <limits>  // `std::numeric_limits`
#include <span>    // `std::span`
#include <utility> // `std::move`, `std::swap`

#include <smashtable/basic_ring.hpp>

#include "harness.hpp"
#include "sequence.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Tests

/** Capacities are powers of two up to the limit, zero holds nothing, and a refusal is reported. */
void test_capacity_rules() {
    expect_equal(basic_ring<int>::make(3).status(), status_t::invalid_argument_k);
    // No power of two above the limit fits a 32-bit `std::size_t`, so there is nothing to refuse
    if constexpr (basic_ring<int>::capacity_limit_k <= (std::numeric_limits<std::size_t>::max)() / 2)
        expect_equal(basic_ring<int>::make(basic_ring<int>::capacity_limit_k * 2).status(),
                     status_t::invalid_argument_k);
    using refused_ring_t = basic_ring<int, refusing_allocator<int>>;
    expect_equal(refused_ring_t::make(16).status(), status_t::out_of_memory_heap_k);

    auto empty = basic_ring<int>::make(0);
    if (!expect_equal(empty.status(), success_k)) return;
    expect_true(empty->empty());
    expect_true(empty->full());
    expect_equal(empty->push(1), status_t::capacity_exhausted_k);
    int destination = 0;
    expect_equal(empty->pop_into(destination), status_t::operation_would_block_k);

    auto ring = basic_ring<int>::make(4);
    if (!expect_equal(ring.status(), success_k)) return;
    for (int value = 0; value < 4; ++value) expect_equal(ring->push(int {value}), success_k);
    expect_true(ring->full());
    expect_equal(ring->push(9), status_t::capacity_exhausted_k);
    expect_equal(ring->size(), 4u);
    expect_equal(ring->free_space(), 0u);
}

/** Random pushes and pops across many wraparounds match an unbounded queue in a plain array. */
void test_order_across_wraparound() {
    split_mix_generator_t generator {case_stream_key()};
    auto ring = basic_ring<std::uint64_t>::make(16);
    if (!expect_equal(ring.status(), success_k)) return;
    std::uint64_t oracle[1 << 16];
    std::size_t oracle_first = 0;
    std::size_t oracle_end = 0;
    std::uint64_t next_value = 0;
    for (std::size_t round = 0; round < 3000 && oracle_end + 32 < std::size(oracle); ++round) {
        std::size_t const pushes = generator() % 9;
        for (std::size_t push = 0; push < pushes; ++push) {
            status_t const status = ring->push(std::uint64_t {next_value});
            if (oracle_end - oracle_first == 16) expect_equal(status, status_t::capacity_exhausted_k);
            else {
                expect_equal(status, success_k);
                oracle[oracle_end++] = next_value;
            }
            ++next_value;
        }
        expect_equal(ring->size(), oracle_end - oracle_first);
        for (std::size_t index = 0; index < ring->size(); ++index) {
            expect_equal((*ring)[index], oracle[oracle_first + index]);
            expect_equal(*ring->at(index), oracle[oracle_first + index]);
        }
        expect_equal(ring->at(ring->size()), nullptr);
        if (!ring->empty()) {
            expect_equal(ring->oldest(), oracle[oracle_first]);
            expect_equal(ring->newest(), oracle[oracle_end - 1]);
        }
        std::size_t const pops = generator() % 9;
        for (std::size_t pop = 0; pop < pops; ++pop) {
            std::uint64_t popped = 0;
            status_t const status = ring->pop_into(popped);
            if (oracle_first == oracle_end) expect_equal(status, status_t::operation_would_block_k);
            else {
                expect_equal(status, success_k);
                expect_equal(popped, oracle[oracle_first++]);
            }
        }
    }
}

/** An evicting push moves the oldest out only when the ring is full. */
void test_push_evicting() {
    auto ring = basic_ring<int>::make(2);
    if (!expect_equal(ring.status(), success_k)) return;
    int evicted = -1;
    expect_equal(ring->push_evicting(1, evicted), ring_eviction_t::kept_every_element_k);
    expect_equal(ring->push_evicting(2, evicted), ring_eviction_t::kept_every_element_k);
    expect_equal(evicted, -1);
    expect_equal(ring->push_evicting(3, evicted), ring_eviction_t::evicted_oldest_k);
    expect_equal(evicted, 1);
    expect_equal(ring->oldest(), 2);
    expect_equal(ring->newest(), 3);
    ring->pop();
    expect_equal(ring->push(assume_reserved, 4), success_k);
    expect_equal((*ring)[0], 3);
    expect_equal((*ring)[1], 4);
}

/** Bulk calls take as much as fits and report how much that was. */
void test_bulk_push_and_pop() {
    auto ring = basic_ring<int>::make(8);
    if (!expect_equal(ring.status(), success_k)) return;
    int const incoming[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    expect_equal(ring->push_many(std::span<int const>(incoming, 5)), 5u);
    expect_equal(ring->push_many(std::span<int const>(incoming + 5, 6)), 3u);
    int outgoing[16] = {};
    expect_equal(ring->pop_many(std::span<int>(outgoing, 3)), 3u);
    expect_equal(ring->push_many(std::span<int const>(incoming + 8, 3)), 3u);
    expect_equal(ring->pop_many(std::span<int>(outgoing + 3, 16 - 3)), 8u);
    for (int index = 0; index < 11; ++index) expect_equal(outgoing[index], index);
    expect_true(ring->empty());
}

/** Every element in a slot is destroyed exactly once: popped, cleared, or dropped with the ring. */
void test_element_lifetimes() {
    counted_key_t::reset();
    {
        expected<basic_ring<counted_key_t>> ring = basic_ring<counted_key_t>::make(8);
        if (!expect_equal(ring.status(), success_k)) return;
        for (trivial_id_t value = 0; value < 6; ++value) expect_equal(ring->push(counted_key_t {value}), success_k);
        expect_equal(counted_key_t::alive(), std::ptrdiff_t {6});
        ring->pop();
        counted_key_t destination;
        expect_equal(ring->pop_into(destination), success_k);
        expect_equal(destination.unique_id, trivial_id_t {1});
        expect_equal(counted_key_t::alive(), std::ptrdiff_t {5});
        ring->clear();
        expect_equal(counted_key_t::alive(), std::ptrdiff_t {1});
        for (trivial_id_t value = 0; value < 8; ++value) expect_equal(ring->push(counted_key_t {value}), success_k);

        basic_ring<counted_key_t> moved = std::move(*ring);
        expect_equal(moved.size(), 8u);
        expect_equal(ring->size(), 0u);
        expect_equal(ring->capacity(), 0u);
        basic_ring<counted_key_t> swapped;
        std::swap(moved, swapped);
        expect_equal(swapped.oldest().unique_id, trivial_id_t {0});
        expect_equal(swapped.newest().unique_id, trivial_id_t {7});
        expect_equal(counted_key_t::alive(), std::ptrdiff_t {9});
    }
    counted_key_t::check_balanced();
}

/** The suite every sequence answers, over rings of counted elements and tracked allocations. */
void test_sequence_suite() {
    check_sequence_tags<basic_ring<counted_key_t>>();
    check_sequence_order<basic_ring<counted_key_t>>();
    check_sequence_element_lifetimes<basic_ring<counted_key_t>>();
    check_sequence_move_semantics<basic_ring<counted_key_t>>();
    check_sequence_allocator_ledger<basic_ring<counted_key_t, stateful_allocator<counted_key_t>>>();
}

#pragma endregion Tests

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

int main(int argc, char **argv) {
    start_test_run(argc, argv);

    run_test_case("test_capacity_rules", test_capacity_rules);
    run_test_case("test_order_across_wraparound", test_order_across_wraparound);
    run_test_case("test_push_evicting", test_push_evicting);
    run_test_case("test_bulk_push_and_pop", test_bulk_push_and_pop);
    run_test_case("test_element_lifetimes", test_element_lifetimes);
    run_test_case("test_sequence_suite", test_sequence_suite);

    return finish_test_run();
}
