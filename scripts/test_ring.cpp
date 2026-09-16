/**
 *  @brief Tests for @c basic_ring: capacity rules, first-in first-out order across wraparound, eviction and lifetimes.
 *  @author Ash Vardanian
 *  @file scripts/test_ring.cpp
 *  @date September 15, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define ST_STRICT_CALLBACK_CHECKS_ 1

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <random>      // `std::mt19937_64`
#include <span>        // `std::span`
#include <type_traits> // `std::true_type`
#include <utility>     // `std::move`, `std::swap`

#include <smashtable/basic_ring.hpp>

#include "test.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

namespace {

#pragma region Element Types

/** An element that counts its live instances, so a leaked or doubly destroyed slot shows up as a nonzero balance. */
struct counted_element_t {
    static inline long live_count = 0;

    std::uint64_t value {0};

    counted_element_t() noexcept { ++live_count; }
    explicit counted_element_t(std::uint64_t initial) noexcept : value(initial) { ++live_count; }
    counted_element_t(counted_element_t &&other) noexcept : value(other.value) { ++live_count; }
    counted_element_t(counted_element_t const &other) noexcept : value(other.value) { ++live_count; }
    counted_element_t &operator=(counted_element_t &&) noexcept = default;
    counted_element_t &operator=(counted_element_t const &) noexcept = default;
    ~counted_element_t() noexcept { --live_count; }
};

/** An allocator that refuses every request. */
template <typename value_type_>
struct refusing_allocator {
    using value_type = value_type_;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type;

    constexpr refusing_allocator() noexcept = default;
    template <typename other_type_>
    constexpr refusing_allocator(refusing_allocator<other_type_> const &) noexcept {}

    [[nodiscard]] value_type_ *allocate(std::size_t) noexcept { return nullptr; }
    void deallocate(value_type_ *, std::size_t) noexcept {}

    template <typename other_type_>
    constexpr bool operator==(refusing_allocator<other_type_> const &) const noexcept {
        return true;
    }
};

#pragma endregion Element Types

#pragma region Tests

/** Capacities are powers of two up to the limit, zero holds nothing, and a refused allocation is reported. */
void ring_capacity_rules() {
    st_verify_eq_(basic_ring<int>::make(3).status(), status_t::invalid_argument_k);
    st_verify_eq_(basic_ring<int>::make(basic_ring<int>::capacity_limit_k * 2).status(), status_t::invalid_argument_k);
    using refused_ring_t = basic_ring<int, refusing_allocator<int>>;
    st_verify_eq_(refused_ring_t::make(16).status(), status_t::out_of_memory_heap_k);

    auto empty = basic_ring<int>::make(0);
    st_verify_(empty);
    st_verify_(empty->empty());
    st_verify_(empty->full());
    st_verify_eq_(empty->push(1), status_t::capacity_exhausted_k);
    int destination = 0;
    st_verify_eq_(empty->pop_into(destination), status_t::operation_would_block_k);

    auto ring = basic_ring<int>::make(4);
    st_verify_(ring);
    for (int value = 0; value < 4; ++value) st_verify_(ring->push(int {value}));
    st_verify_(ring->full());
    st_verify_eq_(ring->push(9), status_t::capacity_exhausted_k);
    st_verify_eq_(ring->size(), 4u);
    st_verify_eq_(ring->free_space(), 0u);
}

/** Random pushes and pops across many wraparounds match an unbounded queue drawn in a plain array. */
void ring_order_across_wraparound() {
    std::mt19937_64 generator(test_seed_for(__func__));
    auto ring = basic_ring<std::uint64_t>::make(16);
    st_verify_(ring);
    std::uint64_t oracle[1 << 16];
    std::size_t oracle_first = 0;
    std::size_t oracle_end = 0;
    std::uint64_t next_value = 0;
    for (std::size_t round = 0; round < 3000 && oracle_end + 32 < std::size(oracle); ++round) {
        std::size_t const pushes = generator() % 9;
        for (std::size_t push = 0; push < pushes; ++push) {
            status_t const status = ring->push(std::uint64_t {next_value});
            if (oracle_end - oracle_first == 16) st_verify_eq_(status, status_t::capacity_exhausted_k);
            else {
                st_verify_(status);
                oracle[oracle_end++] = next_value;
            }
            ++next_value;
        }
        st_verify_eq_(ring->size(), oracle_end - oracle_first);
        for (std::size_t index = 0; index < ring->size(); ++index) {
            st_verify_eq_((*ring)[index], oracle[oracle_first + index]);
            st_verify_eq_(*ring->at(index), oracle[oracle_first + index]);
        }
        st_verify_eq_(ring->at(ring->size()), nullptr);
        if (!ring->empty()) {
            st_verify_eq_(ring->oldest(), oracle[oracle_first]);
            st_verify_eq_(ring->newest(), oracle[oracle_end - 1]);
        }
        std::size_t const pops = generator() % 9;
        for (std::size_t pop = 0; pop < pops; ++pop) {
            std::uint64_t popped = 0;
            status_t const status = ring->pop_into(popped);
            if (oracle_first == oracle_end) st_verify_eq_(status, status_t::operation_would_block_k);
            else {
                st_verify_(status);
                st_verify_eq_(popped, oracle[oracle_first++]);
            }
        }
    }
}

/** An evicting push moves the oldest out only when the ring is full. */
void ring_push_evicting() {
    auto ring = basic_ring<int>::make(2);
    st_verify_(ring);
    int evicted = -1;
    st_verify_eq_(ring->push_evicting(1, evicted), ring_eviction_t::kept_every_element_k);
    st_verify_eq_(ring->push_evicting(2, evicted), ring_eviction_t::kept_every_element_k);
    st_verify_eq_(evicted, -1);
    st_verify_eq_(ring->push_evicting(3, evicted), ring_eviction_t::evicted_oldest_k);
    st_verify_eq_(evicted, 1);
    st_verify_eq_(ring->oldest(), 2);
    st_verify_eq_(ring->newest(), 3);
    ring->pop();
    st_verify_(ring->push(assume_reserved, 4));
    st_verify_eq_((*ring)[0], 3);
    st_verify_eq_((*ring)[1], 4);
}

/** Bulk calls take as much as fits and report how much that was. */
void ring_bulk_push_and_pop() {
    auto ring = basic_ring<int>::make(8);
    st_verify_(ring);
    int const incoming[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    st_verify_eq_(ring->push_many(std::span<int const>(incoming, 5)), 5u);
    st_verify_eq_(ring->push_many(std::span<int const>(incoming + 5, 6)), 3u);
    int outgoing[16] = {};
    st_verify_eq_(ring->pop_many(std::span<int>(outgoing, 3)), 3u);
    st_verify_eq_(ring->push_many(std::span<int const>(incoming + 8, 3)), 3u);
    st_verify_eq_(ring->pop_many(std::span<int>(outgoing + 3, 16 - 3)), 8u);
    for (int index = 0; index < 11; ++index) st_verify_eq_(outgoing[index], index);
    st_verify_(ring->empty());
}

/** Every element constructed into a slot is destroyed exactly once, whether popped, cleared or dropped with the ring.
 */
void ring_element_lifetimes() {
    counted_element_t::live_count = 0;
    {
        auto ring = basic_ring<counted_element_t>::make(8);
        st_verify_(ring);
        for (std::uint64_t value = 0; value < 6; ++value) st_verify_(ring->push(counted_element_t {value}));
        st_verify_eq_(counted_element_t::live_count, 6);
        ring->pop();
        counted_element_t destination;
        st_verify_(ring->pop_into(destination));
        st_verify_eq_(destination.value, 1u);
        st_verify_eq_(counted_element_t::live_count, 5);
        ring->clear();
        st_verify_eq_(counted_element_t::live_count, 1);
        for (std::uint64_t value = 0; value < 8; ++value) st_verify_(ring->push(counted_element_t {value}));

        basic_ring<counted_element_t> moved = std::move(*ring);
        st_verify_eq_(moved.size(), 8u);
        st_verify_eq_(ring->size(), 0u);
        st_verify_eq_(ring->capacity(), 0u);
        basic_ring<counted_element_t> swapped;
        std::swap(moved, swapped);
        st_verify_eq_(swapped.oldest().value, 0u);
        st_verify_eq_(swapped.newest().value, 7u);
        st_verify_eq_(counted_element_t::live_count, 9);
    }
    st_verify_eq_(counted_element_t::live_count, 0);
}

#pragma endregion Tests

} // namespace

int main() {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "ring.capacity_rules", ring_capacity_rules);
    failures += run_test(filter, "ring.order_across_wraparound", ring_order_across_wraparound);
    failures += run_test(filter, "ring.push_evicting", ring_push_evicting);
    failures += run_test(filter, "ring.bulk_push_and_pop", ring_bulk_push_and_pop);
    failures += run_test(filter, "ring.element_lifetimes", ring_element_lifetimes);

    return report_test_failures(failures);
}
