/**
 *  @file test/pools.cpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief Tests for @c basic_fixed_pool, @c basic_growing_pool and @c intrusive_list: taking
 *      until refused, stale handles, record lifetimes, growth without moving, readers beside
 *      growth, and unlinking from either kind of linked record.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <limits>  // `std::numeric_limits`
#include <thread>  // `std::thread`
#include <utility> // `std::move`

#include "smashtable/basic_growing_pool.hpp"
#include "smashtable/basic_fixed_pool.hpp"
#include "smashtable/intrusive_list.hpp"

#include "fixtures.hpp"
#include "harness.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Fixtures

/** A record tagged with its generation. */
struct watched_record_t {
    std::uint32_t descriptor {0};
    std::uint32_t generation {0};
};

/** A record wider than the default new alignment, as one holding cache-line-aligned words is. */
struct alignas(64) over_aligned_record_t {
    std::uint32_t generation {0};
};

/** A record linked one way. */
struct singly_linked_record_t {
    std::uint32_t next {intrusive_list<>::absent_k};
    std::uint32_t payload {0};
};

/** A record linked both ways. */
struct doubly_linked_record_t {
    std::uint32_t next {intrusive_list<>::absent_k};
    std::uint32_t previous {intrusive_list<>::absent_k};
};

static_assert(generation_tagged<watched_record_t, std::uint32_t> && !generation_tagged<std::uint64_t, std::uint32_t>);
static_assert(sizeof(basic_fixed_pool<watched_record_t>::handle_t) == 8, "a handle fits one event word");
static_assert(singly_linked_records<singly_linked_record_t[4], std::uint32_t> &&
              !doubly_linked_records<singly_linked_record_t[4], std::uint32_t>);
static_assert(doubly_linked_records<basic_fixed_pool<doubly_linked_record_t>, std::uint32_t>);

#pragma endregion Fixtures

#pragma region Suites

/** Pushes, erases from the head, the middle and the tail, and pops, over either kind of record. */
template <typename records_type_>
void check_list_unlinks(records_type_ &records) {
    using list_t = intrusive_list<>;
    list_t list;
    expect_true(list.empty());
    expect_equal(list.pop(records), list_t::absent_k);
    expect_equal(list.erase(records, 3u), status_t::key_not_found_k, "a record never pushed");

    for (std::uint32_t index = 0; index != 6; ++index) list.push(records, index);
    expect_equal(list.erase(records, 2u), success_k);
    expect_equal(list.erase(records, 2u), status_t::key_not_found_k, "a record already unlinked");
    expect_equal(list.erase(records, 0u), success_k);
    expect_equal(list.erase(records, 5u), success_k);
    expect_equal(list.oldest, 1u);
    list.push(records, 2u);

    std::uint32_t const expected_order[] = {1, 3, 4, 2};
    for (std::uint32_t const expected : expected_order) expect_equal(list.pop(records), expected);
    expect_true(list.empty());
    expect_equal(list.newest, list_t::absent_k);

    list.push(records, 4u);
    expect_equal(list.erase(records, 4u), success_k);
    expect_true(list.empty());
    expect_equal(list.newest, list_t::absent_k);
}

#pragma endregion Suites

#pragma region Tests

/** Takes hand out ascending indices until refused, a vacated index comes back, and refusals are
 *  reported. */
void test_fixed_pool_take_until_refused() {
    using pool_t = basic_fixed_pool<std::uint64_t>;
    expect_equal(pool_t::make((std::numeric_limits<std::uint32_t>::max)()).status(), status_t::invalid_argument_k);
    using refused_pool_t = basic_fixed_pool<std::uint64_t, std::uint32_t, refusing_allocator<std::byte>>;
    expect_equal(refused_pool_t::make(4).status(), status_t::out_of_memory_heap_k);

    auto empty = pool_t::make(0);
    if (!expect_equal(empty.status(), success_k)) return;
    expect_equal(empty->take(), pool_t::absent_k);

    auto pool = pool_t::make(5);
    if (!expect_equal(pool.status(), success_k)) return;
    for (std::uint32_t expected = 0; expected != 5; ++expected) expect_equal(pool->take(), expected);
    expect_true(pool->full());
    expect_equal(pool->take(), pool_t::absent_k);
    pool->vacate(2);
    expect_equal(pool->size(), 4u);
    expect_equal(pool->take(), 2u);
    expect_equal(pool->index_of((*pool)[3]), 3u);
    expect_equal(pool->records().size(), 5u);
}

/** A handle resolves while its record is taken, and to nothing once it was vacated, even after
 *  the same index is taken again. */
void test_fixed_pool_stale_handle_resolves_to_nothing() {
    using pool_t = basic_fixed_pool<watched_record_t>;
    auto pool = pool_t::make(2);
    if (!expect_equal(pool.status(), success_k)) return;
    std::uint32_t const taken = pool->take();
    (*pool)[taken].descriptor = 7;
    pool_t::handle_t const first = pool->handle_of(taken);
    expect_equal(pool->at(first), &(*pool)[taken]);

    pool->vacate(taken);
    expect_equal(pool->at(first), nullptr);
    expect_equal(pool->take(), taken);
    expect_equal(pool->at(first), nullptr, "the next holder is not the one the handle named");
    expect_equal((*pool)[taken].descriptor, 7u, "a record is handed out as its last holder left it");
    pool_t::handle_t const second = pool->handle_of(taken);
    expect_equal(pool->at(second), &(*pool)[taken]);
    expect_equal(pool->at(pool_t::handle_t {2, 0}), nullptr, "an index past the pool");
}

/** Every record is built once at @c make and destroyed once with the pool, whatever is taken. */
void test_fixed_pool_records_live_with_the_pool() {
    counted_key_t::reset();
    {
        auto pool = basic_fixed_pool<counted_key_t>::make(6);
        if (!expect_equal(pool.status(), success_k)) return;
        expect_equal(counted_key_t::alive(), std::ptrdiff_t {6});
        std::uint32_t const taken = pool->take();
        (*pool)[taken].unique_id = 9;
        pool->vacate(taken);
        expect_equal(counted_key_t::alive(), std::ptrdiff_t {6}, "taking and vacating builds nothing");
        basic_fixed_pool<counted_key_t> moved = std::move(*pool);
        expect_equal(moved[taken].unique_id, trivial_id_t {9});
        expect_equal(counted_key_t::alive(), std::ptrdiff_t {6});
    }
    expect_equal(counted_key_t::alive(), std::ptrdiff_t {0});
}

/** A chunked pool grows by doubling when a take finds none free, never moving a record, and a
 *  refused growth answers @c absent_k with every record and handle intact. */
void test_growing_pool_grows_without_moving() {
    using pool_t = basic_growing_pool<watched_record_t>;
    expect_equal(pool_t::make(5, 4).status(), status_t::invalid_argument_k, "a limit below the capacity");
    expect_equal(pool_t::make(0, pool_t::capacity_max_k + 1).status(), status_t::invalid_argument_k);

    constexpr std::uint32_t limit = 100;
    auto pool = pool_t::make(3, limit);
    if (!expect_equal(pool.status(), success_k)) return;
    expect_equal(pool->capacity(), 4u, "the first chunk rounds up to a power of two");
    expect_equal(pool->capacity_limit(), std::size_t {limit});
    watched_record_t *addresses[limit];
    pool_t::handle_t handles[limit];
    for (std::uint32_t expected = 0; expected != limit; ++expected) {
        expect_equal(pool->take(), expected);
        (*pool)[expected].descriptor = expected * 3;
        addresses[expected] = &(*pool)[expected];
        handles[expected] = pool->handle_of(expected);
    }
    expect_equal(pool->capacity(), std::size_t {limit}, "the last chunk is clipped to the limit");
    expect_true(pool->full());
    expect_equal(pool->take(), pool_t::absent_k);
    for (std::uint32_t index = 0; index != limit; ++index) {
        expect_equal(&(*pool)[index], addresses[index], "a record keeps its address as the pool grows");
        expect_equal(pool->at(handles[index]), addresses[index]);
        expect_equal(pool->index_of(*addresses[index]), index);
        expect_equal((*pool)[index].descriptor, index * 3);
    }
    std::uint32_t walked = 0;
    for (watched_record_t const &record : pool->records()) expect_equal(record.descriptor, walked++ * 3);
    expect_equal(walked, limit);

    pool->vacate(70);
    expect_equal(pool->at(handles[70]), nullptr);
    expect_equal(pool->take(), 70u);

    allocation_ledger_t ledger;
    using ledgered_t = basic_growing_pool<watched_record_t, std::uint32_t, stateful_allocator_t>;
    auto limited = ledgered_t::make(2, 64, stateful_allocator_t {ledger});
    if (!expect_equal(limited.status(), success_k)) return;
    expect_equal(limited->take(), 0u);
    expect_equal(limited->take(), 1u);
    ledgered_t::handle_t const kept = limited->handle_of(1);
    ledger.refuse_everything();
    expect_equal(limited->take(), ledgered_t::absent_k, "a refused growth answers absent");
    expect_equal(limited->capacity(), 2u);
    expect_true(!limited->full(), "the pool may still grow once the allocator allows");
    expect_equal(limited->at(kept), &(*limited)[1]);
    ledger.reset();
    expect_equal(limited->take(), 2u);
    expect_equal(limited->capacity(), 4u);
}

/** Every chunk a growing pool adds keeps an over-aligned record on its alignment. */
void test_growing_pool_over_aligned_records() {
    using pool_t = basic_growing_pool<over_aligned_record_t>;
    constexpr std::uint32_t limit = 100;
    auto pool = pool_t::make(1, limit);
    if (!expect_equal(pool.status(), success_k)) return;
    for (std::uint32_t expected = 0; expected != limit; ++expected) {
        expect_equal(pool->take(), expected);
        expect_equal(reinterpret_cast<std::uintptr_t>(&(*pool)[expected]) % alignof(over_aligned_record_t),
                     std::uintptr_t {0});
    }
}

/** A reader resolving handles taken earlier, holding no claim, keeps finding their records while
 *  the owner grows the chunked pool beside it. */
void test_growing_pool_readers_beside_growth() {
    using pool_t = basic_growing_pool<watched_record_t>;
    constexpr std::uint32_t limit = 1u << 14, early = 8;
    auto pool = pool_t::make(1, limit);
    if (!expect_equal(pool.status(), success_k)) return;
    pool_t::handle_t handles[early];
    for (pool_t::handle_t &handle : handles) {
        std::uint32_t const taken = pool->take();
        (*pool)[taken].descriptor = taken + 1;
        handle = pool->handle_of(taken);
    }
    std::uint32_t done = 0, reading = 0;
    std::size_t misses = 0, resolved = 0, probed = 0;
    std::thread reader([&]() noexcept {
        // Probing past the earlier handles reads records of chunks as they are published.
        for (std::uint32_t probe = 0; !atomic_load(done); probe = (probe + 1) % limit) {
            for (pool_t::handle_t const &handle : handles) {
                watched_record_t const *const record = pool->at(handle);
                misses += !record || record->descriptor != handle.index + 1;
                ++resolved;
            }
            probed += pool->at(pool_t::handle_t {probe, 0}) != nullptr;
            atomic_store(reading, 1u);
        }
    });
    // A loaded machine may start the reader only after the owner finished growing, so it waits.
    expect_true(spin_until([&] { return atomic_load(reading) != 0; }));
    for (std::uint32_t index = early; index != limit; ++index) {
        std::uint32_t const taken = pool->take();
        (*pool)[taken].descriptor = taken + 1;
    }
    atomic_store(done, 1u);
    reader.join();
    expect_equal(misses, 0u);
    expect_true(resolved > 0u);
    expect_true(probed > 0u);
    expect_equal(pool->take(), pool_t::absent_k);
}

/** Records without @c previous unlink by a walk from the oldest. */
void test_intrusive_list_singly_linked() {
    singly_linked_record_t records[6];
    check_list_unlinks(records);
}

/** Records with @c previous unlink in place, here kept in a fixed pool. */
void test_intrusive_list_doubly_linked() {
    auto records = basic_fixed_pool<doubly_linked_record_t>::make(6);
    if (!expect_equal(records.status(), success_k)) return;
    check_list_unlinks(*records);
}

#pragma endregion Tests

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

int main(int argc, char **argv) {
    start_test_run(argc, argv);

    run_test_case("test_fixed_pool_take_until_refused", test_fixed_pool_take_until_refused);
    run_test_case("test_fixed_pool_stale_handle_resolves_to_nothing", test_fixed_pool_stale_handle_resolves_to_nothing);
    run_test_case("test_fixed_pool_records_live_with_the_pool", test_fixed_pool_records_live_with_the_pool);
    run_test_case("test_growing_pool_grows_without_moving", test_growing_pool_grows_without_moving);
    run_test_case("test_growing_pool_readers_beside_growth", test_growing_pool_readers_beside_growth);
    run_test_case("test_growing_pool_over_aligned_records", test_growing_pool_over_aligned_records);
    run_test_case("test_intrusive_list_singly_linked", test_intrusive_list_singly_linked);
    run_test_case("test_intrusive_list_doubly_linked", test_intrusive_list_doubly_linked);

    return finish_test_run();
}
