/**
 *  @file test/slot_pool.cpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief Tests for @c basic_slot_pool, @c basic_chunked_slot_pool and @c intrusive_list: taking
 *      until refused, stale handles, record lifetimes, growth without moving, readers beside
 *      growth, and unlinking from either kind of linked record.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <limits>  // `std::numeric_limits`
#include <thread>  // `std::thread`
#include <utility> // `std::move`

#include "smashtable/basic_chunked_slot_pool.hpp"
#include "smashtable/basic_slot_pool.hpp"
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
static_assert(sizeof(basic_slot_pool<watched_record_t>::handle_t) == 8, "a handle fits one event word");
static_assert(singly_linked_records<singly_linked_record_t[4], std::uint32_t> &&
              !doubly_linked_records<singly_linked_record_t[4], std::uint32_t>);
static_assert(doubly_linked_records<basic_slot_pool<doubly_linked_record_t>, std::uint32_t>);

#pragma endregion Fixtures

#pragma region Suites

/** Pushes, erases from the head, the middle and the tail, and pops, over either kind of record. */
template <typename records_type_>
void verify_list_unlinks(records_type_ &records) {
    using list_t = intrusive_list<>;
    list_t list;
    st_verify_(list.empty());
    st_verify_eq_(list.pop(records), list_t::absent_k);
    st_verify_eq_(list.erase(records, 3u), status_t::key_not_found_k, "a record never pushed");

    for (std::uint32_t index = 0; index != 6; ++index) list.push(records, index);
    st_verify_(list.erase(records, 2u));
    st_verify_eq_(list.erase(records, 2u), status_t::key_not_found_k, "a record already unlinked");
    st_verify_(list.erase(records, 0u));
    st_verify_(list.erase(records, 5u));
    st_verify_eq_(list.oldest, 1u);
    list.push(records, 2u);

    std::uint32_t const expected_order[] = {1, 3, 4, 2};
    for (std::uint32_t const expected : expected_order) st_verify_eq_(list.pop(records), expected);
    st_verify_(list.empty());
    st_verify_eq_(list.newest, list_t::absent_k);

    list.push(records, 4u);
    st_verify_(list.erase(records, 4u));
    st_verify_(list.empty());
    st_verify_eq_(list.newest, list_t::absent_k);
}

#pragma endregion Suites

#pragma region Tests

/** Takes hand out ascending indices until refused, a vacated index comes back, and refusals are
 *  reported. */
void slot_pool_take_until_refused() {
    using pool_t = basic_slot_pool<std::uint64_t>;
    st_verify_eq_(pool_t::make(std::numeric_limits<std::uint32_t>::max()).status(), status_t::invalid_argument_k);
    using refused_pool_t = basic_slot_pool<std::uint64_t, std::uint32_t, refusing_allocator<std::byte>>;
    st_verify_eq_(refused_pool_t::make(4).status(), status_t::out_of_memory_heap_k);

    auto empty = pool_t::make(0);
    st_verify_(empty);
    st_verify_eq_(empty->take(), pool_t::absent_k);

    auto pool = pool_t::make(5);
    st_verify_(pool);
    for (std::uint32_t expected = 0; expected != 5; ++expected) st_verify_eq_(pool->take(), expected);
    st_verify_(pool->full());
    st_verify_eq_(pool->take(), pool_t::absent_k);
    pool->vacate(2);
    st_verify_eq_(pool->size(), 4u);
    st_verify_eq_(pool->take(), 2u);
    st_verify_eq_(pool->index_of((*pool)[3]), 3u);
    st_verify_eq_(pool->records().size(), 5u);
}

/** A handle resolves while its record is taken, and to nothing once it was vacated, even after
 *  the same index is taken again. */
void slot_pool_stale_handle_resolves_to_nothing() {
    using pool_t = basic_slot_pool<watched_record_t>;
    auto pool = pool_t::make(2);
    st_verify_(pool);
    std::uint32_t const taken = pool->take();
    (*pool)[taken].descriptor = 7;
    pool_t::handle_t const first = pool->handle_of(taken);
    st_verify_eq_(pool->at(first), &(*pool)[taken]);

    pool->vacate(taken);
    st_verify_eq_(pool->at(first), nullptr);
    st_verify_eq_(pool->take(), taken);
    st_verify_eq_(pool->at(first), nullptr, "the next holder is not the one the handle named");
    st_verify_eq_((*pool)[taken].descriptor, 7u, "a record is handed out as its last holder left it");
    pool_t::handle_t const second = pool->handle_of(taken);
    st_verify_eq_(pool->at(second), &(*pool)[taken]);
    st_verify_eq_(pool->at(pool_t::handle_t {2, 0}), nullptr, "an index past the pool");
}

/** Every record is built once at @c make and destroyed once with the pool, whatever is taken. */
void slot_pool_records_live_with_the_pool() {
    counted_key_t::reset();
    {
        auto pool = basic_slot_pool<counted_key_t>::make(6);
        st_verify_(pool);
        st_verify_eq_(counted_key_t::alive(), std::ptrdiff_t {6});
        std::uint32_t const taken = pool->take();
        (*pool)[taken].unique_id = 9;
        pool->vacate(taken);
        st_verify_eq_(counted_key_t::alive(), std::ptrdiff_t {6}, "taking and vacating builds nothing");
        basic_slot_pool<counted_key_t> moved = std::move(*pool);
        st_verify_eq_(moved[taken].unique_id, trivial_id_t {9});
        st_verify_eq_(counted_key_t::alive(), std::ptrdiff_t {6});
    }
    st_verify_eq_(counted_key_t::alive(), std::ptrdiff_t {0});
}

/** A chunked pool grows by doubling when a take finds none free, never moving a record, and a
 *  refused growth answers @c absent_k with every record and handle intact. */
void chunked_slot_pool_grows_without_moving() {
    using pool_t = basic_chunked_slot_pool<watched_record_t>;
    st_verify_eq_(pool_t::make(5, 4).status(), status_t::invalid_argument_k, "a limit below the capacity");
    st_verify_eq_(pool_t::make(0, pool_t::capacity_max_k + 1).status(), status_t::invalid_argument_k);

    constexpr std::uint32_t limit = 100;
    auto pool = pool_t::make(3, limit);
    st_verify_(pool);
    st_verify_eq_(pool->capacity(), 4u, "the first chunk rounds up to a power of two");
    st_verify_eq_(pool->capacity_limit(), std::size_t {limit});
    watched_record_t *addresses[limit];
    pool_t::handle_t handles[limit];
    for (std::uint32_t expected = 0; expected != limit; ++expected) {
        st_verify_eq_(pool->take(), expected);
        (*pool)[expected].descriptor = expected * 3;
        addresses[expected] = &(*pool)[expected];
        handles[expected] = pool->handle_of(expected);
    }
    st_verify_eq_(pool->capacity(), std::size_t {limit}, "the last chunk is clipped to the limit");
    st_verify_(pool->full());
    st_verify_eq_(pool->take(), pool_t::absent_k);
    for (std::uint32_t index = 0; index != limit; ++index) {
        st_verify_eq_(&(*pool)[index], addresses[index], "a record keeps its address as the pool grows");
        st_verify_eq_(pool->at(handles[index]), addresses[index]);
        st_verify_eq_(pool->index_of(*addresses[index]), index);
        st_verify_eq_((*pool)[index].descriptor, index * 3);
    }
    std::uint32_t walked = 0;
    for (watched_record_t const &record : pool->records()) st_verify_eq_(record.descriptor, walked++ * 3);
    st_verify_eq_(walked, limit);

    pool->vacate(70);
    st_verify_eq_(pool->at(handles[70]), nullptr);
    st_verify_eq_(pool->take(), 70u);

    allocation_ledger_t ledger;
    using ledgered_t = basic_chunked_slot_pool<watched_record_t, std::uint32_t, stateful_allocator_t>;
    auto limited = ledgered_t::make(2, 64, stateful_allocator_t {ledger});
    st_verify_(limited);
    st_verify_eq_(limited->take(), 0u);
    st_verify_eq_(limited->take(), 1u);
    ledgered_t::handle_t const kept = limited->handle_of(1);
    ledger.refuse_everything();
    st_verify_eq_(limited->take(), ledgered_t::absent_k, "a refused growth answers absent");
    st_verify_eq_(limited->capacity(), 2u);
    st_verify_(!limited->full() && "the pool may still grow once the allocator allows");
    st_verify_eq_(limited->at(kept), &(*limited)[1]);
    ledger.reset();
    st_verify_eq_(limited->take(), 2u);
    st_verify_eq_(limited->capacity(), 4u);
}

/** A reader resolving handles taken earlier, holding no claim, keeps finding their records while
 *  the owner grows the chunked pool beside it. */
void chunked_slot_pool_readers_beside_growth() {
    using pool_t = basic_chunked_slot_pool<watched_record_t>;
    constexpr std::uint32_t limit = 1u << 14, early = 8;
    auto pool = pool_t::make(1, limit);
    st_verify_(pool);
    pool_t::handle_t handles[early];
    for (pool_t::handle_t &handle : handles) {
        std::uint32_t const taken = pool->take();
        (*pool)[taken].descriptor = taken + 1;
        handle = pool->handle_of(taken);
    }
    std::uint32_t done = 0;
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
        }
    });
    for (std::uint32_t index = early; index != limit; ++index) {
        std::uint32_t const taken = pool->take();
        (*pool)[taken].descriptor = taken + 1;
    }
    atomic_store(done, 1u);
    reader.join();
    st_verify_eq_(misses, 0u);
    st_verify_gt_(resolved, 0u);
    st_verify_gt_(probed, 0u);
    st_verify_eq_(pool->take(), pool_t::absent_k);
}

/** Records without @c previous unlink by a walk from the oldest. */
void intrusive_list_singly_linked() {
    singly_linked_record_t records[6];
    verify_list_unlinks(records);
}

/** Records with @c previous unlink in place, here kept in a slot pool. */
void intrusive_list_doubly_linked() {
    auto records = basic_slot_pool<doubly_linked_record_t>::make(6);
    st_verify_(records);
    verify_list_unlinks(*records);
}

#pragma endregion Tests

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

int main(int, char **arguments) {
    settings_t const settings = read_settings(arguments[0]);
    install_test_signal_handlers();
    print(probe_machine());
    print(settings);
    test_tally_t tally;

    tally += run_test(settings, "slot_pool.take_until_refused", slot_pool_take_until_refused);
    tally +=
        run_test(settings, "slot_pool.stale_handle_resolves_to_nothing", slot_pool_stale_handle_resolves_to_nothing);
    tally += run_test(settings, "slot_pool.records_live_with_the_pool", slot_pool_records_live_with_the_pool);
    tally += run_test(settings, "chunked_slot_pool.grows_without_moving", chunked_slot_pool_grows_without_moving);
    tally += run_test(settings, "chunked_slot_pool.readers_beside_growth", chunked_slot_pool_readers_beside_growth);
    tally += run_test(settings, "intrusive_list.singly_linked", intrusive_list_singly_linked);
    tally += run_test(settings, "intrusive_list.doubly_linked", intrusive_list_doubly_linked);

    return report_test_failures(settings, tally);
}
