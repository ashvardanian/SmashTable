/**
 *  @file test/slot_pool.cpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief Tests for @c basic_slot_pool and @c intrusive_list: taking until refused, stale handles,
 *      record lifetimes, and unlinking from either kind of linked record.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <limits>  // `std::numeric_limits`
#include <utility> // `std::move`

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
    tally += run_test(settings, "intrusive_list.singly_linked", intrusive_list_singly_linked);
    tally += run_test(settings, "intrusive_list.doubly_linked", intrusive_list_doubly_linked);

    return report_test_failures(settings, tally);
}
