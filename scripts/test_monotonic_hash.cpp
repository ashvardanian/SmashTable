/**
 *  @brief Test instantiations for the transactional store over an open-addressed hash table. Covers only the
 *      point-access surface - insert, upsert, update, erase, find, watch and the two-phase commit - since an
 *      unordered core supplies no bounds, ranges, or order statistics.
 *  @author Ash Vardanian
 *  @file scripts/test_monotonic_hash.cpp
 *  @date August 17, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define ST_STRICT_CALLBACK_CHECKS_ 1

#include <cstddef> // `std::size_t`

#include <vector> // `std::vector`

#include <smashtable/monotonic_store.hpp>

#include "test.hpp"
#include "test_basic.hpp"
#include "test_commit_stamp.hpp"
#include "test_consistency.hpp"
#include "test_fixture_coverage.hpp"
#include "test_monotonic_store_defects.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

#pragma region Type Aliases

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Transaction: ✓
 *  Tests: Baseline transactional correctness over the cheapest possible key
 */
using transactional_trivial_set_t = monotonic_hash_set<trivial_key_t>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() → expected<T> | Memory: Heap | Transaction: ✓
 *  Tests: Watch copy OOM, rollback with a heap-allocating key
 */
using transactional_heavy_set_t = monotonic_hash_set<heavy_key_t>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial (key & value) | Memory: Stack | Transaction: ✓
 *  Value: int | Tests: Transactional map operations, value overwrites
 */
using transactional_trivial_map_t = monotonic_hash_map<trivial_key_t, int>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Key trivial, value .copy() | Memory: Heap (value) | Transaction: ✓
 *  Value: guarded_payload_t | Tests: Rollback with non-trivial values
 */
using transactional_composite_map_t = monotonic_hash_map<composite_key_t, guarded_payload_t>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() on key & value | Memory: Heap (both) | Transaction: ✓
 *  Value: guarded_payload_t | Tests: Dual-heap staging and rollback
 */
using transactional_heavy_map_t = monotonic_hash_map<heavy_key_t, guarded_payload_t>;

#pragma endregion Type Aliases

#pragma region Point Access Tests

/** @brief Tests that the four insert strategies differ exactly as documented */
template <typename container_type_>
static void test_point_insert_strategies() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    auto first = container.transaction();
    st_verify_(first.has_value());

    st_verify_(first->insert(trivial_id_to_member<member_t>(1)));
    st_verify_eq_(first->insert(trivial_id_to_member<member_t>(1)), status_t::key_already_exists_k);
    st_verify_(first->insert_if_missing(trivial_id_to_member<member_t>(1)));
    st_verify_(first->upsert(trivial_id_to_member<member_t>(1)));
    st_verify_(first->update(trivial_id_to_member<member_t>(1)));
    st_verify_eq_(first->update(trivial_id_to_member<member_t>(2)), status_t::key_not_found_k);

    st_verify_(first->stage());
    st_verify_(first->commit());
    st_verify_eq_(container.size(), 1);
    st_verify_eq_(container.contains(trivial_id_to_key<member_t>(1)), true);
    st_verify_eq_(container.contains(trivial_id_to_key<member_t>(2)), false);
}

/** @brief Tests that an erase staged in a transaction only becomes visible on commit */
template <typename container_type_>
static void test_point_erase_visibility() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (std::size_t index = 0; index < 8; ++index) st_verify_(container.upsert(trivial_id_to_member<member_t>(index)));
    st_verify_eq_(container.size(), 8);

    auto erasing = container.transaction();
    st_verify_(erasing.has_value());
    for (std::size_t index = 0; index < 4; ++index) st_verify_(erasing->erase(trivial_id_to_key<member_t>(index)));
    st_verify_(erasing->stage());
    for (std::size_t index = 0; index < 4; ++index) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(index)), true);
    }

    st_verify_(erasing->commit());
    for (std::size_t index = 0; index < 4; ++index) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(index)), false);
    }
    for (std::size_t index = 4; index < 8; ++index) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(index)), true);
    }
}

/** @brief Tests that a rollback pulls every staged version back and leaves the store as it was */
template <typename container_type_>
static void test_point_rollback_restores_store() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1)));

    auto writing = container.transaction();
    st_verify_(writing.has_value());
    for (std::size_t index = 1; index <= 5; ++index) st_verify_(writing->upsert(trivial_id_to_member<member_t>(index)));
    st_verify_(writing->stage());
    st_verify_(writing->rollback());

    st_verify_eq_(container.size(), 1);
    st_verify_eq_(container.contains(trivial_id_to_key<member_t>(1)), true);
    for (std::size_t index = 2; index <= 5; ++index) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(index)), false);
    }

    // The rolled-back versions are still the transaction's, so a second staging lands them all
    st_verify_(writing->stage());
    st_verify_(writing->commit());
    st_verify_eq_(container.size(), 5);
}

/** @brief Tests that staging enough keys to outgrow the slab several times loses none of them */
template <typename container_type_>
static void test_point_staging_survives_growth(std::size_t size = 500) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    auto writing = container.transaction();
    st_verify_(writing.has_value());
    for (std::size_t index = 0; index < size; ++index)
        st_verify_(writing->upsert(trivial_id_to_member<member_t>(index)));
    st_verify_(writing->stage());
    st_verify_(writing->commit());

    st_verify_eq_(container.size(), size);
    for (std::size_t index = 0; index < size; ++index) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(index)), true);
    }
}

/** @brief Tests that the unordered walk reports every member once, over a store and over a transaction */
template <typename container_type_>
static void test_point_enumeration_visits_every_member() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using key_t = decltype(trivial_id_to_key<member_t>(0));
    std::size_t const size = 8;

    container_t container;
    std::size_t empty_visits = 0;
    st_verify_(container.for_each([&](member_t const &) noexcept { ++empty_visits; }));
    st_verify_eq_(empty_visits, 0);

    // The keys are materialized up front, since a heap-allocating one cannot be built inside a
    // `noexcept` callback, and a tally of one per key is what proves nothing was reported twice.
    std::vector<key_t> keys;
    for (std::size_t index = 0; index < size; ++index) {
        keys.push_back(trivial_id_to_key<member_t>(index));
        st_verify_(container.upsert(trivial_id_to_member<member_t>(index)));
    }

    std::vector<std::size_t> tally(size, 0);
    std::size_t visits = 0;
    st_verify_(container.for_each([&](member_t const &member) noexcept {
        ++visits;
        for (std::size_t index = 0; index < size; ++index)
            if (mapping_key_or_itself<member_t>(member) == keys[index]) ++tally[index];
    }));
    st_verify_eq_(visits, container.size());
    for (std::size_t index = 0; index < size; ++index) st_verify_eq_(tally[index], 1);

    // A transaction reports its own staged writes and hides its own tombstones.
    auto writing = container.transaction();
    st_verify_(writing.has_value());
    st_verify_(writing->erase(trivial_id_to_key<member_t>(0)));
    st_verify_(writing->upsert(trivial_id_to_member<member_t>(size)));

    std::vector<std::size_t> staged_tally(size, 0);
    std::size_t staged_visits = 0;
    st_verify_(writing->for_each([&](member_t const &member) noexcept {
        ++staged_visits;
        for (std::size_t index = 0; index < size; ++index)
            if (mapping_key_or_itself<member_t>(member) == keys[index]) ++staged_tally[index];
    }));
    st_verify_eq_(staged_visits, size);
    st_verify_eq_(staged_tally[0], 0);
    for (std::size_t index = 1; index < size; ++index) st_verify_eq_(staged_tally[index], 1);

    // Nothing the transaction staged is published, so the store still reports what it did before.
    st_verify_(writing->stage());
    std::size_t published_visits = 0;
    st_verify_(container.for_each([&](member_t const &) noexcept { ++published_visits; }));
    st_verify_eq_(published_visits, size);
}

static void point_access_enumeration_visits_every_member() {
    test_point_enumeration_visits_every_member<transactional_trivial_set_t>();
    test_point_enumeration_visits_every_member<transactional_heavy_set_t>();
    test_point_enumeration_visits_every_member<transactional_trivial_map_t>();
    test_point_enumeration_visits_every_member<transactional_composite_map_t>();
    test_point_enumeration_visits_every_member<transactional_heavy_map_t>();
}

static void point_access_insert_strategies() {
    test_point_insert_strategies<transactional_trivial_set_t>();
    test_point_insert_strategies<transactional_heavy_set_t>();
    test_point_insert_strategies<transactional_trivial_map_t>();
    test_point_insert_strategies<transactional_composite_map_t>();
    test_point_insert_strategies<transactional_heavy_map_t>();
}

static void point_access_erase_visibility() {
    test_point_erase_visibility<transactional_trivial_set_t>();
    test_point_erase_visibility<transactional_heavy_set_t>();
    test_point_erase_visibility<transactional_trivial_map_t>();
    test_point_erase_visibility<transactional_composite_map_t>();
    test_point_erase_visibility<transactional_heavy_map_t>();
}

static void point_access_rollback_restores_store() {
    test_point_rollback_restores_store<transactional_trivial_set_t>();
    test_point_rollback_restores_store<transactional_heavy_set_t>();
    test_point_rollback_restores_store<transactional_trivial_map_t>();
    test_point_rollback_restores_store<transactional_composite_map_t>();
    test_point_rollback_restores_store<transactional_heavy_map_t>();
}

static void point_access_staging_survives_growth() {
    test_point_staging_survives_growth<transactional_trivial_set_t>();
    test_point_staging_survives_growth<transactional_heavy_set_t>();
    test_point_staging_survives_growth<transactional_trivial_map_t>();
    test_point_staging_survives_growth<transactional_composite_map_t>();
    test_point_staging_survives_growth<transactional_heavy_map_t>();
}

static void point_access_insertion_patterns() {
    test_basic_insertion_patterns<transactional_trivial_set_t>();
    test_basic_insertion_patterns<transactional_heavy_set_t>();
    test_basic_insertion_patterns<transactional_trivial_map_t>();
    test_basic_insertion_patterns<transactional_composite_map_t>();
    test_basic_insertion_patterns<transactional_heavy_map_t>();
}

static void point_access_bulk_insertion_iterators() {
    test_bulk_insertion_from_iterators<transactional_trivial_set_t>();
    test_bulk_insertion_from_iterators<transactional_heavy_set_t>();
    test_bulk_insertion_from_iterators<transactional_trivial_map_t>();
    test_bulk_insertion_from_iterators<transactional_composite_map_t>();
    test_bulk_insertion_from_iterators<transactional_heavy_map_t>();
}

static void point_access_bulk_upsert_with_duplicates() { //
    test_bulk_upsert_with_duplicates<transactional_trivial_map_t>();
}

#pragma endregion Point Access Tests

#pragma region Consistency and Transaction Tests

static void transactional_consistency_empty_transaction_commit() {
    test_empty_transaction_commit<transactional_trivial_set_t>();
    test_empty_transaction_commit<transactional_heavy_set_t>();
    test_empty_transaction_commit<transactional_trivial_map_t>();
    test_empty_transaction_commit<transactional_composite_map_t>();
    test_empty_transaction_commit<transactional_heavy_map_t>();
}

static void transactional_consistency_no_dirty_reads_multi_key() {
    test_no_dirty_reads_multi_key<transactional_trivial_map_t>();
    test_no_dirty_reads_multi_key<transactional_composite_map_t>();
    test_no_dirty_reads_multi_key<transactional_heavy_map_t>();
}

static void transactional_consistency_new_transaction_sees_nothing_staged() {
    test_new_transaction_sees_nothing_staged<transactional_trivial_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_trivial_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_map_t>();
}

static void transactional_consistency_committed_immediately_visible() {
    test_committed_immediately_visible<transactional_trivial_map_t>();
    test_committed_immediately_visible<transactional_composite_map_t>();
    test_committed_immediately_visible<transactional_heavy_map_t>();
}

static void transactional_consistency_multi_key_atomicity_10_keys() {
    test_multi_key_atomicity_10_keys<transactional_trivial_map_t>();
    test_multi_key_atomicity_10_keys<transactional_composite_map_t>();
    test_multi_key_atomicity_10_keys<transactional_heavy_map_t>();
}

static void transactional_consistency_rollback_makes_all_invisible() {
    test_rollback_makes_all_invisible<transactional_trivial_map_t>();
    test_rollback_makes_all_invisible<transactional_composite_map_t>();
    test_rollback_makes_all_invisible<transactional_heavy_map_t>();
}

static void transactional_consistency_fractured_read_prevention() {
    test_fractured_read_prevention<transactional_trivial_map_t>();
    test_fractured_read_prevention<transactional_composite_map_t>();
    test_fractured_read_prevention<transactional_heavy_map_t>();
}

/** @brief Ordering of mapped values is only meaningful where they are numbers, so @c int maps only. */
static void transactional_consistency_sequential_updates_never_regress() {
    test_sequential_updates_never_regress<transactional_trivial_map_t>();
}

/** @brief Ordering of mapped values is only meaningful where they are numbers, so @c int maps only. */
static void transactional_consistency_transaction_commits_maintain_order() {
    test_transaction_commits_maintain_order<transactional_trivial_map_t>();
}

static void transactional_consistency_concurrent_transactions_on_same_key() {
    test_concurrent_transactions_on_same_key<transactional_trivial_map_t>();
    test_concurrent_transactions_on_same_key<transactional_composite_map_t>();
    test_concurrent_transactions_on_same_key<transactional_heavy_map_t>();
}

static void transactional_consistency_multi_key_conflict_any_key_fails() {
    test_multi_key_conflict_any_key_fails<transactional_trivial_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_composite_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_detects_external_direct_modification() {
    test_watch_detects_external_direct_modification<transactional_trivial_map_t>();
    test_watch_detects_external_direct_modification<transactional_composite_map_t>();
    test_watch_detects_external_direct_modification<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_detects_staged_invisible_writes() {
    test_watch_detects_staged_invisible_writes<transactional_trivial_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_composite_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_detects_staged_writes_of_older_generation() {
    test_watch_detects_staged_writes_of_older_generation<transactional_trivial_map_t>();
    test_watch_detects_staged_writes_of_older_generation<transactional_composite_map_t>();
    test_watch_detects_staged_writes_of_older_generation<transactional_heavy_map_t>();
}

static void transactional_consistency_abandoned_transaction_leaves_no_trace() {
    test_abandoned_transaction_leaves_no_trace<transactional_trivial_map_t>();
    test_abandoned_transaction_leaves_no_trace<transactional_composite_map_t>();
    test_abandoned_transaction_leaves_no_trace<transactional_heavy_map_t>();
}

static void transactional_consistency_moved_transaction_unwinds_once() {
    test_moved_transaction_unwinds_once<transactional_trivial_map_t>();
    test_moved_transaction_unwinds_once<transactional_composite_map_t>();
    test_moved_transaction_unwinds_once<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_on_erased_key_can_commit() {
    test_watch_on_erased_key_can_commit<transactional_trivial_map_t>();
    test_watch_on_erased_key_can_commit<transactional_composite_map_t>();
    test_watch_on_erased_key_can_commit<transactional_heavy_map_t>();
}

static void transactional_consistency_absent_watch_survives_rollback() {
    test_absent_watch_survives_rollback<transactional_trivial_map_t>();
    test_absent_watch_survives_rollback<transactional_composite_map_t>();
    test_absent_watch_survives_rollback<transactional_heavy_map_t>();
}

static void transactional_consistency_disjoint_keys_both_succeed() {
    test_disjoint_keys_both_succeed<transactional_trivial_map_t>();
    test_disjoint_keys_both_succeed<transactional_composite_map_t>();
    test_disjoint_keys_both_succeed<transactional_heavy_map_t>();
}

static void transactional_consistency_lost_update_matches_isolation() {
    test_lost_update_matches_isolation<transactional_trivial_map_t>();
    test_lost_update_matches_isolation<transactional_composite_map_t>();
    test_lost_update_matches_isolation<transactional_heavy_map_t>();
}

static void transactional_consistency_repeated_read_matches_isolation() {
    test_repeated_read_matches_isolation<transactional_trivial_map_t>();
    test_repeated_read_matches_isolation<transactional_composite_map_t>();
    test_repeated_read_matches_isolation<transactional_heavy_map_t>();
}

static void transactional_consistency_reset_clears_transaction_state() {
    test_reset_clears_transaction_state<transactional_trivial_map_t>();
    test_reset_clears_transaction_state<transactional_composite_map_t>();
    test_reset_clears_transaction_state<transactional_heavy_map_t>();
}

/**
 *  @brief Tests that a watch on an erased version records the shape validation re-derives.
 *
 *  The tombstone is built by hand because the public surface hides one: a caller that kept the
 *  version it erased holds exactly this, and handing it back must not doom every later commit.
 */
template <typename container_type_>
static void test_watch_on_erased_version_commits() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using versioned_entry_t = typename container_t::versioned_entry_t;

    container_t container;
    {
        auto writer = container.transaction();
        st_verify_(writer.has_value());
        st_verify_(writer->insert(trivial_id_to_member<member_t>(1)));
        st_verify_(writer->stage());
        st_verify_(writer->commit());
    }

    generation_t erased_generation = 0;
    {
        auto eraser = container.transaction();
        st_verify_(eraser.has_value());
        st_verify_(eraser->erase(trivial_id_to_key<member_t>(1)));
        erased_generation = eraser->generation();
        st_verify_(eraser->stage());
        st_verify_(eraser->commit());
    }
    st_verify_eq_(container.contains(trivial_id_to_key<member_t>(1)), false);

    auto watcher = container.transaction();
    st_verify_(watcher.has_value());
    versioned_entry_t tombstone {trivial_id_to_member<member_t>(1)};
    tombstone.generation = erased_generation;
    tombstone.presence = presence_t::erased_k;
    st_verify_(watcher->watch(tombstone));
    st_verify_(watcher->insert(trivial_id_to_member<member_t>(2)));
    st_verify_(watcher->stage());
    st_verify_(watcher->commit());
    st_verify_eq_(container.contains(trivial_id_to_key<member_t>(2)), true);
}

static void transactional_consistency_watch_on_erased_version_commits() {
    test_watch_on_erased_version_commits<transactional_trivial_set_t>();
}

#pragma endregion Consistency and Transaction Tests

#pragma region Transactional Store Defects

static void transactional_defects_direct_write_spares_staged_version() {
    test_direct_write_spares_staged_version<transactional_trivial_map_t>();
    test_direct_write_spares_staged_version<transactional_composite_map_t>();
    test_direct_write_spares_staged_version<transactional_heavy_map_t>();
}

static void transactional_defects_direct_erase_spares_staged_version() {
    test_direct_erase_spares_staged_version<transactional_trivial_map_t>();
    test_direct_erase_spares_staged_version<transactional_composite_map_t>();
    test_direct_erase_spares_staged_version<transactional_heavy_map_t>();
}

static void transactional_defects_clear_keeps_generations_moving() {
    test_clear_keeps_generations_moving<transactional_trivial_map_t>();
    test_clear_keeps_generations_moving<transactional_composite_map_t>();
    test_clear_keeps_generations_moving<transactional_heavy_map_t>();
}

static void transactional_defects_committed_erase_hidden_from_point_reads() {
    test_committed_erase_hidden_from_point_reads<transactional_trivial_map_t>();
    test_committed_erase_hidden_from_point_reads<transactional_composite_map_t>();
    test_committed_erase_hidden_from_point_reads<transactional_heavy_map_t>();
}

static void transactional_defects_vacuum_reclaims_committed_tombstones() {
    test_vacuum_reclaims_committed_tombstones<transactional_trivial_map_t>();
    test_vacuum_reclaims_committed_tombstones<transactional_composite_map_t>();
    test_vacuum_reclaims_committed_tombstones<transactional_heavy_map_t>();
}

static void transactional_defects_vacuum_spares_staged_versions() {
    test_vacuum_spares_staged_versions<transactional_trivial_map_t>();
    test_vacuum_spares_staged_versions<transactional_composite_map_t>();
    test_vacuum_spares_staged_versions<transactional_heavy_map_t>();
}

static void transactional_consistency_group_commits_participants_together() {
    test_group_commits_participants_together<transactional_trivial_map_t>();
    test_group_commits_participants_together<transactional_composite_map_t>();
    test_group_commits_participants_together<transactional_heavy_map_t>();
}

static void transactional_consistency_group_unwinds_every_participant_on_conflict() {
    test_group_unwinds_every_participant_on_conflict<transactional_trivial_map_t>();
    test_group_unwinds_every_participant_on_conflict<transactional_composite_map_t>();
    test_group_unwinds_every_participant_on_conflict<transactional_heavy_map_t>();
}

#pragma endregion Transactional Store Defects

#pragma region Reserve

/**
 *  @brief A capacity hint reaches the slab the table grows, rather than being swallowed.
 *
 *  The ledger is what makes the difference observable: a hint that is honoured asks the allocator for
 *  a slab before any element is written, and a hint that is merely accepted asks for nothing.
 */
static void test_reserve_reaches_the_slab() {
    using member_t = mapping<trivial_key_t, int>;
    using ledgered_map_t = monotonic_hash_map<trivial_key_t, int, default_hash_t, equal_to_t, stateful_allocator_t>;

    allocation_ledger_t ledger;
    {
        ledgered_map_t container {stateful_allocator_t {ledger}};
        st_verify_eq_(ledger.granted_count, std::size_t {0});

        st_verify_(container.reserve(64));
        st_verify_(ledger.granted_count > 0);
        st_verify_(ledger.largest_request_elements() > 0);

        for (trivial_id_t identifier = 0; identifier != 64; ++identifier)
            st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier, 1)));
        st_verify_eq_(container.size(), 64);
    }
    ledger.verify_balanced();

    // A slab the budget refuses is reported rather than thrown, which is the whole point of a status.
    allocation_ledger_t starved;
    starved.refuse_everything();
    {
        ledgered_map_t container {stateful_allocator_t {starved}};
        st_verify_eq_(container.reserve(64), status_t::out_of_memory_heap_k);
        st_verify_eq_(container.size(), 0);
    }
    starved.verify_balanced();
}

#pragma endregion Reserve

#pragma region Commit Stamp

static void commit_stamp_rolled_back_stage_does_not_abort_a_peer() {
    test_rolled_back_stage_does_not_abort_a_peer<transactional_trivial_map_t>();
}

static void commit_stamp_lost_update_is_refused() { test_lost_update_is_refused<transactional_trivial_map_t>(); }

static void commit_stamp_find_and_watch_records_absence() {
    test_find_and_watch_records_absence<transactional_trivial_map_t>();
}

#pragma endregion Commit Stamp

static void transactional_consistency_find_does_not_watch() {
    test_find_does_not_watch<transactional_trivial_map_t>();
    test_find_does_not_watch<transactional_composite_map_t>();
    test_find_does_not_watch<transactional_heavy_map_t>();
}

static void transactional_defects_transaction_equal_range_sees_staged_writes() {
    test_transaction_equal_range_sees_staged_writes<transactional_trivial_map_t>();
    test_transaction_equal_range_sees_staged_writes<transactional_composite_map_t>();
    test_transaction_equal_range_sees_staged_writes<transactional_heavy_map_t>();
}

static void transactional_defects_insert_reports_key_already_exists() {
    test_insert_reports_key_already_exists<transactional_trivial_map_t>();
    test_insert_reports_key_already_exists<transactional_composite_map_t>();
    test_insert_reports_key_already_exists<transactional_heavy_map_t>();
}

static void transactional_defects_find_copy_reports_key_not_found() {
    test_find_copy_reports_key_not_found<transactional_trivial_map_t>();
    test_find_copy_reports_key_not_found<transactional_composite_map_t>();
    test_find_copy_reports_key_not_found<transactional_heavy_map_t>();
}

static void transactional_defects_second_stage_is_rejected() {
    test_second_stage_is_rejected<transactional_trivial_map_t>();
    test_second_stage_is_rejected<transactional_composite_map_t>();
    test_second_stage_is_rejected<transactional_heavy_map_t>();
}

#pragma region Fixture Coverage

static void fixture_coverage_container_balances_counted_keys() {
    test_container_balances_counted_keys<monotonic_hash_map<counted_key_t, int>>();
}

static void fixture_coverage_rollback_balances_counted_keys() {
    test_rollback_balances_counted_keys<monotonic_hash_map<counted_key_t, int>>();
}

static void fixture_coverage_container_walks_collision_runs() {
    test_container_walks_collision_runs<monotonic_hash_map<colliding_key_t, int>>();
}

static void fixture_coverage_transaction_walks_collision_runs() {
    test_transaction_walks_collision_runs<monotonic_hash_map<colliding_key_t, int>>();
}

#pragma endregion Fixture Coverage

using budgeted_transactional_set_t = monotonic_hash_set<budgeted_key_t>;

static void fixture_coverage_find_copy_reports_a_refused_copy() {
    test_find_copy_reports_a_refused_copy<budgeted_transactional_set_t>();
}

int main() {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "point_access.insert_strategies", point_access_insert_strategies);
    failures +=
        run_test(filter, "point_access.enumeration_visits_every_member", point_access_enumeration_visits_every_member);
    failures += run_test(filter, "point_access.erase_visibility", point_access_erase_visibility);
    failures += run_test(filter, "point_access.rollback_restores_store", point_access_rollback_restores_store);
    failures += run_test(filter, "point_access.staging_survives_growth", point_access_staging_survives_growth);
    failures += run_test(filter, "point_access.insertion_patterns", point_access_insertion_patterns);
    failures += run_test(filter, "point_access.bulk_insertion_iterators", point_access_bulk_insertion_iterators);
    failures += run_test(filter, "point_access.bulk_upsert_with_duplicates", point_access_bulk_upsert_with_duplicates);

    failures += run_test(filter, "transactional_consistency.empty_transaction_commit",
                         transactional_consistency_empty_transaction_commit);
    failures += run_test(filter, "transactional_consistency.no_dirty_reads_multi_key",
                         transactional_consistency_no_dirty_reads_multi_key);
    failures += run_test(filter, "transactional_consistency.new_transaction_sees_nothing_staged",
                         transactional_consistency_new_transaction_sees_nothing_staged);
    failures += run_test(filter, "transactional_consistency.committed_immediately_visible",
                         transactional_consistency_committed_immediately_visible);
    failures += run_test(filter, "transactional_consistency.multi_key_atomicity_10_keys",
                         transactional_consistency_multi_key_atomicity_10_keys);
    failures += run_test(filter, "transactional_consistency.rollback_makes_all_invisible",
                         transactional_consistency_rollback_makes_all_invisible);
    failures += run_test(filter, "transactional_consistency.fractured_read_prevention",
                         transactional_consistency_fractured_read_prevention);
    failures += run_test(filter, "transactional_consistency.sequential_updates_never_regress",
                         transactional_consistency_sequential_updates_never_regress);
    failures += run_test(filter, "transactional_consistency.transaction_commits_maintain_order",
                         transactional_consistency_transaction_commits_maintain_order);
    failures += run_test(filter, "transactional_consistency.concurrent_transactions_on_same_key",
                         transactional_consistency_concurrent_transactions_on_same_key);
    failures += run_test(filter, "transactional_consistency.multi_key_conflict_any_key_fails",
                         transactional_consistency_multi_key_conflict_any_key_fails);
    failures += run_test(filter, "transactional_consistency.watch_detects_external_direct_modification",
                         transactional_consistency_watch_detects_external_direct_modification);
    failures += run_test(filter, "transactional_consistency.watch_detects_staged_invisible_writes",
                         transactional_consistency_watch_detects_staged_invisible_writes);
    failures += run_test(filter, "transactional_consistency.watch_on_erased_version_commits",
                         transactional_consistency_watch_on_erased_version_commits);
    failures += run_test(filter, "transactional_consistency.watch_detects_staged_writes_of_older_generation",
                         transactional_consistency_watch_detects_staged_writes_of_older_generation);
    failures += run_test(filter, "transactional_consistency.abandoned_transaction_leaves_no_trace",
                         transactional_consistency_abandoned_transaction_leaves_no_trace);
    failures += run_test(filter, "transactional_consistency.moved_transaction_unwinds_once",
                         transactional_consistency_moved_transaction_unwinds_once);
    failures += run_test(filter, "transactional_consistency.watch_on_erased_key_can_commit",
                         transactional_consistency_watch_on_erased_key_can_commit);
    failures += run_test(filter, "transactional_consistency.absent_watch_survives_rollback",
                         transactional_consistency_absent_watch_survives_rollback);
    failures += run_test(filter, "transactional_consistency.disjoint_keys_both_succeed",
                         transactional_consistency_disjoint_keys_both_succeed);
    failures += run_test(filter, "transactional_consistency.lost_update_matches_isolation",
                         transactional_consistency_lost_update_matches_isolation);
    failures += run_test(filter, "transactional_consistency.repeated_read_matches_isolation",
                         transactional_consistency_repeated_read_matches_isolation);
    failures += run_test(filter, "transactional_consistency.reset_clears_transaction_state",
                         transactional_consistency_reset_clears_transaction_state);

    failures += run_test(filter, "transactional_defects.validate_refuses_before_publishing",
                         [] { test_validate_refuses_before_publishing<transactional_trivial_map_t>(); });
    failures += run_test(filter, "transactional_defects.publish_cannot_refuse",
                         [] { test_publish_cannot_refuse<transactional_trivial_map_t>(); });
    failures += run_test(filter, "transactional_defects.clear_refuses_while_staged",
                         [] { test_clear_refuses_while_staged<transactional_trivial_map_t>(); });
    failures += run_test(filter, "transactional_defects.direct_write_spares_staged_version",
                         transactional_defects_direct_write_spares_staged_version);
    failures += run_test(filter, "transactional_defects.direct_erase_spares_staged_version",
                         transactional_defects_direct_erase_spares_staged_version);
    failures += run_test(filter, "transactional_defects.clear_keeps_generations_moving",
                         transactional_defects_clear_keeps_generations_moving);
    failures += run_test(filter, "transactional_defects.committed_erase_hidden_from_point_reads",
                         transactional_defects_committed_erase_hidden_from_point_reads);
    failures += run_test(filter, "reserve.reaches_the_slab", test_reserve_reaches_the_slab);

    failures += run_test(filter, "transactional_defects.vacuum_reclaims_committed_tombstones",
                         transactional_defects_vacuum_reclaims_committed_tombstones);
    failures += run_test(filter, "transactional_defects.vacuum_spares_staged_versions",
                         transactional_defects_vacuum_spares_staged_versions);
    failures += run_test(filter, "transactional_consistency.group_commits_participants_together",
                         transactional_consistency_group_commits_participants_together);
    failures += run_test(filter, "transactional_consistency.group_unwinds_every_participant_on_conflict",
                         transactional_consistency_group_unwinds_every_participant_on_conflict);

    failures += run_test(filter, "commit_stamp.rolled_back_stage_does_not_abort_a_peer",
                         commit_stamp_rolled_back_stage_does_not_abort_a_peer);
    failures += run_test(filter, "commit_stamp.lost_update_is_refused", commit_stamp_lost_update_is_refused);
    failures +=
        run_test(filter, "commit_stamp.find_and_watch_records_absence", commit_stamp_find_and_watch_records_absence);

    failures += run_test(filter, "transactional_consistency.find_does_not_watch",
                         transactional_consistency_find_does_not_watch);

    failures += run_test(filter, "transactional_defects.transaction_equal_range_sees_staged_writes",
                         transactional_defects_transaction_equal_range_sees_staged_writes);
    failures += run_test(filter, "transactional_defects.insert_reports_key_already_exists",
                         transactional_defects_insert_reports_key_already_exists);
    failures += run_test(filter, "transactional_defects.find_copy_reports_key_not_found",
                         transactional_defects_find_copy_reports_key_not_found);
    failures += run_test(filter, "transactional_defects.second_stage_is_rejected",
                         transactional_defects_second_stage_is_rejected);

    failures += run_test(filter, "fixture_coverage.container_balances_counted_keys",
                         fixture_coverage_container_balances_counted_keys);
    failures += run_test(filter, "fixture_coverage.rollback_balances_counted_keys",
                         fixture_coverage_rollback_balances_counted_keys);
    failures += run_test(filter, "fixture_coverage.container_walks_collision_runs",
                         fixture_coverage_container_walks_collision_runs);
    failures += run_test(filter, "fixture_coverage.transaction_walks_collision_runs",
                         fixture_coverage_transaction_walks_collision_runs);

    failures += run_test(filter, "fixture_coverage.find_copy_reports_a_refused_copy",
                         fixture_coverage_find_copy_reports_a_refused_copy);

    return report_test_failures(failures);
}
