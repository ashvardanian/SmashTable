/**
 *  @file test/monotonic_hash.cpp
 *  @author Ash Vardanian
 *  @date August 17, 2026
 *  @brief Test instantiations for the transactional store over an open-addressed hash table, which
 *      answers points rather than windows.
 *
 *  The point-access surface is all an unordered core supplies - insert, upsert, update, erase,
 *  find, watch and the two-phase commit - so there are no bounds, no ranges and no order statistics
 *  here. Around it run the commit-stamp, fixture-coverage and transactional-consistency families.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`

#include <vector> // `std::vector`

#include <smashtable/monotonic_store.hpp>

#include "harness.hpp"
#include "fixtures.hpp"
#include "commit_stamp.hpp"
#include "consistency.hpp"
#include "fixture_coverage.hpp"
#include "monotonic_store_defects.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Type Aliases

/**
 *  Tests: Baseline transactional correctness over the cheapest possible key.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Transaction: ✓.
 */
using transactional_trivial_set_t = monotonic_hash_set<trivial_key_t>;

/**
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() → expected<T>.
 *  Memory: Heap |.
 *  Transaction: ✓ Tests: Watch copy OOM, rollback with a heap-allocating key.
 */
using transactional_heavy_set_t = monotonic_hash_set<heavy_key_t>;

/**
 *  Tests: Transactional map operations, value overwrites.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Stack.
 *  Transaction: ✓.
 *  Value: int.
 */
using transactional_trivial_map_t = monotonic_hash_map<trivial_key_t, int>;

/**
 *  Tests: Rollback with non-trivial values.
 *  Heterogeneous lookup: ✗.
 *  Copy: Key trivial, value .copy().
 *  Memory: Heap, over @c value Transaction: ✓ Value: guarded_payload_t.
 */
using transactional_composite_map_t = monotonic_hash_map<composite_key_t, guarded_payload_t>;

/**
 *  Tests: Dual-heap staging and rollback.
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() on key & value.
 *  Memory: Heap, over @c both |.
 *  Transaction: ✓ Value: guarded_payload_t.
 */
using transactional_heavy_map_t = monotonic_hash_map<heavy_key_t, guarded_payload_t>;

#pragma endregion Type Aliases

#pragma region Helpers

/** Where @p member's key sits in @p keys, or one slot past the last where none of them is it. */
template <typename member_type_, typename key_type_>
static std::size_t key_slot(member_type_ const &member, std::vector<key_type_> const &keys) noexcept {
    for (std::size_t slot = 0; slot != keys.size(); ++slot)
        if (mapping_key_or_itself<member_type_>(member) == keys[slot]) return slot;
    return keys.size();
}

#pragma endregion Helpers

#pragma region Point Access Tests

/** Tests that the four insert strategies differ exactly as documented */
template <typename container_type_>
static void check_point_insert_strategies() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    auto [first, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;

    expect_equal(first.insert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(first.insert(trivial_id_to_member<member_t>(1)), status_t::key_already_exists_k);
    expect_equal(first.insert_if_missing(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(first.update(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(first.update(trivial_id_to_member<member_t>(2)), status_t::key_not_found_k);

    expect_equal(first.stage(), success_k);
    expect_equal(first.commit(), success_k);
    expect_equal(container.size(), 1);
    expect_equal(container.contains(trivial_id_to_key<member_t>(1)), true);
    expect_equal(container.contains(trivial_id_to_key<member_t>(2)), false);
}

/** Tests that an erase staged in a transaction only becomes visible on commit */
template <typename container_type_>
static void check_point_erase_visibility() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (std::size_t index = 0; index < 8; ++index)
        expect_equal(container.upsert(trivial_id_to_member<member_t>(index)), success_k);
    expect_equal(container.size(), 8);

    auto [erasing, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    for (std::size_t index = 0; index < 4; ++index)
        expect_equal(erasing.erase(trivial_id_to_key<member_t>(index)), success_k);
    expect_equal(erasing.stage(), success_k);
    for (std::size_t index = 0; index < 4; ++index) {
        expect_equal(container.contains(trivial_id_to_key<member_t>(index)), true);
    }

    expect_equal(erasing.commit(), success_k);
    for (std::size_t index = 0; index < 4; ++index) {
        expect_equal(container.contains(trivial_id_to_key<member_t>(index)), false);
    }
    for (std::size_t index = 4; index < 8; ++index) {
        expect_equal(container.contains(trivial_id_to_key<member_t>(index)), true);
    }
}

/** Tests that a rollback pulls every staged version back and leaves the store as it was */
template <typename container_type_>
static void check_point_rollback_restores_store() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1)), success_k);

    auto [writing, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    for (std::size_t index = 1; index <= 5; ++index)
        expect_equal(writing.upsert(trivial_id_to_member<member_t>(index)), success_k);
    expect_equal(writing.stage(), success_k);
    expect_equal(writing.rollback(), success_k);

    expect_equal(container.size(), 1);
    expect_equal(container.contains(trivial_id_to_key<member_t>(1)), true);
    for (std::size_t index = 2; index <= 5; ++index) {
        expect_equal(container.contains(trivial_id_to_key<member_t>(index)), false);
    }

    // The rolled-back versions are still the transaction's, so a second staging lands them all
    expect_equal(writing.stage(), success_k);
    expect_equal(writing.commit(), success_k);
    expect_equal(container.size(), 5);
}

/** Tests that staging enough keys to outgrow the slab several times loses none of them */
template <typename container_type_>
static void check_point_staging_survives_growth(std::size_t size = 500) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    auto [writing, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    for (std::size_t index = 0; index < size; ++index)
        expect_equal(writing.upsert(trivial_id_to_member<member_t>(index)), success_k);
    expect_equal(writing.stage(), success_k);
    expect_equal(writing.commit(), success_k);

    expect_equal(container.size(), size);
    for (std::size_t index = 0; index < size; ++index) {
        expect_equal(container.contains(trivial_id_to_key<member_t>(index)), true);
    }
}

/** Tests that the unordered walk reports every member once, over a store and over a transaction */
template <typename container_type_>
static void check_point_enumeration_visits_every_member() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using key_t = decltype(trivial_id_to_key<member_t>(0));
    std::size_t const size = 8;

    container_t container;
    std::size_t empty_visits = 0;
    expect_equal(container.for_each([&](member_t const &) noexcept { ++empty_visits; }), success_k);
    expect_equal(empty_visits, 0);

    // The keys are materialized up front, since a heap-allocating one cannot be built inside a
    // `noexcept` callback, and a tally of one per key is what proves nothing was reported twice.
    std::vector<key_t> keys;
    for (std::size_t index = 0; index < size; ++index) {
        keys.push_back(trivial_id_to_key<member_t>(index));
        expect_equal(container.upsert(trivial_id_to_member<member_t>(index)), success_k);
    }

    // One slot per key, plus a trailing one for whatever no key claims.
    std::vector<std::size_t> tally(size + 1, 0);
    std::size_t visits = 0;
    expect_equal(container.for_each([&](member_t const &member) noexcept {
        ++visits;
        ++tally[key_slot<member_t>(member, keys)];
    }),
                 success_k);
    expect_equal(visits, container.size());
    for (std::size_t index = 0; index < size; ++index) expect_equal(tally[index], 1);

    // A transaction reports its own staged writes and hides its own tombstones.
    auto [writing, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(writing.erase(trivial_id_to_key<member_t>(0)), success_k);
    expect_equal(writing.upsert(trivial_id_to_member<member_t>(size)), success_k);

    std::vector<std::size_t> staged_tally(size + 1, 0);
    std::size_t staged_visits = 0;
    expect_equal(writing.for_each([&](member_t const &member) noexcept {
        ++staged_visits;
        ++staged_tally[key_slot<member_t>(member, keys)];
    }),
                 success_k);
    expect_equal(staged_visits, size);
    expect_equal(staged_tally[0], 0);
    for (std::size_t index = 1; index < size; ++index) expect_equal(staged_tally[index], 1);

    // Nothing the transaction staged is published, so the store still reports what it did before.
    expect_equal(writing.stage(), success_k);
    std::size_t published_visits = 0;
    expect_equal(container.for_each([&](member_t const &) noexcept { ++published_visits; }), success_k);
    expect_equal(published_visits, size);
}

static void test_enumeration_visits_every_member() {
    check_point_enumeration_visits_every_member<transactional_trivial_set_t>();
    check_point_enumeration_visits_every_member<transactional_heavy_set_t>();
    check_point_enumeration_visits_every_member<transactional_trivial_map_t>();
    check_point_enumeration_visits_every_member<transactional_composite_map_t>();
    check_point_enumeration_visits_every_member<transactional_heavy_map_t>();
}

static void test_insert_strategies() {
    check_point_insert_strategies<transactional_trivial_set_t>();
    check_point_insert_strategies<transactional_heavy_set_t>();
    check_point_insert_strategies<transactional_trivial_map_t>();
    check_point_insert_strategies<transactional_composite_map_t>();
    check_point_insert_strategies<transactional_heavy_map_t>();
}

static void test_erase_visibility() {
    check_point_erase_visibility<transactional_trivial_set_t>();
    check_point_erase_visibility<transactional_heavy_set_t>();
    check_point_erase_visibility<transactional_trivial_map_t>();
    check_point_erase_visibility<transactional_composite_map_t>();
    check_point_erase_visibility<transactional_heavy_map_t>();
}

static void test_rollback_restores_store() {
    check_point_rollback_restores_store<transactional_trivial_set_t>();
    check_point_rollback_restores_store<transactional_heavy_set_t>();
    check_point_rollback_restores_store<transactional_trivial_map_t>();
    check_point_rollback_restores_store<transactional_composite_map_t>();
    check_point_rollback_restores_store<transactional_heavy_map_t>();
}

static void test_staging_survives_growth() {
    check_point_staging_survives_growth<transactional_trivial_set_t>();
    check_point_staging_survives_growth<transactional_heavy_set_t>();
    check_point_staging_survives_growth<transactional_trivial_map_t>();
    check_point_staging_survives_growth<transactional_composite_map_t>();
    check_point_staging_survives_growth<transactional_heavy_map_t>();
}

static void test_insertion_patterns() {
    check_basic_insertion_patterns<transactional_trivial_set_t>();
    check_basic_insertion_patterns<transactional_heavy_set_t>();
    check_basic_insertion_patterns<transactional_trivial_map_t>();
    check_basic_insertion_patterns<transactional_composite_map_t>();
    check_basic_insertion_patterns<transactional_heavy_map_t>();
}

static void test_bulk_insertion_iterators() {
    check_bulk_insertion_from_iterators<transactional_trivial_set_t>();
    check_bulk_insertion_from_iterators<transactional_heavy_set_t>();
    check_bulk_insertion_from_iterators<transactional_trivial_map_t>();
    check_bulk_insertion_from_iterators<transactional_composite_map_t>();
    check_bulk_insertion_from_iterators<transactional_heavy_map_t>();
}

static void test_bulk_upsert_with_duplicates() { //
    check_bulk_upsert_with_duplicates<transactional_trivial_map_t>();
}

#pragma endregion Point Access Tests

#pragma region Consistency and Transaction Tests

static void test_empty_transaction_commit() {
    check_empty_transaction_commit<transactional_trivial_set_t>();
    check_empty_transaction_commit<transactional_heavy_set_t>();
    check_empty_transaction_commit<transactional_trivial_map_t>();
    check_empty_transaction_commit<transactional_composite_map_t>();
    check_empty_transaction_commit<transactional_heavy_map_t>();
}

static void test_no_dirty_reads_multi_key() {
    check_no_dirty_reads_multi_key<transactional_trivial_map_t>();
    check_no_dirty_reads_multi_key<transactional_composite_map_t>();
    check_no_dirty_reads_multi_key<transactional_heavy_map_t>();
}

static void test_new_transaction_sees_nothing_staged() {
    check_new_transaction_sees_nothing_staged<transactional_trivial_set_t>();
    check_new_transaction_sees_nothing_staged<transactional_heavy_set_t>();
    check_new_transaction_sees_nothing_staged<transactional_trivial_map_t>();
    check_new_transaction_sees_nothing_staged<transactional_composite_map_t>();
    check_new_transaction_sees_nothing_staged<transactional_heavy_map_t>();
}

static void test_committed_immediately_visible() {
    check_committed_immediately_visible<transactional_trivial_map_t>();
    check_committed_immediately_visible<transactional_composite_map_t>();
    check_committed_immediately_visible<transactional_heavy_map_t>();
}

static void test_multi_key_atomicity_10_keys() {
    check_multi_key_atomicity_10_keys<transactional_trivial_map_t>();
    check_multi_key_atomicity_10_keys<transactional_composite_map_t>();
    check_multi_key_atomicity_10_keys<transactional_heavy_map_t>();
}

static void test_rollback_makes_all_invisible() {
    check_rollback_makes_all_invisible<transactional_trivial_map_t>();
    check_rollback_makes_all_invisible<transactional_composite_map_t>();
    check_rollback_makes_all_invisible<transactional_heavy_map_t>();
}

static void test_fractured_read_prevention() {
    check_fractured_read_prevention<transactional_trivial_map_t>();
    check_fractured_read_prevention<transactional_composite_map_t>();
    check_fractured_read_prevention<transactional_heavy_map_t>();
}

/** Ordering of mapped values is only meaningful where they are numbers, so @c int maps only. */
static void test_sequential_updates_never_regress() {
    check_sequential_updates_never_regress<transactional_trivial_map_t>();
}

/** Ordering of mapped values is only meaningful where they are numbers, so @c int maps only. */
static void test_transaction_commits_maintain_order() {
    check_transaction_commits_maintain_order<transactional_trivial_map_t>();
}

static void test_concurrent_transactions_on_same_key() {
    check_concurrent_transactions_on_same_key<transactional_trivial_map_t>();
    check_concurrent_transactions_on_same_key<transactional_composite_map_t>();
    check_concurrent_transactions_on_same_key<transactional_heavy_map_t>();
}

static void test_multi_key_conflict_any_key_fails() {
    check_multi_key_conflict_any_key_fails<transactional_trivial_map_t>();
    check_multi_key_conflict_any_key_fails<transactional_composite_map_t>();
    check_multi_key_conflict_any_key_fails<transactional_heavy_map_t>();
}

static void test_watch_detects_external_direct_modification() {
    check_watch_detects_external_direct_modification<transactional_trivial_map_t>();
    check_watch_detects_external_direct_modification<transactional_composite_map_t>();
    check_watch_detects_external_direct_modification<transactional_heavy_map_t>();
}

static void test_watch_detects_staged_invisible_writes() {
    check_watch_detects_staged_invisible_writes<transactional_trivial_map_t>();
    check_watch_detects_staged_invisible_writes<transactional_composite_map_t>();
    check_watch_detects_staged_invisible_writes<transactional_heavy_map_t>();
}

static void test_watch_detects_staged_writes_of_older_generation() {
    check_watch_detects_staged_writes_of_older_generation<transactional_trivial_map_t>();
    check_watch_detects_staged_writes_of_older_generation<transactional_composite_map_t>();
    check_watch_detects_staged_writes_of_older_generation<transactional_heavy_map_t>();
}

static void test_abandoned_transaction_leaves_no_trace() {
    check_abandoned_transaction_leaves_no_trace<transactional_trivial_map_t>();
    check_abandoned_transaction_leaves_no_trace<transactional_composite_map_t>();
    check_abandoned_transaction_leaves_no_trace<transactional_heavy_map_t>();
}

static void test_moved_transaction_unwinds_once() {
    check_moved_transaction_unwinds_once<transactional_trivial_map_t>();
    check_moved_transaction_unwinds_once<transactional_composite_map_t>();
    check_moved_transaction_unwinds_once<transactional_heavy_map_t>();
}

static void test_watch_on_erased_key_can_commit() {
    check_watch_on_erased_key_can_commit<transactional_trivial_map_t>();
    check_watch_on_erased_key_can_commit<transactional_composite_map_t>();
    check_watch_on_erased_key_can_commit<transactional_heavy_map_t>();
}

static void test_watch_catches_an_insert_and_erase_under_it() {
    check_watch_catches_an_insert_and_erase_under_it<transactional_trivial_map_t>();
    check_watch_catches_an_insert_and_erase_under_it<transactional_composite_map_t>();
    check_watch_catches_an_insert_and_erase_under_it<transactional_heavy_map_t>();
}

static void test_absent_watch_survives_rollback() {
    check_absent_watch_survives_rollback<transactional_trivial_map_t>();
    check_absent_watch_survives_rollback<transactional_composite_map_t>();
    check_absent_watch_survives_rollback<transactional_heavy_map_t>();
}

static void test_disjoint_keys_both_succeed() {
    check_disjoint_keys_both_succeed<transactional_trivial_map_t>();
    check_disjoint_keys_both_succeed<transactional_composite_map_t>();
    check_disjoint_keys_both_succeed<transactional_heavy_map_t>();
}

static void test_lost_update_matches_isolation() {
    check_lost_update_matches_isolation<transactional_trivial_map_t>();
    check_lost_update_matches_isolation<transactional_composite_map_t>();
    check_lost_update_matches_isolation<transactional_heavy_map_t>();
}

static void test_repeated_read_matches_isolation() {
    check_repeated_read_matches_isolation<transactional_trivial_map_t>();
    check_repeated_read_matches_isolation<transactional_composite_map_t>();
    check_repeated_read_matches_isolation<transactional_heavy_map_t>();
}

/** Only the map with an arithmetic value: both anomalies total the mapped values to see whether the
 *  invariant survived, which @c guarded_payload_t cannot answer. */
static void test_write_skew_matches_isolation() { check_write_skew_matches_isolation<transactional_trivial_map_t>(); }

static void test_read_conflict_matches_isolation() {
    check_read_conflict_matches_isolation<transactional_trivial_map_t>();
}

static void test_reset_clears_transaction_state() {
    check_reset_clears_transaction_state<transactional_trivial_map_t>();
    check_reset_clears_transaction_state<transactional_composite_map_t>();
    check_reset_clears_transaction_state<transactional_heavy_map_t>();
}

/**
 *  @brief Tests that a watch on an erased version records the shape validation re-derives.
 *
 *  The tombstone is built by hand because the public surface hides one: a caller that kept the
 *  version it erased holds exactly this, and handing it back must not doom every later commit.
 */
template <typename container_type_>
static void check_watch_on_erased_version_commits() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using versioned_entry_t = typename container_t::versioned_entry_t;

    container_t container;
    {
        auto [writer, writer_status] = container.transaction();
        if (!expect_equal(writer_status, success_k)) return;
        expect_equal(writer.insert(trivial_id_to_member<member_t>(1)), success_k);
        expect_equal(writer.stage(), success_k);
        expect_equal(writer.commit(), success_k);
    }

    generation_t erased_generation = 0;
    {
        auto [eraser, eraser_status] = container.transaction();
        if (!expect_equal(eraser_status, success_k)) return;
        expect_equal(eraser.erase(trivial_id_to_key<member_t>(1)), success_k);
        erased_generation = eraser.generation();
        expect_equal(eraser.stage(), success_k);
        expect_equal(eraser.commit(), success_k);
    }
    expect_equal(container.contains(trivial_id_to_key<member_t>(1)), false);

    auto [watcher, watcher_status] = container.transaction();
    if (!expect_equal(watcher_status, success_k)) return;
    versioned_entry_t tombstone {trivial_id_to_member<member_t>(1)};
    tombstone.generation = erased_generation;
    tombstone.presence = presence_t::erased_k;
    expect_equal(watcher.watch(tombstone), success_k);
    expect_equal(watcher.insert(trivial_id_to_member<member_t>(2)), success_k);
    expect_equal(watcher.stage(), success_k);
    expect_equal(watcher.commit(), success_k);
    expect_equal(container.contains(trivial_id_to_key<member_t>(2)), true);
}

static void test_watch_on_erased_version_commits() {
    check_watch_on_erased_version_commits<transactional_trivial_set_t>();
}

#pragma endregion Consistency and Transaction Tests

#pragma region Transactional Store Defects

static void test_direct_write_spares_staged_version() {
    check_direct_write_spares_staged_version<transactional_trivial_map_t>();
    check_direct_write_spares_staged_version<transactional_composite_map_t>();
    check_direct_write_spares_staged_version<transactional_heavy_map_t>();
}

static void test_direct_erase_spares_staged_version() {
    check_direct_erase_spares_staged_version<transactional_trivial_map_t>();
    check_direct_erase_spares_staged_version<transactional_composite_map_t>();
    check_direct_erase_spares_staged_version<transactional_heavy_map_t>();
}

static void test_clear_keeps_generations_moving() {
    check_clear_keeps_generations_moving<transactional_trivial_map_t>();
    check_clear_keeps_generations_moving<transactional_composite_map_t>();
    check_clear_keeps_generations_moving<transactional_heavy_map_t>();
}

static void test_committed_erase_hidden_from_point_reads() {
    check_committed_erase_hidden_from_point_reads<transactional_trivial_map_t>();
    check_committed_erase_hidden_from_point_reads<transactional_composite_map_t>();
    check_committed_erase_hidden_from_point_reads<transactional_heavy_map_t>();
}

static void test_vacuum_reclaims_committed_tombstones() {
    check_vacuum_reclaims_committed_tombstones<transactional_trivial_map_t>();
    check_vacuum_reclaims_committed_tombstones<transactional_composite_map_t>();
    check_vacuum_reclaims_committed_tombstones<transactional_heavy_map_t>();
}

static void test_vacuum_spares_staged_versions() {
    check_vacuum_spares_staged_versions<transactional_trivial_map_t>();
    check_vacuum_spares_staged_versions<transactional_composite_map_t>();
    check_vacuum_spares_staged_versions<transactional_heavy_map_t>();
}

static void test_group_commits_participants_together() {
    check_group_commits_participants_together<transactional_trivial_map_t>();
    check_group_commits_participants_together<transactional_composite_map_t>();
    check_group_commits_participants_together<transactional_heavy_map_t>();
}

static void test_group_unwinds_every_participant_on_conflict() {
    check_group_unwinds_every_participant_on_conflict<transactional_trivial_map_t>();
    check_group_unwinds_every_participant_on_conflict<transactional_composite_map_t>();
    check_group_unwinds_every_participant_on_conflict<transactional_heavy_map_t>();
}

#pragma endregion Transactional Store Defects

#pragma region Reserve

/**
 *  @brief A capacity hint reaches the slab the table grows, rather than being swallowed.
 *
 *  The ledger is what makes the difference observable: a hint that is honoured asks the allocator
 *  for a slab before any element is written, and a hint that is merely accepted asks for nothing.
 */
static void test_reserve_reaches_the_slab() {
    using member_t = mapping<trivial_key_t, int>;
    using ledgered_map_t = monotonic_hash_map<trivial_key_t, int, default_hash_t, equal_to_t, stateful_allocator_t>;
    using ledgered_entry_t = typename ledgered_map_t::versioned_entry_t;
    constexpr std::size_t reserved_k = 64;

    allocation_ledger_t ledger;
    {
        ledgered_map_t container {stateful_allocator_t {ledger}};
        expect_equal(ledger.allocations, std::size_t {0});

        expect_equal(container.reserve(reserved_k), success_k);
        expect_equal(ledger.allocations, std::size_t {1});
        // The allocator hands out bytes, so the one slab has to cover what the reserved entries occupy.
        expect_true(ledger.largest_request_elements() >= reserved_k * sizeof(ledgered_entry_t));

        for (trivial_id_t identifier = 0; identifier != reserved_k; ++identifier)
            expect_equal(container.upsert(trivial_id_to_member<member_t>(identifier, 1)), success_k);
        expect_equal(container.size(), reserved_k);
        // Honoured rather than merely accepted: the writes the hint covered grew nothing.
        expect_equal(ledger.allocations, std::size_t {1});
    }
    ledger.check_balanced();

    // A slab the budget refuses is reported rather than thrown, which is the whole point of a status.
    allocation_ledger_t starved;
    starved.refuse_everything();
    {
        ledgered_map_t container {stateful_allocator_t {starved}};
        expect_equal(container.reserve(reserved_k), status_t::out_of_memory_heap_k);
        expect_equal(container.size(), 0);
    }
    starved.check_balanced();
}

#pragma endregion Reserve

#pragma region Commit Stamp

static void test_rolled_back_stage_does_not_abort_a_peer() {
    check_rolled_back_stage_does_not_abort_a_peer<transactional_trivial_map_t>();
}

static void test_lost_update_is_refused() { check_lost_update_is_refused<transactional_trivial_map_t>(); }

static void test_find_and_watch_records_absence() {
    check_find_and_watch_records_absence<transactional_trivial_map_t>();
}

#pragma endregion Commit Stamp

static void test_find_does_not_watch() {
    check_find_does_not_watch<transactional_trivial_map_t>();
    check_find_does_not_watch<transactional_composite_map_t>();
    check_find_does_not_watch<transactional_heavy_map_t>();
}

static void test_transaction_equal_range_sees_staged_writes() {
    check_transaction_equal_range_sees_staged_writes<transactional_trivial_map_t>();
    check_transaction_equal_range_sees_staged_writes<transactional_composite_map_t>();
    check_transaction_equal_range_sees_staged_writes<transactional_heavy_map_t>();
}

static void test_insert_reports_key_already_exists() {
    check_insert_reports_key_already_exists<transactional_trivial_map_t>();
    check_insert_reports_key_already_exists<transactional_composite_map_t>();
    check_insert_reports_key_already_exists<transactional_heavy_map_t>();
}

static void test_find_copy_reports_key_not_found() {
    check_find_copy_reports_key_not_found<transactional_trivial_map_t>();
    check_find_copy_reports_key_not_found<transactional_composite_map_t>();
    check_find_copy_reports_key_not_found<transactional_heavy_map_t>();
}

static void test_second_stage_is_rejected() {
    check_second_stage_is_rejected<transactional_trivial_map_t>();
    check_second_stage_is_rejected<transactional_composite_map_t>();
    check_second_stage_is_rejected<transactional_heavy_map_t>();
}

#pragma region Fixture Coverage

static void test_container_balances_counted_keys() {
    check_container_balances_counted_keys<monotonic_hash_map<counted_key_t, int>>();
}

static void test_rollback_balances_counted_keys() {
    check_rollback_balances_counted_keys<monotonic_hash_map<counted_key_t, int>>();
}

static void test_container_walks_collision_runs() {
    check_container_walks_collision_runs<monotonic_hash_map<colliding_key_t, int>>();
}

static void test_transaction_walks_collision_runs() {
    check_transaction_walks_collision_runs<monotonic_hash_map<colliding_key_t, int>>();
}

#pragma endregion Fixture Coverage

using budgeted_transactional_set_t = monotonic_hash_set<budgeted_key_t>;

static void test_find_copy_reports_a_refused_copy() {
    check_find_copy_reports_a_refused_copy<budgeted_transactional_set_t>();
}

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

int main(int argc, char **argv) {
    start_test_run(argc, argv);

    run_test_case("test_insert_strategies", test_insert_strategies);
    run_test_case("test_enumeration_visits_every_member", test_enumeration_visits_every_member);
    run_test_case("test_erase_visibility", test_erase_visibility);
    run_test_case("test_rollback_restores_store", test_rollback_restores_store);
    run_test_case("test_staging_survives_growth", test_staging_survives_growth);
    run_test_case("test_insertion_patterns", test_insertion_patterns);
    run_test_case("test_bulk_insertion_iterators", test_bulk_insertion_iterators);
    run_test_case("test_bulk_upsert_with_duplicates", test_bulk_upsert_with_duplicates);

    run_test_case("test_empty_transaction_commit", test_empty_transaction_commit);
    run_test_case("test_no_dirty_reads_multi_key", test_no_dirty_reads_multi_key);
    run_test_case("test_new_transaction_sees_nothing_staged", test_new_transaction_sees_nothing_staged);
    run_test_case("test_committed_immediately_visible", test_committed_immediately_visible);
    run_test_case("test_multi_key_atomicity_10_keys", test_multi_key_atomicity_10_keys);
    run_test_case("test_rollback_makes_all_invisible", test_rollback_makes_all_invisible);
    run_test_case("test_fractured_read_prevention", test_fractured_read_prevention);
    run_test_case("test_sequential_updates_never_regress", test_sequential_updates_never_regress);
    run_test_case("test_transaction_commits_maintain_order", test_transaction_commits_maintain_order);
    run_test_case("test_concurrent_transactions_on_same_key", test_concurrent_transactions_on_same_key);
    run_test_case("test_multi_key_conflict_any_key_fails", test_multi_key_conflict_any_key_fails);
    run_test_case("test_watch_detects_external_direct_modification", test_watch_detects_external_direct_modification);
    run_test_case("test_watch_detects_staged_invisible_writes", test_watch_detects_staged_invisible_writes);
    run_test_case("test_watch_on_erased_version_commits", test_watch_on_erased_version_commits);
    run_test_case("test_watch_detects_staged_writes_of_older_generation",
                  test_watch_detects_staged_writes_of_older_generation);
    run_test_case("test_abandoned_transaction_leaves_no_trace", test_abandoned_transaction_leaves_no_trace);
    run_test_case("test_moved_transaction_unwinds_once", test_moved_transaction_unwinds_once);
    run_test_case("test_watch_on_erased_key_can_commit", test_watch_on_erased_key_can_commit);
    run_test_case("test_watch_catches_an_insert_and_erase_under_it", test_watch_catches_an_insert_and_erase_under_it);
    run_test_case("test_absent_watch_survives_rollback", test_absent_watch_survives_rollback);
    run_test_case("test_disjoint_keys_both_succeed", test_disjoint_keys_both_succeed);
    run_test_case("test_lost_update_matches_isolation", test_lost_update_matches_isolation);
    run_test_case("test_repeated_read_matches_isolation", test_repeated_read_matches_isolation);
    run_test_case("test_write_skew_matches_isolation", test_write_skew_matches_isolation);
    run_test_case("test_read_conflict_matches_isolation", test_read_conflict_matches_isolation);
    run_test_case("test_reset_clears_transaction_state", test_reset_clears_transaction_state);

    run_test_case("test_validate_refuses_before_publishing<transactional_trivial_map_t>",
                  test_validate_refuses_before_publishing<transactional_trivial_map_t>);
    run_test_case("test_publish_cannot_refuse<transactional_trivial_map_t>",
                  test_publish_cannot_refuse<transactional_trivial_map_t>);
    run_test_case("test_clear_refuses_while_staged<transactional_trivial_map_t>",
                  test_clear_refuses_while_staged<transactional_trivial_map_t>);
    run_test_case("test_direct_write_spares_staged_version", test_direct_write_spares_staged_version);
    run_test_case("test_direct_erase_spares_staged_version", test_direct_erase_spares_staged_version);
    run_test_case("test_clear_keeps_generations_moving", test_clear_keeps_generations_moving);
    run_test_case("test_committed_erase_hidden_from_point_reads", test_committed_erase_hidden_from_point_reads);
    run_test_case("test_reserve_reaches_the_slab", test_reserve_reaches_the_slab);

    run_test_case("test_vacuum_reclaims_committed_tombstones", test_vacuum_reclaims_committed_tombstones);
    run_test_case("test_vacuum_spares_staged_versions", test_vacuum_spares_staged_versions);
    run_test_case("test_group_commits_participants_together", test_group_commits_participants_together);
    run_test_case("test_group_unwinds_every_participant_on_conflict", test_group_unwinds_every_participant_on_conflict);

    run_test_case("test_rolled_back_stage_does_not_abort_a_peer", test_rolled_back_stage_does_not_abort_a_peer);
    run_test_case("test_lost_update_is_refused", test_lost_update_is_refused);
    run_test_case("test_find_and_watch_records_absence", test_find_and_watch_records_absence);

    run_test_case("test_find_does_not_watch", test_find_does_not_watch);

    run_test_case("test_transaction_equal_range_sees_staged_writes", test_transaction_equal_range_sees_staged_writes);
    run_test_case("test_insert_reports_key_already_exists", test_insert_reports_key_already_exists);
    run_test_case("test_find_copy_reports_key_not_found", test_find_copy_reports_key_not_found);
    run_test_case("test_second_stage_is_rejected", test_second_stage_is_rejected);

    run_test_case("test_container_balances_counted_keys", test_container_balances_counted_keys);
    run_test_case("test_rollback_balances_counted_keys", test_rollback_balances_counted_keys);
    run_test_case("test_container_walks_collision_runs", test_container_walks_collision_runs);
    run_test_case("test_transaction_walks_collision_runs", test_transaction_walks_collision_runs);

    run_test_case("test_find_copy_reports_a_refused_copy", test_find_copy_reports_a_refused_copy);

    return finish_test_run();
}
