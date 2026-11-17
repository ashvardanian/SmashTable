/**
 *  @file test/std_set.cpp
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Test instantiations for the @c std::set-backed transactional store. The baseline
 *      reference design, exercised by the same suites as the tree containers.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <smashtable/reference_store.hpp>

#include "harness.hpp"
#include "fixtures.hpp"
#include "commit_stamp.hpp"
#include "consistency.hpp"
#include "reference_store_defects.hpp"
#include "monotonic_store_defects.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Type Aliases

/**
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 */
using transactional_trivial_set_t =
    reference_store<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  Heterogeneous lookup: ✓.
 *  Copy: Trivial.
 *  Memory: Tracked.
 */
using transactional_tracking_set_t = reference_store<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓, over @c uint64_t.
 *  Copy: Trivial.
 *  Memory: Stack.
 */
using transactional_composite_set_t =
    reference_store<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/**
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() → expected<T>.
 *  Memory: Heap.
 */
using transactional_heavy_set_t = reference_store<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/**
 *  Value: int.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Stack.
 */
using transactional_trivial_map_t =
    reference_store<mapping<trivial_key_t, int>, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Value: int.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Tracked.
 */
using transactional_tracking_map_t =
    reference_store<mapping<trivial_key_t, int>, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Value: guarded_payload_t.
 *  Copy: Key trivial, value .copy().
 *  Memory: Heap, over @c value.
 */
using transactional_composite_map_t =
    reference_store<mapping<composite_key_t, guarded_payload_t>, composite_key_compare_t,
                    std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/**
 *  Value: guarded_payload_t.
 *  Copy: .copy() on key & value.
 *  Memory: Heap, over @c both.
 */
using transactional_heavy_map_t = reference_store<mapping<heavy_key_t, guarded_payload_t>, std::less<void>,
                                                  std::allocator<mapping<heavy_key_t, guarded_payload_t>>>;

#pragma endregion Type Aliases

#pragma region Basic Operations Tests

/** Tests operations on empty container don't crash */
static void test_empty_container_operations() {
    check_empty_container_operations<transactional_trivial_set_t>();
    check_empty_container_operations<transactional_tracking_set_t>();
    check_empty_container_operations<transactional_composite_set_t>();
    check_empty_container_operations<transactional_heavy_set_t>();
    check_empty_container_operations<transactional_trivial_map_t>();
    check_empty_container_operations<transactional_tracking_map_t>();
    check_empty_container_operations<transactional_composite_map_t>();
    check_empty_container_operations<transactional_heavy_map_t>();
}

/** Tests operations on single-element container */
static void test_single_element_operations() {
    check_single_element_operations<transactional_trivial_set_t>();
    check_single_element_operations<transactional_tracking_set_t>();
    check_single_element_operations<transactional_composite_set_t>();
    check_single_element_operations<transactional_heavy_set_t>();
    check_single_element_operations<transactional_trivial_map_t>();
    check_single_element_operations<transactional_tracking_map_t>();
    check_single_element_operations<transactional_composite_map_t>();
    check_single_element_operations<transactional_heavy_map_t>();
}

/** Tests ascending, descending, and random insertion patterns across every container here. */
static void test_insertion_patterns() {
    check_basic_insertion_patterns<transactional_trivial_set_t>();
    check_basic_insertion_patterns<transactional_tracking_set_t>();
    check_basic_insertion_patterns<transactional_composite_set_t>();
    check_basic_insertion_patterns<transactional_heavy_set_t>();
    check_basic_insertion_patterns<transactional_trivial_map_t>();
    check_basic_insertion_patterns<transactional_tracking_map_t>();
    check_basic_insertion_patterns<transactional_composite_map_t>();
    check_basic_insertion_patterns<transactional_heavy_map_t>();
}

/** Tests bulk insertion from iterators across every container here. */
static void test_bulk_insertion_iterators() {
    check_bulk_insertion_from_iterators<transactional_trivial_set_t>();
    check_bulk_insertion_from_iterators<transactional_tracking_set_t>();
    check_bulk_insertion_from_iterators<transactional_composite_set_t>();
    check_bulk_insertion_from_iterators<transactional_heavy_set_t>();
    check_bulk_insertion_from_iterators<transactional_trivial_map_t>();
    check_bulk_insertion_from_iterators<transactional_tracking_map_t>();
    check_bulk_insertion_from_iterators<transactional_composite_map_t>();
    check_bulk_insertion_from_iterators<transactional_heavy_map_t>();
}

/** Tests bulk upsert correctly overwrites duplicate keys */
static void test_bulk_upsert_with_duplicate_pairs() {
    check_bulk_upsert_with_duplicates<transactional_trivial_map_t>();
    check_bulk_upsert_with_duplicates<transactional_tracking_map_t>();
}

/** Tests range queries on committed HEAD state */
static void test_range_query_head_state() {
    check_range_query_head_state<transactional_trivial_set_t>();
    check_range_query_head_state<transactional_tracking_set_t>();
    check_range_query_head_state<transactional_composite_set_t>();
    check_range_query_head_state<transactional_trivial_map_t>();
    check_range_query_head_state<transactional_tracking_map_t>();
    check_range_query_head_state<transactional_composite_map_t>();
}

/** Tests erase_range on committed HEAD state */
static void test_erase_range_head_state() {
    check_erase_range_head_state<transactional_trivial_set_t>();
    check_erase_range_head_state<transactional_tracking_set_t>();
    check_erase_range_head_state<transactional_composite_set_t>();
    check_erase_range_head_state<transactional_heavy_set_t>();
    check_erase_range_head_state<transactional_trivial_map_t>();
    check_erase_range_head_state<transactional_tracking_map_t>();
    check_erase_range_head_state<transactional_composite_map_t>();
    check_erase_range_head_state<transactional_heavy_map_t>();
}

/** Tests heterogeneous lookup for composite and heavy key types */
static void test_heterogeneous_lookups() {
    check_heterogeneous_composite_find<transactional_composite_set_t>();
    check_heterogeneous_heavy_string_view_find<transactional_heavy_set_t>();
    check_heterogeneous_composite_find<transactional_composite_map_t>();
    check_heterogeneous_heavy_string_view_find<transactional_heavy_map_t>();
}

#pragma endregion Basic Operations Tests

#pragma region Consistency and Transaction Tests for Sets

static void test_empty_transaction_commit() {
    check_empty_transaction_commit<transactional_trivial_set_t>();
    check_empty_transaction_commit<transactional_tracking_set_t>();
    check_empty_transaction_commit<transactional_composite_set_t>();
    check_empty_transaction_commit<transactional_heavy_set_t>();
    check_empty_transaction_commit<transactional_trivial_map_t>();
    check_empty_transaction_commit<transactional_tracking_map_t>();
    check_empty_transaction_commit<transactional_composite_map_t>();
    check_empty_transaction_commit<transactional_heavy_map_t>();
}

static void test_no_dirty_reads_multi_key() {
    check_no_dirty_reads_multi_key<transactional_trivial_map_t>();
    check_no_dirty_reads_multi_key<transactional_tracking_map_t>();
    check_no_dirty_reads_multi_key<transactional_composite_map_t>();
    check_no_dirty_reads_multi_key<transactional_heavy_map_t>();
}

static void test_new_transaction_sees_nothing_staged() {
    check_new_transaction_sees_nothing_staged<transactional_trivial_set_t>();
    check_new_transaction_sees_nothing_staged<transactional_tracking_set_t>();
    check_new_transaction_sees_nothing_staged<transactional_composite_set_t>();
    check_new_transaction_sees_nothing_staged<transactional_heavy_set_t>();
    check_new_transaction_sees_nothing_staged<transactional_trivial_map_t>();
    check_new_transaction_sees_nothing_staged<transactional_tracking_map_t>();
    check_new_transaction_sees_nothing_staged<transactional_composite_map_t>();
    check_new_transaction_sees_nothing_staged<transactional_heavy_map_t>();
}

static void test_committed_immediately_visible() {
    check_committed_immediately_visible<transactional_trivial_map_t>();
    check_committed_immediately_visible<transactional_tracking_map_t>();
    check_committed_immediately_visible<transactional_composite_map_t>();
    check_committed_immediately_visible<transactional_heavy_map_t>();
}

static void test_multi_key_atomicity_10_keys() {
    check_multi_key_atomicity_10_keys<transactional_trivial_map_t>();
    check_multi_key_atomicity_10_keys<transactional_tracking_map_t>();
    check_multi_key_atomicity_10_keys<transactional_composite_map_t>();
    check_multi_key_atomicity_10_keys<transactional_heavy_map_t>();
}

static void test_rollback_makes_all_invisible() {
    check_rollback_makes_all_invisible<transactional_trivial_map_t>();
    check_rollback_makes_all_invisible<transactional_tracking_map_t>();
    check_rollback_makes_all_invisible<transactional_composite_map_t>();
    check_rollback_makes_all_invisible<transactional_heavy_map_t>();
}

static void test_range_query_sees_atomic_boundaries() {
    check_range_query_sees_atomic_boundaries<transactional_trivial_set_t>();
    check_range_query_sees_atomic_boundaries<transactional_tracking_set_t>();
    check_range_query_sees_atomic_boundaries<transactional_composite_set_t>();
    check_range_query_sees_atomic_boundaries<transactional_heavy_set_t>();
}

static void test_fractured_read_prevention() {
    check_fractured_read_prevention<transactional_trivial_map_t>();
    check_fractured_read_prevention<transactional_tracking_map_t>();
    check_fractured_read_prevention<transactional_composite_map_t>();
    check_fractured_read_prevention<transactional_heavy_map_t>();
}

static void test_sequential_updates_never_regress() {
    check_sequential_updates_never_regress<transactional_trivial_map_t>();
    check_sequential_updates_never_regress<transactional_tracking_map_t>();
}

/** Ordering of mapped values is only meaningful where they are numbers, so @c int maps only. */
static void test_transaction_commits_maintain_order() {
    check_transaction_commits_maintain_order<transactional_trivial_map_t>();
    check_transaction_commits_maintain_order<transactional_tracking_map_t>();
}

static void test_concurrent_transactions_on_same_key() {
    check_concurrent_transactions_on_same_key<transactional_trivial_map_t>();
    check_concurrent_transactions_on_same_key<transactional_tracking_map_t>();
    check_concurrent_transactions_on_same_key<transactional_composite_map_t>();
    check_concurrent_transactions_on_same_key<transactional_heavy_map_t>();
}

static void test_multi_key_conflict_any_key_fails() {
    check_multi_key_conflict_any_key_fails<transactional_trivial_map_t>();
    check_multi_key_conflict_any_key_fails<transactional_tracking_map_t>();
    check_multi_key_conflict_any_key_fails<transactional_composite_map_t>();
    check_multi_key_conflict_any_key_fails<transactional_heavy_map_t>();
}

static void test_watch_detects_external_direct_modification() {
    check_watch_detects_external_direct_modification<transactional_trivial_map_t>();
    check_watch_detects_external_direct_modification<transactional_tracking_map_t>();
    check_watch_detects_external_direct_modification<transactional_composite_map_t>();
    check_watch_detects_external_direct_modification<transactional_heavy_map_t>();
}

static void test_abandoned_transaction_leaves_no_trace() {
    check_abandoned_transaction_leaves_no_trace<transactional_trivial_map_t>();
    check_abandoned_transaction_leaves_no_trace<transactional_tracking_map_t>();
    check_abandoned_transaction_leaves_no_trace<transactional_composite_map_t>();
    check_abandoned_transaction_leaves_no_trace<transactional_heavy_map_t>();
}

static void test_moved_transaction_unwinds_once() {
    check_moved_transaction_unwinds_once<transactional_trivial_map_t>();
    check_moved_transaction_unwinds_once<transactional_tracking_map_t>();
    check_moved_transaction_unwinds_once<transactional_composite_map_t>();
    check_moved_transaction_unwinds_once<transactional_heavy_map_t>();
}

static void test_watch_on_erased_key_can_commit() {
    check_watch_on_erased_key_can_commit<transactional_trivial_map_t>();
    check_watch_on_erased_key_can_commit<transactional_tracking_map_t>();
    check_watch_on_erased_key_can_commit<transactional_composite_map_t>();
    check_watch_on_erased_key_can_commit<transactional_heavy_map_t>();
}

static void test_watch_catches_an_insert_and_erase_under_it() {
    check_watch_catches_an_insert_and_erase_under_it<transactional_trivial_map_t>();
    check_watch_catches_an_insert_and_erase_under_it<transactional_tracking_map_t>();
    check_watch_catches_an_insert_and_erase_under_it<transactional_composite_map_t>();
    check_watch_catches_an_insert_and_erase_under_it<transactional_heavy_map_t>();
}

static void test_absent_watch_survives_rollback() {
    check_absent_watch_survives_rollback<transactional_trivial_map_t>();
    check_absent_watch_survives_rollback<transactional_tracking_map_t>();
    check_absent_watch_survives_rollback<transactional_composite_map_t>();
    check_absent_watch_survives_rollback<transactional_heavy_map_t>();
}

static void test_group_commits_participants_together() {
    check_group_commits_participants_together<transactional_trivial_map_t>();
    check_group_commits_participants_together<transactional_tracking_map_t>();
    check_group_commits_participants_together<transactional_composite_map_t>();
    check_group_commits_participants_together<transactional_heavy_map_t>();
}

static void test_group_unwinds_every_participant_on_conflict() {
    check_group_unwinds_every_participant_on_conflict<transactional_trivial_map_t>();
    check_group_unwinds_every_participant_on_conflict<transactional_tracking_map_t>();
    check_group_unwinds_every_participant_on_conflict<transactional_composite_map_t>();
    check_group_unwinds_every_participant_on_conflict<transactional_heavy_map_t>();
}

static void test_watch_detects_staged_invisible_writes() {
    check_watch_detects_staged_invisible_writes<transactional_trivial_map_t>();
    check_watch_detects_staged_invisible_writes<transactional_tracking_map_t>();
    check_watch_detects_staged_invisible_writes<transactional_composite_map_t>();
    check_watch_detects_staged_invisible_writes<transactional_heavy_map_t>();
}

static void test_watch_detects_staged_writes_of_older_generation() {
    check_watch_detects_staged_writes_of_older_generation<transactional_trivial_map_t>();
    check_watch_detects_staged_writes_of_older_generation<transactional_tracking_map_t>();
    check_watch_detects_staged_writes_of_older_generation<transactional_composite_map_t>();
    check_watch_detects_staged_writes_of_older_generation<transactional_heavy_map_t>();
}

static void test_disjoint_keys_both_succeed() {
    check_disjoint_keys_both_succeed<transactional_trivial_map_t>();
    check_disjoint_keys_both_succeed<transactional_tracking_map_t>();
    check_disjoint_keys_both_succeed<transactional_composite_map_t>();
    check_disjoint_keys_both_succeed<transactional_heavy_map_t>();
}

static void test_lost_update_matches_isolation() {
    check_lost_update_matches_isolation<transactional_trivial_map_t>();
    check_lost_update_matches_isolation<transactional_tracking_map_t>();
    check_lost_update_matches_isolation<transactional_composite_map_t>();
    check_lost_update_matches_isolation<transactional_heavy_map_t>();
}

static void test_repeated_read_matches_isolation() {
    check_repeated_read_matches_isolation<transactional_trivial_map_t>();
    check_repeated_read_matches_isolation<transactional_tracking_map_t>();
    check_repeated_read_matches_isolation<transactional_composite_map_t>();
    check_repeated_read_matches_isolation<transactional_heavy_map_t>();
}

static void test_repeated_range_matches_isolation() {
    check_repeated_range_matches_isolation<transactional_trivial_set_t>();
    check_repeated_range_matches_isolation<transactional_tracking_set_t>();
    check_repeated_range_matches_isolation<transactional_composite_set_t>();
    check_repeated_range_matches_isolation<transactional_heavy_set_t>();
    check_repeated_range_matches_isolation<transactional_trivial_map_t>();
    check_repeated_range_matches_isolation<transactional_tracking_map_t>();
    check_repeated_range_matches_isolation<transactional_composite_map_t>();
    check_repeated_range_matches_isolation<transactional_heavy_map_t>();
}

static void test_delete_visibility() {
    check_delete_visibility<transactional_trivial_set_t>();
    check_delete_visibility<transactional_tracking_set_t>();
    check_delete_visibility<transactional_composite_set_t>();
    check_delete_visibility<transactional_heavy_set_t>();
}

static void test_reset_clears_transaction_state() {
    check_reset_clears_transaction_state<transactional_trivial_map_t>();
    check_reset_clears_transaction_state<transactional_tracking_map_t>();
    check_reset_clears_transaction_state<transactional_composite_map_t>();
    check_reset_clears_transaction_state<transactional_heavy_map_t>();
}

static void test_stateful_comparator_is_consulted() {
    check_stateful_comparator_is_consulted<transactional_tracking_set_t>();
}

#pragma endregion Consistency and Transaction Tests for Sets

#pragma region Std Store Defects

static void test_committed_tombstones_are_reclaimable() {
    check_committed_tombstones_are_reclaimable<transactional_trivial_map_t>();
    check_committed_tombstones_are_reclaimable<transactional_composite_map_t>();
    check_committed_tombstones_are_reclaimable<transactional_heavy_map_t>();
}

static void test_vacuum_respects_its_window() {
    check_vacuum_respects_its_window<transactional_trivial_map_t>();
    check_vacuum_respects_its_window<transactional_composite_map_t>();
    check_vacuum_respects_its_window<transactional_heavy_map_t>();
}

static void test_erase_reports_and_reclaims_tombstone() {
    check_erase_reports_and_reclaims_tombstone<transactional_trivial_map_t>();
    check_erase_reports_and_reclaims_tombstone<transactional_composite_map_t>();
    check_erase_reports_and_reclaims_tombstone<transactional_heavy_map_t>();
}

static void test_erase_range_skips_tombstones() {
    check_erase_range_skips_tombstones<transactional_trivial_map_t>();
    check_erase_range_skips_tombstones<transactional_composite_map_t>();
    check_erase_range_skips_tombstones<transactional_heavy_map_t>();
}

static void test_vacuum_leaves_staged_entries() {
    check_vacuum_leaves_staged_entries<transactional_trivial_map_t>();
    check_vacuum_leaves_staged_entries<transactional_composite_map_t>();
    check_vacuum_leaves_staged_entries<transactional_heavy_map_t>();
}

static void test_staged_versions_survive_a_clear() {
    check_staged_versions_survive_a_clear<transactional_trivial_map_t>();
    check_staged_versions_survive_a_clear<transactional_composite_map_t>();
    check_staged_versions_survive_a_clear<transactional_heavy_map_t>();
}

static void test_commit_reports_success_when_published() {
    check_commit_reports_success_when_published<transactional_trivial_map_t>();
    check_commit_reports_success_when_published<transactional_composite_map_t>();
    check_commit_reports_success_when_published<transactional_heavy_map_t>();
}

static void test_staged_entries_keep_one_generation() {
    check_staged_entries_keep_one_generation<transactional_trivial_map_t>();
    check_staged_entries_keep_one_generation<transactional_composite_map_t>();
    check_staged_entries_keep_one_generation<transactional_heavy_map_t>();
}

#pragma endregion Std Store Defects

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

static void test_clear_keeps_generations_running() {
    check_clear_keeps_generations_running<transactional_trivial_map_t>();
    check_clear_keeps_generations_running<transactional_composite_map_t>();
    check_clear_keeps_generations_running<transactional_heavy_map_t>();
}

static void test_insert_refuses_with_key_already_exists() {
    check_insert_refuses_with_key_already_exists<transactional_trivial_map_t>();
    check_insert_refuses_with_key_already_exists<transactional_composite_map_t>();
    check_insert_refuses_with_key_already_exists<transactional_heavy_map_t>();
}

static void test_insert_reports_the_stored_element() {
    check_insert_reports_the_stored_element<transactional_trivial_map_t>();
    check_insert_reports_the_stored_element<transactional_composite_map_t>();
    check_insert_reports_the_stored_element<transactional_heavy_map_t>();
}

static void test_insert_after_local_erase_succeeds() {
    check_insert_after_local_erase_succeeds<transactional_trivial_map_t>();
    check_insert_after_local_erase_succeeds<transactional_composite_map_t>();
    check_insert_after_local_erase_succeeds<transactional_heavy_map_t>();
}

static void test_stage_refuses_when_already_staged() {
    check_stage_refuses_when_already_staged<transactional_trivial_map_t>();
    check_stage_refuses_when_already_staged<transactional_composite_map_t>();
    check_stage_refuses_when_already_staged<transactional_heavy_map_t>();
}

static void test_rollback_recovers_what_it_staged() {
    check_rollback_recovers_what_it_staged<transactional_trivial_map_t>();
    check_rollback_recovers_what_it_staged<transactional_composite_map_t>();
    check_rollback_recovers_what_it_staged<transactional_heavy_map_t>();
}

static void test_transaction_bounds_skip_locally_erased() {
    check_transaction_bounds_skip_locally_erased<transactional_trivial_map_t>();
    check_transaction_bounds_skip_locally_erased<transactional_composite_map_t>();
    check_transaction_bounds_skip_locally_erased<transactional_heavy_map_t>();
}

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

int main(int argc, char **argv) {
    start_test_run(argc, argv);

    run_test_case("test_empty_container_operations", test_empty_container_operations);
    run_test_case("test_single_element_operations", test_single_element_operations);
    run_test_case("test_insertion_patterns", test_insertion_patterns);
    run_test_case("test_bulk_insertion_iterators", test_bulk_insertion_iterators);
    run_test_case("test_bulk_upsert_with_duplicate_pairs", test_bulk_upsert_with_duplicate_pairs);
    run_test_case("test_range_query_head_state", test_range_query_head_state);
    run_test_case("test_erase_range_head_state", test_erase_range_head_state);
    run_test_case("test_heterogeneous_lookups", test_heterogeneous_lookups);

    run_test_case("test_empty_transaction_commit", test_empty_transaction_commit);
    run_test_case("test_no_dirty_reads_multi_key", test_no_dirty_reads_multi_key);
    run_test_case("test_new_transaction_sees_nothing_staged", test_new_transaction_sees_nothing_staged);
    run_test_case("test_committed_immediately_visible", test_committed_immediately_visible);
    run_test_case("test_multi_key_atomicity_10_keys", test_multi_key_atomicity_10_keys);
    run_test_case("test_rollback_makes_all_invisible", test_rollback_makes_all_invisible);
    run_test_case("test_range_query_sees_atomic_boundaries", test_range_query_sees_atomic_boundaries);
    run_test_case("test_fractured_read_prevention", test_fractured_read_prevention);
    run_test_case("test_sequential_updates_never_regress", test_sequential_updates_never_regress);
    run_test_case("test_transaction_commits_maintain_order", test_transaction_commits_maintain_order);
    run_test_case("test_concurrent_transactions_on_same_key", test_concurrent_transactions_on_same_key);
    run_test_case("test_multi_key_conflict_any_key_fails", test_multi_key_conflict_any_key_fails);
    run_test_case("test_watch_detects_external_direct_modification", test_watch_detects_external_direct_modification);
    run_test_case("test_watch_detects_staged_invisible_writes", test_watch_detects_staged_invisible_writes);
    run_test_case("test_watch_detects_staged_writes_of_older_generation",
                  test_watch_detects_staged_writes_of_older_generation);
    run_test_case("test_group_commits_participants_together", test_group_commits_participants_together);
    run_test_case("test_group_unwinds_every_participant_on_conflict", test_group_unwinds_every_participant_on_conflict);
    run_test_case("test_abandoned_transaction_leaves_no_trace", test_abandoned_transaction_leaves_no_trace);
    run_test_case("test_moved_transaction_unwinds_once", test_moved_transaction_unwinds_once);
    run_test_case("test_watch_on_erased_key_can_commit", test_watch_on_erased_key_can_commit);
    run_test_case("test_watch_catches_an_insert_and_erase_under_it", test_watch_catches_an_insert_and_erase_under_it);
    run_test_case("test_absent_watch_survives_rollback", test_absent_watch_survives_rollback);
    run_test_case("test_disjoint_keys_both_succeed", test_disjoint_keys_both_succeed);
    run_test_case("test_lost_update_matches_isolation", test_lost_update_matches_isolation);
    run_test_case("test_repeated_read_matches_isolation", test_repeated_read_matches_isolation);
    run_test_case("test_repeated_range_matches_isolation", test_repeated_range_matches_isolation);
    run_test_case("test_delete_visibility", test_delete_visibility);
    run_test_case("test_reset_clears_transaction_state", test_reset_clears_transaction_state);

    run_test_case("test_stateful_comparator_is_consulted", test_stateful_comparator_is_consulted);

    run_test_case("test_committed_tombstones_are_reclaimable", test_committed_tombstones_are_reclaimable);
    run_test_case("test_vacuum_respects_its_window", test_vacuum_respects_its_window);
    run_test_case("test_erase_reports_and_reclaims_tombstone", test_erase_reports_and_reclaims_tombstone);
    run_test_case("test_erase_range_skips_tombstones", test_erase_range_skips_tombstones);
    run_test_case("test_vacuum_leaves_staged_entries", test_vacuum_leaves_staged_entries);
    run_test_case("test_staged_versions_survive_a_clear", test_staged_versions_survive_a_clear);
    run_test_case("test_commit_reports_success_when_published", test_commit_reports_success_when_published);
    run_test_case("test_staged_entries_keep_one_generation", test_staged_entries_keep_one_generation);

    run_test_case("test_rolled_back_stage_does_not_abort_a_peer", test_rolled_back_stage_does_not_abort_a_peer);
    run_test_case("test_lost_update_is_refused", test_lost_update_is_refused);
    run_test_case("test_find_and_watch_records_absence", test_find_and_watch_records_absence);

    run_test_case("test_find_does_not_watch", test_find_does_not_watch);

    run_test_case("test_clear_keeps_generations_running", test_clear_keeps_generations_running);
    run_test_case("test_insert_refuses_with_key_already_exists", test_insert_refuses_with_key_already_exists);
    run_test_case("test_insert_reports_the_stored_element", test_insert_reports_the_stored_element);
    run_test_case("test_insert_after_local_erase_succeeds", test_insert_after_local_erase_succeeds);
    run_test_case("test_validate_refuses_before_publishing<transactional_trivial_map_t>",
                  test_validate_refuses_before_publishing<transactional_trivial_map_t>);
    run_test_case("test_publish_cannot_refuse<transactional_trivial_map_t>",
                  test_publish_cannot_refuse<transactional_trivial_map_t>);
    run_test_case("test_clear_refuses_while_staged<transactional_trivial_map_t>",
                  test_clear_refuses_while_staged<transactional_trivial_map_t>);
    run_test_case("test_stage_refuses_when_already_staged", test_stage_refuses_when_already_staged);
    run_test_case("test_rollback_recovers_what_it_staged", test_rollback_recovers_what_it_staged);
    run_test_case("test_transaction_bounds_skip_locally_erased", test_transaction_bounds_skip_locally_erased);

    return finish_test_run();
}
