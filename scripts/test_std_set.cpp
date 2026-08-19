/**
 *  @brief Test instantiations for the @c std::set-backed transactional store. The baseline reference design, exercised
 *      by the same suites as the tree containers.
 *  @author Ash Vardanian
 *  @file scripts/test_std_set.cpp
 *  @date August 16, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define ST_STRICT_CALLBACK_CHECKS_ 1

#include <smashtable/reference_store.hpp>

#include "test.hpp"
#include "test_basic.hpp"
#include "test_commit_stamp.hpp"
#include "test_consistency.hpp"
#include "test_reference_store_defects.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

#pragma region Type Aliases

/** Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack */
using transactional_trivial_set_t =
    reference_store<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/** Heterogeneous lookup: ✓ | Copy: Trivial | Memory: Tracked */
using transactional_tracking_set_t = reference_store<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/** Heterogeneous lookup: ✓ (uint64_t) | Copy: Trivial | Memory: Stack */
using transactional_composite_set_t =
    reference_store<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/** Heterogeneous lookup: ✓ (string_view) | Copy: .copy() → expected<T> | Memory: Heap */
using transactional_heavy_set_t = reference_store<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/** Value: int | Copy: Trivial (key & value) | Memory: Stack */
using transactional_trivial_map_t =
    reference_store<mapping<trivial_key_t, int>, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/** Value: int | Copy: Trivial (key & value) | Memory: Tracked */
using transactional_tracking_map_t =
    reference_store<mapping<trivial_key_t, int>, stateful_comparator_t, stateful_allocator_t>;

/** Value: guarded_payload_t | Copy: Key trivial, value .copy() | Memory: Heap (value) */
using transactional_composite_map_t =
    reference_store<mapping<composite_key_t, guarded_payload_t>, composite_key_compare_t,
                    std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/** Value: guarded_payload_t | Copy: .copy() on key & value | Memory: Heap (both) */
using transactional_heavy_map_t = reference_store<mapping<heavy_key_t, guarded_payload_t>, std::less<void>,
                                                  std::allocator<mapping<heavy_key_t, guarded_payload_t>>>;

/** @brief Tests operations on empty container don't crash */
static void basic_ops_empty_container_operations() {
    test_empty_container_operations<transactional_trivial_set_t>();
    test_empty_container_operations<transactional_tracking_set_t>();
    test_empty_container_operations<transactional_composite_set_t>();
    test_empty_container_operations<transactional_heavy_set_t>();
    test_empty_container_operations<transactional_trivial_map_t>();
    test_empty_container_operations<transactional_tracking_map_t>();
    test_empty_container_operations<transactional_composite_map_t>();
    test_empty_container_operations<transactional_heavy_map_t>();
}

/** @brief Tests operations on single-element container */
static void basic_ops_single_element_operations() {
    test_single_element_operations<transactional_trivial_set_t>();
    test_single_element_operations<transactional_tracking_set_t>();
    test_single_element_operations<transactional_composite_set_t>();
    test_single_element_operations<transactional_heavy_set_t>();
    test_single_element_operations<transactional_trivial_map_t>();
    test_single_element_operations<transactional_tracking_map_t>();
    test_single_element_operations<transactional_composite_map_t>();
    test_single_element_operations<transactional_heavy_map_t>();
}

/** @brief Tests insertion patterns (ascending, descending, random) for all AVL containers */
static void basic_ops_insertion_patterns() {
    test_basic_insertion_patterns<transactional_trivial_set_t>();
    test_basic_insertion_patterns<transactional_tracking_set_t>();
    test_basic_insertion_patterns<transactional_composite_set_t>();
    test_basic_insertion_patterns<transactional_heavy_set_t>();
    test_basic_insertion_patterns<transactional_trivial_map_t>();
    test_basic_insertion_patterns<transactional_tracking_map_t>();
    test_basic_insertion_patterns<transactional_composite_map_t>();
    test_basic_insertion_patterns<transactional_heavy_map_t>();
}

/** @brief Tests bulk insertion from iterators for all AVL containers */
static void basic_ops_bulk_insertion_iterators() {
    test_bulk_insertion_from_iterators<transactional_trivial_set_t>();
    test_bulk_insertion_from_iterators<transactional_tracking_set_t>();
    test_bulk_insertion_from_iterators<transactional_composite_set_t>();
    test_bulk_insertion_from_iterators<transactional_heavy_set_t>();
    test_bulk_insertion_from_iterators<transactional_trivial_map_t>();
    test_bulk_insertion_from_iterators<transactional_tracking_map_t>();
    test_bulk_insertion_from_iterators<transactional_composite_map_t>();
    test_bulk_insertion_from_iterators<transactional_heavy_map_t>();
}

/** @brief Tests bulk upsert correctly overwrites duplicate keys */
static void basic_ops_bulk_upsert_with_duplicate_pairs() {
    test_bulk_upsert_with_duplicates<transactional_trivial_map_t>();
    test_bulk_upsert_with_duplicates<transactional_tracking_map_t>();
}

/** @brief Tests range queries on committed HEAD state */
static void basic_ops_range_query_head_state() {
    test_range_query_head_state<transactional_trivial_set_t>();
    test_range_query_head_state<transactional_tracking_set_t>();
    test_range_query_head_state<transactional_composite_set_t>();
    test_range_query_head_state<transactional_trivial_map_t>();
    test_range_query_head_state<transactional_tracking_map_t>();
    test_range_query_head_state<transactional_composite_map_t>();
}

/** @brief Tests erase_range on committed HEAD state */
static void basic_ops_erase_range_head_state() {
    test_erase_range_head_state<transactional_trivial_set_t>();
    test_erase_range_head_state<transactional_tracking_set_t>();
    test_erase_range_head_state<transactional_composite_set_t>();
    test_erase_range_head_state<transactional_heavy_set_t>();
    test_erase_range_head_state<transactional_trivial_map_t>();
    test_erase_range_head_state<transactional_tracking_map_t>();
    test_erase_range_head_state<transactional_composite_map_t>();
    test_erase_range_head_state<transactional_heavy_map_t>();
}

/** @brief Tests heterogeneous lookup for composite and heavy key types */
static void basic_ops_heterogeneous_lookups() {
    test_heterogeneous_composite_find<transactional_composite_set_t>();
    test_heterogeneous_heavy_string_view_find<transactional_heavy_set_t>();
    test_heterogeneous_composite_find<transactional_composite_map_t>();
    test_heterogeneous_heavy_string_view_find<transactional_heavy_map_t>();
}

static void transactional_consistency_empty_transaction_commit() {
    test_empty_transaction_commit<transactional_trivial_set_t>();
    test_empty_transaction_commit<transactional_tracking_set_t>();
    test_empty_transaction_commit<transactional_composite_set_t>();
    test_empty_transaction_commit<transactional_heavy_set_t>();
    test_empty_transaction_commit<transactional_trivial_map_t>();
    test_empty_transaction_commit<transactional_tracking_map_t>();
    test_empty_transaction_commit<transactional_composite_map_t>();
    test_empty_transaction_commit<transactional_heavy_map_t>();
}

static void transactional_consistency_no_dirty_reads_multi_key() {
    test_no_dirty_reads_multi_key<transactional_trivial_map_t>();
    test_no_dirty_reads_multi_key<transactional_tracking_map_t>();
    test_no_dirty_reads_multi_key<transactional_composite_map_t>();
    test_no_dirty_reads_multi_key<transactional_heavy_map_t>();
}

static void transactional_consistency_new_transaction_sees_nothing_staged() {
    test_new_transaction_sees_nothing_staged<transactional_trivial_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_tracking_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_trivial_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_tracking_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_map_t>();
}

static void transactional_consistency_committed_immediately_visible() {
    test_committed_immediately_visible<transactional_trivial_map_t>();
    test_committed_immediately_visible<transactional_tracking_map_t>();
    test_committed_immediately_visible<transactional_composite_map_t>();
    test_committed_immediately_visible<transactional_heavy_map_t>();
}

static void transactional_consistency_multi_key_atomicity_10_keys() {
    test_multi_key_atomicity_10_keys<transactional_trivial_map_t>();
    test_multi_key_atomicity_10_keys<transactional_tracking_map_t>();
    test_multi_key_atomicity_10_keys<transactional_composite_map_t>();
    test_multi_key_atomicity_10_keys<transactional_heavy_map_t>();
}

static void transactional_consistency_rollback_makes_all_invisible() {
    test_rollback_makes_all_invisible<transactional_trivial_map_t>();
    test_rollback_makes_all_invisible<transactional_tracking_map_t>();
    test_rollback_makes_all_invisible<transactional_composite_map_t>();
    test_rollback_makes_all_invisible<transactional_heavy_map_t>();
}

static void transactional_consistency_range_query_sees_atomic_boundaries() {
    test_range_query_sees_atomic_boundaries<transactional_trivial_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_tracking_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_composite_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_heavy_set_t>();
}

static void transactional_consistency_fractured_read_prevention() {
    test_fractured_read_prevention<transactional_trivial_map_t>();
    test_fractured_read_prevention<transactional_tracking_map_t>();
    test_fractured_read_prevention<transactional_composite_map_t>();
    test_fractured_read_prevention<transactional_heavy_map_t>();
}

static void transactional_consistency_sequential_updates_never_regress() {
    test_sequential_updates_never_regress<transactional_trivial_map_t>();
    test_sequential_updates_never_regress<transactional_tracking_map_t>();
}

/** @brief Ordering of mapped values is only meaningful where they are numbers, so @c int maps only. */
static void transactional_consistency_transaction_commits_maintain_order() {
    test_transaction_commits_maintain_order<transactional_trivial_map_t>();
    test_transaction_commits_maintain_order<transactional_tracking_map_t>();
}

static void transactional_consistency_concurrent_transactions_on_same_key() {
    test_concurrent_transactions_on_same_key<transactional_trivial_map_t>();
    test_concurrent_transactions_on_same_key<transactional_tracking_map_t>();
    test_concurrent_transactions_on_same_key<transactional_composite_map_t>();
    test_concurrent_transactions_on_same_key<transactional_heavy_map_t>();
}

static void transactional_consistency_multi_key_conflict_any_key_fails() {
    test_multi_key_conflict_any_key_fails<transactional_trivial_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_tracking_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_composite_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_detects_external_direct_modification() {
    test_watch_detects_external_direct_modification<transactional_trivial_map_t>();
    test_watch_detects_external_direct_modification<transactional_tracking_map_t>();
    test_watch_detects_external_direct_modification<transactional_composite_map_t>();
    test_watch_detects_external_direct_modification<transactional_heavy_map_t>();
}

static void transactional_consistency_abandoned_transaction_leaves_no_trace() {
    test_abandoned_transaction_leaves_no_trace<transactional_trivial_map_t>();
    test_abandoned_transaction_leaves_no_trace<transactional_tracking_map_t>();
    test_abandoned_transaction_leaves_no_trace<transactional_composite_map_t>();
    test_abandoned_transaction_leaves_no_trace<transactional_heavy_map_t>();
}

static void transactional_consistency_moved_transaction_unwinds_once() {
    test_moved_transaction_unwinds_once<transactional_trivial_map_t>();
    test_moved_transaction_unwinds_once<transactional_tracking_map_t>();
    test_moved_transaction_unwinds_once<transactional_composite_map_t>();
    test_moved_transaction_unwinds_once<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_on_erased_key_can_commit() {
    test_watch_on_erased_key_can_commit<transactional_trivial_map_t>();
    test_watch_on_erased_key_can_commit<transactional_tracking_map_t>();
    test_watch_on_erased_key_can_commit<transactional_composite_map_t>();
    test_watch_on_erased_key_can_commit<transactional_heavy_map_t>();
}

static void transactional_consistency_absent_watch_survives_rollback() {
    test_absent_watch_survives_rollback<transactional_trivial_map_t>();
    test_absent_watch_survives_rollback<transactional_tracking_map_t>();
    test_absent_watch_survives_rollback<transactional_composite_map_t>();
    test_absent_watch_survives_rollback<transactional_heavy_map_t>();
}

static void transactional_consistency_group_commits_participants_together() {
    test_group_commits_participants_together<transactional_trivial_map_t>();
    test_group_commits_participants_together<transactional_tracking_map_t>();
    test_group_commits_participants_together<transactional_composite_map_t>();
    test_group_commits_participants_together<transactional_heavy_map_t>();
}

static void transactional_consistency_group_unwinds_every_participant_on_conflict() {
    test_group_unwinds_every_participant_on_conflict<transactional_trivial_map_t>();
    test_group_unwinds_every_participant_on_conflict<transactional_tracking_map_t>();
    test_group_unwinds_every_participant_on_conflict<transactional_composite_map_t>();
    test_group_unwinds_every_participant_on_conflict<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_detects_staged_invisible_writes() {
    test_watch_detects_staged_invisible_writes<transactional_trivial_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_tracking_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_composite_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_detects_staged_writes_of_older_generation() {
    test_watch_detects_staged_writes_of_older_generation<transactional_trivial_map_t>();
    test_watch_detects_staged_writes_of_older_generation<transactional_tracking_map_t>();
    test_watch_detects_staged_writes_of_older_generation<transactional_composite_map_t>();
    test_watch_detects_staged_writes_of_older_generation<transactional_heavy_map_t>();
}

static void transactional_consistency_disjoint_keys_both_succeed() {
    test_disjoint_keys_both_succeed<transactional_trivial_map_t>();
    test_disjoint_keys_both_succeed<transactional_tracking_map_t>();
    test_disjoint_keys_both_succeed<transactional_composite_map_t>();
    test_disjoint_keys_both_succeed<transactional_heavy_map_t>();
}

static void transactional_consistency_lost_update_matches_isolation() {
    test_lost_update_matches_isolation<transactional_trivial_map_t>();
    test_lost_update_matches_isolation<transactional_tracking_map_t>();
    test_lost_update_matches_isolation<transactional_composite_map_t>();
    test_lost_update_matches_isolation<transactional_heavy_map_t>();
}

static void transactional_consistency_repeated_read_matches_isolation() {
    test_repeated_read_matches_isolation<transactional_trivial_map_t>();
    test_repeated_read_matches_isolation<transactional_tracking_map_t>();
    test_repeated_read_matches_isolation<transactional_composite_map_t>();
    test_repeated_read_matches_isolation<transactional_heavy_map_t>();
}

static void transactional_consistency_repeated_range_matches_isolation() {
    test_repeated_range_matches_isolation<transactional_trivial_set_t>();
    test_repeated_range_matches_isolation<transactional_tracking_set_t>();
    test_repeated_range_matches_isolation<transactional_composite_set_t>();
    test_repeated_range_matches_isolation<transactional_heavy_set_t>();
    test_repeated_range_matches_isolation<transactional_trivial_map_t>();
    test_repeated_range_matches_isolation<transactional_tracking_map_t>();
    test_repeated_range_matches_isolation<transactional_composite_map_t>();
    test_repeated_range_matches_isolation<transactional_heavy_map_t>();
}

static void transactional_consistency_delete_visibility() {
    test_delete_visibility<transactional_trivial_set_t>();
    test_delete_visibility<transactional_tracking_set_t>();
    test_delete_visibility<transactional_composite_set_t>();
    test_delete_visibility<transactional_heavy_set_t>();
}

static void transactional_consistency_reset_clears_transaction_state() {
    test_reset_clears_transaction_state<transactional_trivial_map_t>();
    test_reset_clears_transaction_state<transactional_tracking_map_t>();
    test_reset_clears_transaction_state<transactional_composite_map_t>();
    test_reset_clears_transaction_state<transactional_heavy_map_t>();
}

#pragma endregion Type Aliases

static void transactional_consistency_stateful_comparator_is_consulted() {
    test_stateful_comparator_is_consulted<transactional_tracking_set_t>();
}

#pragma region Std Store Defects

static void std_store_defects_committed_tombstones_are_reclaimable() {
    test_committed_tombstones_are_reclaimable<transactional_trivial_map_t>();
    test_committed_tombstones_are_reclaimable<transactional_composite_map_t>();
    test_committed_tombstones_are_reclaimable<transactional_heavy_map_t>();
}

static void std_store_defects_vacuum_respects_its_window() {
    test_vacuum_respects_its_window<transactional_trivial_map_t>();
    test_vacuum_respects_its_window<transactional_composite_map_t>();
    test_vacuum_respects_its_window<transactional_heavy_map_t>();
}

static void std_store_defects_erase_reports_and_reclaims_tombstone() {
    test_erase_reports_and_reclaims_tombstone<transactional_trivial_map_t>();
    test_erase_reports_and_reclaims_tombstone<transactional_composite_map_t>();
    test_erase_reports_and_reclaims_tombstone<transactional_heavy_map_t>();
}

static void std_store_defects_erase_range_skips_tombstones() {
    test_erase_range_skips_tombstones<transactional_trivial_map_t>();
    test_erase_range_skips_tombstones<transactional_composite_map_t>();
    test_erase_range_skips_tombstones<transactional_heavy_map_t>();
}

static void std_store_defects_vacuum_leaves_staged_entries() {
    test_vacuum_leaves_staged_entries<transactional_trivial_map_t>();
    test_vacuum_leaves_staged_entries<transactional_composite_map_t>();
    test_vacuum_leaves_staged_entries<transactional_heavy_map_t>();
}

static void std_store_defects_commit_reports_lost_versions() {
    test_commit_reports_lost_versions<transactional_trivial_map_t>();
    test_commit_reports_lost_versions<transactional_composite_map_t>();
    test_commit_reports_lost_versions<transactional_heavy_map_t>();
}

static void std_store_defects_commit_reports_success_when_published() {
    test_commit_reports_success_when_published<transactional_trivial_map_t>();
    test_commit_reports_success_when_published<transactional_composite_map_t>();
    test_commit_reports_success_when_published<transactional_heavy_map_t>();
}

static void std_store_defects_staged_entries_keep_one_generation() {
    test_staged_entries_keep_one_generation<transactional_trivial_map_t>();
    test_staged_entries_keep_one_generation<transactional_composite_map_t>();
    test_staged_entries_keep_one_generation<transactional_heavy_map_t>();
}

#pragma endregion Std Store Defects

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

static void std_store_defects_clear_keeps_generations_running() {
    test_clear_keeps_generations_running<transactional_trivial_map_t>();
    test_clear_keeps_generations_running<transactional_composite_map_t>();
    test_clear_keeps_generations_running<transactional_heavy_map_t>();
}

static void std_store_defects_insert_refuses_with_key_already_exists() {
    test_insert_refuses_with_key_already_exists<transactional_trivial_map_t>();
    test_insert_refuses_with_key_already_exists<transactional_composite_map_t>();
    test_insert_refuses_with_key_already_exists<transactional_heavy_map_t>();
}

static void std_store_defects_insert_reports_the_stored_element() {
    test_insert_reports_the_stored_element<transactional_trivial_map_t>();
    test_insert_reports_the_stored_element<transactional_composite_map_t>();
    test_insert_reports_the_stored_element<transactional_heavy_map_t>();
}

static void std_store_defects_insert_after_local_erase_succeeds() {
    test_insert_after_local_erase_succeeds<transactional_trivial_map_t>();
    test_insert_after_local_erase_succeeds<transactional_composite_map_t>();
    test_insert_after_local_erase_succeeds<transactional_heavy_map_t>();
}

static void std_store_defects_stage_refuses_when_already_staged() {
    test_stage_refuses_when_already_staged<transactional_trivial_map_t>();
    test_stage_refuses_when_already_staged<transactional_composite_map_t>();
    test_stage_refuses_when_already_staged<transactional_heavy_map_t>();
}

static void std_store_defects_rollback_reports_lost_versions() {
    test_rollback_reports_lost_versions<transactional_trivial_map_t>();
    test_rollback_reports_lost_versions<transactional_composite_map_t>();
    test_rollback_reports_lost_versions<transactional_heavy_map_t>();
}

static void std_store_defects_transaction_bounds_skip_locally_erased() {
    test_transaction_bounds_skip_locally_erased<transactional_trivial_map_t>();
    test_transaction_bounds_skip_locally_erased<transactional_composite_map_t>();
    test_transaction_bounds_skip_locally_erased<transactional_heavy_map_t>();
}

int main() {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "basic_ops.empty_container_operations", basic_ops_empty_container_operations);
    failures += run_test(filter, "basic_ops.single_element_operations", basic_ops_single_element_operations);
    failures += run_test(filter, "basic_ops.insertion_patterns", basic_ops_insertion_patterns);
    failures += run_test(filter, "basic_ops.bulk_insertion_iterators", basic_ops_bulk_insertion_iterators);
    failures +=
        run_test(filter, "basic_ops.bulk_upsert_with_duplicate_pairs", basic_ops_bulk_upsert_with_duplicate_pairs);
    failures += run_test(filter, "basic_ops.range_query_head_state", basic_ops_range_query_head_state);
    failures += run_test(filter, "basic_ops.erase_range_head_state", basic_ops_erase_range_head_state);
    failures += run_test(filter, "basic_ops.heterogeneous_lookups", basic_ops_heterogeneous_lookups);

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
    failures += run_test(filter, "transactional_consistency.range_query_sees_atomic_boundaries",
                         transactional_consistency_range_query_sees_atomic_boundaries);
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
    failures += run_test(filter, "transactional_consistency.watch_detects_staged_writes_of_older_generation",
                         transactional_consistency_watch_detects_staged_writes_of_older_generation);
    failures += run_test(filter, "transactional_consistency.group_commits_participants_together",
                         transactional_consistency_group_commits_participants_together);
    failures += run_test(filter, "transactional_consistency.group_unwinds_every_participant_on_conflict",
                         transactional_consistency_group_unwinds_every_participant_on_conflict);
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
    failures += run_test(filter, "transactional_consistency.repeated_range_matches_isolation",
                         transactional_consistency_repeated_range_matches_isolation);
    failures +=
        run_test(filter, "transactional_consistency.delete_visibility", transactional_consistency_delete_visibility);
    failures += run_test(filter, "transactional_consistency.reset_clears_transaction_state",
                         transactional_consistency_reset_clears_transaction_state);

    failures += run_test(filter, "transactional_consistency.stateful_comparator_is_consulted",
                         transactional_consistency_stateful_comparator_is_consulted);

    failures += run_test(filter, "std_store_defects.committed_tombstones_are_reclaimable",
                         std_store_defects_committed_tombstones_are_reclaimable);
    failures +=
        run_test(filter, "std_store_defects.vacuum_respects_its_window", std_store_defects_vacuum_respects_its_window);
    failures += run_test(filter, "std_store_defects.erase_reports_and_reclaims_tombstone",
                         std_store_defects_erase_reports_and_reclaims_tombstone);
    failures += run_test(filter, "std_store_defects.erase_range_skips_tombstones",
                         std_store_defects_erase_range_skips_tombstones);
    failures += run_test(filter, "std_store_defects.vacuum_leaves_staged_entries",
                         std_store_defects_vacuum_leaves_staged_entries);
    failures += run_test(filter, "std_store_defects.commit_reports_lost_versions",
                         std_store_defects_commit_reports_lost_versions);
    failures += run_test(filter, "std_store_defects.commit_reports_success_when_published",
                         std_store_defects_commit_reports_success_when_published);
    failures += run_test(filter, "std_store_defects.staged_entries_keep_one_generation",
                         std_store_defects_staged_entries_keep_one_generation);

    failures += run_test(filter, "commit_stamp.rolled_back_stage_does_not_abort_a_peer",
                         commit_stamp_rolled_back_stage_does_not_abort_a_peer);
    failures += run_test(filter, "commit_stamp.lost_update_is_refused", commit_stamp_lost_update_is_refused);
    failures +=
        run_test(filter, "commit_stamp.find_and_watch_records_absence", commit_stamp_find_and_watch_records_absence);

    failures += run_test(filter, "transactional_consistency.find_does_not_watch",
                         transactional_consistency_find_does_not_watch);

    failures += run_test(filter, "std_store_defects.clear_keeps_generations_running",
                         std_store_defects_clear_keeps_generations_running);
    failures += run_test(filter, "std_store_defects.insert_refuses_with_key_already_exists",
                         std_store_defects_insert_refuses_with_key_already_exists);
    failures += run_test(filter, "std_store_defects.insert_reports_the_stored_element",
                         std_store_defects_insert_reports_the_stored_element);
    failures += run_test(filter, "std_store_defects.insert_after_local_erase_succeeds",
                         std_store_defects_insert_after_local_erase_succeeds);
    failures += run_test(filter, "std_store_defects.stage_refuses_when_already_staged",
                         std_store_defects_stage_refuses_when_already_staged);
    failures += run_test(filter, "std_store_defects.rollback_reports_lost_versions",
                         std_store_defects_rollback_reports_lost_versions);
    failures += run_test(filter, "std_store_defects.transaction_bounds_skip_locally_erased",
                         std_store_defects_transaction_bounds_skip_locally_erased);

    return report_test_failures(failures);
}
