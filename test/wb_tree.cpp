/**
 *  @file test/wb_tree.cpp
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Test instantiations for weight-balanced tree containers. Covers basic_wb_tree
 *      (non-transactional) and monotonic_store<basic_wb_tree> (transactional). The weight-balanced
 *      tree also carries order statistics - @c rank and @c select.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <map>    // `std::map`
#include <set>    // `std::set`
#include <vector> // `std::vector`

#include <smashtable/basic_wb_tree.hpp>
#include <smashtable/reference_store.hpp>
#include <smashtable/monotonic_store.hpp>

#include "harness.hpp"
#include "fixtures.hpp"
#include "surfaces.hpp"
#include "commit_stamp.hpp"
#include "consistency.hpp"
#include "fixture_coverage.hpp"
#include "monotonic_store_defects.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Type Aliases

/**
 *  Tests: Baseline non-transparent comparator path.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Transaction: ✗.
 */
using trivial_set_t = wb_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  Tests: Resource accounting, allocation failure injection.
 *  Heterogeneous lookup: ✓.
 *  Copy: Trivial.
 *  Memory: Tracked.
 *  Transaction: ✗.
 */
using tracking_set_t = wb_set<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Tests: Identifier extraction, @c composite_key_compare_t::value_type lookups.
 *  Heterogeneous lookup: ✓, over @c uint64_t.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Transaction: ✗.
 */
using composite_set_t = wb_set<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/**
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() → expected<T>.
 *  Memory: Heap |.
 *  Transaction: ✗ Tests: OOM during .copy(), string_view lookups without materialization.
 */
using heavy_set_t = wb_set<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/**
 *  Tests: Baseline map operations, non-transparent path.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Stack.
 *  Transaction: ✗.
 *  Value: int.
 */
using trivial_map_t = wb_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Tests: Map resource accounting, POCCA/POCMA on key-value pairs.
 *  Heterogeneous lookup: ✓.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Tracked.
 *  Transaction: ✗.
 *  Value: int.
 */
using tracking_map_t = wb_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Tests: Mixed trivial and non-trivial members, value OOM scenarios.
 *  Lookup: heterogeneous over uint64_t.
 *  Copy: key trivial, value .copy().
 *  Memory: heap value. Transactions: ✗.
 *  Value: guarded_payload_t.
 */
using composite_map_t = wb_map<composite_key_t, guarded_payload_t, composite_key_compare_t,
                               std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/**
 *  Tests: Dual-heap OOM, worst-case complexity.
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() on key & value.
 *  Memory: Heap, over @c both |.
 *  Transaction: ✗ Value: guarded_payload_t.
 */
using heavy_map_t =
    wb_map<heavy_key_t, guarded_payload_t, std::less<void>, std::allocator<mapping<heavy_key_t, guarded_payload_t>>>;

/**
 *  Tests: Baseline transactional correctness, isolation levels.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Transaction: ✓.
 */
using transactional_trivial_set_t =
    monotonic_wb_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  Tests: Transaction resource accounting, OOM during stage/commit.
 *  Heterogeneous lookup: ✓.
 *  Copy: Trivial.
 *  Memory: Tracked.
 *  Transaction: ✓.
 */
using transactional_tracking_set_t = monotonic_wb_set<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Tests: Heterogeneous watch/find in transactions.
 *  Heterogeneous lookup: ✓, over @c uint64_t.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Transaction: ✓.
 */
using transactional_composite_set_t =
    monotonic_wb_set<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/**
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() → expected<T>.
 *  Memory: Heap |.
 *  Transaction: ✓ Tests: Watch copy OOM, transaction rollback with heap types.
 */
using transactional_heavy_set_t = monotonic_wb_set<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/**
 *  Tests: Transactional map operations, value overwrites.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Stack.
 *  Transaction: ✓.
 *  Value: int.
 */
using transactional_trivial_map_t =
    monotonic_wb_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Tests: Transaction allocation patterns, map POCCA/POCMA.
 *  Heterogeneous lookup: ✓.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Tracked.
 *  Transaction: ✓.
 *  Value: int.
 */
using transactional_tracking_map_t = monotonic_wb_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Tests: Transaction rollback with non-trivial values.
 *  Heterogeneous lookup: ✓, over @c uint64_t.
 *  Copy: Key trivial, value .copy().
 *  Memory: Heap, over @c value |.
 *  Transaction: ✓ Value: guarded_payload_t.
 */
using transactional_composite_map_t = monotonic_wb_map<composite_key_t, guarded_payload_t, composite_key_compare_t,
                                                       std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/**
 *  Tests: Worst-case transactional complexity, dual-heap rollback.
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() on key & value.
 *  Memory: Heap, over @c both |.
 *  Transaction: ✓ Value: guarded_payload_t.
 */
using transactional_heavy_map_t = monotonic_wb_map<heavy_key_t, guarded_payload_t, std::less<void>,
                                                   std::allocator<mapping<heavy_key_t, guarded_payload_t>>>;

#pragma endregion Type Aliases

#pragma region Basic Operations Tests

/** Tests operations on empty container don't crash */
static void test_empty_container_operations() {
    check_empty_container_operations<trivial_set_t>();
    check_empty_container_operations<tracking_set_t>();
    check_empty_container_operations<composite_set_t>();
    check_empty_container_operations<heavy_set_t>();
    check_empty_container_operations<trivial_map_t>();
    check_empty_container_operations<tracking_map_t>();
    check_empty_container_operations<composite_map_t>();
    check_empty_container_operations<heavy_map_t>();
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
    check_single_element_operations<trivial_set_t>();
    check_single_element_operations<tracking_set_t>();
    check_single_element_operations<composite_set_t>();
    check_single_element_operations<heavy_set_t>();
    check_single_element_operations<trivial_map_t>();
    check_single_element_operations<tracking_map_t>();
    check_single_element_operations<composite_map_t>();
    check_single_element_operations<heavy_map_t>();
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
    check_basic_insertion_patterns<trivial_set_t>();
    check_basic_insertion_patterns<tracking_set_t>();
    check_basic_insertion_patterns<composite_set_t>();
    check_basic_insertion_patterns<heavy_set_t>();
    check_basic_insertion_patterns<trivial_map_t>();
    check_basic_insertion_patterns<tracking_map_t>();
    check_basic_insertion_patterns<composite_map_t>();
    check_basic_insertion_patterns<heavy_map_t>();
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
    check_bulk_insertion_from_iterators<trivial_set_t>();
    check_bulk_insertion_from_iterators<tracking_set_t>();
    check_bulk_insertion_from_iterators<composite_set_t>();
    check_bulk_insertion_from_iterators<heavy_set_t>();
    check_bulk_insertion_from_iterators<trivial_map_t>();
    check_bulk_insertion_from_iterators<tracking_map_t>();
    check_bulk_insertion_from_iterators<composite_map_t>();
    check_bulk_insertion_from_iterators<heavy_map_t>();
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
    // ! Only applies to maps with integral mapped values
    check_bulk_upsert_with_duplicates<trivial_map_t>();
    check_bulk_upsert_with_duplicates<tracking_map_t>();
    check_bulk_upsert_with_duplicates<transactional_trivial_map_t>();
    check_bulk_upsert_with_duplicates<transactional_tracking_map_t>();
}

/** Tests range queries on committed HEAD state */
static void test_range_query_head_state() {
    // ! Only applies to "non-heavy" keys to simplify the test implementation
    check_range_query_head_state<trivial_set_t>();
    check_range_query_head_state<tracking_set_t>();
    check_range_query_head_state<composite_set_t>();
    check_range_query_head_state<trivial_map_t>();
    check_range_query_head_state<tracking_map_t>();
    check_range_query_head_state<composite_map_t>();
    check_range_query_head_state<transactional_trivial_set_t>();
    check_range_query_head_state<transactional_tracking_set_t>();
    check_range_query_head_state<transactional_composite_set_t>();
    check_range_query_head_state<transactional_trivial_map_t>();
    check_range_query_head_state<transactional_tracking_map_t>();
    check_range_query_head_state<transactional_composite_map_t>();
}

/** Tests erase_range on committed HEAD state */
static void test_erase_range_head_state() {
    // ! Only applies to "non-heavy" keys to simplify the test implementation
    check_erase_range_head_state<trivial_set_t>();
    check_erase_range_head_state<tracking_set_t>();
    check_erase_range_head_state<composite_set_t>();
    check_erase_range_head_state<heavy_set_t>();
    check_erase_range_head_state<trivial_map_t>();
    check_erase_range_head_state<tracking_map_t>();
    check_erase_range_head_state<composite_map_t>();
    check_erase_range_head_state<heavy_map_t>();
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
    check_heterogeneous_composite_find<composite_set_t>();
    check_heterogeneous_heavy_string_view_find<heavy_set_t>();
    check_heterogeneous_composite_find<composite_map_t>();
    check_heterogeneous_heavy_string_view_find<heavy_map_t>();
    check_heterogeneous_composite_find<transactional_composite_set_t>();
    check_heterogeneous_heavy_string_view_find<transactional_heavy_set_t>();
    check_heterogeneous_composite_find<transactional_composite_map_t>();
    check_heterogeneous_heavy_string_view_find<transactional_heavy_map_t>();
}

#pragma endregion Basic Operations Tests

#pragma region Consistency and Transaction Tests for Sets

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

static void test_erase_range_spares_staged_versions() {
    check_erase_range_spares_staged_versions<transactional_trivial_map_t>();
    check_erase_range_spares_staged_versions<transactional_composite_map_t>();
    check_erase_range_spares_staged_versions<transactional_heavy_map_t>();
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

static void test_committed_erase_hidden_from_ordered_reads() {
    check_committed_erase_hidden_from_ordered_reads<transactional_trivial_map_t>();
    check_committed_erase_hidden_from_ordered_reads<transactional_composite_map_t>();
    check_committed_erase_hidden_from_ordered_reads<transactional_heavy_map_t>();
}

static void test_committed_erase_hidden_from_update_range() {
    check_committed_erase_hidden_from_update_range<transactional_trivial_map_t>();
    check_committed_erase_hidden_from_update_range<transactional_composite_map_t>();
    check_committed_erase_hidden_from_update_range<transactional_heavy_map_t>();
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

static void test_windowed_vacuum_reclaims_one_slice() {
    check_windowed_vacuum_reclaims_one_slice<transactional_trivial_map_t>();
    check_windowed_vacuum_reclaims_one_slice<transactional_composite_map_t>();
    check_windowed_vacuum_reclaims_one_slice<transactional_heavy_map_t>();
}

#pragma endregion Transactional Store Defects

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

#pragma region Weight Balance Invariants

/** A bare @c int set, because the balance oracle below reaches for @c root() and walks raw nodes.
 *  The keys carry no behaviour of their own, so a failure is always the tree's. */
using ordered_set_t = wb_set<int, std::less<int>, std::allocator<int>>;
using ordered_node_t = ordered_set_t::node_t;

/** What one subtree contributes, so the oracle can compare both counts at once. */
struct subtree_counts_t {

    /** Number of nodes in the subtree. */
    std::size_t size = 0;

    /** Number of nodes the augmentation policy counts, zero for an unaugmented tree. */
    std::size_t augmented_size = 0;
};

/** Recomputes one subtree's counts while checking every structural invariant. Δ=3 over @c size+1
 *  weights, size = 1 + size(left) + size(right), and, where the tree carries an augmentation, the
 *  same recurrence over the policy's per-entry count. */
template <typename node_type_>
static subtree_counts_t check_invariants(node_type_ *node) noexcept {
    if (!node) return {};
    subtree_counts_t const left = check_invariants(node->left);
    subtree_counts_t const right = check_invariants(node->right);
    expect_equal(node->size, 1 + left.size + right.size);

    std::size_t const left_weight = left.size + 1, right_weight = right.size + 1;
    expect_true(left_weight <= node_type_::delta_k * right_weight);
    expect_true(right_weight <= node_type_::delta_k * left_weight);

    subtree_counts_t counts;
    counts.size = node->size;
    if constexpr (node_type_::is_augmented_k) {
        counts.augmented_size = node_type_::get_own_augmented_count(node) + left.augmented_size + right.augmented_size;
        expect_equal(node->augmented_size, counts.augmented_size);
    }
    return counts;
}

/** Checks the tree against a @c std::set oracle: invariants, size, and in-order contents. */
static void check_against_oracle(ordered_set_t &tree, std::set<int> const &oracle) noexcept {
    expect_equal(check_invariants(tree.root()).size, oracle.size());
    expect_equal(tree.size(), oracle.size());

    auto expected = oracle.begin();
    expect_equal(tree.for_each([&](int const &element) noexcept {
        if (!expect_true(expected != oracle.end())) return;
        expect_equal(element, *expected++);
    }),
                 success_k);
    expect_true(expected == oracle.end());
}

/** Sequential keys must not degenerate into a spine - every entry point has to rebalance. */
static void test_insert_rebalances() {
    for (std::size_t count : {std::size_t(1), std::size_t(2), std::size_t(1000)}) {
        ordered_set_t ascending, descending;
        std::set<int> oracle;
        for (std::size_t index = 0; index < count; ++index) {
            [[maybe_unused]] auto const added = ascending.insert(int(index));
            [[maybe_unused]] auto const subtracted = descending.insert(int(count - index));
            oracle.insert(int(index));
        }
        check_against_oracle(ascending, oracle);
        [[maybe_unused]] subtree_counts_t const counted = check_invariants(descending.root());
        expect_equal(descending.size(), count);
    }

    // A rejected duplicate leaves the tree untouched and reports the incumbent.
    using placement_t = ordered_set_t::node_t::node_placement_t;
    ordered_set_t tree;
    [[maybe_unused]] auto const first = tree.insert(7);
    auto const duplicate = tree.insert(7);
    expect_true(duplicate.placement == placement_t::matched_k);
    if (!expect_true(duplicate.node != nullptr)) return;
    expect_equal(duplicate.node->payload, 7);
    expect_equal(tree.size(), 1u);
}

/** Promoting the right subtree's minimum over a two-child erase must rebalance the promotion. */
static void test_two_child_erase_rebalances() {
    ordered_set_t tree;
    std::set<int> oracle;
    for (int element : {10, 5, 15, 3, 7}) {
        [[maybe_unused]] auto const added = tree.insert(int(element));
        oracle.insert(element);
    }
    check_against_oracle(tree, oracle);

    expect_true(tree.erase(10));
    oracle.erase(10);
    check_against_oracle(tree, oracle);
}

/** Both halves of a split own their element counts, their allocator, and a balanced shape. */
static void test_split_and_join_track_size() {
    split_mix_generator_t generator {case_stream_key()};
    for (std::size_t trial = 0; trial < 200; ++trial) {
        ordered_set_t tree;
        std::set<int> oracle;
        std::size_t const count = generator() % 400;
        for (std::size_t index = 0; index < count; ++index) {
            int const element = int(generator() % 1000);
            [[maybe_unused]] auto const added = tree.insert(int(element));
            oracle.insert(element);
        }

        int const pivot = int(generator() % 1000);
        auto halves = tree.split(pivot);
        std::set<int> below(oracle.begin(), oracle.lower_bound(pivot));
        std::set<int> above(oracle.lower_bound(pivot), oracle.end());
        expect_equal(halves.left.empty(), below.empty());
        check_against_oracle(halves.left, below);
        check_against_oracle(halves.right, above);

        halves.left.join(halves.right);
        check_against_oracle(halves.left, oracle);
        expect_true(halves.right.empty());
    }
}

/** Half-open [lower, upper) range walks, visited in sorted order. */
static void test_range_is_half_open() {
    ordered_set_t tree;
    std::set<int> oracle;
    for (int element = 0; element < 10; ++element) {
        [[maybe_unused]] auto const added = tree.insert(int(element));
        oracle.insert(element);
    }

    std::vector<int> visited;
    expect_equal(tree.range(2, 5, [&](int const &element) noexcept { visited.push_back(element); }), success_k);
    std::vector<int> const expected {2, 3, 4};
    expect_true(visited == expected);

    // An empty window yields nothing, rather than the single element at its edge.
    visited.clear();
    expect_equal(tree.range(4, 4, [&](int const &element) noexcept { visited.push_back(element); }), success_k);
    expect_true(visited.empty());
}

/** Erasing an iterator range compiles, reports success, and leaves a balanced tree. */
static void test_erase_iterator_range() {
    ordered_set_t tree;
    std::set<int> oracle;
    for (int element = 0; element < 200; ++element) {
        [[maybe_unused]] auto const added = tree.insert(int(element));
        oracle.insert(element);
    }

    auto const result = tree.erase(tree.lower_bound(40), tree.lower_bound(160));
    oracle.erase(oracle.lower_bound(40), oracle.lower_bound(160));
    expect_equal(result.status, success_k);
    check_against_oracle(tree, oracle);

    auto const empty_result = tree.erase(tree.begin(), tree.begin());
    expect_equal(empty_result.status, success_k);
    check_against_oracle(tree, oracle);
}

/** Dropping an arbitrary subset must leave a balanced tree, not merely a correct one. */
static void test_erase_if_rebalances() {
    split_mix_generator_t generator {case_stream_key()};
    for (std::size_t trial = 0; trial < 200; ++trial) {
        ordered_set_t tree;
        std::set<int> oracle;
        for (std::size_t index = 0; index < 300; ++index) {
            int const element = int(generator() % 1000);
            [[maybe_unused]] auto const added = tree.insert(int(element));
            oracle.insert(element);
        }

        int const residue = int(generator() % 7);
        auto const drops = [&](int element) noexcept { return element % 7 == residue; };
        expect_equal(tree.erase_if(drops), std::erase_if(oracle, drops));
        check_against_oracle(tree, oracle);
    }
}

/** A long randomized mutation sequence, re-checking every invariant after each step. */
static void test_randomized_mutations() {
    using placement_t = ordered_set_t::node_t::node_placement_t;
    split_mix_generator_t generator {case_stream_key()};
    ordered_set_t tree;
    std::set<int> oracle;
    for (std::size_t step = 0; step < 20000; ++step) {
        int const element = int(generator() % 800);
        switch (generator() % 4) {
        case 0: {
            auto const result = tree.insert(int(element));
            expect_true(result.node != nullptr);
            expect_equal(result.placement == placement_t::made_k, oracle.insert(element).second);
            break;
        }
        case 1: {
            auto const result = tree.upsert(int(element));
            expect_true(bool(result));
            oracle.insert(element);
            break;
        }
        case 2: expect_equal(tree.erase(element), oracle.erase(element) != 0); break;
        case 3: {
            int const upper = element + int(generator() % 200);
            std::size_t visited = 0;
            tree.erase_range(element, upper, [&](int const &) noexcept { ++visited; });
            std::size_t const expected =
                std::size_t(std::distance(oracle.lower_bound(element), oracle.lower_bound(upper)));
            expect_equal(visited, expected);
            oracle.erase(oracle.lower_bound(element), oracle.lower_bound(upper));
            break;
        }
        }
        check_against_oracle(tree, oracle);
    }
}

#pragma endregion Weight Balance Invariants

#pragma region Augmented Order Statistics

/** Counts only the entries whose mapped value is non-zero. Stands in for a wrapper's "this entry is
 *  the answer for its key" predicate: a property of the entry, decided outside the tree, and free
 *  to flip while the entry sits in place. */
struct live_augmentation_t {
    static std::size_t augmented_count(mapping<int, int> const &entry) noexcept { return entry.mapped != 0 ? 1u : 0u; }
};

using augmented_map_t = wb_map<int, int, std::less<int>, std::allocator<mapping<int, int>>, live_augmentation_t>;
using augmented_node_t = augmented_map_t::node_t;
using plain_map_t = wb_map<int, int, std::less<int>, std::allocator<mapping<int, int>>>;

/** The layout an unaugmented node must match byte for byte, whatever the default policy costs. */
struct unaugmented_reference_layout_t {
    int payload;
    void *left;
    void *right;
    std::size_t size;
};

static_assert(std::is_empty_v<no_augmentation_t>, "The default policy must be an empty type");
static_assert(sizeof(ordered_node_t) == sizeof(unaugmented_reference_layout_t),
              "The default augmentation must not add a byte to the node");
static_assert(alignof(ordered_node_t) == alignof(unaugmented_reference_layout_t),
              "The default augmentation must not change the node's alignment");

/**
 *  @brief The layout an unaugmented @b map node must match, whose element is a pair, not a key.
 *
 *  Kept apart from the set's reference because the two only agree where the padding after a lone
 *  @c int happens to swallow the difference: on LP64 both come to 32 bytes, and on a 32-bit target
 *  the set's is 16 while the map's is 20.
 */
struct unaugmented_map_reference_layout_t {
    mapping<int, int> payload;
    void *left;
    void *right;
    std::size_t size;
};

static_assert(sizeof(plain_map_t::node_t) == sizeof(unaugmented_map_reference_layout_t),
              "A map node is the same shape once the default policy is elided");
static_assert(sizeof(augmented_node_t) == sizeof(plain_map_t::node_t) + sizeof(std::size_t),
              "An augmented node pays exactly one counter, and nothing else");

/** The keys the oracle considers counted, in sorted order. */
static std::vector<int> counted_keys(std::map<int, int> const &oracle) noexcept {
    std::vector<int> keys;
    for (auto const &entry : oracle)
        if (entry.second != 0) keys.push_back(entry.first);
    return keys;
}

/** Checks structure, the augmented total, and every augmented @c select and @c rank answer. */
static void check_augmented_against_oracle(augmented_map_t &tree, std::map<int, int> const &oracle) noexcept {
    subtree_counts_t const counts = check_invariants(tree.root());
    expect_equal(counts.size, oracle.size());
    expect_equal(tree.size(), oracle.size());

    std::vector<int> const live = counted_keys(oracle);
    expect_equal(tree.augmented_size(), live.size());
    expect_equal(counts.augmented_size, live.size());

    for (std::size_t index = 0; index < live.size(); ++index) {
        augmented_node_t const *selected = tree.select_augmented(index);
        if (!expect_true(selected != nullptr)) return;
        expect_equal(selected->payload.key, live[index]);
        expect_equal(tree.rank_augmented(live[index]), index);
    }
    expect_equal(tree.select_augmented(live.size()), nullptr);
}

/** The same mutation fuzz the plain tree runs, plus in-place predicate flips. A flip changes an
 *  entry the tree already holds, which is the transition a version store makes when a newer version
 *  supersedes an older one, and it must cost a path repair rather than a rescan. */
static void test_augmented_randomized_mutations() {
    using placement_t = augmented_map_t::node_t::node_placement_t;
    split_mix_generator_t generator {case_stream_key()};
    augmented_map_t tree;
    std::map<int, int> oracle;
    for (std::size_t step = 0; step < 20000; ++step) {
        int const key = int(generator() % 200);
        switch (generator() % 7) {
        case 0: {
            int const mapped = int(generator() % 2);
            auto const result = tree.insert(mapping<int, int> {key, mapped});
            expect_true(result.node != nullptr);
            expect_equal(result.placement == placement_t::made_k, oracle.emplace(key, mapped).second);
            break;
        }
        case 1: {
            int const mapped = int(generator() % 2);
            auto const result = tree.upsert(mapping<int, int> {key, mapped});
            expect_true(bool(result));
            oracle[key] = mapped;
            break;
        }
        case 2: expect_equal(tree.erase(key), oracle.erase(key) != 0); break;
        case 3: {
            int const upper = key + int(generator() % 60);
            std::size_t visited = 0;
            tree.erase_range(key, upper, [&](mapping<int, int> const &) noexcept { ++visited; });
            std::size_t const expected = std::size_t(std::distance(oracle.lower_bound(key), oracle.lower_bound(upper)));
            expect_equal(visited, expected);
            oracle.erase(oracle.lower_bound(key), oracle.lower_bound(upper));
            break;
        }
        case 4: {
            // Flip the count under the tree's feet, as a publish site would - and flip the successor too,
            // since one publish can supersede a neighbour, retagging two entries for a single write.
            auto position = tree.find(key);
            expect_equal(position != tree.end(), oracle.count(key) != 0);
            for (std::size_t flipped = 0; flipped < 2 && position != tree.end(); ++flipped, ++position) {
                int const flipped_key = position->key;
                position->mapped = position->mapped != 0 ? 0 : 1;
                oracle[flipped_key] = position->mapped;
                expect_true(tree.refresh_augmentation(flipped_key));
            }
            break;
        }
        case 5: {
            // `erase_if` rebuilds both halves through `join`, which has to carry the counts along.
            int const residue = int(generator() % 5);
            std::size_t const dropped =
                tree.erase_if([&](mapping<int, int> const &entry) noexcept { return entry.key % 5 == residue; });
            expect_equal(dropped,
                         std::erase_if(oracle, [&](auto const &entry) noexcept { return entry.first % 5 == residue; }));
            break;
        }
        case 6: {
            // Splitting and re-joining walks `join_with_root` down both spines.
            auto halves = tree.split(key);
            check_augmented_against_oracle(halves.left, std::map<int, int>(oracle.begin(), oracle.lower_bound(key)));
            check_augmented_against_oracle(halves.right, std::map<int, int>(oracle.lower_bound(key), oracle.end()));
            halves.left.join(halves.right);
            tree = std::move(halves.left);
            break;
        }
        }
        check_augmented_against_oracle(tree, oracle);
    }
}

using counted_map_t = wb_map<int, int, counting_comparator_t, std::allocator<mapping<int, int>>, live_augmentation_t>;

/** Depth of @p wanted below the root, which is exactly what a @c select_augmented descent walks. */
static std::size_t depth_of(counted_map_t const &tree, int wanted) noexcept {
    std::size_t depth = 0;
    for (auto const *node = tree.root(); node; ++depth) {
        if (wanted < node->payload.key) node = node->left;
        else if (node->payload.key < wanted) node = node->right;
        else return depth;
    }
    expect_true(false, "The selected key must be reachable from the root");
    return depth;
}

/** Base-2 logarithm of @p count, rounded down, with @c log2(0) reported as 0. */
static std::size_t floor_log2(std::size_t count) noexcept {
    std::size_t bits = 0;
    while (count > 1) count >>= 1, ++bits;
    return bits;
}

/**
 *  @brief Pins augmented @c select and @c rank to @c log2(size) exactly, at two sizes 64× apart.
 *
 *  Ascending keys and a power-of-two count leave the tree perfectly balanced, so its height is
 *  exactly @c log2(n) and every count below is an equality rather than a bound. The probe is
 *  exhaustive because the defect this guards is one pathological path, which a sample could miss.
 */
static void test_augmented_select_is_logarithmic() {
    for (std::size_t element_count : {std::size_t(4096), std::size_t(262144)}) {
        counted_map_t tree(counting_comparator_t {}, std::allocator<counted_map_t::node_t> {});

        // Every third key counts, so the augmented descent cannot degenerate into the plain one.
        std::vector<int> live;
        for (std::size_t index = 0; index < element_count; ++index) {
            int const key = int(index);
            int const mapped = index % 3 == 0 ? 1 : 0;
            [[maybe_unused]] auto const added = tree.upsert(mapping<int, int> {key, mapped});
            if (mapped) live.push_back(key);
        }
        expect_equal(tree.augmented_size(), live.size());

        std::size_t const budget = floor_log2(element_count);
        std::size_t worst_select_steps = 0, worst_rank_comparisons = 0;
        for (std::size_t index = 0; index < live.size(); ++index) {
            auto const *selected = tree.select_augmented(index);
            if (!expect_true(selected != nullptr)) return;
            expect_equal(selected->payload.key, live[index]);
            // The node's own depth, measured by an independent descent - `select_augmented` navigates on
            // subtree sizes and calls no comparator, so nothing can count its steps from the inside.
            std::size_t const steps = depth_of(tree, live[index]) + 1;
            worst_select_steps = steps > worst_select_steps ? steps : worst_select_steps;

            counting_call_tally_t::reset();
            expect_equal(tree.rank_augmented(live[index]), index);
            std::size_t const comparisons = counting_call_tally_t::comparisons_count();
            worst_rank_comparisons = comparisons > worst_rank_comparisons ? comparisons : worst_rank_comparisons;
        }

        // One step a level down a tree of height `budget`. `rank` spends one comparison per left turn and
        // two per right, and the deepest live key here sits on the all-right spine.
        expect_equal(worst_select_steps, budget + 1);
        expect_equal(worst_rank_comparisons, 2 * budget + 2);
    }
}

#pragma endregion Augmented Order Statistics

#pragma region Allocation Failure

/** A key that rewrites itself when moved from, so a consumed argument is visible to the oracle. */
struct traced_key_t {
    int value = 0;
    bool moved_from = false;

    traced_key_t() = default;
    explicit traced_key_t(int initial) noexcept : value(initial) {}
    traced_key_t(traced_key_t const &) = default;
    traced_key_t &operator=(traced_key_t const &) = default;
    traced_key_t(traced_key_t &&other) noexcept : value(other.value) { other.moved_from = true; }
    traced_key_t &operator=(traced_key_t &&other) noexcept {
        value = other.value;
        other.moved_from = true;
        return *this;
    }
};

struct traced_less_t {
    bool operator()(traced_key_t const &first, traced_key_t const &second) const noexcept {
        return first.value < second.value;
    }
};

using traced_set_t = wb_set<traced_key_t, traced_less_t, stateful_allocator<void>>;

/** Neither a duplicate key nor an exhausted allocator may consume the caller's entry. */
static void test_allocation_failure_preserves_the_argument() {
    using placement_t = traced_set_t::node_t::node_placement_t;
    allocation_ledger_t ledger;
    ledger.allow(3);
    {
        traced_set_t tree(traced_less_t {}, stateful_allocator<traced_set_t::node_t>(ledger));
        for (int element = 0; element < 3; ++element) {
            traced_key_t key(element);
            auto const made = tree.insert(std::move(key));
            expect_true(made.placement == placement_t::made_k);
            expect_true(made.node != nullptr);
        }

        // Budget exhausted - the rejected entry must come back untouched.
        traced_key_t rejected(99);
        auto const refused = tree.insert(std::move(rejected));
        expect_equal(refused.node, nullptr);
        expect_true(refused.placement == placement_t::refused_k);
        expect_true(!rejected.moved_from);

        // A duplicate key must likewise leave the argument alone and report the incumbent.
        ledger.allow(10);
        traced_key_t duplicate(1);
        auto const matched = tree.insert(std::move(duplicate));
        expect_true(matched.node != nullptr);
        expect_true(matched.placement == placement_t::matched_k);
        expect_true(!duplicate.moved_from);
        expect_equal(tree.size(), 3u);
    }
    ledger.check_balanced();
}

/**
 *  @brief An upsert must name which of its three outcomes happened.
 *
 *  A node that was made, one that was already there, and an allocation that never came back are
 *  three answers, and none of them should have to be read off a null check.
 */
static void test_upsert_reports_placement() {
    using placement_t = traced_set_t::node_t::node_placement_t;
    allocation_ledger_t ledger;
    ledger.allow(1);
    {
        traced_set_t tree(traced_less_t {}, stateful_allocator<traced_set_t::node_t>(ledger));

        auto const made = tree.upsert(traced_key_t(1));
        expect_equal(made.placement, placement_t::made_k, "a fresh key must report a node of its own");
        expect_true(!made.failed());
        expect_true(made.node != nullptr);

        auto const matched = tree.upsert(traced_key_t(1));
        expect_equal(matched.placement, placement_t::matched_k, "a key already there must report a match");
        expect_true(succeeded(matched), "an overwrite is not a failure");
        expect_true(matched.node != nullptr);

        // The budget is spent, so the second key has no node to live in - the outcome that shares a
        // null match with nothing else.
        auto const refused = tree.upsert(traced_key_t(2));
        expect_equal(refused.placement, placement_t::refused_k, "a refused allocation must say so");
        expect_true(failed(refused));
        expect_equal(refused.node, nullptr);
        expect_equal(tree.size(), 1u);
    }
    ledger.check_balanced();
}

/** @c insert_if_missing must separate "already there" from "out of memory". */
static void test_allocation_failure_is_distinct_from_presence() {
    using placement_t = traced_set_t::node_t::node_placement_t;
    allocation_ledger_t ledger;
    ledger.allow(2);
    {
        traced_set_t tree(traced_less_t {}, stateful_allocator<traced_set_t::node_t>(ledger));
        {
            traced_key_t key(1);
            auto const made = tree.insert(std::move(key));
            expect_true(made.placement == placement_t::made_k);
            expect_true(made.node != nullptr);
        }

        auto const on_present = tree.insert_if_missing(traced_key_t(1));
        expect_equal(on_present.placement, placement_t::matched_k, "an incumbent is not a refusal");
        expect_true(on_present.position != tree.end());

        ledger.refuse_everything();
        auto const on_exhausted = tree.insert_if_missing(traced_key_t(2));
        expect_equal(on_exhausted.placement, placement_t::refused_k, "no node means no insertion");
        expect_equal(on_exhausted.position, tree.end());
        expect_equal(tree.size(), 1u);
    }
    ledger.check_balanced();
}

#pragma endregion Allocation Failure

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

static void test_transaction_range_interleaves_staged_and_committed() {
    check_transaction_range_interleaves_staged_and_committed<transactional_trivial_map_t>();
    check_transaction_range_interleaves_staged_and_committed<transactional_composite_map_t>();
    check_transaction_range_interleaves_staged_and_committed<transactional_heavy_map_t>();
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
    check_container_balances_counted_keys<wb_map<counted_key_t, int>>();
}

static void test_rollback_balances_counted_keys() {
    check_rollback_balances_counted_keys<monotonic_wb_map<counted_key_t, int>>();
}

/** Every range modifier refused at every point it asks for memory leaves the tree as it was. */
void test_batch_is_all_or_nothing() {
    using ledger_set_t = wb_set<trivial_key_t, less_t, stateful_allocator_t>;
    check_every_offered_surface<ledger_set_t>(
        [](allocation_ledger_t &ledger) noexcept { return ledger_set_t(less_t {}, stateful_allocator_t(1, ledger)); });
}

#pragma endregion Fixture Coverage

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

    run_test_case("test_insert_rebalances", test_insert_rebalances);
    run_test_case("test_two_child_erase_rebalances", test_two_child_erase_rebalances);
    run_test_case("test_split_and_join_track_size", test_split_and_join_track_size);
    run_test_case("test_range_is_half_open", test_range_is_half_open);
    run_test_case("test_erase_iterator_range", test_erase_iterator_range);
    run_test_case("test_erase_if_rebalances", test_erase_if_rebalances);
    run_test_case("test_upsert_reports_placement", test_upsert_reports_placement);
    run_test_case("test_randomized_mutations", test_randomized_mutations);

    run_test_case("test_augmented_randomized_mutations", test_augmented_randomized_mutations);
    run_test_case("test_augmented_select_is_logarithmic", test_augmented_select_is_logarithmic);

    run_test_case("test_allocation_failure_preserves_the_argument", test_allocation_failure_preserves_the_argument);
    run_test_case("test_allocation_failure_is_distinct_from_presence",
                  test_allocation_failure_is_distinct_from_presence);

    run_test_case("test_direct_write_spares_staged_version", test_direct_write_spares_staged_version);
    run_test_case("test_direct_erase_spares_staged_version", test_direct_erase_spares_staged_version);
    run_test_case("test_erase_range_spares_staged_versions", test_erase_range_spares_staged_versions);
    run_test_case("test_clear_keeps_generations_moving", test_clear_keeps_generations_moving);
    run_test_case("test_committed_erase_hidden_from_point_reads", test_committed_erase_hidden_from_point_reads);
    run_test_case("test_committed_erase_hidden_from_ordered_reads", test_committed_erase_hidden_from_ordered_reads);
    run_test_case("test_committed_erase_hidden_from_update_range", test_committed_erase_hidden_from_update_range);
    run_test_case("test_vacuum_reclaims_committed_tombstones", test_vacuum_reclaims_committed_tombstones);
    run_test_case("test_vacuum_spares_staged_versions", test_vacuum_spares_staged_versions);
    run_test_case("test_windowed_vacuum_reclaims_one_slice", test_windowed_vacuum_reclaims_one_slice);

    run_test_case("test_rolled_back_stage_does_not_abort_a_peer", test_rolled_back_stage_does_not_abort_a_peer);
    run_test_case("test_lost_update_is_refused", test_lost_update_is_refused);
    run_test_case("test_find_and_watch_records_absence", test_find_and_watch_records_absence);

    run_test_case("test_find_does_not_watch", test_find_does_not_watch);

    run_test_case("test_transaction_range_interleaves_staged_and_committed",
                  test_transaction_range_interleaves_staged_and_committed);
    run_test_case("test_transaction_equal_range_sees_staged_writes", test_transaction_equal_range_sees_staged_writes);
    run_test_case("test_insert_reports_key_already_exists", test_insert_reports_key_already_exists);
    run_test_case("test_find_copy_reports_key_not_found", test_find_copy_reports_key_not_found);
    run_test_case("test_second_stage_is_rejected", test_second_stage_is_rejected);

    run_test_case("test_container_balances_counted_keys", test_container_balances_counted_keys);
    run_test_case("test_rollback_balances_counted_keys", test_rollback_balances_counted_keys);

    run_test_case("test_batch_is_all_or_nothing", test_batch_is_all_or_nothing);

    return finish_test_run();
}
