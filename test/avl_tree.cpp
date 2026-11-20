/**
 *  @file test/avl_tree.cpp
 *  @author Ash Vardanian
 *  @date October 25, 2025
 *  @brief Test instantiations for AVL tree containers. Covers basic_avl_tree (non-transactional)
 *      and monotonic_store<basic_avl_tree> (transactional). Includes AVL-specific algorithms
 *      (merge/split/join) and transaction architecture tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`

#include <iterator> // `std::bidirectional_iterator`
#include <memory>   // `std::unique_ptr`
#include <set>      // `std::set`
#include <vector>   // `std::vector`

#include <smashtable/basic_avl_tree.hpp>
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
using trivial_set_t = avl_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  Tests: Resource accounting, allocation failure injection.
 *  Heterogeneous lookup: ✓.
 *  Copy: Trivial.
 *  Memory: Tracked.
 *  Transaction: ✗.
 */
using tracking_set_t = avl_set<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Tests: Identifier extraction, @c composite_key_compare_t::value_type lookups.
 *  Heterogeneous lookup: ✓, over @c uint64_t.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Transaction: ✗.
 */
using composite_set_t = avl_set<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/**
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() → expected<T>.
 *  Memory: Heap |.
 *  Transaction: ✗ Tests: OOM during .copy(), string_view lookups without materialization.
 */
using heavy_set_t = avl_set<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/**
 *  Tests: Baseline map operations, non-transparent path.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Stack.
 *  Transaction: ✗.
 *  Value: int.
 */
using trivial_map_t =
    avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Tests: Map resource accounting, POCCA/POCMA on key-value pairs.
 *  Heterogeneous lookup: ✓.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Tracked.
 *  Transaction: ✗.
 *  Value: int.
 */
using tracking_map_t = avl_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Tests: Mixed trivial and non-trivial members, value OOM scenarios.
 *  Heterogeneous lookup: ✓ uint64_t.
 *  Copy: Key trivial, value .copy().
 *  Memory: Heap value.
 *  Transaction: ✗.
 *  Value: guarded_payload_t.
 */
using composite_map_t = avl_map<composite_key_t, guarded_payload_t, composite_key_compare_t,
                                std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/**
 *  Tests: Dual-heap OOM, worst-case complexity.
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() on key & value.
 *  Memory: Heap, over @c both |.
 *  Transaction: ✗ Value: guarded_payload_t.
 */
using heavy_map_t =
    avl_map<heavy_key_t, guarded_payload_t, std::less<void>, std::allocator<mapping<heavy_key_t, guarded_payload_t>>>;

/**
 *  Tests: Baseline transactional correctness, isolation levels.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Transaction: ✓.
 */
using transactional_trivial_set_t =
    monotonic_avl_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  Tests: Transaction resource accounting, OOM during stage/commit.
 *  Heterogeneous lookup: ✓.
 *  Copy: Trivial.
 *  Memory: Tracked.
 *  Transaction: ✓.
 */
using transactional_tracking_set_t = monotonic_avl_set<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Tests: Heterogeneous watch/find in transactions.
 *  Heterogeneous lookup: ✓, over @c uint64_t.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Transaction: ✓.
 */
using transactional_composite_set_t =
    monotonic_avl_set<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/**
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() → expected<T>.
 *  Memory: Heap |.
 *  Transaction: ✓ Tests: Watch copy OOM, transaction rollback with heap types.
 */
using transactional_heavy_set_t = monotonic_avl_set<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/**
 *  Tests: Transactional map operations, value overwrites.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Stack.
 *  Transaction: ✓.
 *  Value: int.
 */
using transactional_trivial_map_t =
    monotonic_avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Tests: Transaction allocation patterns, map POCCA/POCMA.
 *  Heterogeneous lookup: ✓.
 *  Copy: Trivial, over `key & value`.
 *  Memory: Tracked.
 *  Transaction: ✓.
 *  Value: int.
 */
using transactional_tracking_map_t = monotonic_avl_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Tests: Transaction rollback with non-trivial values.
 *  Heterogeneous lookup: ✓, over @c uint64_t.
 *  Copy: Key trivial, value .copy().
 *  Memory: Heap, over @c value |.
 *  Transaction: ✓ Value: guarded_payload_t.
 */
using transactional_composite_map_t = monotonic_avl_map<composite_key_t, guarded_payload_t, composite_key_compare_t,
                                                        std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/**
 *  Tests: Worst-case transactional complexity, dual-heap rollback.
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: .copy() on key & value.
 *  Memory: Heap, over @c both |.
 *  Transaction: ✓ Value: guarded_payload_t.
 */
using transactional_heavy_map_t = monotonic_avl_map<heavy_key_t, guarded_payload_t, std::less<void>,
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

#pragma region Structural Invariants

/** Node count and height of a walked subtree. */
struct subtree_shape_t {
    std::size_t count = 0;
    std::ptrdiff_t height = 0;
};

/** Recursively checks parent back-pointers, stored heights, AVL balance, and key order. */
template <typename tree_type_>
static subtree_shape_t check_subtree(typename tree_type_::node_t const *node, typename tree_type_::node_t const *parent,
                                     typename tree_type_::comparator_t const &comparator) {
    if (!node) return {};
    expect_equal(node->parent, parent, "every node points back at its parent");

    subtree_shape_t const left = check_subtree<tree_type_>(node->left, node, comparator);
    subtree_shape_t const right = check_subtree<tree_type_>(node->right, node, comparator);
    std::ptrdiff_t const height = 1 + (left.height > right.height ? left.height : right.height);
    std::ptrdiff_t const balance = left.height - right.height;
    expect_equal(node->height, height, "the stored height matches the walked one");
    expect_true(balance >= -1, "AVL balance never exceeds one");
    expect_true(balance <= 1, "AVL balance never exceeds one");

    if (node->left)
        expect_true(comparator(mapping_key_or_itself(node->left->payload), mapping_key_or_itself(node->payload)),
                    "the left child sorts before its parent");
    if (node->right)
        expect_true(comparator(mapping_key_or_itself(node->payload), mapping_key_or_itself(node->right->payload)),
                    "the right child sorts after its parent");

    return {left.count + right.count + 1, height};
}

/** Walks the whole tree, checking its shape and that the node count agrees with @c size(). */
template <typename tree_type_>
static void check_invariants(tree_type_ const &tree) {
    auto const *root = tree.root();
    expect_true(!root || !root->parent, "the root has no parent");
    typename tree_type_::comparator_t const comparator = tree.key_comp();
    subtree_shape_t const shape = check_subtree<tree_type_>(root, nullptr, comparator);
    expect_equal(shape.count, tree.size());
}

/** A copied tree must carry its own parent links, or the first @c ++it walks into garbage. */
template <typename tree_type_>
static void check_copy_preserves_parent_links() {
    using member_t = typename tree_type_::value_type;
    tree_type_ tree;
    for (trivial_id_t identifier : {5u, 2u, 8u, 1u, 3u, 7u, 9u})
        expect_true(!tree.upsert(trivial_id_to_member<member_t>(identifier)).failed());

    auto [copied, status] = tree.copy();
    if (!expect_equal(status, success_k)) return;
    check_invariants(copied);
    expect_equal(std::distance(copied.begin(), copied.end()), tree.size());
}

/** The perfectly balanced tree built from a sorted range must be linked both ways. */
template <typename tree_type_>
static void check_bulk_sorted_insert_links_parents() {
    using member_t = typename tree_type_::value_type;
    for (std::size_t count = 1; count <= 64; ++count) {
        tree_type_ tree;
        expect_true(!tree.upsert(trivial_id_to_member<member_t>(1000)).failed());

        std::vector<member_t> sorted;
        for (std::size_t index = 0; index < count; ++index) sorted.push_back(trivial_id_to_member<member_t>(index));

        expect_equal(tree.insert_if_missing(std::make_move_iterator(sorted.begin()),
                                            std::make_move_iterator(sorted.end()), assume_sorted_t {}),
                     success_k);
        check_invariants(tree);
        expect_equal(tree.size(), count + 1);
        expect_equal(std::distance(tree.begin(), tree.end()), tree.size());
    }
}

/** Erasing a two-child root promotes its successor, which nobody above it can rebalance. */
template <typename tree_type_>
static void check_erase_root_rebalances_promoted_node() {
    using member_t = typename tree_type_::value_type;
    tree_type_ tree;
    for (trivial_id_t identifier : {10u, 5u, 15u, 3u, 7u})
        expect_true(!tree.upsert(trivial_id_to_member<member_t>(identifier)).failed());
    check_invariants(tree);

    expect_true(tree.erase(trivial_id_to_key<member_t>(10)));
    check_invariants(tree);
    expect_equal(tree.size(), 4);
}

/** The callback range is [lower, upper), matching what @c erase_range removes. */
template <typename tree_type_>
static void check_range_excludes_upper_bound() {
    using member_t = typename tree_type_::value_type;
    tree_type_ tree;
    for (trivial_id_t identifier = 1; identifier <= 5; ++identifier)
        expect_true(!tree.upsert(trivial_id_to_member<member_t>(identifier)).failed());

    std::vector<trivial_id_t> seen;
    expect_equal(
        tree.range(trivial_id_to_key<member_t>(2), trivial_id_to_key<member_t>(4),
                   [&](auto const &member) noexcept { seen.push_back(mapping_key_or_itself(member).unique_id); }),
        success_k);

    if (!expect_equal(seen.size(), 2)) return;
    expect_equal(seen[0], 2);
    expect_equal(seen[1], 3);
}

/** Splitting rebuilds two trees whose sides can differ in height by far more than one. */
template <typename tree_type_>
static void check_split_and_erase_range_stay_balanced() {
    using member_t = typename tree_type_::value_type;
    for (std::size_t total = 1; total <= 128; ++total) {
        tree_type_ tree;
        for (std::size_t index = 0; index < total; ++index)
            expect_true(!tree.upsert(trivial_id_to_member<member_t>(index * 2)).failed());

        auto halves = tree.split(trivial_id_to_key<member_t>(total));
        check_invariants(halves.left);
        check_invariants(halves.right);
        expect_equal(halves.left.size() + halves.right.size(), total);

        tree_type_ rebuilt;
        for (std::size_t index = 0; index < total; ++index)
            expect_true(!rebuilt.upsert(trivial_id_to_member<member_t>(index)).failed());
        rebuilt.erase_range(trivial_id_to_key<member_t>(total / 4), trivial_id_to_key<member_t>(total / 2));
        check_invariants(rebuilt);
        expect_equal(std::distance(rebuilt.begin(), rebuilt.end()), rebuilt.size());
    }
}

/** The split-based union of two interleaved trees must come out balanced and fully linked. */
template <typename tree_type_>
static void check_merge_unique_stays_balanced() {
    using member_t = typename tree_type_::value_type;
    for (std::size_t total = 1; total <= 128; ++total) {
        tree_type_ evens, odds;
        for (std::size_t index = 0; index < total; ++index) {
            expect_true(!evens.upsert(trivial_id_to_member<member_t>(index * 2)).failed());
            expect_true(!odds.upsert(trivial_id_to_member<member_t>(index * 2 + 1)).failed());
        }

        evens.merge(odds, assume_unique_t {});
        check_invariants(evens);
        expect_equal(evens.size(), total * 2);
        expect_equal(std::distance(evens.begin(), evens.end()), evens.size());

        // Fully ordered inputs take the join fast path instead of the split-based one.
        tree_type_ low, high;
        for (std::size_t index = 0; index < total; ++index) {
            expect_true(!low.upsert(trivial_id_to_member<member_t>(index)).failed());
            expect_true(!high.upsert(trivial_id_to_member<member_t>(index + total)).failed());
        }
        low.merge(high, assume_unique_t {});
        check_invariants(low);
        expect_equal(low.size(), total * 2);
    }
}

/** Randomized insert and erase sequence, re-checking every invariant after each mutation. */
template <typename tree_type_>
static void check_random_mutations_preserve_invariants() {
    using member_t = typename tree_type_::value_type;
    split_mix_generator_t generator {case_stream_key()};
    tree_type_ tree;
    std::set<trivial_id_t> oracle;

    for (std::size_t step = 0; step < 4000; ++step) {
        trivial_id_t const identifier = generator() % 300;
        if (generator() % 2) {
            expect_true(!tree.upsert(trivial_id_to_member<member_t>(identifier)).failed());
            oracle.insert(identifier);
        }
        else { expect_equal(tree.erase(trivial_id_to_key<member_t>(identifier)), oracle.erase(identifier) == 1); }
        check_invariants(tree);
        expect_equal(tree.size(), oracle.size());
    }

    auto expected = oracle.begin();
    for (auto const &member : tree) {
        expect_equal(mapping_key_or_itself(member).unique_id, *expected);
        ++expected;
    }
    expect_true(expected == oracle.end());
}

/** Extracts random positions against @c std::set, re-checking every node's balance after each. */
template <typename tree_type_>
static void check_extract_by_position_stays_balanced() {
    using member_t = typename tree_type_::value_type;
    split_mix_generator_t generator {case_stream_key()};
    tree_type_ tree;
    std::set<trivial_id_t> oracle;

    for (std::size_t step = 0; step < 4000; ++step) {
        trivial_id_t const identifier = generator() % 300;
        typename tree_type_::iterator const position = tree.find(trivial_id_to_key<member_t>(identifier));
        expect_equal(position != tree.end(), oracle.count(identifier) == 1);
        if (position == tree.end()) {
            expect_true(!tree.upsert(trivial_id_to_member<member_t>(identifier)).failed());
            oracle.insert(identifier);
        }
        else {
            typename tree_type_::extract_result_t const extracted = tree.extract(position);
            if (expect_true(bool(extracted), "a dereferenceable position always names a node"))
                expect_equal(mapping_key_or_itself(extracted.node_ptr_->payload).unique_id, identifier);
            oracle.erase(identifier);
        }
        check_invariants(tree);
        expect_equal(tree.size(), oracle.size());
    }

    auto expected = oracle.begin();
    for (auto const &member : tree) {
        expect_equal(mapping_key_or_itself(member).unique_id, *expected);
        ++expected;
    }
    expect_true(expected == oracle.end());
}

/** A splice relinks the nodes themselves, so every entry keeps its address, and a key the
 *  destination already holds stays behind in the tree it came from. */
template <typename tree_type_>
static void check_splice_keeps_addresses_and_leaves_ties() {
    using member_t = typename tree_type_::value_type;
    using node_t = typename tree_type_::node_t;
    for (std::size_t total = 1; total <= 96; ++total) {
        tree_type_ destination, source;
        std::size_t tied = 0;
        for (std::size_t index = 0; index < total; ++index) {
            expect_true(!source.upsert(trivial_id_to_member<member_t>(index)).failed());
            if (index % 3 != 0) continue;
            expect_true(!destination.upsert(trivial_id_to_member<member_t>(index)).failed());
            ++tied;
        }
        std::vector<member_t const *> addresses;
        for (member_t const &member : source) addresses.push_back(&member);

        std::size_t linked = 0;
        destination.splice(source, [&](node_t *node) noexcept {
            expect_equal(&*destination.iterator_to(node), &node->payload);
            ++linked;
        });
        check_invariants(destination);
        check_invariants(source);
        expect_equal(linked, total - tied);
        expect_equal(destination.size(), total);
        expect_equal(source.size(), tied, "a key already there stays in the tree it came from");

        for (member_t const &member : destination) {
            std::size_t const index = mapping_key_or_itself(member).unique_id;
            if (index % 3 != 0) expect_equal(&member, addresses[index], "a spliced entry never moves");
        }
        for (member_t const &member : source) {
            std::size_t const index = mapping_key_or_itself(member).unique_id;
            expect_equal(index % 3, 0u);
            expect_equal(&member, addresses[index], "and neither does one left behind");
        }
    }
}

/** Post-decrement must exist for the type to model @c std::bidirectional_iterator. */
template <typename tree_type_>
static void check_iterator_post_decrement() {
    static_assert(std::bidirectional_iterator<typename tree_type_::iterator>, "iterator must be bidirectional");
    static_assert(std::bidirectional_iterator<typename tree_type_::const_iterator>,
                  "const_iterator must be bidirectional");

    using member_t = typename tree_type_::value_type;
    tree_type_ tree;
    for (trivial_id_t identifier = 1; identifier <= 3; ++identifier)
        expect_true(!tree.upsert(trivial_id_to_member<member_t>(identifier)).failed());

    auto position = tree.end();
    auto const previous = position--;
    expect_true(previous == tree.end());
    expect_equal(mapping_key_or_itself(*position).unique_id, 3);
    position--;
    expect_equal(mapping_key_or_itself(*position).unique_id, 2);
}

/** The node-level @c equal_range reports the half-open bounds around a key. */
template <typename tree_type_>
static void check_node_equal_range() {
    using member_t = typename tree_type_::value_type;
    using node_t = typename tree_type_::node_t;
    tree_type_ tree;
    for (trivial_id_t identifier = 1; identifier <= 5; ++identifier)
        expect_true(!tree.upsert(trivial_id_to_member<member_t>(identifier)).failed());

    typename tree_type_::comparator_t const comparator = tree.key_comp();
    auto const present = node_t::equal_range(tree.root(), trivial_id_to_key<member_t>(3), comparator);
    expect_true(present.lower_bound && mapping_key_or_itself(present.lower_bound->payload).unique_id == 3);
    expect_true(present.upper_bound && mapping_key_or_itself(present.upper_bound->payload).unique_id == 4);
    expect_true(present.lowest_common_ancestor &&
                mapping_key_or_itself(present.lowest_common_ancestor->payload).unique_id == 3);

    auto const absent = node_t::equal_range(tree.root(), trivial_id_to_key<member_t>(9), comparator);
    expect_true(!absent.lower_bound && !absent.upper_bound);
}

/** A bulk upsert pays for its nodes while staging them, and the merge that follows cannot fail. */
static void test_bulk_upsert_pays_only_while_staging() {
    using budget_set_t = avl_set<trivial_key_t, std::less<trivial_key_t>, stateful_allocator<trivial_key_t>>;

    allocation_ledger_t ledger;
    ledger.allow(64);
    {
        budget_set_t tree {typename budget_set_t::allocator_t(ledger)};
        for (trivial_id_t identifier : {1u, 2u}) expect_true(!tree.upsert(trivial_key_t(identifier)).failed());

        // Too little for the three-node temporary, so the whole batch fails and this tree is untouched.
        std::vector<trivial_key_t> const members = {trivial_key_t(1), trivial_key_t(2), trivial_key_t(99)};
        ledger.allow(2);
        expect_equal(tree.upsert(members.begin(), members.end()), status_t::out_of_memory_heap_k);
        expect_equal(tree.contains(trivial_key_t(99)), false);
        expect_equal(tree.size(), 2u);

        // Exactly enough for the temporary, and the merge relinks its nodes rather than asking for more.
        std::size_t const granted_before_merge = ledger.allocations;
        std::size_t const refused_before_merge = ledger.refusals;
        ledger.allow(3);
        expect_equal(tree.upsert(members.begin(), members.end()), success_k);
        expect_equal(tree.contains(trivial_key_t(99)), true);
        expect_equal(ledger.allocations - granted_before_merge, std::size_t {3});
        expect_equal(ledger.refusals - refused_before_merge, std::size_t {0});
        check_invariants(tree);
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
    using budget_set_t = avl_set<trivial_key_t, std::less<trivial_key_t>, stateful_allocator<trivial_key_t>>;
    using placement_t = typename budget_set_t::node_t::node_placement_t;
    allocation_ledger_t ledger;
    ledger.allow(1);
    {
        budget_set_t tree {typename budget_set_t::allocator_t(ledger)};

        auto const made = tree.upsert(trivial_key_t(1));
        expect_equal(made.placement, placement_t::made_k, "a fresh key must report a node of its own");
        expect_true(!made.failed());
        expect_true((made.node != nullptr));

        auto const matched = tree.upsert(trivial_key_t(1));
        expect_equal(matched.placement, placement_t::matched_k, "a key already there must report a match");
        expect_true(succeeded(matched), "an overwrite is not a failure");
        expect_true((matched.node != nullptr));

        // The budget is spent, so the second key has no node to live in - the outcome that shares a
        // null match with nothing else.
        auto const refused = tree.upsert(trivial_key_t(2));
        expect_equal(refused.placement, placement_t::refused_k, "a refused allocation must say so");
        expect_true(failed(refused));
        expect_true((refused.node == nullptr));
        expect_equal(tree.size(), 1u);
    }
    ledger.check_balanced();
}

/** Move-only entries must survive the assignment operator of an upsert result. */
static void test_move_only_upsert_assignment() {
    struct move_only_key_t {
        trivial_id_t unique_id = 0;
        std::unique_ptr<int> payload;

        move_only_key_t() noexcept = default;
        explicit move_only_key_t(trivial_id_t identifier) noexcept : unique_id(identifier) {}
        move_only_key_t(move_only_key_t &&) noexcept = default;
        move_only_key_t &operator=(move_only_key_t &&) noexcept = default;

        bool operator<(move_only_key_t const &other) const noexcept { return unique_id < other.unique_id; }
    };

    avl_set<move_only_key_t> tree;
    auto result = tree.upsert(move_only_key_t(5));
    expect_true(static_cast<bool>(result));
    result = move_only_key_t(5); // ! Instantiates `upserted_node_t::operator=`
    expect_equal(tree.size(), 1);
}

static void test_copy_preserves_parent_links() {
    check_copy_preserves_parent_links<trivial_set_t>();
    check_copy_preserves_parent_links<tracking_set_t>();
    check_copy_preserves_parent_links<trivial_map_t>();
}

static void test_bulk_sorted_insert_links_parents() {
    check_bulk_sorted_insert_links_parents<trivial_set_t>();
    check_bulk_sorted_insert_links_parents<tracking_set_t>();
    check_bulk_sorted_insert_links_parents<trivial_map_t>();
}

static void test_erase_root_rebalances_promoted_node() {
    check_erase_root_rebalances_promoted_node<trivial_set_t>();
    check_erase_root_rebalances_promoted_node<tracking_set_t>();
    check_erase_root_rebalances_promoted_node<trivial_map_t>();
}

static void test_range_excludes_upper_bound() {
    check_range_excludes_upper_bound<trivial_set_t>();
    check_range_excludes_upper_bound<tracking_set_t>();
    check_range_excludes_upper_bound<trivial_map_t>();
}

static void test_split_and_erase_range_stay_balanced() {
    check_split_and_erase_range_stay_balanced<trivial_set_t>();
    check_split_and_erase_range_stay_balanced<trivial_map_t>();
}

static void test_merge_unique_stays_balanced() {
    check_merge_unique_stays_balanced<trivial_set_t>();
    check_merge_unique_stays_balanced<trivial_map_t>();
}

static void test_random_mutations_preserve_invariants() {
    check_random_mutations_preserve_invariants<trivial_set_t>();
    check_random_mutations_preserve_invariants<tracking_set_t>();
    check_random_mutations_preserve_invariants<trivial_map_t>();
}

static void test_extract_by_position_stays_balanced() {
    check_extract_by_position_stays_balanced<trivial_set_t>();
    check_extract_by_position_stays_balanced<tracking_set_t>();
    check_extract_by_position_stays_balanced<trivial_map_t>();
}

static void test_splice_keeps_addresses_and_leaves_ties() {
    check_splice_keeps_addresses_and_leaves_ties<trivial_set_t>();
    check_splice_keeps_addresses_and_leaves_ties<trivial_map_t>();
}

static void test_iterator_post_decrement() {
    check_iterator_post_decrement<trivial_set_t>();
    check_iterator_post_decrement<trivial_map_t>();
}

static void test_node_equal_range() {
    check_node_equal_range<trivial_set_t>();
    check_node_equal_range<trivial_map_t>();
}

/** Erasing via a @c const_iterator must compile, unlink the node, and leave the tree balanced. */
static void test_erase_const_iterator() {
    trivial_set_t tree;
    for (trivial_id_t identifier : {1u, 2u, 3u}) expect_true(!tree.upsert(trivial_key_t(identifier)).failed());

    trivial_set_t::const_iterator const position = tree.find(trivial_key_t(2));
    expect_true(position != tree.end());
    auto const erased = tree.erase(position);
    expect_equal(erased.status, success_k);
    expect_equal(tree.contains(trivial_key_t(2)), false);
    expect_equal(tree.size(), 2u);
    check_invariants(tree);
}

#pragma endregion Structural Invariants

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
    check_container_balances_counted_keys<avl_map<counted_key_t, int>>();
}

static void test_rollback_balances_counted_keys() {
    check_rollback_balances_counted_keys<monotonic_avl_map<counted_key_t, int>>();
}

using overaligned_set_t = monotonic_avl_set<overaligned_key_t>;

static void test_container_honours_over_alignment() { check_container_honours_over_alignment<overaligned_set_t>(); }

#pragma endregion Fixture Coverage

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

using traced_set_t = avl_set<traced_key_t, traced_less_t, stateful_allocator<void>>;

/** A key already there costs no node, so it must report presence rather than exhaustion. */
static void test_allocation_failure_insert_probes_before_allocating() {
    allocation_ledger_t ledger;
    ledger.allow(1);
    {
        traced_set_t tree(traced_less_t {}, stateful_allocator<traced_set_t::node_t>(ledger));
        expect_true(bool(tree.insert(traced_key_t(1))));

        // The budget is spent, and a duplicate must not go looking for memory it does not need.
        ledger.refuse_everything();
        traced_key_t duplicate(1);
        auto const matched = tree.insert(std::move(duplicate));
        expect_equal(matched.status(), status_t::key_already_exists_k, "presence is not exhaustion");
        expect_true(!duplicate.moved_from, "a refused entry stays with its caller");
        expect_equal(ledger.refusals, std::size_t {0});
        expect_equal(tree.size(), 1u);
    }
    ledger.check_balanced();
}

/** Every range modifier refused at every point it asks for memory or copies an element leaves the
 *  tree as it was. */
void test_allocation_failure_batch_is_all_or_nothing() {
    using ledger_set_t = avl_set<trivial_key_t, less_t, stateful_allocator_t>;
    check_every_offered_surface<ledger_set_t>(
        [](allocation_ledger_t &ledger) noexcept { return ledger_set_t(less_t {}, stateful_allocator_t(1, ledger)); });
    check_batch_refuses_copies_whole<avl_set<budgeted_key_t, std::less<void>, std::allocator<budgeted_key_t>>>();
}

#pragma endregion Allocation Failure

#pragma region Failure Policy

/** Every cause this store can produce, the refused heap included, reports a status of its own. */
static void test_failure_policy_budgeted_distinct_causes() {
    check_budgeted_store_maps_causes_to_distinct_statuses<transactional_tracking_map_t>();
}

#pragma endregion Failure Policy

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

    run_test_case("test_copy_preserves_parent_links", test_copy_preserves_parent_links);
    run_test_case("test_bulk_sorted_insert_links_parents", test_bulk_sorted_insert_links_parents);
    run_test_case("test_erase_root_rebalances_promoted_node", test_erase_root_rebalances_promoted_node);
    run_test_case("test_range_excludes_upper_bound", test_range_excludes_upper_bound);
    run_test_case("test_split_and_erase_range_stay_balanced", test_split_and_erase_range_stay_balanced);
    run_test_case("test_merge_unique_stays_balanced", test_merge_unique_stays_balanced);
    run_test_case("test_random_mutations_preserve_invariants", test_random_mutations_preserve_invariants);
    run_test_case("test_extract_by_position_stays_balanced", test_extract_by_position_stays_balanced);
    run_test_case("test_splice_keeps_addresses_and_leaves_ties", test_splice_keeps_addresses_and_leaves_ties);
    run_test_case("test_iterator_post_decrement", test_iterator_post_decrement);
    run_test_case("test_node_equal_range", test_node_equal_range);
    run_test_case("test_bulk_upsert_pays_only_while_staging", test_bulk_upsert_pays_only_while_staging);
    run_test_case("test_move_only_upsert_assignment", test_move_only_upsert_assignment);
    run_test_case("test_upsert_reports_placement", test_upsert_reports_placement);

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

    run_test_case("test_erase_const_iterator", test_erase_const_iterator);
    run_test_case("test_allocation_failure_insert_probes_before_allocating",
                  test_allocation_failure_insert_probes_before_allocating);

    run_test_case("test_container_honours_over_alignment", test_container_honours_over_alignment);

    run_test_case("test_failure_policy_budgeted_distinct_causes", test_failure_policy_budgeted_distinct_causes);

    run_test_case("test_allocation_failure_batch_is_all_or_nothing", test_allocation_failure_batch_is_all_or_nothing);

    return finish_test_run();
}
