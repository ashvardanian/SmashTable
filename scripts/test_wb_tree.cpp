/**
 *  @brief Test instantiations for weight-balanced tree containers. Covers basic_wb_tree (non-transactional) and
 *      monotonic_store<basic_wb_tree> (transactional). The weight-balanced tree also carries order
 *      statistics - @c rank and @c select.
 *  @author Ash Vardanian
 *  @file scripts/test_wb_tree.cpp
 *  @date August 16, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define ST_STRICT_CALLBACK_CHECKS_ 1

#include <algorithm> // `std::equal`
#include <map>       // `std::map`
#include <random>    // `std::mt19937`
#include <set>       // `std::set`
#include <vector>    // `std::vector`

#include <smashtable/basic_wb_tree.hpp>
#include <smashtable/reference_store.hpp>
#include <smashtable/monotonic_store.hpp>

#include "test.hpp"
#include "test_basic.hpp"
#include "test_commit_stamp.hpp"
#include "test_consistency.hpp"
#include "test_fixture_coverage.hpp"
#include "test_monotonic_store_defects.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

namespace {

#pragma region Type Aliases

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Transaction: ✗
 *  Tests: Baseline non-transparent comparator path
 */
using trivial_set_t = wb_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial | Memory: Tracked | Transaction: ✗
 *  Tests: Resource accounting, allocation failure injection
 */
using tracking_set_t = wb_set<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Trivial | Memory: Stack | Transaction: ✗
 *  Tests: Identifier extraction, composite_key_compare_t::value_type lookups
 */
using composite_set_t = wb_set<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() → expected<T> | Memory: Heap | Transaction: ✗
 *  Tests: OOM during .copy(), string_view lookups without materialization
 */
using heavy_set_t = wb_set<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial (key & value) | Memory: Stack | Transaction: ✗
 *  Value: int | Tests: Baseline map operations, non-transparent path
 */
using trivial_map_t = wb_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial (key & value) | Memory: Tracked | Transaction: ✗
 *  Value: int | Tests: Map resource accounting, POCCA/POCMA on key-value pairs
 */
using tracking_map_t = wb_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Key trivial, value .copy() | Memory: Heap (value) | Transaction: ✗
 *  Value: guarded_payload_t | Tests: Mixed trivial/non-trivial, value OOM scenarios
 */
using composite_map_t = wb_map<composite_key_t, guarded_payload_t, composite_key_compare_t,
                               std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() on key & value | Memory: Heap (both) | Transaction: ✗
 *  Value: guarded_payload_t | Tests: Dual-heap OOM, worst-case complexity
 */
using heavy_map_t =
    wb_map<heavy_key_t, guarded_payload_t, std::less<void>, std::allocator<mapping<heavy_key_t, guarded_payload_t>>>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Transaction: ✓
 *  Tests: Baseline transactional correctness, isolation levels
 */
using transactional_trivial_set_t =
    monotonic_wb_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial | Memory: Tracked | Transaction: ✓
 *  Tests: Transaction resource accounting, OOM during stage/commit
 */
using transactional_tracking_set_t = monotonic_wb_set<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Trivial | Memory: Stack | Transaction: ✓
 *  Tests: Heterogeneous watch/find in transactions
 */
using transactional_composite_set_t =
    monotonic_wb_set<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() → expected<T> | Memory: Heap | Transaction: ✓
 *  Tests: Watch copy OOM, transaction rollback with heap types
 */
using transactional_heavy_set_t = monotonic_wb_set<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial (key & value) | Memory: Stack | Transaction: ✓
 *  Value: int | Tests: Transactional map operations, value overwrites
 */
using transactional_trivial_map_t =
    monotonic_wb_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial (key & value) | Memory: Tracked | Transaction: ✓
 *  Value: int | Tests: Transaction allocation patterns, map POCCA/POCMA
 */
using transactional_tracking_map_t = monotonic_wb_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Key trivial, value .copy() | Memory: Heap (value) | Transaction: ✓
 *  Value: guarded_payload_t | Tests: Transaction rollback with non-trivial values
 */
using transactional_composite_map_t = monotonic_wb_map<composite_key_t, guarded_payload_t, composite_key_compare_t,
                                                       std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() on key & value | Memory: Heap (both) | Transaction: ✓
 *  Value: guarded_payload_t | Tests: Worst-case transactional complexity, dual-heap rollback
 */
using transactional_heavy_map_t = monotonic_wb_map<heavy_key_t, guarded_payload_t, std::less<void>,
                                                   std::allocator<mapping<heavy_key_t, guarded_payload_t>>>;

#pragma endregion Type Aliases

#pragma region Basic Operations Tests

/** @brief Tests operations on empty container don't crash */
static void basic_ops_empty_container_operations() {
    test_empty_container_operations<trivial_set_t>();
    test_empty_container_operations<tracking_set_t>();
    test_empty_container_operations<composite_set_t>();
    test_empty_container_operations<heavy_set_t>();
    test_empty_container_operations<trivial_map_t>();
    test_empty_container_operations<tracking_map_t>();
    test_empty_container_operations<composite_map_t>();
    test_empty_container_operations<heavy_map_t>();
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
    test_single_element_operations<trivial_set_t>();
    test_single_element_operations<tracking_set_t>();
    test_single_element_operations<composite_set_t>();
    test_single_element_operations<heavy_set_t>();
    test_single_element_operations<trivial_map_t>();
    test_single_element_operations<tracking_map_t>();
    test_single_element_operations<composite_map_t>();
    test_single_element_operations<heavy_map_t>();
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
    test_basic_insertion_patterns<trivial_set_t>();
    test_basic_insertion_patterns<tracking_set_t>();
    test_basic_insertion_patterns<composite_set_t>();
    test_basic_insertion_patterns<heavy_set_t>();
    test_basic_insertion_patterns<trivial_map_t>();
    test_basic_insertion_patterns<tracking_map_t>();
    test_basic_insertion_patterns<composite_map_t>();
    test_basic_insertion_patterns<heavy_map_t>();
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
    test_bulk_insertion_from_iterators<trivial_set_t>();
    test_bulk_insertion_from_iterators<tracking_set_t>();
    test_bulk_insertion_from_iterators<composite_set_t>();
    test_bulk_insertion_from_iterators<heavy_set_t>();
    test_bulk_insertion_from_iterators<trivial_map_t>();
    test_bulk_insertion_from_iterators<tracking_map_t>();
    test_bulk_insertion_from_iterators<composite_map_t>();
    test_bulk_insertion_from_iterators<heavy_map_t>();
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
    // ! Only applies to maps with integral mapped values
    test_bulk_upsert_with_duplicates<trivial_map_t>();
    test_bulk_upsert_with_duplicates<tracking_map_t>();
    test_bulk_upsert_with_duplicates<transactional_trivial_map_t>();
    test_bulk_upsert_with_duplicates<transactional_tracking_map_t>();
}

/** @brief Tests range queries on committed HEAD state */
static void basic_ops_range_query_head_state() {
    // ! Only applies to "non-heavy" keys to simplify the test implementation
    test_range_query_head_state<trivial_set_t>();
    test_range_query_head_state<tracking_set_t>();
    test_range_query_head_state<composite_set_t>();
    test_range_query_head_state<trivial_map_t>();
    test_range_query_head_state<tracking_map_t>();
    test_range_query_head_state<composite_map_t>();
    test_range_query_head_state<transactional_trivial_set_t>();
    test_range_query_head_state<transactional_tracking_set_t>();
    test_range_query_head_state<transactional_composite_set_t>();
    test_range_query_head_state<transactional_trivial_map_t>();
    test_range_query_head_state<transactional_tracking_map_t>();
    test_range_query_head_state<transactional_composite_map_t>();
}

/** @brief Tests erase_range on committed HEAD state */
static void basic_ops_erase_range_head_state() {
    // ! Only applies to "non-heavy" keys to simplify the test implementation
    test_erase_range_head_state<trivial_set_t>();
    test_erase_range_head_state<tracking_set_t>();
    test_erase_range_head_state<composite_set_t>();
    test_erase_range_head_state<heavy_set_t>();
    test_erase_range_head_state<trivial_map_t>();
    test_erase_range_head_state<tracking_map_t>();
    test_erase_range_head_state<composite_map_t>();
    test_erase_range_head_state<heavy_map_t>();
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
    test_heterogeneous_composite_find<composite_set_t>();
    test_heterogeneous_heavy_string_view_find<heavy_set_t>();
    test_heterogeneous_composite_find<composite_map_t>();
    test_heterogeneous_heavy_string_view_find<heavy_map_t>();
    test_heterogeneous_composite_find<transactional_composite_set_t>();
    test_heterogeneous_heavy_string_view_find<transactional_heavy_set_t>();
    test_heterogeneous_composite_find<transactional_composite_map_t>();
    test_heterogeneous_heavy_string_view_find<transactional_heavy_map_t>();
}

#pragma endregion Basic Operations Tests

#pragma region Consistency and Transaction Tests for Sets

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

static void transactional_defects_erase_range_spares_staged_versions() {
    test_erase_range_spares_staged_versions<transactional_trivial_map_t>();
    test_erase_range_spares_staged_versions<transactional_composite_map_t>();
    test_erase_range_spares_staged_versions<transactional_heavy_map_t>();
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

static void transactional_defects_committed_erase_hidden_from_ordered_reads() {
    test_committed_erase_hidden_from_ordered_reads<transactional_trivial_map_t>();
    test_committed_erase_hidden_from_ordered_reads<transactional_composite_map_t>();
    test_committed_erase_hidden_from_ordered_reads<transactional_heavy_map_t>();
}

static void transactional_defects_committed_erase_hidden_from_update_range() {
    test_committed_erase_hidden_from_update_range<transactional_trivial_map_t>();
    test_committed_erase_hidden_from_update_range<transactional_composite_map_t>();
    test_committed_erase_hidden_from_update_range<transactional_heavy_map_t>();
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

static void transactional_defects_windowed_vacuum_reclaims_one_slice() {
    test_windowed_vacuum_reclaims_one_slice<transactional_trivial_map_t>();
    test_windowed_vacuum_reclaims_one_slice<transactional_composite_map_t>();
    test_windowed_vacuum_reclaims_one_slice<transactional_heavy_map_t>();
}

#pragma endregion Transactional Store Defects

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

#pragma endregion Consistency and Transaction Tests for Sets

static void transactional_consistency_stateful_comparator_is_consulted() {
    test_stateful_comparator_is_consulted<transactional_tracking_set_t>();
}

#pragma region Weight Balance Invariants

/**
 *  A bare @c int set, because the balance oracle below reaches for @c root() and walks raw nodes.
 *  The keys carry no behaviour of their own, so a failure is always the tree's.
 */
using ordered_set_t = wb_set<int, std::less<int>, std::allocator<int>>;
using ordered_node_t = ordered_set_t::node_t;

/** @brief What one subtree contributes, so the oracle can compare both counts at once. */
struct subtree_counts_t {
    /** @brief Number of nodes in the subtree. */
    std::size_t size = 0;
    /** @brief Number of nodes the augmentation policy counts, zero for an unaugmented tree. */
    std::size_t augmented_size = 0;
};

/**
 *  @brief Recomputes one subtree's counts while checking every structural invariant.
 *    Δ=3 over @c size+1 weights, @c size @c = @c 1 @c + @c size(left) @c + @c size(right), and - where the
 *    tree carries an augmentation - the same recurrence over the policy's per-entry count.
 */
template <typename node_type_>
static subtree_counts_t verify_invariants(node_type_ *node) noexcept {
    if (!node) return {};
    subtree_counts_t const left = verify_invariants(node->left);
    subtree_counts_t const right = verify_invariants(node->right);
    st_verify_eq_(node->size, 1 + left.size + right.size);

    std::size_t const left_weight = left.size + 1, right_weight = right.size + 1;
    st_verify_le_(left_weight, node_type_::delta_k * right_weight);
    st_verify_le_(right_weight, node_type_::delta_k * left_weight);

    subtree_counts_t counts;
    counts.size = node->size;
    if constexpr (node_type_::is_augmented_k) {
        counts.augmented_size = node_type_::get_own_augmented_count(node) + left.augmented_size + right.augmented_size;
        st_verify_eq_(node->augmented_size, counts.augmented_size);
    }
    return counts;
}

/** @brief Checks the tree against a @c std::set oracle: invariants, element count, and in-order contents. */
static void verify_against_oracle(ordered_set_t &tree, std::set<int> const &oracle) noexcept {
    st_verify_eq_(verify_invariants(tree.root()).size, oracle.size());
    st_verify_eq_(tree.size(), oracle.size());

    std::vector<int> walked;
    st_verify_(tree.for_each([&](int const &element) noexcept { walked.push_back(element); }));
    st_verify_eq_(walked.size(), oracle.size());
    st_verify_(std::equal(walked.begin(), walked.end(), oracle.begin()));
}

/** @brief Sequential keys must not degenerate into a spine - every entry point has to rebalance. */
static void weight_balance_insert_rebalances() {
    for (std::size_t count : {std::size_t(1), std::size_t(2), std::size_t(1000)}) {
        ordered_set_t ascending, descending;
        std::set<int> oracle;
        for (std::size_t index = 0; index < count; ++index) {
            [[maybe_unused]] auto const added = ascending.insert(int(index));
            [[maybe_unused]] auto const subtracted = descending.insert(int(count - index));
            oracle.insert(int(index));
        }
        verify_against_oracle(ascending, oracle);
        [[maybe_unused]] subtree_counts_t const counted = verify_invariants(descending.root());
        st_verify_eq_(descending.size(), count);
    }

    // A rejected duplicate leaves the tree untouched and reports the incumbent.
    using placement_t = ordered_set_t::node_t::node_placement_t;
    ordered_set_t tree;
    [[maybe_unused]] auto const first = tree.insert(7);
    auto const duplicate = tree.insert(7);
    st_verify_(duplicate.placement == placement_t::matched_k);
    st_verify_ne_(duplicate.node, nullptr);
    st_verify_eq_(duplicate.node->fruit, 7);
    st_verify_eq_(tree.size(), 1u);
}

/** @brief Promoting the right subtree's minimum over a two-child erase must rebalance the promotion. */
static void weight_balance_two_child_erase_rebalances() {
    ordered_set_t tree;
    std::set<int> oracle;
    for (int element : {10, 5, 15, 3, 7}) {
        [[maybe_unused]] auto const added = tree.insert(int(element));
        oracle.insert(element);
    }
    verify_against_oracle(tree, oracle);

    st_verify_(tree.erase(10));
    oracle.erase(10);
    verify_against_oracle(tree, oracle);
}

/** @brief Both halves of a split own their element counts, their allocator, and a balanced shape. */
static void weight_balance_split_and_join_track_size() {
    std::mt19937 generator(1337);
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
        st_verify_eq_(halves.left.empty(), below.empty());
        verify_against_oracle(halves.left, below);
        verify_against_oracle(halves.right, above);

        halves.left.join(halves.right);
        verify_against_oracle(halves.left, oracle);
        st_verify_(halves.right.empty());
    }
}

/** @brief Half-open @c [lower, @c upper) range walks, visited in sorted order. */
static void weight_balance_range_is_half_open() {
    ordered_set_t tree;
    std::set<int> oracle;
    for (int element = 0; element < 10; ++element) {
        [[maybe_unused]] auto const added = tree.insert(int(element));
        oracle.insert(element);
    }

    std::vector<int> visited;
    st_verify_(tree.range(2, 5, [&](int const &element) noexcept { visited.push_back(element); }));
    std::vector<int> const expected {2, 3, 4};
    st_verify_(visited == expected);

    // An empty window yields nothing, rather than the single element at its edge.
    visited.clear();
    st_verify_(tree.range(4, 4, [&](int const &element) noexcept { visited.push_back(element); }));
    st_verify_(visited.empty());
}

/** @brief Erasing an iterator range compiles, reports success, and leaves a balanced tree. */
static void weight_balance_erase_iterator_range() {
    ordered_set_t tree;
    std::set<int> oracle;
    for (int element = 0; element < 200; ++element) {
        [[maybe_unused]] auto const added = tree.insert(int(element));
        oracle.insert(element);
    }

    auto const result = tree.erase(tree.lower_bound(40), tree.lower_bound(160));
    oracle.erase(oracle.lower_bound(40), oracle.lower_bound(160));
    st_verify_(result.status);
    verify_against_oracle(tree, oracle);

    auto const empty_result = tree.erase(tree.begin(), tree.begin());
    st_verify_(empty_result.status);
    verify_against_oracle(tree, oracle);
}

/** @brief Dropping an arbitrary subset must leave a balanced tree, not merely a correct one. */
static void weight_balance_erase_if_rebalances() {
    std::mt19937 generator(2026);
    for (std::size_t trial = 0; trial < 200; ++trial) {
        ordered_set_t tree;
        std::set<int> oracle;
        for (std::size_t index = 0; index < 300; ++index) {
            int const element = int(generator() % 1000);
            [[maybe_unused]] auto const added = tree.insert(int(element));
            oracle.insert(element);
        }

        int const residue = int(generator() % 7);
        std::size_t const dropped = tree.erase_if([&](int const &element) noexcept { return element % 7 == residue; });
        std::size_t counted = 0;
        for (auto position = oracle.begin(); position != oracle.end();) {
            if (*position % 7 == residue) position = oracle.erase(position), ++counted;
            else ++position;
        }
        st_verify_eq_(dropped, counted);
        verify_against_oracle(tree, oracle);
    }
}

/** @brief A long randomized mutation sequence, re-checking every invariant after each step. */
static void weight_balance_randomized_mutations() {
    using placement_t = ordered_set_t::node_t::node_placement_t;
    std::mt19937 generator(20260817);
    ordered_set_t tree;
    std::set<int> oracle;
    for (std::size_t step = 0; step < 20000; ++step) {
        int const element = int(generator() % 800);
        switch (generator() % 4) {
        case 0: {
            auto const result = tree.insert(int(element));
            st_verify_ne_(result.node, nullptr);
            st_verify_eq_(result.placement == placement_t::made_k, oracle.insert(element).second);
            break;
        }
        case 1: {
            auto const result = tree.upsert(int(element));
            st_verify_(bool(result));
            oracle.insert(element);
            break;
        }
        case 2: st_verify_eq_(tree.erase(element), oracle.erase(element) != 0); break;
        case 3: {
            int const upper = element + int(generator() % 200);
            std::size_t visited = 0;
            tree.erase_range(element, upper, [&](int const &) noexcept { ++visited; });
            std::size_t const expected =
                std::size_t(std::distance(oracle.lower_bound(element), oracle.lower_bound(upper)));
            st_verify_eq_(visited, expected);
            oracle.erase(oracle.lower_bound(element), oracle.lower_bound(upper));
            break;
        }
        }
        verify_against_oracle(tree, oracle);
    }
}

#pragma endregion Weight Balance Invariants

#pragma region Augmented Order Statistics

/**
 *  @brief Counts only the entries whose mapped value is non-zero.
 *    Stands in for a wrapper's "this entry is the answer for its key" predicate: a property of the entry,
 *    decided outside the tree, and free to flip while the entry sits in place.
 */
struct live_augmentation_t {
    static std::size_t augmented_count(mapping<int, int> const &entry) noexcept { return entry.mapped != 0 ? 1u : 0u; }
};

using augmented_map_t = wb_map<int, int, std::less<int>, std::allocator<mapping<int, int>>, live_augmentation_t>;
using augmented_node_t = augmented_map_t::node_t;
using plain_map_t = wb_map<int, int, std::less<int>, std::allocator<mapping<int, int>>>;

/** @brief The layout an unaugmented node must match byte for byte, whatever the default policy costs. */
struct unaugmented_reference_layout_t {
    int fruit;
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
 *  @brief The layout an unaugmented @b map node must match, whose element is a pair rather than a key.
 *
 *  Kept apart from the set's reference because the two only agree where the padding after a lone @c int
 *  happens to swallow the difference: on LP64 both come to 32 bytes, and on a 32-bit target the set's is
 *  16 while the map's is 20.
 */
struct unaugmented_map_reference_layout_t {
    mapping<int, int> fruit;
    void *left;
    void *right;
    std::size_t size;
};

static_assert(sizeof(plain_map_t::node_t) == sizeof(unaugmented_map_reference_layout_t),
              "A map node is the same shape once the default policy is elided");
static_assert(sizeof(augmented_node_t) == sizeof(plain_map_t::node_t) + sizeof(std::size_t),
              "An augmented node pays exactly one counter, and nothing else");

/** @brief The keys the oracle considers counted, in sorted order. */
static std::vector<int> counted_keys(std::map<int, int> const &oracle) noexcept {
    std::vector<int> keys;
    for (auto const &entry : oracle)
        if (entry.second != 0) keys.push_back(entry.first);
    return keys;
}

/** @brief Checks structure, the augmented total, and every augmented @c select and @c rank answer. */
static void verify_augmented_against_oracle(augmented_map_t &tree, std::map<int, int> const &oracle) noexcept {
    subtree_counts_t const counts = verify_invariants(tree.root());
    st_verify_eq_(counts.size, oracle.size());
    st_verify_eq_(tree.size(), oracle.size());

    std::vector<int> const live = counted_keys(oracle);
    st_verify_eq_(tree.augmented_size(), live.size());
    st_verify_eq_(counts.augmented_size, live.size());

    for (std::size_t index = 0; index < live.size(); ++index) {
        augmented_node_t const *selected = tree.select_augmented(index);
        st_verify_ne_(selected, nullptr);
        st_verify_eq_(selected->fruit.key, live[index]);
        st_verify_eq_(tree.rank_augmented(live[index]), index);
    }
    st_verify_eq_(tree.select_augmented(live.size()), nullptr);
}

/**
 *  @brief The same mutation fuzz the plain tree runs, plus in-place predicate flips.
 *    A flip changes an entry the tree already holds, which is the transition a version store makes when a
 *    newer version supersedes an older one, and it must cost a path repair rather than a rescan.
 */
static void augmented_randomized_mutations() {
    using placement_t = augmented_map_t::node_t::node_placement_t;
    std::mt19937 generator(20260818);
    augmented_map_t tree;
    std::map<int, int> oracle;
    for (std::size_t step = 0; step < 20000; ++step) {
        int const key = int(generator() % 200);
        switch (generator() % 7) {
        case 0: {
            int const mapped = int(generator() % 2);
            auto const result = tree.insert(mapping<int, int> {key, mapped});
            st_verify_ne_(result.node, nullptr);
            st_verify_eq_(result.placement == placement_t::made_k, oracle.emplace(key, mapped).second);
            break;
        }
        case 1: {
            int const mapped = int(generator() % 2);
            auto const result = tree.upsert(mapping<int, int> {key, mapped});
            st_verify_(bool(result));
            oracle[key] = mapped;
            break;
        }
        case 2: st_verify_eq_(tree.erase(key), oracle.erase(key) != 0); break;
        case 3: {
            int const upper = key + int(generator() % 60);
            std::size_t visited = 0;
            tree.erase_range(key, upper, [&](mapping<int, int> const &) noexcept { ++visited; });
            std::size_t const expected = std::size_t(std::distance(oracle.lower_bound(key), oracle.lower_bound(upper)));
            st_verify_eq_(visited, expected);
            oracle.erase(oracle.lower_bound(key), oracle.lower_bound(upper));
            break;
        }
        case 4: {
            // Flip the count under the tree's feet, as a publish site would - and flip the successor too,
            // since one publish can supersede a neighbour, retagging two entries for a single write.
            auto position = tree.find(key);
            bool const present = position != tree.end();
            st_verify_eq_(present, oracle.count(key) != 0);
            if (!present) break;
            position->mapped = position->mapped != 0 ? 0 : 1;
            oracle[key] = position->mapped;
            st_verify_(tree.refresh_augmentation(key));

            auto successor = position;
            if (++successor != tree.end()) {
                int const successor_key = successor->key;
                successor->mapped = successor->mapped != 0 ? 0 : 1;
                oracle[successor_key] = successor->mapped;
                st_verify_(tree.refresh_augmentation(successor_key));
            }
            break;
        }
        case 5: {
            // `erase_if` rebuilds both halves through `join`, which has to carry the counts along.
            int const residue = int(generator() % 5);
            std::size_t const dropped =
                tree.erase_if([&](mapping<int, int> const &entry) noexcept { return entry.key % 5 == residue; });
            std::size_t counted = 0;
            for (auto position = oracle.begin(); position != oracle.end();) {
                if (position->first % 5 == residue) position = oracle.erase(position), ++counted;
                else ++position;
            }
            st_verify_eq_(dropped, counted);
            break;
        }
        case 6: {
            // Splitting and re-joining walks `join_with_root` down both spines.
            auto halves = tree.split(key);
            verify_augmented_against_oracle(halves.left, std::map<int, int>(oracle.begin(), oracle.lower_bound(key)));
            verify_augmented_against_oracle(halves.right, std::map<int, int>(oracle.lower_bound(key), oracle.end()));
            halves.left.join(halves.right);
            tree = std::move(halves.left);
            break;
        }
        }
        verify_augmented_against_oracle(tree, oracle);
    }
}

using counted_map_t = wb_map<int, int, counting_comparator_t, std::allocator<mapping<int, int>>, live_augmentation_t>;

/** @brief Depth of @p wanted below the root, which is exactly what a @c select_augmented descent walks. */
static std::size_t depth_of(counted_map_t const &tree, int wanted) noexcept {
    std::size_t depth = 0;
    for (auto const *node = tree.root(); node; ++depth) {
        if (wanted < node->fruit.key) node = node->left;
        else if (node->fruit.key < wanted) node = node->right;
        else return depth;
    }
    st_verify_(false && "The selected key must be reachable from the root");
    return depth;
}

/** @brief Base-2 logarithm of @p count, rounded down, with @c log2(0) reported as 0. */
static std::size_t floor_log2(std::size_t count) noexcept {
    std::size_t bits = 0;
    while (count > 1) count >>= 1, ++bits;
    return bits;
}

/**
 *  @brief Bounds augmented @c select and @c rank against @c log2(size) at two sizes.
 *    Two sizes 64x apart, so a linear cost cannot pass as a logarithmic one: a linear descent would grow
 *    with the same factor, while these bounds only allow the measured counts to grow with the height.
 */
static void augmented_select_is_logarithmic() {
    std::size_t previous_select_steps = 0, previous_rank_comparisons = 0;
    for (std::size_t element_count : {std::size_t(4096), std::size_t(262144)}) {
        counted_map_t tree(counting_comparator_t {}, std::allocator<counted_map_t::node_t> {});

        // Every third key counts, so the augmented descent cannot degenerate into the plain one.
        std::mt19937 generator(4242);
        std::vector<int> live;
        for (std::size_t index = 0; index < element_count; ++index) {
            int const key = int(index);
            int const mapped = index % 3 == 0 ? 1 : 0;
            [[maybe_unused]] auto const added = tree.upsert(mapping<int, int> {key, mapped});
            if (mapped) live.push_back(key);
        }
        st_verify_eq_(tree.augmented_size(), live.size());

        std::size_t const budget = floor_log2(element_count);
        std::size_t worst_select_steps = 0, worst_rank_comparisons = 0;
        for (std::size_t probe = 0; probe < 512; ++probe) {
            std::size_t const index = generator() % live.size();

            auto const *selected = tree.select_augmented(index);
            st_verify_ne_(selected, nullptr);
            st_verify_eq_(selected->fruit.key, live[index]);
            // `select_augmented` follows one root-to-node path, so the node's depth is its exact step count.
            std::size_t const steps = depth_of(tree, live[index]) + 1;
            worst_select_steps = steps > worst_select_steps ? steps : worst_select_steps;

            call_tally_t::reset();
            st_verify_eq_(tree.rank_augmented(live[index]), index);
            std::size_t const comparisons = call_tally_t::comparisons_count();
            worst_rank_comparisons = comparisons > worst_rank_comparisons ? comparisons : worst_rank_comparisons;
        }

        // Δ=3 caps the height at log(n)/log(4/3) ≈ 2.41 log2(n), and `rank` spends two comparisons a level.
        st_verify_le_(worst_select_steps, 3 * budget + 4);
        st_verify_le_(worst_rank_comparisons, 6 * budget + 8);
        st_verify_ge_(worst_select_steps, previous_select_steps);
        st_verify_ge_(worst_rank_comparisons, previous_rank_comparisons);
        previous_select_steps = worst_select_steps;
        previous_rank_comparisons = worst_rank_comparisons;
    }

    // A linear scan of the larger tree would have cost tens of thousands of steps, not a few dozen.
    st_verify_lt_(previous_select_steps, 64);
    st_verify_lt_(previous_rank_comparisons, 128);
}

#pragma endregion Augmented Order Statistics

#pragma region Allocation Failure

/** @brief A key that rewrites itself when moved from, so a consumed argument is visible to the oracle. */
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

/** @brief Neither a duplicate key nor an exhausted allocator may consume the caller's entry. */
static void allocation_failure_preserves_the_argument() {
    using placement_t = traced_set_t::node_t::node_placement_t;
    allocation_ledger_t ledger;
    ledger.allow(3);
    {
        traced_set_t tree(traced_less_t {}, stateful_allocator<traced_set_t::node_t>(ledger));
        for (int element = 0; element < 3; ++element) {
            traced_key_t key(element);
            auto const made = tree.insert(std::move(key));
            st_verify_(made.placement == placement_t::made_k);
            st_verify_ne_(made.node, nullptr);
        }

        // Budget exhausted - the rejected entry must come back untouched.
        traced_key_t rejected(99);
        auto const refused = tree.insert(std::move(rejected));
        st_verify_eq_(refused.node, nullptr);
        st_verify_(refused.placement == placement_t::refused_k);
        st_verify_(!rejected.moved_from);

        // A duplicate key must likewise leave the argument alone and report the incumbent.
        ledger.allow(10);
        traced_key_t duplicate(1);
        auto const matched = tree.insert(std::move(duplicate));
        st_verify_ne_(matched.node, nullptr);
        st_verify_(matched.placement == placement_t::matched_k);
        st_verify_(!duplicate.moved_from);
        st_verify_eq_(tree.size(), 3u);
    }
    ledger.verify_balanced();
}

/**
 *  @brief An upsert must name which of its three outcomes happened.
 *
 *  A node that was made, one that was already there, and an allocation that never came back are
 *  three answers, and none of them should have to be read off a null check.
 */
static void upsert_reports_placement() {
    using placement_t = traced_set_t::node_t::node_placement_t;
    allocation_ledger_t ledger;
    ledger.allow(1);
    {
        traced_set_t tree(traced_less_t {}, stateful_allocator<traced_set_t::node_t>(ledger));

        auto const made = tree.upsert(traced_key_t(1));
        st_verify_eq_(made.placement, placement_t::made_k, "a fresh key must report a node of its own");
        st_verify_(made);
        st_verify_ne_(made.node, nullptr);

        auto const matched = tree.upsert(traced_key_t(1));
        st_verify_eq_(matched.placement, placement_t::matched_k, "a key already there must report a match");
        st_verify_(succeeded(matched) && "an overwrite is not a failure");
        st_verify_ne_(matched.node, nullptr);

        // The budget is spent, so the second key has no node to live in - the outcome that shares a
        // null match with nothing else.
        auto const refused = tree.upsert(traced_key_t(2));
        st_verify_eq_(refused.placement, placement_t::refused_k, "a refused allocation must say so");
        st_verify_(failed(refused));
        st_verify_eq_(refused.node, nullptr);
        st_verify_eq_(tree.size(), 1u);
    }
    ledger.verify_balanced();
}

/** @brief @c insert_if_missing must separate "already there" from "out of memory". */
static void allocation_failure_is_distinct_from_presence() {
    using placement_t = traced_set_t::node_t::node_placement_t;
    allocation_ledger_t ledger;
    ledger.allow(2);
    {
        traced_set_t tree(traced_less_t {}, stateful_allocator<traced_set_t::node_t>(ledger));
        {
            traced_key_t key(1);
            auto const made = tree.insert(std::move(key));
            st_verify_(made.placement == placement_t::made_k);
            st_verify_ne_(made.node, nullptr);
        }

        auto const on_present = tree.insert_if_missing(traced_key_t(1));
        st_verify_eq_(on_present.placement, placement_t::matched_k, "an incumbent is not a refusal");
        st_verify_ne_(on_present.position, tree.end());

        ledger.refuse_everything();
        auto const on_exhausted = tree.insert_if_missing(traced_key_t(2));
        st_verify_eq_(on_exhausted.placement, placement_t::refused_k, "no node means no insertion");
        st_verify_eq_(on_exhausted.position, tree.end());
        st_verify_eq_(tree.size(), 1u);
    }
    ledger.verify_balanced();
}

#pragma endregion Allocation Failure

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

static void transactional_defects_transaction_range_interleaves_staged_and_committed() {
    test_transaction_range_interleaves_staged_and_committed<transactional_trivial_map_t>();
    test_transaction_range_interleaves_staged_and_committed<transactional_composite_map_t>();
    test_transaction_range_interleaves_staged_and_committed<transactional_heavy_map_t>();
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
    test_container_balances_counted_keys<wb_map<counted_key_t, int>>();
}

static void fixture_coverage_rollback_balances_counted_keys() {
    test_rollback_balances_counted_keys<monotonic_wb_map<counted_key_t, int>>();
}

#pragma endregion Fixture Coverage

} // namespace

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

    failures += run_test(filter, "weight_balance.insert_rebalances", weight_balance_insert_rebalances);
    failures +=
        run_test(filter, "weight_balance.two_child_erase_rebalances", weight_balance_two_child_erase_rebalances);
    failures += run_test(filter, "weight_balance.split_and_join_track_size", weight_balance_split_and_join_track_size);
    failures += run_test(filter, "weight_balance.range_is_half_open", weight_balance_range_is_half_open);
    failures += run_test(filter, "weight_balance.erase_iterator_range", weight_balance_erase_iterator_range);
    failures += run_test(filter, "weight_balance.erase_if_rebalances", weight_balance_erase_if_rebalances);
    failures += run_test(filter, "weight_balance.upsert_reports_placement", upsert_reports_placement);
    failures += run_test(filter, "weight_balance.randomized_mutations", weight_balance_randomized_mutations);

    failures += run_test(filter, "augmented.randomized_mutations", augmented_randomized_mutations);
    failures += run_test(filter, "augmented.select_is_logarithmic", augmented_select_is_logarithmic);

    failures +=
        run_test(filter, "allocation_failure.preserves_the_argument", allocation_failure_preserves_the_argument);
    failures +=
        run_test(filter, "allocation_failure.is_distinct_from_presence", allocation_failure_is_distinct_from_presence);

    failures += run_test(filter, "transactional_defects.direct_write_spares_staged_version",
                         transactional_defects_direct_write_spares_staged_version);
    failures += run_test(filter, "transactional_defects.direct_erase_spares_staged_version",
                         transactional_defects_direct_erase_spares_staged_version);
    failures += run_test(filter, "transactional_defects.erase_range_spares_staged_versions",
                         transactional_defects_erase_range_spares_staged_versions);
    failures += run_test(filter, "transactional_defects.clear_keeps_generations_moving",
                         transactional_defects_clear_keeps_generations_moving);
    failures += run_test(filter, "transactional_defects.committed_erase_hidden_from_point_reads",
                         transactional_defects_committed_erase_hidden_from_point_reads);
    failures += run_test(filter, "transactional_defects.committed_erase_hidden_from_ordered_reads",
                         transactional_defects_committed_erase_hidden_from_ordered_reads);
    failures += run_test(filter, "transactional_defects.committed_erase_hidden_from_update_range",
                         transactional_defects_committed_erase_hidden_from_update_range);
    failures += run_test(filter, "transactional_defects.vacuum_reclaims_committed_tombstones",
                         transactional_defects_vacuum_reclaims_committed_tombstones);
    failures += run_test(filter, "transactional_defects.vacuum_spares_staged_versions",
                         transactional_defects_vacuum_spares_staged_versions);
    failures += run_test(filter, "transactional_defects.windowed_vacuum_reclaims_one_slice",
                         transactional_defects_windowed_vacuum_reclaims_one_slice);

    failures += run_test(filter, "commit_stamp.rolled_back_stage_does_not_abort_a_peer",
                         commit_stamp_rolled_back_stage_does_not_abort_a_peer);
    failures += run_test(filter, "commit_stamp.lost_update_is_refused", commit_stamp_lost_update_is_refused);
    failures +=
        run_test(filter, "commit_stamp.find_and_watch_records_absence", commit_stamp_find_and_watch_records_absence);

    failures += run_test(filter, "transactional_consistency.find_does_not_watch",
                         transactional_consistency_find_does_not_watch);

    failures += run_test(filter, "transactional_defects.transaction_range_interleaves_staged_and_committed",
                         transactional_defects_transaction_range_interleaves_staged_and_committed);
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

    return report_test_failures(failures);
}
