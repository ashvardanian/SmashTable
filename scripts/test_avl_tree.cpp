/**
 *  @brief Test instantiations for AVL tree containers. Covers basic_avl_tree (non-transactional) and
 *      transactional_store<basic_avl_tree> (transactional). Includes AVL-specific algorithms
 *      (merge/split/join) and transaction architecture tests.
 *  @author Ash Vardanian
 *  @file scripts/test_avl_tree.cpp
 *  @date October 25, 2025
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define ST_STRICT_CALLBACK_CHECKS_ 1

#include <cstddef> // `std::size_t`

#include <iterator> // `std::bidirectional_iterator`
#include <memory>   // `std::unique_ptr`
#include <random>   // `std::mt19937`
#include <set>      // `std::set`
#include <vector>   // `std::vector`

#include <smashtable/basic_avl_tree.hpp>
#include <smashtable/transactional_store.hpp>
#include <smashtable/transactional_std_store.hpp>

#include "test.hpp"
#include "test_basic.hpp"
#include "test_commit_stamp.hpp"
#include "test_consistency.hpp"
#include "test_fixture_coverage.hpp"
#include "test_transactional_store_defects.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

#pragma region Type Aliases

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Transaction: ✗
 *  Tests: Baseline non-transparent comparator path
 */
using trivial_set_t = avl_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial | Memory: Tracked | Transaction: ✗
 *  Tests: Resource accounting, allocation failure injection
 */
using tracking_set_t = avl_set<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Trivial | Memory: Stack | Transaction: ✗
 *  Tests: Identifier extraction, composite_key_compare_t::value_type lookups
 */
using composite_set_t = avl_set<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() → expected<T> | Memory: Heap | Transaction: ✗
 *  Tests: OOM during .copy(), string_view lookups without materialization
 */
using heavy_set_t = avl_set<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial (key & value) | Memory: Stack | Transaction: ✗
 *  Value: int | Tests: Baseline map operations, non-transparent path
 */
using trivial_map_t =
    avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial (key & value) | Memory: Tracked | Transaction: ✗
 *  Value: int | Tests: Map resource accounting, POCCA/POCMA on key-value pairs
 */
using tracking_map_t = avl_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Key trivial, value .copy() | Memory: Heap (value) | Transaction: ✗
 *  Value: guarded_payload_t | Tests: Mixed trivial/non-trivial, value OOM scenarios
 */
using composite_map_t = avl_map<composite_key_t, guarded_payload_t, composite_key_compare_t,
                                std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() on key & value | Memory: Heap (both) | Transaction: ✗
 *  Value: guarded_payload_t | Tests: Dual-heap OOM, worst-case complexity
 */
using heavy_map_t =
    avl_map<heavy_key_t, guarded_payload_t, std::less<void>, std::allocator<mapping<heavy_key_t, guarded_payload_t>>>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Transaction: ✓
 *  Tests: Baseline transactional correctness, isolation levels
 */
using transactional_trivial_set_t =
    transactional_avl_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial | Memory: Tracked | Transaction: ✓
 *  Tests: Transaction resource accounting, OOM during stage/commit
 */
using transactional_tracking_set_t = transactional_avl_set<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Trivial | Memory: Stack | Transaction: ✓
 *  Tests: Heterogeneous watch/find in transactions
 */
using transactional_composite_set_t =
    transactional_avl_set<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() → expected<T> | Memory: Heap | Transaction: ✓
 *  Tests: Watch copy OOM, transaction rollback with heap types
 */
using transactional_heavy_set_t = transactional_avl_set<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial (key & value) | Memory: Stack | Transaction: ✓
 *  Value: int | Tests: Transactional map operations, value overwrites
 */
using transactional_trivial_map_t =
    transactional_avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial (key & value) | Memory: Tracked | Transaction: ✓
 *  Value: int | Tests: Transaction allocation patterns, map POCCA/POCMA
 */
using transactional_tracking_map_t =
    transactional_avl_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Key trivial, value .copy() | Memory: Heap (value) | Transaction: ✓
 *  Value: guarded_payload_t | Tests: Transaction rollback with non-trivial values
 */
using transactional_composite_map_t =
    transactional_avl_map<composite_key_t, guarded_payload_t, composite_key_compare_t,
                          std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() on key & value | Memory: Heap (both) | Transaction: ✓
 *  Value: guarded_payload_t | Tests: Worst-case transactional complexity, dual-heap rollback
 */
using transactional_heavy_map_t = transactional_avl_map<heavy_key_t, guarded_payload_t, std::less<void>,
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

#pragma region Structural Invariants

/** @brief Node count and height of a walked subtree. */
struct subtree_shape_t {
    std::size_t count = 0;
    std::ptrdiff_t height = 0;
};

/** @brief Recursively checks parent back-pointers, stored heights, AVL balance, and key order. */
template <typename tree_type_>
static subtree_shape_t verify_subtree(typename tree_type_::node_t const *node,
                                      typename tree_type_::node_t const *parent,
                                      typename tree_type_::comparator_t const &comparator) {
    if (!node) return {};
    st_verify_(node->parent == parent && "every node points back at its parent");

    subtree_shape_t const left = verify_subtree<tree_type_>(node->left, node, comparator);
    subtree_shape_t const right = verify_subtree<tree_type_>(node->right, node, comparator);
    std::ptrdiff_t const height = 1 + (left.height > right.height ? left.height : right.height);
    std::ptrdiff_t const balance = left.height - right.height;
    st_verify_(node->height == height && "the stored height matches the walked one");
    st_verify_(balance >= -1 && balance <= 1 && "AVL balance never exceeds one");

    if (node->left)
        st_verify_(comparator(mapping_key_or_itself(node->left->fruit), mapping_key_or_itself(node->fruit)) &&
                   "the left child sorts before its parent");
    if (node->right)
        st_verify_(comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(node->right->fruit)) &&
                   "the right child sorts after its parent");

    return {left.count + right.count + 1, height};
}

/** @brief Walks the whole tree, checking its shape and that the node count agrees with @c size(). */
template <typename tree_type_>
static void verify_invariants(tree_type_ const &tree) {
    auto const *root = tree.root();
    st_verify_((!root || !root->parent) && "the root has no parent");
    typename tree_type_::comparator_t const comparator = tree.key_comp();
    subtree_shape_t const shape = verify_subtree<tree_type_>(root, nullptr, comparator);
    st_verify_eq_(shape.count, tree.size());
}

/** @brief A copied tree must carry its own parent links, or the first @c ++it walks into garbage. */
template <typename tree_type_>
static void test_copy_preserves_parent_links() {
    using member_t = typename tree_type_::value_type;
    tree_type_ tree;
    for (trivial_id_t identifier : {5u, 2u, 8u, 1u, 3u, 7u, 9u})
        st_verify_(succeeded(tree.upsert(trivial_id_to_member<member_t>(identifier))));

    auto copied = tree.copy();
    st_verify_(copied.has_value());
    verify_invariants(*copied);

    std::size_t walked = 0;
    for ([[maybe_unused]] auto const &member : *copied) ++walked;
    st_verify_eq_(walked, tree.size());
}

/** @brief The perfectly balanced tree built from a sorted range must be linked both ways. */
template <typename tree_type_>
static void test_bulk_sorted_insert_links_parents() {
    using member_t = typename tree_type_::value_type;
    for (std::size_t count = 1; count <= 64; ++count) {
        tree_type_ tree;
        st_verify_(succeeded(tree.upsert(trivial_id_to_member<member_t>(1000))));

        std::vector<member_t> sorted;
        for (std::size_t index = 0; index < count; ++index) sorted.push_back(trivial_id_to_member<member_t>(index));

        st_verify_(succeeded(tree.insert_if_missing(std::make_move_iterator(sorted.begin()),
                                                    std::make_move_iterator(sorted.end()), assume_sorted_t {})));
        verify_invariants(tree);
        st_verify_eq_(tree.size(), count + 1);

        std::size_t walked = 0;
        for ([[maybe_unused]] auto const &member : tree) ++walked;
        st_verify_eq_(walked, tree.size());
    }
}

/** @brief Erasing a two-child root promotes its successor, which nobody above it can rebalance. */
template <typename tree_type_>
static void test_erase_root_rebalances_promoted_node() {
    using member_t = typename tree_type_::value_type;
    tree_type_ tree;
    for (trivial_id_t identifier : {10u, 5u, 15u, 3u, 7u})
        st_verify_(succeeded(tree.upsert(trivial_id_to_member<member_t>(identifier))));
    verify_invariants(tree);

    st_verify_(tree.erase(trivial_id_to_key<member_t>(10)));
    verify_invariants(tree);
    st_verify_eq_(tree.size(), 4);
}

/** @brief The callback range is @c [lower, upper), matching what @c erase_range removes. */
template <typename tree_type_>
static void test_range_excludes_upper_bound() {
    using member_t = typename tree_type_::value_type;
    tree_type_ tree;
    for (trivial_id_t identifier = 1; identifier <= 5; ++identifier)
        st_verify_(succeeded(tree.upsert(trivial_id_to_member<member_t>(identifier))));

    std::vector<trivial_id_t> seen;
    tree.range(trivial_id_to_key<member_t>(2), trivial_id_to_key<member_t>(4),
               [&](auto const &member) noexcept { seen.push_back(mapping_key_or_itself(member).unique_id); });

    st_verify_eq_(seen.size(), 2);
    st_verify_eq_(seen[0], 2);
    st_verify_eq_(seen[1], 3);
}

/** @brief Splitting rebuilds two trees whose sides can differ in height by far more than one. */
template <typename tree_type_>
static void test_split_and_erase_range_stay_balanced() {
    using member_t = typename tree_type_::value_type;
    for (std::size_t total = 1; total <= 128; ++total) {
        tree_type_ tree;
        for (std::size_t index = 0; index < total; ++index)
            st_verify_(succeeded(tree.upsert(trivial_id_to_member<member_t>(index * 2))));

        auto halves = tree.split(trivial_id_to_key<member_t>(total));
        verify_invariants(halves.left);
        verify_invariants(halves.right);
        st_verify_eq_(halves.left.size() + halves.right.size(), total);

        tree_type_ rebuilt;
        for (std::size_t index = 0; index < total; ++index)
            st_verify_(succeeded(rebuilt.upsert(trivial_id_to_member<member_t>(index))));
        rebuilt.erase_range(trivial_id_to_key<member_t>(total / 4), trivial_id_to_key<member_t>(total / 2));
        verify_invariants(rebuilt);

        std::size_t walked = 0;
        for ([[maybe_unused]] auto const &member : rebuilt) ++walked;
        st_verify_eq_(walked, rebuilt.size());
    }
}

/** @brief The split-based union of two interleaved trees must come out balanced and fully linked. */
template <typename tree_type_>
static void test_merge_unique_stays_balanced() {
    using member_t = typename tree_type_::value_type;
    for (std::size_t total = 1; total <= 128; ++total) {
        tree_type_ evens, odds;
        for (std::size_t index = 0; index < total; ++index) {
            st_verify_(succeeded(evens.upsert(trivial_id_to_member<member_t>(index * 2))));
            st_verify_(succeeded(odds.upsert(trivial_id_to_member<member_t>(index * 2 + 1))));
        }

        evens.merge(odds, assume_unique_t {});
        verify_invariants(evens);
        st_verify_eq_(evens.size(), total * 2);

        std::size_t walked = 0;
        for ([[maybe_unused]] auto const &member : evens) ++walked;
        st_verify_eq_(walked, evens.size());

        // Fully ordered inputs take the join fast path instead of the split-based one.
        tree_type_ low, high;
        for (std::size_t index = 0; index < total; ++index) {
            st_verify_(succeeded(low.upsert(trivial_id_to_member<member_t>(index))));
            st_verify_(succeeded(high.upsert(trivial_id_to_member<member_t>(index + total))));
        }
        low.merge(high, assume_unique_t {});
        verify_invariants(low);
        st_verify_eq_(low.size(), total * 2);
    }
}

/** @brief Randomized insert and erase sequence, re-checking every invariant after each mutation. */
template <typename tree_type_>
static void test_random_mutations_preserve_invariants(std::size_t steps = 4000, unsigned int seed = 42) {
    using member_t = typename tree_type_::value_type;
    std::mt19937 generator(seed);
    tree_type_ tree;
    std::set<trivial_id_t> oracle;

    for (std::size_t step = 0; step < steps; ++step) {
        trivial_id_t const identifier = generator() % 300;
        if (generator() % 2) {
            st_verify_(succeeded(tree.upsert(trivial_id_to_member<member_t>(identifier))));
            oracle.insert(identifier);
        }
        else { st_verify_eq_(tree.erase(trivial_id_to_key<member_t>(identifier)), oracle.erase(identifier) == 1); }
        verify_invariants(tree);
        st_verify_eq_(tree.size(), oracle.size());
    }

    auto expected = oracle.begin();
    for (auto const &member : tree) {
        st_verify_eq_(mapping_key_or_itself(member).unique_id, *expected);
        ++expected;
    }
    st_verify_(expected == oracle.end());
}

/** @brief Post-decrement must exist for the type to model @c std::bidirectional_iterator. */
template <typename tree_type_>
static void test_iterator_post_decrement() {
    static_assert(std::bidirectional_iterator<typename tree_type_::iterator>, "iterator must be bidirectional");
    static_assert(std::bidirectional_iterator<typename tree_type_::const_iterator>,
                  "const_iterator must be bidirectional");

    using member_t = typename tree_type_::value_type;
    tree_type_ tree;
    for (trivial_id_t identifier = 1; identifier <= 3; ++identifier)
        st_verify_(succeeded(tree.upsert(trivial_id_to_member<member_t>(identifier))));

    auto position = tree.end();
    auto const previous = position--;
    st_verify_(previous == tree.end());
    st_verify_eq_(mapping_key_or_itself(*position).unique_id, 3);
    position--;
    st_verify_eq_(mapping_key_or_itself(*position).unique_id, 2);
}

/** @brief The node-level @c equal_range reports the half-open bounds around a key. */
template <typename tree_type_>
static void test_node_equal_range() {
    using member_t = typename tree_type_::value_type;
    using node_t = typename tree_type_::node_t;
    tree_type_ tree;
    for (trivial_id_t identifier = 1; identifier <= 5; ++identifier)
        st_verify_(succeeded(tree.upsert(trivial_id_to_member<member_t>(identifier))));

    typename tree_type_::comparator_t const comparator = tree.key_comp();
    auto const present = node_t::equal_range(tree.root(), trivial_id_to_key<member_t>(3), comparator);
    st_verify_(present.lower_bound && mapping_key_or_itself(present.lower_bound->fruit).unique_id == 3);
    st_verify_(present.upper_bound && mapping_key_or_itself(present.upper_bound->fruit).unique_id == 4);
    st_verify_(present.lowest_common_ancestor &&
               mapping_key_or_itself(present.lowest_common_ancestor->fruit).unique_id == 3);

    auto const absent = node_t::equal_range(tree.root(), trivial_id_to_key<member_t>(9), comparator);
    st_verify_(!absent.lower_bound && !absent.upper_bound);
}

/** @brief A bulk upsert that runs out of memory mid-merge must say so rather than report success. */
static void test_bulk_upsert_reports_allocation_failure() {
    using budget_set_t = avl_set<trivial_key_t, std::less<trivial_key_t>, stateful_allocator<trivial_key_t>>;
    allocation_ledger_t ledger;
    ledger.allow(64);
    budget_set_t tree {typename budget_set_t::allocator_t(ledger)};
    for (trivial_id_t identifier : {1u, 2u}) st_verify_(succeeded(tree.upsert(trivial_key_t(identifier))));

    // Exactly enough to build the three-node temporary, nothing left for the one new key it carries.
    ledger.reset();
    ledger.allow(3);
    std::vector<trivial_key_t> const members = {trivial_key_t(1), trivial_key_t(2), trivial_key_t(99)};
    st_verify_eq_(tree.upsert(members.begin(), members.end()), status_t::out_of_memory_heap_k);
    st_verify_(!tree.contains(trivial_key_t(99)));

    // The ledger says what was actually asked for.
    st_verify_eq_(ledger.granted_count, std::size_t {3});
    st_verify_((ledger.refused_count > 0) && "the merge must have asked for one more node than it could have");
    verify_invariants(tree);
}

/** @brief Move-only entries must survive the assignment operator of an upsert result. */
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
    st_verify_(static_cast<bool>(result));
    result = move_only_key_t(5); // ! Instantiates `upsert_result_t::operator=`
    st_verify_eq_(tree.size(), 1);
}

static void structure_copy_preserves_parent_links() {
    test_copy_preserves_parent_links<trivial_set_t>();
    test_copy_preserves_parent_links<tracking_set_t>();
    test_copy_preserves_parent_links<trivial_map_t>();
}

static void structure_bulk_sorted_insert_links_parents() {
    test_bulk_sorted_insert_links_parents<trivial_set_t>();
    test_bulk_sorted_insert_links_parents<tracking_set_t>();
    test_bulk_sorted_insert_links_parents<trivial_map_t>();
}

static void structure_erase_root_rebalances_promoted_node() {
    test_erase_root_rebalances_promoted_node<trivial_set_t>();
    test_erase_root_rebalances_promoted_node<tracking_set_t>();
    test_erase_root_rebalances_promoted_node<trivial_map_t>();
}

static void structure_range_excludes_upper_bound() {
    test_range_excludes_upper_bound<trivial_set_t>();
    test_range_excludes_upper_bound<tracking_set_t>();
    test_range_excludes_upper_bound<trivial_map_t>();
}

static void structure_split_and_erase_range_stay_balanced() {
    test_split_and_erase_range_stay_balanced<trivial_set_t>();
    test_split_and_erase_range_stay_balanced<trivial_map_t>();
}

static void structure_merge_unique_stays_balanced() {
    test_merge_unique_stays_balanced<trivial_set_t>();
    test_merge_unique_stays_balanced<trivial_map_t>();
}

static void structure_random_mutations_preserve_invariants() {
    test_random_mutations_preserve_invariants<trivial_set_t>();
    test_random_mutations_preserve_invariants<tracking_set_t>();
    test_random_mutations_preserve_invariants<trivial_map_t>();
}

static void structure_iterator_post_decrement() {
    test_iterator_post_decrement<trivial_set_t>();
    test_iterator_post_decrement<trivial_map_t>();
}

static void structure_node_equal_range() {
    test_node_equal_range<trivial_set_t>();
    test_node_equal_range<trivial_map_t>();
}

#pragma endregion Structural Invariants

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
    test_container_balances_counted_keys<avl_map<counted_key_t, int>>();
}

static void fixture_coverage_rollback_balances_counted_keys() {
    test_rollback_balances_counted_keys<transactional_avl_map<counted_key_t, int>>();
}

#pragma endregion Fixture Coverage

/** @brief Erasing through a const iterator must instantiate; nothing in the tree called it before. */
static void structure_erase_const_iterator() {
    trivial_set_t tree;
    for (trivial_id_t identifier : {1u, 2u, 3u}) st_verify_(succeeded(tree.upsert(trivial_key_t(identifier))));

    trivial_set_t::const_iterator const position = tree.find(trivial_key_t(2));
    st_verify_(position != tree.end());
    auto const erased = tree.erase(position);
    st_verify_(succeeded(erased.status));
    st_verify_(!tree.contains(trivial_key_t(2)));
    st_verify_eq_(tree.size(), 2u);
    verify_invariants(tree);
}

using overaligned_set_t = transactional_avl_set<overaligned_key_t>;

static void fixture_coverage_container_honours_over_alignment() {
    test_container_honours_over_alignment<overaligned_set_t>();
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

    failures += run_test(filter, "structure.copy_preserves_parent_links", structure_copy_preserves_parent_links);
    failures +=
        run_test(filter, "structure.bulk_sorted_insert_links_parents", structure_bulk_sorted_insert_links_parents);
    failures += run_test(filter, "structure.erase_root_rebalances_promoted_node",
                         structure_erase_root_rebalances_promoted_node);
    failures += run_test(filter, "structure.range_excludes_upper_bound", structure_range_excludes_upper_bound);
    failures += run_test(filter, "structure.split_and_erase_range_stay_balanced",
                         structure_split_and_erase_range_stay_balanced);
    failures += run_test(filter, "structure.merge_unique_stays_balanced", structure_merge_unique_stays_balanced);
    failures += run_test(filter, "structure.random_mutations_preserve_invariants",
                         structure_random_mutations_preserve_invariants);
    failures += run_test(filter, "structure.iterator_post_decrement", structure_iterator_post_decrement);
    failures += run_test(filter, "structure.node_equal_range", structure_node_equal_range);
    failures += run_test(filter, "structure.bulk_upsert_reports_allocation_failure",
                         test_bulk_upsert_reports_allocation_failure);
    failures += run_test(filter, "structure.move_only_upsert_assignment", test_move_only_upsert_assignment);

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

    failures += run_test(filter, "structure.erase_const_iterator", structure_erase_const_iterator);

    failures += run_test(filter, "fixture_coverage.container_honours_over_alignment",
                         fixture_coverage_container_honours_over_alignment);

    return report_test_failures(failures);
}
