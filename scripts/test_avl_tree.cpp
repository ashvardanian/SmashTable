/**
 *  @brief Test instantiations for AVL tree containers.
 *    Covers basic_avl_tree (non-transactional) and transactional_binary_tree<basic_avl_tree> (transactional).
 *    Includes AVL-specific algorithms (merge/split/join) and transaction architecture tests.
 *
 *  @file test_avl_tree.cpp
 *  @date October 25, 2025
 *  @author Ash Vardanian
 */
#include <gtest/gtest.h>

#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1
#include <smashtable/basic_avl_tree.hpp>
#include <smashtable/transactional_std_store.hpp>
#include <smashtable/transactional_binary_tree.hpp>

#include "test_basic.hpp"
#include "test_consistency.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

#pragma mark - Type Aliases

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
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Transaction: ✓ (MVCC)
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

#pragma mark - Basic Operations Tests

/** @brief Tests operations on empty container don't crash */
TEST(basic_ops, empty_container_operations) {
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
TEST(basic_ops, single_element_operations) {
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
TEST(basic_ops, insertion_patterns) {
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
TEST(basic_ops, bulk_insertion_iterators) {
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
TEST(basic_ops, bulk_upsert_with_duplicate_pairs) {
    // ! Only applies to maps with integral mapped values
    test_bulk_upsert_with_duplicates<trivial_map_t>();
    test_bulk_upsert_with_duplicates<tracking_map_t>();
    test_bulk_upsert_with_duplicates<transactional_trivial_map_t>();
    test_bulk_upsert_with_duplicates<transactional_tracking_map_t>();
}

/** @brief Tests range queries on committed HEAD state */
TEST(basic_ops, range_query_head_state) {
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
TEST(basic_ops, erase_range_head_state) {
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
TEST(basic_ops, heterogeneous_lookups) {
    test_heterogeneous_composite_find<composite_set_t>();
    test_heterogeneous_heavy_string_view_find<heavy_set_t>();
    test_heterogeneous_composite_find<composite_map_t>();
    test_heterogeneous_heavy_string_view_find<heavy_map_t>();
    test_heterogeneous_composite_find<transactional_composite_set_t>();
    test_heterogeneous_heavy_string_view_find<transactional_heavy_set_t>();
    test_heterogeneous_composite_find<transactional_composite_map_t>();
    test_heterogeneous_heavy_string_view_find<transactional_heavy_map_t>();
}

#pragma mark - Consistency & Transaction Tests: Sets

TEST(transactional_consistency, empty_transaction_commit) {
    test_empty_transaction_commit<transactional_trivial_set_t>();
    test_empty_transaction_commit<transactional_tracking_set_t>();
    test_empty_transaction_commit<transactional_composite_set_t>();
    test_empty_transaction_commit<transactional_heavy_set_t>();
    test_empty_transaction_commit<transactional_trivial_map_t>();
    test_empty_transaction_commit<transactional_tracking_map_t>();
    test_empty_transaction_commit<transactional_composite_map_t>();
    test_empty_transaction_commit<transactional_heavy_map_t>();
}

TEST(transactional_consistency, no_dirty_reads_multi_key) {
    test_no_dirty_reads_multi_key<transactional_trivial_map_t>();
    test_no_dirty_reads_multi_key<transactional_tracking_map_t>();
    test_no_dirty_reads_multi_key<transactional_composite_map_t>();
    test_no_dirty_reads_multi_key<transactional_heavy_map_t>();
}

TEST(transactional_consistency, new_transaction_sees_nothing_staged) {
    test_new_transaction_sees_nothing_staged<transactional_trivial_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_tracking_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_trivial_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_tracking_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_map_t>();
}

TEST(transactional_consistency, committed_immediately_visible) {
    test_committed_immediately_visible<transactional_trivial_map_t>();
    test_committed_immediately_visible<transactional_tracking_map_t>();
    test_committed_immediately_visible<transactional_composite_map_t>();
    test_committed_immediately_visible<transactional_heavy_map_t>();
}

TEST(transactional_consistency, multi_key_atomicity_10_keys) {
    test_multi_key_atomicity_10_keys<transactional_trivial_map_t>();
    test_multi_key_atomicity_10_keys<transactional_tracking_map_t>();
    test_multi_key_atomicity_10_keys<transactional_composite_map_t>();
    test_multi_key_atomicity_10_keys<transactional_heavy_map_t>();
}

TEST(transactional_consistency, rollback_makes_all_invisible) {
    test_rollback_makes_all_invisible<transactional_trivial_map_t>();
    test_rollback_makes_all_invisible<transactional_tracking_map_t>();
    test_rollback_makes_all_invisible<transactional_composite_map_t>();
    test_rollback_makes_all_invisible<transactional_heavy_map_t>();
}

TEST(transactional_consistency, range_query_sees_atomic_boundaries) {
    test_range_query_sees_atomic_boundaries<transactional_trivial_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_tracking_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_composite_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_heavy_set_t>();
}

TEST(transactional_consistency, fractured_read_prevention) {
    test_fractured_read_prevention<transactional_trivial_map_t>();
    test_fractured_read_prevention<transactional_tracking_map_t>();
    test_fractured_read_prevention<transactional_composite_map_t>();
    test_fractured_read_prevention<transactional_heavy_map_t>();
}

TEST(transactional_consistency, sequential_updates_never_regress) {
    test_sequential_updates_never_regress<transactional_trivial_map_t>();
    test_sequential_updates_never_regress<transactional_tracking_map_t>();
}

TEST(transactional_consistency, transaction_commits_maintain_order) {
    test_transaction_commits_maintain_order<transactional_trivial_map_t>();
    test_transaction_commits_maintain_order<transactional_tracking_map_t>();
    test_transaction_commits_maintain_order<transactional_composite_map_t>();
    test_transaction_commits_maintain_order<transactional_heavy_map_t>();
}

TEST(transactional_consistency, concurrent_transactions_on_same_key) {
    test_concurrent_transactions_on_same_key<transactional_trivial_map_t>();
    test_concurrent_transactions_on_same_key<transactional_tracking_map_t>();
    test_concurrent_transactions_on_same_key<transactional_composite_map_t>();
    test_concurrent_transactions_on_same_key<transactional_heavy_map_t>();
}

TEST(transactional_consistency, multi_key_conflict_any_key_fails) {
    test_multi_key_conflict_any_key_fails<transactional_trivial_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_tracking_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_composite_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_heavy_map_t>();
}

TEST(transactional_consistency, watch_detects_external_direct_modification) {
    test_watch_detects_external_direct_modification<transactional_trivial_map_t>();
    test_watch_detects_external_direct_modification<transactional_tracking_map_t>();
    test_watch_detects_external_direct_modification<transactional_composite_map_t>();
    test_watch_detects_external_direct_modification<transactional_heavy_map_t>();
}

TEST(transactional_consistency, watch_detects_staged_invisible_writes) {
    test_watch_detects_staged_invisible_writes<transactional_trivial_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_tracking_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_composite_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_heavy_map_t>();
}

TEST(transactional_consistency, disjoint_keys_both_succeed) {
    test_disjoint_keys_both_succeed<transactional_trivial_map_t>();
    test_disjoint_keys_both_succeed<transactional_tracking_map_t>();
    test_disjoint_keys_both_succeed<transactional_composite_map_t>();
    test_disjoint_keys_both_succeed<transactional_heavy_map_t>();
}

TEST(transactional_consistency, non_repeatable_reads_are_allowed) {
    test_non_repeatable_reads_are_allowed<transactional_trivial_map_t>();
    test_non_repeatable_reads_are_allowed<transactional_tracking_map_t>();
    test_non_repeatable_reads_are_allowed<transactional_composite_map_t>();
    test_non_repeatable_reads_are_allowed<transactional_heavy_map_t>();
}

TEST(transactional_consistency, phantom_reads_are_allowed) {
    test_phantom_reads_are_allowed<transactional_trivial_set_t>();
    test_phantom_reads_are_allowed<transactional_tracking_set_t>();
    test_phantom_reads_are_allowed<transactional_composite_set_t>();
    test_phantom_reads_are_allowed<transactional_heavy_set_t>();
    test_phantom_reads_are_allowed<transactional_trivial_map_t>();
    test_phantom_reads_are_allowed<transactional_tracking_map_t>();
    test_phantom_reads_are_allowed<transactional_composite_map_t>();
    test_phantom_reads_are_allowed<transactional_heavy_map_t>();
}

TEST(transactional_consistency, delete_visibility) {
    test_delete_visibility<transactional_trivial_set_t>();
    test_delete_visibility<transactional_tracking_set_t>();
    test_delete_visibility<transactional_composite_set_t>();
    test_delete_visibility<transactional_heavy_set_t>();
}

TEST(transactional_consistency, reset_clears_transaction_state) {
    test_reset_clears_transaction_state<transactional_trivial_map_t>();
    test_reset_clears_transaction_state<transactional_tracking_map_t>();
    test_reset_clears_transaction_state<transactional_composite_map_t>();
    test_reset_clears_transaction_state<transactional_heavy_map_t>();
}
