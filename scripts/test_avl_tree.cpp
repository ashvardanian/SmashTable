/**
 *  @brief Test instantiations for AVL tree containers.
 *    Covers basic_avl_tree (non-transactional) and transactional_binary_tree<basic_avl_tree> (transactional).
 *    Includes AVL-specific algorithms (merge/split/join) and transaction architecture tests.
 *
 *  @file test_avl_tree.cpp
 *  @date October 25, 2025
 *  @author Ash Vardanian
 */
#include "test_basic.hpp"
#include "test_consistency.hpp"

#include <smashtable/basic_avl_tree.hpp>

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
    avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<association<trivial_key_t, int>>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial (key & value) | Memory: Tracked | Transaction: ✗
 *  Value: int | Tests: Map resource accounting, POCCA/POCMA on key-value pairs
 */
using tracking_map_t = avl_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Key trivial, value .copy() | Memory: Heap (value) | Transaction: ✗
 *  Value: basic_vector<int> | Tests: Mixed trivial/non-trivial, value OOM scenarios
 */
using composite_map_t = avl_map<composite_key_t, basic_vector<int>, composite_key_compare_t,
                                std::allocator<association<composite_key_t, basic_vector<int>>>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() on key & value | Memory: Heap (both) | Transaction: ✗
 *  Value: basic_vector<int> | Tests: Dual-heap OOM, worst-case complexity
 */
using heavy_map_t = avl_map<heavy_key_t, basic_vector<int>, std::less<void>,
                            std::allocator<association<heavy_key_t, basic_vector<int>>>>;

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
using transactional_trivial_map_t = transactional_avl_map<trivial_key_t, int, std::less<trivial_key_t>,
                                                          std::allocator<association<trivial_key_t, int>>>;

/**
 *  Heterogeneous lookup: ✓ | Copy: Trivial (key & value) | Memory: Tracked | Transaction: ✓
 *  Value: int | Tests: Transaction allocation patterns, map POCCA/POCMA
 */
using transactional_tracking_map_t =
    transactional_avl_map<trivial_key_t, int, stateful_comparator_t, stateful_allocator_t>;

/**
 *  Heterogeneous lookup: ✓ (uint64_t) | Copy: Key trivial, value .copy() | Memory: Heap (value) | Transaction: ✓
 *  Value: basic_vector<int> | Tests: Transaction rollback with non-trivial values
 */
using transactional_composite_map_t =
    transactional_avl_map<composite_key_t, basic_vector<int>, composite_key_compare_t,
                          std::allocator<association<composite_key_t, basic_vector<int>>>>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() on key & value | Memory: Heap (both) | Transaction: ✓
 *  Value: basic_vector<int> | Tests: Worst-case transactional complexity, dual-heap rollback
 */
using transactional_heavy_map_t = transactional_avl_map<heavy_key_t, basic_vector<int>, std::less<void>,
                                                        std::allocator<association<heavy_key_t, basic_vector<int>>>>;

#pragma mark - Basic Operations Tests

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

/** @brief Tests bulk insert_or_assign correctly overwrites duplicate keys */
TEST(basic_ops, bulk_upsert_with_duplicates) {
    test_bulk_insert_or_assign_with_duplicates<trivial_set_t>();
    test_bulk_insert_or_assign_with_duplicates<tracking_set_t>();
    test_bulk_insert_or_assign_with_duplicates<composite_set_t>();
    test_bulk_insert_or_assign_with_duplicates<heavy_set_t>();
    test_bulk_insert_or_assign_with_duplicates<trivial_map_t>();
    test_bulk_insert_or_assign_with_duplicates<tracking_map_t>();
    test_bulk_insert_or_assign_with_duplicates<composite_map_t>();
    test_bulk_insert_or_assign_with_duplicates<heavy_map_t>();
    test_bulk_insert_or_assign_with_duplicates<transactional_trivial_set_t>();
    test_bulk_insert_or_assign_with_duplicates<transactional_tracking_set_t>();
    test_bulk_insert_or_assign_with_duplicates<transactional_composite_set_t>();
    test_bulk_insert_or_assign_with_duplicates<transactional_heavy_set_t>();
    test_bulk_insert_or_assign_with_duplicates<transactional_trivial_map_t>();
    test_bulk_insert_or_assign_with_duplicates<transactional_tracking_map_t>();
    test_bulk_insert_or_assign_with_duplicates<transactional_composite_map_t>();
    test_bulk_insert_or_assign_with_duplicates<transactional_heavy_map_t>();
}

/** @brief Tests range queries on committed HEAD state */
TEST(basic_ops, range_query_head_state) {
    test_range_query_head_state<trivial_set_t>();
    test_range_query_head_state<tracking_set_t>();
    test_range_query_head_state<composite_set_t>();
    test_range_query_head_state<heavy_set_t>();
    test_range_query_head_state<trivial_map_t>();
    test_range_query_head_state<tracking_map_t>();
    test_range_query_head_state<composite_map_t>();
    test_range_query_head_state<heavy_map_t>();
    test_range_query_head_state<transactional_trivial_set_t>();
    test_range_query_head_state<transactional_tracking_set_t>();
    test_range_query_head_state<transactional_composite_set_t>();
    test_range_query_head_state<transactional_heavy_set_t>();
    test_range_query_head_state<transactional_trivial_map_t>();
    test_range_query_head_state<transactional_tracking_map_t>();
    test_range_query_head_state<transactional_composite_map_t>();
    test_range_query_head_state<transactional_heavy_map_t>();
}

/** @brief Tests erase_range on committed HEAD state */
TEST(basic_ops, erase_range_head_state) {
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

/** @brief Tests upper_bound query functionality */
TEST(basic_ops, upper_bound) {
    test_upper_bound<trivial_set_t>();
    test_upper_bound<tracking_set_t>();
    test_upper_bound<composite_set_t>();
    test_upper_bound<heavy_set_t>();
    test_upper_bound<trivial_map_t>();
    test_upper_bound<tracking_map_t>();
    test_upper_bound<composite_map_t>();
    test_upper_bound<heavy_map_t>();
    test_upper_bound<transactional_trivial_set_t>();
    test_upper_bound<transactional_tracking_set_t>();
    test_upper_bound<transactional_composite_set_t>();
    test_upper_bound<transactional_heavy_set_t>();
    test_upper_bound<transactional_trivial_map_t>();
    test_upper_bound<transactional_tracking_map_t>();
    test_upper_bound<transactional_composite_map_t>();
    test_upper_bound<transactional_heavy_map_t>();
}

/** @brief Tests clear operation */
TEST(basic_ops, clear) {
    test_clear<trivial_set_t>();
    test_clear<tracking_set_t>();
    test_clear<composite_set_t>();
    test_clear<heavy_set_t>();
    test_clear<trivial_map_t>();
    test_clear<tracking_map_t>();
    test_clear<composite_map_t>();
    test_clear<heavy_map_t>();
    test_clear<transactional_trivial_set_t>();
    test_clear<transactional_tracking_set_t>();
    test_clear<transactional_composite_set_t>();
    test_clear<transactional_heavy_set_t>();
    test_clear<transactional_trivial_map_t>();
    test_clear<transactional_tracking_map_t>();
    test_clear<transactional_composite_map_t>();
    test_clear<transactional_heavy_map_t>();
}

/** @brief Tests size increments correctly after upserts */
TEST(basic_ops, size_after_upserts) {
    test_size_after_upserts<trivial_set_t>();
    test_size_after_upserts<tracking_set_t>();
    test_size_after_upserts<composite_set_t>();
    test_size_after_upserts<heavy_set_t>();
    test_size_after_upserts<trivial_map_t>();
    test_size_after_upserts<tracking_map_t>();
    test_size_after_upserts<composite_map_t>();
    test_size_after_upserts<heavy_map_t>();
    test_size_after_upserts<transactional_trivial_set_t>();
    test_size_after_upserts<transactional_tracking_set_t>();
    test_size_after_upserts<transactional_composite_set_t>();
    test_size_after_upserts<transactional_heavy_set_t>();
    test_size_after_upserts<transactional_trivial_map_t>();
    test_size_after_upserts<transactional_tracking_map_t>();
    test_size_after_upserts<transactional_composite_map_t>();
    test_size_after_upserts<transactional_heavy_map_t>();
}

/** @brief Tests size invariant: duplicate upserts don't increment size */
TEST(basic_ops, size_invariant_on_duplicates) {
    test_size_invariant_on_duplicate_upserts<trivial_set_t>();
    test_size_invariant_on_duplicate_upserts<tracking_set_t>();
    test_size_invariant_on_duplicate_upserts<composite_set_t>();
    test_size_invariant_on_duplicate_upserts<heavy_set_t>();
    test_size_invariant_on_duplicate_upserts<trivial_map_t>();
    test_size_invariant_on_duplicate_upserts<tracking_map_t>();
    test_size_invariant_on_duplicate_upserts<composite_map_t>();
    test_size_invariant_on_duplicate_upserts<heavy_map_t>();
    test_size_invariant_on_duplicate_upserts<transactional_trivial_set_t>();
    test_size_invariant_on_duplicate_upserts<transactional_tracking_set_t>();
    test_size_invariant_on_duplicate_upserts<transactional_composite_set_t>();
    test_size_invariant_on_duplicate_upserts<transactional_heavy_set_t>();
    test_size_invariant_on_duplicate_upserts<transactional_trivial_map_t>();
    test_size_invariant_on_duplicate_upserts<transactional_tracking_map_t>();
    test_size_invariant_on_duplicate_upserts<transactional_composite_map_t>();
    test_size_invariant_on_duplicate_upserts<transactional_heavy_map_t>();
}

/** @brief Tests size decrements correctly after erase_range */
TEST(basic_ops, size_after_erase) {
    test_size_after_erase<trivial_set_t>();
    test_size_after_erase<tracking_set_t>();
    test_size_after_erase<composite_set_t>();
    test_size_after_erase<heavy_set_t>();
    test_size_after_erase<trivial_map_t>();
    test_size_after_erase<tracking_map_t>();
    test_size_after_erase<composite_map_t>();
    test_size_after_erase<heavy_map_t>();
    test_size_after_erase<transactional_trivial_set_t>();
    test_size_after_erase<transactional_tracking_set_t>();
    test_size_after_erase<transactional_composite_set_t>();
    test_size_after_erase<transactional_heavy_set_t>();
    test_size_after_erase<transactional_trivial_map_t>();
    test_size_after_erase<transactional_tracking_map_t>();
    test_size_after_erase<transactional_composite_map_t>();
    test_size_after_erase<transactional_heavy_map_t>();
}

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

/** @brief Tests heterogeneous lookup for composite and heavy key types */
TEST(basic_ops, heterogeneous_lookups) {
    test_heterogeneous_composite_find<composite_set_t>();
    test_heterogeneous_heavy_string_view_find<heavy_set_t>();
    test_heterogeneous_heavy_integer_find<heavy_set_t>();
    test_heterogeneous_composite_find<composite_map_t>();
    test_heterogeneous_heavy_string_view_find<heavy_map_t>();
    test_heterogeneous_heavy_integer_find<heavy_map_t>();
    test_heterogeneous_composite_find<transactional_composite_set_t>();
    test_heterogeneous_heavy_string_view_find<transactional_heavy_set_t>();
    test_heterogeneous_heavy_integer_find<transactional_heavy_set_t>();
    test_heterogeneous_composite_find<transactional_composite_map_t>();
    test_heterogeneous_heavy_string_view_find<transactional_heavy_map_t>();
    test_heterogeneous_heavy_integer_find<transactional_heavy_map_t>();
}

/** @brief Tests AVL split/join operations with erase_range edge cases */
TEST(basic_ops, erase_range_edge_cases) {
    test_erase_range_edge_cases<trivial_set_t>();
    test_erase_range_edge_cases<tracking_set_t>();
    test_erase_range_edge_cases<composite_set_t>();
    test_erase_range_edge_cases<heavy_set_t>();
    test_erase_range_edge_cases<trivial_map_t>();
    test_erase_range_edge_cases<tracking_map_t>();
    test_erase_range_edge_cases<composite_map_t>();
    test_erase_range_edge_cases<heavy_map_t>();
    test_erase_range_edge_cases<transactional_trivial_set_t>();
    test_erase_range_edge_cases<transactional_tracking_set_t>();
    test_erase_range_edge_cases<transactional_composite_set_t>();
    test_erase_range_edge_cases<transactional_heavy_set_t>();
    test_erase_range_edge_cases<transactional_trivial_map_t>();
    test_erase_range_edge_cases<transactional_tracking_map_t>();
    test_erase_range_edge_cases<transactional_composite_map_t>();
    test_erase_range_edge_cases<transactional_heavy_map_t>();
}

/** @brief Tests erase_range with large ranges uses split/join efficiently */
TEST(basic_ops, erase_range_large) {
    test_erase_range_large<trivial_set_t>();
    test_erase_range_large<tracking_set_t>();
    test_erase_range_large<composite_set_t>();
    test_erase_range_large<heavy_set_t>();
    test_erase_range_large<trivial_map_t>();
    test_erase_range_large<tracking_map_t>();
    test_erase_range_large<composite_map_t>();
    test_erase_range_large<heavy_map_t>();
    test_erase_range_large<transactional_trivial_set_t>();
    test_erase_range_large<transactional_tracking_set_t>();
    test_erase_range_large<transactional_composite_set_t>();
    test_erase_range_large<transactional_heavy_set_t>();
    test_erase_range_large<transactional_trivial_map_t>();
    test_erase_range_large<transactional_tracking_map_t>();
    test_erase_range_large<transactional_composite_map_t>();
    test_erase_range_large<transactional_heavy_map_t>();
}

/** @brief Tests erase_range maintains AVL balance property */
TEST(basic_ops, erase_range_maintains_balance) {
    test_erase_range_maintains_balance<trivial_set_t>();
    test_erase_range_maintains_balance<tracking_set_t>();
    test_erase_range_maintains_balance<composite_set_t>();
    test_erase_range_maintains_balance<heavy_set_t>();
    test_erase_range_maintains_balance<trivial_map_t>();
    test_erase_range_maintains_balance<tracking_map_t>();
    test_erase_range_maintains_balance<composite_map_t>();
    test_erase_range_maintains_balance<heavy_map_t>();
    test_erase_range_maintains_balance<transactional_trivial_set_t>();
    test_erase_range_maintains_balance<transactional_tracking_set_t>();
    test_erase_range_maintains_balance<transactional_composite_set_t>();
    test_erase_range_maintains_balance<transactional_heavy_set_t>();
    test_erase_range_maintains_balance<transactional_trivial_map_t>();
    test_erase_range_maintains_balance<transactional_tracking_map_t>();
    test_erase_range_maintains_balance<transactional_composite_map_t>();
    test_erase_range_maintains_balance<transactional_heavy_map_t>();
}

#pragma mark - Consistency & Transaction Tests: Sets

TEST(consistency_transactional_sets, no_dirty_reads_multi_key) {
    test_no_dirty_reads_multi_key<transactional_trivial_set_t>();
    test_no_dirty_reads_multi_key<transactional_tracking_set_t>();
    test_no_dirty_reads_multi_key<transactional_composite_set_t>();
    test_no_dirty_reads_multi_key<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, new_transaction_sees_nothing_staged) {
    test_new_transaction_sees_nothing_staged<transactional_trivial_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_tracking_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, committed_immediately_visible) {
    test_committed_immediately_visible<transactional_trivial_set_t>();
    test_committed_immediately_visible<transactional_tracking_set_t>();
    test_committed_immediately_visible<transactional_composite_set_t>();
    test_committed_immediately_visible<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, multi_key_atomicity_10_keys) {
    test_multi_key_atomicity_10_keys<transactional_trivial_set_t>();
    test_multi_key_atomicity_10_keys<transactional_tracking_set_t>();
    test_multi_key_atomicity_10_keys<transactional_composite_set_t>();
    test_multi_key_atomicity_10_keys<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, rollback_makes_all_invisible) {
    test_rollback_makes_all_invisible<transactional_trivial_set_t>();
    test_rollback_makes_all_invisible<transactional_tracking_set_t>();
    test_rollback_makes_all_invisible<transactional_composite_set_t>();
    test_rollback_makes_all_invisible<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, range_query_sees_atomic_boundaries) {
    test_range_query_sees_atomic_boundaries<transactional_trivial_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_tracking_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_composite_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, fractured_read_prevention) {
    test_fractured_read_prevention<transactional_trivial_set_t>();
    test_fractured_read_prevention<transactional_tracking_set_t>();
    test_fractured_read_prevention<transactional_composite_set_t>();
    test_fractured_read_prevention<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, sequential_updates_never_regress) {
    test_sequential_updates_never_regress<transactional_trivial_set_t>();
    test_sequential_updates_never_regress<transactional_tracking_set_t>();
    test_sequential_updates_never_regress<transactional_composite_set_t>();
    test_sequential_updates_never_regress<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, transaction_commits_maintain_order) {
    test_transaction_commits_maintain_order<transactional_trivial_set_t>();
    test_transaction_commits_maintain_order<transactional_tracking_set_t>();
    test_transaction_commits_maintain_order<transactional_composite_set_t>();
    test_transaction_commits_maintain_order<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, concurrent_transactions_on_same_key) {
    test_concurrent_transactions_on_same_key<transactional_trivial_set_t>();
    test_concurrent_transactions_on_same_key<transactional_tracking_set_t>();
    test_concurrent_transactions_on_same_key<transactional_composite_set_t>();
    test_concurrent_transactions_on_same_key<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, multi_key_conflict_any_key_fails) {
    test_multi_key_conflict_any_key_fails<transactional_trivial_set_t>();
    test_multi_key_conflict_any_key_fails<transactional_tracking_set_t>();
    test_multi_key_conflict_any_key_fails<transactional_composite_set_t>();
    test_multi_key_conflict_any_key_fails<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, watch_detects_external_direct_modification) {
    test_watch_detects_external_direct_modification<transactional_trivial_set_t>();
    test_watch_detects_external_direct_modification<transactional_tracking_set_t>();
    test_watch_detects_external_direct_modification<transactional_composite_set_t>();
    test_watch_detects_external_direct_modification<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, watch_detects_staged_invisible_writes) {
    test_watch_detects_staged_invisible_writes<transactional_trivial_set_t>();
    test_watch_detects_staged_invisible_writes<transactional_tracking_set_t>();
    test_watch_detects_staged_invisible_writes<transactional_composite_set_t>();
    test_watch_detects_staged_invisible_writes<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, disjoint_keys_both_succeed) {
    test_disjoint_keys_both_succeed<transactional_trivial_set_t>();
    test_disjoint_keys_both_succeed<transactional_tracking_set_t>();
    test_disjoint_keys_both_succeed<transactional_composite_set_t>();
    test_disjoint_keys_both_succeed<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, non_repeatable_reads_are_allowed) {
    test_non_repeatable_reads_are_allowed<transactional_trivial_set_t>();
    test_non_repeatable_reads_are_allowed<transactional_tracking_set_t>();
    test_non_repeatable_reads_are_allowed<transactional_composite_set_t>();
    test_non_repeatable_reads_are_allowed<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, phantom_reads_are_allowed) {
    test_phantom_reads_are_allowed<transactional_trivial_set_t>();
    test_phantom_reads_are_allowed<transactional_tracking_set_t>();
    test_phantom_reads_are_allowed<transactional_composite_set_t>();
    test_phantom_reads_are_allowed<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, empty_transaction_commit) {
    test_empty_transaction_commit<transactional_trivial_set_t>();
    test_empty_transaction_commit<transactional_tracking_set_t>();
    test_empty_transaction_commit<transactional_composite_set_t>();
    test_empty_transaction_commit<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, delete_visibility) {
    test_delete_visibility<transactional_trivial_set_t>();
    test_delete_visibility<transactional_tracking_set_t>();
    test_delete_visibility<transactional_composite_set_t>();
    test_delete_visibility<transactional_heavy_set_t>();
}

TEST(consistency_transactional_sets, reset_clears_transaction_state) {
    test_reset_clears_transaction_state<transactional_trivial_set_t>();
    test_reset_clears_transaction_state<transactional_tracking_set_t>();
    test_reset_clears_transaction_state<transactional_composite_set_t>();
    test_reset_clears_transaction_state<transactional_heavy_set_t>();
}

#pragma mark - Consistency & Transaction Tests: Maps

TEST(consistency_transactional_maps, no_dirty_reads_multi_key) {
    test_no_dirty_reads_multi_key<transactional_trivial_map_t>();
    test_no_dirty_reads_multi_key<transactional_tracking_map_t>();
    test_no_dirty_reads_multi_key<transactional_composite_map_t>();
    test_no_dirty_reads_multi_key<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, new_transaction_sees_nothing_staged) {
    test_new_transaction_sees_nothing_staged<transactional_trivial_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_tracking_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, committed_immediately_visible) {
    test_committed_immediately_visible<transactional_trivial_map_t>();
    test_committed_immediately_visible<transactional_tracking_map_t>();
    test_committed_immediately_visible<transactional_composite_map_t>();
    test_committed_immediately_visible<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, multi_key_atomicity_10_keys) {
    test_multi_key_atomicity_10_keys<transactional_trivial_map_t>();
    test_multi_key_atomicity_10_keys<transactional_tracking_map_t>();
    test_multi_key_atomicity_10_keys<transactional_composite_map_t>();
    test_multi_key_atomicity_10_keys<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, rollback_makes_all_invisible) {
    test_rollback_makes_all_invisible<transactional_trivial_map_t>();
    test_rollback_makes_all_invisible<transactional_tracking_map_t>();
    test_rollback_makes_all_invisible<transactional_composite_map_t>();
    test_rollback_makes_all_invisible<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, range_query_sees_atomic_boundaries) {
    test_range_query_sees_atomic_boundaries<transactional_trivial_map_t>();
    test_range_query_sees_atomic_boundaries<transactional_tracking_map_t>();
    test_range_query_sees_atomic_boundaries<transactional_composite_map_t>();
    test_range_query_sees_atomic_boundaries<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, fractured_read_prevention) {
    test_fractured_read_prevention<transactional_trivial_map_t>();
    test_fractured_read_prevention<transactional_tracking_map_t>();
    test_fractured_read_prevention<transactional_composite_map_t>();
    test_fractured_read_prevention<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, sequential_updates_never_regress) {
    test_sequential_updates_never_regress<transactional_trivial_map_t>();
    test_sequential_updates_never_regress<transactional_tracking_map_t>();
    test_sequential_updates_never_regress<transactional_composite_map_t>();
    test_sequential_updates_never_regress<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, transaction_commits_maintain_order) {
    test_transaction_commits_maintain_order<transactional_trivial_map_t>();
    test_transaction_commits_maintain_order<transactional_tracking_map_t>();
    test_transaction_commits_maintain_order<transactional_composite_map_t>();
    test_transaction_commits_maintain_order<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, concurrent_transactions_on_same_key) {
    test_concurrent_transactions_on_same_key<transactional_trivial_map_t>();
    test_concurrent_transactions_on_same_key<transactional_tracking_map_t>();
    test_concurrent_transactions_on_same_key<transactional_composite_map_t>();
    test_concurrent_transactions_on_same_key<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, multi_key_conflict_any_key_fails) {
    test_multi_key_conflict_any_key_fails<transactional_trivial_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_tracking_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_composite_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, watch_detects_external_direct_modification) {
    test_watch_detects_external_direct_modification<transactional_trivial_map_t>();
    test_watch_detects_external_direct_modification<transactional_tracking_map_t>();
    test_watch_detects_external_direct_modification<transactional_composite_map_t>();
    test_watch_detects_external_direct_modification<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, watch_detects_staged_invisible_writes) {
    test_watch_detects_staged_invisible_writes<transactional_trivial_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_tracking_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_composite_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, disjoint_keys_both_succeed) {
    test_disjoint_keys_both_succeed<transactional_trivial_map_t>();
    test_disjoint_keys_both_succeed<transactional_tracking_map_t>();
    test_disjoint_keys_both_succeed<transactional_composite_map_t>();
    test_disjoint_keys_both_succeed<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, non_repeatable_reads_are_allowed) {
    test_non_repeatable_reads_are_allowed<transactional_trivial_map_t>();
    test_non_repeatable_reads_are_allowed<transactional_tracking_map_t>();
    test_non_repeatable_reads_are_allowed<transactional_composite_map_t>();
    test_non_repeatable_reads_are_allowed<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, phantom_reads_are_allowed) {
    test_phantom_reads_are_allowed<transactional_trivial_map_t>();
    test_phantom_reads_are_allowed<transactional_tracking_map_t>();
    test_phantom_reads_are_allowed<transactional_composite_map_t>();
    test_phantom_reads_are_allowed<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, empty_transaction_commit) {
    test_empty_transaction_commit<transactional_trivial_map_t>();
    test_empty_transaction_commit<transactional_tracking_map_t>();
    test_empty_transaction_commit<transactional_composite_map_t>();
    test_empty_transaction_commit<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, delete_visibility) {
    test_delete_visibility<transactional_trivial_map_t>();
    test_delete_visibility<transactional_tracking_map_t>();
    test_delete_visibility<transactional_composite_map_t>();
    test_delete_visibility<transactional_heavy_map_t>();
}

TEST(consistency_transactional_maps, reset_clears_transaction_state) {
    test_reset_clears_transaction_state<transactional_trivial_map_t>();
    test_reset_clears_transaction_state<transactional_tracking_map_t>();
    test_reset_clears_transaction_state<transactional_composite_map_t>();
    test_reset_clears_transaction_state<transactional_heavy_map_t>();
}

#pragma mark - AVL Tree Merge Tests (Non-Transactional basic_avl_tree)

using basic_avl_t = basic_avl_tree<std::size_t, std::less<std::size_t>>;

// Helper to populate tree with range [start, end)
void populate_tree(basic_avl_t &tree, std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) {
        tree.insert(std::size_t(i)); // Create rvalue temporary
    }
}

// Helper to verify tree contains exactly the expected elements
void verify_tree_contents(basic_avl_t const &tree, std::vector<std::size_t> const &expected) {
    EXPECT_EQ(tree.size(), expected.size());
    for (auto val : expected) {
        auto it = tree.find(val);
        EXPECT_NE(it, tree.end()) << "Expected to find " << val;
    }
}

TEST(avl_merge, default_merge_with_duplicates) {
    basic_avl_t tree1, tree2;

    // tree1: [0, 10)
    populate_tree(tree1, 0, 10);

    // tree2: [5, 15) - overlaps with tree1
    populate_tree(tree2, 5, 15);

    EXPECT_EQ(tree1.size(), 10u);
    EXPECT_EQ(tree2.size(), 10u);

    // Merge - duplicates should be discarded
    tree1.merge(tree2);

    // tree1 should have union: [0, 15)
    EXPECT_EQ(tree1.size(), 15u);
    EXPECT_EQ(tree2.size(), 0u);

    // Verify all elements present
    for (std::size_t i = 0; i < 15; ++i) { EXPECT_NE(tree1.find(i), tree1.end()) << "Expected to find " << i; }
}

TEST(avl_merge, assume_unique_small_trees) {
    basic_avl_t tree1, tree2;

    // tree1: [0, 10)
    populate_tree(tree1, 0, 10);

    // tree2: [10, 20) - disjoint
    populate_tree(tree2, 10, 20);

    EXPECT_EQ(tree1.size(), 10u);
    EXPECT_EQ(tree2.size(), 10u);

    // Merge with assume_unique (optimized)
    tree1.merge(tree2, assume_unique);

    EXPECT_EQ(tree1.size(), 20u);
    EXPECT_EQ(tree2.size(), 0u);

    verify_tree_contents(tree1, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19});
}

TEST(avl_merge, assume_unique_unbalanced_sizes_split_based) {
    basic_avl_t tree1, tree2;

    // Large tree: [0, 1000)
    populate_tree(tree1, 0, 1000);

    // Small tree: [1000, 1100) - only 100 elements
    populate_tree(tree2, 1000, 1100);

    EXPECT_EQ(tree1.size(), 1000u);
    EXPECT_EQ(tree2.size(), 100u);

    // Should use split-based merge: O(100 log 10) ≈ 332 operations
    tree1.merge(tree2, assume_unique);

    EXPECT_EQ(tree1.size(), 1100u);
    EXPECT_EQ(tree2.size(), 0u);

    // Spot check some values
    for (std::size_t i : {0, 500, 999, 1000, 1050, 1099}) {
        EXPECT_NE(tree1.find(i), tree1.end()) << "Expected to find " << i;
    }
}

TEST(avl_merge, assume_unique_large_similar_sizes_dsw) {
    basic_avl_t tree1, tree2;

    // Both large and similar size to trigger DSW
    // tree1: even numbers [0, 20000) step 2 = 10000 elements
    for (std::size_t i = 0; i < 20000; i += 2) { tree1.insert(std::size_t(i)); }

    // tree2: odd numbers [1, 20001) step 2 = 10000 elements
    for (std::size_t i = 1; i < 20001; i += 2) { tree2.insert(std::size_t(i)); }

    EXPECT_EQ(tree1.size(), 10000u);
    EXPECT_EQ(tree2.size(), 10000u);

    // Should use DSW: O(20000) with min_size=10000 > DSW_THRESHOLD
    tree1.merge(tree2, assume_unique);

    EXPECT_EQ(tree1.size(), 20000u);
    EXPECT_EQ(tree2.size(), 0u);

    // Verify all elements [0, 20000) are present
    for (std::size_t i : {0, 1, 9999, 10000, 19998, 19999}) {
        EXPECT_NE(tree1.find(i), tree1.end()) << "Expected to find " << i;
    }
}

TEST(avl_merge, assume_unique_fully_ordered_fast_join) {
    basic_avl_t tree1, tree2;

    // tree1: [0, 1000) - all smaller
    populate_tree(tree1, 0, 1000);

    // tree2: [1000, 2000) - all larger
    populate_tree(tree2, 1000, 2000);

    EXPECT_EQ(tree1.size(), 1000u);
    EXPECT_EQ(tree2.size(), 1000u);

    // Should detect ordering and use fast join: O(log n)
    tree1.merge(tree2, assume_unique);

    EXPECT_EQ(tree1.size(), 2000u);
    EXPECT_EQ(tree2.size(), 0u);

    // Verify ordering preserved
    std::size_t prev = 0;
    std::size_t count = 0;
    tree1.for_each([&](auto const &val) noexcept {
        if (count > 0) EXPECT_LT(prev, val);
        prev = val;
        ++count;
    });
    EXPECT_EQ(count, 2000u);
}

TEST(avl_merge, assume_unique_reverse_ordered_fast_join) {
    basic_avl_t tree1, tree2;

    // tree1: [1000, 2000) - all larger
    populate_tree(tree1, 1000, 2000);

    // tree2: [0, 1000) - all smaller
    populate_tree(tree2, 0, 1000);

    EXPECT_EQ(tree1.size(), 1000u);
    EXPECT_EQ(tree2.size(), 1000u);

    // Should detect ordering (tree2.max < tree1.min) and use fast join
    tree1.merge(tree2, assume_unique);

    EXPECT_EQ(tree1.size(), 2000u);
    EXPECT_EQ(tree2.size(), 0u);

    // Verify all elements present
    for (std::size_t i : {0, 500, 999, 1000, 1500, 1999}) {
        EXPECT_NE(tree1.find(i), tree1.end()) << "Expected to find " << i;
    }
}

TEST(avl_merge, empty_trees) {
    basic_avl_t tree1, tree2;

    // Both empty
    tree1.merge(tree2, assume_unique);
    EXPECT_EQ(tree1.size(), 0u);
    EXPECT_EQ(tree2.size(), 0u);

    // tree1 empty, tree2 has elements
    populate_tree(tree2, 0, 10);
    tree1.merge(tree2, assume_unique);
    EXPECT_EQ(tree1.size(), 10u);
    EXPECT_EQ(tree2.size(), 0u);

    // tree1 has elements, tree2 empty
    basic_avl_t tree3;
    tree1.merge(tree3, assume_unique);
    EXPECT_EQ(tree1.size(), 10u);
    EXPECT_EQ(tree3.size(), 0u);
}

TEST(avl_merge, single_element_trees) {
    basic_avl_t tree1, tree2;

    tree1.insert(std::size_t(5));
    tree2.insert(std::size_t(10));

    tree1.merge(tree2, assume_unique);

    EXPECT_EQ(tree1.size(), 2u);
    EXPECT_EQ(tree2.size(), 0u);

    verify_tree_contents(tree1, {5, 10});
}

TEST(avl_split_join, split_at_key) {
    basic_avl_t tree;
    populate_tree(tree, 0, 100);

    // Split at 50
    auto result = tree.split(50);

    EXPECT_EQ(tree.size(), 0u);          // Original emptied
    EXPECT_EQ(result.left.size(), 50u);  // [0, 50)
    EXPECT_EQ(result.right.size(), 50u); // [50, 100)

    // Verify left contains [0, 50)
    for (std::size_t i = 0; i < 50; ++i) {
        EXPECT_NE(result.left.find(i), result.left.end()) << "Expected " << i << " in left tree";
    }

    // Verify right contains [50, 100)
    for (std::size_t i = 50; i < 100; ++i) {
        EXPECT_NE(result.right.find(i), result.right.end()) << "Expected " << i << " in right tree";
    }
}

TEST(avl_split_join, join_ordered_trees) {
    basic_avl_t tree1, tree2;

    populate_tree(tree1, 0, 50);
    populate_tree(tree2, 50, 100);

    // Join with precondition: all(tree1) < all(tree2)
    tree1.join(tree2);

    EXPECT_EQ(tree1.size(), 100u);
    EXPECT_EQ(tree2.size(), 0u);

    // Verify ordering maintained
    std::size_t prev = 0;
    std::size_t count = 0;
    tree1.for_each([&](auto const &val) noexcept {
        if (count > 0) EXPECT_LT(prev, val);
        prev = val;
        ++count;
    });
    EXPECT_EQ(count, 100u);
}
