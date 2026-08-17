/**
 *  @brief Test instantiations for the open-addressing hash table. Covers sets and maps over trivial and
 *      heap-allocating key and value types, and the lock-free atomic operations exercised from several threads.
 *  @author Ash Vardanian
 *  @file scripts/test_hash_table.cpp
 *  @date August 16, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <smashtable/basic_hash_table.hpp>
#include <smashtable/concurrent_hash_table.hpp>

#include "test.hpp"
#include "test_unordered.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

#pragma region Type Aliases

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Values: ✗
 *  Tests: Baseline probing, growth and tombstone reuse on the cheapest possible key
 */
using trivial_set_t = hash_set<std::size_t>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Values: ✗
 *  Tests: A strongly-typed key reaching the table through its @c std::hash specialization
 */
using strong_set_t = hash_set<trivial_key_t>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Values: @c std::size_t
 *  Tests: Baseline map operations, value overwrites, the whole atomic surface
 */
using trivial_map_t = hash_map<std::size_t, std::size_t>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Non-trivial value | Memory: Stack | Values: @c guarded_payload_t
 *  Tests: Value construction, destruction and move paths under a lifecycle-checking payload
 */
using guarded_map_t = hash_map<std::size_t, guarded_payload_t>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: Heap | Memory: Heap | Values: ✗
 *  Tests: Non-trivial key destructors, the non-memcpy rehash path
 */
using string_set_t = hash_set<std::string>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: Heap | Memory: Heap | Values: @c std::string
 *  Tests: Dual-heap lifetimes, heterogeneous lookup, atomics over non-trivial elements
 */
using string_map_t = hash_map<std::string, std::string>;

#pragma endregion Type Aliases

#pragma region Basic Operations Tests

/** @brief Tests that reads and teardown tolerate an empty container */
static void unordered_ops_empty_container_operations() {
    test_unordered_empty_container_operations<trivial_set_t>();
    test_unordered_empty_container_operations<strong_set_t>();
    test_unordered_empty_container_operations<trivial_map_t>();
    test_unordered_empty_container_operations<guarded_map_t>();
    test_unordered_empty_container_operations<string_set_t>();
    test_unordered_empty_container_operations<string_map_t>();
}

/** @brief Tests the insert, lookup, overwrite and erase cycle on a single element */
static void unordered_ops_single_element_operations() {
    test_unordered_single_element_operations<trivial_set_t>();
    test_unordered_single_element_operations<strong_set_t>();
    test_unordered_single_element_operations<trivial_map_t>();
    test_unordered_single_element_operations<guarded_map_t>();
    test_unordered_single_element_operations<string_set_t>();
    test_unordered_single_element_operations<string_map_t>();
}

/** @brief Tests that growth from an unallocated table preserves every element across many rehashes */
static void unordered_ops_growth_through_rehashes() {
    test_unordered_growth_through_rehashes<trivial_set_t>();
    test_unordered_growth_through_rehashes<strong_set_t>();
    test_unordered_growth_through_rehashes<trivial_map_t>();
    test_unordered_growth_through_rehashes<guarded_map_t>();
    // ! The string configurations rehash whole heap-allocated keys, so they run a smaller sweep
    test_unordered_growth_through_rehashes<string_set_t>(1500);
    test_unordered_growth_through_rehashes<string_map_t>(1500);
}

/** @brief Tests that a tombstoned table still finds its survivors, then refills the tombstones */
static void unordered_ops_tombstone_reuse() {
    test_unordered_tombstone_reuse<trivial_set_t>();
    test_unordered_tombstone_reuse<strong_set_t>();
    test_unordered_tombstone_reuse<trivial_map_t>();
    test_unordered_tombstone_reuse<guarded_map_t>();
    test_unordered_tombstone_reuse<string_set_t>(800);
    test_unordered_tombstone_reuse<string_map_t>(800);
}

/** @brief Tests that iteration and for_each each visit every live element exactly once */
static void unordered_ops_full_iteration() {
    test_unordered_full_iteration<trivial_set_t>();
    test_unordered_full_iteration<strong_set_t>();
    test_unordered_full_iteration<trivial_map_t>();
    test_unordered_full_iteration<guarded_map_t>();
    test_unordered_full_iteration<string_set_t>(800);
    test_unordered_full_iteration<string_map_t>(800);
}

/** @brief Tests move construction, move assignment and swap */
static void unordered_ops_moves_and_swaps() {
    test_unordered_moves_and_swaps<trivial_set_t>();
    test_unordered_moves_and_swaps<strong_set_t>();
    test_unordered_moves_and_swaps<trivial_map_t>();
    test_unordered_moves_and_swaps<guarded_map_t>();
    test_unordered_moves_and_swaps<string_set_t>();
    test_unordered_moves_and_swaps<string_map_t>();
}

/** @brief Tests reserve idempotence, explicit rehash, clear and shrink_to_fit */
static void unordered_ops_capacity_management() {
    test_unordered_capacity_management<trivial_set_t>();
    test_unordered_capacity_management<strong_set_t>();
    test_unordered_capacity_management<trivial_map_t>();
    test_unordered_capacity_management<guarded_map_t>();
    test_unordered_capacity_management<string_set_t>();
    test_unordered_capacity_management<string_map_t>();
}

/** @brief Tests that the load factor stays under the documented cap and the counters stay honest */
static void unordered_ops_load_factor_consistency() {
    test_unordered_load_factor_consistency<trivial_set_t>();
    test_unordered_load_factor_consistency<strong_set_t>();
    test_unordered_load_factor_consistency<trivial_map_t>();
    test_unordered_load_factor_consistency<guarded_map_t>();
    test_unordered_load_factor_consistency<string_set_t>(1000);
    test_unordered_load_factor_consistency<string_map_t>(1000);
}

/** @brief Tests lookup by string_view, which only the string-keyed configurations support */
static void unordered_ops_heterogeneous_lookups() {
    test_unordered_heterogeneous_string_view_lookup<string_set_t>();
    test_unordered_heterogeneous_string_view_lookup<string_map_t>();
}

#pragma endregion Basic Operations Tests

#pragma region Concurrency Tests

/** @brief Tests concurrent emplace on a pinned table, then concurrent find and contains */
static void unordered_concurrency_emplace_and_find() {
    test_unordered_concurrent_emplace_and_find<trivial_map_t>();
    test_unordered_concurrent_emplace_and_find<guarded_map_t>();
    test_unordered_concurrent_emplace_and_find<string_map_t>(500);
}

/** @brief Tests concurrent update over existing keys, then concurrent erase */
static void unordered_concurrency_update_and_erase() {
    test_unordered_concurrent_update_and_erase<trivial_map_t>();
    test_unordered_concurrent_update_and_erase<guarded_map_t>();
    test_unordered_concurrent_update_and_erase<string_map_t>(500);
}

#pragma endregion Concurrency Tests

int main() {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "unordered_ops.empty_container_operations", unordered_ops_empty_container_operations);
    failures += run_test(filter, "unordered_ops.single_element_operations", unordered_ops_single_element_operations);
    failures += run_test(filter, "unordered_ops.growth_through_rehashes", unordered_ops_growth_through_rehashes);
    failures += run_test(filter, "unordered_ops.tombstone_reuse", unordered_ops_tombstone_reuse);
    failures += run_test(filter, "unordered_ops.full_iteration", unordered_ops_full_iteration);
    failures += run_test(filter, "unordered_ops.moves_and_swaps", unordered_ops_moves_and_swaps);
    failures += run_test(filter, "unordered_ops.capacity_management", unordered_ops_capacity_management);
    failures += run_test(filter, "unordered_ops.load_factor_consistency", unordered_ops_load_factor_consistency);
    failures += run_test(filter, "unordered_ops.heterogeneous_lookups", unordered_ops_heterogeneous_lookups);

    failures += run_test(filter, "unordered_concurrency.emplace_and_find", unordered_concurrency_emplace_and_find);
    failures += run_test(filter, "unordered_concurrency.update_and_erase", unordered_concurrency_update_and_erase);

    return report_test_failures(failures);
}
