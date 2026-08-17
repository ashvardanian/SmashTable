/**
 *  @brief Test instantiations for the transactional store over an open-addressed hash table. Covers only the
 *      point-access surface - insert, upsert, update, erase, find, watch and the two-phase commit - since an
 *      unordered core supplies no bounds, ranges, or order statistics.
 *  @author Ash Vardanian
 *  @file scripts/test_transactional_hash.cpp
 *  @date August 17, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1

#include <smashtable/transactional_store.hpp>

#include "test.hpp"
#include "test_basic.hpp"
#include "test_consistency.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

#pragma region Type Aliases

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial | Memory: Stack | Transaction: ✓
 *  Tests: Baseline transactional correctness over the cheapest possible key
 */
using transactional_trivial_set_t = transactional_hash_set<trivial_key_t>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() → expected<T> | Memory: Heap | Transaction: ✓
 *  Tests: Watch copy OOM, rollback with a heap-allocating key
 */
using transactional_heavy_set_t = transactional_hash_set<heavy_key_t>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Trivial (key & value) | Memory: Stack | Transaction: ✓
 *  Value: int | Tests: Transactional map operations, value overwrites
 */
using transactional_trivial_map_t = transactional_hash_map<trivial_key_t, int>;

/**
 *  Heterogeneous lookup: ✗ | Copy: Key trivial, value .copy() | Memory: Heap (value) | Transaction: ✓
 *  Value: guarded_payload_t | Tests: Rollback with non-trivial values
 */
using transactional_composite_map_t = transactional_hash_map<composite_key_t, guarded_payload_t>;

/**
 *  Heterogeneous lookup: ✓ (string_view) | Copy: .copy() on key & value | Memory: Heap (both) | Transaction: ✓
 *  Value: guarded_payload_t | Tests: Dual-heap staging and rollback
 */
using transactional_heavy_map_t = transactional_hash_map<heavy_key_t, guarded_payload_t>;

static_assert(transactional_trivial_set_t::isolation_k == isolation_t::monotonic_atomic_view_k);

#pragma endregion Type Aliases

#pragma region Point Access Tests

/** @brief Tests that the four insert strategies differ exactly as documented */
template <typename container_type_>
static void test_point_insert_strategies() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    auto first = container.transaction();
    st_verify_(first.has_value());

    st_verify_(succeeded(first->insert(trivial_id_to_member<member_t>(1))));
    st_verify_(failed(first->insert(trivial_id_to_member<member_t>(1))));
    st_verify_(succeeded(first->insert_if_missing(trivial_id_to_member<member_t>(1))));
    st_verify_(succeeded(first->upsert(trivial_id_to_member<member_t>(1))));
    st_verify_(succeeded(first->update(trivial_id_to_member<member_t>(1))));
    st_verify_(failed(first->update(trivial_id_to_member<member_t>(2))));

    st_verify_(succeeded(first->stage()));
    st_verify_(succeeded(first->commit()));
    st_verify_eq_(container.size(), 1);
    st_verify_(container.contains(trivial_id_to_key<member_t>(1)));
    st_verify_(!container.contains(trivial_id_to_key<member_t>(2)));
}

/** @brief Tests that an erase staged in a transaction only becomes visible on commit */
template <typename container_type_>
static void test_point_erase_visibility() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (std::size_t index = 0; index < 8; ++index)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(index))));
    st_verify_eq_(container.size(), 8);

    auto erasing = container.transaction();
    st_verify_(erasing.has_value());
    for (std::size_t index = 0; index < 4; ++index)
        st_verify_(succeeded(erasing->erase(trivial_id_to_key<member_t>(index))));
    st_verify_(succeeded(erasing->stage()));
    for (std::size_t index = 0; index < 4; ++index) st_verify_(container.contains(trivial_id_to_key<member_t>(index)));

    st_verify_(succeeded(erasing->commit()));
    for (std::size_t index = 0; index < 4; ++index) st_verify_(!container.contains(trivial_id_to_key<member_t>(index)));
    for (std::size_t index = 4; index < 8; ++index) st_verify_(container.contains(trivial_id_to_key<member_t>(index)));
}

/** @brief Tests that a rollback pulls every staged version back and leaves the store as it was */
template <typename container_type_>
static void test_point_rollback_restores_store() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(1))));

    auto writing = container.transaction();
    st_verify_(writing.has_value());
    for (std::size_t index = 1; index <= 5; ++index)
        st_verify_(succeeded(writing->upsert(trivial_id_to_member<member_t>(index))));
    st_verify_(succeeded(writing->stage()));
    st_verify_(succeeded(writing->rollback()));

    st_verify_eq_(container.size(), 1);
    st_verify_(container.contains(trivial_id_to_key<member_t>(1)));
    for (std::size_t index = 2; index <= 5; ++index)
        st_verify_(!container.contains(trivial_id_to_key<member_t>(index)));

    // The rolled-back versions are still the transaction's, so a second staging lands them all
    st_verify_(succeeded(writing->stage()));
    st_verify_(succeeded(writing->commit()));
    st_verify_eq_(container.size(), 5);
}

/** @brief Tests that staging enough keys to outgrow the slab several times loses none of them */
template <typename container_type_>
static void test_point_staging_survives_growth(std::size_t size = 500) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    auto writing = container.transaction();
    st_verify_(writing.has_value());
    for (std::size_t index = 0; index < size; ++index)
        st_verify_(succeeded(writing->upsert(trivial_id_to_member<member_t>(index))));
    st_verify_(succeeded(writing->stage()));
    st_verify_(succeeded(writing->commit()));

    st_verify_eq_(container.size(), size);
    for (std::size_t index = 0; index < size; ++index)
        st_verify_(container.contains(trivial_id_to_key<member_t>(index)));
}

static void point_access_insert_strategies() {
    test_point_insert_strategies<transactional_trivial_set_t>();
    test_point_insert_strategies<transactional_heavy_set_t>();
    test_point_insert_strategies<transactional_trivial_map_t>();
    test_point_insert_strategies<transactional_composite_map_t>();
    test_point_insert_strategies<transactional_heavy_map_t>();
}

static void point_access_erase_visibility() {
    test_point_erase_visibility<transactional_trivial_set_t>();
    test_point_erase_visibility<transactional_heavy_set_t>();
    test_point_erase_visibility<transactional_trivial_map_t>();
    test_point_erase_visibility<transactional_composite_map_t>();
    test_point_erase_visibility<transactional_heavy_map_t>();
}

static void point_access_rollback_restores_store() {
    test_point_rollback_restores_store<transactional_trivial_set_t>();
    test_point_rollback_restores_store<transactional_heavy_set_t>();
    test_point_rollback_restores_store<transactional_trivial_map_t>();
    test_point_rollback_restores_store<transactional_composite_map_t>();
    test_point_rollback_restores_store<transactional_heavy_map_t>();
}

static void point_access_staging_survives_growth() {
    test_point_staging_survives_growth<transactional_trivial_set_t>();
    test_point_staging_survives_growth<transactional_heavy_set_t>();
    test_point_staging_survives_growth<transactional_trivial_map_t>();
    test_point_staging_survives_growth<transactional_composite_map_t>();
    test_point_staging_survives_growth<transactional_heavy_map_t>();
}

static void point_access_insertion_patterns() {
    test_basic_insertion_patterns<transactional_trivial_set_t>();
    test_basic_insertion_patterns<transactional_heavy_set_t>();
    test_basic_insertion_patterns<transactional_trivial_map_t>();
    test_basic_insertion_patterns<transactional_composite_map_t>();
    test_basic_insertion_patterns<transactional_heavy_map_t>();
}

static void point_access_bulk_insertion_iterators() {
    test_bulk_insertion_from_iterators<transactional_trivial_set_t>();
    test_bulk_insertion_from_iterators<transactional_heavy_set_t>();
    test_bulk_insertion_from_iterators<transactional_trivial_map_t>();
    test_bulk_insertion_from_iterators<transactional_composite_map_t>();
    test_bulk_insertion_from_iterators<transactional_heavy_map_t>();
}

static void point_access_bulk_upsert_with_duplicates() { //
    test_bulk_upsert_with_duplicates<transactional_trivial_map_t>();
}

#pragma endregion Point Access Tests

#pragma region Consistency and Transaction Tests

static void transactional_consistency_empty_transaction_commit() {
    test_empty_transaction_commit<transactional_trivial_set_t>();
    test_empty_transaction_commit<transactional_heavy_set_t>();
    test_empty_transaction_commit<transactional_trivial_map_t>();
    test_empty_transaction_commit<transactional_composite_map_t>();
    test_empty_transaction_commit<transactional_heavy_map_t>();
}

static void transactional_consistency_no_dirty_reads_multi_key() {
    test_no_dirty_reads_multi_key<transactional_trivial_map_t>();
    test_no_dirty_reads_multi_key<transactional_composite_map_t>();
    test_no_dirty_reads_multi_key<transactional_heavy_map_t>();
}

static void transactional_consistency_new_transaction_sees_nothing_staged() {
    test_new_transaction_sees_nothing_staged<transactional_trivial_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_trivial_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_heavy_map_t>();
}

static void transactional_consistency_committed_immediately_visible() {
    test_committed_immediately_visible<transactional_trivial_map_t>();
    test_committed_immediately_visible<transactional_composite_map_t>();
    test_committed_immediately_visible<transactional_heavy_map_t>();
}

static void transactional_consistency_multi_key_atomicity_10_keys() {
    test_multi_key_atomicity_10_keys<transactional_trivial_map_t>();
    test_multi_key_atomicity_10_keys<transactional_composite_map_t>();
    test_multi_key_atomicity_10_keys<transactional_heavy_map_t>();
}

static void transactional_consistency_rollback_makes_all_invisible() {
    test_rollback_makes_all_invisible<transactional_trivial_map_t>();
    test_rollback_makes_all_invisible<transactional_composite_map_t>();
    test_rollback_makes_all_invisible<transactional_heavy_map_t>();
}

static void transactional_consistency_fractured_read_prevention() {
    test_fractured_read_prevention<transactional_trivial_map_t>();
    test_fractured_read_prevention<transactional_composite_map_t>();
    test_fractured_read_prevention<transactional_heavy_map_t>();
}

/** @brief Ordering of mapped values is only meaningful where they are numbers, so @c int maps only. */
static void transactional_consistency_sequential_updates_never_regress() {
    test_sequential_updates_never_regress<transactional_trivial_map_t>();
}

/** @brief Ordering of mapped values is only meaningful where they are numbers, so @c int maps only. */
static void transactional_consistency_transaction_commits_maintain_order() {
    test_transaction_commits_maintain_order<transactional_trivial_map_t>();
}

static void transactional_consistency_concurrent_transactions_on_same_key() {
    test_concurrent_transactions_on_same_key<transactional_trivial_map_t>();
    test_concurrent_transactions_on_same_key<transactional_composite_map_t>();
    test_concurrent_transactions_on_same_key<transactional_heavy_map_t>();
}

static void transactional_consistency_multi_key_conflict_any_key_fails() {
    test_multi_key_conflict_any_key_fails<transactional_trivial_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_composite_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_detects_external_direct_modification() {
    test_watch_detects_external_direct_modification<transactional_trivial_map_t>();
    test_watch_detects_external_direct_modification<transactional_composite_map_t>();
    test_watch_detects_external_direct_modification<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_detects_staged_invisible_writes() {
    test_watch_detects_staged_invisible_writes<transactional_trivial_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_composite_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_heavy_map_t>();
}

static void transactional_consistency_abandoned_transaction_leaves_no_trace() {
    test_abandoned_transaction_leaves_no_trace<transactional_trivial_map_t>();
    test_abandoned_transaction_leaves_no_trace<transactional_composite_map_t>();
    test_abandoned_transaction_leaves_no_trace<transactional_heavy_map_t>();
}

static void transactional_consistency_moved_transaction_unwinds_once() {
    test_moved_transaction_unwinds_once<transactional_trivial_map_t>();
    test_moved_transaction_unwinds_once<transactional_composite_map_t>();
    test_moved_transaction_unwinds_once<transactional_heavy_map_t>();
}

static void transactional_consistency_watch_on_erased_key_can_commit() {
    test_watch_on_erased_key_can_commit<transactional_trivial_map_t>();
    test_watch_on_erased_key_can_commit<transactional_composite_map_t>();
    test_watch_on_erased_key_can_commit<transactional_heavy_map_t>();
}

static void transactional_consistency_absent_watch_survives_rollback() {
    test_absent_watch_survives_rollback<transactional_trivial_map_t>();
    test_absent_watch_survives_rollback<transactional_composite_map_t>();
    test_absent_watch_survives_rollback<transactional_heavy_map_t>();
}

static void transactional_consistency_disjoint_keys_both_succeed() {
    test_disjoint_keys_both_succeed<transactional_trivial_map_t>();
    test_disjoint_keys_both_succeed<transactional_composite_map_t>();
    test_disjoint_keys_both_succeed<transactional_heavy_map_t>();
}

static void transactional_consistency_non_repeatable_reads_are_allowed() {
    test_non_repeatable_reads_are_allowed<transactional_trivial_map_t>();
    test_non_repeatable_reads_are_allowed<transactional_composite_map_t>();
    test_non_repeatable_reads_are_allowed<transactional_heavy_map_t>();
}

static void transactional_consistency_reset_clears_transaction_state() {
    test_reset_clears_transaction_state<transactional_trivial_map_t>();
    test_reset_clears_transaction_state<transactional_composite_map_t>();
    test_reset_clears_transaction_state<transactional_heavy_map_t>();
}

#pragma endregion Consistency and Transaction Tests

int main() {
    install_test_signal_handlers();
    char const *const filter = std::getenv("SMASHTABLE_FILTER");
    std::size_t failures = 0;

    failures += run_test(filter, "point_access.insert_strategies", point_access_insert_strategies);
    failures += run_test(filter, "point_access.erase_visibility", point_access_erase_visibility);
    failures += run_test(filter, "point_access.rollback_restores_store", point_access_rollback_restores_store);
    failures += run_test(filter, "point_access.staging_survives_growth", point_access_staging_survives_growth);
    failures += run_test(filter, "point_access.insertion_patterns", point_access_insertion_patterns);
    failures += run_test(filter, "point_access.bulk_insertion_iterators", point_access_bulk_insertion_iterators);
    failures += run_test(filter, "point_access.bulk_upsert_with_duplicates", point_access_bulk_upsert_with_duplicates);

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
    failures += run_test(filter, "transactional_consistency.non_repeatable_reads_are_allowed",
                         transactional_consistency_non_repeatable_reads_are_allowed);
    failures += run_test(filter, "transactional_consistency.reset_clears_transaction_state",
                         transactional_consistency_reset_clears_transaction_state);

    return report_test_failures(failures);
}
