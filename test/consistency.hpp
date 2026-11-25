/**
 *  @file test/consistency.hpp
 *  @author Ash Vardanian
 *  @date January 12, 2023
 *  @brief Template test functions for advanced transactional consistency scenarios. Includes tests
 *      for isolation levels, visibility rules, and conflict resolution.
 */
#pragma once
#include <compare> // `std::compare_three_way`
#include <utility> // `std::move`
#include <vector>  // `std::vector`

#include <smashtable/monotonic_store.hpp> // `monotonic_avl_map`
#include <smashtable/snapshot_store.hpp>  // `snapshot_avl_map`, `serializable_avl_map`

#include "fixtures.hpp"
#include "failure_policy.hpp"
#include "surface_parity.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Level Parity

/**
 *  @brief The store-versus-transaction and level-versus-wrapper folds, over the shipped engines.
 *
 *  Asserted here rather than beside the wrapper fold because these ask about a store on its own: a
 *  surface missing from a store @b and from its transaction passes every wrapper check vacuously,
 *  which is how a whole family of them went unnoticed.
 */
using parity_snapshot_map_t = snapshot_avl_map<std::uint64_t, std::uint64_t>;
using parity_monotonic_map_t = monotonic_avl_map<std::uint64_t, std::uint64_t>;
using parity_serializable_map_t = serializable_avl_map<std::uint64_t, std::uint64_t>;

/** Weight-balanced rather than AVL, so the ordinal pairs are asked of a core that keeps subtree
 *  counts. */
using parity_snapshot_ranked_map_t = snapshot_wb_map<std::uint64_t, std::uint64_t>;
using parity_monotonic_ranked_map_t = monotonic_wb_map<std::uint64_t, std::uint64_t>;

/** The same engines over bare keys, since a fold that only ever saw mappings never asked about the
 *  other shape. */
using parity_snapshot_set_t = snapshot_avl_set<std::uint64_t>;
using parity_monotonic_set_t = monotonic_avl_set<std::uint64_t>;
using parity_serializable_set_t = serializable_avl_set<std::uint64_t>;
using parity_snapshot_ranked_set_t = snapshot_wb_set<std::uint64_t>;
using parity_monotonic_ranked_set_t = monotonic_wb_set<std::uint64_t>;

static_assert(transaction_mirrors_the_store<parity_snapshot_map_t>,
              "a snapshot store's transaction must mirror the store it opens on");
static_assert(transaction_mirrors_the_store<parity_monotonic_map_t>,
              "a monotonic store's transaction must mirror the store it opens on");
static_assert(transaction_mirrors_the_store<parity_serializable_map_t>,
              "a serializable store's transaction must mirror the store it opens on");
static_assert(transaction_mirrors_the_store<parity_snapshot_ranked_map_t>,
              "a ranked store's transaction must mirror the store it opens on");
static_assert(transaction_mirrors_the_store<parity_monotonic_ranked_map_t>,
              "a ranked store's transaction must mirror the store it opens on");

static_assert(transaction_mirrors_the_store<parity_snapshot_set_t>,
              "a set-shaped snapshot store's transaction must mirror the store it opens on");
static_assert(transaction_mirrors_the_store<parity_monotonic_set_t>,
              "a set-shaped monotonic store's transaction must mirror the store it opens on");
static_assert(transaction_mirrors_the_store<parity_serializable_set_t>,
              "a set-shaped serializable store's transaction must mirror the store it opens on");
static_assert(transaction_mirrors_the_store<parity_snapshot_ranked_set_t>,
              "a set-shaped ranked store's transaction must mirror the store it opens on");
static_assert(transaction_mirrors_the_store<parity_monotonic_ranked_set_t>,
              "a set-shaped ranked store's transaction must mirror the store it opens on");

static_assert(a_set_shaped_store<parity_snapshot_set_t> && a_map_shaped_store<parity_snapshot_map_t>,
              "the set and map aliases of one engine must not resolve to the same shape");
static_assert(a_set_shaped_store<parity_monotonic_ranked_set_t> && a_map_shaped_store<parity_monotonic_ranked_map_t>,
              "the set and map aliases of one engine must not resolve to the same shape");

static_assert(wrappers_honour_the_level<parity_serializable_map_t>,
              "a wrapper reporting a level must forward the surface that level is defined by");
static_assert(wrappers_honour_the_level<parity_serializable_set_t>,
              "a set-shaped wrapper reporting a level must forward the same surface");

#pragma endregion Level Parity

#pragma region Isolation Expectations

/** The level at which a read repeated inside one transaction must return what it first saw. Below
 *  it a container is free to show the newer committed value, and the suite asserts that it does -
 *  an anomaly a level permits is a promise that level makes, not an outcome left open. */
inline constexpr isolation_t repeatable_reads_from_k = isolation_t::snapshot_k;

/** The level at which a predicate repeated inside one transaction must see the same members. */
inline constexpr isolation_t stable_predicates_from_k = isolation_t::snapshot_k;

/** The level at which one commit reaches a reader whole, rather than half of it at a time. Below it
 *  a walk spanning several partitions is free to meet two commits at once, and the suite asserts
 *  that it does - which is the anomaly Monotonic Atomic View is named for denying. */
inline constexpr isolation_t whole_commits_from_k = isolation_t::monotonic_atomic_view_k;

/**
 *  @brief The level at which two blind writers to one key are separated, rather than one
 *      overwriting the other.
 *
 *  Below it the later commit wins and the earlier write is lost - the anomaly Berenson numbers P4
 *  and Monotonic Atomic View licenses, since that level constrains what a reader observes rather
 *  than what two writers do to each other, and Bailis shows a totally available model provably
 *  cannot prevent it. From here up the write set is validated whether the caller asks or not, which
 *  is the first-committer -wins rule Snapshot Isolation is defined by.
 */
inline constexpr isolation_t first_committer_wins_from_k = isolation_t::snapshot_k;

/**
 *  @brief The level at which what a transaction @b read is validated, not only what it wrote.
 *
 *  Below it only the write set is checked, so two transactions that each read what the other writes
 *  and then write disjointly both land - the anomaly Berenson numbers A5B, which Snapshot Isolation
 *  permits by construction because no two writes ever collide. From here up the read set is
 *  validated too, which separates a serializable history from a merely snapshot-isolated one.
 */
inline constexpr isolation_t validated_reads_from_k = isolation_t::serializable_k;

#pragma endregion Isolation Expectations

/** Edge Case: Committing empty transaction succeeds */
template <typename container_type_>
void check_empty_transaction_commit() {

    using container_t = container_type_;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;

    // Don't add anything, just commit
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.commit(), success_k);

    expect_equal(container.size(), 0);
}

template <typename container_type_>
void check_no_dirty_reads_multi_key() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    // Initial state: keys 1-5 exist
    container_t container;
    for (std::size_t i = 1; i <= 5; ++i) {
        auto new_member = trivial_id_to_member<member_t>(i, i);
        expect_equal(container.upsert(std::move(new_member)), success_k);
    }

    // T1 modifies all 5 keys but only stages (doesn't commit)
    auto [first, first_status] = container.transaction();
    if (!expect_equal(first_status, success_k)) return;
    for (std::size_t i = 1; i <= 5; ++i) {
        auto new_member = trivial_id_to_member<member_t>(i, i * 100);
        expect_equal(first.upsert(std::move(new_member)), success_k);
    }
    expect_equal(first.stage(), success_k);

    // External reader should see original values (staged changes invisible)
    for (std::size_t i = 1; i <= 5; ++i) {
        auto [found, found_status] = container.find_copy(trivial_id_to_key<member_t>(i));
        if (!expect_equal(found_status, success_k, "key must exist before the transaction stages over it")) return;
        expect_equal((found.mapped), (static_cast<decltype(found.mapped)>(i)),
                     "a reader outside must see the original, not the staged value");
    }
}

/**
 *  @brief Read Committed: New transactions don't see staged (uncommitted) data
 *
 *  Timeline:
 *      T1:  upsert(1,100)  →  upsert(2,200)  →  stage()
 *
 *      T2:  transaction()  →  find(1,2)  →  sees nothing
 *
 *  Expected: T2 sees nothing (T1 not committed yet)
 */
template <typename container_type_>
void check_new_transaction_sees_nothing_staged() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto [first, first_status] = container.transaction();
    if (!expect_equal(first_status, success_k)) return;
    expect_equal(first.upsert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(2)), success_k);
    expect_equal(first.stage(), success_k); // Staged but not committed

    // T2 created after T1 staged - should see nothing
    auto [second, second_status] = container.transaction();
    if (!expect_equal(second_status, success_k)) return;

    expect_equal(second.find_copy(trivial_id_to_key<member_t>(1)).status(), status_t::key_not_found_k,
                 "a later transaction must not see a staged key");
    expect_equal(second.find_copy(trivial_id_to_key<member_t>(2)).status(), status_t::key_not_found_k,
                 "a later transaction must not see a staged key");
}

/**
 *  @brief Read Committed: Committed transactions are immediately visible
 *
 *  Timeline:
 *      T1:  upsert(1,111)  →  upsert(2,222)  →  stage()  →  commit()
 *
 *      T2:  transaction()  →  find(1,2)  →  sees (1,111) and (2,222)
 *
 *  Expected: T2 sees committed data immediately
 */
template <typename container_type_>
void check_committed_immediately_visible() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto [first, first_status] = container.transaction();
    if (!expect_equal(first_status, success_k)) return;
    expect_equal(first.upsert(trivial_id_to_member<member_t>(1, 111)), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(2, 222)), success_k);
    expect_equal(first.stage(), success_k);
    expect_equal(first.commit(), success_k); // NOW committed

    // T2 created after commit - should see everything
    auto [second, second_status] = container.transaction();
    if (!expect_equal(second_status, success_k)) return;

    auto [first_held, first_held_status] = second.find_copy(trivial_id_to_key<member_t>(1));
    auto [second_held, second_held_status] = second.find_copy(trivial_id_to_key<member_t>(2));

    if (!expect_equal(first_held_status, success_k, "a later transaction must see a committed key")) return;
    if (!expect_equal(second_held_status, success_k, "a later transaction must see a committed key")) return;
    expect_equal(first_held.mapped, 111);
    expect_equal(second_held.mapped, 222);
}

/**
 *  @brief Atomic View: 10-key transaction visible as 0 or 10, never partial
 *
 *  Timeline:
 *
 *  @verbatim
 *  T.upsert(0..9)
 *  Observer sees: 0 keys
 *
 *  T.stage()
 *  Observer sees: 0 keys
 *
 *  T.commit()
 *  Observer sees: 10 keys, atomic
 *  @endverbatim
 *
 *  Expected: Never see 1-9 keys, all-or-nothing
 */
template <typename container_type_>
void check_multi_key_atomicity_10_keys() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;

    // Insert 10 keys in transaction
    for (std::size_t i = 0; i < 10; ++i)
        expect_equal(transaction.upsert(trivial_id_to_member<member_t>(i, i * 10)), success_k);

    // Before stage: should see 0
    std::size_t count_before_stage = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        auto const [matched, matched_status] = container.count(trivial_id_to_key<member_t>(i));
        if (!expect_equal(matched_status, success_k)) return;
        count_before_stage += matched;
    }
    expect_equal(count_before_stage, 0);
    expect_equal(transaction.stage(), success_k);

    // After stage, before commit: should see 0
    std::size_t count_after_stage = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        auto const [matched, matched_status] = container.count(trivial_id_to_key<member_t>(i));
        if (!expect_equal(matched_status, success_k)) return;
        count_after_stage += matched;
    }
    expect_equal(count_after_stage, 0);
    expect_equal(transaction.commit(), success_k);

    // After commit: should see all 10
    std::size_t count_after_commit = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        auto const [matched, matched_status] = container.count(trivial_id_to_key<member_t>(i));
        if (!expect_equal(matched_status, success_k)) return;
        count_after_commit += matched;
    }
    expect_equal((count_after_commit), (10), "commit must publish every key at once");
}

/**
 *  @brief Atomic View: Rollback atomically hides all staged changes
 *
 *  Setup: set.upsert(1,1)
 *
 *  Timeline:
 *
 *  @verbatim
 *  T:  upsert(1,100)  →  upsert(2,200)  →  upsert(3,300)  →  stage()  →  rollback()
 *
 *  Observer: ────────────────────────────────────────────────────────────────────────→
 *          sees 1,1 only, not 2 or 3
 *  @endverbatim
 *
 *  Expected: All staged changes rolled back atomically
 */
template <typename container_type_>
void check_rollback_makes_all_invisible() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k); // Initial state

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(1, 100)), success_k); // Modify existing
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(2, 200)), success_k); // Add new
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(3, 300)), success_k); // Add new
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.rollback(), success_k); // ROLLBACK instead of commit

    // Key 1 should have original value
    auto [held, held_status] = container.find_copy(trivial_id_to_key<member_t>(1));
    if (!expect_equal(held_status, success_k, "rollback must leave the committed entry in place")) return;
    expect_equal((held.mapped), (1), "rollback must restore the original value");

    // Keys 2 and 3 should not exist
    expect_equal(container.find_copy(trivial_id_to_key<member_t>(2)).status(), status_t::key_not_found_k,
                 "rollback must discard the staged key");
    expect_equal(container.find_copy(trivial_id_to_key<member_t>(3)).status(), status_t::key_not_found_k,
                 "rollback must discard the staged key");
}

/**
 *  @brief Atomic View: Range queries see complete transactions
 *
 *  Timeline:
 *
 *  @verbatim
 *  T.upsert(10..19)
 *  range(10,20) sees: 0 elements
 *
 *  T.stage()
 *  range(10,20) sees: 0 elements
 *
 *  T.commit()
 *  range(10,20) sees: 10 elements, atomic
 *  @endverbatim
 *
 *  Expected: Range sees 0 or 10, never partial
 */
template <typename container_type_>
void check_range_query_sees_atomic_boundaries() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    for (std::size_t i = 10; i < 20; ++i)
        expect_equal(transaction.upsert(trivial_id_to_member<member_t>(i)), success_k);

    // Before commit: range query sees 0
    std::size_t count_before = 0;
    expect_equal(container.range(trivial_id_to_key<member_t>(10), trivial_id_to_key<member_t>(20),
                                 [&](member_t const &) noexcept { count_before++; }),
                 success_k);
    expect_equal(count_before, 0);

    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.commit(), success_k);

    // After commit: range query sees all 10
    std::size_t count_after = 0;
    expect_equal(container.range(trivial_id_to_key<member_t>(10), trivial_id_to_key<member_t>(20),
                                 [&](member_t const &) noexcept { count_after++; }),
                 success_k);
    expect_equal(count_after, 10);
}

/**
 *  @brief Atomic View: Interleaved commits don't create fractured reads
 *
 *  Timeline:
 *
 *  @verbatim
 *  T1.upsert(1,2,3)  →  T1.stage()
 *  T2.upsert(4,5,6)  →  T2.stage()
 *  Observer sees: 0 keys
 *
 *  T1.commit()
 *  Observer sees: 3 keys, 1 2 3 only
 *
 *  T2.commit()
 *  Observer sees: 6 keys, all
 *  @endverbatim
 *
 *  Expected: No partial views across transactions
 */
template <typename container_type_>
void check_fractured_read_prevention() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;

    // T1 will insert keys 1-3
    auto [first, first_status] = container.transaction();
    if (!expect_equal(first_status, success_k)) return;
    expect_equal(first.upsert(trivial_id_to_member<member_t>(1, 10)), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(2, 20)), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(3, 30)), success_k);

    // T2 will insert keys 4-6
    auto [second, second_status] = container.transaction();
    if (!expect_equal(second_status, success_k)) return;
    expect_equal(second.upsert(trivial_id_to_member<member_t>(4, 40)), success_k);
    expect_equal(second.upsert(trivial_id_to_member<member_t>(5, 50)), success_k);
    expect_equal(second.upsert(trivial_id_to_member<member_t>(6, 60)), success_k);

    // Stage both
    expect_equal(first.stage(), success_k);
    expect_equal(second.stage(), success_k);

    // Before any commits: see 0
    std::size_t count_initial = 0;
    for (std::size_t i = 1; i <= 6; ++i)
        expect_equal(
            container.find(trivial_id_to_key<member_t>(i), [&](member_t const &) noexcept { count_initial++; }),
            success_k);
    expect_equal(count_initial, 0);
    expect_equal(first.commit(), success_k); // Commit T1

    // Should see exactly T1's keys (1-3), not T2's (4-6)
    std::size_t count_t1 = 0;
    for (std::size_t i = 1; i <= 6; ++i)
        expect_equal(container.find(trivial_id_to_key<member_t>(i), [&](member_t const &) noexcept { count_t1++; }),
                     success_k);
    expect_equal((count_t1), (3), "a range must not see another transaction staged keys");
    expect_equal(second.commit(), success_k); // Commit T2

    // Now should see all 6
    std::size_t count_both = 0;
    for (std::size_t i = 1; i <= 6; ++i)
        expect_equal(container.find(trivial_id_to_key<member_t>(i), [&](member_t const &) noexcept { count_both++; }),
                     success_k);
    expect_equal((count_both), (6), "a range must see both committed transactions");
}

/** Monotonic View: Values never go backwards (10 → 20 → 30, never 30 → 20) */
template <typename container_type_>
void check_sequential_updates_never_regress() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");
    static_assert(std::is_integral<typename member_t::mapped_type>::value, "Mapped type must be integral");

    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 10)), success_k);

    std::vector<int> observed_values;

    // Observe initial value
    expect_equal(container.find(trivial_id_to_key<member_t>(1),
                                [&](member_t const &e) noexcept { observed_values.push_back(e.mapped); }),
                 success_k);

    // Update to 20
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 20)), success_k);
    expect_equal(container.find(trivial_id_to_key<member_t>(1),
                                [&](member_t const &e) noexcept { observed_values.push_back(e.mapped); }),
                 success_k);

    // Update to 30
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 30)), success_k);
    expect_equal(container.find(trivial_id_to_key<member_t>(1),
                                [&](member_t const &e) noexcept { observed_values.push_back(e.mapped); }),
                 success_k);

    // Each read answered the write before it, so these three are the sequence in the order it was written.
    if (!expect_equal(observed_values.size(), 3)) return;
    expect_equal(observed_values[0], 10);
    expect_equal(observed_values[1], 20);
    expect_equal(observed_values[2], 30);
}

/** Monotonic View: T1 commits v1, T2 commits v2 → always see v1 → v2, never v2 → v1 */
template <typename container_type_>
void check_transaction_commits_maintain_order() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    // T1: value = 100
    auto [first, first_status] = container.transaction();
    if (!expect_equal(first_status, success_k)) return;
    expect_equal(first.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);
    expect_equal(first.stage(), success_k);
    expect_equal(first.commit(), success_k);

    // Observe T1's value
    auto [first_held, first_held_status] = container.find_copy(trivial_id_to_key<member_t>(1));
    if (!expect_equal(first_held_status, success_k)) return;
    expect_equal(first_held.mapped, 100);

    // T2: value = 200 (higher)
    auto [second, second_status] = container.transaction();
    if (!expect_equal(second_status, success_k)) return;
    expect_equal(second.upsert(trivial_id_to_member<member_t>(1, 200)), success_k);
    expect_equal(second.stage(), success_k);
    expect_equal(second.commit(), success_k);

    // Observe T2's value, which is the later of the two commits and so the one that stands
    auto [second_held, second_held_status] = container.find_copy(trivial_id_to_key<member_t>(1));
    if (!expect_equal(second_held_status, success_k)) return;
    expect_equal(second_held.mapped, 200);
}

/**
 *  @brief Write Conflicts: Concurrent transactions on same key - first wins, second fails
 *
 *  Timeline:
 *
 *  @verbatim
 *  T1:  watch(1)  →  upsert(1,100)  →  stage()  →  commit()    ✓
 *
 *  T2:  watch(1)  →  upsert(1,200)  →  stage()                 ✗ CONFLICT
 *  @endverbatim
 *
 *  Expected: T1 succeeds, T2 fails with consistency_k
 */
template <typename container_type_>
void check_concurrent_transactions_on_same_key() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);

    auto [first, first_status] = container.transaction();
    if (!expect_equal(first_status, success_k)) return;
    auto [second, second_status] = container.transaction();
    if (!expect_equal(second_status, success_k)) return;

    // Both watch the same key
    expect_equal(first.watch(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(second.watch(trivial_id_to_key<member_t>(1)), success_k);

    // Both modify it
    expect_equal(first.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);
    expect_equal(second.upsert(trivial_id_to_member<member_t>(1, 200)), success_k);

    // T1 commits successfully
    expect_equal(first.stage(), success_k);
    expect_equal(first.commit(), success_k);

    // T2's stage should fail (watched value changed)
    auto status = second.stage();
    expect_true(failed(status), "staging must fail once a watched key was modified");
    expect_equal(status, status_t::read_conflict_k);

    // Verify T1's value persisted, T2's did not
    auto [final_member, final_member_status] = container.find_copy(trivial_id_to_key<member_t>(1));
    if (!expect_equal(final_member_status, success_k)) return;
    expect_equal((final_member.mapped), (100), "only the winning transaction value may persist");
}

/**
 *  @brief Write Conflicts: any watched key conflict fails entire transaction
 *
 *  Timeline:
 *
 *  @verbatim
 *  T1:  watch(1,2,3)
 *
 *  External:  set.upsert(2,999)  ← modifies key 2
 *
 *  T1:  upsert(1,2,3)  →  stage()    ✗ CONFLICT on key 2
 *  @endverbatim
 *
 *  Expected: Transaction fails if any watched key changed
 */
template <typename container_type_>
void check_multi_key_conflict_any_key_fails() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);
    expect_equal(container.upsert(trivial_id_to_member<member_t>(2, 2)), success_k);
    expect_equal(container.upsert(trivial_id_to_member<member_t>(3, 3)), success_k);

    auto [first, first_status] = container.transaction();
    if (!expect_equal(first_status, success_k)) return;
    auto [second, second_status] = container.transaction();
    if (!expect_equal(second_status, success_k)) return;

    // T1 watches keys 1, 2, 3
    expect_equal(first.watch(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(first.watch(trivial_id_to_key<member_t>(2)), success_k);
    expect_equal(first.watch(trivial_id_to_key<member_t>(3)), success_k);

    // T2 watches same keys
    expect_equal(second.watch(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(second.watch(trivial_id_to_key<member_t>(2)), success_k);
    expect_equal(second.watch(trivial_id_to_key<member_t>(3)), success_k);

    // External update to just one key (key 2)
    expect_equal(container.upsert(trivial_id_to_member<member_t>(2, 999)), success_k);

    // T1 modifies all three
    expect_equal(first.upsert(trivial_id_to_member<member_t>(1, 10)), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(2, 20)), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(3, 30)), success_k);

    // T1's stage should fail (key 2 was modified externally)
    auto status = first.stage();
    expect_true(failed(status), "staging must fail if any watched key changed");
    expect_equal(status, status_t::read_conflict_k);
}

/**
 *  @brief Write Conflicts: Watches detect direct set.upsert() modifications
 *
 *  Timeline:
 *
 *  @verbatim
 *  T:  watch(1)
 *
 *  External:  set.upsert(1,777)  ← direct modification
 *
 *  T:  upsert(1,888)  →  stage()    ✗ CONFLICT
 *  @endverbatim
 *
 *  Expected: Watch detects external change
 */
template <typename container_type_>
void check_watch_detects_external_direct_modification() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;
    expect_equal(transaction.watch(trivial_id_to_key<member_t>(1)), success_k);

    // Direct modification to the set (not through a transaction)
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 777)), success_k);

    // Transaction attempts to modify
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(1, 888)), success_k);

    // Stage should detect the external change
    auto status = transaction.stage();
    expect_true(failed(status), "a watch must detect a direct write to the store");
    expect_equal(status, status_t::read_conflict_k);
}

/**
 *  @brief Write Conflicts: Disjoint key sets allow both transactions to succeed
 *
 *  Timeline:
 *
 *  @verbatim
 *  T1:  upsert(1,2,3)  →  stage()  →  commit()    ✓
 *
 *  T2:  upsert(4,5,6)  →  stage()  →  commit()    ✓
 *  @endverbatim
 *
 *  Expected: Both succeed (no overlapping keys)
 */
template <typename container_type_>
void check_disjoint_keys_both_succeed() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto [first, first_status] = container.transaction();
    if (!expect_equal(first_status, success_k)) return;
    auto [second, second_status] = container.transaction();
    if (!expect_equal(second_status, success_k)) return;

    // T1 modifies keys 1-3
    expect_equal(first.upsert(trivial_id_to_member<member_t>(1, 10)), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(2, 20)), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(3, 30)), success_k);

    // T2 modifies keys 4-6 (disjoint!)
    expect_equal(second.upsert(trivial_id_to_member<member_t>(4, 40)), success_k);
    expect_equal(second.upsert(trivial_id_to_member<member_t>(5, 50)), success_k);
    expect_equal(second.upsert(trivial_id_to_member<member_t>(6, 60)), success_k);

    // Both should succeed
    expect_equal(first.stage(), success_k);
    expect_equal(second.stage(), success_k);
    expect_equal(first.commit(), success_k);
    expect_equal(second.commit(), success_k);

    // Verify all 6 keys exist
    std::size_t count = 0;
    for (std::size_t i = 1; i <= 6; ++i) {
        auto const [matched, matched_status] = container.count(trivial_id_to_key<member_t>(i));
        if (!expect_equal(matched_status, success_k)) return;
        count += matched;
    }
    expect_equal((count), (6), "disjoint transactions must both commit");
}

/**
 *  @brief Allowed Anomaly: Non-repeatable reads are correct for Read Committed
 *
 *  Setup: set.upsert(1,100)
 *
 *  Timeline:
 *
 *  @verbatim
 *  T:  find(1)  →  sees 100
 *
 *  External:  set.upsert(1,999)
 *
 *  T:  find(1)  →  sees 999    ← CORRECT! (not a bug)
 *  @endverbatim
 *
 *  Expected: Within transaction, reads can see different values
 */
template <typename container_type_>
void check_repeated_read_matches_isolation() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;

    // First read
    auto [first_read, first_read_status] = transaction.find_copy(trivial_id_to_key<member_t>(1));
    if (!expect_equal(first_read_status, success_k)) return;
    expect_equal(first_read.mapped, 100);

    // External modification
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 999)), success_k);

    // Second read in same transaction - can see new value (this is correct!)
    auto [second_read, second_read_status] = transaction.find_copy(trivial_id_to_key<member_t>(1));
    if (!expect_equal(second_read_status, success_k)) return;

    if constexpr (at_least(container_t::isolation_k, repeatable_reads_from_k)) {
        expect_equal((second_read.mapped), (100), "a snapshot must repeat its first read");
        expect_equal(first_read.mapped, second_read.mapped);
    }
    else {
        expect_equal((second_read.mapped), (999), "below snapshot, a read sees the newest commit");
        expect_true(first_read.mapped != second_read.mapped);
    }
}

/**
 *  @brief Allowed Anomaly: Phantom reads are correct for Read Committed
 *
 *  Setup: set.upsert(0..4)
 *
 *  Timeline:
 *
 *  @verbatim
 *  T:  range(0,10)  →  counts 5
 *
 *  External:  set.upsert(5,6)
 *
 *  T:  range(0,10)  →  counts 7    ← CORRECT! (not a bug)
 *  @endverbatim
 *
 *  Expected: Range counts can change within transaction
 */
template <typename container_type_>
void check_repeated_range_matches_isolation() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t i = 0; i < 5; ++i) expect_equal(container.upsert(trivial_id_to_member<member_t>(i)), success_k);

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;

    // The predicate is evaluated through the transaction, not through `container`. Asking the store
    // would be a question about the store rather than about the transaction's view, and would hold at
    // every isolation level including serializable - which is why the earlier shape of this test could
    // not fail. It is spelled with `find` rather than a range scan because that is the one ordered-free
    // surface every transaction type here offers, `partitioned_store`'s included.
    auto count_present = [&]() noexcept {
        std::size_t seen = 0;
        for (std::size_t candidate = 0; candidate != 10; ++candidate) {
            auto const [present, present_status] = transaction.contains(trivial_id_to_key<member_t>(candidate));
            if (!expect_equal(present_status, success_k)) return seen;
            if (present) ++seen;
        }
        return seen;
    };

    std::size_t const first_count = count_present();
    expect_equal(first_count, 5);

    // Committed from outside the transaction, after the first evaluation and before the second.
    expect_equal(container.upsert(trivial_id_to_member<member_t>(5, 5)), success_k);
    expect_equal(container.upsert(trivial_id_to_member<member_t>(6, 6)), success_k);

    std::size_t const second_count = count_present();

    if constexpr (at_least(container_t::isolation_k, stable_predicates_from_k))
        expect_equal((second_count), (5), "a snapshot must not admit phantoms");
    else expect_equal((second_count), (7), "below snapshot, a repeated predicate sees the newest commits");
}

/**
 *  @brief Licensed Anomaly: a lost update is correct below Snapshot.
 *
 *  Two transactions open together and each writes key 1 without reading it. The first commits, then
 *  the second. Whether the second may land is the whole of what separates the two levels here.
 *
 *  Timeline:
 *
 *  @verbatim
 *  A, B open together
 *  A:  upsert(1, 10)
 *  B:  upsert(1, 20)
 *  A:  commit           → lands
 *  B:  commit           → lands below Snapshot, refused from Snapshot up
 *  @endverbatim
 *
 *  Below Snapshot nothing validates the write set, so B lands and A's write is gone. That is the
 *  lost update, and it is a promise this level makes rather than an outcome left open. From
 *  Snapshot up the write set is checked against everything published since B opened, so B is
 *  refused and A survives.
 */
template <typename container_type_>
void check_lost_update_matches_isolation() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 0)), success_k);

    // Both open before either writes, so each is unaware of the other by construction.
    auto [earlier, earlier_status] = container.transaction();
    if (!expect_equal(earlier_status, success_k)) return;
    auto [later, later_status] = container.transaction();
    if (!expect_equal(later_status, success_k)) return;

    expect_equal(earlier.upsert(trivial_id_to_member<member_t>(1, 10)), success_k);
    expect_equal(later.upsert(trivial_id_to_member<member_t>(1, 20)), success_k);

    expect_equal(earlier.stage(), success_k);
    expect_equal(earlier.commit(), success_k);

    // A store that validates the write set refuses at `stage`, one that does not gets as far as `commit`,
    // so the answer is whichever of the two spoke first.
    status_t const staged = later.stage();
    status_t const answered = succeeded(staged) ? later.commit() : staged;

    auto [held, held_status] = container.find_copy(trivial_id_to_key<member_t>(1));
    if (!expect_equal(held_status, success_k)) return;

    if constexpr (at_least(container_t::isolation_k, first_committer_wins_from_k)) {
        expect_equal(answered, status_t::write_conflict_k);
        expect_equal((held.mapped), (10), "first-committer-wins keeps the write that landed first");
    }
    else {
        expect_equal(answered, success_k);
        expect_equal((held.mapped), (20), "below snapshot the later write wins and the earlier one is lost");
    }
}

/**
 *  @brief Write skew lands below the level that validates reads, and is refused from there up.
 *
 *  Two transactions open together over a pair of keys holding an invariant - here that at least one
 *  of them stays set. Each reads both, sees the other one covering the invariant, and clears its
 *  own. The write sets are disjoint, so nothing a write-set validator compares ever collides.
 *
 *  Timeline:
 *
 *  @verbatim
 *  A, B open together, both keys set
 *  A:  reads 1 and 2, sees the invariant held by 2
 *  B:  reads 1 and 2, sees the invariant held by 1
 *  A:  upsert(1, 0), commit    → lands
 *  B:  upsert(2, 0), commit    → lands below the threshold, refused from it up
 *  @endverbatim
 *
 *  Below the threshold both land and the invariant is broken with neither transaction at fault -
 *  the anomaly Snapshot Isolation permits, and a promise this level makes rather than an outcome
 *  left open. From the threshold up B's read of key 1 is checked against A's commit and refused.
 */
template <typename container_type_>
void check_write_skew_matches_isolation() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);
    expect_equal(container.upsert(trivial_id_to_member<member_t>(2, 1)), success_k);

    auto [earlier, earlier_status] = container.transaction();
    if (!expect_equal(earlier_status, success_k)) return;
    auto [later, later_status] = container.transaction();
    if (!expect_equal(later_status, success_k)) return;

    // Each reads the pair and concludes the other key covers the invariant.
    std::size_t earlier_total = 0, later_total = 0;
    for (trivial_id_t key = 1; key != 3; ++key) {
        expect_equal(earlier.find(
                         trivial_id_to_key<member_t>(key),
                         [&](member_t const &held) noexcept { earlier_total += static_cast<std::size_t>(held.mapped); },
                         no_op_t {}),
                     success_k);
        expect_equal(later.find(
                         trivial_id_to_key<member_t>(key),
                         [&](member_t const &held) noexcept { later_total += static_cast<std::size_t>(held.mapped); },
                         no_op_t {}),
                     success_k);
    }
    expect_equal(earlier_total, 2u, "both keys must be set before either transaction clears one");
    expect_equal(later_total, 2u, "both keys must be set before either transaction clears one");

    // Disjoint writes, so a validator comparing only written keys sees no collision at all.
    expect_equal(earlier.upsert(trivial_id_to_member<member_t>(1, 0)), success_k);
    expect_equal(later.upsert(trivial_id_to_member<member_t>(2, 0)), success_k);

    expect_equal(earlier.stage(), success_k);
    expect_equal(earlier.commit(), success_k);

    status_t const staged = later.stage();
    status_t const answered = succeeded(staged) ? later.commit() : staged;

    auto [first_held, first_held_status] = container.find_copy(trivial_id_to_key<member_t>(1));
    auto [second_held, second_held_status] = container.find_copy(trivial_id_to_key<member_t>(2));
    if (!expect_equal(first_held_status, success_k)) return;
    if (!expect_equal(second_held_status, success_k)) return;
    std::size_t const remaining =
        static_cast<std::size_t>(first_held.mapped) + static_cast<std::size_t>(second_held.mapped);

    if constexpr (at_least(container_t::isolation_k, validated_reads_from_k)) {
        expect_equal(answered, status_t::read_conflict_k);
        expect_equal(remaining, 1u, "validating the read set is what keeps the invariant standing");
    }
    else {
        expect_equal(answered, success_k);
        expect_equal(remaining, 0u, "below validated reads both writes land and the invariant is gone");
    }
}

/**
 *  @brief A key read by one transaction and written by another refuses only where reads validate.
 *
 *  The narrower half of the same property. One transaction reads a key it never writes, another
 *  writes that key and commits, and the first then writes somewhere else entirely. Nothing the two
 *  of them wrote overlaps, so only a validator that remembers the read can object.
 *
 *  Timeline:
 *
 *  @verbatim
 *  A, B open together
 *  A:  find(1)                 → records the read where the level validates reads
 *  B:  upsert(1, 99), commit   → lands
 *  A:  upsert(2, 5), commit    → lands below the threshold, refused from it up
 *  @endverbatim
 */
template <typename container_type_>
void check_read_conflict_matches_isolation() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);
    expect_equal(container.upsert(trivial_id_to_member<member_t>(2, 1)), success_k);

    auto [reader, reader_status] = container.transaction();
    if (!expect_equal(reader_status, success_k)) return;
    auto [writer, writer_status] = container.transaction();
    if (!expect_equal(writer_status, success_k)) return;

    std::size_t seen = 0;
    expect_equal(reader.find(
                     trivial_id_to_key<member_t>(1),
                     [&](member_t const &held) noexcept { seen = static_cast<std::size_t>(held.mapped); }, no_op_t {}),
                 success_k);
    expect_equal(seen, 1u);

    expect_equal(writer.upsert(trivial_id_to_member<member_t>(1, 99)), success_k);
    expect_equal(writer.stage(), success_k);
    expect_equal(writer.commit(), success_k);

    // A key the reader never touched, so the write sets are disjoint by construction.
    expect_equal(reader.upsert(trivial_id_to_member<member_t>(2, 5)), success_k);

    status_t const staged = reader.stage();
    status_t const answered = succeeded(staged) ? reader.commit() : staged;

    auto [second_held, second_held_status] = container.find_copy(trivial_id_to_key<member_t>(2));
    if (!expect_equal(second_held_status, success_k)) return;

    if constexpr (at_least(container_t::isolation_k, validated_reads_from_k)) {
        expect_equal(answered, status_t::read_conflict_k);
        expect_equal((second_held.mapped), (1), "a refused transaction must leave its own write behind");
    }
    else {
        expect_equal(answered, success_k);
        expect_equal((second_held.mapped), (5), "below validated reads a stale read does not refuse a commit");
    }
}

/** Edge Case: Deleted entries are invisible to queries */
template <typename container_type_>
void check_delete_visibility() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(container.upsert(trivial_id_to_member<member_t>(2)), success_k);

    // Delete via erase_range
    erase_range_of(container, trivial_id_to_key<member_t>(1), trivial_id_to_key<member_t>(2));

    // Verify deleted key is invisible
    expect_equal(container.contains(trivial_id_to_key<member_t>(1)), false);
    expect_equal(container.size(), 1);
}

/** Edge Case: reset() clears watches and staged changes */
template <typename container_type_>
void check_reset_clears_transaction_state() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(container.upsert(trivial_id_to_member<member_t>(2)), success_k);

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;
    expect_equal(transaction.watch(trivial_id_to_key<member_t>(1)), success_k);
    // Key 4 is absent from the store, so its presence afterwards can only come from this discarded change
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(4)), success_k);

    // Reset the transaction
    expect_equal(transaction.reset(), success_k);

    // After reset, can stage/commit successfully (watches cleared)
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(3)), success_k);
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.commit(), success_k);

    auto const [committed_found, committed_found_status] = container.contains(trivial_id_to_key<member_t>(3));
    auto const [discarded_found, discarded_found_status] = container.contains(trivial_id_to_key<member_t>(4));
    if (!expect_equal(committed_found_status, success_k)) return;
    if (!expect_equal(discarded_found_status, success_k)) return;

    expect_true(!discarded_found, "reset must discard the staged key");
    expect_true(committed_found, "the reused transaction must commit its own key");
    expect_equal(container.contains(trivial_id_to_key<member_t>(2)), true, "reset must not touch committed entries");
}

/** Edge Case: commit() clears watches and recorded reads, so neither refuses the next round */
template <typename container_type_>
void check_commit_clears_transaction_state() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(container.upsert(trivial_id_to_member<member_t>(2)), success_k);

    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(transaction.watch(trivial_id_to_key<member_t>(1)), success_k);
    if constexpr (requires { transaction.for_each(no_op_t {}); })
        expect_equal(transaction.for_each(no_op_t {}), success_k);
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(3)), success_k);
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.commit(), success_k);
    expect_equal(transaction.contains(trivial_id_to_key<member_t>(3)), true, "a commit is read by its own writer");

    // The watched key moves after the commit that watched it, so no later round asked about it.
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 99)), success_k);

    // Enough keys to reach every partition of a sharded container, so no part escapes validation.
    for (trivial_id_t identifier = 100; identifier != 164; ++identifier)
        expect_equal(transaction.upsert(trivial_id_to_member<member_t>(identifier)), success_k);
    expect_equal(transaction.stage(), success_k, "a watch or a read from a committed round refused the next one");
    expect_equal(transaction.commit(), success_k);
    expect_equal(container.contains(trivial_id_to_key<member_t>(163)), true);
}

/**
 *  @brief Write Conflicts: A transaction reused after its own commit still conflicts with a rival
 *
 *  A reused transaction restages a key it already published. Filed under the generation of its
 *  first commit, the restaged version replaces the published one, and the rival below finds
 *  nothing newer than what it read.
 *
 *  @verbatim
 *  T1:  upsert(1,20)  →  commit  →  upsert(1,30)  →  stage()
 *
 *  T2:  opened before T1's commit  →  upsert(1,40)  →  stage()   ✗ must fail, T1 committed key 1
 *  @endverbatim
 *
 *  The second script is the watch: T2 watches key 1 between T1's two commits of it, so the second
 *  commit is the move the watch was taken against.
 */
template <typename container_type_>
void check_reused_committer_still_conflicts() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    if constexpr (at_least(container_t::isolation_k, isolation_t::snapshot_k)) {
        container_t container;
        expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 10)), success_k);

        auto [reused, reused_status] = container.transaction();
        if (!expect_equal(reused_status, success_k)) return;
        auto [rival, rival_status] = container.transaction();
        if (!expect_equal(rival_status, success_k)) return;

        expect_equal(reused.upsert(trivial_id_to_member<member_t>(1, 20)), success_k);
        expect_equal(reused.stage(), success_k);
        expect_equal(reused.commit(), success_k);
        expect_equal(reused.upsert(trivial_id_to_member<member_t>(1, 30)), success_k);
        expect_equal(reused.stage(), success_k);

        expect_equal(rival.upsert(trivial_id_to_member<member_t>(1, 40)), success_k);
        status_t const staged = rival.stage();
        expect_equal(succeeded(staged) ? rival.commit() : staged, status_t::write_conflict_k,
                     "the first committer wins, however often it commits");
    }

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 10)), success_k);

    auto [reused, reused_status] = container.transaction();
    if (!expect_equal(reused_status, success_k)) return;
    expect_equal(reused.upsert(trivial_id_to_member<member_t>(1, 20)), success_k);
    expect_equal(reused.stage(), success_k);
    expect_equal(reused.commit(), success_k);

    auto [rival, rival_status] = container.transaction();
    if (!expect_equal(rival_status, success_k)) return;
    expect_equal(rival.watch(trivial_id_to_key<member_t>(1)), success_k);

    expect_equal(reused.upsert(trivial_id_to_member<member_t>(1, 30)), success_k);
    expect_equal(reused.stage(), success_k);
    expect_equal(reused.commit(), success_k);

    expect_equal(rival.upsert(trivial_id_to_member<member_t>(2, 5)), success_k);
    status_t const staged = rival.stage();
    expect_equal(succeeded(staged) ? rival.commit() : staged, status_t::read_conflict_k,
                 "a watch must see the second commit of a reused transaction");
}

/**
 *  @brief Write Conflicts: Watch validation must detect staged (invisible) writes
 *
 *  This test verifies the fix for the bug where transactional_avl_tree used find() instead of
 *  find_latest_for_watch() during stage validation. The bug allowed two concurrent transactions to
 *  both stage successfully, violating Monotonic Atomic View consistency.
 *
 *  Timeline:
 *
 *  @verbatim
 *  T1:  watch(1)  →  ...
 *
 *  T2:  watch(1)  →  upsert(1,999)  →  stage()    [now invisible but staged]
 *
 *  T1:  ...  →  upsert(1,777)  →  stage()        ✗ must fail, T2 has a staged change
 *  @endverbatim
 *
 *  Expected: T1's stage() fails with consistency_k because T2 has staged (but not committed) a
 *  conflicting write to the watched key.
 */
template <typename container_type_>
void check_watch_detects_staged_invisible_writes() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);

    auto [first, first_status] = container.transaction();
    if (!expect_equal(first_status, success_k)) return;
    auto [second, second_status] = container.transaction();
    if (!expect_equal(second_status, success_k)) return;

    // Both watch the same key at generation 1, value 100
    expect_equal(first.watch(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(second.watch(trivial_id_to_key<member_t>(1)), success_k);

    // Both stage: a write nobody has committed is not yet binding on anyone, so neither is refused
    // here. Refusing at this point would abort a transaction whose rival may still roll back.
    expect_equal(second.upsert(trivial_id_to_member<member_t>(1, 999)), success_k);
    expect_equal(second.stage(), success_k);
    expect_equal(first.upsert(trivial_id_to_member<member_t>(1, 777)), success_k);
    expect_equal(first.stage(), success_k);

    // T2 publishes first, so T1's watch does not match what is committed.
    expect_equal(second.commit(), success_k);
    auto status = first.commit();
    expect_true(failed(status), "committing must detect the conflicting commit");
    expect_equal(status, status_t::read_conflict_k);

    auto [final_member, final_member_status] = container.find_copy(trivial_id_to_key<member_t>(1));
    if (!expect_equal(final_member_status, success_k)) return;
    expect_equal((final_member.mapped), (999), "the accepted writer's value must be the one that lands");
}

/**
 *  @brief A staged write must be honoured even when its generation sits below a published one.
 *
 *  Its sibling above stages from the newer transaction, so the staged write is also the highest
 *  generation and any ranking finds it. The dangerous order is the reverse: a generation is handed
 *  out when a transaction opens, so one that opens early and stages late carries a number below a
 *  version another transaction has already published. Ranking versions by generation looks straight
 *  past that staged write, and two transactions commit from the same base - which is exactly what a
 *  contended retry loop produces, and what a counter ends up one short of.
 */
template <typename container_type_>
void check_watch_detects_staged_writes_of_older_generation() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);

    // `early` opens first, so every version it stages carries the lower generation.
    auto [early, early_status] = container.transaction();
    if (!expect_equal(early_status, success_k)) return;
    auto [later, later_status] = container.transaction();
    if (!expect_equal(later_status, success_k)) return;

    // `later` publishes first, lifting the visible version's generation above `early`'s.
    expect_equal(later.watch(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(later.upsert(trivial_id_to_member<member_t>(1, 101)), success_k);
    expect_equal(later.stage(), success_k);
    expect_equal(later.commit(), success_k);

    // `early` watches what is now visible and stages beneath it.
    expect_equal(early.watch(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(early.upsert(trivial_id_to_member<member_t>(1, 102)), success_k);
    expect_equal(early.stage(), success_k);

    // A transaction opening after all of that sees the same visible version `early` watched, so
    // nothing in its own view says the key is spoken for. The staged write is what must refuse it.
    auto [newcomer, newcomer_status] = container.transaction();
    if (!expect_equal(newcomer_status, success_k)) return;
    expect_equal(newcomer.watch(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(newcomer.upsert(trivial_id_to_member<member_t>(1, 202)), success_k);
    expect_equal(newcomer.stage(), success_k);

    // The commit stamp, not the generation, decides: whoever publishes first wins, and the loser
    // is turned away even though its generation is the higher of the two.
    expect_equal(early.commit(), success_k);
    auto const status = newcomer.commit();
    expect_true(failed(status), "a commit over a watched key must refuse the second writer");
    expect_equal(status, status_t::read_conflict_k);

    auto [final_member, final_member_status] = container.find_copy(trivial_id_to_key<member_t>(1));
    if (!expect_equal(final_member_status, success_k)) return;
    expect_equal((final_member.mapped), (102), "the accepted writer's value must be the one that lands");
}

/** Builds a container around a given comparator, whichever factory shape its family offers.
 *  Partitioned collections pair the comparator with a hasher, trees with an allocator, and some
 *  take it alone. Only the arity differs, so the choice is made here rather than in every test. */
template <typename container_type_, typename comparator_type_>
auto make_around_comparator(comparator_type_ const &comparator) noexcept {
    if constexpr (requires { container_type_::make(comparator, typename container_type_::hash_t {}); })
        return container_type_::make(comparator, typename container_type_::hash_t {});
    else if constexpr (requires { container_type_::make(comparator, typename container_type_::allocator_t {}); })
        return container_type_::make(comparator, typename container_type_::allocator_t {});
    else return container_type_::make(comparator);
}

/**
 *  @brief A comparator carrying state must be consulted, never rebuilt.
 *
 *  Handed a descending comparator, the container has to walk high to low. Every transactional
 *  container routes its comparisons through one wrapper, and a wrapper that default-constructs a
 *  fresh comparator per call silently discards the instance it was given - so the walk comes back
 *  ascending and the container quietly orders by something nobody asked for.
 */
template <typename container_type_>
void check_stateful_comparator_is_consulted() {
    using member_t = typename container_type_::value_t;
    using comparator_t = typename container_type_::comparator_t;

    constexpr std::size_t keys_count_k = 3;
    // Above every key, so the first strict successor under a descending order is the largest key.
    constexpr std::size_t above_every_key_k = 1000;

    auto [container, status] =
        make_around_comparator<container_type_>(comparator_t {stateful_comparator_ordering_t::descending_k});
    if (!expect_equal(status, success_k, "a container must be constructible around a comparator instance")) return;

    for (std::size_t identifier = 1; identifier <= keys_count_k; ++identifier)
        expect_equal(container.upsert(trivial_id_to_member<member_t>(identifier)), success_k);
    expect_equal(container.size(), keys_count_k);

    std::vector<std::size_t> walked;
    auto cursor = trivial_id_to_key<member_t>(above_every_key_k);
    for (std::size_t step = 0; step != keys_count_k; ++step) {
        bool advanced = false;
        expect_equal(container.upper_bound(
                         cursor,
                         [&](member_t const &element) noexcept {
                             auto const &key = mapping_key_or_itself<member_t>(element);
                             walked.push_back(static_cast<std::size_t>(key.unique_id));
                             cursor = key;
                             advanced = true;
                         },
                         []() noexcept {}),
                     success_k);
        expect_true(advanced, "every stored key must be reachable by walking successors");
    }

    // An ignored comparator orders ascending, so the walk would read 1, 2, 3 instead.
    std::vector<std::size_t> const descending {3, 2, 1};
    expect_equal(walked, descending, "the comparator the container was given must decide the order");
}

#pragma region Transaction Lifetime and Watches

/**
 *  @brief A transaction that stages and is then abandoned must leave the store as it found it.
 *
 *  Staged versions live inside the store, invisible, tagged with the transaction's generation. If
 *  nothing unwinds them they stay there for good, unreachable and uncounted, so the destructor has
 *  to do it. Nothing else in the public surface can observe them, which is why this counts entries
 *  through a second transaction rather than through @c size.
 */
template <typename container_type_>
void check_abandoned_transaction_leaves_no_trace() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);

    {
        auto [abandoned, status] = container.transaction();
        if (!expect_equal(status, success_k)) return;
        expect_equal(abandoned.upsert(trivial_id_to_member<member_t>(2, 200)), success_k);
        expect_equal(abandoned.stage(), success_k);
        // Falls out of scope neither committed nor rolled back.
    }

    // A later transaction writing the same key must find nothing of the abandoned one in its way.
    auto [follower, follower_status] = container.transaction();
    if (!expect_equal(follower_status, success_k)) return;
    expect_equal(follower.upsert(trivial_id_to_member<member_t>(2, 222)), success_k);
    expect_equal(follower.stage(), success_k);
    expect_equal(follower.commit(), success_k);

    auto [final_member, final_member_status] = container.find_copy(trivial_id_to_key<member_t>(2));
    if (!expect_equal(final_member_status, success_k)) return;
    expect_equal((final_member.mapped), (222), "the abandoned version must not shadow a later write");
}

/** Moving a staged transaction must hand the claim over, not duplicate it. A defaulted move would
 *  leave both halves pointing at the store, and the destructor would unstage them twice. */
template <typename container_type_>
void check_moved_transaction_unwinds_once() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    {
        auto [original, status] = container.transaction();
        if (!expect_equal(status, success_k)) return;
        expect_equal(original.upsert(trivial_id_to_member<member_t>(5, 500)), success_k);
        expect_equal(original.stage(), success_k);
        auto moved = std::move(original);
        // Both halves are destroyed here; only the one holding the claim may unwind.
    }

    auto [follower, follower_status] = container.transaction();
    if (!expect_equal(follower_status, success_k)) return;
    expect_equal(follower.upsert(trivial_id_to_member<member_t>(5, 555)), success_k);
    expect_equal(follower.stage(), success_k);
    expect_equal(follower.commit(), success_k);

    auto [final_member, final_member_status] = container.find_copy(trivial_id_to_key<member_t>(5));
    if (!expect_equal(final_member_status, success_k)) return;
    expect_equal((final_member.mapped), (555), "a moved transaction must unwind exactly once");
}

/**
 *  @brief A plain read joins no read set, and the watching variant does.
 *
 *  The read set is opt-in because it is memory, and a read that allocates is a read that can fail -
 *  which the callback surface has no way to report. That makes the omission easy to walk into, so
 *  it is pinned here from both sides: reading a key and writing back over a concurrent commit is a
 *  lost update the store accepts, and the same shape through @c find_and_watch is one it refuses.
 */
template <typename container_type_>
void check_find_does_not_watch() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    // A plain read records nothing, so the transaction commits over a value it never saw.
    {
        container_t container;
        expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);

        auto [reader, status] = container.transaction();
        if (!expect_equal(status, success_k)) return;
        expect_equal(
            reader.find(trivial_id_to_key<member_t>(1), [](member_t const &) noexcept {}, []() noexcept {}), success_k);

        expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 999)), success_k);
        expect_equal(reader.upsert(trivial_id_to_member<member_t>(1, 101)), success_k);
        expect_true(succeeded(reader.stage()), "an unwatched read cannot refuse anything");
        expect_true(succeeded(reader.commit()), "an unwatched read cannot refuse anything");
    }

    // The same shape through `find_and_watch` records what it saw, and the drift is caught.
    {
        container_t container;
        expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);

        auto [reader, status] = container.transaction();
        if (!expect_equal(status, success_k)) return;
        expect_equal(reader.find_and_watch(
                         trivial_id_to_key<member_t>(1), [](member_t const &) noexcept {}, []() noexcept {}),
                     success_k);

        expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 999)), success_k);
        expect_equal(reader.upsert(trivial_id_to_member<member_t>(1, 101)), success_k);

        auto const staged = reader.stage();
        auto const committed = succeeded(staged) ? reader.commit() : staged;
        expect_true(failed(committed), "a watched read must refuse a write over a newer commit");
        expect_equal(committed, status_t::read_conflict_k);
    }
}

/**
 *  @brief Watching a key that was erased and committed must not, by itself, refuse a commit.
 *
 *  A committed erase leaves a dated tombstone. The watch has to record the same shape that
 *  validation will later resolve the key to, or it can never match itself and every transaction
 *  touching an erased key conflicts with nothing at all.
 */
template <typename container_type_>
void check_watch_on_erased_key_can_commit() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(3, 300)), success_k);

    {
        auto [remover, remover_status] = container.transaction();
        if (!expect_equal(remover_status, success_k)) return;
        expect_equal(remover.erase(trivial_id_to_key<member_t>(3)), success_k);
        expect_equal(remover.stage(), success_k);
        expect_equal(remover.commit(), success_k);
    }

    auto [observer, observer_status] = container.transaction();
    if (!expect_equal(observer_status, success_k)) return;
    expect_equal(observer.watch(trivial_id_to_key<member_t>(3)), success_k);
    expect_equal(observer.upsert(trivial_id_to_member<member_t>(4, 400)), success_k);
    auto status = observer.stage();
    expect_true(succeeded(status), "watching an erased key must not conflict when nothing moved");
    expect_equal(observer.commit(), success_k);
}

/**
 *  @brief A watch on an absent key must refuse once that key is inserted and erased under it.
 *
 *  Both ends resolve to absence, so a watch comparing only presence matches itself across two
 *  commits it never saw. What separates them is the generation the committed tombstone carries,
 *  which is why absence is dated rather than spelled as one constant.
 *
 *  @warning An erase taken outside a transaction drops the entry rather than tombstoning it, so
 *      this schedule is invisible below @c snapshot_k when the outsider writes to the store itself.
 */
template <typename container_type_>
void check_watch_catches_an_insert_and_erase_under_it() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);

    auto [observer, observer_status] = container.transaction();
    if (!expect_equal(observer_status, success_k)) return;
    expect_equal(observer.watch(trivial_id_to_key<member_t>(7)), success_k);

    {
        auto [outsider, outsider_status] = container.transaction();
        if (!expect_equal(outsider_status, success_k)) return;
        expect_equal(outsider.upsert(trivial_id_to_member<member_t>(7, 700)), success_k);
        expect_equal(outsider.stage(), success_k);
        expect_equal(outsider.commit(), success_k);
    }
    {
        auto [outsider, outsider_status] = container.transaction();
        if (!expect_equal(outsider_status, success_k)) return;
        expect_equal(outsider.erase(trivial_id_to_key<member_t>(7)), success_k);
        expect_equal(outsider.stage(), success_k);
        expect_equal(outsider.commit(), success_k);
    }

    expect_equal(observer.upsert(trivial_id_to_member<member_t>(1, 200)), success_k);
    expect_equal(observer.stage(), status_t::read_conflict_k,
                 "two commits landed under a watched absence, which it must not read as unchanged");
}

/** A watch on an absent key must still match itself after a rollback. Rollback hands the
 *  transaction a new generation, so anchoring absence to the transaction's own generation would
 *  silently invalidate every absent watch it deliberately preserves. */
template <typename container_type_>
void check_absent_watch_survives_rollback() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;
    expect_equal(transaction.watch(trivial_id_to_key<member_t>(9)), success_k);
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(8, 800)), success_k);
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.rollback(), success_k);

    auto status = transaction.stage();
    expect_true(succeeded(status), "an absent watch must survive the rollback that preserved it");
    expect_equal(transaction.commit(), success_k);

    auto [final_member, final_member_status] = container.find_copy(trivial_id_to_key<member_t>(8));
    if (!expect_equal(final_member_status, success_k)) return;
    expect_equal((final_member.mapped), (800), "the rolled-back write must still commit");
}

#pragma endregion Transaction Lifetime and Watches

#pragma region Transaction Groups

/** Two stores committed by one group become visible together, and neither before the commit. Both
 *  participants are the same type here, which is also the case a compile-time ordering could not
 *  have separated - the group orders by store address for exactly that reason. */
template <typename container_type_>
void check_group_commits_participants_together() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t first, second;
    {
        auto [group, status] = make_transaction_group(first, second);
        if (!expect_equal(status, success_k, "a group over two live stores must open")) return;
        expect_equal(group.template participant<0>().upsert(trivial_id_to_member<member_t>(1, 100)), success_k);
        expect_equal(group.template participant<1>().upsert(trivial_id_to_member<member_t>(2, 200)), success_k);

        expect_equal(group.stage(), success_k);
        expect_true(!first.find_copy(trivial_id_to_key<member_t>(1)).has_value(), "a staged write must stay invisible");
        expect_true(!second.find_copy(trivial_id_to_key<member_t>(2)).has_value(),
                    "a staged write must stay invisible");

        expect_equal(group.commit(), success_k);
    }

    expect_true(first.find_copy(trivial_id_to_key<member_t>(1)).has_value());
    expect_true(second.find_copy(trivial_id_to_key<member_t>(2)).has_value());
}

/** Two stores in one order publish a group under one stamp: the watermark moves once, every
 *  participant reports that stamp, and one on another order is refused before anything opens. */
template <typename container_type_>
void test_group_publishes_under_one_stamp() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using order_t = typename container_t::order_t;
    static_assert(transaction_group<container_t, container_t>::shares_one_order_k,
                  "two stores in one order must take the one-stamp branch");

    order_t order;
    container_t first {order}, second {order}, apart;

    generation_t const before = order.published_stamp();
    auto [group, status] = make_transaction_group(first, second);
    if (!expect_equal(status, success_k, "a group over two stores in one order must open")) return;
    auto [ids, names] = group.participants();
    expect_equal(ids.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);
    expect_equal(names.upsert(trivial_id_to_member<member_t>(2, 200)), success_k);
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);

    expect_equal(order.published_stamp(), before + 1, "one commit draws one stamp, however many stores it spans");
    expect_equal(group.commit_stamp(), before + 1);
    expect_equal(ids.commit_stamp(), group.commit_stamp());
    expect_equal(names.commit_stamp(), group.commit_stamp());
    expect_true(first.find_copy(trivial_id_to_key<member_t>(1)).has_value());
    expect_true(second.find_copy(trivial_id_to_key<member_t>(2)).has_value());

    expect_equal(make_transaction_group(first, apart).status(), status_t::invalid_argument_k,
                 "a participant on another order cannot share a stamp");
}

/** @c commit_with runs the body over the participants and then stages and commits, and resets what
 *  refused. */
template <typename container_type_>
void test_group_commit_with_runs_the_body() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using order_t = typename container_t::order_t;

    order_t order;
    container_t first {order}, second {order};

    {
        auto [group, status] = make_transaction_group(first, second);
        if (!expect_equal(status, success_k)) return;
        expect_equal(group.commit_with([&](auto &ids, auto &names) noexcept {
            status_t const one = ids.upsert(trivial_id_to_member<member_t>(1, 100));
            return failed(one) ? one : names.upsert(trivial_id_to_member<member_t>(2, 200));
        }),
                     success_k);
    }
    expect_true(first.find_copy(trivial_id_to_key<member_t>(1)).has_value());
    expect_true(second.find_copy(trivial_id_to_key<member_t>(2)).has_value());

    // A body that refuses publishes nothing and leaves no participant staged.
    {
        auto [group, status] = make_transaction_group(first, second);
        if (!expect_equal(status, success_k)) return;
        expect_equal(group.commit_with([&](auto &ids, auto &) noexcept {
            [[maybe_unused]] status_t const staged = ids.upsert(trivial_id_to_member<member_t>(3, 300));
            return status_t::invalid_argument_k;
        }),
                     status_t::invalid_argument_k);
        expect_true(group.staging() == staging_t::pending_k);
    }
    expect_true(!first.find_copy(trivial_id_to_key<member_t>(3)).has_value(), "a refused body writes nothing");
}

/**
 *  @brief Drives one refused group stage, with @p refusing_index_ naming the refusing participant.
 *
 *  The caller picks that index from the store addresses rather than from the argument order,
 *  because the group stages ascending by address and only a refusal above the first one leaves a
 *  staged prefix for the undo to walk.
 */
template <std::size_t refusing_index_, typename container_type_>
void check_group_unwind_refused_at(container_type_ &first, container_type_ &second) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    constexpr std::size_t surviving_index_k = 1 - refusing_index_;

    container_t &refusing = refusing_index_ == 0 ? first : second;
    container_t &surviving = surviving_index_k == 0 ? first : second;
    expect_equal(refusing.upsert(trivial_id_to_member<member_t>(7, 700)), success_k);

    auto [group, group_status] = make_transaction_group(first, second);
    if (!expect_equal(group_status, success_k)) return;
    expect_equal(group.template participant<refusing_index_>().watch(trivial_id_to_key<member_t>(7)), success_k);
    expect_equal(group.template participant<surviving_index_k>().upsert(trivial_id_to_member<member_t>(3, 300)),
                 success_k);
    expect_equal(group.template participant<refusing_index_>().upsert(trivial_id_to_member<member_t>(4, 400)),
                 success_k);

    // Move the watched key from outside, so this group's stage must be refused.
    {
        auto [interloper, interloper_status] = refusing.transaction();
        if (!expect_equal(interloper_status, success_k)) return;
        expect_equal(interloper.upsert(trivial_id_to_member<member_t>(7, 777)), success_k);
        expect_equal(interloper.stage(), success_k);
        expect_equal(interloper.commit(), success_k);
    }

    auto status = group.stage();
    expect_true(failed(status), "a moved watch must refuse the whole group");
    expect_equal(status, status_t::read_conflict_k);

    // Nothing may be left staged in the participant that did succeed.
    expect_true(!surviving.find_copy(trivial_id_to_key<member_t>(3)).has_value(),
                "a refused group must leave no participant staged");
    expect_true(!refusing.find_copy(trivial_id_to_key<member_t>(4)).has_value(),
                "a refused group must leave no participant staged");

    // A reset would have emptied the change set, and leaving the prefix staged would have moved it
    // into the store, so a pending write here is what says the undo rolled back.
    expect_true(group.template participant<surviving_index_k>().changes_count() != std::size_t {0},
                "an unwound participant must keep the writes it staged");

    // Retrying is what the surviving writes were kept for, sharded or not - a rollback keeps the
    // partition marks precisely so the next stage has somewhere to put them.
    expect_equal(group.template participant<refusing_index_>().reset(), success_k);
    expect_equal(group.stage(), success_k);
    expect_equal(group.commit(), success_k);
    auto [republished, republished_status] = surviving.find_copy(trivial_id_to_key<member_t>(3));
    if (!expect_equal(republished_status, success_k, "a rolled-back write must publish on the retry")) return;
    expect_equal((republished.mapped), (300), "a rolled-back write must publish the value it carried");
}

/**
 *  @brief When one participant refuses to stage, no participant is left staged.
 *
 *  The undo rolls the staged prefix back rather than resetting it, so the writes the caller made
 *  are still pending afterwards and the group can be retried without rebuilding them.
 */
template <typename container_type_>
void check_group_unwinds_every_participant_on_conflict() {

    container_type_ first, second;

    // Ordered by the same total order the group's own constructor asks for, since a builtin `>` over
    // pointers into unrelated objects is unspecified and would leave the undo untested on some layouts.
    if (std::compare_three_way {}(static_cast<void const *>(&second), static_cast<void const *>(&first)) > 0)
        check_group_unwind_refused_at<1>(first, second);
    else check_group_unwind_refused_at<0>(first, second);
}

#pragma endregion Transaction Groups

} // namespace ashvardanian::smashtable::test
