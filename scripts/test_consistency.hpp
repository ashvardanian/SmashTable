/**
 *  @brief Template test functions for advanced transactional consistency scenarios. Includes tests for isolation
 *      levels, visibility rules, and conflict resolution.
 *  @author Ash Vardanian
 *  @file scripts/test_consistency.hpp
 *  @date January 12, 2023
 */
#pragma once
#include <utility> // `std::move`
#include <vector>  // `std::vector`

#include "test_basic.hpp"
#include "test_failure_policy.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Isolation Expectations

/**
 *  @brief The level at which a read repeated inside one transaction must return what it first saw.
 *    Below it a container is free to show the newer committed value, and the suite asserts that it
 *    does - an anomaly a level permits is a promise that level makes, not an outcome left open.
 */
inline constexpr isolation_t repeatable_reads_from_k = isolation_t::snapshot_k;

/** @brief The level at which a predicate repeated inside one transaction must see the same members. */
inline constexpr isolation_t stable_predicates_from_k = isolation_t::snapshot_k;

#pragma endregion Isolation Expectations

/**
 *  @brief Edge Case: Committing empty transaction succeeds
 */
template <typename container_type_>
void test_empty_transaction_commit() {

    using container_t = container_type_;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto transaction = container.transaction();
    st_verify_(transaction.has_value());

    // Don't add anything, just commit
    st_verify_(transaction->stage());
    st_verify_(transaction->commit());

    st_verify_eq_(container.size(), 0);
}

template <typename container_type_>
void test_no_dirty_reads_multi_key() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    // Initial state: keys 1-5 exist
    container_t container;
    for (std::size_t i = 1; i <= 5; ++i) {
        auto new_member = trivial_id_to_member<member_t>(i, i);
        st_verify_(container.upsert(std::move(new_member)));
    }

    // T1 modifies all 5 keys but only stages (doesn't commit)
    auto t1 = container.transaction();
    st_verify_(t1.has_value());
    for (std::size_t i = 1; i <= 5; ++i) {
        auto new_member = trivial_id_to_member<member_t>(i, i * 100);
        st_verify_(t1->upsert(std::move(new_member)));
    }
    st_verify_(t1->stage());

    // External reader should see ORIGINAL values (staged changes invisible)
    for (std::size_t i = 1; i <= 5; ++i) {
        auto maybe_found = container.find_copy(trivial_id_to_key<member_t>(i));
        st_verify_((maybe_found) && "key must exist before the transaction stages over it");
        st_verify_((maybe_found->mapped) == (static_cast<decltype(maybe_found->mapped)>(i)) &&
                   "a reader outside must see the original, not the staged value");
    }
}

/**
 *  @brief Read Committed: New transactions don't see staged (uncommitted) data
 *
 *  Timeline:
 *    T1:  upsert(1,100)  →  upsert(2,200)  →  stage()
 *
 *    T2:  transaction()  →  find(1,2)  →  sees nothing
 *
 *  Expected: T2 sees nothing (T1 not committed yet)
 */
template <typename container_type_>
void test_new_transaction_sees_nothing_staged() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto t1 = container.transaction();
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(1)));
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(2)));
    st_verify_(t1->stage()); // Staged but not committed

    // T2 created AFTER T1 staged - should see nothing
    auto t2 = container.transaction();
    st_verify_(t2.has_value());

    auto maybe1 = t2->find_copy(trivial_id_to_key<member_t>(1));
    auto maybe2 = t2->find_copy(trivial_id_to_key<member_t>(2));

    st_verify_((!maybe1) && "a later transaction must not see a staged key");
    st_verify_((!maybe2) && "a later transaction must not see a staged key");
}

/**
 *  @brief Read Committed: Committed transactions are immediately visible
 *
 *  Timeline:
 *    T1:  upsert(1,111)  →  upsert(2,222)  →  stage()  →  commit()
 *
 *    T2:  transaction()  →  find(1,2)  →  sees (1,111) and (2,222)
 *
 *  Expected: T2 sees committed data immediately
 */
template <typename container_type_>
void test_committed_immediately_visible() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto t1 = container.transaction();
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(1, 111)));
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(2, 222)));
    st_verify_(t1->stage());
    st_verify_(t1->commit()); // NOW committed

    // T2 created AFTER commit - should see everything
    auto t2 = container.transaction();
    st_verify_(t2.has_value());

    auto maybe1 = t2->find_copy(trivial_id_to_key<member_t>(1));
    auto maybe2 = t2->find_copy(trivial_id_to_key<member_t>(2));

    st_verify_((maybe1) && "a later transaction must see a committed key");
    st_verify_((maybe2) && "a later transaction must see a committed key");
    st_verify_eq_(maybe1->mapped, 111);
    st_verify_eq_(maybe2->mapped, 222);
}

/**
 *  @brief Atomic View: 10-key transaction visible as 0 or 10, never partial
 *
 *  Timeline:
 *    T.upsert(0..9)
 *    ─────────────────────────────────
 *    Observer sees: 0 keys
 *
 *    T.stage()
 *    ─────────────────────────────────
 *    Observer sees: 0 keys
 *
 *    T.commit()
 *    ─────────────────────────────────
 *    Observer sees: 10 keys (atomic!)
 *
 *  Expected: Never see 1-9 keys (all-or-nothing)
 */
template <typename container_type_>
void test_multi_key_atomicity_10_keys() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto transaction = container.transaction();
    st_verify_(transaction.has_value());

    // Insert 10 keys in transaction
    for (std::size_t i = 0; i < 10; ++i) st_verify_(transaction->upsert(trivial_id_to_member<member_t>(i, i * 10)));

    // Before stage: should see 0
    std::size_t count_before_stage = 0;
    for (std::size_t i = 0; i < 10; ++i) count_before_stage += container.count(trivial_id_to_key<member_t>(i));
    st_verify_eq_(count_before_stage, 0);
    st_verify_(transaction->stage());

    // After stage, before commit: should see 0
    std::size_t count_after_stage = 0;
    for (std::size_t i = 0; i < 10; ++i) count_after_stage += container.count(trivial_id_to_key<member_t>(i));
    st_verify_eq_(count_after_stage, 0);
    st_verify_(transaction->commit());

    // After commit: should see ALL 10
    std::size_t count_after_commit = 0;
    for (std::size_t i = 0; i < 10; ++i) count_after_commit += container.count(trivial_id_to_key<member_t>(i));
    st_verify_((count_after_commit) == (10) && "commit must publish every key at once");
}

/**
 *  @brief Atomic View: Rollback atomically hides all staged changes
 *
 *  Setup:
 *    set.upsert(1,1)
 *
 *  Timeline:
 *    T:  upsert(1,100)  →  upsert(2,200)  →  upsert(3,300)  →  stage()  →  rollback()
 *
 *    Observer: ────────────────────────────────────────────────────────────────────────→
 *              sees (1,1) only, NOT (2,3)
 *
 *  Expected: All staged changes rolled back atomically
 */
template <typename container_type_>
void test_rollback_makes_all_invisible() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 1))); // Initial state

    auto transaction = container.transaction();
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(1, 100))); // Modify existing
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(2, 200))); // Add new
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(3, 300))); // Add new
    st_verify_(transaction->stage());
    st_verify_(transaction->rollback()); // ROLLBACK instead of commit

    // Key 1 should have original value
    auto maybe1 = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_((maybe1) && "rollback must leave the committed entry in place");
    st_verify_((maybe1->mapped) == (1) && "rollback must restore the original value");

    // Keys 2 and 3 should not exist
    auto maybe2 = container.find_copy(trivial_id_to_key<member_t>(2));
    auto maybe3 = container.find_copy(trivial_id_to_key<member_t>(3));
    st_verify_((!maybe2) && "rollback must discard the staged key");
    st_verify_((!maybe3) && "rollback must discard the staged key");
}

/**
 *  @brief Atomic View: Range queries see complete transactions
 *
 *  Timeline:
 *    T.upsert(10..19)
 *    ─────────────────────────────────
 *    range(10,20) sees: 0 elements
 *
 *    T.stage()
 *    ─────────────────────────────────
 *    range(10,20) sees: 0 elements
 *
 *    T.commit()
 *    ─────────────────────────────────
 *    range(10,20) sees: 10 elements (atomic!)
 *
 *  Expected: Range sees 0 or 10, never partial
 */
template <typename container_type_>
void test_range_query_sees_atomic_boundaries() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto transaction = container.transaction();
    for (std::size_t i = 10; i < 20; ++i) st_verify_(transaction->upsert(trivial_id_to_member<member_t>(i)));

    // Before commit: range query sees 0
    std::size_t count_before = 0;
    container.range(trivial_id_to_key<member_t>(10), trivial_id_to_key<member_t>(20),
                    [&](member_t const &) noexcept { count_before++; });
    st_verify_eq_(count_before, 0);

    st_verify_(transaction->stage());
    st_verify_(transaction->commit());

    // After commit: range query sees ALL 10
    std::size_t count_after = 0;
    container.range(trivial_id_to_key<member_t>(10), trivial_id_to_key<member_t>(20),
                    [&](member_t const &) noexcept { count_after++; });
    st_verify_eq_(count_after, 10);
}

/**
 *  @brief Atomic View: Interleaved commits don't create fractured reads
 *
 *  Timeline:
 *    T1.upsert(1,2,3)  →  T1.stage()
 *    T2.upsert(4,5,6)  →  T2.stage()
 *    ─────────────────────────────────
 *    Observer sees: 0 keys
 *
 *    T1.commit()
 *    ─────────────────────────────────
 *    Observer sees: 3 keys (1,2,3 only)
 *
 *    T2.commit()
 *    ─────────────────────────────────
 *    Observer sees: 6 keys (all)
 *
 *  Expected: No partial views across transactions
 */
template <typename container_type_>
void test_fractured_read_prevention() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;

    // T1 will insert keys 1-3
    auto t1 = container.transaction();
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(1, 10)));
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(2, 20)));
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(3, 30)));

    // T2 will insert keys 4-6
    auto t2 = container.transaction();
    st_verify_(t2->upsert(trivial_id_to_member<member_t>(4, 40)));
    st_verify_(t2->upsert(trivial_id_to_member<member_t>(5, 50)));
    st_verify_(t2->upsert(trivial_id_to_member<member_t>(6, 60)));

    // Stage both
    st_verify_(t1->stage());
    st_verify_(t2->stage());

    // Before any commits: see 0
    std::size_t count_initial = 0;
    for (std::size_t i = 1; i <= 6; ++i)
        container.find(trivial_id_to_key<member_t>(i), [&](member_t const &) noexcept { count_initial++; });
    st_verify_eq_(count_initial, 0);
    st_verify_(t1->commit()); // Commit T1

    // Should see exactly T1's keys (1-3), not T2's (4-6)
    std::size_t count_t1 = 0;
    for (std::size_t i = 1; i <= 6; ++i)
        container.find(trivial_id_to_key<member_t>(i), [&](member_t const &) noexcept { count_t1++; });
    st_verify_((count_t1) == (3) && "a range must not see another transaction staged keys");
    st_verify_(t2->commit()); // Commit T2

    // Now should see ALL 6
    std::size_t count_both = 0;
    for (std::size_t i = 1; i <= 6; ++i)
        container.find(trivial_id_to_key<member_t>(i), [&](member_t const &) noexcept { count_both++; });
    st_verify_((count_both) == (6) && "a range must see both committed transactions");
}

/**
 *  @brief Monotonic View: Values never go backwards (10→20→30, never 30→20)
 */
template <typename container_type_>
void test_sequential_updates_never_regress() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");
    static_assert(std::is_integral<typename member_t::mapped_type>::value, "Mapped type must be integral");

    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 10)));

    std::vector<int> observed_values;

    // Observe initial value
    container.find(trivial_id_to_key<member_t>(1),
                   [&](member_t const &e) noexcept { observed_values.push_back(e.mapped); });

    // Update to 20
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 20)));
    container.find(trivial_id_to_key<member_t>(1),
                   [&](member_t const &e) noexcept { observed_values.push_back(e.mapped); });

    // Update to 30
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 30)));
    container.find(trivial_id_to_key<member_t>(1),
                   [&](member_t const &e) noexcept { observed_values.push_back(e.mapped); });

    // Verify monotonicity: each value >= previous
    st_verify_eq_(observed_values.size(), 3);
    st_verify_eq_(observed_values[0], 10);
    st_verify_eq_(observed_values[1], 20);
    st_verify_eq_(observed_values[2], 30);

    for (size_t i = 1; i < observed_values.size(); ++i)
        st_verify_(((observed_values[i]) >= (observed_values[i - 1])) &&
                   "monotonic violation, the value went backwards");
}

/**
 *  @brief Monotonic View: T1 commits v1, T2 commits v2 → always see v1→v2, never v2→v1
 */
template <typename container_type_>
void test_transaction_commits_maintain_order() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    // T1: value = 100
    auto t1 = container.transaction();
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(1, 100)));
    st_verify_(t1->stage());
    st_verify_(t1->commit());

    // Observe T1's value
    auto val1 = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_(val1.has_value());
    st_verify_eq_(val1->mapped, 100);

    // T2: value = 200 (higher)
    auto t2 = container.transaction();
    st_verify_(t2->upsert(trivial_id_to_member<member_t>(1, 200)));
    st_verify_(t2->stage());
    st_verify_(t2->commit());

    // Observe T2's value - should be >= T1's value
    auto val2 = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_(val2.has_value());
    st_verify_eq_(val2->mapped, 200);
    st_verify_(((val2->mapped) >= (val1->mapped)) && "monotonic violation across transactions");
}

/**
 *  @brief Write Conflicts: Concurrent transactions on same key - first wins, second fails
 *
 *  Timeline:
 *    T1:  watch(1)  →  upsert(1,100)  →  stage()  →  commit()    ✓
 *
 *    T2:  watch(1)  →  upsert(1,200)  →  stage()                 ✗ CONFLICT
 *
 *  Expected: T1 succeeds, T2 fails with consistency_k
 */
template <typename container_type_>
void test_concurrent_transactions_on_same_key() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 1)));

    auto t1 = container.transaction();
    auto t2 = container.transaction();

    // Both watch the same key
    st_verify_(t1->watch(trivial_id_to_key<member_t>(1)));
    st_verify_(t2->watch(trivial_id_to_key<member_t>(1)));

    // Both modify it
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(1, 100)));
    st_verify_(t2->upsert(trivial_id_to_member<member_t>(1, 200)));

    // T1 commits successfully
    st_verify_(t1->stage());
    st_verify_(t1->commit());

    // T2's stage should FAIL (watched value changed)
    auto status = t2->stage();
    st_verify_((failed(status)) && "staging must fail once a watched key was modified");
    st_verify_eq_(status, status_t::consistency_k);

    // Verify T1's value persisted, T2's did not
    auto maybe_final = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_(maybe_final.has_value());
    st_verify_((maybe_final->mapped) == (100) && "only the winning transaction value may persist");
}

/**
 *  @brief Write Conflicts: ANY watched key conflict fails entire transaction
 *
 *  Timeline:
 *    T1:  watch(1,2,3)
 *
 *    External:  set.upsert(2,999)  ← modifies key 2
 *
 *    T1:  upsert(1,2,3)  →  stage()    ✗ CONFLICT on key 2
 *
 *  Expected: Transaction fails if ANY watched key changed
 */
template <typename container_type_>
void test_multi_key_conflict_any_key_fails() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 1)));
    st_verify_(container.upsert(trivial_id_to_member<member_t>(2, 2)));
    st_verify_(container.upsert(trivial_id_to_member<member_t>(3, 3)));

    auto t1 = container.transaction();
    auto t2 = container.transaction();

    // T1 watches keys 1, 2, 3
    st_verify_(t1->watch(trivial_id_to_key<member_t>(1)));
    st_verify_(t1->watch(trivial_id_to_key<member_t>(2)));
    st_verify_(t1->watch(trivial_id_to_key<member_t>(3)));

    // T2 watches same keys
    st_verify_(t2->watch(trivial_id_to_key<member_t>(1)));
    st_verify_(t2->watch(trivial_id_to_key<member_t>(2)));
    st_verify_(t2->watch(trivial_id_to_key<member_t>(3)));

    // External update to just ONE key (key 2)
    st_verify_(container.upsert(trivial_id_to_member<member_t>(2, 999)));

    // T1 modifies all three
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(1, 10)));
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(2, 20)));
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(3, 30)));

    // T1's stage should FAIL (key 2 was modified externally)
    auto status = t1->stage();
    st_verify_((failed(status)) && "staging must fail if any watched key changed");
    st_verify_eq_(status, status_t::consistency_k);
}

/**
 *  @brief Write Conflicts: Watches detect direct set.upsert() modifications
 *
 *  Timeline:
 *    T:  watch(1)
 *
 *    External:  set.upsert(1,777)  ← direct modification
 *
 *    T:  upsert(1,888)  →  stage()    ✗ CONFLICT
 *
 *  Expected: Watch detects external change
 */
template <typename container_type_>
void test_watch_detects_external_direct_modification() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 1)));

    auto transaction = container.transaction();
    st_verify_(transaction->watch(trivial_id_to_key<member_t>(1)));

    // Direct modification to the set (not through a transaction)
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 777)));

    // Transaction attempts to modify
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(1, 888)));

    // Stage should detect the external change
    auto status = transaction->stage();
    st_verify_((failed(status)) && "a watch must detect a direct write to the store");
    st_verify_eq_(status, status_t::consistency_k);
}

/**
 *  @brief Write Conflicts: Disjoint key sets allow both transactions to succeed
 *
 *  Timeline:
 *    T1:  upsert(1,2,3)  →  stage()  →  commit()    ✓
 *
 *    T2:  upsert(4,5,6)  →  stage()  →  commit()    ✓
 *
 *  Expected: Both succeed (no overlapping keys)
 */
template <typename container_type_>
void test_disjoint_keys_both_succeed() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto t1 = container.transaction();
    auto t2 = container.transaction();

    // T1 modifies keys 1-3
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(1, 10)));
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(2, 20)));
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(3, 30)));

    // T2 modifies keys 4-6 (disjoint!)
    st_verify_(t2->upsert(trivial_id_to_member<member_t>(4, 40)));
    st_verify_(t2->upsert(trivial_id_to_member<member_t>(5, 50)));
    st_verify_(t2->upsert(trivial_id_to_member<member_t>(6, 60)));

    // Both should succeed
    st_verify_(t1->stage());
    st_verify_(t2->stage());
    st_verify_(t1->commit());
    st_verify_(t2->commit());

    // Verify all 6 keys exist
    std::size_t count = 0;
    for (std::size_t i = 1; i <= 6; ++i) count += container.count(trivial_id_to_key<member_t>(i));
    st_verify_((count) == (6) && "disjoint transactions must both commit");
}

/**
 *  @brief Allowed Anomaly: Non-repeatable reads are CORRECT for Read Committed
 *
 *  Setup:
 *    set.upsert(1,100)
 *
 *  Timeline:
 *    T:  find(1)  →  sees 100
 *
 *    External:  set.upsert(1,999)
 *
 *    T:  find(1)  →  sees 999    ← CORRECT! (not a bug)
 *
 *  Expected: Within transaction, reads CAN see different values
 */
template <typename container_type_>
void test_repeated_read_matches_isolation() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 100)));

    auto transaction = container.transaction();

    // First read
    auto first_read = transaction->find_copy(trivial_id_to_key<member_t>(1));
    st_verify_(first_read.has_value());
    st_verify_eq_(first_read->mapped, 100);

    // External modification
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 999)));

    // Second read in SAME transaction - CAN see new value (this is correct!)
    auto second_read = transaction->find_copy(trivial_id_to_key<member_t>(1));
    st_verify_(second_read.has_value());

    if constexpr (at_least(container_t::isolation_k, repeatable_reads_from_k)) {
        st_verify_((second_read->mapped) == (100) && "a snapshot must repeat its first read");
        st_verify_eq_(first_read->mapped, second_read->mapped);
    }
    else {
        st_verify_((second_read->mapped) == (999) && "below snapshot, a read sees the newest commit");
        st_verify_ne_(first_read->mapped, second_read->mapped);
    }
}

/**
 *  @brief Allowed Anomaly: Phantom reads are CORRECT for Read Committed
 *
 *  Setup:
 *    set.upsert(0..4)
 *
 *  Timeline:
 *    T:  range(0,10)  →  counts 5
 *
 *    External:  set.upsert(5,6)
 *
 *    T:  range(0,10)  →  counts 7    ← CORRECT! (not a bug)
 *
 *  Expected: Range counts can change within transaction
 */
template <typename container_type_>
void test_repeated_range_matches_isolation() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t i = 0; i < 5; ++i) st_verify_(container.upsert(trivial_id_to_member<member_t>(i)));

    auto transaction = container.transaction();

    // The predicate is evaluated through the transaction, not through `container`. Asking the store
    // would be a question about the store rather than about the transaction's view, and would hold at
    // every isolation level including serializable - which is why the earlier shape of this test could
    // not fail. It is spelled with `find` rather than a range scan because that is the one ordered-free
    // surface every transaction type here offers, `partitioned_store`'s included.
    auto count_present = [&]() noexcept {
        std::size_t present = 0;
        for (std::size_t candidate = 0; candidate != 10; ++candidate)
            if (transaction->contains(trivial_id_to_key<member_t>(candidate))) ++present;
        return present;
    };

    std::size_t const first_count = count_present();
    st_verify_eq_(first_count, 5);

    // Committed from outside the transaction, after the first evaluation and before the second.
    st_verify_(container.upsert(trivial_id_to_member<member_t>(5, 5)));
    st_verify_(container.upsert(trivial_id_to_member<member_t>(6, 6)));

    std::size_t const second_count = count_present();

    if constexpr (at_least(container_t::isolation_k, stable_predicates_from_k)) {
        st_verify_((second_count) == (5) && "a snapshot must not admit phantoms");
        st_verify_eq_(first_count, second_count);
    }
    else {
        st_verify_((second_count) == (7) && "below snapshot, a repeated predicate sees the newest commits");
        st_verify_((second_count) > (first_count));
    }
}

/**
 *  @brief Edge Case: Deleted entries are invisible to queries
 */
template <typename container_type_>
void test_delete_visibility() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1)));
    st_verify_(container.upsert(trivial_id_to_member<member_t>(2)));

    // Delete via erase_range
    erase_range_of(container, trivial_id_to_key<member_t>(1), trivial_id_to_key<member_t>(2));

    // Verify deleted key is invisible
    st_verify_(!(container.contains(trivial_id_to_key<member_t>(1))));
    st_verify_eq_(container.size(), 1);
}

/**
 *  @brief Edge Case: reset() clears watches and staged changes
 */
template <typename container_type_>
void test_reset_clears_transaction_state() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1)));
    st_verify_(container.upsert(trivial_id_to_member<member_t>(2)));

    auto transaction = container.transaction();
    st_verify_(transaction->watch(trivial_id_to_key<member_t>(1)));
    // Key 4 is absent from the store, so its presence afterwards can only come from this discarded change
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(4)));

    // Reset the transaction
    st_verify_(transaction->reset());

    // After reset, can stage/commit successfully (watches cleared)
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(3)));
    st_verify_(transaction->stage());
    st_verify_(transaction->commit());

    auto found3 = container.contains(trivial_id_to_key<member_t>(3));
    auto found4 = container.contains(trivial_id_to_key<member_t>(4));

    st_verify_((!found4) && "reset must discard the staged key");
    st_verify_((found3) && "the reused transaction must commit its own key");
    st_verify_((container.contains(trivial_id_to_key<member_t>(2))) && "reset must not touch committed entries");
}

/**
 *  @brief Write Conflicts: Watch validation must detect staged (invisible) writes
 *
 *  This test verifies the fix for the bug where transactional_avl_tree used find() instead of
 *  find_latest_for_watch() during stage validation. The bug allowed two concurrent
 *  transactions to both stage successfully, violating Monotonic Atomic View consistency.
 *
 *  Timeline:
 *    T1:  watch(1)  →  ...
 *
 *    T2:  watch(1)  →  upsert(1,999)  →  stage()    [now invisible but staged]
 *
 *    T1:  ...  →  upsert(1,777)  →  stage()        ✗ MUST FAIL (T2 has staged change)
 *
 *  Expected: T1's stage() fails with consistency_k because T2 has staged (but not committed)
 *            a conflicting write to the watched key.
 */
template <typename container_type_>
void test_watch_detects_staged_invisible_writes() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 100)));

    auto t1 = container.transaction();
    auto t2 = container.transaction();

    // Both watch the same key at generation 1, value 100
    st_verify_(t1->watch(trivial_id_to_key<member_t>(1)));
    st_verify_(t2->watch(trivial_id_to_key<member_t>(1)));

    // Both stage: a write nobody has committed is not yet binding on anyone, so neither is refused
    // here. Refusing at this point would abort a transaction whose rival may still roll back.
    st_verify_(t2->upsert(trivial_id_to_member<member_t>(1, 999)));
    st_verify_(t2->stage());
    st_verify_(t1->upsert(trivial_id_to_member<member_t>(1, 777)));
    st_verify_(t1->stage());

    // T2 publishes first, so T1's watch does not match what is committed.
    st_verify_(t2->commit());
    auto status = t1->commit();
    st_verify_((failed(status)) && "committing must detect the conflicting commit");
    st_verify_eq_(status, status_t::consistency_k);

    auto maybe_final = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_(maybe_final.has_value());
    st_verify_((maybe_final->mapped) == (999) && "the accepted writer's value must be the one that lands");
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
void test_watch_detects_staged_writes_of_older_generation() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 100)));

    // `early` opens first, so every version it stages carries the lower generation.
    auto early = container.transaction();
    auto later = container.transaction();

    // `later` publishes first, lifting the visible version's generation above `early`'s.
    st_verify_(later->watch(trivial_id_to_key<member_t>(1)));
    st_verify_(later->upsert(trivial_id_to_member<member_t>(1, 101)));
    st_verify_(later->stage());
    st_verify_(later->commit());

    // `early` watches what is now visible and stages beneath it.
    st_verify_(early->watch(trivial_id_to_key<member_t>(1)));
    st_verify_(early->upsert(trivial_id_to_member<member_t>(1, 102)));
    st_verify_(early->stage());

    // A transaction opening after all of that sees the same visible version `early` watched, so
    // nothing in its own view says the key is spoken for. The staged write is what must refuse it.
    auto newcomer = container.transaction();
    st_verify_(newcomer->watch(trivial_id_to_key<member_t>(1)));
    st_verify_(newcomer->upsert(trivial_id_to_member<member_t>(1, 202)));
    st_verify_(newcomer->stage());

    // The commit stamp, not the generation, decides: whoever publishes first wins, and the loser
    // is turned away even though its generation is the higher of the two.
    st_verify_(early->commit());
    auto const status = newcomer->commit();
    st_verify_((failed(status)) && "a commit over a watched key must refuse the second writer");
    st_verify_eq_(status, status_t::consistency_k);

    auto maybe_final = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_(maybe_final.has_value());
    st_verify_((maybe_final->mapped) == (102) && "the accepted writer's value must be the one that lands");
}

/**
 *  @brief Builds a container around a given comparator, whichever factory shape its family offers.
 *    Partitioned collections pair the comparator with a hasher, trees with an allocator, and some
 *    take it alone. Only the arity differs, so the choice is made here rather than in every test.
 */
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
void test_stateful_comparator_is_consulted() {
    using member_t = typename container_type_::value_t;
    using comparator_t = typename container_type_::comparator_t;

    constexpr std::size_t keys_count_k = 3;
    // Above every key, so the first strict successor under a descending order is the largest key.
    constexpr std::size_t above_every_key_k = 1000;

    auto built = make_around_comparator<container_type_>(comparator_t {ordering_t::descending_k});
    st_verify_(built.has_value() && "a container must be constructible around a comparator instance");
    auto &container = *built;

    for (std::size_t identifier = 1; identifier <= keys_count_k; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));
    st_verify_eq_(container.size(), keys_count_k);

    std::vector<std::size_t> walked;
    auto cursor = trivial_id_to_key<member_t>(above_every_key_k);
    for (std::size_t step = 0; step != keys_count_k; ++step) {
        bool advanced = false;
        container.upper_bound(
            cursor,
            [&](member_t const &element) noexcept {
                auto const &key = mapping_key_or_itself<member_t>(element);
                walked.push_back(static_cast<std::size_t>(key.unique_id));
                cursor = key;
                advanced = true;
            },
            []() noexcept {});
        st_verify_(advanced && "every stored key must be reachable by walking successors");
    }

    // An ignored comparator orders ascending, so the walk would read 1, 2, 3 instead.
    std::vector<std::size_t> const descending {3, 2, 1};
    st_verify_((walked == descending) && "the comparator the container was given must decide the order");
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
void test_abandoned_transaction_leaves_no_trace() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 100)));

    {
        auto abandoned = container.transaction();
        st_verify_(abandoned->upsert(trivial_id_to_member<member_t>(2, 200)));
        st_verify_(abandoned->stage());
        // Falls out of scope neither committed nor rolled back.
    }

    // A later transaction writing the same key must find nothing of the abandoned one in its way.
    auto follower = container.transaction();
    st_verify_(follower->upsert(trivial_id_to_member<member_t>(2, 222)));
    st_verify_(follower->stage());
    st_verify_(follower->commit());

    auto maybe_final = container.find_copy(trivial_id_to_key<member_t>(2));
    st_verify_(maybe_final.has_value());
    st_verify_((maybe_final->mapped) == (222) && "the abandoned version must not shadow a later write");
}

/**
 *  @brief Moving a staged transaction must hand the claim over, not duplicate it.
 *    A defaulted move would leave both halves pointing at the store, and the destructor would then
 *    unstage the same versions twice.
 */
template <typename container_type_>
void test_moved_transaction_unwinds_once() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    {
        auto original = container.transaction();
        st_verify_(original->upsert(trivial_id_to_member<member_t>(5, 500)));
        st_verify_(original->stage());
        auto moved = std::move(*original);
        // Both halves are destroyed here; only the one holding the claim may unwind.
    }

    auto follower = container.transaction();
    st_verify_(follower->upsert(trivial_id_to_member<member_t>(5, 555)));
    st_verify_(follower->stage());
    st_verify_(follower->commit());

    auto maybe_final = container.find_copy(trivial_id_to_key<member_t>(5));
    st_verify_(maybe_final.has_value());
    st_verify_((maybe_final->mapped) == (555) && "a moved transaction must unwind exactly once");
}

/**
 *  @brief A plain read joins no read set, and the watching variant does.
 *
 *  The read set is opt-in because it is memory, and a read that allocates is a read that can fail -
 *  which the callback surface has no way to report. That makes the omission easy to walk into, so it
 *  is pinned here from both sides: reading a key and writing back over a concurrent commit is a lost
 *  update the store will accept, and the same shape through @c find_and_watch is one it refuses.
 */
template <typename container_type_>
void test_find_does_not_watch() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    // A plain read records nothing, so the transaction commits over a value it never saw.
    {
        container_t container;
        st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 100)));

        auto reader = container.transaction();
        reader->find(trivial_id_to_key<member_t>(1), [](member_t const &) noexcept {}, []() noexcept {});

        st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 999)));
        st_verify_(reader->upsert(trivial_id_to_member<member_t>(1, 101)));
        st_verify_((succeeded(reader->stage())) && "an unwatched read cannot refuse anything");
        st_verify_((succeeded(reader->commit())) && "an unwatched read cannot refuse anything");
    }

    // The same shape through `find_and_watch` records what it saw, and the drift is caught.
    {
        container_t container;
        st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 100)));

        auto reader = container.transaction();
        st_verify_(
            reader->find_and_watch(trivial_id_to_key<member_t>(1), [](member_t const &) noexcept {}, []() noexcept {}));

        st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 999)));
        st_verify_(reader->upsert(trivial_id_to_member<member_t>(1, 101)));

        auto const staged = reader->stage();
        auto const committed = succeeded(staged) ? reader->commit() : staged;
        st_verify_((failed(committed)) && "a watched read must refuse a write over a newer commit");
        st_verify_eq_(committed, status_t::consistency_k);
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
void test_watch_on_erased_key_can_commit() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(3, 300)));

    {
        auto remover = container.transaction();
        st_verify_(remover->erase(trivial_id_to_key<member_t>(3)));
        st_verify_(remover->stage());
        st_verify_(remover->commit());
    }

    auto observer = container.transaction();
    st_verify_(observer->watch(trivial_id_to_key<member_t>(3)));
    st_verify_(observer->upsert(trivial_id_to_member<member_t>(4, 400)));
    auto status = observer->stage();
    st_verify_((succeeded(status)) && "watching an erased key must not conflict when nothing moved");
    st_verify_(observer->commit());
}

/**
 *  @brief A watch on an absent key must still match itself after a rollback.
 *    Rollback hands the transaction a new generation, so anchoring absence to the transaction's own
 *    generation would silently invalidate every absent watch it deliberately preserves.
 */
template <typename container_type_>
void test_absent_watch_survives_rollback() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;

    auto transaction = container.transaction();
    st_verify_(transaction->watch(trivial_id_to_key<member_t>(9)));
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(8, 800)));
    st_verify_(transaction->stage());
    st_verify_(transaction->rollback());

    auto status = transaction->stage();
    st_verify_((succeeded(status)) && "an absent watch must survive the rollback that preserved it");
    st_verify_(transaction->commit());

    auto maybe_final = container.find_copy(trivial_id_to_key<member_t>(8));
    st_verify_(maybe_final.has_value());
    st_verify_((maybe_final->mapped) == (800) && "the rolled-back write must still commit");
}

#pragma endregion Transaction Lifetime and Watches

#pragma region Transaction Groups

/**
 *  @brief Two stores committed by one group become visible together, and neither before the commit.
 *    Both participants are the same type here, which is also the case a compile-time ordering could
 *    not have separated - the group orders by store address for exactly that reason.
 */
template <typename container_type_>
void test_group_commits_participants_together() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t first, second;
    {
        auto group = make_transaction_group(first, second);
        st_verify_((group.has_value()) && "a group over two live stores must open");
        st_verify_(group->template participant<0>().upsert(trivial_id_to_member<member_t>(1, 100)));
        st_verify_(group->template participant<1>().upsert(trivial_id_to_member<member_t>(2, 200)));

        st_verify_(group->stage());
        st_verify_((!first.find_copy(trivial_id_to_key<member_t>(1)).has_value()) &&
                   "a staged write must stay invisible");
        st_verify_((!second.find_copy(trivial_id_to_key<member_t>(2)).has_value()) &&
                   "a staged write must stay invisible");

        st_verify_(group->commit());
    }

    st_verify_(first.find_copy(trivial_id_to_key<member_t>(1)).has_value());
    st_verify_(second.find_copy(trivial_id_to_key<member_t>(2)).has_value());
}

/**
 *  @brief When one participant refuses to stage, no participant is left staged.
 *
 *  The undo rolls the staged prefix back rather than resetting it, so the writes the caller made are
 *  still pending afterwards and the group can be retried without rebuilding them.
 */
template <typename container_type_>
void test_group_unwinds_every_participant_on_conflict() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t first, second;
    st_verify_(second.upsert(trivial_id_to_member<member_t>(7, 700)));

    auto group = make_transaction_group(first, second);
    st_verify_(group.has_value());
    st_verify_(group->template participant<1>().watch(trivial_id_to_key<member_t>(7)));
    st_verify_(group->template participant<0>().upsert(trivial_id_to_member<member_t>(3, 300)));
    st_verify_(group->template participant<1>().upsert(trivial_id_to_member<member_t>(4, 400)));

    // Move the watched key from outside, so this group's stage must be refused.
    {
        auto interloper = second.transaction();
        st_verify_(interloper->upsert(trivial_id_to_member<member_t>(7, 777)));
        st_verify_(interloper->stage());
        st_verify_(interloper->commit());
    }

    auto status = group->stage();
    st_verify_((failed(status)) && "a moved watch must refuse the whole group");
    st_verify_eq_(status, status_t::consistency_k);

    // Nothing may be left staged in the participant that did succeed.
    st_verify_(group->reset());
    st_verify_((!first.find_copy(trivial_id_to_key<member_t>(3)).has_value()) &&
               "a refused group must leave no participant staged");
    st_verify_((!second.find_copy(trivial_id_to_key<member_t>(4)).has_value()) &&
               "a refused group must leave no participant staged");
}

#pragma endregion Transaction Groups

} // namespace ashvardanian::smashtable::scripts
