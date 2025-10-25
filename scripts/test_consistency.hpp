using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

template <typename container_type_>
void test_no_dirty_reads_multi_key() {
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();

    // Initial state: keys 1-5 exist
    for (std::size_t i = 1; i <= 5; ++i) { EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {i, i})); }

    // T1 modifies all 5 keys but only stages (doesn't commit)
    auto t1 = set.transaction();
    ASSERT_TRUE(t1.has_value());
    for (std::size_t i = 1; i <= 5; ++i) { EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {i, i * 100})); }
    EXPECT_TRUE(t1->stage());

    // External reader should see ORIGINAL values (staged changes invisible)
    for (std::size_t i = 1; i <= 5; ++i) {
        bool found = false;
        set.find(id_t {i}, [&](uint64_uint64_pair_t const &e) noexcept {
            found = true;
            EXPECT_EQ(e.value, i) << "Should see original value, not staged value";
        });
        EXPECT_TRUE(found) << "Key " << i << " should exist";
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();

    auto t1 = set.transaction();
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {1, 100}));
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {2, 200}));
    EXPECT_TRUE(t1->stage()); // Staged but not committed

    // T2 created AFTER T1 staged - should see nothing
    auto t2 = set.transaction();
    ASSERT_TRUE(t2.has_value());

    bool found1 = false, found2 = false;
    t2->find(id_t {1}, [&](uint64_uint64_pair_t const &) noexcept { found1 = true; });
    t2->find(id_t {2}, [&](uint64_uint64_pair_t const &) noexcept { found2 = true; });

    EXPECT_FALSE(found1) << "T2 should not see T1's staged key 1";
    EXPECT_FALSE(found2) << "T2 should not see T1's staged key 2";
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();

    auto t1 = set.transaction();
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {1, 111}));
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {2, 222}));
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t1->commit()); // NOW committed

    // T2 created AFTER commit - should see everything
    auto t2 = set.transaction();
    ASSERT_TRUE(t2.has_value());

    std::size_t val1 = 0, val2 = 0;
    bool found1 = false, found2 = false;

    t2->find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept {
        found1 = true;
        val1 = e.value;
    });
    t2->find(id_t {2}, [&](uint64_uint64_pair_t const &e) noexcept {
        found2 = true;
        val2 = e.value;
    });

    EXPECT_TRUE(found1) << "T2 should see committed key 1";
    EXPECT_TRUE(found2) << "T2 should see committed key 2";
    EXPECT_EQ(val1, 111);
    EXPECT_EQ(val2, 222);
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();

    auto txn = set.transaction();
    ASSERT_TRUE(txn.has_value());

    // Insert 10 keys in transaction
    for (std::size_t i = 0; i < 10; ++i) { EXPECT_TRUE(txn->upsert(uint64_uint64_pair_t {i, i * 10})); }

    // Before stage: should see 0
    int count_before_stage = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        set.find(id_t {i}, [&](uint64_uint64_pair_t const &) noexcept { count_before_stage++; });
    }
    EXPECT_EQ(count_before_stage, 0) << "Before stage: should see 0 keys";

    EXPECT_TRUE(txn->stage());

    // After stage, before commit: should see 0
    int count_after_stage = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        set.find(id_t {i}, [&](uint64_uint64_pair_t const &) noexcept { count_after_stage++; });
    }
    EXPECT_EQ(count_after_stage, 0) << "After stage, before commit: should see 0 keys";

    EXPECT_TRUE(txn->commit());

    // After commit: should see ALL 10
    int count_after_commit = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        set.find(id_t {i}, [&](uint64_uint64_pair_t const &) noexcept { count_after_commit++; });
    }
    EXPECT_EQ(count_after_commit, 10) << "After commit: should see ALL 10 keys atomically";
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 1}));

    auto txn = set.transaction();
    EXPECT_TRUE(txn->upsert(uint64_uint64_pair_t {1, 100})); // Modify existing
    EXPECT_TRUE(txn->upsert(uint64_uint64_pair_t {2, 200})); // Add new
    EXPECT_TRUE(txn->upsert(uint64_uint64_pair_t {3, 300})); // Add new
    EXPECT_TRUE(txn->stage());
    EXPECT_TRUE(txn->rollback()); // ROLLBACK instead of commit

    // Key 1 should have original value
    bool found1 = false;
    set.find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept {
        found1 = true;
        EXPECT_EQ(e.value, 1) << "Rollback should restore original value";
    });
    EXPECT_TRUE(found1);

    // Keys 2 and 3 should not exist
    bool found2 = false, found3 = false;
    set.find(id_t {2}, [&](uint64_uint64_pair_t const &) noexcept { found2 = true; });
    set.find(id_t {3}, [&](uint64_uint64_pair_t const &) noexcept { found3 = true; });

    EXPECT_FALSE(found2) << "Rolled back key 2 should not exist";
    EXPECT_FALSE(found3) << "Rolled back key 3 should not exist";
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();

    auto txn = set.transaction();
    for (std::size_t i = 10; i < 20; ++i) { EXPECT_TRUE(txn->upsert(uint64_uint64_pair_t {i, i})); }

    // Before commit: range query sees 0
    int count_before = 0;
    set.range(id_t {10}, id_t {20}, [&](uint64_uint64_pair_t const &) noexcept { count_before++; });
    EXPECT_EQ(count_before, 0) << "Range query before commit sees nothing";

    EXPECT_TRUE(txn->stage());
    EXPECT_TRUE(txn->commit());

    // After commit: range query sees ALL 10
    int count_after = 0;
    set.range(id_t {10}, id_t {20}, [&](uint64_uint64_pair_t const &) noexcept { count_after++; });
    EXPECT_EQ(count_after, 10) << "Range query after commit sees all atomically";
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();

    // T1 will insert keys 1-3
    auto t1 = set.transaction();
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {1, 10}));
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {2, 20}));
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {3, 30}));

    // T2 will insert keys 4-6
    auto t2 = set.transaction();
    EXPECT_TRUE(t2->upsert(uint64_uint64_pair_t {4, 40}));
    EXPECT_TRUE(t2->upsert(uint64_uint64_pair_t {5, 50}));
    EXPECT_TRUE(t2->upsert(uint64_uint64_pair_t {6, 60}));

    // Stage both
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t2->stage());

    // Before any commits: see 0
    int count_0 = 0;
    for (std::size_t i = 1; i <= 6; ++i) {
        set.find(id_t {i}, [&](uint64_uint64_pair_t const &) noexcept { count_0++; });
    }
    EXPECT_EQ(count_0, 0);

    // Commit T1
    EXPECT_TRUE(t1->commit());

    // Should see exactly T1's keys (1-3), not T2's (4-6)
    int count_t1 = 0;
    for (std::size_t i = 1; i <= 6; ++i) {
        set.find(id_t {i}, [&](uint64_uint64_pair_t const &) noexcept { count_t1++; });
    }
    EXPECT_EQ(count_t1, 3) << "Should see only T1's 3 keys, not T2's";

    // Commit T2
    EXPECT_TRUE(t2->commit());

    // Now should see ALL 6
    int count_both = 0;
    for (std::size_t i = 1; i <= 6; ++i) {
        set.find(id_t {i}, [&](uint64_uint64_pair_t const &) noexcept { count_both++; });
    }
    EXPECT_EQ(count_both, 6) << "Should see both transactions' keys";
}

/**
 *  @brief Monotonic View: Values never go backwards (10→20→30, never 30→20)
 */
template <typename container_type_>
void test_sequential_updates_never_regress() {
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 10}));

    std::vector<std::size_t> observed_values;

    // Observe initial value
    set.find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept { observed_values.push_back(e.value); });

    // Update to 20
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 20}));
    set.find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept { observed_values.push_back(e.value); });

    // Update to 30
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 30}));
    set.find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept { observed_values.push_back(e.value); });

    // Verify monotonicity: each value >= previous
    ASSERT_EQ(observed_values.size(), 3);
    EXPECT_EQ(observed_values[0], 10);
    EXPECT_EQ(observed_values[1], 20);
    EXPECT_EQ(observed_values[2], 30);

    for (size_t i = 1; i < observed_values.size(); ++i) {
        EXPECT_GE(observed_values[i], observed_values[i - 1]) << "Monotonic violation: value went backwards!";
    }
}

/**
 *  @brief Monotonic View: T1 commits v1, T2 commits v2 → always see v1→v2, never v2→v1
 */
template <typename container_type_>
void test_transaction_commits_maintain_order() {
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();

    // T1: value = 100
    auto t1 = set.transaction();
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {1, 100}));
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t1->commit());

    // Observe T1's value
    std::size_t val1 = 0;
    set.find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept { val1 = e.value; });
    EXPECT_EQ(val1, 100);

    // T2: value = 200 (higher)
    auto t2 = set.transaction();
    EXPECT_TRUE(t2->upsert(uint64_uint64_pair_t {1, 200}));
    EXPECT_TRUE(t2->stage());
    EXPECT_TRUE(t2->commit());

    // Observe T2's value - should be >= T1's value
    std::size_t val2 = 0;
    set.find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept { val2 = e.value; });
    EXPECT_EQ(val2, 200);
    EXPECT_GE(val2, val1) << "Monotonic violation across transactions!";
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 1}));

    auto t1 = set.transaction();
    auto t2 = set.transaction();

    // Both watch the same key
    EXPECT_TRUE(t1->watch(id_t {1}));
    EXPECT_TRUE(t2->watch(id_t {1}));

    // Both modify it
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {1, 100}));
    EXPECT_TRUE(t2->upsert(uint64_uint64_pair_t {1, 200}));

    // T1 commits successfully
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t1->commit());

    // T2's stage should FAIL (watched value changed)
    auto status = t2->stage();
    EXPECT_FALSE(status) << "T2 should fail - watched key was modified by T1";
    EXPECT_EQ(status.errc, errc_t::consistency_k);

    // Verify T1's value persisted, T2's did not
    std::size_t final_value = 0;
    set.find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept { final_value = e.value; });
    EXPECT_EQ(final_value, 100) << "Only T1's value should persist";
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 1}));
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {2, 2}));
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {3, 3}));

    auto t1 = set.transaction();
    auto t2 = set.transaction();

    // T1 watches keys 1, 2, 3
    EXPECT_TRUE(t1->watch(id_t {1}));
    EXPECT_TRUE(t1->watch(id_t {2}));
    EXPECT_TRUE(t1->watch(id_t {3}));

    // T2 watches same keys
    EXPECT_TRUE(t2->watch(id_t {1}));
    EXPECT_TRUE(t2->watch(id_t {2}));
    EXPECT_TRUE(t2->watch(id_t {3}));

    // External update to just ONE key (key 2)
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {2, 999}));

    // T1 modifies all three
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {1, 10}));
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {2, 20}));
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {3, 30}));

    // T1's stage should FAIL (key 2 was modified externally)
    auto status = t1->stage();
    EXPECT_FALSE(status) << "Transaction should fail if ANY watched key changed";
    EXPECT_EQ(status.errc, errc_t::consistency_k);
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 1}));

    auto txn = set.transaction();
    EXPECT_TRUE(txn->watch(id_t {1}));

    // Direct modification to the set (not through a transaction)
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 777}));

    // Transaction attempts to modify
    EXPECT_TRUE(txn->upsert(uint64_uint64_pair_t {1, 888}));

    // Stage should detect the external change
    auto status = txn->stage();
    EXPECT_FALSE(status) << "Watch should detect direct set.upsert()";
    EXPECT_EQ(status.errc, errc_t::consistency_k);
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();

    auto t1 = set.transaction();
    auto t2 = set.transaction();

    // T1 modifies keys 1-3
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {1, 10}));
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {2, 20}));
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {3, 30}));

    // T2 modifies keys 4-6 (disjoint!)
    EXPECT_TRUE(t2->upsert(uint64_uint64_pair_t {4, 40}));
    EXPECT_TRUE(t2->upsert(uint64_uint64_pair_t {5, 50}));
    EXPECT_TRUE(t2->upsert(uint64_uint64_pair_t {6, 60}));

    // Both should succeed
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t2->stage());
    EXPECT_TRUE(t1->commit());
    EXPECT_TRUE(t2->commit());

    // Verify all 6 keys exist
    int count = 0;
    for (std::size_t i = 1; i <= 6; ++i) {
        set.find(id_t {i}, [&](uint64_uint64_pair_t const &) noexcept { count++; });
    }
    EXPECT_EQ(count, 6) << "Both transactions should succeed with disjoint keys";
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
void test_non_repeatable_reads_are_allowed() {
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 100}));

    auto txn = set.transaction();

    // First read
    std::size_t first_read = 0;
    txn->find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept { first_read = e.value; });
    EXPECT_EQ(first_read, 100);

    // External modification
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 999}));

    // Second read in SAME transaction - CAN see new value (this is correct!)
    std::size_t second_read = 0;
    txn->find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept { second_read = e.value; });

    // With Read Committed, second read sees committed changes
    EXPECT_EQ(second_read, 999) << "Non-repeatable reads are ALLOWED in Read Committed";
    EXPECT_NE(first_read, second_read) << "This is correct behavior!";
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
void test_phantom_reads_are_allowed() {
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    for (std::size_t i = 0; i < 5; ++i) { EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {i, i})); }

    auto txn = set.transaction();

    // First range query: see 5 items (reads are on committed state)
    int first_count = 0;
    set.range(id_t {0}, id_t {10}, [&](uint64_uint64_pair_t const &) noexcept { first_count++; });
    EXPECT_EQ(first_count, 5);

    // External insert
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {5, 5}));
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {6, 6}));

    // Second range query while txn still active - CAN see new items (phantom reads)
    // In Read Committed, reads always see latest committed state
    int second_count = 0;
    set.range(id_t {0}, id_t {10}, [&](uint64_uint64_pair_t const &) noexcept { second_count++; });

    EXPECT_EQ(second_count, 7) << "Phantom reads are ALLOWED in Read Committed";
    EXPECT_GT(second_count, first_count) << "This is correct behavior!";
}

/**
 *  @brief Edge Case: Committing empty transaction succeeds
 */
template <typename container_type_>
void test_empty_transaction_commit() {
    auto set = *container_type_::make();

    auto txn = set.transaction();
    ASSERT_TRUE(txn.has_value());

    // Don't add anything, just commit
    EXPECT_TRUE(txn->stage());
    EXPECT_TRUE(txn->commit());

    EXPECT_EQ(set.size(), 0) << "Empty transaction should leave set empty";
}

/**
 *  @brief Edge Case: Deleted entries are invisible to queries
 */
template <typename container_type_>
void test_delete_visibility() {
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 1}));
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {2, 2}));

    // Delete via erase_range
    set.erase_range(id_t {1}, id_t {2});

    // Verify deleted key is invisible
    bool found = false;
    set.find(id_t {1}, [&](uint64_uint64_pair_t const &) noexcept { found = true; });
    EXPECT_FALSE(found) << "Deleted entry should not be visible";

    EXPECT_EQ(set.size(), 1) << "Should have 1 item (key 2)";
}

/**
 *  @brief Edge Case: reset() clears watches and staged changes
 */
template <typename container_type_>
void test_reset_clears_transaction_state() {
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 1}));

    auto txn = set.transaction();
    EXPECT_TRUE(txn->watch(id_t {1}));
    EXPECT_TRUE(txn->upsert(uint64_uint64_pair_t {2, 2}));

    // Reset the transaction
    EXPECT_TRUE(txn->reset());

    // After reset, can stage/commit successfully (watches cleared)
    EXPECT_TRUE(txn->upsert(uint64_uint64_pair_t {3, 3}));
    EXPECT_TRUE(txn->stage());
    EXPECT_TRUE(txn->commit());

    // Key 3 should exist, key 2 should not
    bool found2 = false, found3 = false;
    set.find(id_t {2}, [&](uint64_uint64_pair_t const &) noexcept { found2 = true; });
    set.find(id_t {3}, [&](uint64_uint64_pair_t const &) noexcept { found3 = true; });

    EXPECT_FALSE(found2) << "Reset should have cleared key 2";
    EXPECT_TRUE(found3) << "New transaction should have added key 3";
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
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {1, 100}));

    auto t1 = set.transaction();
    auto t2 = set.transaction();

    // Both watch the same key at generation 1, value 100
    EXPECT_TRUE(t1->watch(id_t {1}));
    EXPECT_TRUE(t2->watch(id_t {1}));

    // T2 modifies and stages (but doesn't commit)
    EXPECT_TRUE(t2->upsert(uint64_uint64_pair_t {1, 999}));
    EXPECT_TRUE(t2->stage()); // Now gen=2, visible=false

    // T1 should FAIL to stage because T2 has a staged (invisible) write
    // This tests that find_latest_for_watch() is used, not find()
    EXPECT_TRUE(t1->upsert(uint64_uint64_pair_t {1, 777}));
    auto status = t1->stage();
    EXPECT_FALSE(status) << "T1 should fail - T2 has staged invisible write on watched key";
    EXPECT_EQ(status.errc, errc_t::consistency_k);

    // Clean up: rollback T2, verify original value persists
    EXPECT_TRUE(t2->rollback());
    std::size_t final_value = 0;
    set.find(id_t {1}, [&](uint64_uint64_pair_t const &e) noexcept { final_value = e.value; });
    EXPECT_EQ(final_value, 100) << "Original value should persist after rollback";
}
