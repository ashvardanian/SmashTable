#include <iostream>
#include <cstdlib>
#include <thread>
#include <ctime>

#include <gtest/gtest.h>

#include <smashtable/transactional_std_set.hpp>
#include <smashtable/transactional_avl_tree.hpp>
#include <smashtable/partitioned_collection.hpp>

using namespace ashvardanian::smashtable;

constexpr std::size_t size = 128;

struct pair_t {
    std::size_t key;
    std::size_t value;

    pair_t(std::size_t key = 0, std::size_t value = 0) noexcept : key(key), value(value) {}
    explicit operator std::size_t() const noexcept { return key; }
    operator bool() const noexcept { return key != -1; }
};

struct pair_compare_t {
    using value_type = std::size_t;
    // Deliberately omit is_transparent to demonstrate heterogeneous lookups work without it
    bool operator()(pair_t a, pair_t b) const noexcept { return a.key < b.key; }
    bool operator()(std::size_t a, pair_t b) const noexcept { return a < b.key; }
    bool operator()(pair_t a, std::size_t b) const noexcept { return a.key < b; }
};

struct pair_compare_transparent_t {
    using value_type = std::size_t;
    using is_transparent = void; // Explicitly enable heterogeneous lookups (C++14 style)
    bool operator()(pair_t a, pair_t b) const noexcept { return a.key < b.key; }
    bool operator()(std::size_t a, pair_t b) const noexcept { return a < b.key; }
    bool operator()(pair_t a, std::size_t b) const noexcept { return a.key < b; }
};

// Element type for heterogeneous lookup tests (string-based)
struct string_element_t {
    std::string key;
    int value;
    string_element_t() noexcept = default;
    string_element_t(std::string k, int v = 0) : key(std::move(k)), value(v) {}
    explicit operator std::string_view() const noexcept { return key; }
    operator bool() const noexcept { return !key.empty(); }
};

// Comparator WITH is_transparent for heterogeneous lookups
struct string_compare_transparent_t {
    using value_type = std::string_view;
    using is_transparent = void;
    bool operator()(string_element_t const &a, string_element_t const &b) const noexcept { return a.key < b.key; }
    bool operator()(std::string_view a, string_element_t const &b) const noexcept { return a < b.key; }
    bool operator()(string_element_t const &a, std::string_view b) const noexcept { return a.key < b; }
};

// Comparator WITHOUT is_transparent (tests that SmashTable still enables heterogeneous lookups)
struct string_compare_t {
    using value_type = std::string_view;
    // Deliberately omit is_transparent
    bool operator()(string_element_t const &a, string_element_t const &b) const noexcept { return a.key < b.key; }
    bool operator()(std::string_view a, string_element_t const &b) const noexcept { return a < b.key; }
    bool operator()(string_element_t const &a, std::string_view b) const noexcept { return a.key < b; }
};

using stl_t = transactional_std_set<pair_t, pair_compare_t>;
using avl_t = transactional_avl_tree<pair_t, pair_compare_t>;
using partitioned_stl_t = partitioned_collection<stl_t>;
using partitioned_avl_t = partitioned_collection<avl_t>;
using id_stl_t = typename stl_t::identifier_t;
using id_avl_t = typename avl_t::identifier_t;

template <typename container_type_>
void test_with_threads(std::size_t threads_count) {
    auto cont = *container_type_::make();
    std::vector<std::thread> threads;
    threads.reserve(threads_count);

    auto upsert = [&](std::size_t offset, std::size_t length) {
        for (std::size_t idx = offset; idx < length; ++idx) EXPECT_TRUE(cont.upsert(pair_t {idx, idx}));
    };

    std::size_t shift = (size / threads_count);
    for (std::size_t idx = 0; idx < threads_count; ++idx)
        threads.push_back(std::thread(upsert, idx * shift, idx * shift + shift));

    for (std::size_t idx = 0; idx < threads_count; ++idx) threads[idx].join();

    EXPECT_EQ(cont.size(), size);
    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.find(idx, [](auto const &) noexcept {}));
}

// NOTE: Disabled - transactional_std_set and transactional_avl_tree are NOT thread-safe
// Use locked_collection<> or partitioned_collection<> wrappers for thread-safe access
// TEST(upsert_and_find_set, with_threads) {
//     test_with_threads<stl_t>(2);
//     test_with_threads<stl_t>(4);
//     test_with_threads<stl_t>(8);
//     test_with_threads<stl_t>(16);
//     test_with_threads<avl_t>(2);
//     test_with_threads<avl_t>(4);
//     test_with_threads<avl_t>(8);
//     test_with_threads<avl_t>(16);
// }

/**
 * @brief Tests insertion in ascending, descending, and random order (exercises AVL rebalancing)
 */
template <typename container_type_>
void test_basic_insertion_patterns() {
    using id_t = typename container_type_::identifier_t;
    auto cont = *container_type_::make();

    // Test 1: Ascending insertion
    for (std::size_t idx = 0; idx < size; ++idx) {
        EXPECT_TRUE(cont.upsert(pair_t {idx, idx}));
        EXPECT_TRUE(cont.find(idx, [](auto const &) noexcept {}));
    }
    EXPECT_EQ(cont.size(), size);
    EXPECT_TRUE(cont.clear());

    // Test 2: Descending insertion (tests AVL rebalancing)
    for (std::size_t idx = size; idx > 0; --idx) {
        EXPECT_TRUE(cont.upsert(pair_t {idx, idx}));
        EXPECT_TRUE(cont.find(idx, [](auto const &) noexcept {}));
    }
    EXPECT_EQ(cont.size(), size);
    EXPECT_TRUE(cont.clear());

    // Test 3: Random insertion (tests worst-case AVL patterns)
    std::srand(42); // Fixed seed for reproducibility
    for (std::size_t idx = 0; idx < size; ++idx) {
        std::size_t val = std::rand();
        EXPECT_TRUE(cont.upsert(pair_t {val, val}));
        EXPECT_TRUE(cont.find(val, [](auto const &) noexcept {}));
    }
}

/**
 *  @brief Tests bulk insertion via move iterators
 */
template <typename container_type_>
void test_bulk_insertion_from_iterators() {
    using id_t = typename container_type_::identifier_t;
    std::vector<pair_t> vec(size);
    auto cont = *container_type_::make();

    for (std::size_t idx = 0; idx < size; ++idx) vec[idx] = pair_t {idx, idx};

    EXPECT_TRUE(cont.upsert(std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end())));
    EXPECT_EQ(cont.size(), size);

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.find(idx, [](auto const &) noexcept {}));
}

/**
 *  @brief Tests range queries on committed HEAD state
 */
template <typename container_type_>
void test_range_query_head_state() {
    using id_t = typename container_type_::identifier_t;
    auto cont = *container_type_::make();

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.upsert(pair_t {idx, idx}));

    // Query in windows and verify we get elements within the range
    for (std::size_t idx = 0; idx < size; idx += 10) {
        std::size_t count = 0;
        std::size_t min_key = size, max_key = 0;
        EXPECT_TRUE(cont.range(idx, idx + 9, [&](auto const &rhs) noexcept {
            min_key = std::min(min_key, rhs.key);
            max_key = std::max(max_key, rhs.key);
            count++;
        }));
        EXPECT_GT(count, 0) << "Range query should return at least some elements";
        if (count > 0) {
            EXPECT_GE(min_key, idx) << "Min key should be >= lower bound";
            EXPECT_LE(max_key, idx + 10) << "Max key should be near upper bound";
        }
    }
}

/**
 *  @brief Tests `erase_range` on committed HEAD state
 */
template <typename container_type_>
void test_erase_range_head_state() {
    using id_t = typename container_type_::identifier_t;
    auto cont = *container_type_::make();

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.upsert(pair_t {idx, idx}));

    bool state = true;
    for (std::size_t idx = 0; idx < size; idx += 10) {
        EXPECT_TRUE(cont.erase_range(idx, idx + 10, [](auto const &) noexcept {}));
        for (std::size_t i = idx; i < idx + 10; ++i) {
            EXPECT_TRUE(cont.find(i, [&](auto const &) noexcept { state = false; }));
            EXPECT_TRUE(state);
        }
    }
}

/**
 *  @brief Tests `upper_bound` query returns first `element > key`
 */
template <typename container_type_>
void test_upper_bound() {
    using id_t = typename container_type_::identifier_t;
    auto cont = *container_type_::make();

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.upsert(pair_t {idx, idx}));

    for (std::size_t idx = 0; idx < size - 1; ++idx) {
        EXPECT_TRUE(cont.upper_bound(idx, [&](auto const &rhs) noexcept { EXPECT_TRUE(pair_compare_t {}(idx, rhs)); }));
    }
}

/**
 *  @brief Tests `clear` operation resets size to 0
 */
template <typename container_type_>
void test_clear() {
    auto cont = *container_type_::make();

    EXPECT_EQ(cont.size(), 0);

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.upsert(pair_t {idx, idx}));

    EXPECT_EQ(cont.size(), size);
    EXPECT_TRUE(cont.clear());
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests `size` increments correctly after each upsert
 */
template <typename container_type_>
void test_size_after_upserts() {
    auto cont = *container_type_::make();
    EXPECT_EQ(cont.size(), 0);

    for (std::size_t idx = 0; idx < 50; ++idx) {
        EXPECT_TRUE(cont.upsert(pair_t {idx, idx}));
        EXPECT_EQ(cont.size(), idx + 1);
    }
}

/**
 *  @brief Tests size remains unchanged when upserting duplicate keys
 */
template <typename container_type_>
void test_size_invariant_on_duplicate_upserts() {
    auto cont = *container_type_::make();

    // Insert initial keys
    for (std::size_t idx = 0; idx < 50; ++idx) { EXPECT_TRUE(cont.upsert(pair_t {idx, idx})); }
    EXPECT_EQ(cont.size(), 50);

    // Upsert same keys again with different values - size should remain same
    for (std::size_t idx = 0; idx < 50; ++idx) {
        EXPECT_TRUE(cont.upsert(pair_t {idx, idx * 2}));
        EXPECT_EQ(cont.size(), 50) << "Duplicate upsert should not change size";
    }
}

/**
 *  @brief Tests size decrements correctly after erase_range
 */
template <typename container_type_>
void test_size_after_erase() {
    using id_t = typename container_type_::identifier_t;
    auto cont = *container_type_::make();

    for (std::size_t idx = 0; idx < 50; ++idx) EXPECT_TRUE(cont.upsert(pair_t {idx, idx}));
    EXPECT_EQ(cont.size(), 50);

    // Erase in ranges
    for (std::size_t idx = 0; idx < 50; idx += 10) {
        EXPECT_TRUE(cont.erase_range(id_t {idx}, id_t {idx + 10}));
        EXPECT_EQ(cont.size(), 50 - (idx + 10));
    }
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests operations on empty container don't crash
 */
template <typename container_type_>
void test_empty_container_operations() {
    using id_t = typename container_type_::identifier_t;
    auto cont = *container_type_::make();

    // Operations on empty container should not crash
    bool found = false;
    EXPECT_TRUE(cont.find(id_t {1}, [&](auto const &) { found = true; }));
    EXPECT_FALSE(found);

    EXPECT_TRUE(cont.upper_bound(id_t {1}, [&](auto const &) { found = true; }));
    EXPECT_FALSE(found);

    EXPECT_TRUE(cont.erase_range(id_t {0}, id_t {10}));
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests operations on single-element container
 */
template <typename container_type_>
void test_single_element_operations() {
    using id_t = typename container_type_::identifier_t;
    auto cont = *container_type_::make();
    EXPECT_TRUE(cont.upsert(pair_t {42, 42}));
    EXPECT_EQ(cont.size(), 1);

    bool found = false;
    EXPECT_TRUE(cont.find(id_t {42}, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.value, 42);
    }));
    EXPECT_TRUE(found);

    EXPECT_TRUE(cont.erase_range(id_t {42}, id_t {43}));
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests reserve() pre-allocates capacity (STL containers only)
 *
 *  AVL trees don't support collection-level reserve
 */
template <typename container_type_>
void test_reserve() {
    auto cont = *container_type_::make();
    EXPECT_TRUE(cont.reserve(100));
    EXPECT_TRUE(cont.empty());
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests heterogeneous lookups with comparator that HAS is_transparent
 *
 *  Uses std::string elements with std::string_view lookups (truly distinct types, no implicit conversion)
 */
template <template <typename, typename> class base_collection_>
void test_heterogeneous_lookups_with_transparent() {
    using namespace std::literals;
    using set_t = base_collection_<string_element_t, string_compare_transparent_t>;
    auto set = *set_t::make();

    EXPECT_TRUE(set.insert(string_element_t {"10", 100}));
    EXPECT_TRUE(set.insert(string_element_t {"20", 200}));
    EXPECT_TRUE(set.insert(string_element_t {"30", 300}));

    // Test heterogeneous find()
    bool found = false;
    EXPECT_TRUE(set.find("20"sv, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.key, "20");
        EXPECT_EQ(e.element.value, 200);
    }));
    EXPECT_TRUE(found);

    // Test heterogeneous count()
    EXPECT_EQ(set.count("20"sv), 1);
    EXPECT_EQ(set.count("99"sv), 0);

    // Test heterogeneous lower_bound()
    found = false;
    EXPECT_TRUE(set.lower_bound("15"sv, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.key, "20");
    }));
    EXPECT_TRUE(found);

    // Test heterogeneous upper_bound()
    found = false;
    EXPECT_TRUE(set.upper_bound("20"sv, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.key, "30");
    }));
    EXPECT_TRUE(found);

    // Test heterogeneous equal_range()
    found = false;
    EXPECT_TRUE(set.equal_range("20"sv, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.key, "20");
    }));
    EXPECT_TRUE(found);
}

/**
 *  @brief Tests heterogeneous lookups with comparator that LACKS is_transparent
 *
 *  Verifies SmashTable enables heterogeneous lookups even without is_transparent
 */
template <template <typename, typename> class base_collection_>
void test_heterogeneous_lookups_without_transparent() {
    using namespace std::literals;
    using set_t = base_collection_<string_element_t, string_compare_t>;
    auto set = *set_t::make();

    EXPECT_TRUE(set.insert(string_element_t {"10", 100}));
    EXPECT_TRUE(set.insert(string_element_t {"20", 200}));
    EXPECT_TRUE(set.insert(string_element_t {"30", 300}));

    // Test heterogeneous find()
    bool found = false;
    EXPECT_TRUE(set.find("20"sv, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.key, "20");
        EXPECT_EQ(e.element.value, 200);
    }));
    EXPECT_TRUE(found);

    // Test heterogeneous count()
    EXPECT_EQ(set.count("20"sv), 1);
    EXPECT_EQ(set.count("99"sv), 0);

    // Test heterogeneous lower_bound()
    found = false;
    EXPECT_TRUE(set.lower_bound("15"sv, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.key, "20");
    }));
    EXPECT_TRUE(found);

    // Test heterogeneous upper_bound()
    found = false;
    EXPECT_TRUE(set.upper_bound("20"sv, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.key, "30");
    }));
    EXPECT_TRUE(found);

    // Test heterogeneous equal_range()
    found = false;
    EXPECT_TRUE(set.equal_range("20"sv, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.key, "20");
    }));
    EXPECT_TRUE(found);
}

/** @brief Tests insertion in ascending, descending, and random order (AVL balancing) */
TEST(basic_ops_standard_set, insertion_patterns) { test_basic_insertion_patterns<stl_t>(); }

/** @brief Tests bulk insertion from iterators (unique API) */
TEST(basic_ops_standard_set, bulk_insertion_iterators) { test_bulk_insertion_from_iterators<stl_t>(); }

/** @brief Tests range queries on committed HEAD state */
TEST(basic_ops_standard_set, range_query_head_state) { test_range_query_head_state<stl_t>(); }

/** @brief Tests erase_range on committed HEAD state */
TEST(basic_ops_standard_set, erase_range_head_state) { test_erase_range_head_state<stl_t>(); }

/** @brief Tests upper_bound query functionality */
TEST(basic_ops_standard_set, upper_bound) { test_upper_bound<stl_t>(); }

/** @brief Tests clear operation */
TEST(basic_ops_standard_set, clear) { test_clear<stl_t>(); }

/** @brief Tests size increments correctly after upserts */
TEST(basic_ops_standard_set, size_after_upserts) { test_size_after_upserts<stl_t>(); }

/** @brief Tests size invariant: duplicate upserts don't increment size */
TEST(basic_ops_standard_set, size_invariant_on_duplicates) { test_size_invariant_on_duplicate_upserts<stl_t>(); }

/** @brief Tests size decrements correctly after erase_range */
TEST(basic_ops_standard_set, size_after_erase) { test_size_after_erase<stl_t>(); }

/** @brief Tests operations on empty container don't crash */
TEST(basic_ops_standard_set, empty_container_operations) { test_empty_container_operations<stl_t>(); }

/** @brief Tests operations on single-element container */
TEST(basic_ops_standard_set, single_element_operations) { test_single_element_operations<stl_t>(); }

/** @brief Tests reserve() pre-allocates capacity (STL only) */
TEST(basic_ops_standard_set, reserve) { test_reserve<stl_t>(); }

/** @brief Tests insertion in ascending, descending, and random order (AVL balancing) */
TEST(basic_ops_avl_tree, insertion_patterns) { test_basic_insertion_patterns<avl_t>(); }

/** @brief Tests bulk insertion from iterators (unique API) */
TEST(basic_ops_avl_tree, bulk_insertion_iterators) { test_bulk_insertion_from_iterators<avl_t>(); }

/** @brief Tests range queries on committed HEAD state */
TEST(basic_ops_avl_tree, range_query_head_state) { test_range_query_head_state<avl_t>(); }

/** @brief Tests erase_range on committed HEAD state */
TEST(basic_ops_avl_tree, erase_range_head_state) { test_erase_range_head_state<avl_t>(); }

/** @brief Tests upper_bound query functionality */
TEST(basic_ops_avl_tree, upper_bound) { test_upper_bound<avl_t>(); }

/** @brief Tests clear operation */
TEST(basic_ops_avl_tree, clear) { test_clear<avl_t>(); }

/** @brief Tests size increments correctly after upserts */
TEST(basic_ops_avl_tree, size_after_upserts) { test_size_after_upserts<avl_t>(); }

/** @brief Tests size invariant: duplicate upserts don't increment size */
TEST(basic_ops_avl_tree, size_invariant_on_duplicates) { test_size_invariant_on_duplicate_upserts<avl_t>(); }

/** @brief Tests size decrements correctly after erase_range */
TEST(basic_ops_avl_tree, size_after_erase) { test_size_after_erase<avl_t>(); }

/** @brief Tests operations on empty container don't crash */
TEST(basic_ops_avl_tree, empty_container_operations) { test_empty_container_operations<avl_t>(); }

/** @brief Tests operations on single-element container */
TEST(basic_ops_avl_tree, single_element_operations) { test_single_element_operations<avl_t>(); }

/**
 *  @brief Read Committed: Staged transactions are invisible to external readers
 *
 *  Setup:
 *    set.upsert(1..5)
 *
 *  Timeline:
 *    T1:       upsert(1..5, *100)  →  stage()
 *
 *    Observer: ────────────────────────────────→  sees original values (1..5)
 *
 *  Expected: Staged changes invisible until commit
 */
template <typename container_type_>
void test_no_dirty_reads_multi_key() {
    using id_t = typename container_type_::identifier_t;
    auto set = *container_type_::make();

    // Initial state: keys 1-5 exist
    for (std::size_t i = 1; i <= 5; ++i) { EXPECT_TRUE(set.upsert(pair_t {i, i})); }

    // T1 modifies all 5 keys but only stages (doesn't commit)
    auto t1 = set.transaction();
    ASSERT_TRUE(t1.has_value());
    for (std::size_t i = 1; i <= 5; ++i) { EXPECT_TRUE(t1->upsert(pair_t {i, i * 100})); }
    EXPECT_TRUE(t1->stage());

    // External reader should see ORIGINAL values (staged changes invisible)
    for (std::size_t i = 1; i <= 5; ++i) {
        bool found = false;
        EXPECT_TRUE(set.find(id_t {i}, [&](auto const &e) {
            found = true;
            EXPECT_EQ(e.element.value, i) << "Should see original value, not staged value";
        }));
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
    EXPECT_TRUE(t1->upsert(pair_t {1, 100}));
    EXPECT_TRUE(t1->upsert(pair_t {2, 200}));
    EXPECT_TRUE(t1->stage()); // Staged but not committed

    // T2 created AFTER T1 staged - should see nothing
    auto t2 = set.transaction();
    ASSERT_TRUE(t2.has_value());

    bool found1 = false, found2 = false;
    EXPECT_TRUE(t2->find(id_t {1}, [&](auto const &) { found1 = true; }));
    EXPECT_TRUE(t2->find(id_t {2}, [&](auto const &) { found2 = true; }));

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
    EXPECT_TRUE(t1->upsert(pair_t {1, 111}));
    EXPECT_TRUE(t1->upsert(pair_t {2, 222}));
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t1->commit()); // NOW committed

    // T2 created AFTER commit - should see everything
    auto t2 = set.transaction();
    ASSERT_TRUE(t2.has_value());

    std::size_t val1 = 0, val2 = 0;
    bool found1 = false, found2 = false;

    EXPECT_TRUE(t2->find(id_t {1}, [&](auto const &e) {
        found1 = true;
        val1 = e.element.value;
    }));
    EXPECT_TRUE(t2->find(id_t {2}, [&](auto const &e) {
        found2 = true;
        val2 = e.element.value;
    }));

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
    for (std::size_t i = 0; i < 10; ++i) { EXPECT_TRUE(txn->upsert(pair_t {i, i * 10})); }

    // Before stage: should see 0
    int count_before_stage = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_TRUE(set.find(id_t {i}, [&](auto const &) { count_before_stage++; }));
    }
    EXPECT_EQ(count_before_stage, 0) << "Before stage: should see 0 keys";

    EXPECT_TRUE(txn->stage());

    // After stage, before commit: should see 0
    int count_after_stage = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_TRUE(set.find(id_t {i}, [&](auto const &) { count_after_stage++; }));
    }
    EXPECT_EQ(count_after_stage, 0) << "After stage, before commit: should see 0 keys";

    EXPECT_TRUE(txn->commit());

    // After commit: should see ALL 10
    int count_after_commit = 0;
    for (std::size_t i = 0; i < 10; ++i) {
        EXPECT_TRUE(set.find(id_t {i}, [&](auto const &) { count_after_commit++; }));
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
    EXPECT_TRUE(set.upsert(pair_t {1, 1}));

    auto txn = set.transaction();
    EXPECT_TRUE(txn->upsert(pair_t {1, 100})); // Modify existing
    EXPECT_TRUE(txn->upsert(pair_t {2, 200})); // Add new
    EXPECT_TRUE(txn->upsert(pair_t {3, 300})); // Add new
    EXPECT_TRUE(txn->stage());
    EXPECT_TRUE(txn->rollback()); // ROLLBACK instead of commit

    // Key 1 should have original value
    bool found1 = false;
    EXPECT_TRUE(set.find(id_t {1}, [&](auto const &e) {
        found1 = true;
        EXPECT_EQ(e.element.value, 1) << "Rollback should restore original value";
    }));
    EXPECT_TRUE(found1);

    // Keys 2 and 3 should not exist
    bool found2 = false, found3 = false;
    EXPECT_TRUE(set.find(id_t {2}, [&](auto const &) { found2 = true; }));
    EXPECT_TRUE(set.find(id_t {3}, [&](auto const &) { found3 = true; }));

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
    for (std::size_t i = 10; i < 20; ++i) { EXPECT_TRUE(txn->upsert(pair_t {i, i})); }

    // Before commit: range query sees 0
    int count_before = 0;
    EXPECT_TRUE(set.range(id_t {10}, id_t {20}, [&](auto const &) noexcept { count_before++; }));
    EXPECT_EQ(count_before, 0) << "Range query before commit sees nothing";

    EXPECT_TRUE(txn->stage());
    EXPECT_TRUE(txn->commit());

    // After commit: range query sees ALL 10
    int count_after = 0;
    EXPECT_TRUE(set.range(id_t {10}, id_t {20}, [&](auto const &) noexcept { count_after++; }));
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
    EXPECT_TRUE(t1->upsert(pair_t {1, 10}));
    EXPECT_TRUE(t1->upsert(pair_t {2, 20}));
    EXPECT_TRUE(t1->upsert(pair_t {3, 30}));

    // T2 will insert keys 4-6
    auto t2 = set.transaction();
    EXPECT_TRUE(t2->upsert(pair_t {4, 40}));
    EXPECT_TRUE(t2->upsert(pair_t {5, 50}));
    EXPECT_TRUE(t2->upsert(pair_t {6, 60}));

    // Stage both
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t2->stage());

    // Before any commits: see 0
    int count_0 = 0;
    for (std::size_t i = 1; i <= 6; ++i) {
        EXPECT_TRUE(set.find(id_t {i}, [&](auto const &) { count_0++; }));
    }
    EXPECT_EQ(count_0, 0);

    // Commit T1
    EXPECT_TRUE(t1->commit());

    // Should see exactly T1's keys (1-3), not T2's (4-6)
    int count_t1 = 0;
    for (std::size_t i = 1; i <= 6; ++i) {
        EXPECT_TRUE(set.find(id_t {i}, [&](auto const &) { count_t1++; }));
    }
    EXPECT_EQ(count_t1, 3) << "Should see only T1's 3 keys, not T2's";

    // Commit T2
    EXPECT_TRUE(t2->commit());

    // Now should see ALL 6
    int count_both = 0;
    for (std::size_t i = 1; i <= 6; ++i) {
        EXPECT_TRUE(set.find(id_t {i}, [&](auto const &) { count_both++; }));
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
    EXPECT_TRUE(set.upsert(pair_t {1, 10}));

    std::vector<std::size_t> observed_values;

    // Observe initial value
    EXPECT_TRUE(set.find(id_t {1}, [&](auto const &e) { observed_values.push_back(e.element.value); }));

    // Update to 20
    EXPECT_TRUE(set.upsert(pair_t {1, 20}));
    EXPECT_TRUE(set.find(id_t {1}, [&](auto const &e) { observed_values.push_back(e.element.value); }));

    // Update to 30
    EXPECT_TRUE(set.upsert(pair_t {1, 30}));
    EXPECT_TRUE(set.find(id_t {1}, [&](auto const &e) { observed_values.push_back(e.element.value); }));

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
    EXPECT_TRUE(t1->upsert(pair_t {1, 100}));
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t1->commit());

    // Observe T1's value
    std::size_t val1 = 0;
    EXPECT_TRUE(set.find(id_t {1}, [&](auto const &e) { val1 = e.element.value; }));
    EXPECT_EQ(val1, 100);

    // T2: value = 200 (higher)
    auto t2 = set.transaction();
    EXPECT_TRUE(t2->upsert(pair_t {1, 200}));
    EXPECT_TRUE(t2->stage());
    EXPECT_TRUE(t2->commit());

    // Observe T2's value - should be >= T1's value
    std::size_t val2 = 0;
    EXPECT_TRUE(set.find(id_t {1}, [&](auto const &e) { val2 = e.element.value; }));
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
    EXPECT_TRUE(set.upsert(pair_t {1, 1}));

    auto t1 = set.transaction();
    auto t2 = set.transaction();

    // Both watch the same key
    EXPECT_TRUE(t1->watch(id_t {1}));
    EXPECT_TRUE(t2->watch(id_t {1}));

    // Both modify it
    EXPECT_TRUE(t1->upsert(pair_t {1, 100}));
    EXPECT_TRUE(t2->upsert(pair_t {1, 200}));

    // T1 commits successfully
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t1->commit());

    // T2's stage should FAIL (watched value changed)
    auto status = t2->stage();
    EXPECT_FALSE(status) << "T2 should fail - watched key was modified by T1";
    EXPECT_EQ(status.errc, errc_t::consistency_k);

    // Verify T1's value persisted, T2's did not
    std::size_t final_value = 0;
    EXPECT_TRUE(set.find(id_t {1}, [&](auto const &e) { final_value = e.element.value; }));
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
    EXPECT_TRUE(set.upsert(pair_t {1, 1}));
    EXPECT_TRUE(set.upsert(pair_t {2, 2}));
    EXPECT_TRUE(set.upsert(pair_t {3, 3}));

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
    EXPECT_TRUE(set.upsert(pair_t {2, 999}));

    // T1 modifies all three
    EXPECT_TRUE(t1->upsert(pair_t {1, 10}));
    EXPECT_TRUE(t1->upsert(pair_t {2, 20}));
    EXPECT_TRUE(t1->upsert(pair_t {3, 30}));

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
    EXPECT_TRUE(set.upsert(pair_t {1, 1}));

    auto txn = set.transaction();
    EXPECT_TRUE(txn->watch(id_t {1}));

    // Direct modification to the set (not through a transaction)
    EXPECT_TRUE(set.upsert(pair_t {1, 777}));

    // Transaction attempts to modify
    EXPECT_TRUE(txn->upsert(pair_t {1, 888}));

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
    EXPECT_TRUE(t1->upsert(pair_t {1, 10}));
    EXPECT_TRUE(t1->upsert(pair_t {2, 20}));
    EXPECT_TRUE(t1->upsert(pair_t {3, 30}));

    // T2 modifies keys 4-6 (disjoint!)
    EXPECT_TRUE(t2->upsert(pair_t {4, 40}));
    EXPECT_TRUE(t2->upsert(pair_t {5, 50}));
    EXPECT_TRUE(t2->upsert(pair_t {6, 60}));

    // Both should succeed
    EXPECT_TRUE(t1->stage());
    EXPECT_TRUE(t2->stage());
    EXPECT_TRUE(t1->commit());
    EXPECT_TRUE(t2->commit());

    // Verify all 6 keys exist
    int count = 0;
    for (std::size_t i = 1; i <= 6; ++i) {
        EXPECT_TRUE(set.find(id_t {i}, [&](auto const &) { count++; }));
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
    EXPECT_TRUE(set.upsert(pair_t {1, 100}));

    auto txn = set.transaction();

    // First read
    std::size_t first_read = 0;
    EXPECT_TRUE(txn->find(id_t {1}, [&](auto const &e) { first_read = e.element.value; }));
    EXPECT_EQ(first_read, 100);

    // External modification
    EXPECT_TRUE(set.upsert(pair_t {1, 999}));

    // Second read in SAME transaction - CAN see new value (this is correct!)
    std::size_t second_read = 0;
    EXPECT_TRUE(txn->find(id_t {1}, [&](auto const &e) { second_read = e.element.value; }));

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
    for (std::size_t i = 0; i < 5; ++i) { EXPECT_TRUE(set.upsert(pair_t {i, i})); }

    auto txn = set.transaction();

    // First range query: see 5 items (reads are on committed state)
    int first_count = 0;
    EXPECT_TRUE(set.range(id_t {0}, id_t {10}, [&](auto const &) noexcept { first_count++; }));
    EXPECT_EQ(first_count, 5);

    // External insert
    EXPECT_TRUE(set.upsert(pair_t {5, 5}));
    EXPECT_TRUE(set.upsert(pair_t {6, 6}));

    // Second range query while txn still active - CAN see new items (phantom reads)
    // In Read Committed, reads always see latest committed state
    int second_count = 0;
    EXPECT_TRUE(set.range(id_t {0}, id_t {10}, [&](auto const &) noexcept { second_count++; }));

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
    EXPECT_TRUE(set.upsert(pair_t {1, 1}));
    EXPECT_TRUE(set.upsert(pair_t {2, 2}));

    // Delete via erase_range
    EXPECT_TRUE(set.erase_range(id_t {1}, id_t {2}));

    // Verify deleted key is invisible
    bool found = false;
    EXPECT_TRUE(set.find(id_t {1}, [&](auto const &) { found = true; }));
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
    EXPECT_TRUE(set.upsert(pair_t {1, 1}));

    auto txn = set.transaction();
    EXPECT_TRUE(txn->watch(id_t {1}));
    EXPECT_TRUE(txn->upsert(pair_t {2, 2}));

    // Reset the transaction
    EXPECT_TRUE(txn->reset());

    // After reset, can stage/commit successfully (watches cleared)
    EXPECT_TRUE(txn->upsert(pair_t {3, 3}));
    EXPECT_TRUE(txn->stage());
    EXPECT_TRUE(txn->commit());

    // Key 3 should exist, key 2 should not
    bool found2 = false, found3 = false;
    EXPECT_TRUE(set.find(id_t {2}, [&](auto const &) { found2 = true; }));
    EXPECT_TRUE(set.find(id_t {3}, [&](auto const &) { found3 = true; }));

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
    EXPECT_TRUE(set.upsert(pair_t {1, 100}));

    auto t1 = set.transaction();
    auto t2 = set.transaction();

    // Both watch the same key at generation 1, value 100
    EXPECT_TRUE(t1->watch(id_t {1}));
    EXPECT_TRUE(t2->watch(id_t {1}));

    // T2 modifies and stages (but doesn't commit)
    EXPECT_TRUE(t2->upsert(pair_t {1, 999}));
    EXPECT_TRUE(t2->stage()); // Now gen=2, visible=false

    // T1 should FAIL to stage because T2 has a staged (invisible) write
    // This tests that find_latest_for_watch() is used, not find()
    EXPECT_TRUE(t1->upsert(pair_t {1, 777}));
    auto status = t1->stage();
    EXPECT_FALSE(status) << "T1 should fail - T2 has staged invisible write on watched key";
    EXPECT_EQ(status.errc, errc_t::consistency_k);

    // Clean up: rollback T2, verify original value persists
    EXPECT_TRUE(t2->rollback());
    std::size_t final_value = 0;
    EXPECT_TRUE(set.find(id_t {1}, [&](auto const &e) { final_value = e.element.value; }));
    EXPECT_EQ(final_value, 100) << "Original value should persist after rollback";
}

/** @brief Read Committed: Staged transactions with multiple keys must be completely invisible */
TEST(consistency_standard_set, no_dirty_reads_multi_key) { test_no_dirty_reads_multi_key<stl_t>(); }

/** @brief Read Committed: Transactions created after staging should not see staged changes */
TEST(consistency_standard_set, new_transaction_sees_nothing_staged) {
    test_new_transaction_sees_nothing_staged<stl_t>();
}

/** @brief Read Committed: Once committed, new transactions immediately see changes */
TEST(consistency_standard_set, committed_immediately_visible) { test_committed_immediately_visible<stl_t>(); }

/** @brief Atomic View: In a 10-key transaction, observer sees 0 or 10 keys, never 1-9 */
TEST(consistency_standard_set, multi_key_atomicity_10_keys) { test_multi_key_atomicity_10_keys<stl_t>(); }

/** @brief Atomic View: Rollback atomically hides all staged changes */
TEST(consistency_standard_set, rollback_makes_all_invisible) { test_rollback_makes_all_invisible<stl_t>(); }

/** @brief Atomic View: Range queries see complete transactions, not partial */
TEST(consistency_standard_set, range_query_sees_atomic_boundaries) { test_range_query_sees_atomic_boundaries<stl_t>(); }

/** @brief Atomic View: Interleaved commits don't create fractured reads */
TEST(consistency_standard_set, fractured_read_prevention) { test_fractured_read_prevention<stl_t>(); }

/** @brief Monotonic View: Sequential reads never see values go backwards */
TEST(consistency_standard_set, sequential_updates_never_regress) { test_sequential_updates_never_regress<stl_t>(); }

/** @brief Monotonic View: T1 commits v1, T2 commits v2 → reads see v1→v2, never v2→v1 */
TEST(consistency_standard_set, transaction_commits_maintain_order) { test_transaction_commits_maintain_order<stl_t>(); }

/** @brief Write Conflicts: T1 and T2 watch + modify same key → first wins, second fails */
TEST(consistency_standard_set, concurrent_transactions_on_same_key) {
    test_concurrent_transactions_on_same_key<stl_t>();
}

/** @brief Write Conflicts: If ANY watched key conflicts, entire transaction fails */
TEST(consistency_standard_set, multi_key_conflict_any_key_fails) { test_multi_key_conflict_any_key_fails<stl_t>(); }

/** @brief Write Conflicts: Watches detect modifications outside of transactions */
TEST(consistency_standard_set, watch_detects_external_direct_modification) {
    test_watch_detects_external_direct_modification<stl_t>();
}

/** @brief Write Conflicts: Watch validation must detect staged (invisible) writes */
TEST(consistency_standard_set, watch_detects_staged_invisible_writes) {
    test_watch_detects_staged_invisible_writes<stl_t>();
}

/** @brief Write Conflicts: Transactions on disjoint keys can both succeed */
TEST(consistency_standard_set, disjoint_keys_both_succeed) { test_disjoint_keys_both_succeed<stl_t>(); }

/** @brief Allowed Anomalies: Within a transaction, reading same key twice CAN see different values (CORRECT for Read
 * Committed) */
TEST(consistency_standard_set, non_repeatable_reads_are_allowed) { test_non_repeatable_reads_are_allowed<stl_t>(); }

/** @brief Allowed Anomalies: Range query can return different counts on repeated execution (CORRECT for Read Committed)
 */
TEST(consistency_standard_set, phantom_reads_are_allowed) { test_phantom_reads_are_allowed<stl_t>(); }

/** @brief Edge Cases: Committing a transaction with no changes should succeed */
TEST(consistency_standard_set, empty_transaction_commit) { test_empty_transaction_commit<stl_t>(); }

/** @brief Edge Cases: Deleted entries should not appear in queries */
TEST(consistency_standard_set, delete_visibility) { test_delete_visibility<stl_t>(); }

/** @brief Edge Cases: reset() should clear all transaction state */
TEST(consistency_standard_set, reset_clears_transaction_state) { test_reset_clears_transaction_state<stl_t>(); }

/** @brief Read Committed: Staged transactions with multiple keys must be completely invisible */
TEST(consistency_avl_tree, no_dirty_reads_multi_key) { test_no_dirty_reads_multi_key<avl_t>(); }

/** @brief Read Committed: Transactions created after staging should not see staged changes */
TEST(consistency_avl_tree, new_transaction_sees_nothing_staged) { test_new_transaction_sees_nothing_staged<avl_t>(); }

/** @brief Read Committed: Once committed, new transactions immediately see changes */
TEST(consistency_avl_tree, committed_immediately_visible) { test_committed_immediately_visible<avl_t>(); }

/** @brief Atomic View: In a 10-key transaction, observer sees 0 or 10 keys, never 1-9 */
TEST(consistency_avl_tree, multi_key_atomicity_10_keys) { test_multi_key_atomicity_10_keys<avl_t>(); }

/** @brief Atomic View: Rollback atomically hides all staged changes */
TEST(consistency_avl_tree, rollback_makes_all_invisible) { test_rollback_makes_all_invisible<avl_t>(); }

/** @brief Atomic View: Range queries see complete transactions, not partial */
TEST(consistency_avl_tree, range_query_sees_atomic_boundaries) { test_range_query_sees_atomic_boundaries<avl_t>(); }

/** @brief Atomic View: Interleaved commits don't create fractured reads */
TEST(consistency_avl_tree, fractured_read_prevention) { test_fractured_read_prevention<avl_t>(); }

/** @brief Monotonic View: Sequential reads never see values go backwards */
TEST(consistency_avl_tree, sequential_updates_never_regress) { test_sequential_updates_never_regress<avl_t>(); }

/** @brief Monotonic View: T1 commits v1, T2 commits v2 → reads see v1→v2, never v2→v1 */
TEST(consistency_avl_tree, transaction_commits_maintain_order) { test_transaction_commits_maintain_order<avl_t>(); }

/** @brief Write Conflicts: T1 and T2 watch + modify same key → first wins, second fails */
TEST(consistency_avl_tree, concurrent_transactions_on_same_key) { test_concurrent_transactions_on_same_key<avl_t>(); }

/** @brief Write Conflicts: If ANY watched key conflicts, entire transaction fails */
TEST(consistency_avl_tree, multi_key_conflict_any_key_fails) { test_multi_key_conflict_any_key_fails<avl_t>(); }

/** @brief Write Conflicts: Watches detect modifications outside of transactions */
TEST(consistency_avl_tree, watch_detects_external_direct_modification) {
    test_watch_detects_external_direct_modification<avl_t>();
}

/** @brief Write Conflicts: Watch validation must detect staged (invisible) writes */
TEST(consistency_avl_tree, watch_detects_staged_invisible_writes) {
    test_watch_detects_staged_invisible_writes<avl_t>();
}

/** @brief Write Conflicts: Transactions on disjoint keys can both succeed */
TEST(consistency_avl_tree, disjoint_keys_both_succeed) { test_disjoint_keys_both_succeed<avl_t>(); }

/** @brief Allowed Anomalies: Within a transaction, reading same key twice CAN see different values (CORRECT for Read
 * Committed) */
TEST(consistency_avl_tree, non_repeatable_reads_are_allowed) { test_non_repeatable_reads_are_allowed<avl_t>(); }

/** @brief Allowed Anomalies: Range query can return different counts on repeated execution (CORRECT for Read Committed)
 */
TEST(consistency_avl_tree, phantom_reads_are_allowed) { test_phantom_reads_are_allowed<avl_t>(); }

/** @brief Edge Cases: Committing a transaction with no changes should succeed */
TEST(consistency_avl_tree, empty_transaction_commit) { test_empty_transaction_commit<avl_t>(); }

/** @brief Edge Cases: Deleted entries should not appear in queries */
TEST(consistency_avl_tree, delete_visibility) { test_delete_visibility<avl_t>(); }

/** @brief Edge Cases: reset() should clear all transaction state */
TEST(consistency_avl_tree, reset_clears_transaction_state) { test_reset_clears_transaction_state<avl_t>(); }

/** @brief Basic Operations: Ascending, descending, random insertion with AVL rebalancing */
TEST(basic_ops_partitioned_stl, insertion_patterns) { test_basic_insertion_patterns<partitioned_stl_t>(); }

/** @brief Basic Operations: Bulk insertion via move iterators */
TEST(basic_ops_partitioned_stl, bulk_insertion_iterators) { test_bulk_insertion_from_iterators<partitioned_stl_t>(); }

/** @brief Basic Operations: Range queries on committed HEAD state */
TEST(basic_ops_partitioned_stl, range_query_head_state) { test_range_query_head_state<partitioned_stl_t>(); }

/** @brief Basic Operations: erase_range on committed HEAD state */
TEST(basic_ops_partitioned_stl, erase_range_head_state) { test_erase_range_head_state<partitioned_stl_t>(); }

/** @brief Basic Operations: upper_bound returns first element > key */
TEST(basic_ops_partitioned_stl, upper_bound) { test_upper_bound<partitioned_stl_t>(); }

/** @brief Basic Operations: clear resets size to 0 */
TEST(basic_ops_partitioned_stl, clear) { test_clear<partitioned_stl_t>(); }

/** @brief Basic Operations: size increments correctly */
TEST(basic_ops_partitioned_stl, size_after_upserts) { test_size_after_upserts<partitioned_stl_t>(); }

/** @brief Basic Operations: size unchanged on duplicate upserts */
TEST(basic_ops_partitioned_stl, size_invariant_on_duplicates) {
    test_size_invariant_on_duplicate_upserts<partitioned_stl_t>();
}

/** @brief Basic Operations: size decrements after erase */
TEST(basic_ops_partitioned_stl, size_after_erase) { test_size_after_erase<partitioned_stl_t>(); }

/** @brief Basic Operations: empty container operations don't crash */
TEST(basic_ops_partitioned_stl, empty_container_operations) { test_empty_container_operations<partitioned_stl_t>(); }

/** @brief Basic Operations: single element operations */
TEST(basic_ops_partitioned_stl, single_element_operations) { test_single_element_operations<partitioned_stl_t>(); }

/** @brief Basic Operations: reserve() pre-allocates capacity (STL only) */
TEST(basic_ops_partitioned_stl, reserve) { test_reserve<partitioned_stl_t>(); }

/** @brief Basic Operations: Ascending, descending, random insertion with AVL rebalancing */
TEST(basic_ops_partitioned_avl, insertion_patterns) { test_basic_insertion_patterns<partitioned_avl_t>(); }

/** @brief Basic Operations: Bulk insertion via move iterators */
TEST(basic_ops_partitioned_avl, bulk_insertion_iterators) { test_bulk_insertion_from_iterators<partitioned_avl_t>(); }

/** @brief Basic Operations: Range queries on committed HEAD state */
TEST(basic_ops_partitioned_avl, range_query_head_state) { test_range_query_head_state<partitioned_avl_t>(); }

/** @brief Basic Operations: erase_range on committed HEAD state */
TEST(basic_ops_partitioned_avl, erase_range_head_state) { test_erase_range_head_state<partitioned_avl_t>(); }

/** @brief Basic Operations: upper_bound returns first element > key */
TEST(basic_ops_partitioned_avl, upper_bound) { test_upper_bound<partitioned_avl_t>(); }

/** @brief Basic Operations: clear resets size to 0 */
TEST(basic_ops_partitioned_avl, clear) { test_clear<partitioned_avl_t>(); }

/** @brief Basic Operations: size increments correctly */
TEST(basic_ops_partitioned_avl, size_after_upserts) { test_size_after_upserts<partitioned_avl_t>(); }

/** @brief Basic Operations: size unchanged on duplicate upserts */
TEST(basic_ops_partitioned_avl, size_invariant_on_duplicates) {
    test_size_invariant_on_duplicate_upserts<partitioned_avl_t>();
}

/** @brief Basic Operations: size decrements after erase */
TEST(basic_ops_partitioned_avl, size_after_erase) { test_size_after_erase<partitioned_avl_t>(); }

/** @brief Basic Operations: empty container operations don't crash */
TEST(basic_ops_partitioned_avl, empty_container_operations) { test_empty_container_operations<partitioned_avl_t>(); }

/** @brief Basic Operations: single element operations */
TEST(basic_ops_partitioned_avl, single_element_operations) { test_single_element_operations<partitioned_avl_t>(); }

/** @brief Read Committed: Staged transactions with multiple keys must be completely invisible */
TEST(consistency_partitioned_stl, no_dirty_reads_multi_key) { test_no_dirty_reads_multi_key<partitioned_stl_t>(); }

/** @brief Read Committed: Transactions created after staging should not see staged changes */
TEST(consistency_partitioned_stl, new_transaction_sees_nothing_staged) {
    test_new_transaction_sees_nothing_staged<partitioned_stl_t>();
}

/** @brief Read Committed: Once committed, new transactions immediately see changes */
TEST(consistency_partitioned_stl, committed_immediately_visible) {
    test_committed_immediately_visible<partitioned_stl_t>();
}

/** @brief Atomic View: In a 10-key transaction, observer sees 0 or 10 keys, never 1-9 */
TEST(consistency_partitioned_stl, multi_key_atomicity_10_keys) {
    test_multi_key_atomicity_10_keys<partitioned_stl_t>();
}

/** @brief Atomic View: Rollback atomically hides all staged changes */
TEST(consistency_partitioned_stl, rollback_makes_all_invisible) {
    test_rollback_makes_all_invisible<partitioned_stl_t>();
}

/** @brief Atomic View: Range queries see complete transactions, not partial */
TEST(consistency_partitioned_stl, range_query_sees_atomic_boundaries) {
    test_range_query_sees_atomic_boundaries<partitioned_stl_t>();
}

/** @brief Atomic View: Interleaved commits don't create fractured reads */
TEST(consistency_partitioned_stl, fractured_read_prevention) { test_fractured_read_prevention<partitioned_stl_t>(); }

/** @brief Monotonic View: Sequential reads never see values go backwards */
TEST(consistency_partitioned_stl, sequential_updates_never_regress) {
    test_sequential_updates_never_regress<partitioned_stl_t>();
}

/** @brief Monotonic View: T1 commits v1, T2 commits v2 → reads see v1→v2, never v2→v1 */
TEST(consistency_partitioned_stl, transaction_commits_maintain_order) {
    test_transaction_commits_maintain_order<partitioned_stl_t>();
}

/** @brief Write Conflicts: T1 and T2 watch + modify same key → first wins, second fails */
TEST(consistency_partitioned_stl, concurrent_transactions_on_same_key) {
    test_concurrent_transactions_on_same_key<partitioned_stl_t>();
}

/** @brief Write Conflicts: If ANY watched key conflicts, entire transaction fails */
TEST(consistency_partitioned_stl, multi_key_conflict_any_key_fails) {
    test_multi_key_conflict_any_key_fails<partitioned_stl_t>();
}

/** @brief Write Conflicts: Watches detect modifications outside of transactions */
TEST(consistency_partitioned_stl, watch_detects_external_direct_modification) {
    test_watch_detects_external_direct_modification<partitioned_stl_t>();
}

/** @brief Write Conflicts: Watch validation must detect staged (invisible) writes */
TEST(consistency_partitioned_stl, watch_detects_staged_invisible_writes) {
    test_watch_detects_staged_invisible_writes<partitioned_stl_t>();
}

/** @brief Write Conflicts: Transactions on disjoint keys can both succeed */
TEST(consistency_partitioned_stl, disjoint_keys_both_succeed) { test_disjoint_keys_both_succeed<partitioned_stl_t>(); }

/** @brief Allowed Anomalies: Within a transaction, reading same key twice CAN see different values (CORRECT for Read
 * Committed) */
TEST(consistency_partitioned_stl, non_repeatable_reads_are_allowed) {
    test_non_repeatable_reads_are_allowed<partitioned_stl_t>();
}

/** @brief Allowed Anomalies: Range query can return different counts on repeated execution (CORRECT for Read Committed)
 */
TEST(consistency_partitioned_stl, phantom_reads_are_allowed) { test_phantom_reads_are_allowed<partitioned_stl_t>(); }

/** @brief Edge Cases: Committing a transaction with no changes should succeed */
TEST(consistency_partitioned_stl, empty_transaction_commit) { test_empty_transaction_commit<partitioned_stl_t>(); }

/** @brief Edge Cases: Deleted entries should not appear in queries */
TEST(consistency_partitioned_stl, delete_visibility) { test_delete_visibility<partitioned_stl_t>(); }

/** @brief Edge Cases: reset() should clear all transaction state */
TEST(consistency_partitioned_stl, reset_clears_transaction_state) {
    test_reset_clears_transaction_state<partitioned_stl_t>();
}

/** @brief Read Committed: Staged transactions with multiple keys must be completely invisible */
TEST(consistency_partitioned_avl, no_dirty_reads_multi_key) { test_no_dirty_reads_multi_key<partitioned_avl_t>(); }

/** @brief Read Committed: Transactions created after staging should not see staged changes */
TEST(consistency_partitioned_avl, new_transaction_sees_nothing_staged) {
    test_new_transaction_sees_nothing_staged<partitioned_avl_t>();
}

/** @brief Read Committed: Once committed, new transactions immediately see changes */
TEST(consistency_partitioned_avl, committed_immediately_visible) {
    test_committed_immediately_visible<partitioned_avl_t>();
}

/** @brief Atomic View: In a 10-key transaction, observer sees 0 or 10 keys, never 1-9 */
TEST(consistency_partitioned_avl, multi_key_atomicity_10_keys) {
    test_multi_key_atomicity_10_keys<partitioned_avl_t>();
}

/** @brief Atomic View: Rollback atomically hides all staged changes */
TEST(consistency_partitioned_avl, rollback_makes_all_invisible) {
    test_rollback_makes_all_invisible<partitioned_avl_t>();
}

/** @brief Atomic View: Range queries see complete transactions, not partial */
TEST(consistency_partitioned_avl, range_query_sees_atomic_boundaries) {
    test_range_query_sees_atomic_boundaries<partitioned_avl_t>();
}

/** @brief Atomic View: Interleaved commits don't create fractured reads */
TEST(consistency_partitioned_avl, fractured_read_prevention) { test_fractured_read_prevention<partitioned_avl_t>(); }

/** @brief Monotonic View: Sequential reads never see values go backwards */
TEST(consistency_partitioned_avl, sequential_updates_never_regress) {
    test_sequential_updates_never_regress<partitioned_avl_t>();
}

/** @brief Monotonic View: T1 commits v1, T2 commits v2 → reads see v1→v2, never v2→v1 */
TEST(consistency_partitioned_avl, transaction_commits_maintain_order) {
    test_transaction_commits_maintain_order<partitioned_avl_t>();
}

/** @brief Write Conflicts: T1 and T2 watch + modify same key → first wins, second fails */
TEST(consistency_partitioned_avl, concurrent_transactions_on_same_key) {
    test_concurrent_transactions_on_same_key<partitioned_avl_t>();
}

/** @brief Write Conflicts: If ANY watched key conflicts, entire transaction fails */
TEST(consistency_partitioned_avl, multi_key_conflict_any_key_fails) {
    test_multi_key_conflict_any_key_fails<partitioned_avl_t>();
}

/** @brief Write Conflicts: Watches detect modifications outside of transactions */
TEST(consistency_partitioned_avl, watch_detects_external_direct_modification) {
    test_watch_detects_external_direct_modification<partitioned_avl_t>();
}

/** @brief Write Conflicts: Watch validation must detect staged (invisible) writes */
TEST(consistency_partitioned_avl, watch_detects_staged_invisible_writes) {
    test_watch_detects_staged_invisible_writes<partitioned_avl_t>();
}

/** @brief Write Conflicts: Transactions on disjoint keys can both succeed */
TEST(consistency_partitioned_avl, disjoint_keys_both_succeed) { test_disjoint_keys_both_succeed<partitioned_avl_t>(); }

/** @brief Allowed Anomalies: Within a transaction, reading same key twice CAN see different values (CORRECT for Read
 * Committed) */
TEST(consistency_partitioned_avl, non_repeatable_reads_are_allowed) {
    test_non_repeatable_reads_are_allowed<partitioned_avl_t>();
}

/** @brief Allowed Anomalies: Range query can return different counts on repeated execution (CORRECT for Read Committed)
 */
TEST(consistency_partitioned_avl, phantom_reads_are_allowed) { test_phantom_reads_are_allowed<partitioned_avl_t>(); }

/** @brief Edge Cases: Committing a transaction with no changes should succeed */
TEST(consistency_partitioned_avl, empty_transaction_commit) { test_empty_transaction_commit<partitioned_avl_t>(); }

/** @brief Edge Cases: Deleted entries should not appear in queries */
TEST(consistency_partitioned_avl, delete_visibility) { test_delete_visibility<partitioned_avl_t>(); }

/** @brief Edge Cases: reset() should clear all transaction state */
TEST(consistency_partitioned_avl, reset_clears_transaction_state) {
    test_reset_clears_transaction_state<partitioned_avl_t>();
}

/** @brief Heterogeneous Lookups: WITH is_transparent in comparator (standard_set) */
TEST(heterogeneous_lookups_standard_set, with_is_transparent) {
    test_heterogeneous_lookups_with_transparent<transactional_std_set>();
}

/** @brief Heterogeneous Lookups: WITHOUT is_transparent in comparator (standard_set) */
TEST(heterogeneous_lookups_standard_set, without_is_transparent) {
    test_heterogeneous_lookups_without_transparent<transactional_std_set>();
}

/** @brief Heterogeneous Lookups: WITH is_transparent in comparator (avl_tree) */
TEST(heterogeneous_lookups_avl_tree, with_is_transparent) {
    test_heterogeneous_lookups_with_transparent<transactional_avl_tree>();
}

/** @brief Heterogeneous Lookups: WITHOUT is_transparent in comparator (avl_tree) */
TEST(heterogeneous_lookups_avl_tree, without_is_transparent) {
    test_heterogeneous_lookups_without_transparent<transactional_avl_tree>();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
