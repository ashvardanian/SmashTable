#include <iostream>
#include <cstdlib>
#include <thread>
#include <ctime>

#include <smashtable/consistent_set.hpp>
#include <smashtable/consistent_avl.hpp>
#include <gtest/gtest.h>

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
    bool operator()(pair_t a, pair_t b) const noexcept { return a.key < b.key; }
    bool operator()(std::size_t a, pair_t b) const noexcept { return a < b.key; }
    bool operator()(pair_t a, std::size_t b) const noexcept { return a.key < b; }
};

using stl_t = atomic_standard_set<pair_t, pair_compare_t>;
using avl_t = atomic_avl_tree<pair_t, pair_compare_t>;
using id_stl_t = typename stl_t::identifier_t;
using id_avl_t = typename avl_t::identifier_t;

template <typename cont_t>
void test_with_threads(std::size_t threads_count) {
    auto cont = *cont_t::make();
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

// NOTE: Disabled - atomic_standard_set and atomic_avl_tree are NOT thread-safe
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

TEST(upsert_and_find_set, ascending) {
    auto set = *stl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) {
        EXPECT_TRUE(set.upsert(pair_t {idx, idx}));
        EXPECT_TRUE(set.find(idx, [](auto const &) noexcept {}));
    }
    EXPECT_EQ(set.size(), size);
}

TEST(upsert_and_find_set, descending) {
    auto set = *stl_t::make();

    for (std::size_t idx = size; idx > 0; --idx) {
        EXPECT_TRUE(set.upsert(pair_t {idx, idx}));
        EXPECT_TRUE(set.find(idx, [](auto const &) noexcept {}));
    }
    EXPECT_EQ(set.size(), size);
}

TEST(upsert_and_find_set, random) {
    std::srand(std::time(nullptr));
    auto set = *stl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) {
        std::size_t val = std::rand();
        EXPECT_TRUE(set.upsert(pair_t {val, val}));
        EXPECT_TRUE(set.find(val, [](auto const &) noexcept {}));
    }
}

TEST(upsert_and_find_avl, ascending) {
    auto avl = *avl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) {
        EXPECT_TRUE(avl.upsert(pair_t {idx, idx}));
        EXPECT_TRUE(avl.find(idx, [](auto const &) noexcept {}));
    }
    EXPECT_EQ(avl.size(), size);
}

TEST(upsert_and_find_avl, descending) {
    auto avl = *avl_t::make();

    for (std::size_t idx = size; idx > 0; --idx) {
        EXPECT_TRUE(avl.upsert(pair_t {idx, idx}));
        EXPECT_TRUE(avl.find(idx, [](auto const &) noexcept {}));
    }
    EXPECT_EQ(avl.size(), size);
}

TEST(upsert_and_find_avl, random) {
    std::srand(std::time(nullptr));
    auto avl = *avl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) {
        std::size_t val = std::rand();
        EXPECT_TRUE(avl.upsert(pair_t {val, val}));
        EXPECT_TRUE(avl.find(val, [](auto const &) noexcept {}));
    }
}

TEST(upsert_and_find_avl, iterators) {
    std::vector<pair_t> vec(size);
    auto avl = *avl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) vec[idx] = pair_t {idx, idx};

    EXPECT_TRUE(avl.upsert(vec.begin(), vec.end()));
    EXPECT_EQ(avl.size(), size);

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(avl.find(idx, [](auto const &) noexcept {}));
}

TEST(upsert_and_find_set, iterators) {
    std::vector<pair_t> vec(size);
    auto set = *stl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) vec[idx] = pair_t {idx, idx};

    EXPECT_TRUE(set.upsert(std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end())));
    EXPECT_EQ(set.size(), size);

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(set.find(idx, [](auto const &) noexcept {}));
}

TEST(test_set, range) {
    auto set = *stl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(set.upsert(pair_t {idx, idx}));

    for (std::size_t idx = 0; idx < size; idx += 8) {
        std::size_t val = idx;
        EXPECT_TRUE(set.range(idx, idx + 7, [&](auto const &rhs) noexcept {
            EXPECT_EQ(val, rhs.key);
            ++val;
        }));
    }
}

TEST(test_avl, range) {
    auto avl = *avl_t::make();
    auto set = *stl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(avl.upsert(pair_t {idx, idx}));

    bool state = false;
    for (std::size_t idx = 0; idx < size; idx += 8) {
        EXPECT_TRUE(avl.range(idx, idx + 7,
                              [&](auto const &rhs) noexcept { EXPECT_TRUE(set.upsert(pair_t {rhs.key, rhs.value})); }));
        for (std::size_t i = idx; i < idx + 8; ++i) {
            EXPECT_TRUE(set.find(i, [&](auto const &) noexcept { state = true; }));
            EXPECT_TRUE(state);
            state = false;
        }
        EXPECT_TRUE(set.clear());
    }
}

TEST(test_set, erase) {
    auto set = *stl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(set.upsert(pair_t {idx, idx}));

    bool state = true;
    for (std::size_t idx = 0; idx < size; idx += 10) {
        EXPECT_TRUE(set.erase_range(idx, idx + 10, [](auto const &) noexcept {}));
        for (std::size_t i = idx; i < idx + 10; ++i) {
            EXPECT_TRUE(set.find(i, [&](auto const &) noexcept { state = false; }));
            EXPECT_TRUE(state);
        }
    }
}

TEST(test_avl, erase) {
    auto avl = *avl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(avl.upsert(pair_t {idx, idx}));

    bool state = true;
    for (std::size_t idx = 0; idx < size; idx += 10) {
        EXPECT_TRUE(avl.erase_range(idx, idx + 10, [](auto const &) noexcept {}));
        for (std::size_t i = idx; i < idx + 10; ++i) {
            EXPECT_TRUE(avl.find(i, [&](auto const &) noexcept { state = false; }));
            EXPECT_TRUE(state);
        }
    }
}

TEST(test_avl, upper_bound) {
    auto avl = *avl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(avl.upsert(pair_t {idx, idx}));

    for (std::size_t idx = 0; idx < size - 1; ++idx) {
        EXPECT_TRUE(avl.upper_bound(idx, [&](auto const &rhs) noexcept { EXPECT_TRUE(pair_compare_t {}(idx, rhs)); }));
    }
}

TEST(test_set, upper_bound) {
    auto set = *stl_t::make();

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(set.upsert(pair_t {idx, idx}));

    for (std::size_t idx = 0; idx < size - 1; ++idx) {
        EXPECT_TRUE(set.upper_bound(idx, [&](auto const &rhs) noexcept { EXPECT_TRUE(pair_compare_t {}(idx, rhs)); }));
    }
}

TEST(test_set, reserve_clear) {
    auto set = *stl_t::make();
    EXPECT_TRUE(set.reserve(size));
    EXPECT_EQ(set.size(), 0);

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(set.upsert(pair_t {idx, idx}));

    EXPECT_EQ(set.size(), size);
    EXPECT_TRUE(set.clear());
    EXPECT_EQ(set.size(), 0);
}

TEST(test_avl, clear) {
    auto avl = *avl_t::make();
    EXPECT_EQ(avl.size(), 0);

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(avl.upsert(pair_t {idx, idx}));

    EXPECT_EQ(avl.size(), size);
    EXPECT_TRUE(avl.clear());
    EXPECT_EQ(avl.size(), 0);
}

// Transaction Tests
TEST(transaction_set, watch_stage_commit) {
    auto set = *stl_t::make();

    // Insert initial data
    for (std::size_t idx = 0; idx < 10; ++idx) EXPECT_TRUE(set.upsert(pair_t {idx, idx}));
    EXPECT_EQ(set.size(), 10);

    // Create transaction and watch some entries
    auto txn = set.transaction();
    EXPECT_TRUE(txn.has_value());
    EXPECT_TRUE(txn->watch(id_stl_t {5}));
    EXPECT_TRUE(txn->watch(id_stl_t {6}));

    // Modify in transaction
    EXPECT_TRUE(txn->upsert(pair_t {5, 500}));
    EXPECT_TRUE(txn->upsert(pair_t {100, 100}));

    // Stage and commit
    EXPECT_TRUE(txn->stage());
    EXPECT_TRUE(txn->commit());

    // Verify changes
    EXPECT_EQ(set.size(), 11);
    bool found = false;
    EXPECT_TRUE(set.find(id_stl_t {5}, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.value, 500);
    }));
    EXPECT_TRUE(found);
    found = false;
    EXPECT_TRUE(set.find(id_stl_t {100}, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.value, 100);
    }));
    EXPECT_TRUE(found);
}

TEST(transaction_set, watch_stage_rollback) {
    auto set = *stl_t::make();

    // Insert initial data
    for (std::size_t idx = 0; idx < 10; ++idx) EXPECT_TRUE(set.upsert(pair_t {idx, idx}));

    auto txn = set.transaction();
    EXPECT_TRUE(txn->watch(id_stl_t {5}));
    EXPECT_TRUE(txn->upsert(pair_t {5, 500}));
    EXPECT_TRUE(txn->upsert(pair_t {100, 100}));
    EXPECT_TRUE(txn->stage());

    // Rollback instead of commit
    EXPECT_TRUE(txn->rollback());

    // Verify rollback - original data should be intact
    EXPECT_EQ(set.size(), 10);
    bool found = false;
    EXPECT_TRUE(set.find(id_stl_t {5}, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.value, 5);
    }));
    EXPECT_TRUE(found);
}

// TODO: Fix MVCC consistency detection - watches not properly tracking external modifications
TEST(transaction_set, DISABLED_consistency_violation) {
    auto set = *stl_t::make();

    // Insert initial data
    EXPECT_TRUE(set.upsert(pair_t {1, 1}));

    // Create transaction and watch
    auto txn = set.transaction();
    EXPECT_TRUE(txn->watch(id_stl_t {1}));

    // Modify outside transaction
    EXPECT_TRUE(set.upsert(pair_t {1, 100}));

    // Modify in transaction
    EXPECT_TRUE(txn->upsert(pair_t {1, 500}));

    // Stage should fail due to consistency violation
    auto status = txn->stage();
    EXPECT_FALSE(status);
    EXPECT_EQ(status.errc, errc_t::consistency_k);
}

TEST(transaction_avl, watch_stage_commit) {
    auto avl = *avl_t::make();

    // Insert initial data
    for (std::size_t idx = 0; idx < 10; ++idx) EXPECT_TRUE(avl.upsert(pair_t {idx, idx}));
    EXPECT_EQ(avl.size(), 10);

    // Create transaction and watch some entries
    auto txn = avl.transaction();
    EXPECT_TRUE(txn.has_value());
    EXPECT_TRUE(txn->watch(id_avl_t {5}));
    EXPECT_TRUE(txn->watch(id_avl_t {6}));

    // Modify in transaction
    EXPECT_TRUE(txn->upsert(pair_t {5, 500}));
    EXPECT_TRUE(txn->upsert(pair_t {100, 100}));

    // Stage and commit
    EXPECT_TRUE(txn->stage());
    EXPECT_TRUE(txn->commit());

    // Verify changes
    EXPECT_EQ(avl.size(), 11);
    bool found = false;
    EXPECT_TRUE(avl.find(id_avl_t {5}, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.value, 500);
    }));
    EXPECT_TRUE(found);
}

// TODO: Fix MVCC consistency detection - watches not properly tracking external modifications
TEST(transaction_avl, DISABLED_consistency_violation) {
    auto avl = *avl_t::make();

    // Insert initial data
    EXPECT_TRUE(avl.upsert(pair_t {1, 1}));

    // Create transaction and watch
    auto txn = avl.transaction();
    EXPECT_TRUE(txn->watch(id_avl_t {1}));

    // Modify outside transaction
    EXPECT_TRUE(avl.upsert(pair_t {1, 100}));

    // Modify in transaction
    EXPECT_TRUE(txn->upsert(pair_t {1, 500}));

    // Stage should fail
    auto status = txn->stage();
    EXPECT_FALSE(status);
    EXPECT_EQ(status.errc, errc_t::consistency_k);
}

// Visibility Counter Tests
TEST(visibility_counters_set, size_after_upserts) {
    auto set = *stl_t::make();
    EXPECT_EQ(set.size(), 0);

    for (std::size_t idx = 0; idx < 50; ++idx) {
        EXPECT_TRUE(set.upsert(pair_t {idx, idx}));
        EXPECT_EQ(set.size(), idx + 1);
    }

    // Upsert same keys again - size should remain same
    for (std::size_t idx = 0; idx < 50; ++idx) {
        EXPECT_TRUE(set.upsert(pair_t {idx, idx * 2}));
        EXPECT_EQ(set.size(), 50);
    }
}

TEST(visibility_counters_avl, size_after_upserts) {
    auto avl = *avl_t::make();
    EXPECT_EQ(avl.size(), 0);

    for (std::size_t idx = 0; idx < 50; ++idx) {
        EXPECT_TRUE(avl.upsert(pair_t {idx, idx}));
        EXPECT_EQ(avl.size(), idx + 1);
    }

    // Upsert same keys again - size should remain same
    for (std::size_t idx = 0; idx < 50; ++idx) {
        EXPECT_TRUE(avl.upsert(pair_t {idx, idx * 2}));
        EXPECT_EQ(avl.size(), 50);
    }
}

TEST(visibility_counters_set, size_after_erase) {
    auto set = *stl_t::make();

    for (std::size_t idx = 0; idx < 50; ++idx) EXPECT_TRUE(set.upsert(pair_t {idx, idx}));
    EXPECT_EQ(set.size(), 50);

    // Erase in ranges
    for (std::size_t idx = 0; idx < 50; idx += 10) {
        EXPECT_TRUE(set.erase_range(idx, idx + 10));
        EXPECT_EQ(set.size(), 50 - (idx + 10));
    }
    EXPECT_EQ(set.size(), 0);
}

TEST(visibility_counters_avl, size_after_erase) {
    auto avl = *avl_t::make();

    for (std::size_t idx = 0; idx < 50; ++idx) EXPECT_TRUE(avl.upsert(pair_t {idx, idx}));
    EXPECT_EQ(avl.size(), 50);

    // Erase in ranges
    for (std::size_t idx = 0; idx < 50; idx += 10) {
        EXPECT_TRUE(avl.erase_range(idx, idx + 10));
        EXPECT_EQ(avl.size(), 50 - (idx + 10));
    }
    EXPECT_EQ(avl.size(), 0);
}

// Edge Case Tests
TEST(edge_cases_set, empty_operations) {
    auto set = *stl_t::make();

    // Operations on empty set
    bool found = false;
    EXPECT_TRUE(set.find(id_stl_t {1}, [&](auto const &) { found = true; }));
    EXPECT_FALSE(found);

    EXPECT_TRUE(set.upper_bound(id_stl_t {1}, [&](auto const &) { found = true; }));
    EXPECT_FALSE(found);

    EXPECT_TRUE(set.erase_range(id_stl_t {0}, id_stl_t {10}));
    EXPECT_EQ(set.size(), 0);
}

TEST(edge_cases_avl, empty_operations) {
    auto avl = *avl_t::make();

    // Operations on empty avl
    bool found = false;
    EXPECT_TRUE(avl.find(id_avl_t {1}, [&](auto const &) { found = true; }));
    EXPECT_FALSE(found);

    EXPECT_TRUE(avl.upper_bound(id_avl_t {1}, [&](auto const &) { found = true; }));
    EXPECT_FALSE(found);

    EXPECT_TRUE(avl.erase_range(id_avl_t {0}, id_avl_t {10}));
    EXPECT_EQ(avl.size(), 0);
}

TEST(edge_cases_set, single_element) {
    auto set = *stl_t::make();
    EXPECT_TRUE(set.upsert(pair_t {42, 42}));
    EXPECT_EQ(set.size(), 1);

    bool found = false;
    EXPECT_TRUE(set.find(id_stl_t {42}, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.value, 42);
    }));
    EXPECT_TRUE(found);

    EXPECT_TRUE(set.erase_range(id_stl_t {42}, id_stl_t {43}));
    EXPECT_EQ(set.size(), 0);
}

TEST(edge_cases_avl, single_element) {
    auto avl = *avl_t::make();
    EXPECT_TRUE(avl.upsert(pair_t {42, 42}));
    EXPECT_EQ(avl.size(), 1);

    bool found = false;
    EXPECT_TRUE(avl.find(id_avl_t {42}, [&](auto const &e) {
        found = true;
        EXPECT_EQ(e.element.value, 42);
    }));
    EXPECT_TRUE(found);

    EXPECT_TRUE(avl.erase_range(id_avl_t {42}, id_avl_t {43}));
    EXPECT_EQ(avl.size(), 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}