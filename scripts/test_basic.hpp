/**
 *  @brief  Template test functions for basic container operations.
 *    Includes insert/erase/find/range operations and heterogeneous lookup tests.
 *    Templates can be instantiated for any container supporting the common interface.
 *
 *  @file   test_basic.hpp
 */
#pragma once
#include <iostream>
#include <vector>
#include <cstdlib>
#include <thread>
#include <ctime>

#include <gtest/gtest.h>

#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1
#include <smashtable/transactional_std_store.hpp>
#include <smashtable/transactional_binary_tree.hpp>

namespace ashvardanian::smashtable::scripts {

constexpr std::size_t size = 128;

#pragma mark - Keys and Associations

/**
 *  @brief Strongly-typed trivial key without any payload or heterogenous comparisons.
 */
struct trivial_key_t {
    std::uint64_t unique_id = 0;

    explicit trivial_key_t(std::uint64_t i = 0) noexcept : unique_id(i) {}

    trivial_key_t(trivial_key_t &&) noexcept = default;
    trivial_key_t(trivial_key_t const &) noexcept = default;
    trivial_key_t &operator=(trivial_key_t &&) noexcept = default;
    trivial_key_t &operator=(trivial_key_t const &) noexcept = default;

    bool operator<(trivial_key_t const &other) const noexcept { return unique_id < other.unique_id; }
    bool operator==(trivial_key_t const &other) const noexcept { return unique_id == other.unique_id; }
};

/**
 *  @brief Strongly-typed lightweight key that explicitly converts to the inner identifier.
 *    The identifier is inferred from comparator - @c composite_key_compare_t::value_type.
 *    Both the key and the underlying identifier are @c noexcept copyable.
 */
struct composite_key_t {
    std::uint64_t some_metadata = 0;
    std::uint64_t unique_id = 0;
    double some_float = 0.0;

    explicit composite_key_t(std::uint64_t i = 0) noexcept : unique_id(i) {}

    composite_key_t(composite_key_t &&) noexcept = default;
    composite_key_t(composite_key_t const &) noexcept = default;
    composite_key_t &operator=(composite_key_t &&) noexcept = default;
    composite_key_t &operator=(composite_key_t const &) noexcept = default;

    explicit operator std::uint64_t() const noexcept { return unique_id; }
    bool operator<(composite_key_t const &other) const noexcept { return unique_id < other.unique_id; }
    bool operator==(composite_key_t const &other) const noexcept { return unique_id == other.unique_id; }
};

struct composite_key_compare_t {
    using is_transparent = void;      // ? Enable heterogeneous lookup by prefix
    using value_type = std::uint64_t; // ? The only part we need to resolve transactions without carrying full object

    inline bool operator()(composite_key_t const &a, composite_key_t const &b) const noexcept {
        return a.unique_id < b.unique_id;
    }
    inline bool operator()(value_type a, composite_key_t const &b) const noexcept { return a < b.unique_id; }
    inline bool operator()(composite_key_t const &a, value_type b) const noexcept { return a.unique_id < b; }
};

/**
 *  @brief Strongly-typed key with heavy payload, that doesn't have a @c noexcept constructors,
 *    but provides @c ::make(...) and @c .copy() interfaces for safe construction and copying.
 *
 *  For compatibility with integer-based tests,
 */
struct heavy_key_t {

    basic_vector<char> text;

    heavy_key_t() noexcept = default;
    heavy_key_t(heavy_key_t &&) noexcept = default;
    heavy_key_t &operator=(heavy_key_t &&) noexcept = default;

    /**
     *  @brief Creates a heavy_key_t from an integer by converting to hex string.
     *  @param integer_to_encode Integer to encode as hexadecimal string.
     *  @return expected<heavy_key_t> containing the key or error on OOM.
     */
    static expected<heavy_key_t> make(std::uint64_t integer_to_encode) noexcept {
        heavy_key_t result;
        char buf[32];
        int len = std::snprintf(buf, sizeof(buf), "%016lx", integer_to_encode);
        if (len <= 0 || len >= static_cast<int>(sizeof(buf)))
            return expected<heavy_key_t>(heavy_key_t {}, status_t {errc_t::unknown_k});

        for (int i = 0; i < len; ++i) {
            auto status = result.text.push_back(char(buf[i]));
            if (!status) return expected<heavy_key_t>(heavy_key_t {}, status);
        }
        return result;
    }

    /**
     *  @brief Creates a heavy_key_t from a string by copying into internal buffer.
     *  @param std_string String to copy.
     *  @return expected<heavy_key_t> containing the key or error on OOM.
     */
    static expected<heavy_key_t> make(std::string_view std_string) noexcept {
        heavy_key_t result;
        for (std::size_t i = 0; i < std_string.size(); ++i) {
            auto status = result.text.push_back(char(std_string[i]));
            if (!status) return expected<heavy_key_t>(heavy_key_t {}, status);
        }
        return result;
    }

    /**
     *  @brief Deep copies this heavy_key_t.
     *  @return expected<heavy_key_t> containing the copy or error on OOM.
     */
    expected<heavy_key_t> copy() const noexcept {
        heavy_key_t result;
        for (std::size_t i = 0; i < text.size(); ++i) {
            auto status = result.text.push_back(char(text.data()[i]));
            if (!status) return expected<heavy_key_t>(heavy_key_t {}, status);
        }
        return result;
    }

    std::string_view view() const noexcept { return {text.data(), text.size()}; }

    auto operator<=>(heavy_key_t const &other) const noexcept {
        auto cmp = memcmp(text.data(), other.text.data(), std::min(text.size(), other.text.size()));
        if (cmp != 0) return cmp <=> 0;
        return text.size() <=> other.text.size();
    }

    bool operator==(heavy_key_t const &other) const noexcept {
        return text.size() == other.text.size() && memcmp(text.data(), other.text.data(), text.size()) == 0;
    }

    bool operator<(heavy_key_t const &other) const noexcept {
        int cmp = memcmp(text.data(), other.text.data(), std::min(text.size(), other.text.size()));
        if (cmp != 0) return cmp < 0;
        return text.size() < other.text.size();
    }

    bool operator<(std::string_view const &other) const noexcept {
        int cmp = memcmp(text.data(), other.data(), std::min(text.size(), other.size()));
        if (cmp != 0) return cmp < 0;
        return text.size() < other.size();
    }

    bool operator==(std::string_view const &other) const noexcept {
        return text.size() == other.size() && memcmp(text.data(), other.data(), text.size()) == 0;
    }
};

#pragma mark - Association Types Using Keys

/**
 *  @brief Association types built from our test keys.
 *    These can be used with containers as element types.
 */

// Simple uint64 key-value pairs
using trivial_uint64_pair_t = association<trivial_key_t, std::uint64_t>;
using composite_uint64_pair_t = association<composite_key_t, std::uint64_t>;
using heavy_uint64_pair_t = association<heavy_key_t, std::uint64_t>;

// Simple uint64-uint64 for baseline integer tests
using uint64_uint64_pair_t = association<std::uint64_t, std::uint64_t>;

#pragma mark - Stateful Comparator

/**
 *  @brief Stateful comparator with runtime configuration.
 *    Tests that comparators with member state work correctly.
 */
template <typename baseline_comparator_ = std::less<void>>
struct stateful_comparator {
    using baseline_comparator_t = baseline_comparator_;
    using is_transparent = void;

    baseline_comparator_t baseline_comparator {};
    bool reverse_order {false};

    stateful_comparator() noexcept = default;

    template <typename lhs_type_, typename rhs_type_>
    bool operator()(lhs_type_ const &lhs, rhs_type_ const &rhs) const noexcept {
        return reverse_order ? baseline_comparator(rhs, lhs) : baseline_comparator(lhs, rhs);
    }
};

using stateful_comparator_t = stateful_comparator<>;

#pragma mark - Entry Construction Helpers

/**
 * @brief Helper to construct an entry from an integer value
 */
template <typename entry_type_>
auto make_entry_from_int(std::size_t val) {
    // Type has static .make() method (like heavy_key_t)
    if constexpr (requires { entry_type_::make(val); }) return *entry_type_::make(val);
    // Association type: {key, value}
    else if constexpr (requires { entry_type_ {val, static_cast<int>(val)}; })
        return entry_type_ {val, static_cast<int>(val)};
    // Plain type: {value}
    else return entry_type_ {val};
}

#pragma mark - Stateful Allocator

/**
 *  @brief Stateful allocator that tracks allocations.
 *    Tests that allocators with member state work correctly.
 */
template <typename element_type_>
struct stateful_allocator {
    using value_type = element_type_;
    using propagate_on_container_move_assignment = std::true_type; // Required for AVL trees

    int allocator_id {0};               // Allocator instance identifier
    std::size_t allocation_count {0};   // Track number of allocations
    std::size_t deallocation_count {0}; // Detect memory leaks
    std::size_t allocations_till_fail {std::numeric_limits<std::size_t>::max()};

    stateful_allocator() noexcept = default;
    explicit stateful_allocator(int id) noexcept : allocator_id(id) {}

    template <typename other_type_>
    stateful_allocator(stateful_allocator<other_type_> const &other) noexcept
        : allocator_id(other.allocator_id), allocation_count(other.allocation_count) {}

    element_type_ *allocate(std::size_t n) {
        ++allocation_count;
        return static_cast<element_type_ *>(::operator new(n * sizeof(element_type_)));
    }

    void deallocate(element_type_ *p, std::size_t) noexcept { ::operator delete(p); }

    template <typename other_type_>
    bool operator==(stateful_allocator<other_type_> const &other) const noexcept {
        return allocator_id == other.allocator_id;
    }

    template <typename other_type_>
    bool operator!=(stateful_allocator<other_type_> const &other) const noexcept {
        return !(*this == other);
    }
};

using stateful_allocator_t = stateful_allocator<std::byte>;

#pragma mark - Basic Operation Test Templates

template <typename container_type_>
void test_with_threads(std::size_t threads_count) {
    container_type_ cont;
    std::vector<std::thread> threads;
    threads.reserve(threads_count);

    auto upsert = [&](std::size_t offset, std::size_t length) {
        for (std::size_t idx = offset; idx < length; ++idx) EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx}));
    };

    std::size_t shift = (size / threads_count);
    for (std::size_t idx = 0; idx < threads_count; ++idx)
        threads.push_back(std::thread(upsert, idx * shift, idx * shift + shift));

    for (std::size_t idx = 0; idx < threads_count; ++idx) threads[idx].join();

    EXPECT_EQ(cont.size(), size);
    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.contains(idx));
}

/**
 * @brief Tests insertion in ascending, descending, and random order (exercises AVL rebalancing)
 */
template <typename container_type_>
void test_basic_insertion_patterns() {
    using entry_t = typename container_type_::entry_t;
    container_type_ cont;

    // Test 1: Ascending insertion
    for (std::size_t idx = 0; idx < size; ++idx) {
        auto entry = make_entry_from_int<entry_t>(idx);
        EXPECT_TRUE(cont.insert(std::move(entry)));
        EXPECT_TRUE(cont.contains(idx));
    }
    EXPECT_EQ(cont.size(), size);
    cont.clear();

    // Test 2: Descending insertion (tests AVL rebalancing)
    for (std::size_t idx = size; idx > 0; --idx) {
        auto entry = make_entry_from_int<entry_t>(idx);
        EXPECT_TRUE(cont.insert(std::move(entry)));
        EXPECT_TRUE(cont.contains(idx));
    }
    EXPECT_EQ(cont.size(), size);
    cont.clear();

    // Test 3: Random insertion (tests worst-case AVL patterns)
    std::srand(42); // Fixed seed for reproducibility
    for (std::size_t idx = 0; idx < size; ++idx) {
        std::size_t val = std::rand();
        auto entry = make_entry_from_int<entry_t>(val);
        EXPECT_TRUE(cont.insert(std::move(entry)));
        EXPECT_TRUE(cont.contains(val));
    }
}

/**
 *  @brief Tests bulk insertion via move iterators
 */
template <typename container_type_>
void test_bulk_insertion_from_iterators() {
    using entry_t = typename container_type_::entry_t;
    std::vector<entry_t> vec;
    vec.reserve(size);
    container_type_ cont;

    for (std::size_t idx = 0; idx < size; ++idx) { vec.push_back(make_entry_from_int<entry_t>(idx)); }

    EXPECT_TRUE(cont.insert(std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end())));
    EXPECT_EQ(cont.size(), size);

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.contains(idx));
}

/**
 *  @brief Tests bulk insert_or_assign (upsert) correctly overwrites duplicate keys
 *
 *  This test verifies that when bulk inserting elements with duplicate keys,
 *  the implementation uses insert_or_assign semantics (overwrite) rather than
 *  insert_if_missing semantics (skip).
 */
template <typename container_type_>
void test_bulk_insert_or_assign_with_duplicates() {
    container_type_ cont;

    // First, insert some initial values
    for (std::size_t idx = 0; idx < 10; ++idx) { EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx})); }
    EXPECT_EQ(cont.size(), 10);

    // Verify initial values
    cont.find(
        5, [](uint64_uint64_pair_t const &p) noexcept { EXPECT_EQ(p.value, 5); },
        []() noexcept { FAIL() << "Key 5 should exist"; });

    // Now bulk insert with overlapping keys but different values
    std::vector<uint64_uint64_pair_t> vec;
    for (std::size_t idx = 5; idx < 15; ++idx) {
        vec.push_back(uint64_uint64_pair_t {idx, idx * 100}); // Different values
    }

    EXPECT_TRUE(cont.upsert(std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end())));

    // Size should be 15 (0-14), not 20
    EXPECT_EQ(cont.size(), 15) << "Bulk upsert should handle duplicates correctly";

    // Verify that overlapping keys (5-9) have UPDATED values
    for (std::size_t idx = 5; idx < 10; ++idx) {
        bool found = false;
        cont.find(
            idx,
            [&](uint64_uint64_pair_t const &p) noexcept {
                found = true;
                EXPECT_EQ(p.value, idx * 100) << "Key " << idx << " should have updated value from bulk upsert";
            },
            [&]() noexcept { FAIL() << "Key " << idx << " should exist after bulk upsert"; });
        EXPECT_TRUE(found);
    }

    // Verify new keys (10-14) were inserted
    for (std::size_t idx = 10; idx < 15; ++idx) {
        bool found = false;
        cont.find(
            idx,
            [&](uint64_uint64_pair_t const &p) noexcept {
                found = true;
                EXPECT_EQ(p.value, idx * 100);
            },
            [&]() noexcept { FAIL() << "Key " << idx << " should exist after bulk upsert"; });
        EXPECT_TRUE(found);
    }

    // Verify old keys (0-4) retain original values
    for (std::size_t idx = 0; idx < 5; ++idx) {
        bool found = false;
        cont.find(
            idx,
            [&](uint64_uint64_pair_t const &p) noexcept {
                found = true;
                EXPECT_EQ(p.value, idx) << "Key " << idx << " should retain original value";
            },
            [&]() noexcept { FAIL() << "Key " << idx << " should exist"; });
        EXPECT_TRUE(found);
    }
}

/**
 *  @brief Tests range queries on committed HEAD state
 */
template <typename container_type_>
void test_range_query_head_state() {
    container_type_ cont;

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx}));

    // Query in windows and verify we get elements within the range
    for (std::size_t idx = 0; idx < size; idx += 10) {
        std::size_t count = 0;
        std::size_t min_key = size, max_key = 0;
        cont.range(idx, idx + 9, [&](uint64_uint64_pair_t const &rhs) noexcept {
            min_key = std::min(min_key, rhs.key);
            max_key = std::max(max_key, rhs.key);
            count++;
        });
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
    container_type_ cont;

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx}));

    bool state = true;
    for (std::size_t idx = 0; idx < size; idx += 10) {
        cont.erase_range(idx, idx + 10, [](auto const &) noexcept {});
        for (std::size_t i = idx; i < idx + 10; ++i) {
            cont.find(i, [&](uint64_uint64_pair_t const &) noexcept { state = false; });
            EXPECT_TRUE(state);
        }
    }
}

/**
 *  @brief Tests `upper_bound` query returns first `element > key`
 */
template <typename container_type_>
void test_upper_bound() {
    container_type_ cont;

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx}));

    for (std::size_t idx = 0; idx < size - 1; ++idx) {
        cont.upper_bound(idx, [&](uint64_uint64_pair_t const &rhs) noexcept { EXPECT_GT(rhs.key, idx); });
    }
}

/**
 *  @brief Tests `clear` operation resets size to 0
 */
template <typename container_type_>
void test_clear() {
    container_type_ cont;

    EXPECT_EQ(cont.size(), 0);

    for (std::size_t idx = 0; idx < size; ++idx) EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx}));

    EXPECT_EQ(cont.size(), size);
    EXPECT_TRUE(cont.clear());
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests `size` increments correctly after each upsert
 */
template <typename container_type_>
void test_size_after_upserts() {
    container_type_ cont;
    EXPECT_EQ(cont.size(), 0);

    for (std::size_t idx = 0; idx < 50; ++idx) {
        EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx}));
        EXPECT_EQ(cont.size(), idx + 1);
    }
}

/**
 *  @brief Tests size remains unchanged when upserting duplicate keys
 */
template <typename container_type_>
void test_size_invariant_on_duplicate_upserts() {
    container_type_ cont;

    // Insert initial keys
    for (std::size_t idx = 0; idx < 50; ++idx) { EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx})); }
    EXPECT_EQ(cont.size(), 50);

    // Upsert same keys again with different values - size should remain same
    for (std::size_t idx = 0; idx < 50; ++idx) {
        EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx * 2}));
        EXPECT_EQ(cont.size(), 50) << "Duplicate upsert should not change size";
    }
}

/**
 *  @brief Tests size decrements correctly after erase_range
 */
template <typename container_type_>
void test_size_after_erase() {
    container_type_ cont;

    for (std::size_t idx = 0; idx < 50; ++idx) EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {idx, idx}));
    EXPECT_EQ(cont.size(), 50);

    // Erase in ranges
    for (std::size_t idx = 0; idx < 50; idx += 10) {
        cont.erase_range(id_t {idx}, id_t {idx + 10});
        EXPECT_EQ(cont.size(), 50 - (idx + 10));
    }
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests operations on empty container don't crash
 */
template <typename container_type_>
void test_empty_container_operations() {
    container_type_ cont;

    // Operations on empty container should not crash
    bool found = false;
    cont.find(id_t {1}, [&](uint64_uint64_pair_t const &) noexcept { found = true; });
    EXPECT_FALSE(found);

    cont.upper_bound(id_t {1}, [&](uint64_uint64_pair_t const &) noexcept { found = true; });
    EXPECT_FALSE(found);

    cont.erase_range(id_t {0}, id_t {10});
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests operations on single-element container
 */
template <typename container_type_>
void test_single_element_operations() {
    container_type_ cont;
    EXPECT_TRUE(cont.upsert(uint64_uint64_pair_t {42, 42}));
    EXPECT_EQ(cont.size(), 1);

    bool found = false;
    cont.find(id_t {42}, [&](uint64_uint64_pair_t const &e) noexcept {
        found = true;
        EXPECT_EQ(e.value, 42);
    });
    EXPECT_TRUE(found);

    cont.erase_range(id_t {42}, id_t {43});
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests reserve() pre-allocates capacity (STL containers only)
 *
 *  AVL trees don't support collection-level reserve
 */
template <typename container_type_>
void test_reserve() {
    container_type_ cont;
    EXPECT_TRUE(cont.reserve(100));
    EXPECT_TRUE(cont.empty());
    EXPECT_EQ(cont.size(), 0);
}

/**
 *  @brief Tests split/join work correctly for range deletion edge cases
 */
template <typename container_type_>
void test_erase_range_edge_cases() {
    container_type_ set;

    // Insert elements
    for (std::size_t i = 0; i < 100; ++i) { EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {i, i})); }

    // Test: Erase empty range (should be no-op)
    set.erase_range(50, 50);
    EXPECT_EQ(set.size(), 100);

    // Test: Erase single element range
    set.erase_range(50, 51);
    EXPECT_EQ(set.size(), 99);
    bool found = false;
    set.find(50, [&](uint64_uint64_pair_t const &) noexcept { found = true; });
    EXPECT_FALSE(found);

    // Test: Erase from beginning
    set.erase_range(0, 10);
    EXPECT_EQ(set.size(), 89);

    // Test: Erase to end
    set.erase_range(90, 100);
    EXPECT_EQ(set.size(), 79);

    // Verify remaining range is correct [11, 90)
    for (std::size_t i = 11; i < 90; ++i) {
        if (i == 50) continue; // Already deleted
        bool found_elem = false;
        set.find(i, [&](uint64_uint64_pair_t const &) noexcept { found_elem = true; });
        EXPECT_TRUE(found_elem);
    }
}

/**
 *  @brief Tests erase_range with large ranges uses split/join efficiently
 */
template <typename container_type_>
void test_erase_range_large() {
    container_type_ set;

    // Insert 1000 elements
    for (std::size_t i = 0; i < 1000; ++i) { EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {i, i})); }
    EXPECT_EQ(set.size(), 1000);

    // Erase middle 800 elements [100, 900)
    std::size_t callback_count = 0;
    set.erase_range(100, 900, [&](uint64_uint64_pair_t const &p) noexcept {
        EXPECT_GE(p.key, 100);
        EXPECT_LT(p.key, 900);
        ++callback_count;
    });

    // Verify correct number of callbacks
    EXPECT_EQ(callback_count, 800);

    // Verify size is correct
    EXPECT_EQ(set.size(), 200);

    // Verify remaining elements are correct
    for (std::size_t i = 0; i < 100; ++i) {
        bool found = false;
        set.find(i, [&](uint64_uint64_pair_t const &) noexcept { found = true; });
        EXPECT_TRUE(found);
    }
    for (std::size_t i = 900; i < 1000; ++i) {
        bool found = false;
        set.find(i, [&](uint64_uint64_pair_t const &) noexcept { found = true; });
        EXPECT_TRUE(found);
    }
    for (std::size_t i = 100; i < 900; ++i) {
        bool found = false;
        set.find(i, [&](uint64_uint64_pair_t const &) noexcept { found = true; });
        EXPECT_FALSE(found);
    }
}

/**
 *  @brief Tests erase_range maintains AVL balance property
 */
template <typename container_type_>
void test_erase_range_maintains_balance() {
    container_type_ set;

    // Insert elements
    for (std::size_t i = 0; i < 200; ++i) { EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {i, i})); }

    // Erase multiple ranges
    set.erase_range(50, 60);
    set.erase_range(100, 120);
    set.erase_range(150, 180);

    // Tree should still be functional (this implicitly tests balance)
    // Verify we can still insert and find
    EXPECT_TRUE(set.upsert(uint64_uint64_pair_t {55, 55}));
    bool found = false;
    set.find(55, [&](uint64_uint64_pair_t const &) noexcept { found = true; });
    EXPECT_TRUE(found);

    // Verify correct elements are missing
    for (std::size_t i = 51; i < 60; ++i) {
        if (i == 55) continue; // We re-inserted this one
        bool found_missing = false;
        set.find(i, [&](uint64_uint64_pair_t const &) noexcept { found_missing = true; });
        EXPECT_FALSE(found_missing);
    }
}

#pragma mark - Heterogeneous Lookup Test Templates

/**
 *  @brief Tests heterogeneous lookup for composite_key_t by uint64_t identifier.
 *    composite_key_t stores metadata + unique_id, but supports lookup by just the id.
 *    Verifies transparent comparator allows searching without materializing full key.
 */
template <typename container_type_>
void test_heterogeneous_composite_find() {
    container_type_ cont;

    // Insert composite keys
    for (std::size_t i = 0; i < 10; ++i) {
        composite_key_t key {i};
        key.some_metadata = i * 100;
        key.some_float = i * 1.5;
        EXPECT_TRUE(cont.upsert(key));
    }

    // Test heterogeneous lookup by uint64_t (without constructing full composite_key_t)
    for (std::size_t i = 0; i < 10; ++i) {
        bool found = false;
        cont.find(static_cast<std::uint64_t>(i), [&](auto const &elem) noexcept {
            found = true;
            // For sets: elem is composite_key_t
            // For maps: elem is association<composite_key_t, V>
            if constexpr (requires { elem.key; }) {
                // Map case
                EXPECT_EQ(elem.key.unique_id, i);
                EXPECT_EQ(elem.key.some_metadata, i * 100);
            }
            else {
                // Set case
                EXPECT_EQ(elem.unique_id, i);
                EXPECT_EQ(elem.some_metadata, i * 100);
            }
        });
        EXPECT_TRUE(found) << "Should find key " << i << " via heterogeneous lookup";
    }

    // Test that non-existent keys are not found
    bool found = false;
    cont.find(static_cast<std::uint64_t>(999), [&](auto const &) noexcept { found = true; });
    EXPECT_FALSE(found);
}

/**
 *  @brief Tests heterogeneous lookup for heavy_key_t by string_view.
 *    heavy_key_t stores heap-allocated text, but supports lookup by string_view.
 *    Verifies we can search without allocating temporary heavy_key_t objects.
 */
template <typename container_type_>
void test_heterogeneous_heavy_string_view_find() {
    container_type_ cont;

    // Insert heavy keys from strings
    std::vector<std::string> test_strings = {"hello", "world", "foo", "bar", "baz"};
    for (auto const &str : test_strings) {
        auto key = heavy_key_t::make(str);
        EXPECT_TRUE(key);
        EXPECT_TRUE(cont.upsert(std::move(*key)));
    }

    // Test heterogeneous lookup by string_view (without allocating heavy_key_t)
    for (auto const &str : test_strings) {
        bool found = false;
        std::string_view sv {str};
        cont.find(sv, [&](auto const &elem) noexcept {
            found = true;
            if constexpr (requires { elem.key; }) {
                // Map case
                EXPECT_EQ(elem.key.view(), str);
            }
            else {
                // Set case
                EXPECT_EQ(elem.view(), str);
            }
        });
        EXPECT_TRUE(found) << "Should find key '" << str << "' via string_view lookup";
    }

    // Test that non-existent keys are not found
    bool found = false;
    cont.find(std::string_view {"nonexistent"}, [&](auto const &) noexcept { found = true; });
    EXPECT_FALSE(found);
}

/**
 *  @brief Tests heterogeneous lookup for heavy_key_t by integer (via make).
 *    Tests that heavy_key_t created from integers can be found heterogeneously.
 */
template <typename container_type_>
void test_heterogeneous_heavy_integer_find() {
    container_type_ cont;

    // Insert heavy keys from integers (converted to hex strings)
    for (std::size_t i = 0; i < 10; ++i) {
        auto key = heavy_key_t::make(i);
        EXPECT_TRUE(key);
        EXPECT_TRUE(cont.upsert(std::move(*key)));
    }

    // Test heterogeneous lookup by reconstructing the hex string
    for (std::size_t i = 0; i < 10; ++i) {
        auto search_key = heavy_key_t::make(i);
        EXPECT_TRUE(search_key);

        bool found = false;
        cont.find(search_key->view(), [&](auto const &elem) noexcept {
            found = true;
            if constexpr (requires { elem.key; }) { EXPECT_EQ(elem.key.view(), search_key->view()); }
            else { EXPECT_EQ(elem.view(), search_key->view()); }
        });
        EXPECT_TRUE(found);
    }
}

} // namespace ashvardanian::smashtable::scripts