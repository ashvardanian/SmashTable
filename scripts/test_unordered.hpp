/**
 *  @brief Template test functions for unordered containers, exercising the hash-specific surface -
 *      growth through rehashes, tombstone reuse, iteration, and the lock-free atomic operations.
 *  @author Ash Vardanian
 *  @file scripts/test_unordered.hpp
 *  @date August 16, 2026
 *
 *  @section test_unordered_scope Scope
 *
 *  The suites in @c test_basic.hpp are written against an ordered, transactional surface -
 *  @c upsert, @c find_copy, @c range, @c erase_range, @c transaction - none of which a hash table
 *  offers, so nothing there is reusable here. What the two files do share are the key fixtures,
 *  which is why this header includes @c test_basic.hpp rather than restating them.
 *
 *  @section test_unordered_oracle Oracle
 *
 *  Every non-trivial suite mirrors the container into a @c std::unordered_map of integer
 *  identifiers, and every membership claim is checked in both directions - present keys are found,
 *  absent keys are not. A broken probe sequence hides from a size check but not from that pair.
 *
 *  @section test_unordered_threads Threading
 *
 *  The concurrent suites hand each thread a disjoint key range against a table that was reserved
 *  before the first thread started, so no operation can trigger a reallocation and no two threads
 *  ever touch the same key. Threads are joined before anything is asserted, leaving the outcome
 *  independent of the interleaving.
 */
#pragma once
#include <cstddef> // `std::size_t`

#include <atomic>        // `std::atomic`
#include <limits>        // `std::numeric_limits`
#include <set>           // `std::set`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view`
#include <thread>        // `std::thread`
#include <type_traits>   // `std::is_void`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include <smashtable/basic_hash_table.hpp>
#include <smashtable/concurrent_hash_table.hpp>

#include "test_basic.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Fixture Helpers

/** @brief True when the container stores values alongside keys, false for sets. */
template <typename container_type_>
inline constexpr bool unordered_has_values_k = !std::is_void_v<typename container_type_::mapped_type>;

/** @brief Builds a key of the container's key type from an integer identifier. */
template <typename key_type_>
key_type_ unordered_key_from(std::size_t identifier) noexcept {
    if constexpr (std::is_same_v<key_type_, std::string>) return std::string("key-") + std::to_string(identifier);
    else return key_type_(identifier);
}

/** @brief Builds a mapped value of the container's value type from an integer identifier. */
template <typename value_type_>
value_type_ unordered_value_from(std::size_t identifier) noexcept {
    if constexpr (std::is_same_v<value_type_, std::string>) return std::string("value-") + std::to_string(identifier);
    else return value_type_(identifier);
}

/** @brief Inserts the element for @p identifier, choosing the set or map arity of @c emplace. */
template <typename container_type_, typename... tags_types_>
void unordered_emplace(container_type_ &container, std::size_t identifier, tags_types_... tags) noexcept {
    using key_t = typename container_type_::key_type;
    if constexpr (unordered_has_values_k<container_type_>) {
        using mapped_t = typename container_type_::mapped_type;
        container.emplace(unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(identifier), tags...);
    }
    else { container.emplace(unordered_key_from<key_t>(identifier), tags...); }
}

/** @brief Inserts the key of @p identifier carrying the value of @p value_identifier, for maps only. */
template <typename container_type_, typename... tags_types_>
void unordered_emplace_valued(container_type_ &container, std::size_t identifier, std::size_t value_identifier,
                              tags_types_... tags) noexcept {
    using key_t = typename container_type_::key_type;
    using mapped_t = typename container_type_::mapped_type;
    static_assert(unordered_has_values_k<container_type_>, "Only maps carry a separate value");
    container.emplace(unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(value_identifier), tags...);
}

/** @brief Checks that @p identifier is present and, on maps, that it still carries @p value_identifier. */
template <typename container_type_>
void unordered_verify_present(container_type_ &container, std::size_t identifier,
                              std::size_t value_identifier) noexcept {
    using key_t = typename container_type_::key_type;
    auto const key = unordered_key_from<key_t>(identifier);
    st_verify_((container.contains(key)) && "a key that was inserted must be found");
    st_verify_eq_(container.count(key), 1u);
    st_verify_((container.find(key) != container.end()) && "find must reach the element");
    if constexpr (unordered_has_values_k<container_type_>) {
        using mapped_t = typename container_type_::mapped_type;
        st_verify_((container.at(key) == unordered_value_from<mapped_t>(value_identifier)) &&
                   "the mapped value must survive every rehash");
    }
}

/** @brief Checks that @p identifier is absent, both through @c contains and through @c find. */
template <typename container_type_>
void unordered_verify_absent(container_type_ &container, std::size_t identifier) noexcept {
    using key_t = typename container_type_::key_type;
    auto const key = unordered_key_from<key_t>(identifier);
    st_verify_(!(container.contains(key)));
    st_verify_eq_(container.count(key), 0u);
    st_verify_((container.find(key) == container.end()) && "find must report the end for a missing key");
}

/**
 *  @brief Collects the keys reachable by iteration, and how many visits it took to reach them.
 *  @note The set alone cannot catch a duplicate visit - it swallows one - so the count comes back
 *    with it, and a caller compares that against @c size() to pin one visit per element.
 */
template <typename container_type_>
std::pair<std::set<typename container_type_::key_type>, std::size_t> unordered_keys_by_iteration(
    container_type_ &container) noexcept {
    std::set<typename container_type_::key_type> collected;
    std::size_t visits = 0;
    for (auto position = container.begin(); position != container.end(); ++position, ++visits)
        collected.insert(position.key());
    return {std::move(collected), visits};
}

/**
 *  @brief Collects the keys reachable by @c for_each, and how many visits it took to reach them.
 *  @note The count is what catches a duplicate visit; the set on its own would hide it.
 */
template <typename container_type_>
std::pair<std::set<typename container_type_::key_type>, std::size_t> unordered_keys_by_for_each(
    container_type_ &container) noexcept {
    std::set<typename container_type_::key_type> collected;
    std::size_t visits = 0;
    container.for_each([&](auto const &address) noexcept {
        collected.insert(address.key());
        ++visits;
    });
    return {std::move(collected), visits};
}

/** @brief The keys the container is expected to hold, built from the same identifiers. */
template <typename container_type_>
std::set<typename container_type_::key_type> unordered_expected_keys(
    std::unordered_map<std::size_t, std::size_t> const &oracle) noexcept {
    using key_t = typename container_type_::key_type;
    std::set<key_t> expected;
    for (auto const &entry : oracle) expected.insert(unordered_key_from<key_t>(entry.first));
    return expected;
}

/** @brief Re-checks the whole container against the oracle, in both directions. */
template <typename container_type_>
void unordered_verify_against_oracle(container_type_ &container,
                                     std::unordered_map<std::size_t, std::size_t> const &oracle,
                                     std::size_t identifiers_bound) noexcept {
    st_verify_eq_(container.size(), oracle.size());
    st_verify_eq_(container.empty(), oracle.empty());
    for (std::size_t identifier = 0; identifier < identifiers_bound; ++identifier) {
        auto const found = oracle.find(identifier);
        if (found != oracle.end()) unordered_verify_present(container, identifier, found->second);
        else unordered_verify_absent(container, identifier);
    }
    auto const [walked_keys, walked_visits] = unordered_keys_by_iteration(container);
    st_verify_eq_(walked_visits, container.size());
    st_verify_((walked_keys == unordered_expected_keys<container_type_>(oracle)) &&
               "iteration must reach exactly the live keys");
}

#pragma endregion Fixture Helpers

#pragma region Basic Unordered Operations

/** @brief Tests that every read and every teardown path tolerates an empty container. */
template <typename container_type_>
void test_unordered_empty_container_operations() {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    st_verify_(container.empty());
    st_verify_eq_(container.size(), 0u);
    st_verify_eq_(container.bucket_count(), 0u);
    unordered_verify_absent(container, 1);
    st_verify_((container.begin() == container.end()) && "an empty container iterates over nothing");
    st_verify_eq_(unordered_keys_by_for_each(container).first.size(), 0u);
    st_verify_(!(container.erase(unordered_key_from<key_t>(7))));

    // The same operations must also survive on a container that owns memory but holds nothing.
    auto allocated = container_t::make(128);
    st_verify_((allocated) && "allocating a small table must succeed");
    container_t reserved = *std::move(allocated);
    st_verify_(reserved.empty());
    st_verify_eq_(reserved.size(), 0u);
    st_verify_((reserved.bucket_count()) > (0u));
    unordered_verify_absent(reserved, 1);
    st_verify_((reserved.begin() == reserved.end()) && "a reserved but empty container iterates over nothing");
    st_verify_eq_(unordered_keys_by_for_each(reserved).first.size(), 0u);

    // Teardown on an empty container must be repeatable.
    reserved.clear();
    reserved.clear();
    st_verify_eq_(reserved.size(), 0u);
    reserved.shrink_to_fit();
    st_verify_eq_(reserved.bucket_count(), 0u);
    st_verify_(reserved.empty());
}

/** @brief Tests the insert, lookup, overwrite and erase cycle on a single element. */
template <typename container_type_>
void test_unordered_single_element_operations() {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    unordered_emplace(container, 42);
    st_verify_eq_(container.size(), 1u);
    st_verify_(!(container.empty()));
    unordered_verify_present(container, 42, 42);
    unordered_verify_absent(container, 43);

    auto const [collected, collected_visits] = unordered_keys_by_iteration(container);
    st_verify_eq_(collected_visits, container.size());
    st_verify_eq_(collected.size(), 1u);
    st_verify_((*collected.begin() == unordered_key_from<key_t>(42)) && "the only key must be the inserted one");

    // Re-inserting the same key overwrites rather than duplicates.
    if constexpr (unordered_has_values_k<container_t>) {
        unordered_emplace_valued(container, 42, 99);
        st_verify_eq_(container.size(), 1u);
        unordered_verify_present(container, 42, 99);
    }
    else {
        unordered_emplace(container, 42);
        st_verify_eq_(container.size(), 1u);
    }

    st_verify_(container.erase(unordered_key_from<key_t>(42)));
    st_verify_eq_(container.size(), 0u);
    st_verify_(container.empty());
    unordered_verify_absent(container, 42);
    st_verify_(!(container.erase(unordered_key_from<key_t>(42))));

    // A tombstoned slot must accept the same key again.
    unordered_emplace(container, 42);
    st_verify_eq_(container.size(), 1u);
    unordered_verify_present(container, 42, 42);
}

/** @brief Tests that growth from an unallocated table preserves every element across many rehashes. */
template <typename container_type_>
void test_unordered_growth_through_rehashes(std::size_t size = 4000) {

    using container_t = container_type_;

    container_t container;
    std::unordered_map<std::size_t, std::size_t> oracle;
    std::size_t rehashes = 0;
    auto previous_buckets = container.bucket_count();

    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
        st_verify_eq_(container.size(), oracle.size());
        if (container.bucket_count() != previous_buckets) {
            previous_buckets = container.bucket_count();
            ++rehashes;
            // Everything inserted so far must have survived the reallocation.
            for (std::size_t earlier = 0; earlier <= identifier; ++earlier)
                unordered_verify_present(container, earlier, earlier);
        }
    }

    st_verify_((rehashes) > (3) && "growing to thousands of elements must rehash repeatedly");
    unordered_verify_against_oracle(container, oracle, size + 100);
}

/** @brief Tests that a tombstoned table still finds its survivors, then refills the tombstones. */
template <typename container_type_>
void test_unordered_tombstone_reuse(std::size_t size = 2000) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    std::unordered_map<std::size_t, std::size_t> oracle;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }
    unordered_verify_against_oracle(container, oracle, size);

    // Erase three quarters of the keys, leaving long runs of tombstones behind. A probe sequence
    // that treats a tombstone as the end of the run loses the survivors right here.
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        if (identifier % 4 == 0) continue;
        st_verify_(container.erase(unordered_key_from<key_t>(identifier)));
        oracle.erase(identifier);
    }
    unordered_verify_against_oracle(container, oracle, size);

    // Refill the tombstones with the very keys that made them.
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        if (identifier % 4 == 0) continue;
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }
    unordered_verify_against_oracle(container, oracle, size);

    // And with keys that were never present, so the tombstones take unfamiliar hashes.
    for (std::size_t identifier = size; identifier < size + 500; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }
    unordered_verify_against_oracle(container, oracle, size + 500);
}

/** @brief Tests that iteration and @c for_each each visit every live element exactly once. */
template <typename container_type_>
void test_unordered_full_iteration(std::size_t size = 1500) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    std::unordered_map<std::size_t, std::size_t> oracle;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }

    auto const expected = unordered_expected_keys<container_t>(oracle);
    st_verify_eq_(expected.size(), size);

    // A count alone would pass while a slot is visited twice and another skipped, so compare sets.
    std::size_t visits = 0;
    std::set<key_t> visited;
    container.for_each([&](auto const &address) noexcept {
        visited.insert(address.key());
        ++visits;
    });
    st_verify_eq_(visits, size);
    st_verify_((visited == expected) && "for_each must visit every live key exactly once");
    auto const [iterated, iterated_visits] = unordered_keys_by_iteration(container);
    st_verify_eq_(iterated_visits, container.size());
    st_verify_((iterated == expected) && "iteration must agree with for_each");

    // The const overloads walk the same slots through the const element reference.
    container_t const &const_container = container;
    std::size_t const_visits = 0;
    std::set<key_t> const_visited;
    const_container.for_each([&](auto const &address) noexcept {
        const_visited.insert(address.key());
        ++const_visits;
    });
    st_verify_eq_(const_visits, size);
    st_verify_((const_visited == expected) && "the const for_each must see the same keys");

    // Half the elements go away, and both walks must follow.
    for (std::size_t identifier = 0; identifier < size; identifier += 2) {
        st_verify_(container.erase(unordered_key_from<key_t>(identifier)));
        oracle.erase(identifier);
    }
    auto const survivors = unordered_expected_keys<container_t>(oracle);
    auto const [seen_by_for_each, for_each_visits] = unordered_keys_by_for_each(container);
    auto const [seen_by_walk, walk_visits] = unordered_keys_by_iteration(container);
    st_verify_eq_(for_each_visits, container.size());
    st_verify_eq_(walk_visits, container.size());
    st_verify_((seen_by_for_each == survivors) && "for_each must skip tombstones");
    st_verify_((seen_by_walk == survivors) && "iteration must skip tombstones");
}

/** @brief Tests move construction, move assignment and @c swap against the oracle. */
template <typename container_type_>
void test_unordered_moves_and_swaps(std::size_t size = 600) {

    using container_t = container_type_;

    container_t source;
    std::unordered_map<std::size_t, std::size_t> oracle;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(source, identifier);
        oracle[identifier] = identifier;
    }

    container_t moved_into(std::move(source));
    unordered_verify_against_oracle(moved_into, oracle, size);

    container_t assigned_into;
    assigned_into = std::move(moved_into);
    unordered_verify_against_oracle(assigned_into, oracle, size);

    // Swapping with an empty container hands the whole payload over and takes nothing back.
    container_t empty_partner;
    assigned_into.swap(empty_partner);
    st_verify_(assigned_into.empty());
    st_verify_eq_(assigned_into.size(), 0u);
    unordered_verify_against_oracle(empty_partner, oracle, size);

    // Swapping two populated containers exchanges them wholesale.
    container_t other;
    std::unordered_map<std::size_t, std::size_t> other_oracle;
    for (std::size_t identifier = size; identifier < size + 200; ++identifier) {
        unordered_emplace(other, identifier);
        other_oracle[identifier] = identifier;
    }
    empty_partner.swap(other);
    unordered_verify_against_oracle(empty_partner, other_oracle, size + 200);
    unordered_verify_against_oracle(other, oracle, size + 200);
}

/** @brief Tests @c reserve idempotence, explicit @c rehash, @c clear and @c shrink_to_fit. */
template <typename container_type_>
void test_unordered_capacity_management(std::size_t size = 800) {

    using container_t = container_type_;

    using reserve_result_t = typename container_t::reserve_result_t;

    container_t container;
    st_verify_((container.reserve(size) == reserve_result_t::relayouted_k) &&
               "the first reserve on an empty table must allocate");
    st_verify_((container.reserve(size) == reserve_result_t::unchanged_k) &&
               "a second reserve for the same count must be a no-op");
    st_verify_((container.reserve(size / 2) == reserve_result_t::unchanged_k) &&
               "shrinking through reserve must be a no-op");
    auto const reserved_buckets = container.bucket_count();
    st_verify_((reserved_buckets) > (0u));

    std::unordered_map<std::size_t, std::size_t> oracle;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }
    st_verify_eq_(container.bucket_count(), reserved_buckets);
    unordered_verify_against_oracle(container, oracle, size);

    // An explicit rehash moves every element into a differently sized table. The bucket count is
    // scaled by a power of two, because the probe mask assumes a power-of-two slot count and
    // @c rehash forwards its argument without rounding.
    container.rehash(reserved_buckets * 4);
    st_verify_eq_(container.bucket_count(), reserved_buckets * 4);
    unordered_verify_against_oracle(container, oracle, size);

    // Shrinking gives the memory back without losing anything.
    container.shrink_to_fit();
    st_verify_((container.bucket_count()) <= (reserved_buckets * 4));
    unordered_verify_against_oracle(container, oracle, size);

    container.clear();
    st_verify_(container.empty());
    st_verify_eq_(container.size(), 0u);
    st_verify_((container.bucket_count()) > (0u) && "clear keeps the memory");
    unordered_verify_absent(container, 0);

    container.shrink_to_fit();
    st_verify_eq_(container.bucket_count(), 0u);

    // A cleared and shrunk container is still usable.
    unordered_emplace(container, 5);
    st_verify_eq_(container.size(), 1u);
    unordered_verify_present(container, 5, 5);
}

/** @brief Tests that the load factor stays under the documented cap and the counters stay honest. */
template <typename container_type_>
void test_unordered_load_factor_consistency(std::size_t size = 3000) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        st_verify_eq_(container.size(), identifier + 1);
        st_verify_(!(container.empty()));

        // The header lives in whole buckets, and the table never runs past three quarters full.
        st_verify_eq_(container.bucket_count() * hash_bucket_capacity_k, container.slots_count().raw);
        st_verify_((container.size() * 4) <= (container.slots_count().raw * 3) &&
                   "the load factor must stay under the 75% cap");
        st_verify_((container.size()) <= (container.capacity()));
    }

    // Erasures lower the size without lowering the slot count.
    auto const slots_before_erasures = container.slots_count().raw;
    for (std::size_t identifier = 0; identifier < size; identifier += 3)
        st_verify_(container.erase(unordered_key_from<key_t>(identifier)));
    st_verify_eq_(container.size(), size - ((size + 2) / 3));
    st_verify_eq_(container.slots_count().raw, slots_before_erasures);
    st_verify_(!(container.empty()));

    container.clear();
    st_verify_(container.empty());
    st_verify_eq_(container.size(), 0u);
    st_verify_eq_(container.slots_count().raw, slots_before_erasures);
}

#pragma endregion Basic Unordered Operations

#pragma region Heterogeneous Lookup

/**
 *  @brief Tests lookup by @c std::string_view against a container of @c std::string keys.
 *    Only meaningful where the equality predicate is transparent and the hash of a view matches
 *    the hash of the string it views, which the standard guarantees.
 */
template <typename container_type_>
void test_unordered_heterogeneous_string_view_lookup(std::size_t size = 400) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;
    static_assert(std::is_same_v<key_t, std::string>, "This suite is written for string-keyed containers");
    static_assert(
        requires { typename container_t::key_equal::is_transparent; },
        "Heterogeneous lookup needs a transparent equality predicate");

    container_t container;
    for (std::size_t identifier = 0; identifier < size; ++identifier) unordered_emplace(container, identifier);

    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        std::string const owned = unordered_key_from<key_t>(identifier);
        std::string_view const borrowed {owned};
        st_verify_((container.contains(borrowed)) && "a view of a present key must be found");
        st_verify_eq_(container.count(borrowed), 1u);
        st_verify_((container.find(borrowed) != container.end()) && "find must accept a view");
        if constexpr (unordered_has_values_k<container_t>) {
            using mapped_t = typename container_t::mapped_type;
            st_verify_((container.at(borrowed) == unordered_value_from<mapped_t>(identifier)) &&
                       "a view must reach the same value the owned key does");
        }
    }

    using namespace std::literals::string_view_literals;
    st_verify_(!(container.contains("key-missing"sv)));
    st_verify_eq_(container.count("key-missing"sv), 0u);

    // Erasing through a view must retire the same slot the owned key would have.
    std::string const first = unordered_key_from<key_t>(0);
    st_verify_(container.erase(std::string_view {first}));
    st_verify_(!(container.contains(first)));
    st_verify_eq_(container.size(), size - 1);
}

#pragma endregion Heterogeneous Lookup

#pragma region Concurrency

/** @brief The thread count the concurrent suites use, fixed so a failure reproduces. */
inline constexpr std::size_t unordered_threads_count_k = 4;

/** @brief The pinned table matching a growable one, since neither type names the other. */
template <typename container_type_>
struct unordered_pinned_of;

template <typename element_type_, typename hasher_type_, typename equals_type_, typename allocator_type_>
struct unordered_pinned_of<basic_hash_table<element_type_, hasher_type_, equals_type_, allocator_type_>> {
    using type = concurrent_hash_table<element_type_, hasher_type_, equals_type_, allocator_type_>;
};

/** @brief The pinned counterpart of @p container_type_, reached through @c release and @c adopt. */
template <typename container_type_>
using unordered_pinned_t = typename unordered_pinned_of<container_type_>::type;

/**
 *  @brief Tests concurrent @c emplace followed by concurrent @c find and @c contains on a frozen
 *    table, with each thread owning a disjoint range of keys.
 */
template <typename container_type_>
void test_unordered_concurrent_emplace_and_find(std::size_t per_thread = 2000) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;
    using mapped_t = typename container_t::mapped_type;
    static_assert(unordered_has_values_k<container_t>, "The atomic suites are written for maps");

    std::size_t const total = unordered_threads_count_k * per_thread;
    auto allocated = container_t::make(total * 2);
    st_verify_((allocated) && "the growable table must build");
    auto container = unordered_pinned_t<container_t>::adopt((*std::move(allocated)).release());
    st_verify_((container.capacity() >= total) && "the pinned table must hold what the writers will store");
    auto const slots_before = container.slots_count();

    std::vector<std::thread> threads;
    threads.reserve(unordered_threads_count_k);
    for (std::size_t thread_index = 0; thread_index < unordered_threads_count_k; ++thread_index)
        threads.emplace_back([&container, thread_index, per_thread]() noexcept {
            for (std::size_t offset = 0; offset < per_thread; ++offset) {
                std::size_t const identifier = thread_index * per_thread + offset;
                [[maybe_unused]] status_t const stored = container.emplace(unordered_key_from<key_t>(identifier),
                                                                           unordered_value_from<mapped_t>(identifier));
            }
        });
    for (auto &thread : threads) thread.join();

    st_verify_eq_(container.size(), total);
    st_verify_eq_(container.slots_count(), slots_before);

    // The readers run over the same disjoint ranges, so a miss is a lost element, not a race.
    std::atomic<std::size_t> found_by_contains {0};
    std::atomic<std::size_t> found_by_callback {0};
    std::atomic<std::size_t> wrong_values {0};
    threads.clear();
    for (std::size_t thread_index = 0; thread_index < unordered_threads_count_k; ++thread_index)
        threads.emplace_back([&, thread_index]() noexcept {
            std::size_t contains_hits = 0, callback_hits = 0, mismatches = 0;
            for (std::size_t offset = 0; offset < per_thread; ++offset) {
                std::size_t const identifier = thread_index * per_thread + offset;
                auto const key = unordered_key_from<key_t>(identifier);
                if (container.contains(key)) ++contains_hits;
                bool const reached = container.find(key, [&](auto const &address) noexcept {
                    if (!(address.value() == unordered_value_from<mapped_t>(identifier))) ++mismatches;
                });
                callback_hits += reached;
            }
            found_by_contains += contains_hits;
            found_by_callback += callback_hits;
            wrong_values += mismatches;
        });
    for (auto &thread : threads) thread.join();

    st_verify_eq_(found_by_contains.load(), total);
    st_verify_eq_(found_by_callback.load(), total);
    st_verify_eq_(wrong_values.load(), 0u);

    // The single-threaded view must agree with what the threads believe they wrote.
    std::size_t single_threaded_hits = 0;
    for (std::size_t identifier = 0; identifier < total; ++identifier)
        if (container.contains(unordered_key_from<key_t>(identifier))) ++single_threaded_hits;
    st_verify_eq_(single_threaded_hits, total);
}

/**
 *  @brief Tests concurrent @c update over pre-populated keys and concurrent @c erase afterwards,
 *    each thread again confined to its own range.
 *
 *  The table is filled while it is still growable, pinned for the concurrent phases, and handed back
 *  to a growable table to verify - the whole lifecycle @c release and @c adopt exist to express.
 */
template <typename container_type_>
void test_unordered_concurrent_update_and_erase(std::size_t per_thread = 1000) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;
    using mapped_t = typename container_t::mapped_type;
    static_assert(unordered_has_values_k<container_t>, "The atomic suites are written for maps");

    std::size_t const total = unordered_threads_count_k * per_thread;
    auto allocated = container_t::make(total * 2);
    st_verify_((allocated) && "the growable table must build");
    container_t growable = *std::move(allocated);
    for (std::size_t identifier = 0; identifier < total; ++identifier)
        unordered_emplace(growable, identifier, assume_reserved_t {});
    st_verify_eq_(growable.size(), total);

    auto container = unordered_pinned_t<container_t>::adopt(std::move(growable).release());
    st_verify_((container.capacity() >= total) && "the pinned table must hold every key it was filled with");

    // Every key already exists, so every update must land.
    std::atomic<std::size_t> updates_landed {0};
    std::vector<std::thread> threads;
    threads.reserve(unordered_threads_count_k);
    for (std::size_t thread_index = 0; thread_index < unordered_threads_count_k; ++thread_index)
        threads.emplace_back([&, thread_index]() noexcept {
            std::size_t landed = 0;
            for (std::size_t offset = 0; offset < per_thread; ++offset) {
                std::size_t const identifier = thread_index * per_thread + offset;
                landed += container.update(unordered_key_from<key_t>(identifier),
                                           unordered_value_from<mapped_t>(identifier + total));
            }
            updates_landed += landed;
        });
    for (auto &thread : threads) thread.join();

    st_verify_eq_(updates_landed.load(), total);
    st_verify_eq_(container.size(), total);
    for (std::size_t identifier = 0; identifier < total; ++identifier) {
        bool const reached = container.find(unordered_key_from<key_t>(identifier), [&](auto const &slot) noexcept {
            st_verify_((slot.value() == unordered_value_from<mapped_t>(identifier + total)) &&
                       "every update must be visible once the writers have joined");
        });
        st_verify_((reached) && "an updated key must still be present");
    }

    // Half of each thread's range is retired, and a missing key must never report an erasure.
    std::atomic<std::size_t> erasures_landed {0};
    std::atomic<std::size_t> phantom_erasures {0};
    threads.clear();
    for (std::size_t thread_index = 0; thread_index < unordered_threads_count_k; ++thread_index)
        threads.emplace_back([&, thread_index]() noexcept {
            std::size_t landed = 0, phantoms = 0;
            for (std::size_t offset = 0; offset < per_thread; offset += 2) {
                std::size_t const identifier = thread_index * per_thread + offset;
                landed += container.erase(unordered_key_from<key_t>(identifier));
                phantoms += container.erase(unordered_key_from<key_t>(identifier + total));
            }
            erasures_landed += landed;
            phantom_erasures += phantoms;
        });
    for (auto &thread : threads) thread.join();

    st_verify_eq_(erasures_landed.load(), total / 2);
    st_verify_eq_(phantom_erasures.load(), 0u);
    st_verify_eq_(container.size(), total - total / 2);
    st_verify_eq_(container.deleted_count(), total / 2);

    // Handing the allocation back gives the iterable surface back, so the survivors are checked the
    // same way every other suite checks them.
    container_t compacted = container_t::adopt(std::move(container).release());
    st_verify_eq_(compacted.size(), total - total / 2);
    for (std::size_t identifier = 0; identifier < total; ++identifier) {
        bool const survived = (identifier % per_thread) % 2 == 1;
        if (survived) unordered_verify_present(compacted, identifier, identifier + total);
        else unordered_verify_absent(compacted, identifier);
    }
}

#pragma endregion Concurrency

} // namespace ashvardanian::smashtable::scripts
