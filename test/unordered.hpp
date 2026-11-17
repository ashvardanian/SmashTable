/**
 *  @file test/unordered.hpp
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Template test functions for unordered containers, exercising the hash-specific surface -
 *      growth through rehashes, tombstone reuse, iteration, and the per-slot atomic operations.
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
 *  ever touch the same key. A worker tallies what it saw into an atomic and asserts nothing; every
 *  check runs after the join, leaving the outcome independent of the interleaving.
 */
#pragma once
#include <cstddef> // `std::size_t`

#include <atomic>        // `std::atomic`
#include <limits>        // `std::numeric_limits`
#include <new>           // `std::nothrow`
#include <set>           // `std::set`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view`
#include <thread>        // `std::thread`
#include <type_traits>   // `std::is_void`
#include <unordered_map> // `std::unordered_map`
#include <vector>        // `std::vector`

#include <smashtable/basic_hash_table.hpp>
#include <smashtable/atomic_hash_table.hpp>

#include "fixtures.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Fixture Helpers

/** True when the container stores values alongside keys, false for sets. */
template <typename container_type_>
inline constexpr bool unordered_has_values = !std::is_void_v<typename container_type_::mapped_type>;

/** Builds a key of the container's key type from an integer identifier. */
template <typename key_type_>
key_type_ unordered_key_from(std::size_t identifier) noexcept {
    if constexpr (std::is_same_v<key_type_, std::string>) return std::string("key-") + std::to_string(identifier);
    // Four keys share each high word and every low word repeats, so a half-compared key collides.
    else if constexpr (std::is_same_v<key_type_, key128_t>) return key128_t {identifier >> 2, identifier & 3};
    else return key_type_(identifier);
}

/** Builds a mapped value of the container's mapped type from an integer identifier. */
template <typename mapped_type_>
mapped_type_ unordered_value_from(std::size_t identifier) noexcept {
    if constexpr (std::is_same_v<mapped_type_, std::string>) return std::string("value-") + std::to_string(identifier);
    else return mapped_type_(identifier);
}

/** Inserts the element for @p identifier, choosing the set or map arity of @c emplace. */
template <typename container_type_, typename... tags_types_>
void unordered_emplace(container_type_ &container, std::size_t identifier, tags_types_... tags) noexcept {
    using key_t = typename container_type_::key_type;
    if constexpr (unordered_has_values<container_type_>) {
        using mapped_t = typename container_type_::mapped_type;
        container.emplace(unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(identifier), tags...);
    }
    else { container.emplace(unordered_key_from<key_t>(identifier), tags...); }
}

/** Inserts the key of @p identifier carrying the value of @p value_identifier, for maps only. */
template <typename container_type_, typename... tags_types_>
void unordered_emplace_valued(container_type_ &container, std::size_t identifier, std::size_t value_identifier,
                              tags_types_... tags) noexcept {
    using key_t = typename container_type_::key_type;
    using mapped_t = typename container_type_::mapped_type;
    static_assert(unordered_has_values<container_type_>, "Only maps carry a separate value");
    container.emplace(unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(value_identifier), tags...);
}

/** Files the element of @p identifier carrying @p value_identifier through @c upsert. */
template <typename container_type_>
[[nodiscard]] status_t unordered_upsert(container_type_ &container, std::size_t identifier,
                                        [[maybe_unused]] std::size_t value_identifier) noexcept {
    using key_t = typename container_type_::key_type;
    using owned_t = typename container_type_::owned_value_type;
    if constexpr (unordered_has_values<container_type_>) {
        using mapped_t = typename container_type_::mapped_type;
        return container.upsert(
            owned_t {unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(value_identifier)});
    }
    else return container.upsert(owned_t {unordered_key_from<key_t>(identifier)});
}

/** Checks that @p identifier is present and, on maps, that it still carries @p value_identifier. */
template <typename container_type_>
void check_unordered_present(container_type_ &container, std::size_t identifier,
                             std::size_t value_identifier) noexcept {
    using key_t = typename container_type_::key_type;

    auto const key = unordered_key_from<key_t>(identifier);
    expect_equal(container.contains(key), true, "a key that was inserted must be found");
    expect_equal(container.count(key), 1u);
    expect_true(container.find(key) != container.end(), "find must reach the element");
    if constexpr (unordered_has_values<container_type_>) {
        using mapped_t = typename container_type_::mapped_type;
        expect_equal(container.at(key), unordered_value_from<mapped_t>(value_identifier),
                     "the mapped value must survive every rehash");
    }
}

/** Checks that @p identifier is absent, both through @c contains and through @c find. */
template <typename container_type_>
void check_unordered_absent(container_type_ &container, std::size_t identifier) noexcept {
    using key_t = typename container_type_::key_type;

    auto const key = unordered_key_from<key_t>(identifier);
    expect_equal(container.contains(key), false);
    expect_equal(container.count(key), 0u);
    expect_equal(container.find(key), container.end(), "find must report the end for a missing key");
}

/**
 *  @brief Collects the keys reachable by iteration, and how many visits it took to reach them.
 *  @note The set alone cannot catch a duplicate visit - it swallows one - so the count comes back
 *      with it, and a caller compares that against @c size() to pin one visit per element.
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
    expect_equal(container.for_each([&](auto const &address) noexcept {
        collected.insert(address.key());
        ++visits;
    }),
                 success_k);
    return {std::move(collected), visits};
}

/** The keys the container is expected to hold, built from the same identifiers. */
template <typename container_type_>
std::set<typename container_type_::key_type> unordered_expected_keys(
    std::unordered_map<std::size_t, std::size_t> const &oracle) noexcept {
    using key_t = typename container_type_::key_type;
    std::set<key_t> expected;
    for (auto const &entry : oracle) expected.insert(unordered_key_from<key_t>(entry.first));
    return expected;
}

/** Re-checks the whole container against the oracle, in both directions. */
template <typename container_type_>
void check_unordered_against_oracle(container_type_ &container,
                                    std::unordered_map<std::size_t, std::size_t> const &oracle,
                                    std::size_t identifiers_bound) noexcept {
    expect_equal(container.size(), oracle.size());
    expect_equal(container.empty(), oracle.empty());
    for (std::size_t identifier = 0; identifier < identifiers_bound; ++identifier) {
        auto const found = oracle.find(identifier);
        if (found != oracle.end()) check_unordered_present(container, identifier, found->second);
        else check_unordered_absent(container, identifier);
    }
    auto const [walked_keys, walked_visits] = unordered_keys_by_iteration(container);
    expect_equal(walked_visits, container.size());
    expect_equal(walked_keys, unordered_expected_keys<container_type_>(oracle),
                 "iteration must reach exactly the live keys");
}

#pragma endregion Fixture Helpers

#pragma region Basic Unordered Operations

/** Tests that every read and every teardown path tolerates an empty container. */
template <typename container_type_>
void check_unordered_empty_container_operations() {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    expect_true(container.empty());
    expect_equal(container.size(), 0u);
    expect_equal(container.bucket_count(), 0u);
    check_unordered_absent(container, 1);
    expect_equal(container.begin(), container.end(), "an empty container iterates over nothing");
    expect_equal(unordered_keys_by_for_each(container).first.size(), 0u);
    expect_true(!(container.erase(unordered_key_from<key_t>(7))));

    // The same operations must also survive on a container that owns memory but holds nothing.
    auto allocated = container_t::make(128);
    if (!expect_true(allocated.has_value(), "allocating a small table must succeed")) return;
    container_t reserved = *std::move(allocated);
    expect_true(reserved.empty());
    expect_equal(reserved.size(), 0u);
    expect_true(reserved.bucket_count() > 0u);
    check_unordered_absent(reserved, 1);
    expect_equal(reserved.begin(), reserved.end(), "a reserved but empty container iterates over nothing");
    expect_equal(unordered_keys_by_for_each(reserved).first.size(), 0u);

    // Teardown on an empty container must be repeatable.
    reserved.clear();
    reserved.clear();
    expect_equal(reserved.size(), 0u);
    reserved.shrink_to_fit();
    expect_equal(reserved.bucket_count(), 0u);
    expect_true(reserved.empty());
}

/** Tests the insert, lookup, overwrite and erase cycle on a single element. */
template <typename container_type_>
void check_unordered_single_element_operations() {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    unordered_emplace(container, 42);
    expect_equal(container.size(), 1u);
    expect_true(!(container.empty()));
    check_unordered_present(container, 42, 42);
    check_unordered_absent(container, 43);

    auto const [collected, collected_visits] = unordered_keys_by_iteration(container);
    expect_equal(collected_visits, container.size());
    expect_equal(collected.size(), 1u);
    expect_equal(*collected.begin(), unordered_key_from<key_t>(42), "the only key must be the inserted one");

    // Re-inserting the same key overwrites rather than duplicates.
    if constexpr (unordered_has_values<container_t>) {
        unordered_emplace_valued(container, 42, 99);
        expect_equal(container.size(), 1u);
        check_unordered_present(container, 42, 99);
    }
    else {
        unordered_emplace(container, 42);
        expect_equal(container.size(), 1u);
    }

    expect_true(container.erase(unordered_key_from<key_t>(42)));
    expect_equal(container.size(), 0u);
    expect_true(container.empty());
    check_unordered_absent(container, 42);
    expect_true(!(container.erase(unordered_key_from<key_t>(42))));

    // A tombstoned slot must accept the same key again.
    unordered_emplace(container, 42);
    expect_equal(container.size(), 1u);
    check_unordered_present(container, 42, 42);
}

/** Tests that growth from an unallocated table preserves every element across many rehashes. */
template <typename container_type_>
void check_unordered_growth_through_rehashes(std::size_t size = 4000) {

    using container_t = container_type_;

    container_t container;
    std::unordered_map<std::size_t, std::size_t> oracle;
    std::size_t rehashes = 0;
    auto previous_buckets = container.bucket_count();

    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
        expect_equal(container.size(), oracle.size());
        if (container.bucket_count() != previous_buckets) {
            previous_buckets = container.bucket_count();
            ++rehashes;
            // Everything inserted so far must have survived the reallocation.
            for (std::size_t earlier = 0; earlier <= identifier; ++earlier)
                check_unordered_present(container, earlier, earlier);
        }
    }

    expect_true(rehashes > 3, "growing to thousands of elements must rehash repeatedly");
    check_unordered_against_oracle(container, oracle, size + 100);
}

/** Tests that a tombstoned table still finds its survivors, then refills the tombstones. */
template <typename container_type_>
void check_unordered_tombstone_reuse(std::size_t size = 2000) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    std::unordered_map<std::size_t, std::size_t> oracle;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }
    check_unordered_against_oracle(container, oracle, size);

    // Erase three quarters of the keys, leaving long runs of tombstones behind. A probe sequence
    // that treats a tombstone as the end of the run loses the survivors right here.
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        if (identifier % 4 == 0) continue;
        expect_true(container.erase(unordered_key_from<key_t>(identifier)));
        oracle.erase(identifier);
    }
    check_unordered_against_oracle(container, oracle, size);

    // Refill the tombstones with the very keys that made them.
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        if (identifier % 4 == 0) continue;
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }
    check_unordered_against_oracle(container, oracle, size);

    // And with keys that were never present, so the tombstones take unfamiliar hashes.
    for (std::size_t identifier = size; identifier < size + 500; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }
    check_unordered_against_oracle(container, oracle, size + 500);
}

/** Tests that iteration and @c for_each each visit every live element exactly once. */
template <typename container_type_>
void check_unordered_full_iteration(std::size_t size = 1500) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    std::unordered_map<std::size_t, std::size_t> oracle;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }

    auto const expected = unordered_expected_keys<container_t>(oracle);
    expect_equal(expected.size(), size);

    // A count alone would pass while a slot is visited twice and another skipped, so compare sets.
    std::size_t visits = 0;
    std::set<key_t> visited;
    expect_equal(container.for_each([&](auto const &address) noexcept {
        visited.insert(address.key());
        ++visits;
    }),
                 success_k);
    expect_equal(visits, size);
    expect_equal(visited, expected, "for_each must visit every live key exactly once");
    auto const [iterated, iterated_visits] = unordered_keys_by_iteration(container);
    expect_equal(iterated_visits, container.size());
    expect_equal(iterated, expected, "iteration must agree with for_each");

    // The const overloads walk the same slots through the const element reference.
    container_t const &const_container = container;
    std::size_t const_visits = 0;
    std::set<key_t> const_visited;
    expect_equal(const_container.for_each([&](auto const &address) noexcept {
        const_visited.insert(address.key());
        ++const_visits;
    }),
                 success_k);
    expect_equal(const_visits, size);
    expect_equal(const_visited, expected, "the const for_each must see the same keys");

    // Half the elements go away, and both walks must follow.
    for (std::size_t identifier = 0; identifier < size; identifier += 2) {
        expect_true(container.erase(unordered_key_from<key_t>(identifier)));
        oracle.erase(identifier);
    }
    auto const survivors = unordered_expected_keys<container_t>(oracle);
    auto const [seen_by_for_each, for_each_visits] = unordered_keys_by_for_each(container);
    auto const [seen_by_walk, walk_visits] = unordered_keys_by_iteration(container);
    expect_equal(for_each_visits, container.size());
    expect_equal(walk_visits, container.size());
    expect_equal(seen_by_for_each, survivors, "for_each must skip tombstones");
    expect_equal(seen_by_walk, survivors, "iteration must skip tombstones");
}

/** Tests move construction, move assignment and @c swap against the oracle. */
template <typename container_type_>
void check_unordered_moves_and_swaps(std::size_t size = 600) {

    using container_t = container_type_;

    container_t source;
    std::unordered_map<std::size_t, std::size_t> oracle;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(source, identifier);
        oracle[identifier] = identifier;
    }

    container_t moved_into(std::move(source));
    check_unordered_against_oracle(moved_into, oracle, size);

    container_t assigned_into;
    assigned_into = std::move(moved_into);
    check_unordered_against_oracle(assigned_into, oracle, size);

    // Swapping with an empty container hands the whole payload over and takes nothing back.
    container_t empty_partner;
    assigned_into.swap(empty_partner);
    expect_true(assigned_into.empty());
    expect_equal(assigned_into.size(), 0u);
    check_unordered_against_oracle(empty_partner, oracle, size);

    // Swapping two populated containers exchanges them wholesale.
    container_t other;
    std::unordered_map<std::size_t, std::size_t> other_oracle;
    for (std::size_t identifier = size; identifier < size + 200; ++identifier) {
        unordered_emplace(other, identifier);
        other_oracle[identifier] = identifier;
    }
    empty_partner.swap(other);
    check_unordered_against_oracle(empty_partner, other_oracle, size + 200);
    check_unordered_against_oracle(other, oracle, size + 200);
}

/** Tests @c reserve idempotence, explicit @c rehash, @c clear and @c shrink_to_fit. */
template <typename container_type_>
void check_unordered_capacity_management(std::size_t size = 800) {

    using container_t = container_type_;

    using reserve_result_t = typename container_t::reserve_result_t;

    container_t container;
    expect_equal(container.reserve(size), reserve_result_t::reallocated_k,
                 "the first reserve on an empty table must allocate");
    expect_equal(container.reserve(size), reserve_result_t::unchanged_k,
                 "a second reserve for the same count must be a no-op");
    expect_equal(container.reserve(size / 2), reserve_result_t::unchanged_k,
                 "shrinking through reserve must be a no-op");
    auto const reserved_buckets = container.bucket_count();
    expect_true(reserved_buckets > 0u);

    std::unordered_map<std::size_t, std::size_t> oracle;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }
    expect_equal(container.bucket_count(), reserved_buckets);
    check_unordered_against_oracle(container, oracle, size);

    // An explicit rehash moves every element into a differently sized table. The bucket count is
    // scaled by a power of two, because the probe mask assumes a power-of-two slot count and
    // @c rehash forwards its argument without rounding.
    expect_equal(container.rehash(reserved_buckets * 4), reserve_result_t::reallocated_k,
                 "a rehash onto a different size moves every element, so it must report reallocating");
    expect_equal(container.bucket_count(), reserved_buckets * 4);
    check_unordered_against_oracle(container, oracle, size);

    // Shrinking gives the four-fold growth back and lands on the count the reserve chose, so a
    // @c shrink_to_fit that did nothing at all is a failure rather than a pass.
    container.shrink_to_fit();
    expect_equal(container.bucket_count(), reserved_buckets);
    check_unordered_against_oracle(container, oracle, size);

    container.clear();
    expect_true(container.empty());
    expect_equal(container.size(), 0u);
    expect_true(container.bucket_count() > 0u, "clear keeps the memory");
    check_unordered_absent(container, 0);

    container.shrink_to_fit();
    expect_equal(container.bucket_count(), 0u);

    // A cleared and shrunk container is still usable.
    unordered_emplace(container, 5);
    expect_equal(container.size(), 1u);
    check_unordered_present(container, 5, 5);
}

/** Tests that the load factor stays under the documented cap and the counters stay honest. */
template <typename container_type_>
void check_unordered_load_factor_consistency(std::size_t size = 3000) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;

    container_t container;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        expect_equal(container.size(), identifier + 1);
        expect_true(!(container.empty()));

        // The header lives in whole buckets, and the table never runs past three quarters full.
        expect_equal(container.bucket_count() * hash_bucket_capacity_k, container.slots_count().raw);
        expect_true(container.size() * 4 <= container.slots_count().raw * 3,
                    "the load factor must stay under the 75% cap");
        expect_true(container.size() <= container.capacity());
    }

    // Erasures lower the size without lowering the slot count.
    auto const slots_before_erasures = container.slots_count().raw;
    for (std::size_t identifier = 0; identifier < size; identifier += 3)
        expect_true(container.erase(unordered_key_from<key_t>(identifier)));
    expect_equal(container.size(), size - ((size + 2) / 3));
    expect_equal(container.slots_count().raw, slots_before_erasures);
    expect_true(!(container.empty()));

    container.clear();
    expect_true(container.empty());
    expect_equal(container.size(), 0u);
    expect_equal(container.slots_count().raw, slots_before_erasures);
}

#pragma endregion Basic Unordered Operations

#pragma region Heterogeneous Lookup

/** Tests lookup by @c std::string_view against a container of @c std::string keys. Only meaningful
 *  where the equality predicate is transparent and the hash of a view matches the hash of the
 *  string it views, which the standard guarantees. */
template <typename container_type_>
void check_unordered_heterogeneous_string_view_lookup(std::size_t size = 400) {

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
        expect_equal(container.contains(borrowed), true, "a view of a present key must be found");
        expect_equal(container.count(borrowed), 1u);
        expect_true(container.find(borrowed) != container.end(), "find must accept a view");
        if constexpr (unordered_has_values<container_t>) {
            using mapped_t = typename container_t::mapped_type;
            expect_equal(container.at(borrowed), unordered_value_from<mapped_t>(identifier),
                         "a view must reach the same value the owned key does");
        }
    }

    using namespace std::literals::string_view_literals;
    expect_equal(container.contains("key-missing"sv), false);
    expect_equal(container.count("key-missing"sv), 0u);

    // Erasing through a view must retire the same slot the owned key would have.
    std::string const first = unordered_key_from<key_t>(0);
    expect_true(container.erase(std::string_view {first}));
    expect_equal(container.contains(first), false);
    expect_equal(container.size(), size - 1);
}

#pragma endregion Heterogeneous Lookup

#pragma region Saturation and Refusal

/**
 *  @brief Tests that a table whose allocator stops supplying memory stops storing, rather than
 *      filling to the last slot and then probing a table with no free slot forever.
 *
 *  @note The hang this pins only exists where @c assert is compiled out, so the suite must be run
 *      from a release build to be worth anything.
 */
template <typename container_type_>
void check_unordered_exhausted_allocator_insertions(std::size_t attempts = 4000) {

    using container_t = container_type_;

    // Exactly one allocation is granted, so the table gets its first buffer and can never grow again.
    allocation_ledger_t ledger;
    ledger.allow(1);
    {
        auto allocated = container_t::make(std::size_t {64}, {}, {}, stateful_allocator<std::byte> {ledger});
        if (!expect_true(allocated.has_value(), "the one permitted allocation must succeed")) return;
        container_t container = *std::move(allocated);
        std::size_t const slots = container.slots_count().raw;
        expect_true(slots > 0u);

        std::unordered_map<std::size_t, std::size_t> oracle;
        for (std::size_t identifier = 0; identifier < attempts; ++identifier) {
            std::size_t const size_before = container.size();
            unordered_emplace(container, identifier);
            if (container.size() != size_before) oracle[identifier] = identifier;
            // Neither the counters nor the layout may run past the single buffer that was handed out.
            expect_true(container.size() + container.deleted_count() <= slots);
            expect_equal(container.slots_count().raw, slots);
        }

        // The load factor caps at 75%, so the one buffer accepts three of every four slots and no more.
        expect_equal(container.size(), slots * 3 / 4, "a table that cannot grow stops at its load cap");
        expect_equal(container.size(), oracle.size());
        for (auto const &entry : oracle) check_unordered_present(container, entry.first, entry.second);
    }
    ledger.check_balanced();
}

/** Inserts the element for @p identifier through the reporting overload of @c emplace. */
template <typename container_type_, typename... tags_types_>
auto unordered_emplace_reporting(container_type_ &container, std::size_t identifier, tags_types_... tags) noexcept {
    using key_t = typename container_type_::key_type;
    using report_t = typename container_type_::emplace_report_t;
    if constexpr (unordered_has_values<container_type_>) {
        using mapped_t = typename container_type_::mapped_type;
        return container.template emplace<report_t::insert_result_k>(
            unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(identifier), tags...);
    }
    else return container.template emplace<report_t::insert_result_k>(unordered_key_from<key_t>(identifier), tags...);
}

/** Tests that a reporting insertion separates a fresh key from one already taken. Two of the three
 *  outcomes; the refusal is a table that cannot grow, tested separately. */
template <typename container_type_>
void check_unordered_insert_reports_outcome() {

    using container_t = container_type_;
    using upsert_result_t = typename container_t::upsert_result_t;

    container_t container;
    auto const stored = unordered_emplace_reporting(container, 1);
    expect_equal(stored.outcome, upsert_result_t::inserted_k, "a fresh key must report an insertion");
    expect_true(stored.position != container.end(), "an insertion must name the slot it landed in");
    expect_true(!(stored.failed()));

    auto const duplicate = unordered_emplace_reporting(container, 1);
    expect_equal(duplicate.outcome, upsert_result_t::updated_k, "a key already there must report an update");
    expect_true(duplicate.position != container.end(), "an update must name the slot the key sits in");
    expect_true(!(duplicate.failed()), "a key already there is not a failure");
    expect_equal(container.size(), 1u);
}

/** Tests that an insertion the table had no room for is distinguishable from a duplicate. Both
 *  leave the size alone, and only the reported outcome tells them apart. */
template <typename container_type_>
void check_unordered_insert_reports_refusal() {

    using container_t = container_type_;
    using upsert_result_t = typename container_t::upsert_result_t;

    // One allocation is granted and then filled to the last slot, so the next key has nowhere to go
    // and the table has no way to make room.
    allocation_ledger_t ledger;
    ledger.allow(1);
    {
        auto allocated = container_t::make(std::size_t {64}, {}, {}, stateful_allocator<std::byte> {ledger});
        if (!expect_true(allocated.has_value(), "the one permitted allocation must succeed")) return;
        container_t container = *std::move(allocated);
        std::size_t const slots = container.slots_count().raw;
        for (std::size_t identifier = 0; identifier < slots; ++identifier)
            unordered_emplace(container, identifier, assume_reserved_t {}, assume_unique_t {});
        expect_equal(container.size(), slots);

        // Both calls promise the reservation, so neither asks the refusing allocator for room and the
        // only thing separating them is what the probe found.
        auto const duplicate = unordered_emplace_reporting(container, 0, assume_reserved_t {});
        expect_equal(duplicate.outcome, upsert_result_t::updated_k, "a saturated table still updates what it holds");
        expect_true(!(duplicate.failed()));

        auto const refused = unordered_emplace_reporting(container, slots + 1, assume_reserved_t {});
        expect_equal(refused.outcome, upsert_result_t::no_slot_k, "a table with no room must report the refusal");
        expect_true(refused.failed(), "a refusal is the only way an insertion fails");
        expect_equal(refused.position, container.end(), "a refusal names no slot");
        expect_equal(container.size(), slots);
    }
    ledger.check_balanced();
}

/** Tests that filing a key the table already holds never reaches the allocator. A table sitting
 *  exactly on its growth threshold has no headroom left, and asking for some before the probe has
 *  established the key is new turns a plain overwrite - which needs no slot of its own - into a
 *  refusal the caller cannot do anything about. */
template <typename container_type_>
void check_unordered_present_key_needs_no_room() {

    using container_t = container_type_;
    using upsert_result_t = typename container_t::upsert_result_t;

    // The table is sized so that its load factor leaves no headroom once it is filled below.
    allocation_ledger_t ledger;
    {
        auto allocated = container_t::make(std::size_t {24}, {}, {}, stateful_allocator<std::byte> {ledger});
        if (!expect_true(allocated.has_value(), "the first allocation must succeed")) return;
        container_t container = *std::move(allocated);
        std::size_t const threshold = container.capacity();
        expect_true(threshold > 0u);

        for (std::size_t identifier = 0; identifier < threshold; ++identifier) unordered_emplace(container, identifier);
        expect_equal(container.size(), threshold);
        std::size_t const slots_before = container.slots_count().raw;
        std::size_t const granted_before = ledger.allocations;

        // From here the allocator answers nothing, so anything asking it for a slot fails.
        ledger.refuse_everything();

        for (std::size_t identifier = 0; identifier < threshold; ++identifier) {
            status_t const overwritten = unordered_upsert(container, identifier, identifier + threshold);
            expect_equal(overwritten, success_k, "an overwrite needs no room and must not report a refusal");
        }
        expect_equal(ledger.allocations, granted_before);
        expect_equal(ledger.refusals, std::size_t {0});
        expect_equal(container.slots_count().raw, slots_before);
        expect_equal(container.size(), threshold);
        for (std::size_t identifier = 0; identifier < threshold; ++identifier)
            check_unordered_present(container, identifier, identifier + threshold);

        // The reporting insertion answers the same question, and an update is not a failure.
        auto const duplicate = unordered_emplace_reporting(container, 0);
        expect_equal(duplicate.outcome, upsert_result_t::updated_k, "a key already there is an update, not a refusal");
        expect_true(!(duplicate.failed()), "an overwrite the allocator never saw cannot have failed");
        expect_true(duplicate.position != container.end(), "an update must name the slot the key sits in");
        expect_equal(ledger.refusals, std::size_t {0});

        // A genuinely new key does need a slot, and that refusal is the one the heap is to blame for.
        status_t const refused = unordered_upsert(container, threshold, threshold);
        expect_equal(refused, out_of_memory_heap_k,
                     "a new key the table cannot make room for is an allocation failure");
        expect_equal(container.size(), threshold);
        expect_equal(container.slots_count().raw, slots_before);

        auto const refused_report = unordered_emplace_reporting(container, threshold + 1);
        expect_equal(refused_report.outcome, upsert_result_t::no_memory_k,
                     "a refused growth must not be reported as an exhausted probe");
        expect_true(refused_report.failed(), "nothing was stored, so the insertion failed");
        expect_equal(refused_report.position, container.end(), "a refusal names no slot");
        expect_equal(container.size(), threshold);
    }
    ledger.check_balanced();
}

/** Tests that a table with every slot taken refuses further keys, terminates while doing so, and
 *  does not count a store that never happened. */
template <typename container_type_>
void check_unordered_full_table_refusals(std::size_t extra_attempts = 64) {

    using container_t = container_type_;

    auto allocated = container_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the table must build")) return;
    container_t container = *std::move(allocated);
    std::size_t const slots = container.slots_count().raw;

    // Promising the reservation is what lets the table run past its own load factor, right up to
    // the last slot, which is the state both probe loops have to survive.
    for (std::size_t identifier = 0; identifier < slots; ++identifier)
        unordered_emplace(container, identifier, assume_reserved_t {}, assume_unique_t {});
    expect_equal(container.size(), slots);
    expect_equal(container.slots_count().raw, slots);

    // The unique-key probe has nowhere to store, and must say so by leaving the size alone.
    for (std::size_t identifier = slots; identifier < slots + extra_attempts; ++identifier) {
        unordered_emplace(container, identifier, assume_reserved_t {}, assume_unique_t {});
        expect_equal(container.size(), slots);
    }

    // The upsert probe has to terminate on the same table, and leave the size alone too.
    for (std::size_t identifier = slots; identifier < slots + extra_attempts; ++identifier) {
        unordered_emplace(container, identifier, assume_reserved_t {});
        expect_equal(container.size(), slots);
    }

    // Everything that did fit is still reachable, and an overwrite of a present key still lands.
    for (std::size_t identifier = 0; identifier < slots; ++identifier)
        check_unordered_present(container, identifier, identifier);

    // A walk that visits every match wraps the whole full table, and still meets each key once.
    for (std::size_t identifier = 0; identifier < slots; ++identifier) {
        std::size_t visits = 0;
        container.probe_to_visit(unordered_key_from<typename container_t::key_type>(identifier),
                                 [&](auto const &) noexcept {
                                     ++visits;
                                     return walk_control_t::resume_k;
                                 });
        expect_equal(visits, 1u);
    }
    if constexpr (unordered_has_values<container_t>) {
        unordered_emplace_valued(container, 0, 7, assume_reserved_t {});
        expect_equal(container.size(), slots);
        check_unordered_present(container, 0, 7);
    }
}

/** Tests that a rehash asking for nothing compacts the table instead of emptying it into a layout
 *  with no slots at all. */
template <typename container_type_>
void check_unordered_rehash_to_nothing(std::size_t size = 300) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;
    using reserve_result_t = typename container_t::reserve_result_t;

    container_t container;
    std::unordered_map<std::size_t, std::size_t> oracle;
    for (std::size_t identifier = 0; identifier < size; ++identifier) {
        unordered_emplace(container, identifier);
        oracle[identifier] = identifier;
    }

    // Tombstones make the smallest layout that still fits smaller than the current one.
    for (std::size_t identifier = 0; identifier < size; identifier += 2) {
        expect_true(container.erase(unordered_key_from<key_t>(identifier)));
        oracle.erase(identifier);
    }

    expect_true(container.rehash(0) != reserve_result_t::failed_k, "a rehash to nothing must still succeed");
    expect_true(container.bucket_count() > 0u, "the survivors need slots to live in");
    expect_equal(container.deleted_count(), 0u);
    check_unordered_against_oracle(container, oracle, size);

    // On a table holding nothing the same call may genuinely give the memory back.
    container.clear();
    expect_true((container.rehash(0) != reserve_result_t::failed_k));
    expect_true(container.empty());
    check_unordered_absent(container, 0);
}

/** Tests that an element count no power of two can cover fails instead of quietly yielding the
 *  smallest possible table. */
template <typename container_type_>
void check_unordered_unrepresentable_capacity() {

    using container_t = container_type_;
    using reserve_result_t = typename container_t::reserve_result_t;

    // The middle one is the nastiest: the load-factor multiplication wraps to exactly zero, which
    // the power-of-two rounding then reads as a request for the smallest table there is.
    constexpr std::size_t past_the_load_factor_k = (std::numeric_limits<std::size_t>::max)() / 2;
    constexpr std::size_t wrapping_to_nothing_k = std::size_t {1} << (std::numeric_limits<std::size_t>::digits - 2);
    constexpr std::size_t past_every_power_of_two_k = (std::numeric_limits<std::size_t>::max)();

    for (std::size_t elements : {past_the_load_factor_k, wrapping_to_nothing_k, past_every_power_of_two_k}) {
        auto allocated = container_t::make(elements);
        expect_true(!(allocated), "an unrepresentable element count must not report success");

        container_t container;
        expect_equal(container.reserve(elements), reserve_result_t::failed_k,
                     "an unrepresentable reservation must fail rather than allocate one bucket");
        expect_equal(container.bucket_count(), 0u);
        expect_true(container.empty());
    }
}

#pragma endregion Saturation and Refusal

#pragma region Concurrency

/** The thread count the concurrent suites use, fixed so a failure reproduces. */
inline constexpr std::size_t unordered_threads_count_k = 4;

/** The pinned table matching a growable one, since neither type names the other. */
template <typename container_type_>
struct unordered_pinned_of;

template <typename value_type_, typename hasher_type_, typename equals_type_, typename allocator_type_>
struct unordered_pinned_of<basic_hash_table<value_type_, hasher_type_, equals_type_, allocator_type_>> {
    using type = atomic_hash_table<value_type_, hasher_type_, equals_type_, allocator_type_>;
};

/** The pinned counterpart of @p container_type_, reached through @c release and @c adopt. */
template <typename container_type_>
using unordered_pinned = typename unordered_pinned_of<container_type_>::type;

/**
 *  @brief Tests that the pinned table answers a missing key with a status and a callback.
 *
 *  @c update and @c erase report @c key_not_found_k rather than a bare @c false, and @c find hands
 *  the miss to a second callback rather than returning one, the shape every store here uses.
 */
template <typename container_type_>
void check_unordered_pinned_reports_status() {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;
    using mapped_t = typename container_t::mapped_type;
    static_assert(unordered_has_values<container_t>, "The pinned suites are written for maps");

    auto allocated = container_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the growable table must build")) return;
    auto container = unordered_pinned<container_t>::adopt((*std::move(allocated)).release());

    expect_equal(container.emplace(unordered_key_from<key_t>(1), unordered_value_from<mapped_t>(1)), success_k,
                 "an empty pinned table must take the first key");

    expect_equal(container.update(unordered_key_from<key_t>(1), unordered_value_from<mapped_t>(2)), success_k,
                 "updating a key that is there must succeed");
    expect_equal(container.update(unordered_key_from<key_t>(9), unordered_value_from<mapped_t>(2)), key_not_found_k,
                 "updating a key that was never stored must name the reason");

    // The miss reaches a callback of its own, so a caller never infers absence from a return value.
    std::size_t found_calls = 0, missing_calls = 0;
    expect_equal(container.find(
                     unordered_key_from<key_t>(1), [&](auto const &) noexcept { ++found_calls; },
                     [&]() noexcept { ++missing_calls; }),
                 success_k);
    expect_equal(found_calls, 1u);
    expect_equal(missing_calls, 0u);
    expect_equal(container.find(
                     unordered_key_from<key_t>(9), [&](auto const &) noexcept { ++found_calls; },
                     [&]() noexcept { ++missing_calls; }),
                 success_k);
    expect_equal(found_calls, 1u);
    expect_equal(missing_calls, 1u);

    expect_equal(container.erase(unordered_key_from<key_t>(1)), success_k, "erasing a live key must succeed");
    expect_equal(container.erase(unordered_key_from<key_t>(1)), key_not_found_k,
                 "erasing a tombstone must name the reason");
    expect_equal(container.erase(unordered_key_from<key_t>(9)), key_not_found_k,
                 "erasing a key that was never stored must name the reason");
    expect_equal(container.size(), 0u);
}

/** Tests that a pinned table with no free slot reports the refusal instead of probing forever. */
template <typename container_type_>
void check_unordered_pinned_saturation() {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;
    using mapped_t = typename container_t::mapped_type;

    static_assert(unordered_has_values<container_t>, "The pinned suites are written for maps");

    auto allocated = container_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the growable table must build")) return;
    auto container = unordered_pinned<container_t>::adopt((*std::move(allocated)).release());
    std::size_t const slots = container.slots_count();
    expect_true(slots > 0u);

    // A pinned table has no load factor to respect, so every slot can be taken.
    for (std::size_t identifier = 0; identifier < slots; ++identifier) {
        status_t const stored =
            container.emplace(unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(identifier));
        expect_equal(stored, success_k, "every slot of an empty pinned table must be available");
    }
    expect_equal(container.size(), slots);

    // A new key has nowhere to go, and the probe must come back rather than wrap forever.
    for (std::size_t identifier = slots; identifier < slots + 8; ++identifier) {
        status_t const refused =
            container.emplace(unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(identifier));
        expect_equal(refused, capacity_exhausted_k, "a full pinned table must report the refusal");
        expect_true(refused != out_of_memory_heap_k, "no allocation was attempted, so none can have failed");
    }
    expect_equal(container.size(), slots);

    // Keys that are already there stay reachable and overwritable even with no free slot left.
    for (std::size_t identifier = 0; identifier < slots; ++identifier) {
        expect_equal(container.contains(unordered_key_from<key_t>(identifier)), true,
                     "a saturated table must still find what it holds");
        status_t const updated = container.emplace(unordered_key_from<key_t>(identifier),
                                                   unordered_value_from<mapped_t>(identifier + slots));
        expect_equal(updated, success_k, "an overwrite needs no free slot");
    }
    for (std::size_t identifier = 0; identifier < slots; ++identifier) {
        bool reached = false;
        expect_equal(container.find(unordered_key_from<key_t>(identifier),
                                    [&](auto const &slot) noexcept {
                                        reached = true;
                                        expect_equal(slot.value(), unordered_value_from<mapped_t>(identifier + slots),
                                                     "the overwrite must be the value that survives");
                                    }),
                     success_k);
        expect_true(reached, "an overwritten key must still be present");
    }
}

/** Tests that a pinned table left holding nothing but tombstones names a cause a caller can act on.
 *  The probe walks past a tombstone rather than reclaiming it, so such a table refuses every new
 *  key; reporting that as an allocation failure sends the caller to free memory that was never the
 *  problem, when the remedy is a rehash into fresh storage. */
template <typename container_type_>
void check_unordered_pinned_tombstone_saturation() {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;
    using mapped_t = typename container_t::mapped_type;
    static_assert(unordered_has_values<container_t>, "The pinned suites are written for maps");

    auto allocated = container_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the growable table must build")) return;
    auto container = unordered_pinned<container_t>::adopt((*std::move(allocated)).release());
    std::size_t const slots = container.slots_count();
    expect_true(slots > 0u);

    for (std::size_t identifier = 0; identifier < slots; ++identifier) {
        status_t const stored =
            container.emplace(unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(identifier));
        expect_equal(stored, success_k, "every slot of an empty pinned table must be available");
    }
    for (std::size_t identifier = 0; identifier < slots; ++identifier) {
        status_t const erased = container.erase(unordered_key_from<key_t>(identifier));
        expect_equal(erased, success_k, "every key stored must be erasable");
    }
    expect_equal(container.size(), 0u);

    // Every slot is a tombstone now, so the table is empty and still has nowhere to put a key.
    for (std::size_t identifier = slots; identifier < slots + 4; ++identifier) {
        status_t const refused =
            container.emplace(unordered_key_from<key_t>(identifier), unordered_value_from<mapped_t>(identifier));
        expect_equal(refused, capacity_exhausted_k, "an exhausted probe must name the exhausted probe");
        expect_true(refused != out_of_memory_heap_k,
                    "a table of tombstones is not short of memory, and freeing some would not help");
    }
    expect_equal(container.size(), 0u);
}

/** Tests concurrent @c emplace followed by concurrent @c find and @c contains on a frozen table,
 *  with each thread owning a disjoint range of keys. */
template <typename container_type_>
void check_unordered_concurrent_emplace_and_find(std::size_t per_thread = 2000) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;
    using mapped_t = typename container_t::mapped_type;

    static_assert(unordered_has_values<container_t>, "The atomic suites are written for maps");

    std::size_t const total = unordered_threads_count_k * per_thread;
    auto allocated = container_t::make(total * 2);
    if (!expect_true(allocated.has_value(), "the growable table must build")) return;
    auto container = unordered_pinned<container_t>::adopt((*std::move(allocated)).release());
    expect_true(container.capacity() >= total, "the pinned table must hold what the writers will store");
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

    expect_equal(container.size(), total);
    expect_equal(container.slots_count(), slots_before);

    // The readers run over the same disjoint ranges, so a miss is a lost element, not a race.
    std::atomic<std::size_t> found_by_contains {0};
    std::atomic<std::size_t> found_by_callback {0};
    std::atomic<std::size_t> wrong_values {0};
    std::atomic<std::size_t> refused_reads {0};
    threads.clear();
    for (std::size_t thread_index = 0; thread_index < unordered_threads_count_k; ++thread_index)
        threads.emplace_back([&, thread_index]() noexcept {
            std::size_t contains_hits = 0, callback_hits = 0, mismatches = 0, refusals = 0;
            for (std::size_t offset = 0; offset < per_thread; ++offset) {
                std::size_t const identifier = thread_index * per_thread + offset;
                auto const key = unordered_key_from<key_t>(identifier);
                // A refusal is tallied rather than checked here, so the verdict waits for the join
                // rather than racing siblings still writing into the container.
                expected<bool> const seen = container.contains(key);
                if (!seen) ++refusals;
                else if (*seen) ++contains_hits;
                status_t const walked = container.find(key, [&](auto const &address) noexcept {
                    ++callback_hits;
                    if (!(address.value() == unordered_value_from<mapped_t>(identifier))) ++mismatches;
                });
                if (failed(walked)) ++refusals;
            }
            found_by_contains += contains_hits;
            found_by_callback += callback_hits;
            wrong_values += mismatches;
            refused_reads += refusals;
        });
    for (auto &thread : threads) thread.join();

    expect_equal(refused_reads.load(), 0u, "a frozen table must answer every read it is handed");
    expect_equal(found_by_contains.load(), total);
    expect_equal(found_by_callback.load(), total);
    expect_equal(wrong_values.load(), 0u);

    // The single-threaded view must agree with what the threads believe they wrote.
    std::size_t single_threaded_hits = 0;
    for (std::size_t identifier = 0; identifier < total; ++identifier) {
        expected<bool> const seen = container.contains(unordered_key_from<key_t>(identifier));
        if (!expect_equal(seen.status(), success_k)) return;
        if (*seen) ++single_threaded_hits;
    }
    expect_equal(single_threaded_hits, total);
}

/**
 *  @brief Tests concurrent @c update over pre-populated keys and concurrent @c erase afterwards,
 *      each thread again confined to its own range.
 *
 *  The table is filled while it is still growable, pinned for the concurrent phases, and handed
 *  back to a growable table to verify - the whole lifecycle @c release and @c adopt express.
 */
template <typename container_type_>
void check_unordered_concurrent_update_and_erase(std::size_t per_thread = 1000) {

    using container_t = container_type_;
    using key_t = typename container_t::key_type;
    using mapped_t = typename container_t::mapped_type;
    static_assert(unordered_has_values<container_t>, "The atomic suites are written for maps");

    std::size_t const total = unordered_threads_count_k * per_thread;
    auto allocated = container_t::make(total * 2);
    if (!expect_true(allocated.has_value(), "the growable table must build")) return;
    container_t growable = *std::move(allocated);
    for (std::size_t identifier = 0; identifier < total; ++identifier)
        unordered_emplace(growable, identifier, assume_reserved_t {});
    expect_equal(growable.size(), total);

    auto container = unordered_pinned<container_t>::adopt(std::move(growable).release());
    expect_true(container.capacity() >= total, "the pinned table must hold every key it was filled with");

    // Every key already exists, so every update must land.
    std::atomic<std::size_t> updates_landed {0};
    std::vector<std::thread> threads;
    threads.reserve(unordered_threads_count_k);
    for (std::size_t thread_index = 0; thread_index < unordered_threads_count_k; ++thread_index)
        threads.emplace_back([&, thread_index]() noexcept {
            std::size_t landed = 0;
            for (std::size_t offset = 0; offset < per_thread; ++offset) {
                std::size_t const identifier = thread_index * per_thread + offset;
                landed += succeeded(container.update(unordered_key_from<key_t>(identifier),
                                                     unordered_value_from<mapped_t>(identifier + total)));
            }
            updates_landed += landed;
        });
    for (auto &thread : threads) thread.join();

    expect_equal(updates_landed.load(), total);
    expect_equal(container.size(), total);
    for (std::size_t identifier = 0; identifier < total; ++identifier) {
        bool reached = false;
        expect_equal(container.find(unordered_key_from<key_t>(identifier),
                                    [&](auto const &slot) noexcept {
                                        reached = true;
                                        expect_equal(slot.value(), unordered_value_from<mapped_t>(identifier + total),
                                                     "every update must be visible once the writers have joined");
                                    }),
                     success_k);
        expect_true(reached, "an updated key must still be present");
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
                landed += succeeded(container.erase(unordered_key_from<key_t>(identifier)));
                phantoms += succeeded(container.erase(unordered_key_from<key_t>(identifier + total)));
            }
            erasures_landed += landed;
            phantom_erasures += phantoms;
        });
    for (auto &thread : threads) thread.join();

    expect_equal(erasures_landed.load(), total / 2);
    expect_equal(phantom_erasures.load(), 0u);
    expect_equal(container.size(), total - total / 2);
    expect_equal(container.deleted_count(), total / 2);

    // Handing the allocation back gives the iterable surface back, so the survivors are checked the
    // same way every other suite checks them.
    container_t compacted = container_t::adopt(std::move(container).release());
    expect_equal(compacted.size(), total - total / 2);
    for (std::size_t identifier = 0; identifier < total; ++identifier) {
        bool const survived = (identifier % per_thread) % 2 == 1;
        if (survived) check_unordered_present(compacted, identifier, identifier + total);
        else check_unordered_absent(compacted, identifier);
    }
}

#pragma endregion Concurrency

#pragma region Multi Match Probe Walks

/** A key filed once per generation, so several versions of one identifier coexist. */
struct versioned_key_t {
    std::size_t identifier = 0;
    std::size_t generation = 0;
};

/** A @c versioned_key_t without its generation, equal to every generation of it. */
struct unversioned_key_t {
    std::size_t identifier = 0;
};

/** Peels both key shapes to the identifier, so every version shares one home slot. */
struct versioned_hash_t {
    std::size_t operator()(versioned_key_t const &key) const noexcept { return key.identifier; }
    std::size_t operator()(unversioned_key_t const &key) const noexcept { return key.identifier; }
};

/** Separates generations of one identifier, unless the probe asks for the bare key. */
struct per_key_equals_t {
    using is_transparent = void;
    bool operator()(versioned_key_t const &first, versioned_key_t const &second) const noexcept {
        return first.identifier == second.identifier && first.generation == second.generation;
    }
    bool operator()(versioned_key_t const &first, unversioned_key_t const &second) const noexcept {
        return first.identifier == second.identifier;
    }
    bool operator()(unversioned_key_t const &first, versioned_key_t const &second) const noexcept {
        return first.identifier == second.identifier;
    }
};

using versioned_set_t = hash_set<versioned_key_t, versioned_hash_t, per_key_equals_t>;

/** Collects the generations @c probe_to_visit reaches for @p bare, in probe order. */
inline std::vector<std::size_t> unordered_visit_generations(versioned_set_t const &container,
                                                            std::size_t bare) noexcept {
    std::vector<std::size_t> generations;
    container.probe_to_visit(unversioned_key_t {bare}, [&](auto const &slot) noexcept {
        generations.push_back(slot.key().generation);
        return walk_control_t::resume_k;
    });
    return generations;
}

/** Tests that @c probe_to_visit reaches every version of one key sharing a probe run, and none of
 *  the foreign keys sharing it. */
inline void check_unordered_visit_every_match(std::size_t versions = 5) {

    auto allocated = versioned_set_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the table must build")) return;
    versioned_set_t container = *std::move(allocated);
    std::size_t const slots = container.slots_count().raw;

    // Two identifiers a slot count apart hash to the same home slot, so their versions interleave
    // inside a single probe run - exactly the layout that makes "the first match" the wrong answer.
    for (std::size_t generation = 0; generation < versions; ++generation) {
        container.emplace(versioned_key_t {1, generation}, assume_reserved_t {}, assume_unique_t {});
        container.emplace(versioned_key_t {1 + slots, generation}, assume_reserved_t {}, assume_unique_t {});
    }
    expect_equal(container.size(), versions * 2);

    std::vector<std::size_t> const wanted = unordered_visit_generations(container, 1);
    if (!expect_equal(wanted.size(), versions)) return;
    for (std::size_t generation = 0; generation < versions; ++generation) expect_equal(wanted[generation], generation);

    std::vector<std::size_t> const foreign = unordered_visit_generations(container, 1 + slots);
    expect_equal(foreign.size(), versions);

    // A key that never entered the run must cost zero visits, and so must one whose home slot is free.
    expect_equal(unordered_visit_generations(container, 2).size(), 0u);
    expect_equal(unordered_visit_generations(container, 1 + slots * 2).size(), 0u);
}

/** Tests that a @c halt_k reply stops the walk before the next slot is even read. */
inline void check_unordered_visit_early_exit(std::size_t versions = 5) {

    auto allocated = versioned_set_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the table must build")) return;
    versioned_set_t container = *std::move(allocated);
    for (std::size_t generation = 0; generation < versions; ++generation)
        container.emplace(versioned_key_t {1, generation}, assume_reserved_t {}, assume_unique_t {});

    for (std::size_t stop_after = 1; stop_after <= versions; ++stop_after) {
        std::size_t visits = 0;
        container.probe_to_visit(unversioned_key_t {1}, [&](auto const &) noexcept {
            ++visits;
            return visits == stop_after ? walk_control_t::halt_k : walk_control_t::resume_k;
        });
        expect_equal(visits, stop_after);
    }
}

/** Tests that a tombstone between two matches does not truncate the walk. */
inline void check_unordered_visit_across_tombstones(std::size_t versions = 6) {

    auto allocated = versioned_set_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the table must build")) return;
    versioned_set_t container = *std::move(allocated);
    for (std::size_t generation = 0; generation < versions; ++generation)
        container.emplace(versioned_key_t {1, generation}, assume_reserved_t {}, assume_unique_t {});

    // Erasing every other generation leaves tombstones interleaved with the survivors.
    for (std::size_t generation = 0; generation < versions; generation += 2)
        expect_true(container.erase(versioned_key_t {1, generation}));
    expect_equal(container.deleted_count(), (versions + 1) / 2);

    std::vector<std::size_t> const survivors = unordered_visit_generations(container, 1);
    if (!expect_equal(survivors.size(), versions / 2)) return;
    for (std::size_t index = 0; index < survivors.size(); ++index) expect_equal(survivors[index], index * 2 + 1);
}

/** Tests that an empty table, and one holding no allocation at all, cost zero visits. */
inline void check_unordered_visit_empty_table() {

    versioned_set_t unallocated;
    expect_equal(unordered_visit_generations(unallocated, 1).size(), 0u);

    auto allocated = versioned_set_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the table must build")) return;
    versioned_set_t container = *std::move(allocated);
    expect_equal(unordered_visit_generations(container, 1).size(), 0u);

    // A table emptied back out by erasures must behave like one that never held anything.
    container.emplace(versioned_key_t {1, 0}, assume_reserved_t {}, assume_unique_t {});
    expect_true(container.erase(versioned_key_t {1, 0}));
    expect_equal(unordered_visit_generations(container, 1).size(), 0u);
}

/** Tests that a run starting in the last slot wraps to the front instead of ending there. */
inline void check_unordered_visit_wraparound(std::size_t versions = 5) {

    auto allocated = versioned_set_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the table must build")) return;
    versioned_set_t container = *std::move(allocated);
    std::size_t const slots = container.slots_count().raw;

    std::size_t const last = slots - 1;
    for (std::size_t generation = 0; generation < versions; ++generation)
        container.emplace(versioned_key_t {last, generation}, assume_reserved_t {}, assume_unique_t {});

    std::vector<std::size_t> const generations = unordered_visit_generations(container, last);
    if (!expect_equal(generations.size(), versions)) return;
    for (std::size_t generation = 0; generation < versions; ++generation)
        expect_equal(generations[generation], generation);
}

/**
 *  @brief Tests that a table with no free slot left ends the walk instead of circling forever.
 *  @note The bound must hold in a build where assertions are gone, which is why the harness runs
 *      this suite under a timeout in a release configuration too.
 */
inline void check_unordered_visit_full_table() {

    auto allocated = versioned_set_t::make(std::size_t {64});
    if (!expect_true(allocated.has_value(), "the table must build")) return;
    versioned_set_t container = *std::move(allocated);
    std::size_t const slots = container.slots_count().raw;

    // Every slot taken by one identifier, so the run has neither a free slot nor a foreign key to end on.
    for (std::size_t generation = 0; generation < slots; ++generation)
        container.emplace(versioned_key_t {1, generation}, assume_reserved_t {}, assume_unique_t {});
    expect_equal(container.size(), slots);

    expect_equal(unordered_visit_generations(container, 1).size(), slots);

    // A key absent from a table with nowhere to stop still has to walk exactly once around.
    std::size_t visits = 0;
    container.probe_to_visit(unversioned_key_t {2}, [&](auto const &) noexcept {
        ++visits;
        return walk_control_t::resume_k;
    });
    expect_equal(visits, 0u);

    // Tombstones must not resurrect the loop either.
    for (std::size_t generation = 0; generation < slots; generation += 2)
        expect_true(container.erase(versioned_key_t {1, generation}));
    expect_equal(unordered_visit_generations(container, 1).size(), slots / 2);
}

#pragma endregion Multi Match Probe Walks

} // namespace ashvardanian::smashtable::test
