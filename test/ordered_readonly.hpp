/**
 *  @file test/ordered_readonly.hpp
 *  @author Ash Vardanian
 *  @date September 17, 2026
 *  @brief The suite every build-once ordered container answers: its type tags, its ranks against
 *      @c std::lower_bound, and the values a map form keeps beside its keys.
 *
 *  A read-only ordered container never mutates after @c make, so nothing here erases or inserts.
 *  What it does promise is that a rank, a position and a walk all name the same element, and that a
 *  map form agrees with its set twin key for key.
 */
#pragma once
#include <cstddef> // `std::size_t`

#include <algorithm> // `std::lower_bound`
#include <span>      // `std::span`

#include "harness.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Ordered Readonly Test Templates

/** The standard bound over a sorted span, which is what every rank here must agree with. */
template <typename key_type_>
[[nodiscard]] std::size_t lower_bound_of(std::span<key_type_ const> sorted, key_type_ wanted) {
    return static_cast<std::size_t>(std::lower_bound(sorted.begin(), sorted.end(), wanted) - sorted.begin());
}

/** Tests that a read-only ordered container names its element, its key and its shape. */
template <typename container_type_>
void test_ordered_readonly_tags() {

    using container_t = container_type_;
    static_assert(tagged_collection<container_t>, "every container answers what it stores and whether it maps");
    static_assert(std::is_same_v<typename container_t::value_type, typename container_t::value_t>,
                  "the STL tag and the library alias name one type");
    static_assert(std::is_same_v<typename container_t::key_type, typename container_t::key_t>,
                  "the STL tag and the library alias name one key");
    static_assert(container_t::is_associative::value == is_mapping<typename container_t::value_t>,
                  "the shape tag follows the element, never the other way around");
}

/**
 *  @brief Tests that @p layout agrees with @p sorted on every rank, position and walk.
 *  @param[in] wanted The keys to look up, which need not be present.
 */
template <typename container_type_>
void verify_ordered_readonly(container_type_ const &layout, std::span<typename container_type_::key_t const> sorted,
                             std::span<typename container_type_::key_t const> wanted) {

    using key_t = typename container_type_::key_t;
    st_verify_eq_(layout.size(), sorted.size());
    st_verify_eq_(layout.size_bytes(), container_type_::size_bytes(sorted.size()));
    st_verify_eq_(layout.size_bytes() % container_type_::format_t::bytes_per_row_k, 0u);

    for (key_t const key : wanted) {
        std::size_t const expected_rank = lower_bound_of(sorted, key);
        st_verify_eq_(layout.rank(key), expected_rank);
        expected<std::size_t> const found = layout.find(key);
        if (expected_rank < sorted.size() && sorted[expected_rank] == key) st_verify_eq_(found, expected_rank);
        else st_verify_eq_(found.status(), status_t::key_not_found_k);
        // A range walk from the bound must continue in sorted order.
        typename container_type_::iterator walk = layout.lower_bound(key);
        for (std::size_t step = 0; step < 6 && expected_rank + step < sorted.size(); ++step, ++walk) {
            st_verify_eq_(walk.rank(), expected_rank + step);
            st_verify_(*walk == sorted[expected_rank + step]);
        }
    }

    std::size_t const stride = sorted.size() / 3000 + 1;
    for (std::size_t ordinal = 0; ordinal < sorted.size(); ordinal += stride) {
        st_verify_(layout.select(ordinal) == sorted[ordinal]);
        st_verify_(*layout.at_rank(ordinal) == sorted[ordinal]);
    }

    std::size_t walked = 0;
    for (typename container_type_::iterator walk = layout.begin(); walk != layout.end(); ++walk, ++walked)
        st_verify_(*walk == sorted[walked]);
    st_verify_eq_(walked, sorted.size());
    st_verify_(layout.at_rank(sorted.size()) == layout.end());
}

/**
 *  @brief Tests that a map form holds the keys its set twin holds, and the value each key arrived with.
 *
 *  Both forms take the same keys in the same order, so a repeated key reads back the first value
 *  written for it - which is what makes the rank-ordered array of values load-bearing rather than
 *  incidental.
 */
template <typename map_type_, typename set_type_>
void test_ordered_readonly_mapping(std::size_t size = 300) {

    using key_t = typename map_type_::key_t;
    using mapped_t = typename map_type_::mapped_t;
    using pair_t = typename map_type_::value_t;
    static_assert(map_type_::is_associative::value, "the map form stores a mapping");
    static_assert(!set_type_::is_associative::value, "the set form stores bare keys");

    basic_vector<pair_t> pairs;
    basic_vector<key_t> keys;
    st_verify_(pairs.reserve(size));
    st_verify_(keys.reserve(size));
    for (std::size_t index = 0; index < size; ++index) {
        key_t const key = static_cast<key_t>(index / 2); // ? Every key repeats, so first-wins is observable
        st_verify_(pairs.push_back(pair_t {key, static_cast<mapped_t>(index)}));
        st_verify_(keys.push_back(key_t {key}));
    }

    expected<map_type_> made_map = map_type_::make({pairs.data(), pairs.size()});
    expected<set_type_> made_set = set_type_::make({keys.data(), keys.size()});
    st_verify_(made_map);
    st_verify_(made_set);
    st_verify_eq_(made_map->size(), made_set->size());
    st_verify_gt_(made_map->size_bytes(), made_set->size_bytes());

    for (std::size_t rank = 0; rank < made_map->size(); ++rank) {
        st_verify_(made_map->select(rank) == made_set->select(rank));
        st_verify_eq_(made_map->mapped_at(rank), static_cast<mapped_t>(rank));
    }
    for (std::size_t index = 0; index < size / 2; ++index) {
        key_t const key = static_cast<key_t>(index);
        expected<std::size_t> const in_map = made_map->find(key);
        expected<std::size_t> const in_set = made_set->find(key);
        st_verify_(in_map);
        st_verify_(in_set);
        st_verify_eq_(*in_map, *in_set);
        st_verify_eq_(made_map->mapped_at(*in_map), static_cast<mapped_t>(index * 2));
    }
}

#pragma endregion Ordered Readonly Test Templates

} // namespace ashvardanian::smashtable::test
