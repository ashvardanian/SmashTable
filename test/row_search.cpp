/**
 *  @file test/row_search.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Tests for the row kits and the static layouts: every kit this processor runs against the
 *      serial kit, and both layouts against a sorted array.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstring> // `std::memcmp`

#include <algorithm> // `std::lower_bound`, `std::sort`
#include <concepts>  // `std::same_as`
#include <limits>    // `std::numeric_limits`
#include <span>      // `std::span`

#include <smashtable/basic_vector.hpp>
#include <smashtable/immutable_b_tree.hpp>
#include <smashtable/immutable_splus_tree.hpp>
#include <smashtable/row_search.hpp>

#include "harness.hpp"
#include "ordered_readonly.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Key Generation

/** The distributions a row is drawn from, each aimed at a place a kit could go wrong. */
enum class key_shape_t : std::uint8_t {
    uniform_k,
    all_zeros_k,
    all_ones_k,
    tied_high_words_k,
    integer_identities_k,
    sign_boundary_k,
};

constexpr key_shape_t key_shapes_k[] = {
    key_shape_t::uniform_k,         key_shape_t::all_zeros_k,          key_shape_t::all_ones_k,
    key_shape_t::tied_high_words_k, key_shape_t::integer_identities_k, key_shape_t::sign_boundary_k,
};

template <typename key_type_>
[[nodiscard]] constexpr key_type_ key_from_words(std::uint64_t high, std::uint64_t low) noexcept {
    if constexpr (std::same_as<key_type_, key128_t>) return key128_t {high, low};
    else return static_cast<key_type_>(low);
}

template <typename key_type_>
[[nodiscard]] constexpr key_type_ predecessor_or_self(key_type_ key) noexcept {
    if constexpr (std::same_as<key_type_, key128_t>) {
        if (key.low != 0) return key128_t {key.high, key.low - 1};
        if (key.high != 0) return key128_t {key.high - 1, (std::numeric_limits<std::uint64_t>::max)()};
        return key;
    }
    else return key == (std::numeric_limits<key_type_>::min)() ? key : static_cast<key_type_>(key - 1);
}

/** Fills @p keys from @p shape: a tie sits in a 16-byte key's high word, repeats in an integer. */
template <typename key_type_>
void fill_keys(std::span<key_type_> keys, key_shape_t shape, split_mix_generator_t &generator) {
    std::uint64_t const all_ones = (std::numeric_limits<std::uint64_t>::max)();
    std::uint64_t const tied_word = generator();
    std::uint64_t const boundary = sizeof(key_type_) == 4 ? std::uint64_t {1} << 31 : std::uint64_t {1} << 63;
    for (key_type_ &key : keys) {
        std::uint64_t const random_high = generator();
        std::uint64_t const random_low = generator();
        switch (shape) {
        case key_shape_t::uniform_k: key = key_from_words<key_type_>(random_high, random_low); break;
        case key_shape_t::all_zeros_k: key = key_from_words<key_type_>(0, 0); break;
        case key_shape_t::all_ones_k: key = key_from_words<key_type_>(all_ones, all_ones); break;
        case key_shape_t::tied_high_words_k:
            key = std::same_as<key_type_, key128_t> ? key_from_words<key_type_>(tied_word, random_low)
                                                    : key_from_words<key_type_>(0, tied_word + random_low % 3);
            break;
        case key_shape_t::integer_identities_k: key = key_from_words<key_type_>(0, random_low % 4096); break;
        case key_shape_t::sign_boundary_k:
            key = key_from_words<key_type_>(boundary - 2 + random_high % 4, boundary - 2 + random_low % 4);
            break;
        }
    }
}

/** Collects the keys worth asking about: extremes, the boundary, and the row's own neighbours. */
template <typename key_type_>
void collect_wanted(std::span<key_type_ const> keys, split_mix_generator_t &generator,
                    basic_vector<key_type_> &wanted) {
    std::uint64_t const all_ones = (std::numeric_limits<std::uint64_t>::max)();
    std::uint64_t const boundary = std::uint64_t {1} << 63;
    wanted.clear();
    for (key_type_ const extreme :
         {key_from_words<key_type_>(0, 0), key_from_words<key_type_>(all_ones, all_ones),
          key_from_words<key_type_>(0, all_ones), key_from_words<key_type_>(all_ones, 0),
          key_from_words<key_type_>(boundary, boundary), key_from_words<key_type_>(0, 1u << 31),
          (std::numeric_limits<key_type_>::min)(), (std::numeric_limits<key_type_>::max)()})
        expect_equal(wanted.push_back(key_type_ {extreme}), success_k);
    std::size_t const stride = keys.size() / 24 + 1;
    for (std::size_t index = 0; index < keys.size(); index += stride) {
        expect_equal(wanted.push_back(key_type_ {keys[index]}), success_k);
        expect_equal(wanted.push_back(successor_or_self(keys[index])), success_k);
        expect_equal(wanted.push_back(predecessor_or_self(keys[index])), success_k);
    }
    for (std::size_t draw = 0; draw < 8; ++draw)
        expect_equal(wanted.push_back(key_from_words<key_type_>(generator(), generator())), success_k);
}

#pragma endregion Key Generation

#pragma region Kit Equivalence

/** Splits a 16-byte row into its two columns, the way a split node stores it. */
void split_columns(std::span<key128_t const> keys, basic_vector<std::uint64_t> &high_words,
                   basic_vector<std::uint64_t> &low_words) {
    expect_equal(high_words.resize(keys.size()), success_k);
    expect_equal(low_words.resize(keys.size()), success_k);
    for (std::size_t index = 0; index < keys.size(); ++index) {
        high_words[index] = keys[index].high;
        low_words[index] = keys[index].low;
    }
}

template <typename kit_type_, typename key_type_, std::size_t extent_>
void check_row(std::span<key_type_ const, extent_> row, std::span<key_type_ const> wanted, bool is_sorted) {
    basic_vector<std::uint64_t> high_words;
    basic_vector<std::uint64_t> low_words;
    if constexpr (std::same_as<key_type_, key128_t>) split_columns(row, high_words, low_words);
    for (key_type_ const key : wanted) {
        std::size_t const expected_below = serial_row_kit_t::count_below(row, key);
        expect_equal(kit_type_::count_below(row, key), expected_below);
        expect_equal(count_not_above<kit_type_>(row, key), count_not_above<serial_row_kit_t>(row, key));
        if constexpr (std::same_as<key_type_, key128_t>) {
            std::span<std::uint64_t const, extent_> const highs(high_words.data(), row.size());
            std::span<std::uint64_t const, extent_> const lows(low_words.data(), row.size());
            expect_equal(serial_row_kit_t::count_below(highs, lows, key), expected_below);
            expect_equal(kit_type_::count_below(highs, lows, key), expected_below);
            expect_equal(count_not_above<kit_type_>(highs, lows, key), count_not_above<serial_row_kit_t>(row, key));
        }
        if (is_sorted) {
            std::span<key_type_ const> const sorted(row.data(), row.size());
            expect_equal(count_below_sorted<kit_type_>(sorted, key), lower_bound_of(sorted, key));
            if constexpr (std::same_as<key_type_, key128_t>)
                expect_equal(
                    count_below_sorted<kit_type_>(std::span<std::uint64_t const>(high_words.data(), row.size()),
                                                  std::span<std::uint64_t const>(low_words.data(), row.size()), key),
                    lower_bound_of(sorted, key));
        }
    }
}

template <typename kit_type_, typename key_type_>
void check_rows_of(split_mix_generator_t &generator) {
    constexpr std::size_t long_lengths[] = {127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4099};
    basic_vector<key_type_> keys;
    basic_vector<key_type_> wanted;
    for (key_shape_t const shape : key_shapes_k) {
        for (std::size_t length = 0; length < 70 + std::size(long_lengths); ++length) {
            std::size_t const row_length = length < 70 ? length : long_lengths[length - 70];
            expect_equal(keys.resize(row_length), success_k);
            fill_keys(std::span<key_type_>(keys.data(), row_length), shape, generator);
            std::span<key_type_ const> const row(keys.data(), row_length);
            collect_wanted(row, generator, wanted);
            std::span<key_type_ const> const asked(wanted.data(), wanted.size());
            check_row<kit_type_>(row, asked, false);
            std::sort(keys.begin(), keys.end());
            check_row<kit_type_>(row, asked, true);
        }
        // A static extent is the width a medium fixes, and the kit may unroll its loop for it.
        for (std::size_t const extent : {4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u}) {
            expect_equal(keys.resize(extent), success_k);
            fill_keys(std::span<key_type_>(keys.data(), extent), shape, generator);
            std::sort(keys.begin(), keys.end());
            collect_wanted(std::span<key_type_ const>(keys.data(), extent), generator, wanted);
            std::span<key_type_ const> const asked(wanted.data(), wanted.size());
            switch (extent) {
            case 4: check_row<kit_type_>(std::span<key_type_ const, 4>(keys.data(), 4), asked, true); break;
            case 8: check_row<kit_type_>(std::span<key_type_ const, 8>(keys.data(), 8), asked, true); break;
            case 16: check_row<kit_type_>(std::span<key_type_ const, 16>(keys.data(), 16), asked, true); break;
            case 32: check_row<kit_type_>(std::span<key_type_ const, 32>(keys.data(), 32), asked, true); break;
            case 64: check_row<kit_type_>(std::span<key_type_ const, 64>(keys.data(), 64), asked, true); break;
            case 128: check_row<kit_type_>(std::span<key_type_ const, 128>(keys.data(), 128), asked, true); break;
            case 256: check_row<kit_type_>(std::span<key_type_ const, 256>(keys.data(), 256), asked, true); break;
            case 512: check_row<kit_type_>(std::span<key_type_ const, 512>(keys.data(), 512), asked, true); break;
            default: check_row<kit_type_>(std::span<key_type_ const, 1024>(keys.data(), 1024), asked, true); break;
            }
        }
    }
}

/** Masks of equal keys against a plain loop, the wanted key at every position and at none. */
template <typename kit_type_, typename word_type_>
void check_masks_of(split_mix_generator_t &generator) {
    constexpr std::size_t mask_equal_width_k = kit_type_::mask_equal_width_k;
    word_type_ keys[mask_equal_width_k];
    for (std::size_t round = 0; round != 256; ++round) {
        // Narrow values make repeats likely: several keys can match at once, or in one word only.
        for (word_type_ &key : keys) {
            std::uint64_t const high = round % 2 ? generator() % 2 : generator();
            key = key_from_words<word_type_>(high, round % 2 ? generator() % 4 : generator());
        }
        // The largest word must match too, where a signed compare of unsigned keys would wrap.
        keys[round % mask_equal_width_k] = key_from_words<word_type_>(~0ull, ~0ull);
        std::span<word_type_ const, mask_equal_width_k> const row(keys);
        // Every key in turn, then one drawn afresh that is most likely absent.
        for (std::size_t position = 0; position <= mask_equal_width_k; ++position) {
            word_type_ const wanted =
                position != mask_equal_width_k ? keys[position] : key_from_words<word_type_>(generator(), generator());
            std::uint32_t expected = 0;
            for (std::size_t index = 0; index != mask_equal_width_k; ++index)
                expected |= static_cast<std::uint32_t>(keys[index] == wanted) << index;
            expect_equal(kit_type_::mask_equal(row, wanted), expected);
        }
    }
}

template <typename kit_type_>
void check_kit_rows(kit_type_) {
    split_mix_generator_t generator {case_stream_key()};
    check_rows_of<kit_type_, std::uint32_t>(generator);
    check_rows_of<kit_type_, std::int32_t>(generator);
    check_rows_of<kit_type_, std::uint64_t>(generator);
    check_rows_of<kit_type_, std::int64_t>(generator);
    check_rows_of<kit_type_, key128_t>(generator);
    check_masks_of<kit_type_, std::uint32_t>(generator);
    check_masks_of<kit_type_, std::uint64_t>(generator);
    check_masks_of<kit_type_, key128_t>(generator);
}

#pragma endregion Kit Equivalence

#pragma region Layout Equivalence

/** Saves @p tree into words, so the view reads it from a word boundary as a mapped file would, and
 *  answers the view built over those bytes. */
template <typename tree_type_>
expected<typename tree_type_::view_t> save_and_view(tree_type_ const &tree, basic_vector<std::uint64_t> &words) {
    expect_equal(words.resize(tree.saved_bytes() / sizeof(std::uint64_t) + 1), success_k);
    std::span<std::byte> const bytes(reinterpret_cast<std::byte *>(words.data()), tree.saved_bytes());
    expect_equal(tree.save(bytes), success_k);
    auto [view, status] = tree_type_::view_t::make(std::span<std::byte const>(bytes));
    if (succeeded(status)) expect_equal(view.saved_bytes(), tree.saved_bytes());
    return {std::move(view), status};
}

template <typename kit_type_, typename key_type_, std::size_t keys_per_row_>
void check_layouts_of(split_mix_generator_t &generator) {
    using btree_t = immutable_b_tree<key_type_, keys_per_row_, kit_type_>;
    using splus_t = immutable_splus_tree<key_type_, keys_per_row_, kit_type_>;
    check_ordered_readonly_tags<btree_t>();
    check_ordered_readonly_tags<splus_t>();
    std::size_t const row = keys_per_row_;
    std::size_t const fanout = row + 1;
    std::size_t const sizes[] = {
        0,
        1,
        row - 1,
        row,
        row + 1,
        2 * row + 1,
        row * fanout - 1,
        row * fanout,
        row * fanout + 1,
        row * fanout * fanout / 3 + 7,
        static_cast<std::size_t>(3000 + generator() % 9000),
    };
    basic_vector<key_type_> keys;
    basic_vector<key_type_> wanted;
    basic_vector<std::uint64_t> saved;
    for (key_shape_t const shape : key_shapes_k)
        for (std::size_t const size : sizes) {
            if (size > (std::size_t {1} << 20)) continue;
            expect_equal(keys.resize(size), success_k);
            fill_keys(std::span<key_type_>(keys.data(), size), shape, generator);
            std::sort(keys.begin(), keys.end());
            std::span<key_type_ const> const sorted(keys.data(), size);
            collect_wanted(sorted, generator, wanted);
            std::span<key_type_ const> const asked(wanted.data(), wanted.size());

            auto [btree, btree_status] = btree_t::make(sorted);
            if (!expect_equal(btree_status, success_k)) return;
            check_ordered_readonly(btree, sorted, asked);
            auto [splus, splus_status] = splus_t::make(sorted);
            if (!expect_equal(splus_status, success_k)) return;
            check_ordered_readonly(splus, sorted, asked);
            auto [view, save_status] = save_and_view(splus, saved);
            if (!expect_equal(save_status, success_k)) return;
            check_ordered_readonly(view, sorted, asked);
        }

    // Out-of-order input is refused rather than built into a tree that answers wrongly.
    key_type_ const reversed[] = {key_from_words<key_type_>(0, 2), key_from_words<key_type_>(0, 1)};
    expect_equal(btree_t::make(std::span<key_type_ const>(reversed, 2)).status(), status_t::invalid_argument_k);
    expect_equal(splus_t::make(std::span<key_type_ const>(reversed, 2)).status(), status_t::invalid_argument_k);
}

template <typename kit_type_>
void check_kit_layouts(kit_type_) {
    split_mix_generator_t generator {case_stream_key()};
    // Widths from each medium: both cache lines, a block and a page, and a few odd ones.
    check_layouts_of<kit_type_, key128_t, 2>(generator);
    check_layouts_of<kit_type_, key128_t, 3>(generator);
    check_layouts_of<kit_type_, key128_t, keys_per_row<key128_t>(64u)>(generator);
    check_layouts_of<kit_type_, key128_t, keys_per_row<key128_t>(128u)>(generator);
    check_layouts_of<kit_type_, key128_t, keys_per_row<key128_t>(512u)>(generator);
    check_layouts_of<kit_type_, key128_t, keys_per_row<key128_t>(4096u)>(generator);
    check_layouts_of<kit_type_, std::uint64_t, 5>(generator);
    check_layouts_of<kit_type_, std::uint64_t, keys_per_row<std::uint64_t>(64u)>(generator);
    check_layouts_of<kit_type_, std::uint64_t, keys_per_row<std::uint64_t>(128u)>(generator);
    check_layouts_of<kit_type_, std::uint64_t, keys_per_row<std::uint64_t>(512u)>(generator);
    check_layouts_of<kit_type_, std::uint64_t, keys_per_row<std::uint64_t>(4096u)>(generator);
    check_layouts_of<kit_type_, std::int64_t, keys_per_row<std::int64_t>(64u)>(generator);
    check_layouts_of<kit_type_, std::int32_t, keys_per_row<std::int32_t>(512u)>(generator);
    check_layouts_of<kit_type_, std::uint32_t, keys_per_row<std::uint32_t>(128u)>(generator);
    check_layouts_of<kit_type_, std::uint32_t, keys_per_row<std::uint32_t>(512u)>(generator);
}

#pragma endregion Layout Equivalence

#pragma region Tests

/** The serial kit is the reference, answering the definitions: byte order, bounds and widths. */
void test_serial_matches_definitions() {
    split_mix_generator_t generator {case_stream_key()};
    for (std::size_t draw = 0; draw < 4096; ++draw) {
        std::byte first_bytes[16];
        std::byte second_bytes[16];
        for (std::size_t index = 0; index < 16; ++index) {
            // Few distinct bytes, so equal prefixes of every length occur.
            first_bytes[index] = static_cast<std::byte>(generator() % 3 * 0x7F);
            second_bytes[index] = static_cast<std::byte>(generator() % 3 * 0x7F);
        }
        key128_t const first = key128_t::from_bytes(first_bytes);
        key128_t const second = key128_t::from_bytes(second_bytes);
        int const byte_order = std::memcmp(first_bytes, second_bytes, 16);
        expect_equal(first < second, byte_order < 0);
        expect_equal(first == second, byte_order == 0);
        std::byte round_trip[16];
        first.to_bytes(round_trip);
        expect_equal(std::memcmp(round_trip, first_bytes, 16), 0);
    }

    std::uint64_t const words[] = {0, 1, 5, 5, 9, (std::numeric_limits<std::uint64_t>::max)()};
    std::span<std::uint64_t const> const row(words, 6);
    expect_equal(serial_row_kit_t::count_below(row, std::uint64_t {5}), 2u);
    expect_equal(count_not_above<serial_row_kit_t>(row, std::uint64_t {5}), 4u);
    expect_equal(count_not_above<serial_row_kit_t>(row, (std::numeric_limits<std::uint64_t>::max)()), 6u);

    // An integer identity parks in the low word and orders as its value.
    expect_true((key128_t {0, 42} < key128_t {0, 43}));
    expect_true((key128_t {0, (std::numeric_limits<std::uint64_t>::max)()} < key128_t {1, 0}));

    expect_equal(keys_per_row<key128_t>(512u), 32u);
    expect_equal(keys_per_row<key128_t>(4096u), 256u);
    expect_equal(keys_per_row<key128_t>(64u), 4u);
    expect_equal(keys_per_row<std::uint64_t>(512u), 64u);
    expect_equal(keys_per_row<std::uint32_t>(4096u), 1024u);
    using block_format_t = row_format<key128_t, 32>;
    using page_format_t = row_format<std::uint64_t, 512>;
    expect_equal(block_format_t::bytes_per_row_k, 512u);
    expect_equal(page_format_t::bytes_per_row_k, 4096u);
    expect_equal(immutable_b_tree<key128_t>::keys_per_row_k, cache_line_bytes_k / sizeof(key128_t));

    row_kit_t const detected = detect_row_kit();
    expect_true(row_kit_compiled(detected));
    expect_true(row_kit_supported(detected));
}

/** @c check_kit_rows through @p kit , skipped where this machine cannot run it. */
case_result_t test_kit_rows(row_kit_t kit) {
    if (!row_kit_supported(kit)) return case_result_t::skipped(needs_kit_reason(kit));
    visit_row_kit(kit, [](row_kit auto kit_instance) noexcept { check_kit_rows(kit_instance); });
    return {};
}

/** @c check_kit_layouts through @p kit , skipped where this machine cannot run it. */
case_result_t test_kit_layouts(row_kit_t kit) {
    if (!row_kit_supported(kit)) return case_result_t::skipped(needs_kit_reason(kit));
    visit_row_kit(kit, [](row_kit auto kit_instance) noexcept { check_kit_layouts(kit_instance); });
    return {};
}

/** A map form answers its set twin's keys and returns the value each key arrived with. */
void test_mapped_values() {
    check_ordered_readonly_mapping<immutable_b_map<std::uint64_t, std::uint64_t>, immutable_b_set<std::uint64_t>>();
    check_ordered_readonly_mapping<immutable_splus_map<std::uint64_t, std::uint64_t>,
                                   immutable_splus_set<std::uint64_t>>();
    check_ordered_readonly_mapping<immutable_b_map<std::uint32_t, std::uint32_t>, immutable_b_set<std::uint32_t>>();
    check_ordered_readonly_mapping<immutable_splus_map<std::int64_t, std::uint64_t>,
                                   immutable_splus_set<std::int64_t>>();
}

/** A saved map reads back every value beside its key in place, and a view refuses bytes that are
 *  misaligned, truncated, or saved by a tree of another row width or element. */
void test_saved_views() {
    using map_t = immutable_splus_map<std::uint64_t, std::uint64_t>;
    using pair_t = map_t::value_t;
    basic_vector<pair_t> pairs;
    for (std::uint64_t index = 0; index != 1000; ++index)
        expect_equal(pairs.push_back(pair_t {index * 3, index * 7}), success_k);
    auto [map, map_status] = map_t::make({pairs.data(), pairs.size()});
    if (!expect_equal(map_status, success_k)) return;
    basic_vector<std::uint64_t> words;
    auto [view, save_status] = save_and_view(map, words);
    if (!expect_equal(save_status, success_k)) return;
    for (std::size_t rank = 0; rank != pairs.size(); ++rank) {
        expect_equal(view.select(rank), pairs[rank].key);
        expect_equal(view.mapped_at(rank), pairs[rank].mapped);
    }
    expect_equal(view.find(pairs[500].key), 500u);
    expect_equal(view.find(1).status(), status_t::key_not_found_k);

    std::span<std::byte const> const bytes(reinterpret_cast<std::byte const *>(words.data()), map.saved_bytes());
    expect_equal(map_t::view_t::make(bytes.first(bytes.size() - 1)).status(), status_t::invalid_argument_k);
    expect_equal(map_t::view_t::make(bytes.first(sizeof(splus_tree_header_t) - 1)).status(),
                 status_t::invalid_argument_k);
    expect_equal(map_t::view_t::make(bytes.subspan(1)).status(), status_t::invalid_argument_k, "a misaligned start");
    expect_equal(immutable_splus_set<std::uint64_t>::view_t::make(bytes).status(), status_t::invalid_argument_k,
                 "a set over a map's bytes");
    expect_equal((immutable_splus_tree_view<pair_t, map_t::keys_per_row_k + 1>::make(bytes).status()),
                 status_t::invalid_argument_k, "another row width");
    std::span<std::byte> const short_bytes(reinterpret_cast<std::byte *>(words.data()), map.saved_bytes() - 1);
    expect_equal(map.save(short_bytes), status_t::invalid_argument_k);

    auto [empty, empty_status] = immutable_splus_set<std::uint32_t>::make({});
    if (!expect_equal(empty_status, success_k)) return;
    auto [empty_view, empty_save_status] = save_and_view(empty, words);
    if (!expect_equal(empty_save_status, success_k)) return;
    expect_true(empty_view.empty());
    expect_equal(empty_view.rank(7u), 0u);
}

#pragma endregion Tests

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::test;

int main(int argc, char **argv) {
    start_test_run(argc, argv);

    run_test_case("test_serial_matches_definitions", test_serial_matches_definitions);
    run_test_case("test_kit_rows(row_kit_t::serial_k)", test_kit_rows, row_kit_t::serial_k);
    run_test_case("test_kit_rows(row_kit_t::haswell_k)", test_kit_rows, row_kit_t::haswell_k);
    run_test_case("test_kit_rows(row_kit_t::skylake_k)", test_kit_rows, row_kit_t::skylake_k);
    run_test_case("test_kit_rows(row_kit_t::icelake_k)", test_kit_rows, row_kit_t::icelake_k);
    run_test_case("test_kit_rows(row_kit_t::neon_k)", test_kit_rows, row_kit_t::neon_k);
    run_test_case("test_kit_rows(row_kit_t::sve_k)", test_kit_rows, row_kit_t::sve_k);
    run_test_case("test_kit_rows(row_kit_t::rvv_k)", test_kit_rows, row_kit_t::rvv_k);
    run_test_case("test_kit_rows(row_kit_t::powervsx_k)", test_kit_rows, row_kit_t::powervsx_k);
    run_test_case("test_kit_rows(row_kit_t::v128_k)", test_kit_rows, row_kit_t::v128_k);
    run_test_case("test_kit_layouts(row_kit_t::serial_k)", test_kit_layouts, row_kit_t::serial_k);
    run_test_case("test_kit_layouts(row_kit_t::haswell_k)", test_kit_layouts, row_kit_t::haswell_k);
    run_test_case("test_kit_layouts(row_kit_t::skylake_k)", test_kit_layouts, row_kit_t::skylake_k);
    run_test_case("test_kit_layouts(row_kit_t::icelake_k)", test_kit_layouts, row_kit_t::icelake_k);
    run_test_case("test_kit_layouts(row_kit_t::neon_k)", test_kit_layouts, row_kit_t::neon_k);
    run_test_case("test_kit_layouts(row_kit_t::sve_k)", test_kit_layouts, row_kit_t::sve_k);
    run_test_case("test_kit_layouts(row_kit_t::rvv_k)", test_kit_layouts, row_kit_t::rvv_k);
    run_test_case("test_kit_layouts(row_kit_t::powervsx_k)", test_kit_layouts, row_kit_t::powervsx_k);
    run_test_case("test_kit_layouts(row_kit_t::v128_k)", test_kit_layouts, row_kit_t::v128_k);
    run_test_case("test_mapped_values", test_mapped_values);
    run_test_case("test_saved_views", test_saved_views);

    return finish_test_run();
}
