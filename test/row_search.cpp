/**
 *  @file test/row_search.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Tests for the row kits and the static layouts: every kit this processor runs against the
 *      serial kit, and both layouts against a sorted array.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstring> // `std::memcmp`

#include <algorithm> // `std::lower_bound`, `std::sort`
#include <concepts>  // `std::same_as`
#include <limits>    // `std::numeric_limits`
#include <random>    // `std::mt19937_64`
#include <span>      // `std::span`

#include <smashtable/basic_vector.hpp>
#include <smashtable/immutable_b_tree.hpp>
#include <smashtable/immutable_splus_tree.hpp>
#include <smashtable/row_search.hpp>

#include "harness.hpp"
#include "ordered_readonly.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::test;

namespace {

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
        if (key.high != 0) return key128_t {key.high - 1, std::numeric_limits<std::uint64_t>::max()};
        return key;
    }
    else return key == std::numeric_limits<key_type_>::min() ? key : static_cast<key_type_>(key - 1);
}

/** Fills @p keys from @p shape: a tie sits in a 16-byte key's high word, repeats in an integer. */
template <typename key_type_>
void fill_keys(std::span<key_type_> keys, key_shape_t shape, std::mt19937_64 &generator) {
    std::uint64_t const all_ones = std::numeric_limits<std::uint64_t>::max();
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
void collect_wanted(std::span<key_type_ const> keys, std::mt19937_64 &generator, basic_vector<key_type_> &wanted) {
    std::uint64_t const all_ones = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t const boundary = std::uint64_t {1} << 63;
    wanted.clear();
    for (key_type_ const extreme :
         {key_from_words<key_type_>(0, 0), key_from_words<key_type_>(all_ones, all_ones),
          key_from_words<key_type_>(0, all_ones), key_from_words<key_type_>(all_ones, 0),
          key_from_words<key_type_>(boundary, boundary), key_from_words<key_type_>(0, 1u << 31),
          std::numeric_limits<key_type_>::min(), std::numeric_limits<key_type_>::max()})
        st_verify_(wanted.push_back(key_type_ {extreme}));
    std::size_t const stride = keys.size() / 24 + 1;
    for (std::size_t index = 0; index < keys.size(); index += stride) {
        st_verify_(wanted.push_back(key_type_ {keys[index]}));
        st_verify_(wanted.push_back(successor_or_self(keys[index])));
        st_verify_(wanted.push_back(predecessor_or_self(keys[index])));
    }
    for (std::size_t draw = 0; draw < 8; ++draw)
        st_verify_(wanted.push_back(key_from_words<key_type_>(generator(), generator())));
}

#pragma endregion Key Generation

#pragma region Kit Equivalence

/** Splits a 16-byte row into its two columns, the way a split node stores it. */
void split_columns(std::span<key128_t const> keys, basic_vector<std::uint64_t> &high_words,
                   basic_vector<std::uint64_t> &low_words) {
    st_verify_(high_words.resize(keys.size()));
    st_verify_(low_words.resize(keys.size()));
    for (std::size_t index = 0; index < keys.size(); ++index) {
        high_words[index] = keys[index].high;
        low_words[index] = keys[index].low;
    }
}

template <typename kit_type_, typename key_type_, std::size_t extent_>
void verify_row(std::span<key_type_ const, extent_> row, std::span<key_type_ const> wanted, bool is_sorted) {
    basic_vector<std::uint64_t> high_words;
    basic_vector<std::uint64_t> low_words;
    if constexpr (std::same_as<key_type_, key128_t>) split_columns(row, high_words, low_words);
    for (key_type_ const key : wanted) {
        std::size_t const expected_below = serial_row_kit_t::count_below(row, key);
        st_verify_eq_(kit_type_::count_below(row, key), expected_below);
        st_verify_eq_(count_not_above<kit_type_>(row, key), count_not_above<serial_row_kit_t>(row, key));
        if constexpr (std::same_as<key_type_, key128_t>) {
            std::span<std::uint64_t const, extent_> const highs(high_words.data(), row.size());
            std::span<std::uint64_t const, extent_> const lows(low_words.data(), row.size());
            st_verify_eq_(serial_row_kit_t::count_below(highs, lows, key), expected_below);
            st_verify_eq_(kit_type_::count_below(highs, lows, key), expected_below);
            st_verify_eq_(count_not_above<kit_type_>(highs, lows, key), count_not_above<serial_row_kit_t>(row, key));
        }
        if (is_sorted) {
            std::span<key_type_ const> const sorted(row.data(), row.size());
            st_verify_eq_(count_below_sorted<kit_type_>(sorted, key), lower_bound_of(sorted, key));
            if constexpr (std::same_as<key_type_, key128_t>)
                st_verify_eq_(
                    count_below_sorted<kit_type_>(std::span<std::uint64_t const>(high_words.data(), row.size()),
                                                  std::span<std::uint64_t const>(low_words.data(), row.size()), key),
                    lower_bound_of(sorted, key));
        }
    }
}

template <typename kit_type_, typename key_type_>
void verify_rows_of(std::mt19937_64 &generator) {
    constexpr std::size_t long_lengths[] = {127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4099};
    basic_vector<key_type_> keys;
    basic_vector<key_type_> wanted;
    for (key_shape_t const shape : key_shapes_k) {
        for (std::size_t length = 0; length < 70 + std::size(long_lengths); ++length) {
            std::size_t const row_length = length < 70 ? length : long_lengths[length - 70];
            st_verify_(keys.resize(row_length));
            fill_keys(std::span<key_type_>(keys.data(), row_length), shape, generator);
            std::span<key_type_ const> const row(keys.data(), row_length);
            collect_wanted(row, generator, wanted);
            std::span<key_type_ const> const asked(wanted.data(), wanted.size());
            verify_row<kit_type_>(row, asked, false);
            std::sort(keys.begin(), keys.end());
            verify_row<kit_type_>(row, asked, true);
        }
        // A static extent is the width a medium fixes, and the kit may unroll its loop for it.
        for (std::size_t const extent : {4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u}) {
            st_verify_(keys.resize(extent));
            fill_keys(std::span<key_type_>(keys.data(), extent), shape, generator);
            std::sort(keys.begin(), keys.end());
            collect_wanted(std::span<key_type_ const>(keys.data(), extent), generator, wanted);
            std::span<key_type_ const> const asked(wanted.data(), wanted.size());
            switch (extent) {
            case 4: verify_row<kit_type_>(std::span<key_type_ const, 4>(keys.data(), 4), asked, true); break;
            case 8: verify_row<kit_type_>(std::span<key_type_ const, 8>(keys.data(), 8), asked, true); break;
            case 16: verify_row<kit_type_>(std::span<key_type_ const, 16>(keys.data(), 16), asked, true); break;
            case 32: verify_row<kit_type_>(std::span<key_type_ const, 32>(keys.data(), 32), asked, true); break;
            case 64: verify_row<kit_type_>(std::span<key_type_ const, 64>(keys.data(), 64), asked, true); break;
            case 128: verify_row<kit_type_>(std::span<key_type_ const, 128>(keys.data(), 128), asked, true); break;
            case 256: verify_row<kit_type_>(std::span<key_type_ const, 256>(keys.data(), 256), asked, true); break;
            case 512: verify_row<kit_type_>(std::span<key_type_ const, 512>(keys.data(), 512), asked, true); break;
            default: verify_row<kit_type_>(std::span<key_type_ const, 1024>(keys.data(), 1024), asked, true); break;
            }
        }
    }
}

template <typename kit_type_>
void verify_kit_rows(test_context_t const &context, kit_type_) {
    std::mt19937_64 generator(mix_seed(context.seed, name_of(kit_type_::kit_k)));
    verify_rows_of<kit_type_, std::uint32_t>(generator);
    verify_rows_of<kit_type_, std::int32_t>(generator);
    verify_rows_of<kit_type_, std::uint64_t>(generator);
    verify_rows_of<kit_type_, std::int64_t>(generator);
    verify_rows_of<kit_type_, key128_t>(generator);
}

#pragma endregion Kit Equivalence

#pragma region Layout Equivalence

template <typename kit_type_, typename key_type_, std::size_t keys_per_row_>
void verify_layouts_of(std::mt19937_64 &generator) {
    using btree_t = immutable_b_tree<key_type_, keys_per_row_, kit_type_>;
    using splus_t = immutable_splus_tree<key_type_, keys_per_row_, kit_type_>;
    test_ordered_readonly_tags<btree_t>();
    test_ordered_readonly_tags<splus_t>();
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
        3000 + generator() % 9000,
    };
    basic_vector<key_type_> keys;
    basic_vector<key_type_> wanted;
    for (key_shape_t const shape : key_shapes_k)
        for (std::size_t const size : sizes) {
            if (size > (std::size_t {1} << 20)) continue;
            st_verify_(keys.resize(size));
            fill_keys(std::span<key_type_>(keys.data(), size), shape, generator);
            std::sort(keys.begin(), keys.end());
            std::span<key_type_ const> const sorted(keys.data(), size);
            collect_wanted(sorted, generator, wanted);
            std::span<key_type_ const> const asked(wanted.data(), wanted.size());

            expected<btree_t> btree = btree_t::make(sorted);
            st_verify_(btree);
            verify_ordered_readonly(*btree, sorted, asked);
            expected<splus_t> splus = splus_t::make(sorted);
            st_verify_(splus);
            verify_ordered_readonly(*splus, sorted, asked);
        }

    // Out-of-order input is refused rather than built into a tree that answers wrongly.
    key_type_ const reversed[] = {key_from_words<key_type_>(0, 2), key_from_words<key_type_>(0, 1)};
    st_verify_eq_(btree_t::make(std::span<key_type_ const>(reversed, 2)).status(), status_t::invalid_argument_k);
    st_verify_eq_(splus_t::make(std::span<key_type_ const>(reversed, 2)).status(), status_t::invalid_argument_k);
}

template <typename kit_type_>
void verify_kit_layouts(test_context_t const &context, kit_type_) {
    std::mt19937_64 generator(mix_seed(context.seed, name_of(kit_type_::kit_k)));
    // Widths from each medium - a cache line, a 512-byte and a 4096-byte block - and a few odd ones for the arithmetic.
    verify_layouts_of<kit_type_, key128_t, 2>(generator);
    verify_layouts_of<kit_type_, key128_t, 3>(generator);
    verify_layouts_of<kit_type_, key128_t, keys_per_row<key128_t>(64u)>(generator);
    verify_layouts_of<kit_type_, key128_t, keys_per_row<key128_t>(512u)>(generator);
    verify_layouts_of<kit_type_, key128_t, keys_per_row<key128_t>(4096u)>(generator);
    verify_layouts_of<kit_type_, std::uint64_t, 5>(generator);
    verify_layouts_of<kit_type_, std::uint64_t, keys_per_row<std::uint64_t>(64u)>(generator);
    verify_layouts_of<kit_type_, std::uint64_t, keys_per_row<std::uint64_t>(512u)>(generator);
    verify_layouts_of<kit_type_, std::uint64_t, keys_per_row<std::uint64_t>(4096u)>(generator);
    verify_layouts_of<kit_type_, std::int64_t, keys_per_row<std::int64_t>(64u)>(generator);
    verify_layouts_of<kit_type_, std::int32_t, keys_per_row<std::int32_t>(512u)>(generator);
    verify_layouts_of<kit_type_, std::uint32_t, keys_per_row<std::uint32_t>(512u)>(generator);
}

#pragma endregion Layout Equivalence

#pragma region Tests

void verify_rows_through(test_context_t const &context, row_kit_t kit) {
    if (row_kit_supported(kit))
        visit_row_kit(kit, [&](row_kit auto kit_instance) noexcept { verify_kit_rows(context, kit_instance); });
}

void verify_layouts_through(test_context_t const &context, row_kit_t kit) {
    if (row_kit_supported(kit))
        visit_row_kit(kit, [&](row_kit auto kit_instance) noexcept { verify_kit_layouts(context, kit_instance); });
}

/** The serial kit is the reference, answering the definitions: byte order, bounds and widths. */
void row_search_serial_matches_definitions(test_context_t const &context) {
    std::mt19937_64 generator(mix_seed(context.seed, __func__));
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
        st_verify_eq_(first < second, byte_order < 0);
        st_verify_eq_(first == second, byte_order == 0);
        std::byte round_trip[16];
        first.to_bytes(round_trip);
        st_verify_eq_(std::memcmp(round_trip, first_bytes, 16), 0);
    }

    std::uint64_t const words[] = {0, 1, 5, 5, 9, std::numeric_limits<std::uint64_t>::max()};
    std::span<std::uint64_t const> const row(words, 6);
    st_verify_eq_(serial_row_kit_t::count_below(row, std::uint64_t {5}), 2u);
    st_verify_eq_(count_not_above<serial_row_kit_t>(row, std::uint64_t {5}), 4u);
    st_verify_eq_(count_not_above<serial_row_kit_t>(row, std::numeric_limits<std::uint64_t>::max()), 6u);

    // An integer identity parks in the low word and orders as its value.
    st_verify_((key128_t {0, 42} < key128_t {0, 43}));
    st_verify_((key128_t {0, std::numeric_limits<std::uint64_t>::max()} < key128_t {1, 0}));

    st_verify_eq_(keys_per_row<key128_t>(512u), 32u);
    st_verify_eq_(keys_per_row<key128_t>(4096u), 256u);
    st_verify_eq_(keys_per_row<key128_t>(64u), 4u);
    st_verify_eq_(keys_per_row<std::uint64_t>(512u), 64u);
    st_verify_eq_(keys_per_row<std::uint32_t>(4096u), 1024u);
    using block_format_t = row_format<key128_t, 32>;
    using page_format_t = row_format<std::uint64_t, 512>;
    st_verify_eq_(block_format_t::bytes_per_row_k, 512u);
    st_verify_eq_(page_format_t::bytes_per_row_k, 4096u);
    st_verify_eq_(immutable_b_tree<key128_t>::keys_per_row_k, 32u);

    row_kit_t const detected = detect_row_kit();
    st_verify_(row_kit_compiled(detected));
    st_verify_(row_kit_supported(detected));
}

void row_search_serial_kit(test_context_t const &context) { verify_rows_through(context, row_kit_t::serial_k); }
void row_search_haswell_kit(test_context_t const &context) { verify_rows_through(context, row_kit_t::haswell_k); }
void row_search_skylake_kit(test_context_t const &context) { verify_rows_through(context, row_kit_t::skylake_k); }
void row_search_neon_kit(test_context_t const &context) { verify_rows_through(context, row_kit_t::neon_k); }
void row_search_sve_kit(test_context_t const &context) { verify_rows_through(context, row_kit_t::sve_k); }
void row_search_rvv_kit(test_context_t const &context) { verify_rows_through(context, row_kit_t::rvv_k); }

void layouts_serial_kit(test_context_t const &context) { verify_layouts_through(context, row_kit_t::serial_k); }
void layouts_haswell_kit(test_context_t const &context) { verify_layouts_through(context, row_kit_t::haswell_k); }
void layouts_skylake_kit(test_context_t const &context) { verify_layouts_through(context, row_kit_t::skylake_k); }
void layouts_neon_kit(test_context_t const &context) { verify_layouts_through(context, row_kit_t::neon_k); }
void layouts_sve_kit(test_context_t const &context) { verify_layouts_through(context, row_kit_t::sve_k); }
void layouts_rvv_kit(test_context_t const &context) { verify_layouts_through(context, row_kit_t::rvv_k); }

/** A map form answers its set twin's keys and returns the value each key arrived with. */
void layouts_mapped_values() {
    test_ordered_readonly_mapping<immutable_b_map<std::uint64_t, std::uint64_t>, immutable_b_set<std::uint64_t>>();
    test_ordered_readonly_mapping<immutable_splus_map<std::uint64_t, std::uint64_t>,
                                  immutable_splus_set<std::uint64_t>>();
    test_ordered_readonly_mapping<immutable_b_map<std::uint32_t, std::uint32_t>, immutable_b_set<std::uint32_t>>();
    test_ordered_readonly_mapping<immutable_splus_map<std::int64_t, std::uint64_t>,
                                  immutable_splus_set<std::int64_t>>();
}

#pragma endregion Tests

} // namespace

int main(int, char **arguments) {
    test_environment_t const environment = read_test_environment(arguments[0]);
    install_test_signal_handlers();
    log_environment(environment);
    test_tally_t tally;

    tally += run_test(environment, "row_search.serial_matches_definitions", row_search_serial_matches_definitions);
    tally += run_test(environment, "row_search.serial_kit", row_search_serial_kit);
    tally += run_test(environment, "row_search.haswell_kit", row_search_haswell_kit);
    tally += run_test(environment, "row_search.skylake_kit", row_search_skylake_kit);
    tally += run_test(environment, "row_search.neon_kit", row_search_neon_kit);
    tally += run_test(environment, "row_search.sve_kit", row_search_sve_kit);
    tally += run_test(environment, "row_search.rvv_kit", row_search_rvv_kit);
    tally += run_test(environment, "layouts.serial_kit", layouts_serial_kit);
    tally += run_test(environment, "layouts.haswell_kit", layouts_haswell_kit);
    tally += run_test(environment, "layouts.skylake_kit", layouts_skylake_kit);
    tally += run_test(environment, "layouts.neon_kit", layouts_neon_kit);
    tally += run_test(environment, "layouts.sve_kit", layouts_sve_kit);
    tally += run_test(environment, "layouts.rvv_kit", layouts_rvv_kit);
    tally += run_test(environment, "layouts.mapped_values", layouts_mapped_values);

    return report_test_failures(environment, tally);
}
