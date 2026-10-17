/**
 *  @file test/elias_fano.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Tests for @c immutable_elias_fano and the @c immutable_packed_array keeping its lows:
 *      every value reads back by position, whole, pulled in chunks or viewed in saved bytes, the
 *      refusals hold, and the space stays near the Elias-Fano bound.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstring> // `std::memcpy`

#include <bit>    // `std::bit_width`
#include <limits> // `std::numeric_limits`
#include <random> // `std::mt19937_64`
#include <span>   // `std::span`
#include <vector> // `std::vector`

#include "smashtable/immutable_elias_fano.hpp"
#include "smashtable/immutable_packed_array.hpp"

#include "harness.hpp"
#include "fixtures.hpp"

namespace ashvardanian::smashtable::test {

using sequence_t = immutable_elias_fano<std::uint64_t>;

/** The gaps between consecutive values, each aimed at a different split of low and high parts. */
enum class gaps_t : std::uint8_t {
    dense_k,
    sparse_k,
    runs_k,
    jumps_k,
};

static std::vector<std::uint64_t> make_values(std::mt19937_64 &generator, std::size_t count, gaps_t gaps) {
    std::vector<std::uint64_t> values(count);
    std::uint64_t running = generator() % 1000;
    for (std::uint64_t &value : values) {
        switch (gaps) {
        case gaps_t::dense_k: running += generator() % 4; break;
        case gaps_t::sparse_k: running += generator() % 1000000; break;
        case gaps_t::runs_k: running += generator() % 16 == 0 ? generator() % 64 : 0; break;
        case gaps_t::jumps_k: running += generator() % 128 == 0 ? std::uint64_t(1) << (generator() % 40) : 1; break;
        }
        value = running;
    }
    return values;
}

/** Builds @p values pulled @p chunk at a time, the way a reader streams an offsets column. */
template <typename sequence_type_ = sequence_t>
static expected<sequence_type_> make_chunked(std::span<std::uint64_t const> values, std::size_t chunk) {
    std::uint64_t const universe = values.empty() ? 0 : values.back();
    return sequence_type_::make(values.size(), universe, [values, chunk](std::size_t first) noexcept {
        std::size_t const pulled = values.size() - first < chunk ? values.size() - first : chunk;
        return expected<std::span<std::uint64_t const>>(values.subspan(first, pulled), success_k);
    });
}

#pragma region Tests

/** An empty sequence stores nothing, and single values read back at both ends of the range. */
void elias_fano_edges() {
    auto empty = sequence_t::make(std::span<std::uint64_t const> {});
    st_verify_(empty);
    st_verify_(empty->empty());
    st_verify_eq_(empty->size_bytes(), 0u);

    for (std::uint64_t const value : {std::uint64_t(0), std::uint64_t(7), std::numeric_limits<std::uint64_t>::max()}) {
        std::uint64_t const values[1] = {value};
        auto single = sequence_t::make(std::span<std::uint64_t const>(values));
        st_verify_(single);
        st_verify_eq_(single->select(0), value);
    }

    std::vector<std::uint64_t> const zeros(1000, 0);
    auto flat = sequence_t::make(std::span<std::uint64_t const>(zeros));
    st_verify_(flat);
    for (std::size_t position = 0; position != zeros.size(); ++position) st_verify_eq_(flat->select(position), 0u);
}

/** Every value of every gap shape reads back at its position, whole or pulled in chunks, across
 *  the sample stride, through one kit's select scan. */
template <typename kit_type_>
void verify_kit(test_context_t const &context, kit_type_) {
    using kit_sequence_t = immutable_elias_fano<std::uint64_t, default_allocator<std::uint64_t>, kit_type_>;
    std::mt19937_64 generator(mix_seed(context.seed, name_of(kit_type_::kit_k)));
    for (std::size_t const count : {std::size_t(1), std::size_t(255), std::size_t(256), std::size_t(257),
                                    std::size_t(4099), std::size_t(1) << 18})
        for (gaps_t const gaps : {gaps_t::dense_k, gaps_t::sparse_k, gaps_t::runs_k, gaps_t::jumps_k}) {
            std::vector<std::uint64_t> const values = make_values(generator, count, gaps);
            auto whole = kit_sequence_t::make(std::span<std::uint64_t const>(values));
            auto chunked = make_chunked<kit_sequence_t>(values, 1 + generator() % 1000);
            st_verify_(whole);
            st_verify_(chunked);
            st_verify_eq_(whole->size(), count);
            st_verify_eq_(chunked->size_bytes(), whole->size_bytes());
            for (std::size_t position = 0; position != count; ++position) {
                st_verify_eq_(whole->select(position), values[position]);
                st_verify_eq_(chunked->select(position), values[position]);
            }
        }

    // Only a corrupted sequence asks past its last set bit, and the scan stops at the span instead
    // of reading beyond it - through the serial tail, and through a kit's own wider steps.
    using scan_kit_t = std::conditional_t<select_kit<kit_type_>, kit_type_, serial_row_kit_t>;
    for (std::size_t const length : {std::size_t(1), std::size_t(3), std::size_t(17)}) {
        std::vector<std::uint64_t> words(length, 0);
        words[0] = 1;
        st_verify_eq_(scan_kit_t::select_set_bit(std::span<std::uint64_t const>(words), 0, 1), length * 64);
    }
}

void verify_through(test_context_t const &context, row_kit_t kit) {
    if (row_kit_supported(kit))
        visit_row_kit(kit, [&](row_kit auto kit_instance) noexcept { verify_kit(context, kit_instance); });
}

void elias_fano_serial_kit(test_context_t const &context) { verify_through(context, row_kit_t::serial_k); }
void elias_fano_haswell_kit(test_context_t const &context) { verify_through(context, row_kit_t::haswell_k); }
void elias_fano_skylake_kit(test_context_t const &context) { verify_through(context, row_kit_t::skylake_k); }
void elias_fano_icelake_kit(test_context_t const &context) { verify_through(context, row_kit_t::icelake_k); }
void elias_fano_neon_kit(test_context_t const &context) { verify_through(context, row_kit_t::neon_k); }
void elias_fano_sve_kit(test_context_t const &context) { verify_through(context, row_kit_t::sve_k); }
void elias_fano_rvv_kit(test_context_t const &context) { verify_through(context, row_kit_t::rvv_k); }
void elias_fano_v128_kit(test_context_t const &context) { verify_through(context, row_kit_t::v128_k); }

/** Narrower values take the same layout, read back at their own width. */
void elias_fano_narrow_values(test_context_t const &context) {
    std::mt19937_64 generator(mix_seed(context.seed, __func__));
    std::vector<std::uint32_t> values(5000);
    std::uint32_t running = 0;
    for (std::uint32_t &value : values) value = running += static_cast<std::uint32_t>(generator() % 300);
    auto sequence = immutable_elias_fano<std::uint32_t>::make(std::span<std::uint32_t const>(values));
    st_verify_(sequence);
    for (std::size_t position = 0; position != values.size(); ++position)
        st_verify_eq_(sequence->select(position), values[position]);
}

/** Decreasing values, values past the universe, a short fill, a failing fill and a refusing
 *  allocator are all reported, and nothing is built. */
void elias_fano_refusals() {
    std::uint64_t const decreasing[3] = {5, 9, 8};
    st_verify_eq_(sequence_t::make(std::span<std::uint64_t const>(decreasing)).status(), invalid_argument_k);

    std::uint64_t const values[3] = {1, 2, 30};
    auto pull_all = [&](std::size_t first) noexcept {
        return expected<std::span<std::uint64_t const>>(std::span<std::uint64_t const>(values).subspan(first),
                                                        success_k);
    };
    st_verify_eq_(sequence_t::make(3, 29, pull_all).status(), invalid_argument_k);
    st_verify_eq_(sequence_t::make(4, 30, pull_all).status(), invalid_argument_k);
    auto failing = [](std::size_t) noexcept { return expected<std::span<std::uint64_t const>>(input_output_k); };
    st_verify_eq_(sequence_t::make(3, 30, failing).status(), input_output_k);

    using refused_t = immutable_elias_fano<std::uint64_t, refusing_allocator<std::uint64_t>>;
    st_verify_eq_(refused_t::make(std::span<std::uint64_t const>(values)).status(), out_of_memory_heap_k);
}

/** The space stays within the Elias-Fano bound - the low part and at most three unary bits per
 *  value, which is 2 + ceil(log2(universe / count)), plus a quarter bit of samples - for the gaps
 *  of variable-length rows. */
void elias_fano_space_bound(test_context_t const &context) {
    std::mt19937_64 generator(mix_seed(context.seed, __func__));
    std::size_t const count = std::size_t(1) << 16;
    std::vector<std::uint64_t> values(count);
    std::uint64_t running = 0;
    for (std::uint64_t &value : values) value = running += 200 + generator() % 400;
    auto sequence = sequence_t::make(std::span<std::uint64_t const>(values));
    st_verify_(sequence);
    std::size_t const low_bits = std::bit_width(values.back() / count) - 1;
    // Three words round the three arrays up, and seven pad the high parts for the kits' scans.
    std::size_t const bound_bits = count * (3 + low_bits) + count / 4 + (3 + 7) * 64;
    st_verify_le_(sequence->size_bytes() * 8, bound_bits);
}

using packed_t = immutable_packed_array<std::uint64_t>;

/** Pulls @p values @p chunk at a time, the way a run streams its stamps in from storage. */
template <typename value_type_>
static auto pull_in_chunks(std::span<value_type_ const> values, std::size_t chunk) noexcept {
    return [values, chunk](std::size_t first) noexcept {
        std::size_t const pulled = values.size() - first < chunk ? values.size() - first : chunk;
        return expected<std::span<value_type_ const>>(values.subspan(first, pulled), success_k);
    };
}

/** The bytes @p count fields of @p bits bits take, rounded up to whole words. */
static std::size_t packed_bytes(std::size_t count, unsigned bits) noexcept {
    return (std::uint64_t(count) * bits + 63) / 64 * sizeof(std::uint64_t);
}

/** Saves @p array @p offset bytes into a buffer, so the view reads it at that alignment, and checks
 *  that the view answers every position as the array does. */
template <typename array_type_>
static void verify_saved_view(array_type_ const &array, std::size_t offset) {
    using view_t = immutable_packed_array_view<typename array_type_::value_t>;
    std::vector<std::byte> bytes(offset + array.saved_bytes());
    st_verify_eq_(array.save(std::span<std::byte>(bytes).subspan(offset)), success_k);
    auto view = view_t::make(std::span<std::byte const>(bytes).subspan(offset));
    st_verify_(view);
    st_verify_eq_(view->size(), array.size());
    st_verify_eq_(view->bits(), array.bits());
    st_verify_eq_(view->base(), array.base());
    st_verify_eq_(view->saved_bytes(), array.saved_bytes());
    for (std::size_t position = 0; position != array.size(); ++position)
        st_verify_eq_((*view)[position], array[position]);
}

/** Empty and one-element arrays, a zero width that stores no words, the full 64 bits, and fields
 *  straddling a word boundary all read back. */
void packed_array_edges() {
    auto empty = packed_t::make(std::span<std::uint64_t const> {});
    st_verify_(empty);
    st_verify_(empty->empty());
    st_verify_eq_(empty->bits(), 0u);
    st_verify_eq_(empty->size_bytes(), 0u);
    verify_saved_view(*empty, 0);
    auto never_pulled = [](std::size_t) noexcept { return expected<std::span<std::uint64_t const>>(input_output_k); };
    auto empty_pulled = packed_t::make(0, 5, 9, never_pulled);
    st_verify_(empty_pulled);
    st_verify_eq_(empty_pulled->bits(), 0u);
    st_verify_eq_(empty_pulled->base(), 0u);

    std::uint64_t const greatest = std::numeric_limits<std::uint64_t>::max();
    for (std::uint64_t const value : {std::uint64_t(0), std::uint64_t(7), greatest}) {
        std::uint64_t const values[1] = {value};
        auto single = packed_t::make(std::span<std::uint64_t const>(values));
        st_verify_(single);
        st_verify_eq_(single->bits(), 0u);
        st_verify_eq_(single->size_bytes(), 0u);
        st_verify_eq_((*single)[0], value);
        auto widest = packed_t::make(1, 0, greatest, pull_in_chunks(std::span<std::uint64_t const>(values), 1));
        st_verify_(widest);
        st_verify_eq_(widest->bits(), 64u);
        st_verify_eq_(widest->size_bytes(), sizeof(std::uint64_t));
        st_verify_eq_((*widest)[0], value);
    }

    std::vector<std::uint64_t> const flat(1000, 42);
    auto constant = packed_t::make(std::span<std::uint64_t const>(flat));
    st_verify_(constant);
    st_verify_eq_(constant->size_bytes(), 0u, "a zero width stores no words");
    st_verify_eq_(constant->saved_bytes(), sizeof(packed_array_header_t));
    for (std::size_t position = 0; position != flat.size(); ++position) st_verify_eq_((*constant)[position], 42u);
    verify_saved_view(*constant, 3);

    // Both ends of the 64-bit range, alternating, so every field is all ones or all zeros.
    std::vector<std::uint64_t> extremes(129);
    for (std::size_t position = 0; position != extremes.size(); ++position)
        extremes[position] = position % 2 ? greatest - position / 2 : position / 2;
    auto full = packed_t::make(std::span<std::uint64_t const>(extremes));
    st_verify_(full);
    st_verify_eq_(full->bits(), 64u);
    for (std::size_t position = 0; position != extremes.size(); ++position)
        st_verify_eq_((*full)[position], extremes[position]);

    // Widths that do not divide 64 put fields across word boundaries, all ones to catch a lost bit.
    for (unsigned const bits : {3u, 7u, 13u, 33u, 63u}) {
        std::uint64_t const top = (std::uint64_t {1} << bits) - 1;
        std::vector<std::uint64_t> values(130);
        for (std::size_t position = 0; position != values.size(); ++position) values[position] = position % 3 ? top : 0;
        auto straddling = packed_t::make(std::span<std::uint64_t const>(values));
        st_verify_(straddling);
        st_verify_eq_(straddling->bits(), bits);
        st_verify_eq_(straddling->size_bytes(), packed_bytes(values.size(), bits));
        for (std::size_t position = 0; position != values.size(); ++position)
            st_verify_eq_((*straddling)[position], values[position]);
    }
}

/** Every width from 0 to 64 at random bases agrees with a plain vector, pulled whole or in chunks,
 *  and read back from bytes saved at any alignment. */
void packed_array_random_widths(test_context_t const &context) {
    std::mt19937_64 generator(mix_seed(context.seed, __func__));
    std::uint64_t const greatest = std::numeric_limits<std::uint64_t>::max();
    for (unsigned bits = 0; bits <= 64; ++bits)
        for (std::size_t const count :
             {std::size_t(1), std::size_t(2), std::size_t(63), std::size_t(64), std::size_t(65), std::size_t(1000)}) {
            std::uint64_t const spread = bits == 64 ? greatest : (std::uint64_t {1} << bits) - 1;
            std::uint64_t base = generator();
            if (base > greatest - spread) base = greatest - spread;
            std::uint64_t const maximum = base + spread;

            std::vector<std::uint64_t> values(count);
            for (std::uint64_t &value : values)
                value = base + (spread == greatest ? generator() : generator() % (spread + 1));
            values.front() = base;
            if (count > 1) values.back() = maximum;

            auto chunked = packed_t::make(
                count, base, maximum, pull_in_chunks(std::span<std::uint64_t const>(values), 1 + generator() % 100));
            auto whole = packed_t::make(std::span<std::uint64_t const>(values));
            st_verify_(chunked);
            st_verify_(whole);
            st_verify_eq_(chunked->bits(), bits);
            st_verify_eq_(chunked->base(), base);
            st_verify_eq_(chunked->size_bytes(), packed_bytes(count, bits));
            st_verify_eq_(whole->base(), base);
            // One value is its own base and maximum, so held whole it packs at zero width.
            st_verify_eq_(whole->bits(), count > 1 ? bits : 0u);
            for (std::size_t position = 0; position != count; ++position) {
                st_verify_eq_((*chunked)[position], values[position]);
                st_verify_eq_((*whole)[position], values[position]);
            }
            verify_saved_view(*chunked, generator() % sizeof(std::uint64_t));
        }
}

/** Narrower values pack at their own width, and a view at that width refuses a saved form wider
 *  than it can hold. */
void packed_array_narrow_values(test_context_t const &context) {
    std::mt19937_64 generator(mix_seed(context.seed, __func__));
    std::vector<std::uint32_t> values(5000);
    for (std::uint32_t &value : values) value = 1000 + static_cast<std::uint32_t>(generator() % 300000);
    auto array = immutable_packed_array<std::uint32_t>::make(std::span<std::uint32_t const>(values));
    st_verify_(array);
    for (std::size_t position = 0; position != values.size(); ++position)
        st_verify_eq_((*array)[position], values[position]);
    verify_saved_view(*array, 1);

    std::uint64_t const wide[2] = {0, std::uint64_t {1} << 40};
    auto saved = packed_t::make(std::span<std::uint64_t const>(wide));
    st_verify_(saved);
    std::vector<std::byte> bytes(saved->saved_bytes());
    st_verify_eq_(saved->save(std::span<std::byte>(bytes)), success_k);
    st_verify_eq_(immutable_packed_array_view<std::uint32_t>::make(std::span<std::byte const>(bytes)).status(),
                  invalid_argument_k);
}

/** Values outside the base and the maximum, a base above the maximum, a short or overlong fill, a
 *  failing fill, a refusing allocator, and truncated or malformed saved bytes are all reported. */
void packed_array_refusals() {
    std::uint64_t const values[3] = {10, 12, 30};
    auto pull_all = pull_in_chunks(std::span<std::uint64_t const>(values), 3);
    st_verify_(packed_t::make(3, 10, 30, pull_all));
    st_verify_eq_(packed_t::make(3, 11, 30, pull_all).status(), invalid_argument_k, "a value below the base");
    st_verify_eq_(packed_t::make(3, 10, 29, pull_all).status(), invalid_argument_k, "a value above the maximum");
    st_verify_eq_(packed_t::make(3, 31, 30, pull_all).status(), invalid_argument_k, "a base above the maximum");
    st_verify_eq_(packed_t::make(4, 10, 30, pull_all).status(), invalid_argument_k, "a fill falling short");
    st_verify_eq_(packed_t::make(2, 10, 30, pull_all).status(), invalid_argument_k, "a fill running over");
    auto failing = [](std::size_t) noexcept { return expected<std::span<std::uint64_t const>>(input_output_k); };
    st_verify_eq_(packed_t::make(3, 10, 30, failing).status(), input_output_k);

    using refused_t = immutable_packed_array<std::uint64_t, refusing_allocator<std::uint64_t>>;
    st_verify_eq_(refused_t::make(std::span<std::uint64_t const>(values)).status(), out_of_memory_heap_k);

    auto array = packed_t::make(std::span<std::uint64_t const>(values));
    st_verify_(array);
    std::vector<std::byte> bytes(array->saved_bytes());
    st_verify_eq_(array->save(std::span<std::byte>(bytes).first(bytes.size() - 1)), invalid_argument_k);
    st_verify_eq_(array->save(std::span<std::byte>(bytes)), success_k);
    using view_t = immutable_packed_array_view<std::uint64_t>;
    std::span<std::byte const> const saved(bytes);
    st_verify_(view_t::make(saved));
    st_verify_eq_(view_t::make(saved.first(sizeof(packed_array_header_t) - 1)).status(), invalid_argument_k);
    st_verify_eq_(view_t::make(saved.first(saved.size() - 1)).status(), invalid_argument_k);

    packed_array_header_t const too_wide {.count = 1, .base = 0, .bits = 65};
    std::vector<std::byte> malformed(sizeof(too_wide) + 2 * sizeof(std::uint64_t));
    std::memcpy(malformed.data(), &too_wide, sizeof(too_wide));
    st_verify_eq_(view_t::make(std::span<std::byte const>(malformed)).status(), invalid_argument_k);

    packed_array_header_t const wraps_narrow {.count = 1, .base = 200, .bits = 6};
    std::memcpy(malformed.data(), &wraps_narrow, sizeof(wraps_narrow));
    st_verify_eq_(immutable_packed_array_view<std::uint8_t>::make(std::span<std::byte const>(malformed)).status(),
                  invalid_argument_k);
}

#pragma endregion Tests

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::test;

int main(int, char **arguments) {
    settings_t const settings = read_settings(arguments[0]);
    install_test_signal_handlers();
    print(probe_machine());
    print(settings);
    test_tally_t tally;

    tally += run_test(settings, "elias_fano.edges", elias_fano_edges);
    tally += run_test(settings, "elias_fano.serial_kit", elias_fano_serial_kit);
    tally += run_test(settings, "elias_fano.haswell_kit", elias_fano_haswell_kit);
    tally += run_test(settings, "elias_fano.skylake_kit", elias_fano_skylake_kit);
    tally += run_test(settings, "elias_fano.icelake_kit", elias_fano_icelake_kit);
    tally += run_test(settings, "elias_fano.neon_kit", elias_fano_neon_kit);
    tally += run_test(settings, "elias_fano.sve_kit", elias_fano_sve_kit);
    tally += run_test(settings, "elias_fano.rvv_kit", elias_fano_rvv_kit);
    tally += run_test(settings, "elias_fano.v128_kit", elias_fano_v128_kit);
    tally += run_test(settings, "elias_fano.narrow_values", elias_fano_narrow_values);
    tally += run_test(settings, "elias_fano.refusals", elias_fano_refusals);
    tally += run_test(settings, "elias_fano.space_bound", elias_fano_space_bound);
    tally += run_test(settings, "packed_array.edges", packed_array_edges);
    tally += run_test(settings, "packed_array.random_widths", packed_array_random_widths);
    tally += run_test(settings, "packed_array.narrow_values", packed_array_narrow_values);
    tally += run_test(settings, "packed_array.refusals", packed_array_refusals);

    return report_test_failures(settings, tally);
}
