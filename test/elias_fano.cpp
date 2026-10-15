/**
 *  @file test/elias_fano.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Tests for @c immutable_elias_fano: every value reads back by position, whole or pulled
 *      in chunks, the refusals hold, and the space stays near the Elias-Fano bound.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <bit>    // `std::bit_width`
#include <limits> // `std::numeric_limits`
#include <random> // `std::mt19937_64`
#include <span>   // `std::span`
#include <vector> // `std::vector`

#include "smashtable/immutable_elias_fano.hpp"

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
    tally += run_test(settings, "elias_fano.narrow_values", elias_fano_narrow_values);
    tally += run_test(settings, "elias_fano.refusals", elias_fano_refusals);
    tally += run_test(settings, "elias_fano.space_bound", elias_fano_space_bound);

    return report_test_failures(settings, tally);
}
