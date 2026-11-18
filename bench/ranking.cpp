/**
 *  @file bench/ranking.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Benchmark of ranking a key among sorted ones: by the row kits within one row of each
 *      medium width, and by the static B-tree, the S+ tree and a binary search among all of them.
 *
 *  Prints the machine and the settings, then one row per benchmark in two phases: the kits over
 *  single rows of each medium width, and both layouts over the same sorted keys. The settings it
 *  reads are listed in CONTRIBUTING.md.
 */
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`, `std::uintptr_t`
#include <cstring> // `std::strcmp`

#include <algorithm>   // `std::lower_bound`, `std::sort`
#include <chrono>      // `std::chrono::steady_clock`
#include <concepts>    // `std::same_as`
#include <format>      // `std::format`
#include <span>        // `std::span`
#include <string>      // `std::string`
#include <type_traits> // `std::true_type`

#include <forkunion.hpp> // `fu::portable_aligned_allocator`

#include <smashtable/basic_vector.hpp>
#include <smashtable/row_search.hpp>
#include <smashtable/immutable_b_tree.hpp>
#include <smashtable/immutable_splus_tree.hpp>

#include "harness.hpp"

namespace ashvardanian::smashtable::bench {

namespace fu = ashvardanian::forkunion;

#pragma region Workloads

/** How the 16-byte keys are drawn: independent words, or integer identities whose high words all
 *  tie. */
enum class key_draw_t : std::uint8_t {
    uniform_k,
    integer_identities_k,
};

template <typename key_type_>
[[nodiscard]] key_type_ draw_key(split_mix_generator_t &generator, key_draw_t draw) noexcept {
    if constexpr (std::same_as<key_type_, key128_t>)
        return draw == key_draw_t::uniform_k ? key128_t {generator(), generator()} : key128_t {0, generator() >> 8};
    else return static_cast<key_type_>(generator());
}

/** Calls @p search with each of @p queries_count query indices in turn until @p loop ends. */
template <typename search_type_>
void search_queries(loop_t &loop, std::size_t queries_count, search_type_ &&search) noexcept {
    std::size_t checksum = 0, query = 0;
    for ([[maybe_unused]] std::size_t call : loop) {
        checksum += search(query);
        query = query + 1 == queries_count ? 0 : query + 1;
    }
    do_not_optimize(checksum);
}

#pragma endregion Workloads

#pragma region Row Kits Phase

/** Times the kit over interleaved rows, or over split columns when @p high_words holds them. */
template <typename kit_type_, typename key_type_, std::size_t keys_per_row_>
void time_rows(loop_t &loop, std::span<key_type_ const> rows, std::span<std::uint64_t const> high_words,
               std::span<std::uint64_t const> low_words, std::span<key_type_ const> queries) noexcept {
    std::size_t const rows_count = rows.size() / keys_per_row_;
    search_queries(loop, queries.size(), [&](std::size_t query) noexcept {
        std::size_t const row = query % rows_count * keys_per_row_;
        if constexpr (std::same_as<key_type_, key128_t>)
            if (!high_words.empty())
                return kit_type_::count_below(
                    std::span<std::uint64_t const, keys_per_row_>(high_words.data() + row, keys_per_row_),
                    std::span<std::uint64_t const, keys_per_row_>(low_words.data() + row, keys_per_row_),
                    queries[query]);
        return kit_type_::count_below(std::span<key_type_ const, keys_per_row_>(rows.data() + row, keys_per_row_),
                                      queries[query]);
    });
}

template <typename key_type_, std::size_t keys_per_row_>
void bench_rows_of(environment_t const &env, char const *label, key_draw_t draw, char const *row_layout) {
    bench_settings_t const &settings = env.settings;
    std::string const prefix =
        std::format("ranking.rows.{}b.{}.{}.", sizeof(key_type_) * keys_per_row_, label, row_layout);
    split_mix_generator_t generator {stream_key(settings.seed, prefix)};
    std::size_t const keys_count = settings.rows * keys_per_row_;
    basic_vector<key_type_> rows;
    basic_vector<key_type_> queries;
    basic_vector<std::uint64_t> high_words;
    basic_vector<std::uint64_t> low_words;
    if (failed(rows.resize(keys_count)) || failed(queries.resize(settings.queries))) return;
    for (key_type_ &key : rows) key = draw_key<key_type_>(generator, draw);
    for (std::size_t row = 0; row < settings.rows; ++row)
        std::sort(rows.data() + row * keys_per_row_, rows.data() + (row + 1) * keys_per_row_);
    for (std::size_t query = 0; query < settings.queries; ++query)
        queries[query] = rows[(query % settings.rows) * keys_per_row_ + generator() % keys_per_row_];
    bool const splits = std::strcmp(row_layout, "split") == 0;
    if constexpr (std::same_as<key_type_, key128_t>)
        if (splits) {
            if (failed(high_words.resize(keys_count)) || failed(low_words.resize(keys_count))) return;
            for (std::size_t index = 0; index < keys_count; ++index) {
                high_words[index] = rows[index].high;
                low_words[index] = rows[index].low;
            }
        }

    std::span<key_type_ const> const row_span(rows.data(), rows.size());
    std::span<std::uint64_t const> const high_span(high_words.data(), high_words.size());
    std::span<std::uint64_t const> const low_span(low_words.data(), low_words.size());
    std::span<key_type_ const> const query_span(queries.data(), queries.size());

    run(env, prefix + "binary", 0, [&](loop_t &loop) noexcept {
        search_queries(loop, settings.queries, [&](std::size_t query) noexcept {
            key_type_ const *const row = rows.data() + query % settings.rows * keys_per_row_;
            return static_cast<std::size_t>(std::lower_bound(row, row + keys_per_row_, queries[query]) - row);
        });
    });
    double serial = 0;
    for (row_kit_t const kit : every_row_kit_k) {
        if (!row_kit_supported(kit)) continue;
        double const calls_per_second = visit_row_kit(kit, [&]<typename kit_type_>(kit_type_) noexcept {
            return run(env, prefix + name_of(kit), serial, [&](loop_t &loop) noexcept {
                time_rows<kit_type_, key_type_, keys_per_row_>(loop, row_span, high_span, low_span, query_span);
            });
        });
        if (kit == row_kit_t::serial_k) serial = calls_per_second;
    }
}

/** The row widths the sweep walks: two common cache lines, a block, and a common page. */
constexpr std::size_t swept_row_bytes_k[] = {64u, 128u, 512u, 4096u};

void bench_row_kits(environment_t const &env) {
    print_line(stdout, "\nRow kits over {} sorted rows, the speed-up against serial last", env.settings.rows);
    for (std::size_t const row_bytes : swept_row_bytes_k) {
        switch (row_bytes) {
        case 64u:
            bench_rows_of<std::uint32_t, 16>(env, "u32", key_draw_t::uniform_k, "column");
            bench_rows_of<std::uint64_t, 8>(env, "u64", key_draw_t::uniform_k, "column");
            bench_rows_of<std::int64_t, 8>(env, "i64", key_draw_t::uniform_k, "column");
            bench_rows_of<key128_t, 4>(env, "key128.uniform", key_draw_t::uniform_k, "split");
            bench_rows_of<key128_t, 4>(env, "key128.identities", key_draw_t::integer_identities_k, "split");
            bench_rows_of<key128_t, 4>(env, "key128.uniform", key_draw_t::uniform_k, "interleaved");
            break;
        case 128u:
            bench_rows_of<std::uint32_t, 32>(env, "u32", key_draw_t::uniform_k, "column");
            bench_rows_of<std::uint64_t, 16>(env, "u64", key_draw_t::uniform_k, "column");
            bench_rows_of<std::int64_t, 16>(env, "i64", key_draw_t::uniform_k, "column");
            bench_rows_of<key128_t, 8>(env, "key128.uniform", key_draw_t::uniform_k, "split");
            bench_rows_of<key128_t, 8>(env, "key128.identities", key_draw_t::integer_identities_k, "split");
            bench_rows_of<key128_t, 8>(env, "key128.uniform", key_draw_t::uniform_k, "interleaved");
            break;
        case 512u:
            bench_rows_of<std::uint32_t, 128>(env, "u32", key_draw_t::uniform_k, "column");
            bench_rows_of<std::uint64_t, 64>(env, "u64", key_draw_t::uniform_k, "column");
            bench_rows_of<std::int64_t, 64>(env, "i64", key_draw_t::uniform_k, "column");
            bench_rows_of<key128_t, 32>(env, "key128.uniform", key_draw_t::uniform_k, "split");
            bench_rows_of<key128_t, 32>(env, "key128.identities", key_draw_t::integer_identities_k, "split");
            bench_rows_of<key128_t, 32>(env, "key128.uniform", key_draw_t::uniform_k, "interleaved");
            break;
        case 4096u:
            bench_rows_of<std::uint32_t, 1024>(env, "u32", key_draw_t::uniform_k, "column");
            bench_rows_of<std::uint64_t, 512>(env, "u64", key_draw_t::uniform_k, "column");
            bench_rows_of<std::int64_t, 512>(env, "i64", key_draw_t::uniform_k, "column");
            bench_rows_of<key128_t, 256>(env, "key128.uniform", key_draw_t::uniform_k, "split");
            bench_rows_of<key128_t, 256>(env, "key128.identities", key_draw_t::integer_identities_k, "split");
            bench_rows_of<key128_t, 256>(env, "key128.uniform", key_draw_t::uniform_k, "interleaved");
            break;
        default: break;
        }
    }
}

#pragma endregion Row Kits Phase

#pragma region Layouts Phase

template <typename layout_type_>
void bench_layout(environment_t const &env, std::string const &prefix, char const *layout_name,
                  std::span<typename layout_type_::key_t const> sorted,
                  std::span<typename layout_type_::key_t const> queries, double binary) {
    std::string const name =
        std::format("{}{}b.{}.{}", prefix, layout_type_::keys_per_row_k * sizeof(typename layout_type_::key_t),
                    layout_name, name_of(layout_type_::kit_t::kit_k));
    if (!env.settings.regex_filter.selects(name)) return;
    auto const built_at = steady_clock_t::now();
    auto layout = layout_type_::make(sorted);
    double const build_seconds = std::chrono::duration<double>(steady_clock_t::now() - built_at).count();
    run(env, name, binary, [&](loop_t &loop) noexcept {
        if (!layout) return loop.skip(name_of(layout.status()));
        if (reinterpret_cast<std::uintptr_t>(layout->words().data()) % cache_line_bytes_k != 0)
            return loop.skip("rows start off the cache line");
        loop.counter("MB", static_cast<double>(layout->size_bytes()) / (1 << 20));
        loop.counter("build seconds", build_seconds);
        search_queries(loop, queries.size(), [&](std::size_t query) noexcept { return layout->rank(queries[query]); });
    });
}

template <typename kit_type_, typename key_type_, std::size_t keys_per_row_>
void bench_layouts_with(environment_t const &env, std::string const &prefix, std::span<key_type_ const> sorted,
                        std::span<key_type_ const> queries, double binary) {
    using allocator_t = fu::portable_aligned_allocator<key_type_>;
    bench_layout<immutable_b_tree<key_type_, keys_per_row_, kit_type_, allocator_t>>(env, prefix, "b_tree", sorted,
                                                                                     queries, binary);
    bench_layout<immutable_splus_tree<key_type_, keys_per_row_, kit_type_, allocator_t>>(env, prefix, "splus_tree",
                                                                                         sorted, queries, binary);
}

template <typename key_type_>
void bench_layouts_of(environment_t const &env, char const *label, key_draw_t draw) {
    bench_settings_t const &settings = env.settings;
    std::string const prefix = std::format("ranking.layouts.{}.", label);
    // Two copies of the keys plus the largest tree, which stays under a tenth of the keys again.
    split_mix_generator_t generator {stream_key(settings.seed, prefix)};
    basic_vector<key_type_> sorted;
    basic_vector<key_type_> queries;
    if (failed(sorted.resize(settings.keys)) || failed(queries.resize(settings.queries))) {
        print_line(stdout, "{:<64} skipped: needs memory for {} keys", prefix + "binary", settings.keys);
        return;
    }
    for (key_type_ &key : sorted) key = draw_key<key_type_>(generator, draw);
    std::sort(sorted.begin(), sorted.end());
    for (key_type_ &query : queries) query = sorted[generator() % settings.keys];
    std::span<key_type_ const> const sorted_span(sorted.data(), sorted.size());
    std::span<key_type_ const> const query_span(queries.data(), queries.size());

    double const binary = run(env, prefix + "binary", 0, [&](loop_t &loop) noexcept {
        search_queries(loop, settings.queries, [&](std::size_t query) noexcept {
            return static_cast<std::size_t>(std::lower_bound(sorted.begin(), sorted.end(), queries[query]) -
                                            sorted.begin());
        });
    });
    for (row_kit_t const kit : every_row_kit_k) {
        if (!row_kit_supported(kit)) continue;
        visit_row_kit(kit, [&]<typename kit_type_>(kit_type_) noexcept {
            if constexpr (std::same_as<key_type_, key128_t>) {
                bench_layouts_with<kit_type_, key128_t, keys_per_row<key128_t>(64u)>(env, prefix, sorted_span,
                                                                                     query_span, binary);
                bench_layouts_with<kit_type_, key128_t, keys_per_row<key128_t>(128u)>(env, prefix, sorted_span,
                                                                                      query_span, binary);
                bench_layouts_with<kit_type_, key128_t, keys_per_row<key128_t>(512u)>(env, prefix, sorted_span,
                                                                                      query_span, binary);
                bench_layouts_with<kit_type_, key128_t, keys_per_row<key128_t>(4096u)>(env, prefix, sorted_span,
                                                                                       query_span, binary);
            }
            else {
                bench_layouts_with<kit_type_, key_type_, keys_per_row<key_type_>(64u)>(env, prefix, sorted_span,
                                                                                       query_span, binary);
                bench_layouts_with<kit_type_, key_type_, keys_per_row<key_type_>(128u)>(env, prefix, sorted_span,
                                                                                        query_span, binary);
                bench_layouts_with<kit_type_, key_type_, keys_per_row<key_type_>(512u)>(env, prefix, sorted_span,
                                                                                        query_span, binary);
                bench_layouts_with<kit_type_, key_type_, keys_per_row<key_type_>(4096u)>(env, prefix, sorted_span,
                                                                                         query_span, binary);
            }
        });
    }
}

void bench_layouts(environment_t const &env) {
    print_line(stdout, "\nLayouts over {} sorted keys, queries drawn from the keys, the speed-up against binary last",
               env.settings.keys);
    bench_layouts_of<std::uint32_t>(env, "u32", key_draw_t::uniform_k);
    bench_layouts_of<std::uint64_t>(env, "u64", key_draw_t::uniform_k);
    bench_layouts_of<key128_t>(env, "key128.uniform", key_draw_t::uniform_k);
    bench_layouts_of<key128_t>(env, "key128.identities", key_draw_t::integer_identities_k);
}

#pragma endregion Layouts Phase

} // namespace ashvardanian::smashtable::bench

using namespace ashvardanian::smashtable::bench;

int main(int argc, char **argv) {
    environment_t const env {read_bench_settings(argc, argv), probe_machine()};
    print(env.machine);
    print(env.settings);
    bench_row_kits(env);
    bench_layouts(env);
    return 0;
}
