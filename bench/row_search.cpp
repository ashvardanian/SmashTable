/**
 *  @file bench/row_search.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Benchmark of the row kits per medium width, and of the static B-tree against the S+ tree
 *      and a binary search.
 *
 *  Prints the kits compiled and runnable, a machine block, then one block per phase: the kits over
 *  single rows of each medium width, and both layouts over the same sorted keys. The `--rows`,
 *  `--keys` and `--queries` flags all take counts.
 */
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstdio>  // `std::fopen`, `std::fgets`
#include <cstring> // `std::strcmp`, `std::strncmp`

#include <algorithm>    // `std::lower_bound`, `std::sort`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`
#include <concepts>     // `std::same_as`
#include <random>       // `std::mt19937_64`
#include <span>         // `std::span`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`

#include <smashtable/basic_vector.hpp>
#include <smashtable/row_search.hpp>
#include <smashtable/immutable_b_tree.hpp>
#include <smashtable/immutable_splus_tree.hpp>

#include "harness.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::test;

namespace {

#pragma region Options

struct options_t {
    std::size_t rows {4096};
    std::size_t keys {std::size_t {1} << 24};
    std::size_t queries {std::size_t {1} << 20};
};

[[nodiscard]] bool parse_count(char const *text, std::size_t &count) noexcept {
    std::string_view const view(text);
    auto const [end, error] = std::from_chars(view.data(), view.data() + view.size(), count);
    return error == std::errc {} && end == view.data() + view.size();
}

[[nodiscard]] bool parse_options(int arguments_count, char **arguments, options_t &options) noexcept {
    for (int index = 1; index + 1 < arguments_count; index += 2) {
        std::string_view const flag(arguments[index]);
        std::size_t *const target = flag == "--rows"      ? &options.rows
                                    : flag == "--keys"    ? &options.keys
                                    : flag == "--queries" ? &options.queries
                                                          : nullptr;
        if (!target || !parse_count(arguments[index + 1], *target)) return false;
    }
    return arguments_count % 2 == 1;
}

#pragma endregion Options

#pragma region Machine

/** Copies the processor's model name into @p model, or leaves it unknown where the system does not
 *  say. */
void read_cpu_model(std::span<char> model) {
    std::snprintf(model.data(), model.size(), "unknown");
    std::FILE *const cpuinfo = std::fopen("/proc/cpuinfo", "r");
    if (!cpuinfo) return;
    char line[512];
    while (std::fgets(line, sizeof(line), cpuinfo)) {
        char const *const colon = std::strchr(line, ':');
        if (!colon || (std::strncmp(line, "model name", 10) != 0 && std::strncmp(line, "uarch", 5) != 0 &&
                       std::strncmp(line, "CPU part", 8) != 0))
            continue;
        std::snprintf(model.data(), model.size(), "%s", colon + 2);
        model[std::strcspn(model.data(), "\n")] = '\0';
        if (std::strncmp(line, "model name", 10) == 0) break;
    }
    std::fclose(cpuinfo);
}

void print_machine(options_t const &options) {
    char model[256];
    read_cpu_model(model);
    print_line(stdout, "Machine");
    print_line(stdout, "  cpu:            {}", static_cast<char const *>(model)); // ? The array would print whole
#if defined(__clang__)
    print_line(stdout, "  compiler:       Clang {}", __clang_version__);
#elif defined(__GNUC__)
    print_line(stdout, "  compiler:       GCC {}", __VERSION__);
#else
    print_line(stdout, "  compiler:       unrecognized");
#endif
    print_line(stdout, "  kit detected:   {}", name_of(detect_row_kit()));
    print_line(stdout, "  rows:           {}", options.rows);
    print_line(stdout, "  keys:           {}", options.keys);
    print_line(stdout, "  queries:        {}", options.queries);
}

#pragma endregion Machine

#pragma region Workloads

/** How the 16-byte keys are drawn: independent words, or integer identities whose high words all
 *  tie. */
enum class key_draw_t : std::uint8_t {
    uniform_k,
    integer_identities_k,
};

template <typename key_type_>
[[nodiscard]] key_type_ draw_key(std::mt19937_64 &generator, key_draw_t draw) noexcept {
    if constexpr (std::same_as<key_type_, key128_t>)
        return draw == key_draw_t::uniform_k ? key128_t {generator(), generator()} : key128_t {0, generator() >> 8};
    else return static_cast<key_type_>(generator());
}

/** Seconds spent per call of @p body over @p calls calls, after a short warm-up. */
template <typename body_type_>
[[nodiscard]] double nanoseconds_per_call(std::size_t calls, body_type_ &&body) noexcept {
    std::size_t checksum = 0;
    for (std::size_t call = 0; call < calls / 16 + 1; ++call) checksum += body(call);
    auto const started = std::chrono::steady_clock::now();
    for (std::size_t call = 0; call < calls; ++call) checksum += body(call);
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    [[maybe_unused]] volatile std::size_t const kept = checksum;
    return seconds * 1e9 / static_cast<double>(calls);
}

#pragma endregion Workloads

#pragma region Row Kits Phase

/** Times the kit over interleaved rows, or over split columns when @p high_words holds them. */
template <typename kit_type_, typename key_type_, std::size_t keys_per_row_>
[[nodiscard]] double time_rows(std::span<key_type_ const> rows, std::span<std::uint64_t const> high_words,
                               std::span<std::uint64_t const> low_words, std::span<key_type_ const> queries) noexcept {
    std::size_t const rows_count = rows.size() / keys_per_row_;
    return nanoseconds_per_call(queries.size(), [&](std::size_t call) noexcept {
        std::size_t const row = call % rows_count * keys_per_row_;
        if constexpr (std::same_as<key_type_, key128_t>)
            if (!high_words.empty())
                return kit_type_::count_below(
                    std::span<std::uint64_t const, keys_per_row_>(high_words.data() + row, keys_per_row_),
                    std::span<std::uint64_t const, keys_per_row_>(low_words.data() + row, keys_per_row_),
                    queries[call]);
        return kit_type_::count_below(std::span<key_type_ const, keys_per_row_>(rows.data() + row, keys_per_row_),
                                      queries[call]);
    });
}

template <typename key_type_, std::size_t keys_per_row_>
void bench_rows_of(char const *label, key_draw_t draw, char const *row_layout, options_t const &options) {
    std::mt19937_64 generator(42);
    std::size_t const keys_count = options.rows * keys_per_row_;
    basic_vector<key_type_> rows;
    basic_vector<key_type_> queries;
    basic_vector<std::uint64_t> high_words;
    basic_vector<std::uint64_t> low_words;
    if (failed(rows.resize(keys_count)) || failed(queries.resize(options.queries))) return;
    for (key_type_ &key : rows) key = draw_key<key_type_>(generator, draw);
    for (std::size_t row = 0; row < options.rows; ++row)
        std::sort(rows.data() + row * keys_per_row_, rows.data() + (row + 1) * keys_per_row_);
    for (std::size_t query = 0; query < options.queries; ++query)
        queries[query] = rows[(query % options.rows) * keys_per_row_ + generator() % keys_per_row_];
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

    double const binary = nanoseconds_per_call(options.queries, [&](std::size_t call) noexcept {
        key_type_ const *const row = rows.data() + call % options.rows * keys_per_row_;
        return static_cast<std::size_t>(std::lower_bound(row, row + keys_per_row_, queries[call]) - row);
    });
    double serial = 0;
    std::printf("  %-22s %-12s %5zu  %-10s %8.2f\n", label, row_layout, keys_per_row_, "binary", binary);
    for (row_kit_t const kit : every_row_kit_k) {
        if (!row_kit_supported(kit)) continue;
        double const elapsed = visit_row_kit(kit, [&]<typename kit_type_>(kit_type_) noexcept {
            return time_rows<kit_type_, key_type_, keys_per_row_>(row_span, high_span, low_span, query_span);
        });
        if (kit == row_kit_t::serial_k) serial = elapsed;
        std::printf("  %-22s %-12s %5zu  %-10s %8.2f  %5.2fx of serial\n", label, row_layout, keys_per_row_,
                    name_of(kit), elapsed, serial / elapsed);
    }
}

/** The row widths the sweep walks: a common cache line, a block, and a common page. */
constexpr std::size_t swept_row_bytes_k[] = {64u, 512u, 4096u};

void bench_row_kits(options_t const &options) {
    print_line(stdout, "\nPhase: row kits, nanoseconds per row search over {} sorted rows", options.rows);
    std::printf("  %-22s %-12s %5s  %-10s %8s\n", "key", "row layout", "keys", "kit", "ns/row");
    for (std::size_t const row_bytes : swept_row_bytes_k) {
        print_line(stdout, "  row: {} bytes", row_bytes);
        switch (row_bytes) {
        case 64u:
            bench_rows_of<std::uint32_t, 16>("u32", key_draw_t::uniform_k, "column", options);
            bench_rows_of<std::uint64_t, 8>("u64", key_draw_t::uniform_k, "column", options);
            bench_rows_of<std::int64_t, 8>("i64", key_draw_t::uniform_k, "column", options);
            bench_rows_of<key128_t, 4>("key128 uniform", key_draw_t::uniform_k, "split", options);
            bench_rows_of<key128_t, 4>("key128 identities", key_draw_t::integer_identities_k, "split", options);
            bench_rows_of<key128_t, 4>("key128 uniform", key_draw_t::uniform_k, "interleaved", options);
            break;
        case 512u:
            bench_rows_of<std::uint32_t, 128>("u32", key_draw_t::uniform_k, "column", options);
            bench_rows_of<std::uint64_t, 64>("u64", key_draw_t::uniform_k, "column", options);
            bench_rows_of<std::int64_t, 64>("i64", key_draw_t::uniform_k, "column", options);
            bench_rows_of<key128_t, 32>("key128 uniform", key_draw_t::uniform_k, "split", options);
            bench_rows_of<key128_t, 32>("key128 identities", key_draw_t::integer_identities_k, "split", options);
            bench_rows_of<key128_t, 32>("key128 uniform", key_draw_t::uniform_k, "interleaved", options);
            break;
        case 4096u:
            bench_rows_of<std::uint32_t, 1024>("u32", key_draw_t::uniform_k, "column", options);
            bench_rows_of<std::uint64_t, 512>("u64", key_draw_t::uniform_k, "column", options);
            bench_rows_of<std::int64_t, 512>("i64", key_draw_t::uniform_k, "column", options);
            bench_rows_of<key128_t, 256>("key128 uniform", key_draw_t::uniform_k, "split", options);
            bench_rows_of<key128_t, 256>("key128 identities", key_draw_t::integer_identities_k, "split", options);
            bench_rows_of<key128_t, 256>("key128 uniform", key_draw_t::uniform_k, "interleaved", options);
            break;
        default: break;
        }
    }
}

#pragma endregion Row Kits Phase

#pragma region Layouts Phase

template <typename layout_type_>
void bench_layout(char const *name, std::span<typename layout_type_::key_t const> sorted,
                  std::span<typename layout_type_::key_t const> queries, char const *label, double binary) {
    auto const built_at = std::chrono::steady_clock::now();
    auto layout = layout_type_::make(sorted);
    double const build_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - built_at).count();
    if (!layout) {
        print_line(stdout, "  {} {}: build refused with {}", label, name, name_of(layout.status()));
        return;
    }
    double const elapsed =
        nanoseconds_per_call(queries.size(), [&](std::size_t call) noexcept { return layout->rank(queries[call]); });
    std::printf("  %-18s %5zu  %-11s %-8s %8.1f  %5.2fx of binary  %8.1f MiB  %5.2f s build\n", label,
                layout_type_::keys_per_row_k, name, name_of(layout_type_::kit_t::kit_k), elapsed, binary / elapsed,
                static_cast<double>(layout->size_bytes()) / (1 << 20), build_seconds);
}

template <typename kit_type_, typename key_type_, std::size_t keys_per_row_>
void bench_layouts_with(std::span<key_type_ const> sorted, std::span<key_type_ const> queries, char const *label,
                        double binary) {
    bench_layout<immutable_b_tree<key_type_, keys_per_row_, kit_type_>>("b-tree", sorted, queries, label, binary);
    bench_layout<immutable_splus_tree<key_type_, keys_per_row_, kit_type_>>("s+tree", sorted, queries, label, binary);
}

template <typename key_type_>
void bench_layouts_of(char const *label, key_draw_t draw, options_t const &options) {
    // Two copies of the keys plus the largest tree, which stays under a tenth of the keys again.
    std::mt19937_64 generator(7);
    basic_vector<key_type_> sorted;
    basic_vector<key_type_> queries;
    if (failed(sorted.resize(options.keys)) || failed(queries.resize(options.queries))) {
        print_line(stdout, "  {}: not enough memory for {} keys", label, options.keys);
        return;
    }
    for (key_type_ &key : sorted) key = draw_key<key_type_>(generator, draw);
    std::sort(sorted.begin(), sorted.end());
    for (key_type_ &query : queries) query = sorted[generator() % options.keys];
    std::span<key_type_ const> const sorted_span(sorted.data(), sorted.size());
    std::span<key_type_ const> const query_span(queries.data(), queries.size());

    double const binary = nanoseconds_per_call(options.queries, [&](std::size_t call) noexcept {
        return static_cast<std::size_t>(std::lower_bound(sorted.begin(), sorted.end(), queries[call]) - sorted.begin());
    });
    std::printf("  %-18s %5s  %-11s %-8s %8.1f\n", label, "-", "binary", "-", binary);
    for (row_kit_t const kit : every_row_kit_k) {
        if (!row_kit_supported(kit)) continue;
        visit_row_kit(kit, [&]<typename kit_type_>(kit_type_) noexcept {
            if constexpr (std::same_as<key_type_, key128_t>) {
                bench_layouts_with<kit_type_, key128_t, keys_per_row<key128_t>(64u)>(sorted_span, query_span, label,
                                                                                     binary);
                bench_layouts_with<kit_type_, key128_t, keys_per_row<key128_t>(512u)>(sorted_span, query_span, label,
                                                                                      binary);
                bench_layouts_with<kit_type_, key128_t, keys_per_row<key128_t>(4096u)>(sorted_span, query_span, label,
                                                                                       binary);
            }
            else {
                bench_layouts_with<kit_type_, key_type_, keys_per_row<key_type_>(64u)>(sorted_span, query_span, label,
                                                                                       binary);
                bench_layouts_with<kit_type_, key_type_, keys_per_row<key_type_>(512u)>(sorted_span, query_span, label,
                                                                                        binary);
                bench_layouts_with<kit_type_, key_type_, keys_per_row<key_type_>(4096u)>(sorted_span, query_span, label,
                                                                                         binary);
            }
        });
    }
}

void bench_layouts(options_t const &options) {
    print_line(stdout, "\nPhase: layouts, nanoseconds per rank over {} sorted keys, queries drawn from the keys",
               options.keys);
    std::printf("  %-18s %5s  %-11s %-8s %8s\n", "key", "row", "layout", "kit", "ns/rank");
    bench_layouts_of<std::uint64_t>("u64", key_draw_t::uniform_k, options);
    bench_layouts_of<key128_t>("key128 uniform", key_draw_t::uniform_k, options);
    bench_layouts_of<key128_t>("key128 identities", key_draw_t::integer_identities_k, options);
}

#pragma endregion Layouts Phase

} // namespace

int main(int arguments_count, char **arguments) {
    options_t options;
    if (!parse_options(arguments_count, arguments, options) || options.rows == 0 || options.keys == 0 ||
        options.queries == 0) {
        std::fprintf(stderr, "Usage: %s [--rows <count>] [--keys <count>] [--queries <count>]\n", arguments[0]);
        return 1;
    }
    print_row_kits("Compiled for", row_kit_compiled);
    print_row_kits("This machine", row_kit_supported);
    print_machine(options);
    bench_row_kits(options);
    bench_layouts(options);
    return 0;
}
