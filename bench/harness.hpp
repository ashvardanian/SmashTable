/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Harness for the benchmarks - settings, the machine, and the timed loop with its rows.
 *
 *  Environment variables, read once by @c read_settings:
 *
 *  @verbatim
 *  SMASHTABLE_FILTER      unset     ECMAScript regex over benchmark names, or a substring when it does not compile
 *  SMASHTABLE_SEED        42        Seed for the key draws, or "random" to draw one
 *  SMASHTABLE_WARMUP      1s        Untimed run ahead of each benchmark, "<int>ms" or "<int>s"
 *  SMASHTABLE_TIME_LIMIT  10s       Timed run of each benchmark, "<int>ms" or "<int>s"
 *  SMASHTABLE_ROWS        4096      Sorted rows per row-kit benchmark
 *  SMASHTABLE_KEYS        16777216  Sorted keys per layout benchmark
 *  SMASHTABLE_QUERIES     1048576   Queries drawn per benchmark, searched in turn
 *  @endverbatim
 */
#pragma once
#include <cctype>  // `std::tolower`
#include <cstddef> // `std::ptrdiff_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `std::FILE`, `std::fputc`, `std::fopen`, `std::fgets`
#include <cstdlib> // `std::exit`, `std::getenv`
#include <cstring> // `std::strchr`, `std::strncmp`, `std::strcspn`

#include <array>        // `std::array`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <format>       // `std::format_to`, `std::format_string`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`
#include <regex>        // `std::regex`, `std::regex_search`
#include <string>       // `std::string`, `std::to_string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <utility>      // `std::exchange`, `std::forward`, `std::move`

#include <smashtable/row_search.hpp> // `every_row_kit_k`, `row_kit_t`, `name_of`

namespace ashvardanian::smashtable::bench {

using steady_clock_t = std::chrono::steady_clock;
using time_point_t = steady_clock_t::time_point;

/** Output iterator handing each character straight to a @c std::FILE. */
struct file_output_iterator_t {
    using difference_type = std::ptrdiff_t;

    std::FILE *stream {};

    file_output_iterator_t &operator*() noexcept { return *this; }
    file_output_iterator_t &operator++() noexcept { return *this; }
    file_output_iterator_t operator++(int) noexcept { return *this; }
    file_output_iterator_t &operator=(char character) noexcept {
        std::fputc(character, stream);
        return *this;
    }
};

/**
 *  @brief Writes one formatted line to @p stream, checking the pattern against its arguments and
 *      terminating it here, so @p pattern carries no trailing newline of its own.
 *
 *  Formats straight into @p stream rather than into a @c std::string, so there is no allocation and
 *  no buffer to size.
 */
template <typename... args_types_>
inline void print_line(std::FILE *stream, std::format_string<args_types_...> pattern, args_types_ &&...args) noexcept {
    std::format_to(file_output_iterator_t {stream}, pattern, std::forward<args_types_>(args)...);
    std::fputc('\n', stream);
}

/** The text of the environment variable @p name, or nothing when it is unset or empty. */
inline std::optional<std::string_view> env_text(char const *name) noexcept {
    char const *const text = std::getenv(name);
    if (!text || !*text) return std::nullopt;
    return std::string_view(text);
}

/** Parses the environment variable @p name with @p parse, or returns @p fallback when it is unset
 *  or empty. Text that does not parse prints `NAME="text" does not parse, expected <expected>` and
 *  exits with status 1, which leaves crash handlers quiet. */
template <typename value_type_, typename parse_type_>
[[nodiscard]] value_type_ env_parsed(char const *name, value_type_ fallback, parse_type_ &&parse,
                                     char const *expected) noexcept {
    std::optional<std::string_view> const text = env_text(name);
    if (!text) return fallback;
    if (std::optional<value_type_> value = parse(*text)) return *std::move(value);
    print_line(stderr, "{}=\"{}\" does not parse, expected {}", name, *text, expected);
    std::exit(1);
}

/** A positive whole number, like "64". */
inline std::optional<std::size_t> parse_count(std::string_view text) noexcept {
    std::size_t count = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc {} || end != text.data() + text.size() || !count) return std::nullopt;
    return count;
}

/** A positive duration in whole milliseconds or seconds, like "200ms" or "10s". */
inline std::optional<std::chrono::milliseconds> parse_duration(std::string_view text) noexcept {
    std::uint32_t count = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc {} || !count) return std::nullopt;
    std::string_view const unit = text.substr(end - text.data());
    if (unit == "ms") return std::chrono::milliseconds(count);
    if (unit == "s") return std::chrono::seconds(count);
    return std::nullopt;
}

/** A 32-bit seed, or "random" for a fresh draw from @c std::random_device. */
inline std::optional<std::uint32_t> parse_seed(std::string_view text) noexcept {
    if (text == "random") return static_cast<std::uint32_t>(std::random_device {}());
    std::uint32_t seed = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), seed);
    if (error != std::errc {} || end != text.data() + text.size()) return std::nullopt;
    return seed;
}

inline std::size_t env_count(char const *name, std::size_t fallback) noexcept {
    return env_parsed(name, fallback, parse_count, "a positive count");
}

inline std::chrono::milliseconds env_duration(char const *name, std::chrono::milliseconds fallback) noexcept {
    return env_parsed(name, fallback, parse_duration, "a duration like 200ms or 10s");
}

inline std::uint32_t env_seed(char const *name, std::uint32_t fallback) noexcept {
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random");
}

/** Spells @p duration as a user types it: whole seconds as "10s", anything else as "1500ms". */
inline std::string spell_duration(std::chrono::milliseconds duration) {
    auto const count = duration.count();
    return count % 1000 ? std::to_string(count) + "ms" : std::to_string(count / 1000) + "s";
}

/** SplitMix64's finalizer, a bijection that spreads every input bit over the whole output. */
[[nodiscard]] constexpr std::uint64_t mix(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** The key of the generator stream @p name draws from, two mixes away from @p seed, so neither
 *  adjacent seeds nor two names ever give overlapping streams. */
[[nodiscard]] constexpr std::uint64_t stream_key(std::uint32_t seed, std::string_view name) noexcept {
    std::uint64_t hashed = 0xCBF29CE484222325ull;
    for (char const character : name) hashed = (hashed ^ static_cast<unsigned char>(character)) * 0x100000001B3ull;
    return mix(mix(seed ^ hashed));
}

/** Keeps @p value, and every write before it, from being optimized away. */
template <typename value_type_>
inline void do_not_optimize(value_type_ &&value) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    [[maybe_unused]] auto volatile *pointer = &value;
    _ReadWriteBarrier();
#elif defined(__clang__)
    asm volatile("" : "+r,m"(value) : : "memory");
#else
    asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

/** A named amount a benchmark reports next to its rate, printed as it is. */
struct counter_t {
    char const *name = nullptr;
    double value = 0;
};

/** A finished benchmark: its name, calls per second and counters, or why it was skipped. */
struct row_t {
    std::string_view name;
    double calls_per_second = 0;
    std::array<counter_t, 2> counters {};
    char const *skipped = nullptr;
};

/** Prints @p row on one line, with its speed-up over @p baseline calls per second when known. */
inline void print(row_t const &row, double baseline = 0) noexcept {
    if (row.skipped) return print_line(stdout, "{:<56} skipped: {}", row.name, row.skipped);
    double per_second = row.calls_per_second;
    char const *prefix = "";
    for (char const *next : {"k", "M", "G", "T", "P"})
        if (per_second >= 1000) per_second /= 1000, prefix = next;
    file_output_iterator_t output =
        std::format_to(file_output_iterator_t {stdout}, "{:<56}  calls {:.3f} {}/s", row.name, per_second, prefix);
    for (counter_t const &counter : row.counters)
        if (counter.name) output = std::format_to(output, "  {} {:.3f}", counter.name, counter.value);
    if (baseline > 0) output = std::format_to(output, "  {:.2f}x", row.calls_per_second / baseline);
    std::fputc('\n', stdout);
}

/**
 *  @brief One benchmark's timed loop, iterated as `for (std::size_t call : loop)`.
 *
 *  Setup above the loop stays untimed. The loop runs untimed for the warm-up, then counts calls
 *  until the time limit. It reads the clock once per 64th of the calls so far, so a short search
 *  doesn't time the clock itself.
 */
class loop_t {
    std::chrono::milliseconds warmup_, time_limit_;
    time_point_t start_ {};
    steady_clock_t::duration elapsed_ {};
    std::size_t calls_ = 0, next_check_ = 0;
    bool warming_up_ = true;
    char const *skipped_ = nullptr;
    std::array<counter_t, 2> counters_ {};

    bool keep_running() noexcept {
        if (skipped_) return false;
        if (calls_ < next_check_) return true;
        elapsed_ = steady_clock_t::now() - start_;
        if (warming_up_ && elapsed_ >= warmup_)
            warming_up_ = false, start_ = steady_clock_t::now(), elapsed_ = {}, calls_ = 0;
        else if (!warming_up_ && elapsed_ >= time_limit_) return false;
        next_check_ = calls_ + calls_ / 64 + 1;
        return true;
    }

  public:
    struct end_t {};
    struct iterator_t {
        loop_t *loop;
        bool operator!=(end_t) noexcept { return loop->keep_running(); }
        std::size_t operator*() const noexcept { return loop->calls_; }
        void operator++() noexcept { ++loop->calls_; }
    };

    loop_t(std::chrono::milliseconds warmup, std::chrono::milliseconds time_limit) noexcept
        : warmup_(warmup), time_limit_(time_limit) {}

    iterator_t begin() noexcept { return start_ = steady_clock_t::now(), iterator_t {this}; }
    end_t end() const noexcept { return {}; }

    /** Stops the loop and reports @p reason instead of results. */
    void skip(char const *reason) noexcept { skipped_ = reason; }

    /** Reports @p value as @p name, unscaled. */
    void counter(char const *name, double value) noexcept {
        for (counter_t &slot : counters_)
            if (!slot.name) return void(slot = {name, value});
    }

    /** The finished benchmark under @p name. */
    row_t row(std::string_view name) const noexcept {
        if (skipped_) return {name, 0, {}, skipped_};
        return {name, calls_ / std::chrono::duration<double>(elapsed_).count(), counters_, nullptr};
    }
};

/** Every benchmark setting, its default as the initializer, filled once by @c read_settings. */
struct settings_t {

    /** Benchmarks to run, by ECMAScript regex or else substring. */
    std::string_view filter;
    std::optional<std::regex> filter_regex;

    /** Seed every key draw derives its stream from. */
    std::uint32_t seed = 42;

    /** Untimed run ahead of each benchmark. */
    std::chrono::milliseconds warmup = std::chrono::seconds(1);

    /** Timed run of each benchmark. */
    std::chrono::milliseconds time_limit = std::chrono::seconds(10);

    /** Sorted rows each row-kit benchmark searches across. */
    std::size_t rows = 4096;

    /** Sorted keys each layout benchmark builds its trees over. */
    std::size_t keys = std::size_t {1} << 24;

    /** Queries drawn from the keys, searched in turn by every benchmark. */
    std::size_t queries = std::size_t {1} << 20;

    /** Whether @p name passes the filter. */
    bool selects(std::string_view name) const {
        if (filter.empty()) return true;
        if (filter_regex) return std::regex_search(name.begin(), name.end(), *filter_regex);
        return name.find(filter) != std::string_view::npos;
    }
};

/** Reads every @c settings_t variable, rejecting a zero count. */
inline settings_t read_settings() noexcept {
    settings_t settings;
    settings.filter = env_text("SMASHTABLE_FILTER").value_or("");
    if (!settings.filter.empty()) {
        try {
            settings.filter_regex.emplace(settings.filter.begin(), settings.filter.end());
        }
        catch (std::regex_error const &) {
        }
    }
    settings.seed = env_seed("SMASHTABLE_SEED", settings.seed);
    settings.warmup = env_duration("SMASHTABLE_WARMUP", settings.warmup);
    settings.time_limit = env_duration("SMASHTABLE_TIME_LIMIT", settings.time_limit);
    settings.rows = env_count("SMASHTABLE_ROWS", settings.rows);
    settings.keys = env_count("SMASHTABLE_KEYS", settings.keys);
    settings.queries = env_count("SMASHTABLE_QUERIES", settings.queries);
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(settings_t const &settings) noexcept {
    print_line(stdout, "- Seed: {}", settings.seed);
    print_line(stdout, "- Filter: {}", settings.filter.empty() ? std::string_view("none") : settings.filter);
    print_line(stdout, "- Warm-up: {}", spell_duration(settings.warmup));
    print_line(stdout, "- Time limit: {}", spell_duration(settings.time_limit));
    print_line(stdout, "- Rows: {}", settings.rows);
    print_line(stdout, "- Keys: {}", settings.keys);
    print_line(stdout, "- Queries: {}", settings.queries);
}

/** The facts this binary and this machine report: the processor, the compiler, and the kit
 *  dispatch picks. */
struct machine_t {
    char cpu[256] = "unknown";
    row_kit_t detected = row_kit_t::serial_k;
};

/** Reads the processor's model name where the system says it, and the kit dispatch picks. */
inline machine_t probe_machine() noexcept {
    machine_t machine;
    machine.detected = detect_row_kit();
    std::FILE *const cpuinfo = std::fopen("/proc/cpuinfo", "r");
    if (!cpuinfo) return machine;
    char line[512];
    while (std::fgets(line, sizeof(line), cpuinfo)) {
        char const *const colon = std::strchr(line, ':');
        if (!colon || (std::strncmp(line, "model name", 10) != 0 && std::strncmp(line, "uarch", 5) != 0 &&
                       std::strncmp(line, "CPU part", 8) != 0))
            continue;
        std::snprintf(machine.cpu, sizeof(machine.cpu), "%s", colon + 2);
        machine.cpu[std::strcspn(machine.cpu, "\n")] = '\0';
        if (std::strncmp(line, "model name", 10) == 0) break;
    }
    std::fclose(cpuinfo);
    return machine;
}

/** Prints the version line, the kits compiled and runnable, then the processor and the compiler. */
inline void print(machine_t const &machine) noexcept {
    auto const print_kits = [](char const *label, bool (*holds)(row_kit_t) noexcept) noexcept {
        file_output_iterator_t output = std::format_to(file_output_iterator_t {stdout}, "- {}:", label);
        char const *separator = " ";
        for (row_kit_t const kit : every_row_kit_k)
            if (holds(kit)) output = std::format_to(output, "{}{}", std::exchange(separator, ","), name_of(kit));
        std::fputc('\n', stdout);
    };
    print_line(stdout, "SmashTable {}.{}.{}", SMASHTABLE_VERSION_MAJOR, SMASHTABLE_VERSION_MINOR,
               SMASHTABLE_VERSION_PATCH);
    print_kits("Compiled for", row_kit_compiled);
    print_kits("This machine", row_kit_supported);
    print_line(stdout, "- Dispatches to: {}", name_of(machine.detected));
    print_line(stdout, "- CPU: {}", static_cast<char const *>(machine.cpu));
#if defined(__clang__)
    print_line(stdout, "- Compiler: Clang {}", __clang_version__);
#elif defined(__GNUC__)
    print_line(stdout, "- Compiler: GCC {}", __VERSION__);
#else
    print_line(stdout, "- Compiler: unrecognized");
#endif
}

/** Everything a benchmark reads, built once in @c main and passed down by reference. */
struct environment_t {
    settings_t settings;
    machine_t machine;
};

/** Runs @p measure on a fresh loop when the filter selects @p name, then prints its row against
 *  @p baseline. Returns the row's calls per second, or 0 when it was filtered out or skipped. */
template <typename measure_type_>
double run(environment_t const &env, std::string_view name, double baseline, measure_type_ &&measure) noexcept {
    if (!env.settings.selects(name)) return 0;
    loop_t loop(env.settings.warmup, env.settings.time_limit);
    measure(loop);
    row_t const row = loop.row(name);
    print(row, baseline);
    return row.calls_per_second;
}

} // namespace ashvardanian::smashtable::bench
