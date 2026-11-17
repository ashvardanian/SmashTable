/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Harness for the benchmarks - settings, the machine, and the timed loop with its rows.
 *
 *  Every setting is a @c SMASHTABLE_<NAME> variable that a @c --name flag overrides;
 *  CONTRIBUTING.md lists them beside the test suites' own.
 */
#pragma once
#include <cctype>  // `std::tolower`
#include <cmath>   // `std::isfinite`
#include <cstddef> // `std::ptrdiff_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstdio>  // `std::FILE`, `std::fputc`, `std::fopen`, `std::fgets`
#include <cstdlib> // `std::exit`, `std::getenv`, `std::strtod`
#include <cstring> // `std::strchr`, `std::strncmp`, `std::strcspn`, `std::strcmp`

#include <array>        // `std::array`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <format>       // `std::format_to`, `std::format_string`
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

/** Writes one line to @p stream, formatted from @p format and @p args, then its newline. */
template <typename... args_types_>
void print_line(std::FILE *stream, std::format_string<args_types_...> format, args_types_ &&...args) {
    std::format_to(file_output_iterator_t {stream}, format, std::forward<args_types_>(args)...);
    std::fputc('\n', stream);
}

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const *variable, char const *text, char const *expected) noexcept {
    print_line(stderr, "{}=\"{}\" does not parse, expected {}", variable, text, expected);
    std::exit(1);
}

/** The text of a setting: its @p flag when given, else its environment @p variable , else null. */
inline char const *setting_text(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    for (int index = 1; index < argc; ++index) {
        std::string_view const argument {argv[index]};
        if (argument == flag)
            return index + 1 < argc && std::strncmp(argv[index + 1], "--", 2) != 0 ? argv[index + 1] : "1";
        if (argument.size() > flag.size() && argument.starts_with(flag) && argument[flag.size()] == '=')
            return argv[index] + flag.size() + 1;
    }
    char const *text = std::getenv(variable);
    return text && *text ? text : nullptr;
}

/** A switch, @p fallback when unset: 0, false, off and no turn it off, anything else on. */
inline bool read_flag_setting(int argc, char **argv, char const *variable, std::string_view flag,
                              bool fallback = false) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    for (char const *off : {"0", "false", "off", "no"})
        if (std::strcmp(text, off) == 0) return false;
    return true;
}

/** A whole number, @p fallback when unset. */
inline std::size_t read_count_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                      std::size_t fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::size_t count = 0;
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "a whole number");
    return count;
}

/** A positive real, @p fallback when unset. */
inline double read_real_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                double fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    char *end = nullptr;
    double const value = std::strtod(text, &end);
    if (*end != '\0' || !(value > 0) || !std::isfinite(value))
        exit_unparsed(variable, text, "a positive number like 0.1 or 10");
    return value;
}

/** A positive duration such as 200ms, 10s or 2m, @p fallback when unset. */
inline std::chrono::milliseconds read_duration_setting(int argc, char **argv, char const *variable,
                                                       std::string_view flag,
                                                       std::chrono::milliseconds fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::uint32_t count = 0;
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (parse_status == std::errc {} && count != 0) {
        if (unit == "ms") return std::chrono::milliseconds(count);
        if (unit == "s") return std::chrono::seconds(count);
        if (unit == "m") return std::chrono::minutes(count);
    }
    exit_unparsed(variable, text, "a duration like 200ms, 10s or 2m");
}

/** The run's seed: 42 when unset, drawn afresh for @c random , else any unsigned 64-bit value. */
inline std::uint64_t read_seed_setting(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return 42;
    std::string_view const spelled {text};
    if (spelled == "random") {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }
    std::uint64_t seed = 0;
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: a regex search when the pattern compiles, else a substring. */
struct filter_t {
    std::string_view pattern;
    std::regex regex;
    bool is_regex = false;

    bool selects(std::string_view label) const noexcept {
        if (pattern.empty()) return true;
        if (is_regex) return std::regex_search(label.begin(), label.end(), regex);
        return label.find(pattern) != std::string_view::npos;
    }
};

/** The filter a setting spells, compiled once; empty, selecting everything, when unset. */
inline filter_t read_filter_setting(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    filter_t filter;
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.pattern = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.assign(filter.pattern.begin(), filter.pattern.end());
        filter.is_regex = true;
    }
    catch (std::regex_error const &) {
    }
#else
    filter.regex.assign(filter.pattern.begin(), filter.pattern.end());
    filter.is_regex = true;
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    filter_t filter;
    char const *program = "";
};

/** SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits. */
constexpr std::uint64_t split_mix_bits(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** The @p index -th stream of the family @p name under @p seed , identical in every language. */
constexpr std::uint64_t stream_key(std::uint64_t seed, std::string_view name, std::uint64_t index = 0) noexcept {
    std::uint64_t hashed = 0xCBF29CE484222325ull;
    for (char const character : name) hashed = (hashed ^ static_cast<std::uint8_t>(character)) * 0x100000001B3ull;
    return split_mix_bits(split_mix_bits(seed ^ hashed) + index);
}

/**
 *  @brief SplitMix64 over one 64-bit counter, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t counter = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : counter(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(counter += 0x9E3779B97F4A7C15ull); }

    /** A uniform draw below @p bound , zero for an empty bound, without the bias of a modulo. */
    std::uint64_t below(std::uint64_t bound) noexcept {
        if (bound == 0) return 0;
        std::uint64_t const rejection_floor = (0u - bound) % bound;
        std::uint64_t draw = (*this)();
        while (draw < rejection_floor) draw = (*this)();
        return draw % bound;
    }
};

#pragma endregion Shared Settings

/** Spells @p duration as a user types it: whole seconds as "10s", anything else as "1500ms". */
inline std::string spell_duration(std::chrono::milliseconds duration) {
    auto const count = duration.count();
    return count % 1000 ? std::to_string(count) + "ms" : std::to_string(count / 1000) + "s";
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
    if (row.skipped) return print_line(stdout, "{:<64} skipped: {}", row.name, row.skipped);
    double per_second = row.calls_per_second;
    char const *prefix = "";
    for (char const *next : {"k", "M", "G", "T", "P"})
        if (per_second >= 1000) per_second /= 1000, prefix = next;
    file_output_iterator_t output =
        std::format_to(file_output_iterator_t {stdout}, "{:<64}  calls {:.3f} {}/s", row.name, per_second, prefix);
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

/** What a benchmark reads: the common settings, then how long to time and how much to build. */
struct bench_settings_t : common_settings_t {

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
};

/** Reads every @c bench_settings_t field, rejecting a zero count, which every benchmark divides
 *  by. */
inline bench_settings_t read_bench_settings(int argc, char **argv) noexcept {
    bench_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "SMASHTABLE_SEED", "--seed");
    settings.filter = read_filter_setting(argc, argv, "SMASHTABLE_FILTER", "--filter");
    auto const read_positive = [&](char const *variable, std::string_view flag, std::size_t fallback) noexcept {
        std::size_t const count = read_count_setting(argc, argv, variable, flag, fallback);
        if (count == 0) exit_unparsed(variable, "0", "a positive whole number");
        return count;
    };
    settings.warmup = read_duration_setting(argc, argv, "SMASHTABLE_WARMUP", "--warmup", settings.warmup);
    settings.time_limit =
        read_duration_setting(argc, argv, "SMASHTABLE_TIME_LIMIT", "--time-limit", settings.time_limit);
    settings.rows = read_positive("SMASHTABLE_ROWS", "--rows", settings.rows);
    settings.keys = read_positive("SMASHTABLE_KEYS", "--keys", settings.keys);
    settings.queries = read_positive("SMASHTABLE_QUERIES", "--queries", settings.queries);
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(bench_settings_t const &settings) noexcept {
    print_line(stdout, "- Seed: {}", settings.seed);
    print_line(stdout, "- Filter: {}",
               settings.filter.pattern.empty() ? std::string_view("none") : settings.filter.pattern);
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
    bench_settings_t settings;
    machine_t machine;
};

/** Runs @p measure on a fresh loop when the filter selects @p name, then prints its row against
 *  @p baseline. Returns the row's calls per second, or 0 when it was filtered out or skipped. */
template <typename measure_type_>
double run(environment_t const &env, std::string_view name, double baseline, measure_type_ &&measure) noexcept {
    if (!env.settings.filter.selects(name)) return 0;
    loop_t loop(env.settings.warmup, env.settings.time_limit);
    measure(loop);
    row_t const row = loop.row(name);
    print(row, baseline);
    return row.calls_per_second;
}

} // namespace ashvardanian::smashtable::bench
