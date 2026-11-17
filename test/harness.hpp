/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date August 15, 2026
 *  @brief Harness for the C++ suites - checks, a named test runner, and crash localization.
 *
 *  Every setting is a @c SMASHTABLE_<NAME> variable that a @c --name flag overrides;
 *  CONTRIBUTING.md lists them. A filter that matches no case fails the binary rather than passing
 *  an empty suite.
 *
 *  @section test_failure_model Failure Model
 *
 *  A failed check prints its location and details, counts into @c test_run, and lets the case go
 *  on, so one run lists every mismatch it reaches. @c run_test_case diffs that count around each
 *  case, prints a rerun line naming the seed and a filter that selects only the failing case, and
 *  moves on to the next one; @c finish_test_run makes the exit status. Where the next step after a
 *  failed check would dereference or index past what it checked, the case returns on the check's
 *  answer. Checks in a @c noexcept visitor or a spawned thread count the same way, as nothing
 *  unwinds the stack for them.
 */
#pragma once
#include <cctype>  // `std::tolower`
#include <cmath>   // `std::llround`, `std::isfinite`
#include <csignal> // `std::signal`, `std::raise`, `SIGSEGV`, `SIGABRT`, `SIGTRAP`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`, `std::uint_least32_t`
#include <cstdio>  // `std::fputc`, `std::setvbuf`
#include <cstdlib> // `std::exit`, `std::getenv`
#include <cstring> // `std::strcmp`, `std::strncmp`

#include <algorithm>       // `std::max`
#include <atomic>          // `std::atomic`
#include <charconv>        // `std::from_chars`
#include <chrono>          // `std::chrono::steady_clock`
#include <format>          // `std::format_to`, `std::format_string`
#include <functional>      // `std::invoke`
#include <iterator>        // `std::output_iterator`
#include <random>          // `std::random_device`
#include <regex>           // `std::regex`, `std::regex_search`
#include <source_location> // `std::source_location`
#include <string_view>     // `std::string_view`
#include <system_error>    // `std::errc`
#include <thread>          // `std::this_thread::yield`
#include <type_traits>     // `std::is_void_v`, `std::invoke_result_t`, `std::type_identity_t`
#include <utility>         // `std::cmp_equal`, `std::exchange`

#include "smashtable/row_search.hpp" // `every_row_kit_k`, `row_kit_compiled`, `row_kit_supported`
#include "smashtable/shared.hpp"     // `status_t`, `succeeded`, `name_of`

#if SMASHTABLE_OS_WINDOWS_
#include <io.h> // `_write`
#else
#include <unistd.h> // `write`, `STDERR_FILENO`
#endif
/*  Only glibc and Darwin ship `backtrace` in libc itself; FreeBSD and musl need another library. */
#if (defined(__GLIBC__) || SMASHTABLE_OS_APPLE_) && __has_include(<execinfo.h>)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif

namespace ashvardanian::smashtable::test {

#pragma region Formatted Output

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
 *  @warning Never from a signal handler - formatting is not async-signal-safe.
 *
 *  Formats straight into @p stream rather than into a @c std::string, so there is no allocation and
 *  no buffer to size: a long @c what() prints whole instead of being truncated to fit.
 */
template <typename... args_types_>
inline void print_line([[maybe_unused]] std::FILE *stream, [[maybe_unused]] std::format_string<args_types_...> pattern,
                       [[maybe_unused]] args_types_ &&...args) noexcept {
    // Host formatting must stay out of NVCC's relaxed constexpr device pass.
#if !defined(__CUDA_ARCH__)
    std::format_to(file_output_iterator_t {stream}, pattern, std::forward<args_types_>(args)...);
    std::fputc('\n', stream);
#endif
}

#pragma endregion Formatted Output

#pragma region Machine

/** The facts this machine reports next to the build: the kit dispatch picks. */
struct machine_t {
    row_kit_t detected = row_kit_t::serial_k;
};

/** Probes the kit @c machine_t reports. */
inline machine_t probe_machine() noexcept { return {detect_row_kit()}; }

/** Prints the version line, the kits compiled and runnable, and the kit dispatch picks. */
inline void print([[maybe_unused]] machine_t const &machine, [[maybe_unused]] std::FILE *stream = stdout) noexcept {
#if !defined(__CUDA_ARCH__)
    auto const print_kits = [stream](char const *label, bool (*holds)(row_kit_t) noexcept) noexcept {
        file_output_iterator_t output = std::format_to(file_output_iterator_t {stream}, "- {}:", label);
        char const *separator = " ";
        for (row_kit_t const kit : every_row_kit_k)
            if (holds(kit)) output = std::format_to(output, "{}{}", std::exchange(separator, ","), name_of(kit));
        std::fputc('\n', stream);
    };
    print_line(stream, "SmashTable {}.{}.{}", SMASHTABLE_VERSION_MAJOR, SMASHTABLE_VERSION_MINOR,
               SMASHTABLE_VERSION_PATCH);
    print_kits("Compiled for", row_kit_compiled);
    print_kits("This machine", row_kit_supported);
    print_line(stream, "- Dispatches to: {}", name_of(machine.detected));
#endif
}

/** Why a case skips on a machine that cannot run @p kit . */
[[nodiscard]] constexpr char const *needs_kit_reason(row_kit_t kit) noexcept {
    switch (kit) {
    case row_kit_t::serial_k: return "needs row_kit_t::serial_k";
    case row_kit_t::haswell_k: return "needs row_kit_t::haswell_k";
    case row_kit_t::skylake_k: return "needs row_kit_t::skylake_k";
    case row_kit_t::icelake_k: return "needs row_kit_t::icelake_k";
    case row_kit_t::neon_k: return "needs row_kit_t::neon_k";
    case row_kit_t::sve_k: return "needs row_kit_t::sve_k";
    case row_kit_t::rvv_k: return "needs row_kit_t::rvv_k";
    case row_kit_t::powervsx_k: return "needs row_kit_t::powervsx_k";
    case row_kit_t::v128_k: return "needs row_kit_t::v128_k";
    }
    return "needs an unrecognized kit";
}

#pragma endregion Machine

/** Stops in an attached debugger, or dumps core without one; no crash handler catches SIGTRAP. */
inline void trap_into_debugger() noexcept {
#if SMASHTABLE_OS_WINDOWS_
    __debugbreak();
#else
    std::raise(SIGTRAP);
#endif
}

/** Prints the case in progress and a backtrace on a fatal signal; defined beside its run. */
inline void install_test_signal_handlers() noexcept;

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

/** What a run reads: the common settings, then how hard to push and how to treat failures. */
struct test_settings_t : common_settings_t {
    double scale = 1.0;
    bool trap_on_failure = false;
    bool refuse_skips = false;
};

/** Reads every @c test_settings_t field, each from its flag or else its variable. */
inline test_settings_t read_test_settings(int argc, char **argv) noexcept {
    test_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "SMASHTABLE_SEED", "--seed");
    settings.filter = read_filter_setting(argc, argv, "SMASHTABLE_FILTER", "--filter");
    settings.scale = read_real_setting(argc, argv, "SMASHTABLE_SCALE", "--scale", 1.0);
    settings.trap_on_failure = read_flag_setting(argc, argv, "SMASHTABLE_TRAP_ON_FAILURE", "--trap-on-failure");
    settings.refuse_skips = read_flag_setting(argc, argv, "SMASHTABLE_NO_SKIPS", "--no-skips");
    settings.program = argc != 0 ? argv[0] : "";
    return settings;
}

#pragma region Shared Run

/** What a case reports beyond its checks: that it ran, or why it skipped. */
struct case_result_t {
    char const *skip_reason = nullptr;

    [[nodiscard]] static constexpr case_result_t skipped(char const *reason) noexcept { return {reason}; }
};

/**
 *  @brief The run in progress: the harness's one sanctioned mutable global.
 *
 *  Checks fire on worker threads and inside helpers as well as in a case's own body, so the run
 *  counts failed checks in one place and @ref run_test_case diffs that count around each case,
 *  rather than threading a tally through every call. After a crash, printing @c test_run from a
 *  core dump names the case in progress and the last failed check, and the signal handlers print
 *  the case too.
 */
struct test_run_t {
    test_settings_t settings {};
    char const *current_case = nullptr;
    std::atomic<std::size_t> failed_checks {0};
    std::atomic<char const *> last_failed_file {nullptr};
    std::atomic<std::uint_least32_t> last_failed_line {0};
    std::size_t passed_cases = 0;
    std::size_t failed_cases = 0;
    std::size_t skipped_cases = 0;
};

inline test_run_t test_run;

/** How long any wait in a test may take; a passing case never waits it out, so it bounds hangs. */
inline constexpr std::chrono::seconds wait_limit_k {120};

/** Yields until @p predicate holds; false when @ref wait_limit_k passed first. */
template <typename predicate_type_>
bool spin_until(predicate_type_ &&predicate) noexcept {
    auto const deadline = std::chrono::steady_clock::now() + wait_limit_k;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

/** @p baseline iterations under the @c SCALE setting, never fewer than one. */
inline std::size_t scaled_iterations(std::size_t baseline) noexcept {
    double const scaled = static_cast<double>(baseline) * test_run.settings.scale;
    return scaled < 1 ? 1 : static_cast<std::size_t>(std::llround(scaled));
}

/** The step that visits a @p coverage fraction of a sweep, at least one. */
inline std::size_t sweep_stride(double coverage) noexcept {
    double const stride = 1.0 / coverage;
    return stride < 1 ? 1 : static_cast<std::size_t>(std::llround(stride));
}

/** The @p index -th stream of the case in progress, so a filtered run draws what a full one did. */
inline std::uint64_t case_stream_key(std::uint64_t index = 0) noexcept {
    return stream_key(test_run.settings.seed, test_run.current_case ? test_run.current_case : "", index);
}

/** Line-buffers stdout, prints the machine, reads the settings, arms the crash handlers and
 *  prints the settings line, before the first case. */
inline void start_test_run(int argc, char **argv) noexcept {
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ); // before any output, or the C library keeps its buffering
    print(probe_machine());
    test_run.settings = read_test_settings(argc, argv);
    install_test_signal_handlers();
    print_line(stdout, "- Seed: {}, scale: {}, filter: {}", test_run.settings.seed, test_run.settings.scale,
               test_run.settings.filter.pattern.empty() ? "none" : test_run.settings.filter.pattern);
}

/** Counts one failed check at @p where , and traps when the run asked to stop at the first. */
inline void record_failed_check(std::source_location where) noexcept {
    std::fflush(stdout); // so a log that merges both streams keeps the failure under its case
    print_line(stderr, "  check failed at {}:{}", where.file_name(), where.line());
    test_run.last_failed_file.store(where.file_name(), std::memory_order_relaxed);
    test_run.last_failed_line.store(where.line(), std::memory_order_relaxed);
    test_run.failed_checks.fetch_add(1, std::memory_order_relaxed);
    if (test_run.settings.trap_on_failure) trap_into_debugger();
}

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line. */
inline bool expect_true(bool is_satisfied, std::source_location where = std::source_location::current()) noexcept {
    if (!is_satisfied) record_failed_check(where);
    return is_satisfied;
}

/** A case label spelled as a regex that matches only itself, every metacharacter escaped. */
struct regex_escaped_t {
    std::string_view label;
};

} // namespace ashvardanian::smashtable::test

/** Writes a @ref regex_escaped_t straight to the output, a backslash before each metacharacter. */
template <>
struct std::formatter<ashvardanian::smashtable::test::regex_escaped_t> {
    constexpr auto parse(std::format_parse_context &context) { return context.begin(); }
    template <typename context_type_>
    auto format(ashvardanian::smashtable::test::regex_escaped_t const &escaped, context_type_ &context) const {
        auto output = context.out();
        for (char const character : escaped.label) {
            if (std::string_view {"\\^$.|?*+()[]{}"}.find(character) != std::string_view::npos) *output++ = '\\';
            *output++ = character;
        }
        return output;
    }
};

namespace ashvardanian::smashtable::test {

/** Prints one case's verdict, and the line that reruns it alone when it failed. */
inline void report_test_case(char const *label, std::size_t failed_checks, char const *skip_reason,
                             double seconds) noexcept {
    if (skip_reason) {
        print_line(stdout, "- {} ... skipped: {}", label, skip_reason);
        return;
    }
    if (failed_checks == 0) {
        print_line(stdout, "- {} ... ok ({:.2f} s)", label, seconds);
        return;
    }
    std::fflush(stdout);
    print_line(stderr, "- {} ... FAILED: {} failed check(s)", label, failed_checks);
    print_line(stderr, "  rerun: {} --seed {} --filter '^{}$'", test_run.settings.program, test_run.settings.seed,
               regex_escaped_t {label});
}

/** Runs one case unless the filter leaves it out, and records whether it passed or skipped. */
template <typename case_function_type_, typename... argument_types_>
void run_test_case(char const *label, case_function_type_ &&case_function, argument_types_ &&...arguments) noexcept {
    if (!test_run.settings.filter.selects(label)) return;
    test_run.current_case = label;
    std::size_t const failed_before = test_run.failed_checks.load();
    auto const started = std::chrono::steady_clock::now();
    case_result_t result {};
    using returned_type = std::invoke_result_t<case_function_type_, argument_types_...>;
    if constexpr (std::is_void_v<returned_type>)
        std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    else result = std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    std::chrono::duration<double> const elapsed = std::chrono::steady_clock::now() - started;
    std::size_t const failed_checks = test_run.failed_checks.load() - failed_before;
    if (result.skip_reason) ++test_run.skipped_cases;
    else if (failed_checks != 0) ++test_run.failed_cases;
    else ++test_run.passed_cases;
    report_test_case(label, failed_checks, result.skip_reason, elapsed.count());
    test_run.current_case = nullptr;
}

/** The exit code: 1 when a case or a check outside any case failed, or nothing ran; 3 when
 *  skips are refused and one happened; else 0. */
inline int finish_test_run() noexcept {
    print_line(stdout, "{} passed, {} failed, {} skipped, seed {}", test_run.passed_cases, test_run.failed_cases,
               test_run.skipped_cases, test_run.settings.seed);
    bool const failed = test_run.failed_cases != 0 || test_run.failed_checks.load() != 0;
    if (failed || test_run.passed_cases + test_run.skipped_cases == 0) return 1;
    if (test_run.skipped_cases != 0 && test_run.settings.refuse_skips) return 3;
    return 0;
}

#pragma endregion Shared Run

#pragma region Crash Localization

/**
 *  @brief Names the signal and the case in progress, prints a backtrace, then re-raises the signal,
 *      so a crash self-localizes rather than dying silently under CI's output redirection.
 *
 *  Writes through a raw @c write, since the crashing thread may already hold the stdio lock that
 *  @c std::fprintf takes.
 */
inline void test_fatal_signal_handler(int signal_number) noexcept {
    auto const write_text = [](std::string_view text) noexcept {
#if SMASHTABLE_OS_WINDOWS_
        [[maybe_unused]] auto const written = _write(2, text.data(), static_cast<unsigned>(text.size()));
#else
        [[maybe_unused]] auto const written = ::write(STDERR_FILENO, text.data(), text.size());
#endif
    };
    std::string_view signal_name = "SIGSEGV";
    if (signal_number == SIGABRT) signal_name = "SIGABRT";
    else if (signal_number == SIGILL) signal_name = "SIGILL";
    else if (signal_number == SIGFPE) signal_name = "SIGFPE";
#if defined(SIGBUS)
    else if (signal_number == SIGBUS) signal_name = "SIGBUS";
#endif
    write_text("\n*** ");
    write_text(signal_name);
    write_text(" in case '");
    write_text(test_run.current_case ? test_run.current_case : "none");
    write_text("' - backtrace follows ***\n");
#if (defined(__GLIBC__) || SMASHTABLE_OS_APPLE_) && __has_include(<execinfo.h>)
    void *frames[64];
    int const frames_count = backtrace(frames, sizeof(frames) / sizeof(frames[0]));
    // The `_fd` form writes without allocating, which is what makes it usable from a handler.
    backtrace_symbols_fd(frames, frames_count, STDERR_FILENO);
#endif
    // Restore and re-raise, so the shell still sees the real signal and any core dump is produced.
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

inline void install_test_signal_handlers() noexcept {
    std::signal(SIGSEGV, test_fatal_signal_handler);
    std::signal(SIGABRT, test_fatal_signal_handler);
    std::signal(SIGILL, test_fatal_signal_handler);
    std::signal(SIGFPE, test_fatal_signal_handler);
#if defined(SIGBUS) // Windows has no bus error to catch
    std::signal(SIGBUS, test_fatal_signal_handler);
#endif
}

#pragma endregion Crash Localization

#pragma region Checks

/**
 *  @brief Whether both sides are whole numbers, which is what lets a comparison cross signedness.
 *
 *  @c bool is excluded because the promotion it would get is the answer everybody already expects,
 *  and @c std::cmp_equal refuses it outright.
 */
template <typename left_type_, typename right_type_>
constexpr bool whole_numbers_ =
    std::is_integral_v<std::remove_cvref_t<left_type_>> && std::is_integral_v<std::remove_cvref_t<right_type_>> &&
    !std::is_same_v<std::remove_cvref_t<left_type_>, bool> && !std::is_same_v<std::remove_cvref_t<right_type_>, bool>;

/**
 *  @brief Whether the two compare equal, across signedness where both sides are whole numbers.
 *
 *  Comparing whole numbers by value rather than by promotion answers correctly for every pairing,
 *  so an unsigned count checked against a literal @c 0 neither warns nor wraps, and leaves
 *  everything else to its own @c operator==.
 */
template <typename left_type_, typename right_type_>
[[nodiscard]] constexpr bool equal_(left_type_ const &left, right_type_ const &right) noexcept {
    if constexpr (whole_numbers_<left_type_, right_type_>) return std::cmp_equal(left, right);
    else return left == right;
}

/** Whether a failed comparison can name @p type_ : a status, a result holding one, or anything
 *  @c std::format takes. */
template <typename type_>
constexpr bool printable_() noexcept {
    using bare_t = std::remove_cvref_t<type_>;
    if constexpr (std::is_same_v<bare_t, status_t>) return true;
    else if constexpr (requires(bare_t const &value) {
                           value.status();
                           value.has_value();
                           *value;
                       })
        return printable_<decltype(*std::declval<bare_t const &>())>();
    else return std::is_default_constructible_v<std::formatter<bare_t, char>>;
}

/** Prints one side of a failed comparison, naming a status rather than numbering it. */
template <typename type_>
void print_operand_(type_ const &value) noexcept {
    using bare_t = std::remove_cvref_t<type_>;
    if constexpr (std::is_same_v<bare_t, status_t>) std::fputs(name_of(value), stderr);
    else if constexpr (requires {
                           value.status();
                           value.has_value();
                           *value;
                       }) {
        // A result that holds something was compared for what it holds, so that is what a reader
        // needs; the status is the answer only when there is no value to have compared.
        if (!value.has_value()) std::fputs(name_of(value.status()), stderr);
        else print_operand_(*value);
    }
    else std::format_to(file_output_iterator_t {stderr}, "{}", value);
}

/** Prints both sides of a failed equality under its location line, when both can be named. */
template <typename actual_type_, typename wanted_type_>
void print_unequal_([[maybe_unused]] actual_type_ const &actual, [[maybe_unused]] wanted_type_ const &wanted) noexcept {
    if constexpr (printable_<actual_type_>() && printable_<wanted_type_>()) {
        std::fputs("    expected ", stderr);
        print_operand_(wanted);
        std::fputs(", got ", stderr);
        print_operand_(actual);
        std::fputc('\n', stderr);
    }
}

/**
 *  @brief A format string bundled with the @c std::source_location of the call that spelled it.
 *
 *  The implicit conversion from a string literal captures the caller's line behind the variadic
 *  arguments of a check's context, where a defaulted trailing parameter cannot go, and checks the
 *  pattern against those arguments at compile time, as @c std::format_string does.
 */
template <typename... args_types_>
struct located_format {
    std::format_string<args_types_...> format;
    std::source_location location;

    template <typename text_type_>
    consteval located_format(text_type_ const &text,
                             std::source_location location = std::source_location::current()) noexcept
        : format(text), location(location) {}
};

/** Prints @p context formatted with @p args , indented under a failed check's location line. */
template <typename... args_types_>
void print_context_(located_format<std::type_identity_t<args_types_>...> context, args_types_ const &...args) noexcept {
    std::fputs("    ", stderr);
    std::vformat_to(file_output_iterator_t {stderr}, context.format.get(), std::make_format_args(args...));
    std::fputc('\n', stderr);
}

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line and prints
 *  @p context formatted with @p args beneath it. */
template <typename... args_types_>
bool expect_true(bool is_satisfied, located_format<std::type_identity_t<args_types_>...> context,
                 args_types_ const &...args) noexcept {
    if (is_satisfied) return true;
    record_failed_check(context.location);
    print_context_<args_types_...>(context, args...);
    return false;
}

/** Holds when @p actual equals @p wanted ; otherwise counts a failed check at the caller's line and
 *  prints both beneath it. */
template <typename actual_type_, typename wanted_type_>
bool expect_equal(actual_type_ const &actual, wanted_type_ const &wanted,
                  std::source_location where = std::source_location::current()) noexcept {
    if (equal_(actual, wanted)) return true;
    record_failed_check(where);
    print_unequal_(actual, wanted);
    return false;
}

/** The same, printing @p context formatted with @p args beneath both sides on failure. */
template <typename actual_type_, typename wanted_type_, typename... args_types_>
bool expect_equal(actual_type_ const &actual, wanted_type_ const &wanted,
                  located_format<std::type_identity_t<args_types_>...> context, args_types_ const &...args) noexcept {
    if (equal_(actual, wanted)) return true;
    record_failed_check(context.location);
    print_unequal_(actual, wanted);
    print_context_<args_types_...>(context, args...);
    return false;
}

#pragma endregion Checks

#pragma region Container Helpers

/** Clears a container whether or not its @c clear reports a status. */
template <typename container_type_>
void clear_container(container_type_ &container) noexcept {
    if constexpr (std::is_void_v<decltype(container.clear())>) container.clear();
    else { [[maybe_unused]] auto const status = container.clear(); }
}

#pragma endregion Container Helpers

} // namespace ashvardanian::smashtable::test
