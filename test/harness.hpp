/**
 *  @file test/harness.hpp
 *  @author Ash Vardanian
 *  @date August 15, 2026
 *  @brief Harness for the C++ suites - assertions, a named test runner, and crash localization.
 *
 *  @section test_environment_variables Environment Variables
 *
 *  @c SMASHTABLE_FILTER is an ECMAScript regex searched for in a test's "suite.name" label; only
 *  matching tests run, and a pattern that does not compile matches as a plain substring instead.
 *  Unset or empty runs everything. Honored by @c run_test, which announces what it skipped, and a
 *  filter that matched nothing fails the binary rather than passing an empty suite.
 *
 *  @c SMASHTABLE_SEED is the seed every randomized suite draws from, so a failure names the run
 *  that produced it. Unset means @c default_seed_k, 42, which keeps an unattended build
 *  deterministic, and @c random draws a fresh one; @c log_environment prints either. Any other
 *  value that is not a whole number aborts rather than quietly reproducing the default run.
 *
 *  Both are read once, by @c read_test_environment at the top of @c main, into the constant every
 *  @c run_test call is handed.
 *
 *  @section test_failure_model Failure Model
 *
 *  A failed check reports and continues: it prints its diagnostic and throws @c test_failure_t, and
 *  @c run_test catches that, prints a rerun line naming the seed and a filter that selects only the
 *  failing test, and moves on to the next one; @c main exits with 1 at the end. Leaving the test
 *  rather than the statement matters, since many checks guard the dereference or the index on the
 *  very next line. A check inside a @c noexcept visitor or a spawned thread cannot unwind that far
 *  and terminates instead, as a @c <cassert> assert and a crash do; the signal handler then prints
 *  a backtrace, and the last test started on stdout names the one that died.
 */
#pragma once
#include <csignal> // `std::signal`, `std::raise`, `SIGSEGV`, `SIGABRT`
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fprintf`, `std::setvbuf`
#include <cstdlib> // `std::abort`, `std::getenv`

#include <chrono>      // `std::chrono::steady_clock`
#include <exception>   // `std::exception`
#include <format>      // `std::format_to`, `std::format_string`
#include <iterator>    // `std::output_iterator`
#include <limits>      // `std::numeric_limits`
#include <optional>    // `std::optional`
#include <random>      // `std::random_device`
#include <regex>       // `std::regex`, `std::regex_search`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::is_void_v`
#include <utility>     // `std::cmp_equal`, `std::cmp_less`, `std::exchange`

#if defined(_WIN32)
#include <io.h> // `_write`
#else
#include <unistd.h> // `write`, `STDERR_FILENO`
#endif
#if defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#endif

#include <smashtable/row_search.hpp> // `every_row_kit_k`, `row_kit_compiled`, `row_kit_supported`
#include <smashtable/shared.hpp>     // `status_t`, `succeeded`, `name_of`

#pragma region Assertions

/** Adds the status a check was handed, where it was handed one, to the message it prints. */
template <typename type_>
inline void st_explain_(type_ const &answered) noexcept {
    if constexpr (std::is_same_v<std::remove_cvref_t<type_>, ::ashvardanian::smashtable::status_t>)
        std::fprintf(stderr, " (answered %s)", ::ashvardanian::smashtable::name_of(answered));
    else if constexpr (requires { answered.status(); })
        std::fprintf(stderr, " (answered %s)", ::ashvardanian::smashtable::name_of(answered.status()));
}

/**
 *  @brief Whether both sides are whole numbers, which is what lets a comparison cross signedness.
 *
 *  @c bool is excluded because the promotion it would get is the answer everybody already expects,
 *  and @c std::cmp_equal and friends refuse it outright.
 */
template <typename left_type_, typename right_type_>
constexpr bool st_whole_numbers_ =
    std::is_integral_v<std::remove_cvref_t<left_type_>> && std::is_integral_v<std::remove_cvref_t<right_type_>> &&
    !std::is_same_v<std::remove_cvref_t<left_type_>, bool> && !std::is_same_v<std::remove_cvref_t<right_type_>, bool>;

/**
 *  @brief Whether the two compare equal, across signedness where both sides are whole numbers.
 *
 *  Binding each side to a variable is what lets the message name it, and it also turns a literal
 *  @c 0 into an @c int object - so an unsigned count compared against it would warn where the bare
 *  literal did not. Comparing whole numbers by value rather than by promotion answers correctly for
 *  every pairing and leaves everything else to its own @c operator==.
 */
template <typename left_type_, typename right_type_>
[[nodiscard]] constexpr bool st_equal_(left_type_ const &left, right_type_ const &right) noexcept {
    if constexpr (st_whole_numbers_<left_type_, right_type_>) return std::cmp_equal(left, right);
    else return left == right;
}

/** Whether the two differ, so every relation reads the same way round for the shared body. */
template <typename left_type_, typename right_type_>
[[nodiscard]] constexpr bool st_not_equal_(left_type_ const &left, right_type_ const &right) noexcept {
    return !st_equal_(left, right);
}

/** Whether the left side orders below the right, with the same cross-signedness parity. */
template <typename left_type_, typename right_type_>
[[nodiscard]] constexpr bool st_less_(left_type_ const &left, right_type_ const &right) noexcept {
    if constexpr (st_whole_numbers_<left_type_, right_type_>) return std::cmp_less(left, right);
    else return left < right;
}

/** Whether the left side orders below the right or matches it. */
template <typename left_type_, typename right_type_>
[[nodiscard]] constexpr bool st_less_equal_(left_type_ const &left, right_type_ const &right) noexcept {
    if constexpr (st_whole_numbers_<left_type_, right_type_>) return std::cmp_less_equal(left, right);
    else return left <= right;
}

/** Whether the left side orders above the right. */
template <typename left_type_, typename right_type_>
[[nodiscard]] constexpr bool st_greater_(left_type_ const &left, right_type_ const &right) noexcept {
    if constexpr (st_whole_numbers_<left_type_, right_type_>) return std::cmp_greater(left, right);
    else return left > right;
}

/** Whether the left side orders above the right or matches it. */
template <typename left_type_, typename right_type_>
[[nodiscard]] constexpr bool st_greater_equal_(left_type_ const &left, right_type_ const &right) noexcept {
    if constexpr (st_whole_numbers_<left_type_, right_type_>) return std::cmp_greater_equal(left, right);
    else return left >= right;
}

/** Adds one side of a comparison to the message, naming a status rather than numbering it. */
template <typename type_>
inline void st_print_operand_(char const *label, type_ const &value) noexcept {
    using bare_t = std::remove_cvref_t<type_>;
    if constexpr (std::is_same_v<bare_t, ::ashvardanian::smashtable::status_t>)
        std::fprintf(stderr, ", %s = %s", label, ::ashvardanian::smashtable::name_of(value));
    else if constexpr (std::is_same_v<bare_t, bool>) std::fprintf(stderr, ", %s = %s", label, value ? "true" : "false");
    else if constexpr (requires {
                           value.status();
                           value.has_value();
                           *value;
                       }) {
        // A result that holds something was compared for what it holds, so that is what a reader needs;
        // the status is the answer only when there is no value to have compared.
        if (!value.has_value())
            std::fprintf(stderr, ", %s = %s", label, ::ashvardanian::smashtable::name_of(value.status()));
        else st_print_operand_(label, *value);
    }
    else if constexpr (std::is_integral_v<bare_t>)
        std::fprintf(stderr, ", %s = %lld", label, static_cast<long long>(value));
}

/**
 *  @brief Verification that stays active regardless of @c NDEBUG - an oracle never compiles out.
 *
 *  Wrapped in @c do/while(0) so the macro is one statement: it demands its terminating semicolon
 *  and swallows a dangling @c else. Context belongs inside the condition as @c &&"text", which the
 *  stringified expression then prints; there is no streamed message and no object to return.
 */
#define st_verify_(condition)                                            \
    do {                                                                 \
        auto const &st_answered_ = (condition);                          \
        if (!::ashvardanian::smashtable::succeeded(st_answered_)) {      \
            std::fprintf(stderr, "Verification failed: %s", #condition); \
            st_explain_(st_answered_);                                   \
            std::fprintf(stderr, ", %s:%d\n", __FILE__, __LINE__);       \
            ::ashvardanian::smashtable::test::fail_test();               \
        }                                                                \
    } while (0)

/**
 *  @brief The body every relational check shares: bind each side once, and on failure name the
 *      relation, both operands, the caller's message and the line.
 *
 *  @p holds answers whether the assertion passed, so every relation reads the same way round.
 *  Binding before comparing is what lets the message print an operand rather than the expression
 *  that produced it - which for a status is the whole of what a reader needs.
 */
#define st_verify_relation_(first, holds, symbol, second, ...)                              \
    do {                                                                                    \
        auto const &st_left_ = (first);                                                     \
        auto const &st_right_ = (second);                                                   \
        if (!holds(st_left_, st_right_)) {                                                  \
            std::fprintf(stderr, "Verification failed: %s " symbol " %s", #first, #second); \
            st_print_operand_("left", st_left_);                                            \
            st_print_operand_("right", st_right_);                                          \
            __VA_OPT__(std::fprintf(stderr, ", %s", __VA_ARGS__);)                          \
            std::fprintf(stderr, ", %s:%d\n", __FILE__, __LINE__);                          \
            ::ashvardanian::smashtable::test::fail_test();                                  \
        }                                                                                   \
    } while (0)

/** Verification that two values match, naming both when they do not. */
#define st_verify_eq_(first, second, ...) st_verify_relation_(first, st_equal_, "==", second __VA_OPT__(, ) __VA_ARGS__)

/** Verification that two values differ, naming both when they do not. */
#define st_verify_ne_(first, second, ...) \
    st_verify_relation_(first, st_not_equal_, "!=", second __VA_OPT__(, ) __VA_ARGS__)

/** Verification that the first orders below the second, naming both when it does not. */
#define st_verify_lt_(first, second, ...) st_verify_relation_(first, st_less_, "<", second __VA_OPT__(, ) __VA_ARGS__)

/** Verification that the first never rises above the second, naming both when it does. */
#define st_verify_le_(first, second, ...) \
    st_verify_relation_(first, st_less_equal_, "<=", second __VA_OPT__(, ) __VA_ARGS__)

/** Verification that the first orders above the second, naming both when it does not. */
#define st_verify_gt_(first, second, ...) \
    st_verify_relation_(first, st_greater_, ">", second __VA_OPT__(, ) __VA_ARGS__)

/** Verification that the first never falls below the second, naming both when it does. */
#define st_verify_ge_(first, second, ...) \
    st_verify_relation_(first, st_greater_equal_, ">=", second __VA_OPT__(, ) __VA_ARGS__)

#pragma endregion Assertions

namespace ashvardanian::smashtable::test {

#pragma region Environment

/** The seed a randomized suite draws from when @c SMASHTABLE_SEED is unset. */
inline constexpr unsigned int default_seed_k = 42;

/**
 *  @brief Parses @p requested, the text of @c SMASHTABLE_SEED: @c default_seed_k when unset or
 *      empty, a fresh draw for @c random.
 *  @warning Aborts on any other text than a run of decimal digits below 2^32, so a sign or a stray
 *      space names no run rather than wrapping into one.
 */
[[nodiscard]] inline unsigned int parse_test_seed(char const *requested) noexcept {
    if (!requested || requested[0] == '\0') return default_seed_k;
    if (std::string_view(requested) == "random") return std::random_device {}();
    // Digit by digit rather than through `strtoul`, which skips leading spaces, negates a minus, and
    // wraps an overflow - three ways for a typo to come back as a seed nobody chose.
    unsigned long long parsed = 0;
    bool whole = true;
    for (char const *scan = requested; *scan != '\0' && whole; ++scan) {
        whole = *scan >= '0' && *scan <= '9';
        if (whole) parsed = parsed * 10 + static_cast<unsigned long long>(*scan - '0');
        whole = whole && parsed <= std::numeric_limits<unsigned int>::max();
    }
    if (whole) return static_cast<unsigned int>(parsed);
    std::fprintf(stderr, "SMASHTABLE_SEED=\"%s\" does not parse\n", requested);
    std::abort();
}

/**
 *  @brief What @c main reads from the environment, once, and hands to every @c run_test call.
 *
 *  A fuzzer pinned to one literal finds one defect once, and one drawing from the clock finds a
 *  defect nobody can reproduce. The seed is therefore an input the runner prints, so a failing run
 *  names the sequence that produced it, and a sweep is a shell loop rather than a source edit.
 */
struct test_environment_t {
    unsigned int seed {default_seed_k};

    /** @c SMASHTABLE_FILTER, or @c nullptr when it is unset or empty and every test runs. */
    char const *filter {};

    /** The filter compiled as an ECMAScript regex, or nothing when the pattern does not compile. */
    std::optional<std::regex> pattern {};

    /** The binary's @c argv[0], which turns a rerun line into a command. */
    std::string_view program {};

    /** Whether the test labelled @p name runs: the filter searched as a regex, or found as a plain
     *  substring when it does not compile. */
    [[nodiscard]] bool selects(std::string_view name) const noexcept {
        if (!filter) return true;
        if (pattern) return std::regex_search(name.begin(), name.end(), *pattern);
        return name.find(filter) != std::string_view::npos;
    }
};

/**
 *  @brief Reads @c SMASHTABLE_SEED and @c SMASHTABLE_FILTER. Call once, first thing in @c main.
 *  @param[in] program The binary's @c argv[0], which every rerun line ends with.
 *
 *  MSVC deprecates @c std::getenv in favor of the allocating @c _dupenv_s, which buys a suite
 *  nothing: both values are read here, before any thread exists, and the environment block
 *  outlives the run.
 */
[[nodiscard]] inline test_environment_t read_test_environment(char const *program) noexcept {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    char const *const seed = std::getenv("SMASHTABLE_SEED");
    char const *const filter = std::getenv("SMASHTABLE_FILTER");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
    test_environment_t environment;
    environment.seed = parse_test_seed(seed);
    environment.program = program ? program : "";
    if (!filter || filter[0] == '\0') return environment;
    environment.filter = filter;
    try {
        environment.pattern.emplace(filter);
    }
    catch (std::regex_error const &) {
        // Left empty, so `selects` finds the filter as a plain substring instead.
    }
    return environment;
}

#pragma endregion Environment

#pragma region Randomization

/** What a randomized test receives from @c run_test: the run's seed, to mix with its own name. */
struct test_context_t {
    unsigned int seed;
};

/**
 *  @brief The seed one suite draws from: FNV-1a over @p name, from a basis perturbed by @p seed.
 *
 *  Two suites seeded alike walk one sequence between them and cover half of what their count
 *  suggests. Passing @c __func__ keeps the name that separates them the same name the compiler
 *  already knows, so a suite cannot be added, renamed, or moved into another binary and collide
 *  with one already there. Mixed here rather than through @c smashtable::hash, so a change to the
 *  code under test never reshuffles the sequences that test it.
 */
[[nodiscard]] constexpr unsigned int mix_seed(unsigned int seed, std::string_view name) noexcept {
    std::uint32_t mixed = 2166136261u ^ seed;
    for (char const character : name) mixed = (mixed ^ static_cast<unsigned char>(character)) * 16777619u;
    return mixed;
}

#pragma endregion Randomization

#pragma region Crash Localization

/**
 *  @brief Prints a notice and a backtrace on a fatal signal, then re-raises it, so a crash
 *      self-localizes rather than dying silently under CI's output redirection.
 *
 *  Writes through a raw @c write, since the crashing thread may already hold the stdio lock that
 *  @c std::fprintf takes. It names no test: the last one started on stdout is the one that died.
 */
inline void test_fatal_signal_handler(int signal_number) noexcept {
    constexpr std::string_view message = "\n*** Fatal signal - backtrace follows ***\n";
#if defined(_WIN32)
    [[maybe_unused]] auto const written = _write(2, message.data(), static_cast<unsigned>(message.size()));
#else
    [[maybe_unused]] auto const written = ::write(STDERR_FILENO, message.data(), message.size());
#endif
#if defined(__linux__) && defined(__GLIBC__)
    void *frames[64];
    int const frames_count = backtrace(frames, sizeof(frames) / sizeof(frames[0]));
    // The `_fd` form writes without allocating, which is what makes it usable from a handler.
    backtrace_symbols_fd(frames, frames_count, STDERR_FILENO);
#endif
    // Restore and re-raise, so the shell still sees the real signal and any core dump is produced.
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

/** Installs the backtrace handlers and line-buffers stdout. Call once, from @c main. */
inline void install_test_signal_handlers() noexcept {
    // Line-buffered, so progress survives a crash under redirection. The size must be nonzero:
    // Windows ucrt fast-fails on a zero-sized buffering mode.
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
    std::signal(SIGSEGV, test_fatal_signal_handler);
    std::signal(SIGABRT, test_fatal_signal_handler);
    std::signal(SIGILL, test_fatal_signal_handler);
    std::signal(SIGFPE, test_fatal_signal_handler);
#if defined(SIGBUS) // Windows has no bus error to catch
    std::signal(SIGBUS, test_fatal_signal_handler);
#endif
}

#pragma endregion Crash Localization

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
inline void print_line(std::FILE *stream, std::format_string<args_types_...> pattern, args_types_ &&...args) noexcept {
    std::format_to(file_output_iterator_t {stream}, pattern, std::forward<args_types_>(args)...);
    std::fputc('\n', stream);
}

/** Prints one capability line, naming each kit @p holds accepts. */
inline void print_row_kits(char const *label, bool (*holds)(row_kit_t) noexcept) noexcept {
    file_output_iterator_t output = std::format_to(file_output_iterator_t {stdout}, "- {}:", label);
    char const *separator = " ";
    for (row_kit_t const kit : every_row_kit_k)
        if (holds(kit)) output = std::format_to(output, "{}{}", std::exchange(separator, ","), name_of(kit));
    std::fputc('\n', stdout);
}

/** Prints the library version, the kits this build carries, the ones this processor runs, the run's
 *  seed, and how to rerun one test under it. Call once, from a test's @c main. */
inline void log_environment(test_environment_t const &environment) noexcept {
    print_line(stdout, "SmashTable {}.{}.{}", SMASHTABLE_VERSION_MAJOR, SMASHTABLE_VERSION_MINOR,
               SMASHTABLE_VERSION_PATCH);
    print_row_kits("Compiled for", row_kit_compiled);
    print_row_kits("This machine", row_kit_supported);
    print_line(stdout, "- Seed: {}", environment.seed);
    print_line(stdout, "- Rerun one test: SMASHTABLE_SEED={} SMASHTABLE_FILTER='^<name>$' {}", environment.seed,
               environment.program);
}

#pragma endregion Formatted Output

#pragma region Test Runner

/** Thrown by a failed check once it has printed why, for @c run_test to catch and report. */
struct test_failure_t {};

/**
 *  @brief Leaves the running test after a failed check has printed its diagnostic.
 *
 *  A call rather than a @c throw inside the check's own macro, so a check in a @c noexcept visitor
 *  compiles without @c -Wterminate; failing there still terminates, and the handler backtraces it.
 */
[[noreturn]] inline void fail_test() { throw test_failure_t {}; }

/** How many tests ran and how many of those failed, summed over a binary's @c run_test calls. */
struct test_tally_t {
    std::size_t executed {0};
    std::size_t failed {0};

    test_tally_t &operator+=(test_tally_t const &other) noexcept {
        executed += other.executed;
        failed += other.failed;
        return *this;
    }
};

/** The body both @c run_test overloads share, around @p call, which runs the test itself. */
template <typename call_type_>
test_tally_t run_test_(test_environment_t const &environment, std::string_view name, call_type_ const &call) noexcept {
    if (!environment.selects(name)) {
        print_line(stdout, "- {} ... skipped (SMASHTABLE_FILTER)", name);
        std::fflush(stdout);
        return {};
    }

    print_line(stdout, "- {} ...", name);
    std::fflush(stdout);
    auto const started = std::chrono::steady_clock::now();
    try {
        call();
        double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        print_line(stdout, "- {} ... ok ({:.2f} s)", name, seconds);
        std::fflush(stdout);
        return {.executed = 1, .failed = 0};
    }
    catch (test_failure_t const &) {
        print_line(stderr, "- {} ... FAILED: see the check above", name);
    }
    catch (std::exception const &error) {
        print_line(stderr, "- {} ... FAILED: {}", name, error.what());
    }
    print_line(stderr, "  rerun: SMASHTABLE_SEED={} SMASHTABLE_FILTER='^{}$' {}", environment.seed, name,
               environment.program);
    return {.executed = 1, .failed = 1};
}

/**
 *  @brief Runs one named test when @p environment selects it, timing it and reporting the outcome.
 *  @param[in] environment What @c main read: the filter that selects tests and the seed they draw.
 *  @param[in] name The test's "suite.name" label, which is also its filter key.
 *  @param[in] test_function A function taking no arguments.
 *  @return One execution when the test ran, and one failure too when a check failed or it threw.
 *
 *  A failure prints a rerun line beneath it and returns, so the binary goes on to the next test.
 *  The started line prints before the call, so a hard crash leaves the running test as the last
 *  thing on stdout.
 *
 *  Takes a function pointer rather than any callable on purpose: a suite that grows a defaulted
 *  parameter stops being a @c void() and would otherwise hide behind a lambda at every call site
 *  instead of failing here.
 */
inline test_tally_t run_test(test_environment_t const &environment, std::string_view name,
                             void (*test_function)()) noexcept {
    return run_test_(environment, name, test_function);
}

/** Runs one named randomized test, handing it the run's seed; otherwise as the overload above. */
inline test_tally_t run_test(test_environment_t const &environment, std::string_view name,
                             void (*test_function)(test_context_t const &)) noexcept {
    return run_test_(environment, name, [&] { test_function(test_context_t {environment.seed}); });
}

/**
 *  @brief Reports whether every test passed, printing the verdict; its result is the exit status
 *      of @c main.
 *
 *  A filter that matched nothing fails here rather than passing: every test skipping leaves no
 *  failures to count, so a mistyped filter would otherwise be indistinguishable from a green suite.
 */
inline int report_test_failures(test_environment_t const &environment, test_tally_t const &tally) noexcept {
    if (tally.failed != 0) {
        print_line(stderr, "\n{} test(s) failed under SMASHTABLE_SEED={}.", tally.failed, environment.seed);
        return 1;
    }
    if (environment.filter && tally.executed == 0) {
        print_line(stderr, "\nSMASHTABLE_FILTER=\"{}\" matched no test, so nothing ran.", environment.filter);
        return 1;
    }
    print_line(stdout, "\nAll tests passed!");
    return 0;
}

#pragma endregion Test Runner

#pragma region Container Helpers

/** Clears a container whether or not its @c clear reports a status. */
template <typename container_type_>
void clear_container(container_type_ &container) noexcept {
    if constexpr (std::is_void_v<decltype(container.clear())>) container.clear();
    else { [[maybe_unused]] auto const status = container.clear(); }
}

#pragma endregion Container Helpers

} // namespace ashvardanian::smashtable::test
