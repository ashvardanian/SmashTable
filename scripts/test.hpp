/**
 *  @brief Harness for the C++ suites - assertions, a named test runner, and crash localization.
 *  @author Ash Vardanian
 *  @file scripts/test.hpp
 *  @date August 15, 2026
 *
 *  @section test_environment_variables Environment Variables
 *
 *  - @c SMASHTABLE_FILTER : substring matched against a test's "suite.name" label; only matching tests
 *    run. Unset or empty runs everything. Honored by @c run_test, which announces what it skipped, so a
 *    mistyped filter reads as a skip rather than as an empty suite.
 *
 *  @section test_failure_model Failure Model
 *
 *  Assertions abort rather than accumulate. Many of them guard the dereference or the index on the very
 *  next line, so a check that recorded a failure and carried on would hand the following statement a
 *  disengaged optional or an out-of-range subscript. The process dies at the defect, and the installed
 *  signal handler turns that into a backtrace.
 */
#pragma once
#include <cstdio>  // `std::fprintf`, `std::setvbuf`
#include <cstdlib> // `std::abort`, `std::getenv`
#include <csignal> // `std::signal`, `SIGSEGV`, `SIGABRT`
#include <cstring> // `std::strstr`

#include <chrono>      // `std::chrono::steady_clock`
#include <exception>   // `std::exception`
#include <format>      // `std::format_to`, `std::format_string`
#include <iterator>    // `std::output_iterator`
#include <type_traits> // `std::is_void_v`
#include <utility>     // `std::cmp_equal`

#if defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h> // `backtrace`, `backtrace_symbols_fd`
#include <unistd.h>   // `STDERR_FILENO`
#endif

#include <smashtable/shared.hpp> // `status_t`, `succeeded`, `name_of`

#pragma region Assertions

/** @brief Adds the status a check was handed, where it was handed one, to the message it prints. */
template <typename type_>
inline void st_explain_(type_ const &answered) noexcept {
    if constexpr (std::is_same_v<std::remove_cvref_t<type_>, ::ashvardanian::smashtable::status_t>)
        std::fprintf(stderr, " (answered %s)", ::ashvardanian::smashtable::name_of(answered));
    else if constexpr (requires { answered.status(); })
        std::fprintf(stderr, " (answered %s)", ::ashvardanian::smashtable::name_of(answered.status()));
}

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
    using bare_left_t = std::remove_cvref_t<left_type_>;
    using bare_right_t = std::remove_cvref_t<right_type_>;
    constexpr bool both_whole_numbers_k = std::is_integral_v<bare_left_t> && std::is_integral_v<bare_right_t> &&
                                          !std::is_same_v<bare_left_t, bool> && !std::is_same_v<bare_right_t, bool>;
    if constexpr (both_whole_numbers_k) return std::cmp_equal(left, right);
    else return left == right;
}

/** @brief Adds one side of a comparison to the message, naming a status rather than numbering it. */
template <typename type_>
inline void st_print_operand_(char const *label, type_ const &value) noexcept {
    using bare_t = std::remove_cvref_t<type_>;
    if constexpr (std::is_same_v<bare_t, ::ashvardanian::smashtable::status_t>)
        std::fprintf(stderr, ", %s = %s", label, ::ashvardanian::smashtable::name_of(value));
    else if constexpr (requires { value.status(); })
        std::fprintf(stderr, ", %s = %s", label, ::ashvardanian::smashtable::name_of(value.status()));
    else if constexpr (std::is_integral_v<bare_t>)
        std::fprintf(stderr, ", %s = %lld", label, static_cast<long long>(value));
}

/**
 *  @brief Verification that stays active regardless of @c NDEBUG - a test's oracle must never compile out.
 *
 *  Wrapped in @c do/while(0) so the macro is one statement: it demands its terminating semicolon and
 *  swallows a dangling @c else. Context belongs inside the condition as @c &&"text", which the
 *  stringified expression then prints; there is no streamed message and no object to return.
 */
#define st_verify_(condition)                                            \
    do {                                                                 \
        auto const &st_answered_ = (condition);                          \
        if (!::ashvardanian::smashtable::succeeded(st_answered_)) {      \
            std::fprintf(stderr, "Verification failed: %s", #condition); \
            st_explain_(st_answered_);                                   \
            std::fprintf(stderr, ", %s:%d\n", __FILE__, __LINE__);       \
            std::abort();                                                \
        }                                                                \
    } while (0)

/**
 *  @brief Verification that two values match, naming both when they do not.
 *
 *  Stringifying the two expressions says which comparison broke and neither of the values it
 *  compared, which for a status is the whole of what a reader needs.
 */
#define st_verify_eq_(first, second)                                                \
    do {                                                                            \
        auto const &st_left_ = (first);                                             \
        auto const &st_right_ = (second);                                           \
        if (!st_equal_(st_left_, st_right_)) {                                      \
            std::fprintf(stderr, "Verification failed: %s == %s", #first, #second); \
            st_print_operand_("left", st_left_);                                    \
            st_print_operand_("right", st_right_);                                  \
            std::fprintf(stderr, ", %s:%d\n", __FILE__, __LINE__);                  \
            std::abort();                                                           \
        }                                                                           \
    } while (0)

/** @brief Verification that two values differ, naming both when they do not. */
#define st_verify_ne_(first, second)                                                \
    do {                                                                            \
        auto const &st_left_ = (first);                                             \
        auto const &st_right_ = (second);                                           \
        if (st_equal_(st_left_, st_right_)) {                                       \
            std::fprintf(stderr, "Verification failed: %s != %s", #first, #second); \
            st_print_operand_("left", st_left_);                                    \
            st_print_operand_("right", st_right_);                                  \
            std::fprintf(stderr, ", %s:%d\n", __FILE__, __LINE__);                  \
            std::abort();                                                           \
        }                                                                           \
    } while (0)

#pragma endregion Assertions

namespace ashvardanian::smashtable::scripts {

#pragma region Crash Localization

/**
 *  @brief Prints a backtrace on a fatal signal, so an aborting check self-localizes rather than dying
 *    silently under CI's output redirection.
 */
inline void test_fatal_signal_handler(int signal_number) noexcept {
    std::fprintf(stderr, "\n*** Fatal signal %d - backtrace follows ***\n", signal_number);
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

/** @brief Installs the backtrace handlers and line-buffers stdout. Call once, from @c main. */
inline void install_test_signal_handlers() noexcept {
    // Line-buffered, so progress survives a crash under redirection. The size must be nonzero:
    // Windows ucrt fast-fails on a zero-sized buffering mode.
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
    std::signal(SIGSEGV, test_fatal_signal_handler);
    std::signal(SIGABRT, test_fatal_signal_handler);
}

#pragma endregion Crash Localization

#pragma region Formatted Output

/** @brief Output iterator handing each character straight to a @c std::FILE. */
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
 *    terminating it here, so @p pattern carries no trailing newline of its own.
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

#pragma endregion Formatted Output

#pragma region Test Runner

/**
 *  @brief Reads @c SMASHTABLE_FILTER, or @c nullptr when it is unset.
 *
 *  MSVC deprecates @c std::getenv in favor of the allocating @c _dupenv_s, which buys a suite nothing:
 *  the value is read once from @c main, before any thread exists, and the environment block outlives
 *  the run. One place to say so beats the same suppression in every suite.
 */
[[nodiscard]] inline char const *test_filter() noexcept {
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    return std::getenv("SMASHTABLE_FILTER");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
}

/**
 *  @brief Runs one named test, honoring @p filter, timing it, and reporting the outcome.
 *  @param[in] filter Substring matched against @p name, or @c nullptr to run everything.
 *  @param[in] name The test's "suite.name" label, which is also its filter key.
 *  @param[in] test_function A function taking no arguments.
 *  @return The number of failures - 0 on success or when skipped, 1 when the test threw.
 *
 *  A failed assertion aborts before this returns, so the count covers only thrown exceptions; naming
 *  them here beats a bare @c what() at the top of @c main. The started line prints before the call, so
 *  a hard crash leaves the running test as the last thing on stdout.
 *
 *  Takes a function pointer rather than any callable on purpose: a suite that grows a defaulted
 *  parameter stops being a @c void() and would otherwise hide behind a lambda at every call site
 *  instead of failing here.
 */
inline std::size_t run_test(char const *filter, char const *name, void (*test_function)()) noexcept {
    if (filter && filter[0] != '\0' && !std::strstr(name, filter)) {
        print_line(stdout, "- {} ... skipped (SMASHTABLE_FILTER)", name);
        std::fflush(stdout);
        return 0;
    }

    print_line(stdout, "- {} ...", name);
    std::fflush(stdout);
    auto const started = std::chrono::steady_clock::now();
    try {
        test_function();
    }
    catch (std::exception const &error) {
        print_line(stderr, "- {} ... FAILED: {}", name, error.what());
        std::fflush(stderr);
        return 1;
    }
    double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    print_line(stdout, "- {} ... ok ({:.2f} s)", name, seconds);
    std::fflush(stdout);
    return 0;
}

/** @brief Reports whether every test passed, printing the verdict. Use its result as @c main's status. */
inline int report_test_failures(std::size_t failures) noexcept {
    if (failures != 0) {
        print_line(stderr, "\n{} test(s) failed.", failures);
        return 1;
    }
    print_line(stdout, "\nAll tests passed!");
    return 0;
}

#pragma endregion Test Runner

#pragma region Container Helpers

/** @brief Clears a container whether or not its @c clear reports a status. */
template <typename container_type_>
void clear_container(container_type_ &container) noexcept {
    if constexpr (std::is_void_v<decltype(container.clear())>) container.clear();
    else { [[maybe_unused]] auto const status = container.clear(); }
}

#pragma endregion Container Helpers

} // namespace ashvardanian::smashtable::scripts
