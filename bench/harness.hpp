/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date September 25, 2026
 *  @brief Harness for the benchmarks - formatted lines straight onto a stream, and the kit lists.
 */
#pragma once
#include <cstddef> // `std::ptrdiff_t`
#include <cstdio>  // `std::FILE`, `std::fputc`

#include <format>  // `std::format_to`, `std::format_string`
#include <utility> // `std::exchange`, `std::forward`

#include <smashtable/row_search.hpp> // `every_row_kit_k`, `row_kit_t`, `name_of`

namespace ashvardanian::smashtable::bench {

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

/** Prints one capability line, naming each kit @p holds accepts. */
inline void print_row_kits(char const *label, bool (*holds)(row_kit_t) noexcept) noexcept {
    file_output_iterator_t output = std::format_to(file_output_iterator_t {stdout}, "- {}:", label);
    char const *separator = " ";
    for (row_kit_t const kit : every_row_kit_k)
        if (holds(kit)) output = std::format_to(output, "{}{}", std::exchange(separator, ","), name_of(kit));
    std::fputc('\n', stdout);
}

} // namespace ashvardanian::smashtable::bench
