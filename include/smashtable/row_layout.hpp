/**
 *  @brief Shared vocabulary of the row-searched layouts: the wide key, the media a row is sized to, and
 *    where each key of a row sits.
 *  @author Ash Vardanian
 *  @file include/smashtable/row_layout.hpp
 *  @date September 16, 2026
 *
 *  A row is the unit every ordered layout here searches: a span of keys wide enough to be worth one
 *  vectorized pass. This header says what a key may be and how a row is laid out; @c row_search.hpp says
 *  how one is searched, and the layouts above both say what they do with the answer.
 */
#pragma once
#include <cstddef> // `std::size_t`, `offsetof`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`

#include <bit>  // `std::popcount`
#include <span> // `std::span`

#include "shared.hpp"

namespace ashvardanian::smashtable {

#pragma region Keys and Media

/**
 *  A 16-byte key as two words, each holding eight bytes read big-endian, so ordering the words orders the bytes as
 *  @c memcmp would. An integer identity below 2^64 has a zero @c high and its value in @c low.
 */
struct key128_t {
    std::uint64_t high;
    std::uint64_t low;

    /** Reads sixteen bytes in @c memcmp order. */
    [[nodiscard]] static constexpr key128_t from_bytes(std::span<std::byte const, 16> bytes) noexcept {
        key128_t key {0, 0};
        for (std::size_t index = 0; index < 8; ++index)
            key.high = (key.high << 8) | static_cast<std::uint64_t>(bytes[index]);
        for (std::size_t index = 8; index < 16; ++index)
            key.low = (key.low << 8) | static_cast<std::uint64_t>(bytes[index]);
        return key;
    }

    /** Writes the sixteen bytes @c from_bytes reads. */
    constexpr void to_bytes(std::span<std::byte, 16> bytes) const noexcept {
        for (std::size_t index = 0; index < 8; ++index) bytes[index] = static_cast<std::byte>(high >> (56 - 8 * index));
        for (std::size_t index = 0; index < 8; ++index)
            bytes[8 + index] = static_cast<std::byte>(low >> (56 - 8 * index));
    }

    friend constexpr auto operator<=>(key128_t const &, key128_t const &) noexcept = default;
    friend constexpr bool operator==(key128_t const &, key128_t const &) noexcept = default;
};

static_assert(sizeof(key128_t) == 16 && offsetof(key128_t, low) == 8 && std::is_trivially_copyable_v<key128_t>,
              "interleaved kits load a key128_t row as a run of 64-bit words");

/** The key types every kit searches. */
template <typename key_type_>
concept row_searchable_key = std::same_as<key_type_, std::uint32_t> || std::same_as<key_type_, std::int32_t> ||
                             std::same_as<key_type_, std::uint64_t> || std::same_as<key_type_, std::int64_t> ||
                             std::same_as<key_type_, key128_t>;

/**
 *  @brief The row width the ordered layouts take when a caller names none: a 512-byte block.
 *
 *  Sized rather than measured: a cache line is 64 bytes on most cores and 128 on others, and a page is
 *  4096 bytes until an operating system is configured otherwise, so neither is a constant this header
 *  could state. A block sits between them and divides evenly by every key width here.
 */
inline constexpr std::size_t default_row_bytes_k = 512;

/** How many keys of @p key_type_ fill @p medium_bytes, counting both columns of a split 16-byte row. */
template <typename key_type_>
[[nodiscard]] constexpr std::size_t keys_per_row(std::size_t medium_bytes) noexcept {
    return medium_bytes / sizeof(key_type_);
}

#pragma endregion Keys and Media

#pragma region Row Format

/**
 *  How a node of @p keys_per_row_ keys is stored as words: an integer row is one column of keys, and a 16-byte row
 *  is a column of high words followed by a column of low words, so both columns share one medium unit.
 */
template <typename key_type_, std::size_t keys_per_row_>
    requires row_searchable_key<key_type_> && (keys_per_row_ >= 2)
struct row_format {
    using key_t = key_type_;
    using word_t = std::conditional_t<std::same_as<key_type_, key128_t>, std::uint64_t, key_type_>;

    static constexpr bool keys_are_split_k = std::same_as<key_type_, key128_t>;
    static constexpr std::size_t keys_per_row_k = keys_per_row_;
    static constexpr std::size_t words_per_row_k = keys_are_split_k ? 2 * keys_per_row_ : keys_per_row_;
    static constexpr std::size_t bytes_per_row_k = words_per_row_k * sizeof(word_t);

    /** The key unused slots hold, which orders at or after every key so a count below never includes it. */
    static constexpr key_t padding_k = [] {
        if constexpr (keys_are_split_k)
            return key_t {std::numeric_limits<std::uint64_t>::max(), std::numeric_limits<std::uint64_t>::max()};
        else return std::numeric_limits<key_t>::max();
    }();

    [[nodiscard]] static constexpr key_t key_at(word_t const *row, std::size_t index) noexcept {
        if constexpr (keys_are_split_k) return key_t {row[index], row[keys_per_row_ + index]};
        else return row[index];
    }

    static constexpr void store(word_t *row, std::size_t index, key_t key) noexcept {
        if constexpr (keys_are_split_k) {
            row[index] = key.high;
            row[keys_per_row_ + index] = key.low;
        }
        else row[index] = key;
    }
};

#pragma endregion Row Format

} // namespace ashvardanian::smashtable
