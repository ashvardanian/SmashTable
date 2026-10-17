/**
 *  @file include/smashtable/immutable_packed_array.hpp
 *  @author Ash Vardanian
 *  @date October 16, 2026
 *  @brief An immutable packed array: unsigned integers kept as their distance from a base, in the
 *      fewest bits the widest distance needs, and read back by position in constant time.
 *
 *  @section immutable_packed_array_layout Layout
 *
 *  Value @c i is stored as its distance from the base, in bits i × width up to (i + 1) × width of a
 *  run of 64-bit words, where the width is the bit width of the maximum less the base. A field
 *  therefore crosses at most one word boundary, and a zero width stores no words at all.
 *
 *  @section immutable_packed_array_saved Saved Form
 *
 *  @c save writes a @c packed_array_header_t and then the packed words, all in this machine's byte
 *  order. @c immutable_packed_array_view reads that form in place, from borrowed bytes at any
 *  alignment, so a file mapped into memory is read without a copy.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint64_t`
#include <cstring> // `std::memcpy`

#include <bit>     // `std::bit_width`
#include <limits>  // `std::numeric_limits`
#include <memory>  // `std::allocator_traits`
#include <span>    // `std::span`
#include <utility> // `std::exchange`, `std::move`

#include "smashtable/basic_vector.hpp"
#include "smashtable/row_search.hpp" // `row_kit`, which the befriended Elias-Fano sequence is constrained by

namespace ashvardanian::smashtable {

/** What @c save writes ahead of the packed words, one 64-bit word per field, so a view needs
 *  nothing else to read them. */
struct packed_array_header_t {

    /** How many values the array holds. */
    std::uint64_t count {0};

    /** The distance every stored value is measured from. */
    std::uint64_t base {0};

    /** How many bits each value takes. */
    std::uint64_t bits {0};
};

/** How many 64-bit words hold @p count fields of @p bits bits each. */
[[nodiscard]] constexpr std::size_t packed_words_for_(std::uint64_t count, unsigned bits) noexcept {
    return static_cast<std::size_t>((count * bits + 63) / 64);
}

/** The @p bits -bit field starting at bit @p first of the words at @p words, loaded at any
 *  alignment in this machine's byte order. Takes 1 to 64 bits. */
[[nodiscard]] inline std::uint64_t unpack_bits_(std::byte const *words, std::uint64_t first, unsigned bits) noexcept {
    std::size_t const word = static_cast<std::size_t>(first / 64);
    unsigned const shift = static_cast<unsigned>(first % 64);
    std::uint64_t field {};
    std::memcpy(&field, words + word * sizeof(std::uint64_t), sizeof(std::uint64_t));
    field >>= shift;
    if (shift + bits > 64) {
        std::uint64_t next {};
        std::memcpy(&next, words + (word + 1) * sizeof(std::uint64_t), sizeof(std::uint64_t));
        field |= next << (64 - shift);
    }
    return field & (~std::uint64_t {0} >> (64 - bits));
}

/**
 *  @brief An immutable packed array read in place from bytes it does not own, as saved.
 *
 *  A pointer and three fields, so it copies freely and stays valid exactly as long as the bytes it
 *  was made over.
 */
template <std::unsigned_integral value_type_ = std::uint64_t>
class immutable_packed_array_view {
  public:
    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility

    static_assert(sizeof(value_t) <= sizeof(std::uint64_t), "the array packs unsigned values of at most 64 bits");

  private:
    std::byte const *words_ {nullptr};
    std::size_t size_ {0};
    value_t base_ {0};
    unsigned bits_ {0};

    constexpr immutable_packed_array_view(std::byte const *words, std::size_t count, value_t base,
                                          unsigned bits) noexcept
        : words_(words), size_(count), base_(base), bits_(bits) {}

  public:
    constexpr immutable_packed_array_view() noexcept = default;

    /**
     *  @brief Reads the saved form at the front of @p bytes, which may sit at any alignment.
     *  @return The view, or @c invalid_argument_k when @p bytes ends before the words its header
     *      names, or the header holds a base or a width that @c value_t cannot.
     */
    static expected<immutable_packed_array_view> make(std::span<std::byte const> bytes) noexcept {
        packed_array_header_t header;
        if (bytes.size() < sizeof(header)) return invalid_argument_k;
        std::memcpy(&header, bytes.data(), sizeof(header));
        std::uint64_t const widest = std::numeric_limits<value_t>::digits;
        std::uint64_t const limit = std::numeric_limits<value_t>::max();
        if (header.bits > widest || header.base > limit) return invalid_argument_k;
        if (header.bits && (~std::uint64_t {0} >> (64 - header.bits)) > limit - header.base) return invalid_argument_k;
        if (header.count > size_max_k / 64) return invalid_argument_k;
        unsigned const bits = static_cast<unsigned>(header.bits);
        std::size_t const words = packed_words_for_(header.count, bits);
        if ((bytes.size() - sizeof(header)) / sizeof(std::uint64_t) < words) return invalid_argument_k;
        return expected<immutable_packed_array_view>(
            immutable_packed_array_view(bytes.data() + sizeof(header), static_cast<std::size_t>(header.count),
                                        static_cast<value_t>(header.base), bits),
            success_k);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /** How many bits each value takes. */
    [[nodiscard]] unsigned bits() const noexcept { return bits_; }

    /** The distance every stored value is measured from, and the least value it may hold. */
    [[nodiscard]] value_t base() const noexcept { return base_; }

    /** The bytes the saved form spans, so a reader can step past it to whatever follows. */
    [[nodiscard]] std::size_t saved_bytes() const noexcept {
        return sizeof(packed_array_header_t) + packed_words_for_(size_, bits_) * sizeof(std::uint64_t);
    }

    /** The value at @p position. */
    [[nodiscard]] value_t operator[](std::size_t position) const noexcept {
        assert(position < size_ && "a packed array reads positions below its size");
        if (!bits_) return base_;
        return static_cast<value_t>(base_ + unpack_bits_(words_, std::uint64_t(position) * bits_, bits_));
    }
};

/** An immutable array of unsigned integers between a base and a maximum, built once from values
 *  pulled in order, each kept in the bit width of the maximum less the base. */
template <std::unsigned_integral value_type_ = std::uint64_t,
          allocator_of<std::uint64_t> allocator_type_ = default_allocator<std::uint64_t>>
class immutable_packed_array {
  public:
    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility
    using allocator_t = allocator_type_;
    using allocator_type = allocator_t; // ? STL compatibility

    static_assert(sizeof(value_t) <= sizeof(std::uint64_t), "the array packs unsigned values of at most 64 bits");
    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "immutable_packed_array requires allocators that propagate on move assignment");

  private:
    template <typename, allocator_of<std::uint64_t>, row_kit>
    friend class immutable_elias_fano;

    using words_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::uint64_t>;
    using words_t = basic_vector<std::uint64_t, words_allocator_t>;

    words_t words_;
    std::size_t size_ {0};
    value_t base_ {0};
    unsigned bits_ {0};

  public:
    immutable_packed_array() noexcept = default;

    explicit immutable_packed_array(allocator_t allocator) noexcept : words_(words_allocator_t(allocator)) {}

    immutable_packed_array(immutable_packed_array &&other) noexcept
        : words_(std::move(other.words_)), size_(std::exchange(other.size_, 0)),
          base_(std::exchange(other.base_, value_t {0})), bits_(std::exchange(other.bits_, 0)) {}

    immutable_packed_array &operator=(immutable_packed_array &&other) noexcept {
        if (this == &other) return *this;
        words_ = std::move(other.words_);
        size_ = std::exchange(other.size_, 0);
        base_ = std::exchange(other.base_, value_t {0});
        bits_ = std::exchange(other.bits_, 0);
        return *this;
    }

    immutable_packed_array(immutable_packed_array const &) = delete;
    immutable_packed_array &operator=(immutable_packed_array const &) = delete;

    /**
     *  @brief Builds the array of @p count values from @p base to @p maximum inclusive, pulled in
     *      order through @p fill, so no caller ever holds them all at once.
     *
     *  @param[in] fill Answers `expected<std::span<value_t const>>` for a position, holding the
     *      values from that position on - at least one while any remain, possibly all of them.
     *  @return The array, the first failure @p fill reports, @c invalid_argument_k when @p base
     *      exceeds @p maximum or a value falls outside them or short of @p count, or
     *      @c out_of_memory_heap_k.
     */
    template <typename fill_type_>
    static expected<immutable_packed_array> make(std::size_t count, value_t base, value_t maximum, fill_type_ &&fill,
                                                 allocator_t allocator = {}) noexcept {
        if (base > maximum) return invalid_argument_k;
        if (count > size_max_k / 64) return out_of_memory_heap_k;
        immutable_packed_array array(std::move(allocator));
        unsigned const bits = static_cast<unsigned>(std::bit_width(std::uint64_t(maximum) - base));
        if (status_t const status = array.reset_(count, base, bits); failed(status)) return status;

        for (std::size_t position = 0; position < count;) {
            expected<std::span<value_t const>> pulled = fill(position);
            if (failed(pulled)) return pulled.status();
            std::span<value_t const> const values = *pulled;
            if (values.empty() || values.size() > count - position) return invalid_argument_k;
            for (value_t const value : values) {
                if (value < base || value > maximum) return invalid_argument_k;
                array.pack_(position++, std::uint64_t(value) - base);
            }
        }
        return expected<immutable_packed_array>(std::move(array), success_k);
    }

    /** Builds the array of @p values, held whole, based at the least of them and as wide as the
     *  greatest needs. */
    static expected<immutable_packed_array> make(std::span<value_t const> values, allocator_t allocator = {}) noexcept {
        value_t least = values.empty() ? value_t {0} : values.front();
        value_t greatest = least;
        for (value_t const value : values) {
            if (value < least) least = value;
            if (value > greatest) greatest = value;
        }
        return make(
            values.size(), least, greatest,
            [values](std::size_t first) noexcept {
                return expected<std::span<value_t const>>(values.subspan(first), success_k);
            },
            std::move(allocator));
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /** How many bits each value takes: the bit width of the maximum less the base. */
    [[nodiscard]] unsigned bits() const noexcept { return bits_; }

    /** The distance every stored value is measured from, and the least value it may hold. */
    [[nodiscard]] value_t base() const noexcept { return base_; }

    /** The bytes the packed words take, which a zero width keeps at zero. */
    [[nodiscard]] std::size_t size_bytes() const noexcept { return words_.size() * sizeof(std::uint64_t); }

    /** The bytes @c save writes: the header and then the packed words. */
    [[nodiscard]] std::size_t saved_bytes() const noexcept { return sizeof(packed_array_header_t) + size_bytes(); }

    /**
     *  @brief Writes the saved form into the front of @p bytes, for
     *      @c immutable_packed_array_view to read in place.
     *  @return @c success_k, or @c invalid_argument_k when @p bytes is shorter than
     *      @c saved_bytes, in which case nothing is written.
     */
    status_t save(std::span<std::byte> bytes) const noexcept {
        if (bytes.size() < saved_bytes()) return invalid_argument_k;
        packed_array_header_t const header {.count = size_, .base = base_, .bits = bits_};
        std::memcpy(bytes.data(), &header, sizeof(header));
        if (!words_.empty()) std::memcpy(bytes.data() + sizeof(header), words_.data(), size_bytes());
        return success_k;
    }

    /** The value at @p position. */
    [[nodiscard]] value_t operator[](std::size_t position) const noexcept {
        assert(position < size_ && "a packed array reads positions below its size");
        if (!bits_) return base_;
        std::byte const *words = reinterpret_cast<std::byte const *>(words_.data());
        return static_cast<value_t>(base_ + unpack_bits_(words, std::uint64_t(position) * bits_, bits_));
    }

  private:
    /** Sizes the array for @p count fields of @p bits bits each, based at @p base. */
    status_t reset_(std::size_t count, value_t base, unsigned bits) noexcept {
        unsigned const width = count ? bits : 0;
        if (status_t const status = words_.resize(packed_words_for_(count, width)); failed(status)) return status;
        size_ = count;
        base_ = count ? base : value_t {0};
        bits_ = width;
        return success_k;
    }

    /** Ors @p offset into the field at @p position, which starts zeroed. */
    void pack_(std::size_t position, std::uint64_t offset) noexcept {
        if (!bits_) return;
        std::uint64_t const first = std::uint64_t(position) * bits_;
        std::size_t const word = static_cast<std::size_t>(first / 64);
        unsigned const shift = static_cast<unsigned>(first % 64);
        words_[word] |= offset << shift;
        if (shift + bits_ > 64) words_[word + 1] |= offset >> (64 - shift);
    }
};

} // namespace ashvardanian::smashtable
