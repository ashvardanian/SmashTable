/**
 *  @file include/smashtable/immutable_elias_fano.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief An immutable Elias-Fano sequence: non-decreasing integers kept in a few bits apiece and
 *      read back by position in constant time.
 *
 *  @section immutable_elias_fano_layout Layout
 *
 *  Every value splits at @c low_bits, the floor of log2(universe / count). The low parts are
 *  packed side by side, @c low_bits each. The high parts are stored in unary: value @c i sets bit
 *  `(value >> low_bits) + i` of a vector that never exceeds `count + (universe >> low_bits) + 1`
 *  bits, so the position of the @c i -th set bit, less @c i, is the high part of value @c i.
 *
 *  @section immutable_elias_fano_select Select
 *
 *  The position of every @c select_stride_k -th set bit is sampled, a quarter of a bit per value.
 *  A @c select jumps to the sample at or below its position, counts set bits a word at a time, and
 *  halves the last word down to the bit, so it reads a handful of words whatever the count.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <bit>         // `std::bit_width`, `std::popcount`
#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span`
#include <type_traits> // `std::is_unsigned_v`
#include <utility>     // `std::exchange`, `std::move`

#include "smashtable/basic_vector.hpp"

namespace ashvardanian::smashtable {

/** An immutable sequence of non-decreasing unsigned integers, built once from values pulled in
 *  order and read back by position - the offsets of variable-length rows, a few bits apiece. */
template <typename value_type_ = std::uint64_t, typename allocator_type_ = default_allocator<std::uint64_t>>
class immutable_elias_fano {
  public:
    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility
    using allocator_t = allocator_type_;
    using allocator_type = allocator_t; // ? STL compatibility

    /** Set bits between consecutive samples of the select index. */
    static constexpr std::size_t select_stride_k = 256;

    static_assert(std::is_unsigned_v<value_t> && sizeof(value_t) <= sizeof(std::uint64_t),
                  "the sequence splits unsigned values of at most 64 bits");
    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "immutable_elias_fano requires allocators that propagate on move assignment");

  private:
    using words_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::uint64_t>;
    using words_t = basic_vector<std::uint64_t, words_allocator_t>;

    words_t lows_;
    words_t highs_;
    words_t samples_; // ? Positions in `highs_` of every `select_stride_k`-th set bit
    std::size_t size_ {0};
    unsigned low_bits_ {0};

  public:
    immutable_elias_fano() noexcept = default;

    explicit immutable_elias_fano(allocator_t allocator) noexcept
        : lows_(words_allocator_t(allocator)), highs_(words_allocator_t(allocator)),
          samples_(words_allocator_t(allocator)) {}

    immutable_elias_fano(immutable_elias_fano &&other) noexcept
        : lows_(std::move(other.lows_)), highs_(std::move(other.highs_)), samples_(std::move(other.samples_)),
          size_(std::exchange(other.size_, 0)), low_bits_(std::exchange(other.low_bits_, 0)) {}

    immutable_elias_fano &operator=(immutable_elias_fano &&other) noexcept {
        if (this == &other) return *this;
        lows_ = std::move(other.lows_);
        highs_ = std::move(other.highs_);
        samples_ = std::move(other.samples_);
        size_ = std::exchange(other.size_, 0);
        low_bits_ = std::exchange(other.low_bits_, 0);
        return *this;
    }

    immutable_elias_fano(immutable_elias_fano const &) = delete;
    immutable_elias_fano &operator=(immutable_elias_fano const &) = delete;

    /**
     *  @brief Builds the sequence of @p count values, none above @p universe, pulled in order
     *      through @p fill, so no caller ever holds them all at once.
     *
     *  @param[in] fill Answers `expected<std::span<value_t const>>` for a position, holding the
     *      values from that position on - at least one while any remain, possibly all of them.
     *  @return The sequence, the first failure @p fill reports, @c invalid_argument_k when the
     *      values decrease, pass @p universe or fall short of @p count, or @c out_of_memory_heap_k.
     */
    template <typename fill_type_>
    static expected<immutable_elias_fano> make(std::size_t count, value_t universe, fill_type_ &&fill,
                                               allocator_t allocator = {}) noexcept {
        if (count > size_max_k / 64) return out_of_memory_heap_k;
        immutable_elias_fano sequence(std::move(allocator));
        unsigned const low_bits =
            count && universe / count ? static_cast<unsigned>(std::bit_width(std::uint64_t(universe / count)) - 1) : 0;
        std::uint64_t const high_bits = count ? count + (std::uint64_t(universe) >> low_bits) + 1 : 0;
        if (status_t const status = sequence.lows_.resize(words_for_(std::uint64_t(count) * low_bits)); failed(status))
            return status;
        if (status_t const status = sequence.highs_.resize(words_for_(high_bits)); failed(status)) return status;
        if (status_t const status = sequence.samples_.resize((count + select_stride_k - 1) / select_stride_k);
            failed(status))
            return status;
        sequence.size_ = count;
        sequence.low_bits_ = low_bits;

        value_t previous = 0;
        for (std::size_t position = 0; position < count;) {
            expected<std::span<value_t const>> pulled = fill(position);
            if (failed(pulled)) return pulled.status();
            std::span<value_t const> const values = *pulled;
            if (values.empty() || values.size() > count - position) return invalid_argument_k;
            for (value_t const value : values) {
                if (value < previous || value > universe) return invalid_argument_k;
                sequence.append_(position++, value);
                previous = value;
            }
        }
        return expected<immutable_elias_fano>(std::move(sequence), success_k);
    }

    /** Builds the sequence of @p sorted, held whole. */
    static expected<immutable_elias_fano> make(std::span<value_t const> sorted, allocator_t allocator = {}) noexcept {
        value_t const universe = sorted.empty() ? value_t {0} : sorted.back();
        return make(
            sorted.size(), universe,
            [sorted](std::size_t first) noexcept {
                return expected<std::span<value_t const>>(sorted.subspan(first), success_k);
            },
            std::move(allocator));
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /** The bytes this sequence stores: the low parts, the high parts and the select samples. */
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return (lows_.size() + highs_.size() + samples_.size()) * sizeof(std::uint64_t);
    }

    /** The value at @p position. */
    [[nodiscard]] value_t select(std::size_t position) const noexcept {
        assert(position < size_ && "select takes a position below size");
        std::uint64_t const high = select_high_bit_(position) - position;
        if (!low_bits_) return static_cast<value_t>(high);
        std::uint64_t const bit = std::uint64_t(position) * low_bits_;
        std::uint64_t low = lows_[bit / 64] >> (bit % 64);
        if (bit % 64 + low_bits_ > 64) low |= lows_[bit / 64 + 1] << (64 - bit % 64);
        return static_cast<value_t>((high << low_bits_) | (low & low_mask_()));
    }

  private:
    static constexpr std::size_t words_for_(std::uint64_t bits) noexcept {
        return static_cast<std::size_t>((bits + 63) / 64);
    }

    std::uint64_t low_mask_() const noexcept { return (std::uint64_t(1) << low_bits_) - 1; }

    void append_(std::size_t position, value_t value) noexcept {
        std::uint64_t const widened = value;
        if (low_bits_) {
            std::uint64_t const low = widened & low_mask_();
            std::uint64_t const bit = std::uint64_t(position) * low_bits_;
            lows_[bit / 64] |= low << (bit % 64);
            if (bit % 64 + low_bits_ > 64) lows_[bit / 64 + 1] |= low >> (64 - bit % 64);
        }
        std::uint64_t const high_bit = (widened >> low_bits_) + position;
        highs_[high_bit / 64] |= std::uint64_t(1) << (high_bit % 64);
        if (position % select_stride_k == 0) samples_[position / select_stride_k] = high_bit;
    }

    /** Where the @p position -th set bit of the high parts sits: from the sample at or below it,
     *  whole words at a time. */
    std::uint64_t select_high_bit_(std::size_t position) const noexcept {
        std::uint64_t const sampled = samples_[position / select_stride_k];
        std::size_t remaining = position % select_stride_k;
        if (!remaining) return sampled;
        std::size_t word_index = static_cast<std::size_t>(sampled / 64);
        std::uint64_t word = highs_[word_index] & (~std::uint64_t(0) << (sampled % 64));
        word &= word - 1; // The sampled bit is the lowest one left, and it is already counted.
        while (true) {
            std::size_t const ones = static_cast<std::size_t>(std::popcount(word));
            if (remaining <= ones) return std::uint64_t(word_index) * 64 + select_in_word_(word, remaining - 1);
            remaining -= ones;
            word = highs_[++word_index];
        }
    }

    /** The offset of the @p rank -th set bit of @p word, counted from zero, found by halving. */
    static unsigned select_in_word_(std::uint64_t word, std::size_t rank) noexcept {
        unsigned offset = 0;
        for (unsigned half = 32; half; half >>= 1) {
            std::uint64_t const lower = word & ((std::uint64_t(1) << half) - 1);
            std::size_t const ones = static_cast<std::size_t>(std::popcount(lower));
            if (rank < ones) {
                word = lower;
                continue;
            }
            rank -= ones;
            word >>= half;
            offset += half;
        }
        return offset;
    }
};

} // namespace ashvardanian::smashtable
