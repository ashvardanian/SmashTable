/**
 *  @file include/smashtable/immutable_elias_fano.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief An immutable Elias-Fano sequence: non-decreasing integers kept in a few bits apiece and
 *      read back by position in constant time.
 *
 *  @section immutable_elias_fano_layout Layout
 *
 *  Every value splits at @c low_bits, the floor of log2(universe / count). The low parts sit in an
 *  @c immutable_packed_array based at zero, @c low_bits each. The high parts are stored in unary:
 *  value @c i sets bit <tt>(value >> low_bits) + i</tt> of a vector that never exceeds
 *  <tt>count + (universe >> low_bits) + 1</tt> bits, so the position of the @c i -th set bit, less @c i,
 *  is the high part of value @c i.
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
#include <type_traits> // `std::conditional_t`, `std::is_unsigned_v`
#include <utility>     // `std::exchange`, `std::move`

#include "smashtable/basic_vector.hpp"
#include "smashtable/immutable_packed_array.hpp"
#include "smashtable/row_search.hpp" // `row_kit`, `native_row_kit_t`

namespace ashvardanian::smashtable {

/** An immutable sequence of non-decreasing unsigned integers, built once from values pulled in
 *  order and read back by position - the offsets of variable-length rows, a few bits apiece.
 *  @p row_kit_type_ scans the high parts in @c select where it has a scan of its own. */
template <typename value_type_ = std::uint64_t,
          allocator_of<std::uint64_t> allocator_type_ = default_allocator<std::uint64_t>,
          row_kit row_kit_type_ = native_row_kit_t>
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
    using lows_t = immutable_packed_array<value_t, words_allocator_t>;

    lows_t lows_;
    words_t highs_;
    words_t samples_; // ? Positions in `highs_` of every `select_stride_k`-th set bit
    std::size_t size_ {0};

  public:
    immutable_elias_fano() noexcept = default;

    explicit immutable_elias_fano(allocator_t allocator) noexcept
        : lows_(words_allocator_t(allocator)), highs_(words_allocator_t(allocator)),
          samples_(words_allocator_t(allocator)) {}

    immutable_elias_fano(immutable_elias_fano &&other) noexcept
        : lows_(std::move(other.lows_)), highs_(std::move(other.highs_)), samples_(std::move(other.samples_)),
          size_(std::exchange(other.size_, 0)) {}

    immutable_elias_fano &operator=(immutable_elias_fano &&other) noexcept {
        if (this == &other) return *this;
        lows_ = std::move(other.lows_);
        highs_ = std::move(other.highs_);
        samples_ = std::move(other.samples_);
        size_ = std::exchange(other.size_, 0);
        return *this;
    }

    immutable_elias_fano(immutable_elias_fano const &) = delete;
    immutable_elias_fano &operator=(immutable_elias_fano const &) = delete;

    /**
     *  @brief Builds the sequence of @p count values, none above @p universe, pulled in order
     *      through @p fill, so no caller ever holds them all at once.
     *
     *  @param[in] count Number of values in the sequence.
     *  @param[in] universe Largest value any of them may take.
     *  @param[in] fill Answers <tt>expected<std::span<value_t const>></tt> for a position, holding the
     *      values from that position on - at least one while any remain, possibly all of them.
     *  @param[in] allocator The allocator instance to use.
     *  @return The sequence.
     *
     *  @par Statuses:
     *  - The first failure @p fill reports.
     *  - @c invalid_argument_k The values decrease, pass @p universe or fall short of @p count.
     */
    template <typename fill_type_>
    static expected<immutable_elias_fano> make(std::size_t count, value_t universe, fill_type_ &&fill,
                                               allocator_t allocator = {}) noexcept {
        if (count > size_max_k / 64) return out_of_memory_heap_k;
        immutable_elias_fano sequence(std::move(allocator));
        unsigned const low_bits =
            count && universe / count ? static_cast<unsigned>(std::bit_width(std::uint64_t(universe / count)) - 1) : 0;
        std::uint64_t const high_bits = count ? count + (std::uint64_t(universe) >> low_bits) + 1 : 0;
        // A kit's select may read seven words past the one it lands in, so those stay zeroed.
        std::size_t const high_words = count ? packed_words_for_(high_bits, 1) + 7 : 0;
        if (status_t const status = sequence.lows_.reset_(count, value_t {0}, low_bits); failed(status)) return status;
        if (status_t const status = sequence.highs_.resize(high_words); failed(status)) return status;
        if (status_t const status = sequence.samples_.resize((count + select_stride_k - 1) / select_stride_k);
            failed(status))
            return status;
        sequence.size_ = count;

        std::uint64_t const low_mask = (std::uint64_t {1} << low_bits) - 1;
        value_t previous = 0;
        for (std::size_t position = 0; position < count;) {
            expected<std::span<value_t const>> pulled = fill(position);
            if (failed(pulled)) return pulled.status();
            std::span<value_t const> const values = *pulled;
            if (values.empty() || values.size() > count - position) return invalid_argument_k;
            for (value_t const value : values) {
                if (value < previous || value > universe) return invalid_argument_k;
                sequence.lows_.pack_(position, std::uint64_t(value) & low_mask);
                sequence.append_high_(position, value, low_bits);
                previous = value;
                ++position;
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
        return lows_.size_bytes() + (highs_.size() + samples_.size()) * sizeof(std::uint64_t);
    }

    /** The value at @p position. */
    [[nodiscard]] value_t select(std::size_t position) const noexcept {
        assert(position < size_ && "select takes a position below size");
        std::uint64_t const high = select_high_bit_(position) - position;
        return static_cast<value_t>((high << lows_.bits()) | lows_[position]);
    }

  private:
    /** Sets the unary bit of @p value at @p position, and samples it every @c select_stride_k. */
    void append_high_(std::size_t position, value_t value, unsigned low_bits) noexcept {
        std::uint64_t const high_bit = (std::uint64_t(value) >> low_bits) + position;
        highs_[high_bit / 64] |= std::uint64_t(1) << (high_bit % 64);
        if (position % select_stride_k == 0) samples_[position / select_stride_k] = high_bit;
    }

    /** Where the @p position -th set bit of the high parts sits: from the sample at or below it,
     *  whole words at a time. */
    std::uint64_t select_high_bit_(std::size_t position) const noexcept {
        std::uint64_t const sampled = samples_[position / select_stride_k];
        std::size_t const remaining = position % select_stride_k;
        if (!remaining) return sampled;
        using select_kit_t = std::conditional_t<select_kit<row_kit_type_>, row_kit_type_, serial_row_kit_t>;
        return select_kit_t::select_set_bit(std::span<std::uint64_t const>(highs_.data(), highs_.size()), sampled,
                                            remaining);
    }
};

} // namespace ashvardanian::smashtable
