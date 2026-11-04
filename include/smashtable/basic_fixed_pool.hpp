/**
 *  @file include/smashtable/basic_fixed_pool.hpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief A fixed array of records with a stack of the free ones' indices, in one allocation.
 *
 *  @section basic_fixed_pool_layout Layout
 *
 *  The records come first and the free indices follow from the next cache line, so the owner is a
 *  pointer and two counts. Every record is built at @c make and lives as long as the pool: @c take
 *  hands out a record as its last holder left it, and @c vacate only returns the index.
 *
 *  @section basic_fixed_pool_generations Generations
 *
 *  A record carrying an unsigned @c generation member gets handles: an index and the generation it
 *  was taken at, which @c vacate steps, so a handle kept past its record's release resolves to
 *  nothing instead of to whoever took the record next.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <concepts>    // `std::same_as`, `std::unsigned_integral`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::allocator_traits`
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <type_traits> // `std::is_nothrow_default_constructible_v`, `std::is_trivially_destructible_v`
#include <utility>     // `std::exchange`, `std::move`

#include "smashtable/shared.hpp"

namespace ashvardanian::smashtable {

/** A record counting its reuses in a @c generation member of the pool's index type, which a handle
 *  compares. The generation shares the index's width, so a narrow index wraps it sooner. */
template <typename value_type_, typename index_type_>
concept generation_tagged = requires(value_type_ &record) {
    { record.generation } -> std::same_as<index_type_ &>;
};

/**
 *  @brief At most @c capacity records of @p value_type_, taken and vacated by index.
 *
 *  @tparam value_type_ The record, default-constructed once per slot at @c make.
 *  @tparam index_type_ Names a record, whose largest value is @c absent_k.
 *  @tparam allocator_type_ Supplies and reclaims the single byte buffer.
 */
template <typename value_type_, std::unsigned_integral index_type_ = std::uint32_t,
          allocator_of<std::byte> allocator_type_ = default_allocator_t>
class basic_fixed_pool {
  public:
    using value_t = value_type_;
    using index_t = index_type_;
    using allocator_t = allocator_type_;

    /** What @c take answers when every record is taken. */
    static constexpr index_t absent_k = std::numeric_limits<index_t>::max();

    /** A record's index and the generation it was taken at, for a tagged record only. */
    struct handle_t {
        index_t index;
        index_t generation;
    };
    static_assert(sizeof(handle_t) == 2 * sizeof(index_t), "a handle carries no padding");

    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "basic_fixed_pool requires allocators that propagate on move assignment");
    static_assert(std::is_nothrow_default_constructible_v<value_t>,
                  "basic_fixed_pool builds every record at make, in noexcept code");
    static_assert(alignof(value_t) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
                  "An over-aligned record is not accommodated by the single default-aligned allocation");

  private:
    std::byte *memory_ {};
    index_t capacity_ {};
    index_t free_count_ {};
    SMASHTABLE_NO_UNIQUE_ADDRESS_ allocator_t allocator_ {};

    static constexpr std::size_t records_bytes_(std::size_t capacity) noexcept {
        return roundup_to_multiple<std::size_t, cache_line_bytes_k>(capacity * sizeof(value_t));
    }

    static constexpr std::size_t memory_bytes_(std::size_t capacity) noexcept {
        return records_bytes_(capacity) + capacity * sizeof(index_t);
    }

    [[nodiscard]] value_t *records_data_() const noexcept { return reinterpret_cast<value_t *>(memory_); }

    [[nodiscard]] index_t *free_indices_() const noexcept {
        return reinterpret_cast<index_t *>(memory_ + records_bytes_(capacity_));
    }

    void release_() noexcept {
        if (!memory_) return;
        if constexpr (!std::is_trivially_destructible_v<value_t>)
            for (value_t &record : records()) record.~value_t();
        allocator_.deallocate(memory_, memory_bytes_(capacity_));
        memory_ = nullptr;
        capacity_ = 0;
        free_count_ = 0;
    }

  public:
    basic_fixed_pool() noexcept = default;
    explicit basic_fixed_pool(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}
    ~basic_fixed_pool() noexcept { release_(); }

    basic_fixed_pool(basic_fixed_pool &&other) noexcept
        : memory_(std::exchange(other.memory_, nullptr)), capacity_(std::exchange(other.capacity_, 0)),
          free_count_(std::exchange(other.free_count_, 0)), allocator_(std::move(other.allocator_)) {}

    basic_fixed_pool &operator=(basic_fixed_pool &&other) noexcept {
        if (this == &other) return *this;
        release_();
        memory_ = std::exchange(other.memory_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
        free_count_ = std::exchange(other.free_count_, 0);
        allocator_ = std::move(other.allocator_);
        return *this;
    }

    basic_fixed_pool(basic_fixed_pool const &) = delete;
    basic_fixed_pool &operator=(basic_fixed_pool const &) = delete;

    /**
     *  @brief A pool of @p capacity default-constructed records, all of them free.
     *  @return The pool, which allocates nothing for a zero capacity.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k A capacity reaching @c absent_k, or too large to count in bytes.
     */
    static expected<basic_fixed_pool> make(std::size_t capacity, allocator_t allocator = {}) noexcept {
        if (capacity >= absent_k || capacity > (std::numeric_limits<std::size_t>::max() - cache_line_bytes_k) /
                                                   (sizeof(value_t) + sizeof(index_t)))
            return invalid_argument_k;
        basic_fixed_pool pool(std::move(allocator));
        if (capacity == 0) return expected<basic_fixed_pool>(std::move(pool), success_k);
        pool.memory_ = pool.allocator_.allocate(memory_bytes_(capacity));
        if (!pool.memory_) return out_of_memory_heap_k;
        pool.capacity_ = static_cast<index_t>(capacity);
        pool.free_count_ = pool.capacity_;
        for (value_t &record : pool.records()) new (&record) value_t();
        // Stacked in reverse, so the first takes hand out ascending indices.
        index_t *const free_indices = pool.free_indices_();
        for (index_t slot = 0; slot != pool.capacity_; ++slot)
            free_indices[slot] = static_cast<index_t>(pool.capacity_ - 1 - slot);
        return expected<basic_fixed_pool>(std::move(pool), success_k);
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t size() const noexcept { return capacity_ - free_count_; }
    [[nodiscard]] bool empty() const noexcept { return free_count_ == capacity_; }
    [[nodiscard]] bool full() const noexcept { return free_count_ == 0; }

    /** Every record, taken or free, in index order. */
    [[nodiscard]] std::span<value_t> records() noexcept { return {records_data_(), capacity_}; }
    [[nodiscard]] std::span<value_t const> records() const noexcept { return {records_data_(), capacity_}; }

    /** The record at @p index, taken or not, without bounds checking. */
    [[nodiscard]] value_t &operator[](index_t index) noexcept {
        assert(index < capacity_ && "index out of bounds");
        return records_data_()[index];
    }
    [[nodiscard]] value_t const &operator[](index_t index) const noexcept {
        assert(index < capacity_ && "index out of bounds");
        return records_data_()[index];
    }

    /** The index of @p record, which must be one of @c records. */
    [[nodiscard]] index_t index_of(value_t const &record) const noexcept {
        assert(&record >= records_data_() && &record < records_data_() + capacity_ && "a record of another pool");
        return static_cast<index_t>(&record - records_data_());
    }

    /** Takes a free record, as its last holder left it, or answers @c absent_k when none is. */
    [[nodiscard]] index_t take() noexcept { return free_count_ ? free_indices_()[--free_count_] : absent_k; }

    /** Frees the taken record at @p index, stepping its generation so older handles go stale. */
    void vacate(index_t index) noexcept {
        assert(index < capacity_ && free_count_ < capacity_ && "vacating a record that was never taken");
        if constexpr (generation_tagged<value_t, index_t>) ++records_data_()[index].generation;
        free_indices_()[free_count_++] = index;
    }

    /** A handle to the record at @p index, which @c at answers until it is vacated. */
    [[nodiscard]] handle_t handle_of(index_t index) const noexcept
        requires generation_tagged<value_t, index_t>
    {
        assert(index < capacity_ && "index out of bounds");
        return {index, records_data_()[index].generation};
    }

    /** The record @p handle names, or null once that record was vacated. */
    [[nodiscard]] value_t *at(handle_t handle) noexcept
        requires generation_tagged<value_t, index_t>
    {
        if (handle.index >= capacity_) return nullptr;
        value_t &record = records_data_()[handle.index];
        return record.generation == handle.generation ? &record : nullptr;
    }
    [[nodiscard]] value_t const *at(handle_t handle) const noexcept
        requires generation_tagged<value_t, index_t>
    {
        if (handle.index >= capacity_) return nullptr;
        value_t const &record = records_data_()[handle.index];
        return record.generation == handle.generation ? &record : nullptr;
    }
};

static_assert(sizeof(basic_fixed_pool<std::uint64_t>) == 16, "a pointer and two 32-bit counts");

} // namespace ashvardanian::smashtable
