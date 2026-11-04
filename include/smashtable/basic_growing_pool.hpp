/**
 *  @file include/smashtable/basic_growing_pool.hpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief A pool of records growing in chunks that never move, whose records are read beside growth
 *      without a claim.
 *
 *  @section basic_growing_pool_layout Layout
 *
 *  The records sit in chunks, each one allocation, and the free indices in a stack of their own.
 *  The table of chunks is inline in the pool and never moves. Every record is built when its chunk
 *  is allocated and lives as long as the pool: @c take hands out a record as its last holder left
 *  it, and @c vacate only returns the index.
 *
 *  @section basic_growing_pool_growth Growth
 *
 *  A @c take finding no free record adds a chunk, up to the limit fixed at @c make. The first chunk
 *  holds @c 2^b records and every later chunk as many as all the chunks before it, so the record at
 *  index @c i sits in chunk @c bit_width(i>>b), less the records before that chunk. The last chunk
 *  is clipped to the limit. Records never move, so their addresses and generations stay; only the
 *  stack of free indices is reallocated, and only while it is empty.
 *
 *  @section basic_growing_pool_readers Readers
 *
 *  @c take and @c vacate need one caller at a time, but @c at, @c operator[] and @c index_of may
 *  run beside them holding no claim. Growth builds every record of a chunk and then publishes its
 *  table entry with a release store; readers load entries with acquire and bound an index by the
 *  limit, so a reader either finds a chunk null and answers nothing or sees all of it. The capacity
 *  and the free indices belong to the caller holding the claim.
 *
 *  @section basic_growing_pool_generations Generations
 *
 *  A record carrying an unsigned @c generation member gets handles, as in @c basic_fixed_pool: an
 *  index and the generation it was taken at, which @c vacate steps, so a handle kept past its
 *  record's release resolves to nothing instead of to whoever took the record next.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::byte`, `std::size_t`, `std::ptrdiff_t`
#include <cstdint> // `std::uint32_t`, `std::uintptr_t`

#include <bit>         // `std::bit_ceil`, `std::bit_width`, `std::countr_zero`
#include <concepts>    // `std::unsigned_integral`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::allocator_traits`
#include <new>         // placement `new`
#include <type_traits> // `std::is_nothrow_default_constructible_v`, `std::is_trivially_destructible_v`
#include <utility>     // `std::exchange`, `std::move`

#include "smashtable/basic_fixed_pool.hpp" // `generation_tagged`
#include "smashtable/shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief Records of @p value_type_, taken and vacated by index, growing chunk by chunk up to a
 *      limit without moving a record, and readable by index or handle from other threads while the
 *      one owner grows it.
 *
 *  @tparam value_type_ The record, default-constructed once per slot when its chunk is allocated.
 *  @tparam index_type_ Names a record, whose largest value is @c absent_k.
 *  @tparam allocator_type_ Supplies and reclaims the chunks and the stack of free indices.
 */
template <typename value_type_, std::unsigned_integral index_type_ = std::uint32_t,
          allocator_of<std::byte> allocator_type_ = default_allocator_t>
class basic_growing_pool {
  public:
    using value_t = value_type_;
    using index_t = index_type_;
    using allocator_t = allocator_type_;

    /** What @c take answers when every record is taken and no more may be added. */
    static constexpr index_t absent_k = std::numeric_limits<index_t>::max();

    /** The most records a pool holds, every index but @c absent_k. */
    static constexpr std::size_t capacity_max_k = absent_k - 1;

    /** A record's index and the generation it was taken at, for a tagged record only. */
    struct handle_t {
        index_t index;
        index_t generation;
    };
    static_assert(sizeof(handle_t) == 2 * sizeof(index_t), "a handle carries no padding");

    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "basic_growing_pool requires allocators that propagate on move assignment");
    static_assert(std::is_nothrow_default_constructible_v<value_t>,
                  "basic_growing_pool builds every record with its chunk, in noexcept code");

    /** A forward walk over every record, taken or free, in index order. */
    template <typename pool_type_, typename record_type_>
    class records_walk {
        pool_type_ *pool_ {nullptr};
        index_t index_ {0};

      public:
        using value_type = value_t;
        using difference_type = std::ptrdiff_t;

        records_walk() noexcept = default;
        records_walk(pool_type_ *pool, index_t index) noexcept : pool_(pool), index_(index) {}

        [[nodiscard]] record_type_ &operator*() const noexcept { return (*pool_)[index_]; }
        records_walk &operator++() noexcept {
            ++index_;
            return *this;
        }
        records_walk operator++(int) noexcept {
            records_walk const previous = *this;
            ++index_;
            return previous;
        }
        [[nodiscard]] friend bool operator==(records_walk const &left, records_walk const &right) noexcept {
            return left.index_ == right.index_;
        }
    };

    /** Every record of a pool, taken or free, in index order, which a range-for walks. */
    template <typename pool_type_, typename record_type_>
    class records_range {
        pool_type_ *pool_;

      public:
        explicit records_range(pool_type_ *pool) noexcept : pool_(pool) {}
        [[nodiscard]] records_walk<pool_type_, record_type_> begin() const noexcept { return {pool_, 0}; }
        [[nodiscard]] records_walk<pool_type_, record_type_> end() const noexcept {
            return {pool_, static_cast<index_t>(pool_->capacity())};
        }
        [[nodiscard]] std::size_t size() const noexcept { return pool_->capacity(); }
    };

  private:
    /** Allocates the chunks, so an over-aligned record gets its alignment from the allocator. */
    using records_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<value_t>;

    /** More chunks than doubling from one record takes to reach @c absent_k. */
    static constexpr std::size_t chunks_capacity_k = std::numeric_limits<index_t>::digits + 1;

    // Each entry is published with release and loaded with acquire, and the table never moves.
    std::uintptr_t chunks_[chunks_capacity_k] {};
    index_t *free_indices_ {nullptr};
    index_t capacity_ {};
    index_t free_count_ {};
    index_t capacity_limit_ {};
    index_t first_chunk_ {}; // ? A power of two, though the limit may clip chunk 0
    index_t chunks_count_ {};
    SMASHTABLE_NO_UNIQUE_ADDRESS_ allocator_t allocator_ {};

    /** The index of the first record of chunk @p chunk, which is also how many records the chunk
     *  spans. */
    [[nodiscard]] std::size_t chunk_start_(std::size_t chunk) const noexcept {
        return chunk == 0 ? 0 : std::size_t(first_chunk_) << (chunk - 1);
    }

    /** Records chunk @p chunk is allocated with, its span clipped to the limit. */
    [[nodiscard]] std::size_t chunk_held_(std::size_t chunk) const noexcept {
        std::size_t const start = chunk_start_(chunk);
        std::size_t const end = chunk == 0 ? first_chunk_ : start * 2;
        return (end < capacity_limit_ ? end : std::size_t(capacity_limit_)) - start;
    }

    /** The chunk record @p index sits in, by the geometry alone. */
    [[nodiscard]] std::size_t chunk_of_(index_t index) const noexcept {
        unsigned const shift = static_cast<unsigned>(std::countr_zero(first_chunk_));
        return static_cast<std::size_t>(std::bit_width(static_cast<std::size_t>(index) >> shift));
    }

    /** Chunk @p chunk loaded with acquire, null until it is published, so a reader holding no claim
     *  either misses it or sees every record built. */
    [[nodiscard]] value_t *chunk_(std::size_t chunk) const noexcept {
        return reinterpret_cast<value_t *>(atomic_load(chunks_[chunk], memory_order_acquire_k));
    }

    /** The record at @p index, or null past the limit or before its chunk is published. */
    [[nodiscard]] value_t *published_record_(index_t index) const noexcept {
        if (index >= capacity_limit_) return nullptr;
        std::size_t const chunk = chunk_of_(index);
        value_t *const records = chunk_(chunk);
        return records ? records + (index - chunk_start_(chunk)) : nullptr;
    }

    void free_chunk_(value_t *records, std::size_t count) noexcept {
        if constexpr (!std::is_trivially_destructible_v<value_t>)
            for (std::size_t offset = 0; offset != count; ++offset) records[offset].~value_t();
        records_allocator_t(allocator_).deallocate(records, count);
    }

    /** Adds the next chunk, free, and a stack wide enough for every record, or refuses at the
     *  limit; the stack is empty whenever this runs, so nothing is copied out of the old one. */
    [[nodiscard]] status_t grow_() noexcept {
        if (capacity_ >= capacity_limit_ || chunks_count_ == chunks_capacity_k) return capacity_exhausted_k;
        std::size_t const count = chunk_held_(chunks_count_);
        std::size_t const grown = std::size_t(capacity_) + count;
        value_t *const records = records_allocator_t(allocator_).allocate(count);
        if (!records) return out_of_memory_heap_k;
        for (std::size_t offset = 0; offset != count; ++offset) new (records + offset) value_t();
        std::byte *const stack = allocator_.allocate(grown * sizeof(index_t));
        if (!stack) {
            free_chunk_(records, count);
            return out_of_memory_heap_k;
        }
        if (free_indices_)
            allocator_.deallocate(reinterpret_cast<std::byte *>(free_indices_), capacity_ * sizeof(index_t));
        free_indices_ = reinterpret_cast<index_t *>(stack);
        // Readers hold no claim, so the chunk is released only once every record is built.
        atomic_ref<std::uintptr_t>(chunks_[chunks_count_++])
            .store(reinterpret_cast<std::uintptr_t>(records), memory_order_release_k);
        // Stacked in reverse, so the first takes hand out ascending indices.
        for (std::size_t slot = 0; slot != count; ++slot) free_indices_[slot] = static_cast<index_t>(grown - 1 - slot);
        free_count_ = static_cast<index_t>(count);
        capacity_ = static_cast<index_t>(grown);
        return success_k;
    }

    void release_() noexcept {
        for (std::size_t chunk = 0; chunk != chunks_count_; ++chunk) {
            free_chunk_(chunk_(chunk), chunk_held_(chunk));
            chunks_[chunk] = 0;
        }
        if (free_indices_)
            allocator_.deallocate(reinterpret_cast<std::byte *>(free_indices_), capacity_ * sizeof(index_t));
        free_indices_ = nullptr;
        capacity_ = free_count_ = capacity_limit_ = first_chunk_ = chunks_count_ = 0;
    }

    void steal_(basic_growing_pool &other) noexcept {
        for (std::size_t chunk = 0; chunk != chunks_capacity_k; ++chunk)
            chunks_[chunk] = std::exchange(other.chunks_[chunk], 0);
        free_indices_ = std::exchange(other.free_indices_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
        free_count_ = std::exchange(other.free_count_, 0);
        capacity_limit_ = std::exchange(other.capacity_limit_, 0);
        first_chunk_ = std::exchange(other.first_chunk_, 0);
        chunks_count_ = std::exchange(other.chunks_count_, 0);
    }

  public:
    basic_growing_pool() noexcept = default;
    explicit basic_growing_pool(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}
    ~basic_growing_pool() noexcept { release_(); }

    basic_growing_pool(basic_growing_pool &&other) noexcept : allocator_(std::move(other.allocator_)) { steal_(other); }

    basic_growing_pool &operator=(basic_growing_pool &&other) noexcept {
        if (this == &other) return *this;
        release_();
        allocator_ = std::move(other.allocator_);
        steal_(other);
        return *this;
    }

    basic_growing_pool(basic_growing_pool const &) = delete;
    basic_growing_pool &operator=(basic_growing_pool const &) = delete;

    /**
     *  @brief A pool of at least @p capacity free records, rounded up to a power of two and
     *      clipped to @p capacity_limit, which a @c take finding none free grows chunk by chunk up
     *      to that limit.
     *  @return The pool, which allocates nothing for a zero limit.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k A limit past @c capacity_max_k, below @p capacity, or too large to
     *    count in bytes.
     *  - @c out_of_memory_heap_k The allocator refused the first chunk.
     */
    static expected<basic_growing_pool> make(std::size_t capacity, std::size_t capacity_limit,
                                             allocator_t allocator = {}) noexcept {
        if (capacity_limit > capacity_max_k || capacity > capacity_limit ||
            capacity_limit > std::numeric_limits<std::size_t>::max() / sizeof(value_t))
            return invalid_argument_k;
        basic_growing_pool pool(std::move(allocator));
        if (capacity_limit == 0) return expected<basic_growing_pool>(std::move(pool), success_k);
        pool.first_chunk_ = static_cast<index_t>(std::bit_ceil(capacity == 0 ? std::size_t {1} : capacity));
        pool.capacity_limit_ = static_cast<index_t>(capacity_limit);
        if (status_t const grown = pool.grow_(); failed(grown)) {
            pool.first_chunk_ = pool.capacity_limit_ = 0;
            return grown;
        }
        return expected<basic_growing_pool>(std::move(pool), success_k);
    }

    /** Records the pool holds now, taken or free, bookkeeping of whoever serializes @c take. */
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    /** Records the pool may grow to. */
    [[nodiscard]] std::size_t capacity_limit() const noexcept { return capacity_limit_; }

    [[nodiscard]] std::size_t size() const noexcept { return capacity_ - free_count_; }
    [[nodiscard]] bool empty() const noexcept { return free_count_ == capacity_; }

    /** Whether every record is taken and none may be added. */
    [[nodiscard]] bool full() const noexcept { return free_count_ == 0 && capacity_ == capacity_limit_; }

    /** Every record, taken or free, in index order. */
    [[nodiscard]] records_range<basic_growing_pool, value_t> records() noexcept {
        return records_range<basic_growing_pool, value_t>(this);
    }
    [[nodiscard]] records_range<basic_growing_pool const, value_t const> records() const noexcept {
        return records_range<basic_growing_pool const, value_t const>(this);
    }

    /** The record at @p index, taken or not, whose chunk must be published. */
    [[nodiscard]] value_t &operator[](index_t index) noexcept {
        value_t *const record = published_record_(index);
        assert(record && "index out of bounds");
        return *record;
    }
    [[nodiscard]] value_t const &operator[](index_t index) const noexcept {
        value_t const *const record = published_record_(index);
        assert(record && "index out of bounds");
        return *record;
    }

    /** The index of @p record, which must be one of @c records, found among published chunks. */
    [[nodiscard]] index_t index_of(value_t const &record) const noexcept {
        for (std::size_t chunk = 0; chunk != chunks_capacity_k; ++chunk) {
            value_t const *const records = chunk_(chunk);
            if (!records) break;
            if (&record >= records && &record < records + chunk_held_(chunk))
                return static_cast<index_t>(chunk_start_(chunk) + (&record - records));
        }
        assert(false && "a record of another pool");
        return absent_k;
    }

    /** Takes a free record, as its last holder left it, growing the pool when none is free and it
     *  may, or answers @c absent_k when it may not or the allocator refuses. */
    [[nodiscard]] index_t take() noexcept {
        if (!free_count_ && failed(grow_())) return absent_k;
        return free_indices_[--free_count_];
    }

    /** Frees the taken record at @p index, stepping its generation so older handles go stale. */
    void vacate(index_t index) noexcept {
        assert(index < capacity_ && free_count_ < capacity_ && "vacating a record that was never taken");
        if constexpr (generation_tagged<value_t, index_t>) ++(*this)[index].generation;
        free_indices_[free_count_++] = index;
    }

    /** A handle to the record at @p index, which @c at answers until it is vacated. */
    [[nodiscard]] handle_t handle_of(index_t index) const noexcept
        requires generation_tagged<value_t, index_t>
    {
        return {index, (*this)[index].generation};
    }

    /** The record @p handle names, or null once that record was vacated. */
    [[nodiscard]] value_t *at(handle_t handle) noexcept
        requires generation_tagged<value_t, index_t>
    {
        value_t *const record = published_record_(handle.index);
        return record && record->generation == handle.generation ? record : nullptr;
    }
    [[nodiscard]] value_t const *at(handle_t handle) const noexcept
        requires generation_tagged<value_t, index_t>
    {
        value_t const *const record = published_record_(handle.index);
        return record && record->generation == handle.generation ? record : nullptr;
    }
};

} // namespace ashvardanian::smashtable
