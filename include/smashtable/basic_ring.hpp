/**
 *  @file include/smashtable/basic_ring.hpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief A fixed-capacity first-in first-out ring over one allocation, for batching and
 *      read-ahead queues.
 *
 *  Capacity is a power of two chosen at @c make, and the ring never grows. Two 32-bit counters of
 *  pushes and pops wrap together, so their difference is the size and their low bits are the slots,
 *  and a full ring is told apart from an empty one without a spare slot.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <memory>      // `std::allocator_traits`
#include <new>         // placement `new`
#include <span>        // `std::span`
#include <type_traits> // `std::is_nothrow_move_constructible_v`
#include <utility>     // `std::exchange`, `std::move`

#include "shared.hpp"

namespace ashvardanian::smashtable {

/** Whether @c push_evicting had to drop the oldest element to make room. */
enum class ring_eviction_t : bool {
    kept_every_element_k,
    evicted_oldest_k,
};

/** A first-in first-out ring of at most @c capacity elements of @p element_type_, which a failed push reports. */
template <typename element_type_, typename allocator_type_ = default_allocator<element_type_>>
class basic_ring {
  public:
    using element_t = element_type_;
    using allocator_t = allocator_type_;

    /** The largest capacity, so a size always fits the 32-bit counters. */
    static constexpr std::size_t capacity_limit_k = std::size_t {1} << 31;

    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "basic_ring requires allocators that propagate on move assignment");
    static_assert(std::is_nothrow_move_constructible_v<element_t> && std::is_nothrow_move_assignable_v<element_t>,
                  "basic_ring moves elements in noexcept code, so they must move without throwing");

  private:
    element_t *slots_ {nullptr};
    std::uint32_t capacity_ {0};
    std::uint32_t pushed_ {0};
    std::uint32_t popped_ {0};
    ST_NO_UNIQUE_ADDRESS_ allocator_t allocator_ {};

    [[nodiscard]] element_t &slot_(std::uint32_t counter) noexcept { return slots_[counter & (capacity_ - 1)]; }
    [[nodiscard]] element_t const &slot_(std::uint32_t counter) const noexcept {
        return slots_[counter & (capacity_ - 1)];
    }

    void release_() noexcept {
        clear();
        if (slots_) allocator_.deallocate(slots_, capacity_);
        slots_ = nullptr;
        capacity_ = 0;
        pushed_ = 0;
        popped_ = 0;
    }

  public:
    basic_ring() noexcept = default;

    explicit basic_ring(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}

    ~basic_ring() noexcept { release_(); }

    basic_ring(basic_ring &&other) noexcept
        : slots_(std::exchange(other.slots_, nullptr)), capacity_(std::exchange(other.capacity_, 0)),
          pushed_(std::exchange(other.pushed_, 0)), popped_(std::exchange(other.popped_, 0)),
          allocator_(std::move(other.allocator_)) {}

    basic_ring &operator=(basic_ring &&other) noexcept {
        if (this == &other) return *this;
        release_();
        slots_ = std::exchange(other.slots_, nullptr);
        capacity_ = std::exchange(other.capacity_, 0);
        pushed_ = std::exchange(other.pushed_, 0);
        popped_ = std::exchange(other.popped_, 0);
        allocator_ = std::move(other.allocator_);
        return *this;
    }

    basic_ring(basic_ring const &) = delete;
    basic_ring &operator=(basic_ring const &) = delete;

    /**
     *  An empty ring of @p capacity slots, where zero allocates nothing and holds nothing.
     *  @return The ring, @c invalid_argument_k for a capacity that is not a power of two up to
     *      @c capacity_limit_k, or @c out_of_memory_heap_k.
     */
    [[nodiscard]] static expected<basic_ring> make(std::size_t capacity, allocator_t allocator = {}) noexcept {
        if (capacity > capacity_limit_k || (capacity & (capacity - 1)) != 0) return invalid_argument_k;
        basic_ring ring(std::move(allocator));
        if (capacity != 0) {
            ring.slots_ = ring.allocator_.allocate(capacity);
            if (!ring.slots_) return out_of_memory_heap_k;
            ring.capacity_ = static_cast<std::uint32_t>(capacity);
        }
        return expected<basic_ring>(std::move(ring), success_k);
    }

    [[nodiscard]] bool empty() const noexcept { return pushed_ == popped_; }
    [[nodiscard]] bool full() const noexcept { return size() == capacity_; }
    [[nodiscard]] std::size_t size() const noexcept { return static_cast<std::uint32_t>(pushed_ - popped_); }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::size_t free_space() const noexcept { return capacity_ - size(); }

    /** The element the next pop removes; the ring must not be empty. */
    [[nodiscard]] element_t &oldest() noexcept {
        assert(!empty() && "oldest requires a non-empty ring");
        return slot_(popped_);
    }
    [[nodiscard]] element_t const &oldest() const noexcept {
        assert(!empty() && "oldest requires a non-empty ring");
        return slot_(popped_);
    }

    /** The element the last push added; the ring must not be empty. */
    [[nodiscard]] element_t &newest() noexcept {
        assert(!empty() && "newest requires a non-empty ring");
        return slot_(pushed_ - 1);
    }
    [[nodiscard]] element_t const &newest() const noexcept {
        assert(!empty() && "newest requires a non-empty ring");
        return slot_(pushed_ - 1);
    }

    /** The element @p index places after the oldest, without bounds checking. */
    [[nodiscard]] element_t &operator[](std::size_t index) noexcept {
        assert(index < size() && "index out of bounds");
        return slot_(popped_ + static_cast<std::uint32_t>(index));
    }
    [[nodiscard]] element_t const &operator[](std::size_t index) const noexcept {
        assert(index < size() && "index out of bounds");
        return slot_(popped_ + static_cast<std::uint32_t>(index));
    }

    /** The element @p index places after the oldest, or null past the newest. */
    [[nodiscard]] element_t *at(std::size_t index) noexcept {
        return index < size() ? &slot_(popped_ + static_cast<std::uint32_t>(index)) : nullptr;
    }
    [[nodiscard]] element_t const *at(std::size_t index) const noexcept {
        return index < size() ? &slot_(popped_ + static_cast<std::uint32_t>(index)) : nullptr;
    }

    /** Appends @p element to a ring known not to be full. */
    [[nodiscard]] status_t push(assume_reserved_t, element_t &&element) noexcept {
        assert(!full() && "push with assume_reserved requires a free slot");
        new (&slot_(pushed_)) element_t(std::move(element));
        ++pushed_;
        return success_k;
    }

    /** Appends @p element, or answers @c capacity_exhausted_k and leaves the ring alone. */
    [[nodiscard]] status_t push(element_t &&element) noexcept {
        if (full()) return capacity_exhausted_k;
        return push(assume_reserved, std::move(element));
    }

    /** Appends @p element, first moving the oldest into @p evicted when the ring is full. Capacity must be nonzero. */
    [[nodiscard]] ring_eviction_t push_evicting(element_t &&element, element_t &evicted) noexcept {
        assert(capacity_ != 0 && "a ring without slots cannot hold the newest element");
        ring_eviction_t eviction = ring_eviction_t::kept_every_element_k;
        if (full()) {
            evicted = std::move(slot_(popped_));
            pop();
            eviction = ring_eviction_t::evicted_oldest_k;
        }
        [[maybe_unused]] status_t const status = push(assume_reserved, std::move(element));
        return eviction;
    }

    /** Destroys the oldest element; the ring must not be empty. */
    void pop() noexcept {
        assert(!empty() && "pop requires a non-empty ring");
        slot_(popped_).~element_t();
        ++popped_;
    }

    /** Moves the oldest element into @p destination and removes it, or answers @c operation_would_block_k. */
    [[nodiscard]] status_t pop_into(element_t &destination) noexcept {
        if (empty()) return operation_would_block_k;
        destination = std::move(slot_(popped_));
        pop();
        return success_k;
    }

    /** Copies as many of @p elements as fit, oldest first, and returns how many. */
    std::size_t push_many(std::span<element_t const> elements) noexcept
        requires std::is_nothrow_copy_constructible_v<element_t>
    {
        std::size_t const accepted = smaller_of(elements.size(), free_space());
        for (std::size_t index = 0; index < accepted; ++index) {
            new (&slot_(pushed_)) element_t(elements[index]);
            ++pushed_;
        }
        return accepted;
    }

    /** Moves as many elements as @p destination holds out of the ring, oldest first, and returns how many. */
    std::size_t pop_many(std::span<element_t> destination) noexcept {
        std::size_t const delivered = smaller_of(destination.size(), size());
        for (std::size_t index = 0; index < delivered; ++index) {
            destination[index] = std::move(slot_(popped_));
            pop();
        }
        return delivered;
    }

    /** Destroys every element and keeps the slots. */
    void clear() noexcept {
        while (!empty()) pop();
    }
};

} // namespace ashvardanian::smashtable
