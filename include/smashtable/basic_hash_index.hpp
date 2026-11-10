/**
 *  @file include/smashtable/basic_hash_index.hpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief A fixed-capacity index from fixed-width keys to the positions of records kept elsewhere.
 *
 *  @section basic_hash_index_layout Layout
 *
 *  The same single allocation a @c hash_map is carved from: keys in the keys region, so a probe
 *  compares a whole window of them through the row kit without touching a record, positions in the
 *  values region, and the @c populations lane of each header as occupancy. The @c deletions lane
 *  stays zero, since an erase shifts the rest of its run back over the hole rather than leaving a
 *  tombstone, so a table under churn probes as far as one built with the same keys.
 *
 *  @section basic_hash_index_duplicates Duplicate Keys
 *
 *  A key may be a hash of something longer, and two records may share it. Inserting is therefore
 *  always @c assume_unique, a lookup can take a @c confirm that reads the record behind each match,
 *  and an erase names the exact position it removes.
 */
#pragma once
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <concepts>    // `std::unsigned_integral`
#include <limits>      // `std::numeric_limits`
#include <type_traits> // `std::is_empty_v`
#include <utility>     // `std::move`, `std::swap`

#include "smashtable/hash_layout.hpp"
#include "smashtable/row_search.hpp" // `row_kit`, `native_row_kit_t`

namespace ashvardanian::smashtable {

/**
 *  @brief Maps fixed-width keys to record positions, at most @c capacity of them, with linear
 *      probing in windows and backward-shift erasure.
 *
 *  @tparam key_type_ A 4- or 8-byte integer or a @c key128_t, compared a window at a time.
 *  @tparam index_type_ The record position stored per key, whose largest value is @c absent_k.
 *  @tparam hasher_type_ Hashes a key down to its home slot.
 *  @tparam allocator_type_ Supplies and reclaims the single byte buffer.
 *  @tparam row_kit_type_ The kit comparing a window of its @c mask_equal_width_k keys at once.
 */
template <window_key key_type_, std::unsigned_integral index_type_ = std::uint32_t,
          typename hasher_type_ = default_hash_t, allocator_of<std::byte> allocator_type_ = default_allocator_t,
          row_kit row_kit_type_ = native_row_kit_t>
    requires key_hasher<hasher_type_, key_type_>
class basic_hash_index {
  public:
    using key_t = key_type_;
    using index_t = index_type_;
    using hasher_t = hasher_type_;
    using allocator_t = allocator_type_;

    /** The position @c find answers when no key matches, which no record may occupy. */
    static constexpr index_t absent_k = (std::numeric_limits<index_t>::max)();

  private:
    using storage_t = hash_storage<mapping<key_t, index_t>, hasher_t, allocator_t>;
    using offset_t = typename storage_t::offset_t;

    /** The single allocation, its keys, positions and headers, and the population count. */
    storage_t storage_ {};

    /** How many keys @c insert accepts, at most half the slots so every run ends at a free one. */
    offset_t capacity_ {};

    /** Hashes a key down to its home slot. */
    SMASHTABLE_NO_UNIQUE_ADDRESS_ hasher_t hasher_ {};

    basic_hash_index(storage_t storage, offset_t capacity, hasher_t hasher) noexcept
        : storage_(std::move(storage)), capacity_(capacity), hasher_(std::move(hasher)) {}

    [[nodiscard]] offset_t home_of_(key_t const &key) const noexcept {
        return static_cast<offset_t>(hasher_(key)) & (storage_.slots_count - 1);
    }

    /** The first free slot at or after @p slot, wrapping; a table at most half full has one. */
    [[nodiscard]] offset_t free_slot_from_(offset_t slot) const noexcept {
        offset_t const offset_mask = storage_.slots_count - 1;
        for (;;) {
            unsigned const shift = static_cast<unsigned>(slot % hash_bucket_capacity_k);
            hash_bucket_mask_t const free = ~storage_.headers[slot / hash_bucket_capacity_k].lanes.populations >> shift;
            if (free) return (slot + static_cast<offset_t>(countr_zero(free))) & offset_mask;
            slot = (slot - shift + hash_bucket_capacity_k) & offset_mask;
        }
    }

    /** Hands @p on_match the slot of every key equal to @p key, in probe order, until it halts. */
    template <typename on_match_type_>
    void walk_(key_t const &key, on_match_type_ &&on_match) const noexcept {
        if (!storage_.slots_count) return;
        walk_hash_windows<row_kit_type_>(storage_.headers, storage_.keys, storage_.slots_count, home_of_(key), key,
                                         on_match);
    }

  public:
    basic_hash_index() noexcept = default;
    basic_hash_index(basic_hash_index &&other) noexcept { swap(other); }
    basic_hash_index &operator=(basic_hash_index &&other) noexcept {
        swap(other);
        return *this;
    }
    basic_hash_index(basic_hash_index const &) = delete;
    basic_hash_index &operator=(basic_hash_index const &) = delete;

    /**
     *  @brief An empty index accepting up to @p capacity keys, over twice as many slots.
     *  @return The index; a zero capacity allocates nothing and accepts nothing.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k A capacity whose slots no power of two covers.
     *  - @c out_of_memory_heap_k The allocator refused the buffer.
     */
    static expected<basic_hash_index> make(std::size_t capacity, hasher_t hasher = {},
                                           allocator_t allocator = {}) noexcept {
        if (capacity > (std::numeric_limits<std::size_t>::max)() / 2) return invalid_argument_k;
        hash_slots_count_t const slots = hash_slots_count_t::from_slots(capacity * 2);
        if (!slots.is_addressable()) return invalid_argument_k;
        storage_t storage = storage_t::make(slots, std::move(allocator));
        if (slots.raw && !storage.is_allocated()) return out_of_memory_heap_k;
        return expected<basic_hash_index>(
            basic_hash_index(std::move(storage), static_cast<offset_t>(capacity), std::move(hasher)), success_k);
    }

    [[nodiscard]] std::size_t size() const noexcept { return storage_.populated_count; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool empty() const noexcept { return storage_.populated_count == 0; }

    /** The position stored under @p key, the first in probe order when several are, or
     *  @c absent_k. */
    [[nodiscard]] index_t find(key_t key) const noexcept {
        index_t found = absent_k;
        walk_(key, [&](offset_t slot) noexcept {
            found = storage_.values[slot];
            return walk_control_t::halt_k;
        });
        return found;
    }

    /** The first position under @p key that @p confirm accepts, for keys that hash something
     *  longer. */
    template <typename confirm_type_>
    [[nodiscard]] index_t find(key_t key, confirm_type_ &&confirm) const noexcept {
        index_t found = absent_k;
        walk_(key, [&](offset_t slot) noexcept {
            index_t const candidate = storage_.values[slot];
            if (!confirm(candidate)) return walk_control_t::resume_k;
            found = candidate;
            return walk_control_t::halt_k;
        });
        return found;
    }

    /**
     *  @brief Stores @p index under @p key, beside any position already stored under it.
     *
     *  @par Statuses:
     *  - @c capacity_exhausted_k The index already holds @c capacity keys.
     */
    status_t insert(key_t key, index_t index, assume_unique_t) noexcept {
        if (storage_.populated_count == capacity_) return capacity_exhausted_k;
        offset_t const slot = free_slot_from_(home_of_(key));
        storage_.keys[slot] = key;
        storage_.values[slot] = index;
        storage_.headers[slot / hash_bucket_capacity_k].lanes.populations |= hash_bucket_mask_t {1}
                                                                             << (slot % hash_bucket_capacity_k);
        ++storage_.populated_count;
        return success_k;
    }

    /**
     *  @brief Removes the entry pairing @p key with @p index, shifting the rest of its run back.
     *
     *  Each later key in the run moves into the hole when the hole lies between its home and its
     *  slot, so every key stays reachable from its home without a tombstone.
     *
     *  @par Statuses:
     *  - @c key_not_found_k No entry pairs @p key with @p index.
     */
    status_t erase(key_t key, index_t index) noexcept {
        offset_t hole = storage_.slots_count;
        walk_(key, [&](offset_t slot) noexcept {
            if (storage_.values[slot] != index) return walk_control_t::resume_k;
            hole = slot;
            return walk_control_t::halt_k;
        });
        if (hole == storage_.slots_count) return key_not_found_k;

        offset_t const offset_mask = storage_.slots_count - 1;
        offset_t const run_end = free_slot_from_((hole + 1) & offset_mask);
        for (offset_t slot = (hole + 1) & offset_mask; slot != run_end; slot = (slot + 1) & offset_mask) {
            offset_t const home = home_of_(storage_.keys[slot]);
            if (((slot - home) & offset_mask) < ((slot - hole) & offset_mask)) continue;
            storage_.keys[hole] = storage_.keys[slot];
            storage_.values[hole] = storage_.values[slot];
            hole = slot;
        }
        storage_.headers[hole / hash_bucket_capacity_k].lanes.populations &=
            ~(hash_bucket_mask_t {1} << (hole % hash_bucket_capacity_k));
        --storage_.populated_count;
        return success_k;
    }

    /** Forgets every entry and keeps the allocation, zeroing the headers alone. */
    void clear() noexcept { storage_.clear(); }

    void swap(basic_hash_index &other) noexcept {
        storage_.swap(other.storage_);
        std::swap(capacity_, other.capacity_);
        if constexpr (!std::is_empty_v<hasher_t>) std::swap(hasher_, other.hasher_);
    }
};

} // namespace ashvardanian::smashtable
