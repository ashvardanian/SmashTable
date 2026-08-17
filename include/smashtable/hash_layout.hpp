/**
 *  @brief Shared vocabulary of the open-addressed hash tables: bucket metadata, slot references, storage.
 *  @author Ash Vardanian
 *  @file include/smashtable/hash_layout.hpp
 *  @date August 17, 2026
 *
 *  @section hash_layout_overview Overview
 *
 *  Two tables are carved from the same memory layout - a growable single-threaded one and a pinned
 *  lock-free one - and this header is everything they have in common. Neither table names the other,
 *  and the only thing that travels between them is a @c hash_storage, moved as a value.
 *
 *  @section hash_layout_memory Memory Layout
 *
 *  One allocation, three regions at cache-line offsets, so compaction can keep the keys and values
 *  and drop the headers without copying:
 *  @code
 *  // Layout: [keys_region | values_region | headers_region]
 *  key_t *keys = (key_t *)memory;                          // Direct indexing: keys[slot]
 *  value_t *values = (value_t *)(memory + keys_bytes);     // Direct indexing: values[slot]
 *  hash_bucket_head_t *headers = ...;                      // Bucket indexing: headers[slot / 32]
 *  @endcode
 *
 *  The offsets are multiples of the cache line; the base is only as aligned as the allocator makes
 *  it, which for @c std::allocator is the default new alignment, so a region boundary shares a line
 *  with its neighbour rather than starting one. An over-aligned key or value is not accommodated.
 *
 *  @section hash_layout_bucket Bucket Size
 *
 *  Each slot costs 2 bits of metadata, and 32 slots share one 64-bit header. That is 4× denser than
 *  SwissTable's byte per slot, it keeps every metadata update inside a single 64-bit atomic, and it
 *  matches an Nvidia warp for cooperative-group probing.
 *
 *  @see Google's SwissTable: https://abseil.io/docs/cpp/guides/container
 *  @see Nvidia's GPU-friendly cuCollections: https://github.com/NVIDIA/cuCollections
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint32_t`, `std::uint64_t`
#include <cstring> // `std::memcpy`, `std::memset`

#include <algorithm>   // `std::max`
#include <atomic>      // `std::atomic_ref`
#include <bit>         // `std::popcount`, `std::countr_zero`
#include <functional>  // `std::hash`
#include <memory>      // `std::allocator`
#include <type_traits> // `std::is_same`, `std::conditional_t`
#include <utility>     // `std::declval`, `std::move`

#include "shared.hpp"

namespace ashvardanian::smashtable {

#pragma region Bucket Metadata

/** @brief Platform cache line size in bytes, typically 64 on modern CPUs. */
inline constexpr std::size_t cache_line_bytes_k = 64;

/** @brief Number of slots sharing one bucket header, sized so the header fits a 64-bit atomic. */
inline constexpr std::size_t hash_bucket_capacity_k = 32;

/**
 *  @brief Bucket header containing metadata for @c hash_bucket_capacity_k slots.
 *    Stores two parallel 32-bit bitmasks (population and deletion states) that combine to encode
 *    four possible states per slot using 2 bits each.
 *
 *  @section hash_layout_slot_state_encoding Slot State Encoding
 *  Each slot's state is determined by corresponding bits in both masks:
 *  - @c 00 (populations=0, deletions=0): Free slot, never used or fully freed
 *  - @c 01 (populations=0, deletions=1): Deleted slot, tombstone from lazy deletion
 *  - @c 10 (populations=1, deletions=0): Populated slot with valid key-value pair
 *  - @c 11 (populations=1, deletions=1): Locked slot, temporarily held for atomic operations
 *
 *  @c u32s exposes the two lanes separately - @c populations has a set bit for every populated or
 *  locked slot, @c deletions one for every deleted or locked slot. @c u64 views both lanes at once,
 *  which is what enables atomic operations on the entire header (2 × 32 bits = 64 bits).
 */
union hash_bucket_head_t {

    struct {
        std::uint32_t populations;
        std::uint32_t deletions;
    } u32s;
    std::uint64_t u64;
};

static_assert(sizeof(hash_bucket_head_t) == sizeof(std::uint64_t));

/** @brief Bitmask selecting one slot within a bucket header lane. */
using hash_bucket_mask_t = std::uint32_t;

/** @brief The four states a slot can be in, spelled out by the two header lanes. */
enum class hash_slot_state_t : std::uint32_t {
    /** @brief Never used, or fully freed. */
    free_k = 0,
    /** @brief Tombstone left behind by a lazy deletion. */
    deleted_k = 1,
    /** @brief Holds a live key, and a value when the table is a map. */
    populated_k = 2,
    /** @brief Temporarily held by a thread performing an atomic operation. */
    locked_k = 3,
};

/** @brief Reads the state of the slot selected by @p mask out of both header lanes. */
inline hash_slot_state_t hash_slot_state_of(hash_bucket_head_t const &head, hash_bucket_mask_t mask) noexcept {
    std::uint32_t const populated = (head.u32s.populations & mask) != 0;
    std::uint32_t const deleted = (head.u32s.deletions & mask) != 0;
    return static_cast<hash_slot_state_t>(populated * 2u + deleted);
}

/** @brief Drives the slot selected by @p mask into the @c free_k state. */
inline void hash_mark_free(hash_bucket_head_t &head, hash_bucket_mask_t mask) noexcept {
    head.u32s.populations &= ~mask;
    head.u32s.deletions &= ~mask;
}

/** @brief Drives the slot selected by @p mask into the @c populated_k state. */
inline void hash_mark_populated(hash_bucket_head_t &head, hash_bucket_mask_t mask) noexcept {
    head.u32s.populations |= mask;
    head.u32s.deletions &= ~mask;
}

/** @brief Drives the slot selected by @p mask into the @c deleted_k state. */
inline void hash_mark_deleted(hash_bucket_head_t &head, hash_bucket_mask_t mask) noexcept {
    head.u32s.populations &= ~mask;
    head.u32s.deletions |= mask;
}

/**
 *  @brief Scaling schema that computes the required number of slots for a target element count.
 *    Enforces a 75% maximum load factor (4/3 multiplier) to prevent pathological linear probing behavior.
 *    All slot counts are rounded up to powers of two to enable fast modulo operations via bitwise AND masks.
 *
 *  @note The 75% load factor strikes a balance between memory efficiency and probe length:
 *    - Too high (>85%): Linear probing degrades into long search chains
 *    - Too low (<60%): Wastes memory without significant performance gain
 *    - 75%: Industry standard, keeps average probe length under 2 hops
 *
 *  @note The minimum slot count equals @c hash_bucket_capacity_k to ensure at least one full bucket.
 *
 *  @c raw carries the widened count itself. The primary constructor budgets for a target @b element
 *  count and applies the load factor, while @c from_slots takes the slot count directly, for callers
 *  that already sized the storage. @c is_addressable reports whether the probe masks can address the
 *  count, which requires a power of two.
 */
struct hash_slots_count_t {
    std::size_t raw = 0;

    operator std::size_t() const noexcept { return raw; }
    explicit constexpr hash_slots_count_t() noexcept {}
    explicit constexpr hash_slots_count_t(std::size_t elements) noexcept {
        if (elements == 0) return;
        std::size_t needed_slots = (elements * 4ul) / 3ul;
        // We calculate the bucket index with AND masks, so it must be a power of two.
        raw = roundup_to_pow2(needed_slots);
        raw = std::max(raw, hash_bucket_capacity_k);
    }

    static constexpr hash_slots_count_t from_slots(std::size_t slots) noexcept {
        hash_slots_count_t result;
        if (slots == 0) return result;
        result.raw = std::max(roundup_to_pow2(slots), hash_bucket_capacity_k);
        return result;
    }

    constexpr bool is_addressable() const noexcept {
        return raw == 0 || (raw >= hash_bucket_capacity_k && (raw & (raw - 1)) == 0);
    }
};

#pragma endregion Bucket Metadata

#pragma region Layouts

/**
 *  @brief Metadata extraction template that derives type information for hash table elements.
 *    Computes element types, sizes, and alignment requirements from element/hasher template parameters.
 *    Uses the hasher's return type as @c offset_t for slot indexing (typically @c std::size_t).
 *
 *  @tparam element_type_ Key type for sets, or @c mapping<K,V> for maps. May be const-qualified.
 *  @tparam hasher_type_ Hash function object, must be callable with the key type.
 *
 *  @note Provides three element views:
 *    - @c element_t: Mutable reference pair @code mapping<key const&, value&> @endcode for iteration
 *    - @c element_const_t: Immutable reference pair @code mapping<key const&, value const&> @endcode for lookups
 *    - @c element_copy_t: Owned value pair @code mapping<key, value> @endcode for extraction operations
 *
 *  @see https://en.cppreference.com/w/cpp/utility/hash
 */
template <typename element_type_, typename hasher_type_>
struct hash_layout_for {
    using key_t = std::remove_reference_t<element_type_>;
    using value_t = void;
    using value_storage_t = placeholder_t;
    using element_t = key_t;
    using element_const_t = key_t;
    using element_copy_t = key_t;
    using hasher_t = hasher_type_;
    using offset_t = decltype(hasher_type_ {}(std::declval<key_t>()));

    inline static constexpr bool has_values_k = false;
    inline static constexpr std::size_t bytes_for_keys_k =
        roundup_to_multiple<std::size_t, cache_line_bytes_k>(sizeof(key_t) * hash_bucket_capacity_k);
    inline static constexpr std::size_t bytes_for_values_k = 0;
    inline static constexpr std::size_t bytes_in_bucket_k =
        bytes_for_keys_k + bytes_for_values_k + sizeof(hash_bucket_head_t);

    static constexpr bool will_memcpy_keys() noexcept { return std::is_trivially_copy_constructible<key_t>(); }
    static constexpr bool will_memcpy_vals() noexcept { return false; }
};

template <typename key_type_, typename value_type_, typename hasher_type_>
struct hash_layout_for<mapping<key_type_, value_type_>, hasher_type_> {

    using key_t = std::remove_reference_t<key_type_>;
    using value_t = std::remove_reference_t<value_type_>;
    using value_storage_t = value_t;
    using element_t = mapping<key_t const &, value_t &>;
    using element_const_t = mapping<key_t const &, value_t const &>;
    using element_copy_t = mapping<key_t, value_t>;
    using hasher_t = hasher_type_;
    using offset_t = decltype(hasher_type_ {}(std::declval<key_t>()));

    inline static constexpr bool has_values_k = true;
    inline static constexpr std::size_t bytes_for_keys_k =
        roundup_to_multiple<std::size_t, cache_line_bytes_k>(sizeof(key_t) * hash_bucket_capacity_k);
    inline static constexpr std::size_t bytes_for_values_k =
        roundup_to_multiple<std::size_t, cache_line_bytes_k>(sizeof(value_t) * hash_bucket_capacity_k);
    inline static constexpr std::size_t bytes_in_bucket_k =
        bytes_for_keys_k + bytes_for_values_k + sizeof(hash_bucket_head_t);

    static constexpr bool will_memcpy_keys() noexcept { return std::is_trivially_copy_constructible<key_t>(); }
    static constexpr bool will_memcpy_vals() noexcept { return std::is_trivially_copy_constructible<value_t>(); }
};

/** @brief Const-qualified elements reuse the unqualified layout, adding @c const to every view. */
template <typename element_type_, typename hasher_type_>
struct hash_layout_for<element_type_ const, hasher_type_> {

    using unqualified_t = hash_layout_for<element_type_, hasher_type_>;
    using key_t = typename unqualified_t::key_t const;
    using value_t = typename unqualified_t::value_t const;
    using value_storage_t = typename unqualified_t::value_storage_t const;
    using element_t = typename unqualified_t::element_const_t;
    using element_const_t = typename unqualified_t::element_const_t;
    using element_copy_t = typename unqualified_t::element_copy_t;
    using hasher_t = hasher_type_;
    using offset_t = typename unqualified_t::offset_t;

    inline static constexpr bool has_values_k = unqualified_t::has_values_k;
    inline static constexpr std::size_t bytes_for_keys_k = unqualified_t::bytes_for_keys_k;
    inline static constexpr std::size_t bytes_for_values_k = unqualified_t::bytes_for_values_k;
    inline static constexpr std::size_t bytes_in_bucket_k = unqualified_t::bytes_in_bucket_k;

    static constexpr bool will_memcpy_keys() noexcept { return unqualified_t::will_memcpy_keys(); }
    static constexpr bool will_memcpy_vals() noexcept { return unqualified_t::will_memcpy_vals(); }
};

static_assert(hash_layout_for<int, std::hash<int>>::will_memcpy_keys());
static_assert(hash_layout_for<mapping<int, int>, std::hash<int>>::will_memcpy_keys());

/** @brief Wraps @c std::hash default hasher that delays type resolution until invocation. */
struct lazy_std_hash_t {

    template <typename key_type_>
    std::size_t operator()(key_type_ const &key) const noexcept {
        return std::hash<key_type_> {}(key);
    }
};

#pragma endregion Layouts

#pragma region Slot References

/**
 *  @brief Smart reference to a hash table slot, addressing it as base pointers plus a slot index.
 *  @tparam element_type_ Key type for sets, or @c mapping<K,V> for maps. May be const-qualified.
 *  @tparam hasher_type_ Hash function object, only used to derive the offset type.
 *
 *  Probing advances @c slot_ by one, so a step costs a single integer increment rather than three
 *  pointer recomputations. Sets carry no values region and leave @c values_ null.
 *
 *  @c keys_, @c values_ and @c headers_ are the bases of the three regions, the first two indexed by
 *  the absolute @c slot_ and the last by the bucket that owns it. The aliases restate the layout's
 *  own key, value, element and offset types, plus @c dereference_t, which is the element view for
 *  maps and a key reference for sets.
 *
 *  @c header_ref and @c mask_in_bucket resolve the owning header and the single bit inside it;
 *  @c key, @c key_ref, @c value and @c value_ref reach the two payload regions; the @c is_ and
 *  @c mark_ families read and write the slot's two bits. Locking is what a threaded reference
 *  overrides, so here @c lock, @c unlock and @c try_lock collapse to nothing.
 */
template <typename element_type_, typename hasher_type_>
struct hash_slot_ref {

    using layout_t = hash_layout_for<element_type_, hasher_type_>;
    using key_t = typename layout_t::key_t;
    using value_t = typename layout_t::value_t;
    using value_storage_t = typename layout_t::value_storage_t;
    using element_t = typename layout_t::element_t;
    using offset_t = typename layout_t::offset_t;

    inline static constexpr bool has_values_k = layout_t::has_values_k;
    using dereference_t = std::conditional_t<has_values_k, element_t, key_t const &>;

    key_t *keys_ {};
    value_storage_t *values_ {};
    hash_bucket_head_t *headers_ {};
    offset_t slot_ {};

    hash_bucket_head_t &header_ref() const noexcept { return headers_[slot_ / hash_bucket_capacity_k]; }

    hash_bucket_mask_t mask_in_bucket() const noexcept {
        return hash_bucket_mask_t {1} << (slot_ % hash_bucket_capacity_k);
    }

    key_t &key_ref() const noexcept { return keys_[slot_]; }
    key_t const &key() const noexcept { return keys_[slot_]; }
    value_storage_t &value_ref() const noexcept { return values_[slot_]; }
    value_storage_t const &value() const noexcept { return values_[slot_]; }

    bool is_free() const noexcept {
        return hash_slot_state_of(header_ref(), mask_in_bucket()) == hash_slot_state_t::free_k;
    }
    bool is_deleted() const noexcept {
        return hash_slot_state_of(header_ref(), mask_in_bucket()) == hash_slot_state_t::deleted_k;
    }
    bool is_populated() const noexcept {
        return hash_slot_state_of(header_ref(), mask_in_bucket()) == hash_slot_state_t::populated_k;
    }
    bool is_locked() const noexcept {
        return hash_slot_state_of(header_ref(), mask_in_bucket()) == hash_slot_state_t::locked_k;
    }

    void mark_free() const noexcept { hash_mark_free(header_ref(), mask_in_bucket()); }
    void mark_populated() const noexcept { hash_mark_populated(header_ref(), mask_in_bucket()); }
    void mark_deleted() const noexcept { hash_mark_deleted(header_ref(), mask_in_bucket()); }

    constexpr void lock() const noexcept {}
    constexpr void unlock() const noexcept {}

    dereference_t operator*() const noexcept {
        if constexpr (has_values_k) return element_t {key(), value_ref()};
        else return key();
    }
};

/**
 *  @brief Thread-safe smart reference to a hash table slot with metadata accessors.
 *  @tparam element_type_ Key type for sets, or @c mapping<K,V> for maps. May be const-qualified.
 *  @tparam hasher_type_ Hash function object, only used to derive the offset type.
 *
 *  Ideally, we would want to avoid Compare-And-Swap @b (CAS) loops for locking individual slots.
 *  On the locking path, we can use a @c fetch_or atomic operation to set both bits (populated + deleted)
 *  simultaneously and transition to the locked state. If the previous value indicates the slot was
 *  already locked, we retry until we acquire the lock.
 *
 *  The intuition of using the @c fetch_and for the inverse operation, however, is wrong. When unlocking,
 *  we need to restore the previous state (populated/deleted) of the slot or transition to the new one,
 *  depending on the @c mark_populated()/mark_deleted()/mark_free() calls made while the slot was locked.
 *
 *  The bits outside the active ones in each 32-bit word shouldn't be changed. The active ones may have
 *  to be flipped. Assuming the value of the relevant bits couldn't have changed, we can use @c fetch_xor
 *  to control the result in the same lock-free manner. @b XOR-is-all-you-need!
 *
 *  This doesn't resolve @b false-sharing issues native to such a densely packed design, but still results
 *  in very low contention if the duration of atomic operations under the lock is comparable to CPU's
 *  memory latency.
 */
template <typename element_type_, typename hasher_type_>
class hash_atomic_slot_ref : public hash_slot_ref<element_type_, hasher_type_> {

    using base_t = hash_slot_ref<element_type_, hasher_type_>;

    /** @brief State the slot will be driven into once @c unlock() lands. */
    hash_bucket_head_t mutable future_header_ {};

    /** @brief Both lanes of this slot's bit, the exact footprint the lock owns. */
    hash_bucket_head_t header_mask_() const noexcept {
        hash_bucket_head_t mask {};
        mask.u32s.populations = base_t::mask_in_bucket();
        mask.u32s.deletions = base_t::mask_in_bucket();
        return mask;
    }

  public:
    /** @brief The staged header, not the shared one, so reads under the lock stay private. */
    hash_bucket_head_t &header_ref() const noexcept { return future_header_; }

    /** @brief Whether the staged state is @c free_k. */
    bool is_free() const noexcept {
        return hash_slot_state_of(header_ref(), base_t::mask_in_bucket()) == hash_slot_state_t::free_k;
    }
    /** @brief Whether the staged state is @c deleted_k. */
    bool is_deleted() const noexcept {
        return hash_slot_state_of(header_ref(), base_t::mask_in_bucket()) == hash_slot_state_t::deleted_k;
    }
    /** @brief Whether the staged state is @c populated_k. */
    bool is_populated() const noexcept {
        return hash_slot_state_of(header_ref(), base_t::mask_in_bucket()) == hash_slot_state_t::populated_k;
    }
    /** @brief Whether the staged state is @c locked_k. */
    bool is_locked() const noexcept {
        return hash_slot_state_of(header_ref(), base_t::mask_in_bucket()) == hash_slot_state_t::locked_k;
    }

    /** @brief Stages @c free_k, which @c unlock() then publishes. */
    void mark_free() const noexcept { hash_mark_free(header_ref(), base_t::mask_in_bucket()); }
    /** @brief Stages @c populated_k, which @c unlock() then publishes. */
    void mark_populated() const noexcept { hash_mark_populated(header_ref(), base_t::mask_in_bucket()); }
    /** @brief Stages @c deleted_k, which @c unlock() then publishes. */
    void mark_deleted() const noexcept { hash_mark_deleted(header_ref(), base_t::mask_in_bucket()); }

    /** @brief Spins on @c fetch_or until this thread is the one that observed a non-locked slot. */
    void lock() const noexcept {
        hash_bucket_head_t const header_mask = header_mask_();
        std::atomic_ref<std::uint64_t> atomic_header(base_t::header_ref().u64);
        while (true) {
            future_header_.u64 = atomic_header.fetch_or(header_mask.u64, std::memory_order_acquire) & header_mask.u64;
            if (future_header_.u64 != header_mask.u64) break;
        }
    }

    /** @brief Drives the two owned bits from @c locked_k to whatever was staged, with one @c fetch_xor. */
    void unlock() const noexcept {
        // The bits we care about are now set to 11 (locked).
        // After this procedure they must be set to either 00, 01, or 10, depending on the "future header".
        hash_bucket_head_t const header_mask = header_mask_();
        assert(std::popcount(header_mask.u64) == 2 && "only 2 bits must be set in the mask.");

        // Let's compute the "differences" between the old locked state and the new desired state:
        // - for "freed" slots 00: 11 ^ 00 = 11
        // - for "deleted" slots 01: 11 ^ 01 = 10
        // - for "populated" slots 10: 11 ^ 10 = 01
        std::uint64_t const header_differences = header_mask.u64 ^ future_header_.u64;
        assert(std::popcount(header_differences) >= 1 && std::popcount(header_differences) <= 2 &&
               "only 1 or 2 bits can form the difference.");

        // Now if we only XOR the differences:
        // - going from "locked" state to "freed" state: 11 ^ 11 = 00
        // - going from "locked" state to "deleted" state: 11 ^ 10 = 01
        // - going from "locked" state to "populated" state: 11 ^ 01 = 10
        std::atomic_ref<std::uint64_t> atomic_header(base_t::header_ref().u64);
        atomic_header.fetch_xor(header_differences, std::memory_order_release);
    }
};

#pragma endregion Slot References

#pragma region Bucket Algorithms

/**
 *  @brief Iterates over all populated slots in a bucket using optimized bit-scanning.
 *    More efficient than sequential iteration as it skips empty/deleted slots by analyzing
 *    the population bitmap with @c std::countr_zero.
 *
 *  @param[in,out] slot Reference to a slot in the bucket. Its @c slot_ field is modified
 *    during iteration to point to each populated slot sequentially.
 *  @param[in] callback Functor invoked for each populated slot, receiving @c hash_slot_ref.
 */
template <typename element_type_, typename hasher_type_, typename callback_type_>
void for_each_in_hash_bucket(hash_slot_ref<element_type_, hasher_type_> &slot, callback_type_ &&callback) noexcept {

    using offset_t = typename hash_slot_ref<element_type_, hasher_type_>::offset_t;
    offset_t const bucket_start = (slot.slot_ / hash_bucket_capacity_k) * hash_bucket_capacity_k;

    hash_bucket_head_t const &head = slot.header_ref();
    hash_bucket_mask_t populations_left = head.u32s.populations & ~head.u32s.deletions;
    while (populations_left) {
        offset_t const index_in_bucket = static_cast<offset_t>(std::countr_zero(populations_left));
        slot.slot_ = bucket_start + index_in_bucket;
        callback(slot);
        populations_left &= populations_left - 1;
    }
}

/**
 *  @brief Searches for an element within a bucket using optimized bit-scanning.
 *    Stops iteration early when the predicate returns @c true.
 *
 *  @param[in,out] slot Reference to a slot in the bucket. Its @c slot_ field is modified
 *    during iteration to point to each populated slot until a match is found.
 *  @param[in] predicate Functor invoked for each populated slot. Must return @c true if the
 *    element matches (terminating the search) or @c false to continue.
 *  @return True if a matching element was found, false otherwise.
 */
template <typename element_type_, typename hasher_type_, typename predicate_type_>
bool find_in_hash_bucket(hash_slot_ref<element_type_, hasher_type_> &slot, predicate_type_ &&predicate) noexcept {

    using offset_t = typename hash_slot_ref<element_type_, hasher_type_>::offset_t;
    offset_t const bucket_start = (slot.slot_ / hash_bucket_capacity_k) * hash_bucket_capacity_k;

    hash_bucket_head_t const &head = slot.header_ref();
    hash_bucket_mask_t populations_left = head.u32s.populations & ~head.u32s.deletions;
    while (populations_left) {
        offset_t const index_in_bucket = static_cast<offset_t>(std::countr_zero(populations_left));
        slot.slot_ = bucket_start + index_in_bucket;
        if (predicate(slot)) return true;
        populations_left &= populations_left - 1;
    }
    return false;
}

#pragma endregion Bucket Algorithms

#pragma region Storage

/**
 *  @brief The one allocation a table is carved from, and the counters describing it.
 *    Owns the buffer and the elements inside it, and frees both, so a table holding one needs no
 *    destructor of its own. This is also the only thing that passes between a growable table and a
 *    pinned one, which is what lets that hand-off be a move of a value rather than one type reaching
 *    into the other.
 *
 *  @tparam element_type_ Key type for sets, or @c mapping<K,V> for maps.
 *  @tparam hasher_type_ Hash function object, only used to derive the offset type.
 *  @tparam allocator_type_ Supplies and reclaims the single byte buffer.
 */
template <typename element_type_, typename hasher_type_, typename allocator_type_>
struct hash_storage {

    using layout_t = hash_layout_for<element_type_, hasher_type_>;
    using key_t = typename layout_t::key_t;
    using value_storage_t = typename layout_t::value_storage_t;
    using offset_t = typename layout_t::offset_t;
    using allocator_t = allocator_type_;
    using slot_ref_t = hash_slot_ref<element_type_, hasher_type_>;

    /** @brief Whether the element carries a mapped value, making the table a map rather than a set. */
    inline static constexpr bool has_values_k = layout_t::has_values_k;
    /** @brief Bytes one bucket occupies across all three regions, header included. */
    inline static constexpr std::size_t bytes_in_bucket_k = layout_t::bytes_in_bucket_k;
    /** @brief Whether teardown must run a key destructor. */
    inline static constexpr bool destruct_keys_k = !std::is_trivially_destructible<key_t>();
    /** @brief Whether teardown must run a value destructor. */
    inline static constexpr bool destruct_vals_k = has_values_k && !std::is_trivially_destructible<value_storage_t>();

    /** @brief Base of the single allocation the three regions are carved from. */
    std::byte *memory {};
    /** @brief Keys region, indexed by slot. */
    key_t *keys {};
    /** @brief Values region, indexed by slot, null for sets. */
    value_storage_t *values {};
    /** @brief Headers region, indexed by bucket. */
    hash_bucket_head_t *headers {};

    /** @brief Total number of slots, always a power of two. */
    offset_t slots_count {};
    /** @brief Rehash trigger at the 75% load factor. */
    offset_t growth_threshold {};
    /** @brief Count of populated slots. */
    offset_t populated_count {};
    /** @brief Count of deleted slots still holding tombstones. */
    offset_t deleted_count {};

    /** @brief Supplies and reclaims the single byte buffer behind the three regions. */
    [[no_unique_address]] allocator_t allocator {};

    hash_storage() noexcept = default;
    explicit hash_storage(allocator_t allocator_state) noexcept : allocator(std::move(allocator_state)) {}
    hash_storage(hash_storage &&other) noexcept { swap(other); }
    hash_storage &operator=(hash_storage &&other) noexcept {
        swap(other);
        return *this;
    }
    hash_storage(hash_storage const &) = delete;
    hash_storage &operator=(hash_storage const &) = delete;
    ~hash_storage() noexcept {
        destroy_elements();
        deallocate();
    }

    /**
     *  @brief Allocates and zeroes the buffer for @p slots, keeping @p allocator for its release.
     *  @return An empty storage when the allocation fails, which @c is_allocated reports.
     */
    [[nodiscard]] static hash_storage make(hash_slots_count_t slots, allocator_t allocator_state) noexcept {

        hash_storage result(std::move(allocator_state));

        // Every probe masks with `slots_count - 1`, so a count that is not a power of two turns
        // the probe sequence into an endless walk rather than a wrong answer.
        assert(slots.is_addressable() && "Slot count must be a power of two of at least one bucket");

        std::size_t const needed_bytes = memory_usage(slots);
        if (!needed_bytes) return result;

        result.memory = result.allocator.allocate(needed_bytes);
        if (!result.memory) return result;

        std::memset(result.memory, 0, needed_bytes);
        result.slots_count = static_cast<offset_t>(slots.raw);
        result.growth_threshold = result.slots_count * 3ul / 4ul;
        result.retarget_regions();
        return result;
    }

    /** @brief Total memory needed for the three regions of a table this size. */
    static constexpr std::size_t memory_usage(hash_slots_count_t slots) noexcept {
        return (slots.raw / hash_bucket_capacity_k) * bytes_in_bucket_k;
    }

    /** @brief Whether a buffer is behind the three regions. */
    bool is_allocated() const noexcept { return memory != nullptr; }

    offset_t bucket_count() const noexcept { return slots_count / hash_bucket_capacity_k; }
    std::size_t size_bytes() const noexcept { return bucket_count() * bytes_in_bucket_k; }

    /** @brief Recomputes the three region pointers from @c memory and @c slots_count. */
    void retarget_regions() noexcept {
        std::size_t const buckets = static_cast<std::size_t>(slots_count) / hash_bucket_capacity_k;
        keys = reinterpret_cast<key_t *>(memory);
        if constexpr (has_values_k) {
            values = reinterpret_cast<value_storage_t *>(memory + layout_t::bytes_for_keys_k * buckets);
            headers = reinterpret_cast<hash_bucket_head_t *>(
                memory + (layout_t::bytes_for_keys_k + layout_t::bytes_for_values_k) * buckets);
        }
        else {
            values = nullptr;
            headers = reinterpret_cast<hash_bucket_head_t *>(memory + layout_t::bytes_for_keys_k * buckets);
        }
    }

    /** @brief Points @p slot at the three regions and at @p slot_index within them. */
    template <typename slot_ref_type_>
    void retarget_slot(slot_ref_type_ &slot, offset_t slot_index) const noexcept {
        slot.keys_ = keys;
        slot.values_ = values;
        slot.headers_ = headers;
        slot.slot_ = slot_index;
    }

    /** @brief Runs the destructors of every live key and value, leaving the headers as they are. */
    void destroy_elements() noexcept {
        if constexpr (destruct_keys_k || destruct_vals_k) {
            if (!memory) return;
            slot_ref_t slot;
            retarget_slot(slot, 0);
            offset_t const buckets = bucket_count();
            for (offset_t bucket_index = 0; bucket_index != buckets; ++bucket_index) {
                slot.slot_ = bucket_index * hash_bucket_capacity_k;
                for_each_in_hash_bucket(slot, [](slot_ref_t const &live) noexcept {
                    if constexpr (destruct_keys_k) live.key_ref().~key_t();
                    if constexpr (destruct_vals_k) live.value_ref().~value_storage_t();
                });
            }
        }
    }

    /** @brief Destroys every element and zeroes the headers, keeping the buffer. */
    void clear() noexcept {
        if (!memory) return;
        destroy_elements();
        std::memset(memory, 0, size_bytes());
        populated_count = 0;
        deleted_count = 0;
    }

    /**
     *  @brief Returns the buffer to the allocator and drops every pointer and counter.
     *  @warning Elements are not destroyed, so call @c clear or @c destroy_elements first.
     */
    void deallocate() noexcept {
        if (memory) allocator.deallocate(memory, size_bytes());
        memory = nullptr;
        keys = nullptr;
        values = nullptr;
        headers = nullptr;
        slots_count = 0;
        growth_threshold = 0;
        populated_count = 0;
        deleted_count = 0;
    }

    /** @brief A cheap copy-less exchange of the buffer, the counters and the allocator. */
    void swap(hash_storage &other) noexcept {
        std::swap(memory, other.memory);
        std::swap(keys, other.keys);
        std::swap(values, other.values);
        std::swap(headers, other.headers);
        std::swap(slots_count, other.slots_count);
        std::swap(growth_threshold, other.growth_threshold);
        std::swap(populated_count, other.populated_count);
        std::swap(deleted_count, other.deleted_count);
        if constexpr (!std::is_empty<allocator_t>::value) std::swap(allocator, other.allocator);
    }
};

#pragma endregion Storage

} // namespace ashvardanian::smashtable
