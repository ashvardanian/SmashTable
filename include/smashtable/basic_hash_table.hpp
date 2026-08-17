/**
 *  @brief Lock-free concurrent hash-table optimized for CPU and GPU workloads.
 *  @author Ash Vardanian
 *  @file include/smashtable/basic_hash_table.hpp
 *  @date December 21, 2021
 *
 *  @section basic_hash_table_overview Overview
 *
 *  This file implements @c basic_hash_table, a high-performance open-addressing hash table
 *  designed for both CPU and GPU execution. Unlike STL containers, like @c std::unordered_map,
 *  this implementation:
 *  - Never throws exceptions (all operations are @c noexcept)
 *  - Supports lock-free concurrent operations without external synchronization
 *  - Avoids Compare-And-Swap (CAS) in favor of much-cheaper atomic ORs and XORs
 *  - Uses only 2 bits of metadata per slot (vs 1 byte in Google's SwissTable)
 *  - Organizes data in Structure-of-Arrays layout for better cache efficiency
 *  - Being SIMD-friendly and GPU-friendly with minimal modifications
 *
 *  @section basic_hash_table_core_design_goals Core Design Goals
 *
 *  @b Concurrency: Non-allocating operations have lock-free atomic variants ( @c emplace_atomic(),
 *  @c find_atomic(), @c erase_atomic(), @c update_atomic()) that enable fine-grained concurrent
 *  access from multiple threads without external synchronization. Atomic operations require prior
 *  @c reserve() to prevent reallocations. Never mix atomic and non-atomic operations on the same table.
 *
 *  @b Compactness: Each slot requires only 2 bits of overhead stored in a shared bucket header,
 *  encoding four states: free (00), deleted (01), populated (10), and locked (11). This is 4×
 *  more space-efficient than SwissTable's 1 byte per slot. We colocate 32× such slots, cause
 *  single- and dual-bit atomic operations aren't possible, but 64-bit ones are!
 *
 *  @b Cache-Efficiency: Keys and values are stored separately (Structure-of-Arrays) rather
 *  than interleaved (Array-of-Structures). During lookups that don't need values, only keys are
 *  loaded into cache, halving memory traffic for failed lookups.
 *
 *  @b Portability: The bucket size of 32 slots is chosen to:
 *  - Enable 64-bit atomic operations (2 bits × 32 = 64 bits) on all modern platforms
 *  - Match Nvidia GPU warp size for optimal cooperative thread group probing
 *  - Avoid 128-bit atomics which aren't universally supported
 *
 *  @b Performance: Optimized for bulk operations common in analytical and database workloads.
 *  The tags a call carries - @c assume_reserved_t, @c assume_unique_t, @c threadsafe_t - select the
 *  specialized code path at compile time, so no branch survives into the emitted code.
 *
 *  @section basic_hash_table_key_design_decisions Key Design Decisions
 *
 *  @par Load and Probing Strategy
 *  Uses linear probing with step size 1: @code (hash + i) % capacity @endcode.
 *  While simpler than quadratic probing or double hashing, linear probing offers:
 *  - Excellent cache locality (sequential memory access)
 *  - Predictable patterns for hardware prefetchers
 *  - Natural fit for GPU cooperative group strategies (threads probe adjacent slots)
 *  - Minimal instruction overhead in the critical path
 *
 *  It's recommended to keep the 75% load factor cap to ensure reasonable probe sequence lengths.
 *  Keep it under 40% for best performance to maximize single-probe hits leveraging @b SIMD-gathers.
 *
 *  @par Memory Layout
 *  Single allocation carved into three regions at cache-line offsets, organized for compaction.
 *  The offsets are multiples of the cache line; the base is only as aligned as the allocator makes
 *  it, which for @c std::allocator is the default new alignment, so a region boundary shares a line
 *  with its neighbour rather than starting one. An over-aligned key or value is not accommodated:
 *  @code
 *  // Layout: [keys_region | values_region | headers_region]
 *  std::byte* memory_ = allocate(bytes_in_bucket_k * buckets);
 *
 *  key_t* keys_ = (key_t*)memory_;                            // Direct indexing: keys_[slot]
 *  value_t* values_ = (value_t*)(memory_ + keys_bytes);       // Direct indexing: values_[slot]
 *  hash_bucket_head_t* headers_ = (hash_bucket_head_t*)(memory_ + keys_bytes + values_bytes);
 *                                                             // Bucket indexing: headers_[slot/32]
 *  @endcode
 *
 *  This ordering enables @b compaction: when squeezing the hash table into a dense array, the
 *  keys and values are already contiguous at the start of @c memory_. The headers at the end can be
 *  discarded, and the buffer directly reused for storage without copying.
 *
 *  @par SIMD Acceleration
 *  Bulk operations can benefit from vectorized hashing:
 *  - AVX-512: Hash many strings in parallel
 *  - SIMD metadata scanning for iteration
 *  - Prefetching probe sequences to hide memory latency
 *
 *  Note: Line Fill Buffer (LFB) limits on x86 (10 concurrent L1D misses) can bottleneck gather
 *  instructions, so vectorized hashing provides more benefit than vectorized probing.
 *
 *  @par GPU Deployment
 *  The SoA layout and bucket size are GPU-friendly. Cooperative group probing (4-8 threads per
 *  key) achieves coalesced memory access despite random hashing:
 *  @code
 *  // Threads 0-3 probe slots [hash+0, hash+1, hash+2, hash+3] in parallel
 *  // 4 adjacent addresses form a single coalesced transaction
 *  probe_idx = (hash + group.thread_rank()) % capacity;
 *  @endcode
 *
 *  This strategy, used by NVIDIA's cuCollections, shows 13% faster inserts and 40% faster
 *  lookups under high load factors (>=60%) compared to one-thread-per-key approaches. The larger
 *  improvement for lookups occurs because reads don't contend for atomic operations - the full
 *  benefit of coalesced memory access is realized. Inserts see smaller gains as they still
 *  serialize on atomic compare-and-swap operations for the bucket header. Performance measured on
 *  H100 with 64-bit key-value pairs. Cooperative groups of size 4 provide the sweet spot: smaller
 *  groups (1-2) suffer from uncoalesced access, larger groups (16-32) underutilize threads when
 *  probe sequences terminate early.
 *
 *  @par Small String Optimization
 *  Inline storage for keys under 16 bytes can eliminate heap allocations and enable efficient
 *  vectorized hashing of fixed-size data. Two approaches possible:
 *  - Dual tables: separate @c basic_hash_table<array<char,16>, V> for short strings
 *  - Variant storage: @c union of inline buffer and pointer (standard SSO pattern)
 *
 *  @see https://en.wikipedia.org/wiki/Hash_table
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
#include <iterator>    // `std::input_iterator_tag`
#include <limits>      // `std::numeric_limits`
#include <memory>      // `std::allocator`
#include <type_traits> // `std::is_same`, `std::enable_if`
#include <utility>     // `std::declval`, `std::move`

#include "shared.hpp"

namespace ashvardanian::smashtable {

#pragma region Bucket Metadata

/** @brief Platform cache line size in bytes, typically 64 on modern CPUs. */
inline constexpr std::size_t cache_line_bytes_k = 64;

/** @brief Number of slots sharing one bucket header, sized so the header fits a 64-bit atomic. */
inline constexpr std::size_t hash_bucket_capacity_k = 32;

/** @brief Detects iterators by the presence of an @c iterator_category, without hard-erroring on other types. */
template <typename type_, typename = void>
struct is_iterator_type : std::false_type {};

template <typename type_>
struct is_iterator_type<type_, std::void_t<typename std::iterator_traits<type_>::iterator_category>> : std::true_type {
};

/** @brief Type trait to check if a type is an iterator. */
template <typename type_>
constexpr bool is_iterator() {
    return is_iterator_type<std::remove_cvref_t<type_>>::value;
}

/**
 *  @brief Bucket header containing metadata for @c hash_bucket_capacity_k slots.
 *    Stores two parallel 32-bit bitmasks (population and deletion states) that combine to encode
 *    four possible states per slot using 2 bits each.
 *
 *  @section basic_hash_table_slot_state_encoding Slot State Encoding
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
    constexpr bool try_lock() const noexcept { return true; }

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

    /** @brief One @c fetch_or attempt, reporting whether the slot was free to take. */
    bool try_lock() const noexcept {
        hash_bucket_head_t const header_mask = header_mask_();
        std::atomic_ref<std::uint64_t> atomic_header(base_t::header_ref().u64);
        future_header_.u64 = atomic_header.fetch_or(header_mask.u64, std::memory_order_acquire) & header_mask.u64;
        return future_header_.u64 != header_mask.u64;
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

#pragma region Iterators

/**
 *  @brief Forward iterator for hash tables using simplified slot-based addressing.
 *    Inherits the three region pointers and the slot index from @c hash_slot_ref, and adds a
 *    counter of unvisited slots. Implements @c std::forward_iterator_tag for sequential traversal.
 *
 *  @note This iterator is not thread-safe. For concurrent iteration, use @c for_each with
 *    appropriate synchronization or operate on a snapshot of the table.
 *
 *  @c slots_remaining counts the slots left to visit and reaching zero @b is the end, which is what
 *  @c is_end and @c isnt_end report and what the @c end_sentinel_t comparisons consult. Two
 *  iterators compare equal when they name the same slot of the same keys region. @c advance steps
 *  to the next populated slot and must not run at the end, while @c skip_non_populated walks past
 *  free, deleted and locked slots until a populated one or the end is reached.
 */
template <typename element_type_, typename hasher_type_>
struct hash_table_iterator : public hash_slot_ref<element_type_, hasher_type_> {

    using base_t = hash_slot_ref<element_type_, hasher_type_>;
    using element_t = typename base_t::element_t;
    using offset_t = typename base_t::offset_t;
    using base_t::keys_;
    using base_t::slot_;

    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = element_t;
    using reference = typename base_t::dereference_t;
    using pointer = void;

    offset_t slots_remaining {};

    [[nodiscard]] bool operator==(end_sentinel_t) const noexcept { return is_end(); }
    [[nodiscard]] bool operator!=(end_sentinel_t) const noexcept { return isnt_end(); }
    [[nodiscard]] bool isnt_end() const noexcept { return slots_remaining != 0; }
    [[nodiscard]] bool is_end() const noexcept { return slots_remaining == 0; }

    bool operator==(hash_table_iterator const &other) const noexcept {
        return (keys_ == other.keys_) & (slot_ == other.slot_);
    }
    bool operator!=(hash_table_iterator const &other) const noexcept {
        return (keys_ != other.keys_) | (slot_ != other.slot_);
    }

    hash_table_iterator &operator++() noexcept {
        advance();
        return *this;
    }

    hash_table_iterator operator++(int) noexcept {
        hash_table_iterator copy = *this;
        advance();
        return copy;
    }

    void advance() noexcept {
        assert(!is_end() && "Going out of range!");
        --slots_remaining;
        ++slot_;
        skip_non_populated();
    }

    void skip_non_populated() noexcept {
        while (slots_remaining != 0 && !base_t::is_populated()) {
            --slots_remaining;
            ++slot_;
        }
    }
};

/** @brief Wraps @c std::hash default hasher that delays type resolution until invocation. */
struct lazy_std_hash_t {

    template <typename key_type_>
    std::size_t operator()(key_type_ const &key) const noexcept {
        return std::hash<key_type_> {}(key);
    }
};

#pragma endregion Iterators

/**
 *  @brief Lock-free concurrent hash table with linear probing and Structure-of-Arrays layout.
 *    Compatible with @c std::unordered_set interface. See file header for detailed design rationale.
 *
 *  @tparam element_type_ Hashable and equality-comparable key type, or @c mapping<K,V> for maps.
 *  @tparam hasher_type_ Hash function type. Must be copy-constructible. Defaults to a @c std::hash wrapper.
 *  @tparam equals_type_ Equality predicate supporting heterogeneous lookups. Must be copy-constructible.
 *    Defaults to a transparent @c std::equal_to.
 *  @tparam allocator_type_ Allocator for internal memory management. Defaults to @c std::allocator<std::byte>.
 *
 *  @see https://en.cppreference.com/w/cpp/container/unordered_set
 *  @see https://en.cppreference.com/w/cpp/container/unordered_map
 */
template <typename element_type_, typename hasher_type_ = lazy_std_hash_t, typename equals_type_ = std::equal_to<>,
          typename allocator_type_ = std::allocator<std::byte>>
class basic_hash_table {

    using layout_t = hash_layout_for<element_type_, hasher_type_>;
    using key_t = typename layout_t::key_t;
    using value_t = typename layout_t::value_t;
    using value_storage_t = typename layout_t::value_storage_t;
    using element_t = typename layout_t::element_t;
    using offset_t = typename layout_t::offset_t;

    /** @brief Whether the element carries a mapped value, making this table a map rather than a set. */
    inline static constexpr bool has_values_k = layout_t::has_values_k;
    /** @brief Cache-line-aligned size of one bucket's keys region. */
    inline static constexpr std::size_t bytes_for_keys_k = layout_t::bytes_for_keys_k;
    /** @brief Cache-line-aligned size of one bucket's values region, zero for sets. */
    inline static constexpr std::size_t bytes_for_values_k = layout_t::bytes_for_values_k;
    /** @brief Bytes one bucket occupies across all three regions, header included. */
    inline static constexpr std::size_t bytes_in_bucket_k = layout_t::bytes_in_bucket_k;
    /** @brief Whether the hasher is stateless, so swapping tables need not exchange it. */
    inline static constexpr bool hasher_is_empty_k = std::is_empty<hasher_type_>::value;
    /** @brief Whether the equality predicate is stateless, so swapping tables need not exchange it. */
    inline static constexpr bool equals_is_empty_k = std::is_empty<equals_type_>::value;
    /** @brief Whether the allocator is stateless, so swapping tables need not exchange it. */
    inline static constexpr bool allocator_is_empty_k = std::is_empty<allocator_type_>::value;

    using hasher_t = hasher_type_;
    using equals_t = equals_type_;
    using allocator_t = allocator_type_;
    using iterator_t = hash_table_iterator<element_type_, hasher_t>;
    using const_iterator_t = hash_table_iterator<element_type_ const, hasher_t>;
    using slot_ref_t = hash_slot_ref<element_type_, hasher_t>;
    using const_slot_ref_t = hash_slot_ref<element_type_ const, hasher_t>;
    using atomic_slot_ref_t = hash_atomic_slot_ref<element_type_, hasher_t>;
    using const_atomic_slot_ref_t = hash_atomic_slot_ref<element_type_ const, hasher_t>;

    static_assert(!std::is_reference<key_t>(), "Keys can't be references!");
    static_assert(!std::is_reference<value_t>(), "Values can't be references!");
    static_assert(std::is_unsigned<offset_t>(),
                  "Hash value must be an unsigned integer, like std::uint32_t or std::uint64_t!");

    /** @brief Whether erasure and teardown must run a key destructor. */
    inline static constexpr bool destruct_keys_k = !std::is_trivially_destructible<key_t>();
    /** @brief Whether erasure and teardown must run a value destructor. */
    inline static constexpr bool destruct_vals_k = has_values_k && !std::is_trivially_destructible<value_storage_t>();

  public:
    /**
     *  @brief A return type for the insert function, similar to @c insert_return_type in STL.
     *  @c position indicates the inserted or conflicted element, if none exists points to the end.
     *  @c inserted true if the given element was inserted or updated.
     */
    struct insert_result_t {
        iterator_t position;
        bool inserted = false;
    };

    /*  STL-compatibility definitions, identical to that of `std::unordered_map`:
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map
     */
    using key_type = key_t;
    using mapped_type = value_t;
    using result_type = insert_result_t;
    using value_type = element_t;
    /** @brief The element as an owned pair, which @c value_type cannot be: it views two regions. */
    using owned_value_type = typename layout_t::element_copy_t;
    using size_type = offset_t;
    using difference_type = std::ptrdiff_t;
    using key_compare = equals_t;
    using allocator_type = allocator_t;
    using reference = element_t;
    using const_reference = element_t;
    using iterator = iterator_t;
    using const_iterator = const_iterator_t;
    using hasher = hasher_t;
    using key_equal = equals_t;

    /*  Traits the transactional adapters dispatch on.
     */
    using is_associative = std::bool_constant<has_values_k>;
    using callback_reads = std::true_type;
    using is_transactional = std::false_type;

    /**
     *  @brief Restates this table over a different element type, hasher and equality.
     *    A store that decorates its elements - wrapping them in version metadata, say - needs the
     *    equality restated alongside the hasher, since both must address the decorated shape.
     */
    template <typename other_element_type_, typename other_hasher_type_, typename other_equals_type_ = equals_t>
    using rebind = basic_hash_table<other_element_type_, other_hasher_type_, other_equals_type_, allocator_t>;

  private:
    /**
     *  @brief Base of the single allocation the three regions are carved from.
     *    Layout: [keys | values | headers] enabling zero-copy compaction.
     */
    std::byte *memory_ {};
    /** @brief Keys region, indexed by slot. */
    key_t *keys_ {};
    /** @brief Values region, indexed by slot, null for sets. */
    value_storage_t *values_ {};
    /** @brief Headers region, indexed by bucket. */
    hash_bucket_head_t *headers_ {};

    /** @brief Total number of slots, always a power of two. */
    offset_t slots_count_ {};
    /** @brief Rehash trigger at the 75% load factor. */
    offset_t growth_threshold_ {};
    /** @brief Count of populated slots. */
    offset_t populated_count_ {};
    /** @brief Count of deleted slots still holding tombstones. */
    offset_t deleted_count_ {};

    /** @brief Hashes a key down to its initial probe offset. */
    [[no_unique_address]] hasher_t hasher_ {};
    /** @brief Decides whether a probed key matches the wanted one. */
    [[no_unique_address]] equals_t equals_ {};
    /** @brief Supplies and reclaims the single byte buffer behind the three regions. */
    [[no_unique_address]] allocator_t allocator_ {};

    /** @brief Recomputes the three region pointers from @c memory_ and @c slots_count_. */
    void retarget_regions() noexcept {
        std::size_t const buckets = static_cast<std::size_t>(slots_count_) / hash_bucket_capacity_k;
        keys_ = reinterpret_cast<key_t *>(memory_);
        if constexpr (has_values_k) {
            values_ = reinterpret_cast<value_storage_t *>(memory_ + bytes_for_keys_k * buckets);
            headers_ =
                reinterpret_cast<hash_bucket_head_t *>(memory_ + (bytes_for_keys_k + bytes_for_values_k) * buckets);
        }
        else {
            values_ = nullptr;
            headers_ = reinterpret_cast<hash_bucket_head_t *>(memory_ + bytes_for_keys_k * buckets);
        }
    }

    /**
     *  @brief Allocates and zeroes a table of the requested slot count.
     *    On allocation failure the table is left empty, which the factories report as an error.
     */
    basic_hash_table(hash_slots_count_t slots, hasher_t hasher, equals_t equals, allocator_t allocator) noexcept
        : hasher_(std::move(hasher)), equals_(std::move(equals)), allocator_(std::move(allocator)) {

        // Every probe masks with `slots_count_ - 1`, so a count that is not a power of two turns
        // the probe sequence into an endless walk rather than a wrong answer.
        assert(slots.is_addressable() && "Slot count must be a power of two of at least one bucket");

        std::size_t const needed_bytes = memory_usage(slots);
        if (!needed_bytes) return;

        memory_ = allocator_.allocate(needed_bytes);
        if (!memory_) return;

        std::memset(memory_, 0, needed_bytes);
        slots_count_ = static_cast<offset_t>(slots.raw);
        growth_threshold_ = slots_count_ * 3ul / 4ul;
        retarget_regions();
    }

  public:
#pragma region Constructors

    /**
     *  @brief Default constructor, avoiding memory allocations, accepting a pre-constructed hasher functor,
     *    an equality operator, and the allocator state.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/unordered_map
     */
    basic_hash_table(hasher_t hasher = {}, equals_t equals = {}, allocator_t allocator = {}) noexcept
        : hasher_(std::move(hasher)), equals_(std::move(equals)), allocator_(std::move(allocator)) {}

    basic_hash_table(basic_hash_table &&other) noexcept { swap(other); }

    basic_hash_table &operator=(basic_hash_table &&other) noexcept {
        swap(other);
        return *this;
    }

    basic_hash_table(basic_hash_table const &) noexcept = delete;
    basic_hash_table &operator=(basic_hash_table const &) noexcept = delete;
    basic_hash_table(offset_t, hasher_t = {}, equals_t = {}, allocator_t = {}) noexcept = delete;

    ~basic_hash_table() noexcept {
        if (!memory_) return;
        clear(assume_reserved_t {});
        deallocate(assume_reserved_t {});
    }

    /**
     *  @brief Most commonly used @b constructor-like interface, sized by the element count you plan to store.
     *  @return An empty-status @c expected if the allocation has failed.
     */
    [[nodiscard]] static expected<basic_hash_table> make(offset_t planned_elements, hasher_t hasher = {},
                                                         equals_t equals = {}, allocator_t allocator = {}) noexcept {
        return make(hash_slots_count_t {planned_elements}, std::move(hasher), std::move(equals), std::move(allocator));
    }

    /**
     *  @brief Most commonly used @b constructor-like interface, sized by the exact slot count.
     *  @return An empty-status @c expected if the allocation has failed.
     */
    [[nodiscard]] static expected<basic_hash_table> make(hash_slots_count_t slots, hasher_t hasher = {},
                                                         equals_t equals = {}, allocator_t allocator = {}) noexcept {
        basic_hash_table table(slots, std::move(hasher), std::move(equals), std::move(allocator));
        if (slots.raw && !table.memory_) return expected<basic_hash_table>(status_t {out_of_memory_heap_k});
        return expected<basic_hash_table>(std::move(table), status_t {success_k});
    }

    /**
     *  @brief Builds a table pre-sized for the range and fills it.
     *  @return An empty-status @c expected if the allocation has failed.
     */
    template <typename begin_iterator_type_, typename end_iterator_type_,
              typename std::enable_if<is_iterator<begin_iterator_type_>(), int>::type = 0>
    [[nodiscard]] static expected<basic_hash_table> make(begin_iterator_type_ begin, end_iterator_type_ end) noexcept {
        auto table = make(hash_slots_count_t {static_cast<std::size_t>(std::distance(begin, end))});
        if (!table) return table;
        table->insert(begin, end, assume_reserved_t {});
        return table;
    }

#pragma endregion Constructors

#pragma region Metadata

    bool empty() const noexcept { return !populated_count_; }
    offset_t size() const noexcept { return populated_count_; }

    offset_t optimal_capacity() const noexcept { return growth_threshold_; }
    offset_t capacity() const noexcept { return std::max(optimal_capacity(), size()); }

    std::size_t size_bytes() const noexcept { return bucket_count() * bytes_in_bucket_k; }

    hash_slots_count_t slots_count() const noexcept {
        hash_slots_count_t result;
        result.raw = slots_count_;
        return result;
    }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/bucket_count
     */
    offset_t bucket_count() const noexcept { return slots_count_ / hash_bucket_capacity_k; }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/max_bucket_count
     */
    offset_t max_bucket_count() const noexcept { return std::numeric_limits<offset_t>::max() / hash_bucket_capacity_k; }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/hash_function
     */
    hasher_t hash_function() const noexcept { return hasher_; }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/key_eq
     */
    equals_t key_eq() const noexcept { return equals_; }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/get_allocator
     */
    allocator_t get_allocator() const noexcept { return allocator_; }

#pragma endregion Metadata

#pragma region Search

    /**
     *  @brief Search optimized for @c find: checks if a value is present.
     *    Unlike @c search_to_insert or @c search_to_upsert, doesn't track "deleted" slots.
     *
     *  @param[in] wanted Hashable and comparable with key object.
     *  @param[in] call A callback receiving @c const_slot_ref_t or an atomic @c const_atomic_slot_ref_t
     *    to an initialized matching object.
     *  @param[in] tags Markers for special acceleration:
     *    @c assume_reserved_t avoids null checks, @c threadsafe_t enables lock-less concurrency.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    void search_to_find(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_...) const noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), const_atomic_slot_ref_t,
                                         const_slot_ref_t>;
        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (!populated_count_) [[unlikely]]
                return;

        offset_t const offset_mask = slots_count_ - 1;
        offset_t const initial_offset = hasher_(wanted) & offset_mask;
        offset_t off = initial_offset;
        ref_t slot;

        while (true) {
            unsafe_retarget(slot, off);
            slot.lock();

            // Checking for equality isn't safe, if we operate on uninitialized
            // memory and call some complex comparison operators on them.
            // So instead of one runtime `if`, we have two nested `if`s.
            if (slot.is_populated()) {
                if (equals_(slot.key(), wanted)) {
                    call(slot);
                    slot.unlock();
                    break;
                }
                slot.unlock();
                if constexpr (contains_type<first_match_t, tags_types_...>()) break;
                off = (off + 1) & offset_mask;
                if (off == initial_offset) break;
            }
            else if (slot.is_deleted()) {
                slot.unlock();
                if constexpr (contains_type<first_match_t, tags_types_...>()) break;
                off = (off + 1) & offset_mask;
                if (off == initial_offset) break;
            }
            else {
                // A "free" slot ends the probe sequence.
                slot.unlock();
                break;
            }
        }
    }

    /**
     *  @brief Search optimized for @c find: checks if a value is present.
     *    Unlike @c search_to_insert or @c search_to_upsert, doesn't track "deleted" slots.
     *
     *  @param[in] wanted Hashable and comparable with key object.
     *  @param[in] call A callback receiving @c slot_ref_t or an atomic @c atomic_slot_ref_t
     *    to an initialized matching object.
     *  @param[in] tags Markers for special acceleration:
     *    @c assume_reserved_t avoids null checks, @c threadsafe_t enables lock-less concurrency.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    void search_to_find(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_...) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), atomic_slot_ref_t, slot_ref_t>;
        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (!populated_count_) [[unlikely]]
                return;

        offset_t const offset_mask = slots_count_ - 1;
        offset_t const initial_offset = hasher_(wanted) & offset_mask;
        offset_t off = initial_offset;
        ref_t slot;

        while (true) {
            unsafe_retarget(slot, off);
            slot.lock();

            if (slot.is_populated()) {
                if (equals_(slot.key(), wanted)) {
                    call(slot);
                    slot.unlock();
                    break;
                }
                slot.unlock();
                if constexpr (contains_type<first_match_t, tags_types_...>()) break;
                off = (off + 1) & offset_mask;
                if (off == initial_offset) break;
            }
            else if (slot.is_deleted()) {
                slot.unlock();
                if constexpr (contains_type<first_match_t, tags_types_...>()) break;
                off = (off + 1) & offset_mask;
                if (off == initial_offset) break;
            }
            else {
                slot.unlock();
                break;
            }
        }
    }

    /** @brief Adds to a counter, atomically when @c threadsafe_t is present, returning the new value. */
    template <typename... tags_types_>
    offset_t advance(offset_t &count, offset_t addend, tags_types_...) const noexcept {
        if constexpr (contains_type<threadsafe_t, tags_types_...>()) return atomic_add_fetch(count, addend);
        else return count += addend;
    }

    /** @brief Subtracts from a counter, atomically when @c threadsafe_t is present, returning the new value. */
    template <typename... tags_types_>
    offset_t decrement(offset_t &count, offset_t subtrahend, tags_types_...) const noexcept {
        if constexpr (contains_type<threadsafe_t, tags_types_...>()) return atomic_sub_fetch(count, subtrahend);
        else return count -= subtrahend;
    }

    /** @brief Writes the post-update size into a @c return_new_size_t tag, when the caller passed one. */
    template <typename... tags_types_>
    void export_new_size(offset_t new_size, tags_types_ &...tags) const noexcept {
        if constexpr (contains_type<return_new_size_t, tags_types_...>())
            if (auto *destination = find_tag<return_new_size_t>(tags...); destination && destination->out)
                *destination->out = static_cast<std::size_t>(new_size);
    }

    /**
     *  @brief Search optimized for @c insert of a unique key.
     *    Unlike @c search_to_upsert avoids potentially expensive equality comparisons,
     *    knowing that the incoming key is different from all present members.
     *
     *  @param[in] wanted Hashable and comparable with key object.
     *  @param[in] call A callback receiving @c slot_ref_t or an atomic @c atomic_slot_ref_t
     *    to UN-initialized memory, where an element should be built.
     *  @param[in] tags Markers for special acceleration:
     *    @c assume_reserved_t avoids null checks, @c threadsafe_t enables lock-less concurrency.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    void search_to_insert(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), atomic_slot_ref_t, slot_ref_t>;

        // We pre-increment the counter, before beginning the potentially slow
        // search. It helps avoiding premature cleanup from a different thread.
        export_new_size(advance(populated_count_, 1, tags...), tags...);

        offset_t const offset_mask = slots_count_ - 1;
        offset_t off = hasher_(wanted) & offset_mask;
        ref_t slot;

        // Bounded by the table itself. A caller promising @c assume_reserved_t promises a free slot
        // exists, and probing forever is the wrong way to discover the promise was false - it hangs
        // with nothing to show, in a build where the assertion below is gone.
        bool stored = false;
        for (offset_t probes = 0; probes != slots_count_ && !stored; ++probes) {
            unsafe_retarget(slot, off);
            slot.lock();

            if (slot.is_free() || slot.is_deleted()) {
                call(slot);

                // Update the cell state before unlocking.
                bool const did_find_deleted = slot.is_deleted();
                slot.mark_populated();
                slot.unlock();

                // Decrement this counter afterwards - in a relaxed manner.
                // Somebody else might be already searching this slot,
                // we don't want them to wait :)
                decrement(deleted_count_, did_find_deleted, tags...);
                stored = true;
                continue;
            }

            // A slot is "populated", definitely with a different key!
            slot.unlock();
            off = (off + 1) & offset_mask;
        }

        assert(stored && "Reserve before inserting: every slot was taken, so nothing was stored");
    }

    /**
     *  @brief Search optimized for potential @c upserts.
     *    More expensive than @c search_to_find and @c search_to_insert, so pick this method wisely.
     *
     *  @param[in] wanted Hashable and comparable with key object.
     *  @param[in] call_unused A callback receiving @c slot_ref_t or an atomic @c atomic_slot_ref_t
     *    to UN-initialized memory, where an element should be built.
     *  @param[in] call_equal A callback receiving @c slot_ref_t or an atomic @c atomic_slot_ref_t
     *    to an initialized matching object.
     *  @param[in] tags Markers for special acceleration:
     *    @c assume_reserved_t avoids null checks, @c threadsafe_t enables lock-less concurrency.
     */
    template <typename comparable_key_type_, typename callback_unused_type_, typename callback_equal_type_,
              typename... tags_types_>
    void search_to_upsert(comparable_key_type_ &&wanted, callback_unused_type_ &&call_unused,
                          callback_equal_type_ &&call_equal, tags_types_... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), atomic_slot_ref_t, slot_ref_t>;

        offset_t const offset_mask = slots_count_ - 1;
        offset_t const initial_offset = hasher_(wanted) & offset_mask;
        offset_t off = initial_offset;

        // The slot currently under inspection, and the first observed "deleted" slot.
        // The latter stays locked from the moment it is remembered until we commit.
        ref_t current, reusable;
        bool did_find_deleted = false;

        while (true) {
            unsafe_retarget(current, off);
            current.lock();

            if (current.is_populated()) {
                // We have found a filled slot and must check for match:
                // > If matches: unlock the remembered slot, call back, unlock current, exit.
                // > If not: unlock current and jump forward.
                if (equals_(current.key(), wanted)) {
                    if (did_find_deleted) reusable.unlock();
                    call_equal(current);
                    current.unlock();
                    // Overwriting leaves the count alone, but a caller that asked for the new size
                    // still needs it written; a zero delta reads it under the same atomicity.
                    export_new_size(advance(populated_count_, 0, tags...), tags...);
                    return;
                }
                if constexpr (contains_type<first_match_t, tags_types_...>()) {
                    if (did_find_deleted) reusable.unlock();
                    call_unused(current);
                    current.mark_populated();
                    current.unlock();
                    return;
                }
                current.unlock();
                off = (off + 1) & offset_mask;
                assert(off != initial_offset && "Poor hash table usage!");
                continue;
            }

            if (current.is_deleted()) {
                // Found a deleted slot:
                // > If it is the first one: keep it locked for later, jump forward.
                // > If we already saw one: unlock current, just jump forward.
                if (!did_find_deleted) {
                    reusable = current;
                    did_find_deleted = true;
                }
                else { current.unlock(); }
                off = (off + 1) & offset_mask;
                assert(off != initial_offset && "Poor hash table usage!");
                continue;
            }

            // In case of a "free" slot:
            // > If a deleted slot was seen earlier in this probe: reuse it, drop the free one.
            // > Otherwise: take the free slot itself.
            if (did_find_deleted) current.unlock();
            ref_t &chosen = did_find_deleted ? reusable : current;
            call_unused(chosen);
            chosen.mark_populated();
            chosen.unlock();

            // Change these counters afterwards - in a relaxed manner.
            // Somebody else might be already searching for this slot,
            // we don't want them to wait :)
            export_new_size(advance(populated_count_, 1, tags...), tags...);
            decrement(deleted_count_, did_find_deleted, tags...);
            return;
        }
    }

    /**
     *  @brief Retargets an element reference to a specific slot using direct array indexing.
     *  @param[out] slot Slot reference to retarget with new region pointers and a new index.
     *  @param[in] slot_index Absolute slot index in [0, slots_count_).
     */
    template <typename element_ref_type_>
    void unsafe_retarget(element_ref_type_ &slot, offset_t slot_index) const noexcept {
        slot.keys_ = keys_;
        slot.values_ = values_;
        slot.headers_ = headers_;
        slot.slot_ = slot_index;
    }

#pragma endregion Search

#pragma region Lookups

    /** @brief Number of slots between the reference's position and the end of the table. */
    template <typename address_or_iterator_type_>
    offset_t slots_remaining_after(address_or_iterator_type_ &&slot) const noexcept {
        return slots_count_ - slot.slot_;
    }

    /** @brief Reports whether a key equivalent to @p wanted is present. */
    template <typename comparable_key_type_, typename... tags_types_>
    bool contains(comparable_key_type_ &&wanted, tags_types_... tags) const noexcept {
        bool result = false;
        search_to_find(
            std::forward<comparable_key_type_>(wanted), [&result](auto const &) noexcept { result = true; }, tags...);
        return result;
    }

    /**
     *  @brief Similar to STL, searches for the element with the given key.
     *  @return End iterator, iff element with such key is missing.
     */
    template <typename comparable_key_type_ = key_t const &>
    const_iterator_t find(comparable_key_type_ &&wanted) const noexcept {
        const_iterator_t it;
        unsafe_retarget(it, slots_count_);
        it.slots_remaining = 0;
        search_to_find(std::forward<comparable_key_type_>(wanted),
                       [&it](auto const &slot) noexcept { it.slot_ = slot.slot_; });
        it.slots_remaining = slots_remaining_after(it);
        return it;
    }

    /**
     *  @brief Similar to STL, searches for the element with the given key.
     *  @return End iterator, iff element with such key is missing.
     */
    template <typename comparable_key_type_ = key_t const &>
    iterator_t find(comparable_key_type_ &&wanted) noexcept {
        iterator_t it;
        unsafe_retarget(it, slots_count_);
        it.slots_remaining = 0;
        search_to_find(std::forward<comparable_key_type_>(wanted),
                       [&it](auto const &slot) noexcept { it.slot_ = slot.slot_; });
        it.slots_remaining = slots_remaining_after(it);
        return it;
    }

    /**
     *  @brief Lock-free atomic lookup that invokes callback if key is found.
     *    Requires prior @c reserve() to prevent reallocations during concurrent access.
     *
     *  @param[in] wanted Key to search for, hashable and comparable.
     *  @param[in] callback Invoked with an atomic reference to the element if found.
     *  @return True if key was found and callback invoked, false otherwise.
     *
     *  @note Do not mix with non-atomic operations on the same table instance.
     */
    template <typename comparable_key_type_, typename callback_type_>
    bool find_atomic(comparable_key_type_ &&wanted, callback_type_ &&callback) const noexcept {
        bool found = false;
        search_to_find(
            std::forward<comparable_key_type_>(wanted),
            [&](const_atomic_slot_ref_t const &slot) noexcept {
                callback(slot);
                found = true;
            },
            threadsafe_t {}, assume_reserved_t {});
        return found;
    }

    /**
     *  @brief Lock-free atomic check for key existence.
     *  @param[in] wanted Key to search for.
     *  @return True if key exists in table, false otherwise.
     */
    template <typename comparable_key_type_>
    bool contains_atomic(comparable_key_type_ &&wanted) const noexcept {
        return find_atomic(std::forward<comparable_key_type_>(wanted), no_op_fn_t {});
    }

    /**
     *  @brief Returns a reference to the mapped value of the element with the given key.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/at
     */
    template <typename comparable_key_type_ = key_t const &>
    value_storage_t &at(comparable_key_type_ &&key) noexcept {
        static_assert(has_values_k, "at() is only available for maps, not sets");
        slot_ref_t result;
        unsafe_retarget(result, slots_count_);
        search_to_find(std::forward<comparable_key_type_>(key),
                       [&result](slot_ref_t const &slot) noexcept { result.slot_ = slot.slot_; });
        assert(result.slot_ != slots_count_ && "The object must be already present");
        return result.value_ref();
    }

    /**
     *  @brief Returns a const reference to the mapped value of the element with the given key.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/at
     */
    template <typename comparable_key_type_ = key_t const &>
    value_storage_t const &at(comparable_key_type_ &&key) const noexcept {
        static_assert(has_values_k, "at() is only available for maps, not sets");
        const_slot_ref_t result;
        unsafe_retarget(result, slots_count_);
        search_to_find(std::forward<comparable_key_type_>(key),
                       [&result](const_slot_ref_t const &slot) noexcept { result.slot_ = slot.slot_; });
        assert(result.slot_ != slots_count_ && "The object must be already present");
        return result.value_ref();
    }

    /**
     *  @brief A temporarily banned function, that is replaced by @b @c at().
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/operator_at
     */
    template <typename comparable_key_type_ = key_t const &>
    void operator[](comparable_key_type_ &&) = delete;

    /**
     *  @brief Number of elements matching the key, either 1 or 0 as duplicates aren't allowed.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/count
     */
    template <typename comparable_key_type_ = key_t const &>
    std::size_t count(comparable_key_type_ &&key) const noexcept {
        return contains(std::forward<comparable_key_type_>(key));
    }

#pragma endregion Lookups

#pragma region Scans

    iterator_t begin() noexcept {
        iterator_t it;
        unsafe_retarget(it, 0);
        if (empty()) {
            it.slot_ = slots_count_;
            it.slots_remaining = 0;
        }
        else {
            it.slots_remaining = slots_count_;
            it.skip_non_populated();
        }
        return it;
    }

    const_iterator_t cbegin() const noexcept {
        const_iterator_t it;
        unsafe_retarget(it, 0);
        if (empty()) {
            it.slot_ = slots_count_;
            it.slots_remaining = 0;
        }
        else {
            it.slots_remaining = slots_count_;
            it.skip_non_populated();
        }
        return it;
    }

    const_iterator_t begin() const noexcept { return cbegin(); }
    end_sentinel_t end() const noexcept { return end_sentinel_t {}; }
    end_sentinel_t cend() const noexcept { return end_sentinel_t {}; }

    /** @brief Invokes the callback for every populated slot, bucket by bucket. */
    template <typename callback_type_ = no_op_fn<slot_ref_t>>
    void for_each(callback_type_ &&callback) noexcept {
        slot_ref_t slot;
        unsafe_retarget(slot, 0);
        offset_t const buckets = bucket_count();
        for (offset_t bucket_index = 0; bucket_index != buckets; ++bucket_index) {
            slot.slot_ = bucket_index * hash_bucket_capacity_k;
            for_each_in_hash_bucket(slot, callback);
        }
    }

    /** @brief Invokes the callback for every populated slot, bucket by bucket. */
    template <typename callback_type_ = no_op_fn<const_slot_ref_t>>
    void for_each(callback_type_ &&callback) const noexcept {
        const_slot_ref_t slot;
        unsafe_retarget(slot, 0);
        offset_t const buckets = bucket_count();
        for (offset_t bucket_index = 0; bucket_index != buckets; ++bucket_index) {
            slot.slot_ = bucket_index * hash_bucket_capacity_k;
            for_each_in_hash_bucket(slot, callback);
        }
    }

#pragma endregion Scans

#pragma region Insertions

    template <typename convertible_key_type_, typename convertible_value_type_>
    static constexpr bool can_use_map_emplace() {
        return has_values_k && std::is_constructible<key_t, convertible_key_type_ &&>() &&
               std::is_constructible<value_storage_t, convertible_value_type_ &&>();
    }

    template <typename convertible_key_type_>
    static constexpr bool can_use_set_emplace() {
        return !has_values_k && std::is_constructible<key_t, convertible_key_type_ &&>();
    }

    template <typename... tags_types_>
    static constexpr bool emplace_returns() {
        return contains_type<return_position_t, tags_types_...>();
    }

    template <typename... tags_types_>
    using emplace_return = std::conditional_t<emplace_returns<tags_types_...>(), insert_result_t, void>;

    /**
     *  @brief Deleted: hint-based emplace is not supported.
     *    A hint names a position, and an open-addressed table decides position by hashing the key,
     *    so there is nothing a caller could usefully hint at. Declaring it deleted rather than
     *    omitting it turns a port from @c std::unordered_map into a signposted error instead of an
     *    unexplained missing member. Use @c emplace, or @c search_to_upsert for finer control.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/emplace_hint
     */
    template <typename... args_types_>
    iterator_t emplace_hint(const_iterator_t, args_types_ &&...) noexcept = delete;

    /**
     *  @brief Main insertion method for maps, finding the optimal location and overwriting it.
     *    More performant than @c insert, as it avoids temporary object construction.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/emplace
     *
     *  @param[in] tags Markers for special acceleration:
     *    @c assume_reserved_t avoids reserving more memory, @c assume_unique_t skips equality
     *    comparisons, @c return_position_t asks for the resulting iterator, @c threadsafe_t allows
     *    concurrent access, @c return_new_size_t exports the updated @c size().
     *  @return Nothing, unless @c return_position_t is provided.
     */
    template <
        typename convertible_key_type_, typename convertible_value_type_, typename... tags_types_,
        typename std::enable_if<can_use_map_emplace<convertible_key_type_, convertible_value_type_>(), int>::type = 0>
    emplace_return<tags_types_...> emplace(convertible_key_type_ &&key, convertible_value_type_ &&value,
                                           tags_types_... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), atomic_slot_ref_t, slot_ref_t>;
        if constexpr (contains_type<threadsafe_t, tags_types_...>()) {
            static_assert(contains_type<assume_reserved_t, tags_types_...>(),
                          "Atomic operations can't cause reallocations!");
            static_assert(!contains_type<return_position_t, tags_types_...>(),
                          "Atomic operations can't return a persistent iterator!");
        }

        if constexpr (!contains_type<assume_reserved_t, tags_types_...>()) reserve_more(1);

        constexpr bool return_k = emplace_returns<tags_types_...>();
        using result_t = std::conditional_t<return_k, insert_result_t, placeholder_t>;

        [[maybe_unused]] result_t result;
        if constexpr (return_k) {
            unsafe_retarget(result.position, slots_count_);
            result.position.slots_remaining = 0;
        }

        auto callback_unused = [&](ref_t &unused_slot) noexcept {
            new (&unused_slot.key_ref()) key_t(std::forward<convertible_key_type_>(key));
            new (&unused_slot.value_ref()) value_storage_t(std::forward<convertible_value_type_>(value));
            if constexpr (return_k) {
                result.position.slot_ = unused_slot.slot_;
                result.inserted = true;
            }
        };

        if constexpr (contains_type<assume_unique_t, tags_types_...>())
            search_to_insert(key, callback_unused, assume_reserved_t {}, tags...);
        else
            search_to_upsert(
                key, callback_unused,
                [&](ref_t &equal_slot) noexcept {
                    equal_slot.value_ref() = std::forward<convertible_value_type_>(value);
                    if constexpr (return_k) result.position.slot_ = equal_slot.slot_;
                },
                assume_reserved_t {}, tags...);

        if constexpr (return_k) {
            result.position.slots_remaining = slots_remaining_after(result.position);
            return result;
        }
    }

    /**
     *  @brief Main insertion method for sets, finding the optimal location and overwriting it.
     *    More performant than @c insert, as it avoids temporary object construction.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_set/emplace
     *
     *  @param[in] tags Markers for special acceleration:
     *    @c assume_reserved_t avoids reserving more memory, @c assume_unique_t skips equality
     *    comparisons, @c return_position_t asks for the resulting iterator, @c threadsafe_t allows
     *    concurrent access, @c return_new_size_t exports the updated @c size().
     *  @return Nothing, unless @c return_position_t is provided.
     */
    template <typename convertible_key_type_, typename... tags_types_,
              typename std::enable_if<can_use_set_emplace<convertible_key_type_>(), int>::type = 0>
    emplace_return<tags_types_...> emplace(convertible_key_type_ &&key, tags_types_... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), atomic_slot_ref_t, slot_ref_t>;
        if constexpr (contains_type<threadsafe_t, tags_types_...>()) {
            static_assert(contains_type<assume_reserved_t, tags_types_...>(),
                          "Atomic operations can't cause reallocations!");
            static_assert(!contains_type<return_position_t, tags_types_...>(),
                          "Atomic operations can't return a persistent iterator!");
        }

        if constexpr (!contains_type<assume_reserved_t, tags_types_...>()) reserve_more(1);

        constexpr bool return_k = emplace_returns<tags_types_...>();
        using result_t = std::conditional_t<return_k, insert_result_t, placeholder_t>;

        [[maybe_unused]] result_t result;
        if constexpr (return_k) {
            unsafe_retarget(result.position, slots_count_);
            result.position.slots_remaining = 0;
        }

        auto callback_unused = [&](ref_t &unused_slot) noexcept {
            new (&unused_slot.key_ref()) key_t(std::forward<convertible_key_type_>(key));
            if constexpr (return_k) {
                result.position.slot_ = unused_slot.slot_;
                result.inserted = true;
            }
        };

        if constexpr (contains_type<assume_unique_t, tags_types_...>())
            search_to_insert(key, callback_unused, assume_reserved_t {}, tags...);
        else if constexpr (return_k)
            search_to_upsert(
                key, callback_unused, [&](ref_t &equal_slot) noexcept { result.position.slot_ = equal_slot.slot_; },
                assume_reserved_t {}, tags...);
        else search_to_upsert(key, callback_unused, no_op_fn_t {}, assume_reserved_t {}, tags...);

        if constexpr (return_k) {
            result.position.slots_remaining = slots_remaining_after(result.position);
            return result;
        }
    }

    /**
     *  @brief Main insertion and upsertion method, compatible with STL code.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/insert
     *
     *  @note With maps this expects a @c mapping, exposing @c key and @c mapped members.
     *  @return Nothing, unless @c return_position_t is provided.
     */
    template <typename temporary_pack_type_, typename... tags_types_,
              typename std::enable_if<!is_iterator<temporary_pack_type_>(), int>::type = 0>
    emplace_return<tags_types_...> insert(temporary_pack_type_ &&pack, tags_types_... tags) noexcept {
        if constexpr (has_values_k)
            return emplace(std::forward<temporary_pack_type_>(pack).key,
                           std::forward<temporary_pack_type_>(pack).mapped, tags...);
        else return emplace(std::forward<temporary_pack_type_>(pack), tags...);
    }

    /**
     *  @brief Inserts elements from the range [begin, end), keeping the first of equivalent keys.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/insert
     */
    template <typename begin_iterator_type_, typename end_iterator_type_, typename... tags_types_,
              typename std::enable_if<is_iterator<begin_iterator_type_>(), int>::type = 0>
    void insert(begin_iterator_type_ begin, end_iterator_type_ end, tags_types_... tags) noexcept {
        for (; begin != end; ++begin) insert(*begin, tags...);
    }

    /**
     *  @brief Files an element under its own key, replacing whatever key compared equal to it.
     *    Unlike @c insert, which keeps the incumbent, this one always leaves the argument in place,
     *    which is what a store layering its own versioning on top of the table needs.
     *
     *  @param[in] element Element to file, moved into the table.
     *  @return @c out_of_memory_heap_k when the table had no room for a new key and could not grow.
     */
    template <typename convertible_element_type_>
    [[nodiscard]] status_t upsert(convertible_element_type_ &&element) noexcept {
        // `reserve` reports whether the layout changed rather than whether it could, so the room is
        // checked instead of assumed: a growth that failed leaves the threshold where it was.
        reserve_more(1);
        if (populated_count_ + deleted_count_ >= growth_threshold_) return status_t {out_of_memory_heap_k};

        if constexpr (has_values_k)
            search_to_upsert(
                element.key,
                [&](slot_ref_t &unused_slot) noexcept {
                    new (&unused_slot.key_ref()) key_t(std::move(element.key));
                    new (&unused_slot.value_ref()) value_storage_t(std::move(element.mapped));
                },
                [&](slot_ref_t &equal_slot) noexcept { equal_slot.value_ref() = std::move(element.mapped); },
                assume_reserved_t {});
        else
            search_to_upsert(
                element,
                [&](slot_ref_t &unused_slot) noexcept { new (&unused_slot.key_ref()) key_t(std::move(element)); },
                [&](slot_ref_t &equal_slot) noexcept { equal_slot.key_ref() = std::move(element); },
                assume_reserved_t {});
        return status_t {success_k};
    }

    /**
     *  @brief An interface similar to @c std::unordered_map::merge, banned in favor of
     *    a manual @c insert(begin, end).
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/merge
     */
    template <typename other_type_>
    void merge(other_type_ &&) = delete;

    /**
     *  @brief Lock-free atomic insertion without reallocation, requires prior @c reserve().
     *  @param[in] key Key to insert or update, forwarded to in-place construction.
     *  @param[in] value Value to insert or update, forwarded to in-place construction.
     *  @note Do not mix with non-atomic operations. If the key exists, updates the value.
     */
    template <
        typename convertible_key_type_, typename convertible_value_type_,
        typename std::enable_if<can_use_map_emplace<convertible_key_type_, convertible_value_type_>(), int>::type = 0>
    void emplace_atomic(convertible_key_type_ &&key, convertible_value_type_ &&value) noexcept {
        search_to_upsert(
            key,
            [&](atomic_slot_ref_t &unused_slot) noexcept {
                new (&unused_slot.key_ref()) key_t(std::forward<convertible_key_type_>(key));
                new (&unused_slot.value_ref()) value_storage_t(std::forward<convertible_value_type_>(value));
            },
            [&](atomic_slot_ref_t &equal_slot) noexcept {
                equal_slot.value_ref() = std::forward<convertible_value_type_>(value);
            },
            threadsafe_t {}, assume_reserved_t {});
    }

    /**
     *  @brief Lock-free atomic value update for an existing key, a no-op when the key is missing.
     *  @param[in] key Key to search for.
     *  @param[in] new_value New value to assign if the key is found.
     *  @return True if key was found and value updated, false otherwise.
     */
    template <typename comparable_key_type_, typename convertible_value_type_>
    bool update_atomic(comparable_key_type_ &&key, convertible_value_type_ &&new_value) noexcept {
        static_assert(has_values_k, "update_atomic() only available for maps, not sets");
        return find_atomic(std::forward<comparable_key_type_>(key), [&](const_atomic_slot_ref_t const &slot) noexcept {
            const_cast<value_storage_t &>(slot.value_ref()) = std::forward<convertible_value_type_>(new_value);
        });
    }

#pragma endregion Insertions

#pragma region Erasures

    /**
     *  @brief Removes the element the reference points to, leaving a tombstone behind.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/erase
     *  @warning The reference must point at a populated slot.
     */
    void erase(slot_ref_t slot) noexcept {
        if constexpr (destruct_keys_k) slot.key_ref().~key_t();
        if constexpr (destruct_vals_k) slot.value_ref().~value_storage_t();
        slot.mark_deleted();
        ++deleted_count_;
        --populated_count_;
    }

    void erase(iterator_t slot) noexcept { erase(static_cast<slot_ref_t>(slot)); }

    void erase(const_iterator_t) = delete;

    /**
     *  @brief Removes a value by key in the most efficient fashion.
     *    If you already have the iterator, use the faster overload that avoids the search entirely.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/erase
     *
     *  @param[in] tags Markers for special acceleration:
     *    @c assume_reserved_t assumes the container isn't empty, @c threadsafe_t allows concurrency.
     *  @return True if the wanted key was found.
     *
     *  @note Erasures never deallocate; memory stays cluttered until the next @c force_resize().
     */
    template <typename comparable_key_type_, typename... tags_types_>
    bool erase(comparable_key_type_ &&key, tags_types_... tags) noexcept {

        bool result = false;
        search_to_find(
            std::forward<comparable_key_type_>(key),
            // This can't call `erase` on the returned references, as it allows even
            // atomic erasures, which must be done within the locked block.
            [&](auto const &slot) noexcept {
                if constexpr (destruct_keys_k) slot.key_ref().~key_t();
                if constexpr (destruct_vals_k) slot.value_ref().~value_storage_t();
                slot.mark_deleted();
                advance(deleted_count_, 1, tags...);
                decrement(populated_count_, 1, tags...);
                result = true;
            },
            tags...);

        return result;
    }

    /**
     *  @brief Lock-free atomic erasure by key, requires prior @c reserve().
     *  @param[in] key Key to erase, hashable and comparable.
     *  @return True if the key was found and erased, false if it didn't exist.
     *
     *  @note The slot is marked as deleted (tombstone) rather than freed, preserving probe sequences.
     */
    template <typename comparable_key_type_>
    bool erase_atomic(comparable_key_type_ &&key) noexcept {
        return erase(std::forward<comparable_key_type_>(key), threadsafe_t {}, assume_reserved_t {});
    }

#pragma endregion Erasures

#pragma region Memory Management

    /**
     *  @brief Total memory needed for the three regions of a table this size.
     *  @param[in] slots_count Total number of slots, a power of two and a multiple of 32.
     *  @return Total bytes needed for the single allocation containing all three regions.
     */
    static constexpr std::size_t memory_usage(hash_slots_count_t slots_count) noexcept {
        return (slots_count.raw / hash_bucket_capacity_k) * bytes_in_bucket_k;
    }

    /**
     *  @brief Grows the table to fit that many elements, if the current capacity is short.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/reserve
     *  @return True when the underlying capacity has changed, invalidating older iterators.
     */
    bool reserve(offset_t planned_elements) noexcept {
        if (planned_elements <= growth_threshold_) return false;
        force_resize(hash_slots_count_t {planned_elements});
        return true;
    }

    /**
     *  @brief The recommended entry point for bulk insertions, sizing for the elements already present.
     *  @return True when the underlying capacity has changed, invalidating older iterators.
     */
    bool reserve_more(offset_t new_elements) noexcept {
        return reserve(new_elements + populated_count_ + deleted_count_);
    }

    /**
     *  @brief Exports the data into a fresh table of the requested size and swaps the contents.
     *    Also used to clean the tombstones left behind by deletions.
     *  @warning The new capacity can't be less than @c size().
     */
    void force_resize(hash_slots_count_t slots_count) noexcept {
        auto resized = move_to_new(slots_count);
        swap(resized);
    }

    /**
     *  @brief A cheap copy-less swap mechanism used in move-constructors.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/swap
     */
    void swap(basic_hash_table &other) noexcept {
        std::swap(memory_, other.memory_);
        std::swap(keys_, other.keys_);
        std::swap(values_, other.values_);
        std::swap(headers_, other.headers_);
        std::swap(slots_count_, other.slots_count_);
        std::swap(growth_threshold_, other.growth_threshold_);
        std::swap(populated_count_, other.populated_count_);
        std::swap(deleted_count_, other.deleted_count_);

        if constexpr (!hasher_is_empty_k) std::swap(hasher_, other.hasher_);
        if constexpr (!equals_is_empty_k) std::swap(equals_, other.equals_);
        if constexpr (!allocator_is_empty_k) std::swap(allocator_, other.allocator_);
    }

    /**
     *  @brief Erases all elements, keeping the memory. For deallocation call @c shrink_to_fit().
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/clear
     */
    void clear() noexcept {
        if (memory_) clear(assume_reserved_t {});
    }

    /**
     *  @brief Erases all elements, keeping the memory. For deallocation call @c shrink_to_fit().
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/clear
     */
    void clear(assume_reserved_t) noexcept {
        if constexpr (destruct_keys_k || destruct_vals_k)
            for (auto it = begin(); it != end(); it.advance()) {
                if constexpr (destruct_keys_k) it.key_ref().~key_t();
                if constexpr (destruct_vals_k) it.value_ref().~value_storage_t();
            }

        std::memset(memory_, 0, size_bytes());
        deleted_count_ = 0;
        populated_count_ = 0;
    }

    /**
     *  @brief Brings the memory consumption of the container to the minimum.
     *  @see https://en.cppreference.com/w/cpp/container/vector/shrink_to_fit
     */
    void shrink_to_fit() noexcept {
        hash_slots_count_t const new_capacity {size()};
        if (new_capacity.raw == slots_count_) return;
        if (size()) force_resize(new_capacity);
        else deallocate();
    }

    /**
     *  @brief Deallocates all the memory used by the container.
     *  @warning Call @c clear() first to destruct all the non-trivial elements.
     */
    void deallocate() noexcept {
        if (memory_) deallocate(assume_reserved_t {});
    }

    /**
     *  @brief Deallocates all the memory used by the container.
     *  @warning Call @c clear() first to destruct all the non-trivial elements.
     */
    void deallocate(assume_reserved_t) noexcept {
        allocator_.deallocate(memory_, size_bytes());
        unsafe_reset();
    }

    /** @brief Drops every pointer and counter without touching the memory they described. */
    void unsafe_reset() noexcept {
        memory_ = nullptr;
        keys_ = nullptr;
        values_ = nullptr;
        headers_ = nullptr;
        slots_count_ = 0;
        growth_threshold_ = 0;
        populated_count_ = 0;
        deleted_count_ = 0;
    }

    /**
     *  @brief Sets the number of buckets to the desired value, rehashing in the process.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/rehash
     */
    void rehash(offset_t count_buckets) noexcept {
        force_resize(hash_slots_count_t::from_slots(count_buckets * hash_bucket_capacity_k));
    }

#pragma endregion Memory Management

#pragma region Copies and Moves

    /**
     *  @brief Creates a new buffer of the requested capacity and rehashes the present data into it.
     *    Move-constructors are used, avoiding copies, and this table is left deallocated.
     *  @warning The new capacity can't be less than the current @c size().
     */
    basic_hash_table move_to_new(hash_slots_count_t slots_count) noexcept {

        basic_hash_table target(slots_count, hasher_, equals_, allocator_);
        assert((!slots_count.raw || target.memory_) && "Allocation has failed!");
        assert(target.capacity() >= size() && "Not enough space in the new Hash-Table!");

        for_each([&target](slot_ref_t const &source_slot) noexcept {
            if constexpr (has_values_k)
                target.emplace(std::move(source_slot.key_ref()), std::move(source_slot.value_ref()),
                               assume_reserved_t {}, assume_unique_t {});
            else target.emplace(std::move(source_slot.key_ref()), assume_reserved_t {}, assume_unique_t {});
            if constexpr (destruct_keys_k) source_slot.key_ref().~key_t();
            if constexpr (destruct_vals_k) source_slot.value_ref().~value_storage_t();
        });

        assert(target.populated_count_ == populated_count_ && "Element counts must match!");
        deallocate();
        return target;
    }

    /** @brief Duplicates the table, reusing the exact bucket layout when the elements are trivial. */
    basic_hash_table copy_to_new() const noexcept {

        basic_hash_table target(slots_count(), hasher_, equals_, allocator_);
        if (!target.memory_) return target;

        if constexpr (layout_t::will_memcpy_keys() && (!has_values_k || layout_t::will_memcpy_vals())) {
            std::memcpy(target.memory_, memory_, size_bytes());
            target.populated_count_ = populated_count_;
            target.deleted_count_ = deleted_count_;
        }
        else
            for_each([&target](const_slot_ref_t const &source_slot) noexcept {
                if constexpr (has_values_k)
                    target.emplace(source_slot.key(), source_slot.value(), assume_reserved_t {}, assume_unique_t {});
                else target.emplace(source_slot.key(), assume_reserved_t {}, assume_unique_t {});
            });

        return target;
    }

    /** @brief Duplicates the table into a differently sized one, rehashing every element. */
    basic_hash_table copy_to_new(hash_slots_count_t slots_count) const noexcept {
        basic_hash_table target(slots_count, hasher_, equals_, allocator_);
        if (!target.memory_) return target;
        for_each([&target](const_slot_ref_t const &source_slot) noexcept {
            if constexpr (has_values_k)
                target.emplace(source_slot.key(), source_slot.value(), assume_reserved_t {}, assume_unique_t {});
            else target.emplace(source_slot.key(), assume_reserved_t {}, assume_unique_t {});
        });
        return target;
    }

#pragma endregion Copies and Moves
};

#pragma region Aliases

template <typename key_type_, typename value_type_, typename hasher_type_ = lazy_std_hash_t,
          typename equals_type_ = std::equal_to<>, typename allocator_type_ = std::allocator<std::byte>>
using hash_map = basic_hash_table<mapping<key_type_, value_type_>, hasher_type_, equals_type_, allocator_type_>;

template <typename key_type_, typename hasher_type_ = lazy_std_hash_t, typename equals_type_ = std::equal_to<>,
          typename allocator_type_ = std::allocator<std::byte>>
using hash_set = basic_hash_table<key_type_, hasher_type_, equals_type_, allocator_type_>;

static_assert(sizeof(hash_set<int>) >= 3 * sizeof(void *), "Hash-Table is too small!");

#pragma endregion Aliases

} // namespace ashvardanian::smashtable
