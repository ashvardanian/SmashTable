/**
 *  @brief  Lock-free concurrent hash-table optimized for CPU and GPU workloads.
 *
 *  @file   basic_hash_table2.hpp
 *  @author Ash Vardanian
 *
 *  @section Overview
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
 *  @section Core Design Goals
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
 *  @b Performance: Optimized for bulk operations and set operations common in analytical and
 *  database workloads. Methods like @c intersection_size() and @c merge_size() enable efficient
 *  multi-table operations. Explicit API methods ( @c emplace_reserved(), @c emplace_atomic()) provide
 *  zero-overhead access to specialized code paths without runtime branching.
 *
 *  @section Key Design Decisions
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
 *  Single allocation with three cache-line-aligned regions organized for efficient compaction:
 *  @code
 *  // Layout: [keys_region | values_region | headers_region]
 *  std::byte* memory_ = allocate(keys_bytes + values_bytes + header_bytes);
 *
 *  key_t* keys_ = (key_t*)memory_;                           // Direct indexing: keys_[slot]
 *  value_t* values_ = (value_t*)(memory_ + keys_bytes);     // Direct indexing: values_[slot]
 *  bht_bucket_head_t* headers_ = (bht_bucket_head_t*)(memory_ + keys_bytes + values_bytes);
 *                                                            // Bucket indexing: headers_[slot/32]
 *
 *  // Each region padded to 64-byte cache lines to prevent false sharing
 *  keys_bytes = roundup_to_cache_line(slots * sizeof(key_t));
 *  values_bytes = roundup_to_cache_line(slots * sizeof(value_t));
 *  header_bytes = (slots / 32) * sizeof(bht_bucket_head_t);
 *  @endcode
 *
 *  This ordering enables @b compaction: when squeezing the hash table into a dense array, the
 *  keys and values are already contiguous at the start of @c memory_. The headers at the end can be
 *  discarded, and the buffer directly reused for storage without copying.
 *
 *  @par SIMD Acceleration
 *  Bulk operations ( @c intersection_size, batch insertions) can benefit from vectorized hashing:
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
 *  // → 4 adjacent addresses = single coalesced transaction
 *  probe_idx = (hash + group.thread_rank()) % capacity;
 *  @endcode
 *
 *  This strategy, used by NVIDIA's cuCollections, shows 13% faster inserts and 40% faster
 *  lookups under high load factors (≥60%) compared to one-thread-per-key approaches. The larger
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
#include <cassert>     // `assert`
#include <algorithm>   // `std::max`
#include <bit>         // `std::popcount`, `std::countl_zero`
#include <cstdint>     // `std::uint32_t`, `std::uint64_t`
#include <cstring>     // `std::memcpy`
#include <iterator>    // `std::input_iterator_tag`
#include <memory>      // `std::allocator`, `std::assume_aligned`
#include <atomic>      // `std::atomic_ref`
#include <optional>    // `std::optional`
#include <random>      // `std::uniform_int_distribution`
#include <type_traits> // `std::is_same`, `std::enable_if`
#include <utility>     // `std::exchange`, `std::pair`

#include "shared.hpp"

namespace ashvardanian::smashtable {

/** @brief Platform cache line size in bytes, typically 64 on modern CPUs. */
inline constexpr std::size_t cache_line_bytes_k = 64;

/** @brief Type trait to check if a type is an iterator. */
template <typename type_>
constexpr bool is_iterator() {
    return std::is_base_of<std::input_iterator_tag, typename std::iterator_traits<type_>::iterator_category>::value ||
           std::is_same<typename std::iterator_traits<type_>::iterator_category, std::output_iterator_tag>::value;
}

/** @brief Function object that extracts the second element of a mapping. */
struct take_second_t {
    template <typename key_type_, typename value_type_>
    decltype(auto) operator()(mapping<key_type_, value_type_> const &p) const noexcept {
        return p.value;
    }
    template <typename key_type_, typename value_type_>
    decltype(auto) operator()(mapping<key_type_, value_type_> &p) const noexcept {
        return p.value;
    }
};

inline static constexpr std::size_t bht_bucket_capacity_k = 32;

/**
 *  @brief Bucket header containing metadata for @c bht_bucket_capacity_k slots.
 *    Stores two parallel 32-bit bitmasks (population and deletion states) that combine to encode
 *    four possible states per slot using 2 bits each.
 *
 *  @section Slot State Encoding
 *  Each slot's state is determined by corresponding bits in both masks:
 *  - @c 00 (populations=0, deletions=0): Free slot, never used or fully freed
 *  - @c 01 (populations=0, deletions=1): Deleted slot, tombstone from lazy deletion
 *  - @c 10 (populations=1, deletions=0): Populated slot with valid key-value pair
 *  - @c 11 (populations=1, deletions=1): Locked slot, temporarily held for atomic operations
 *
 *  The @c u64 field enables atomic operations on the entire header (2 × 32 bits = 64 bits).
 */
union bht_bucket_head_t {

    struct {
        std::uint32_t populations; /**< Bitmask where set bits indicate populated/locked slots. */
        std::uint32_t deletions;   /**< Bitmask where set bits indicate deleted/locked slots. */
    } u32s;                        /**< Separate 32-bit views for populations and deletions. */
    std::uint64_t u64;             /**< Combined 64-bit view for atomic operations. */
};

static_assert(sizeof(bht_bucket_head_t) == sizeof(std::uint64_t));

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
 *  @note The minimum slot count equals @c bht_bucket_capacity_k to ensure at least one full bucket.
 */
struct bht_slots_count_t {
    std::size_t raw = 0;

    operator std::size_t() const noexcept { return raw; }
    explicit constexpr bht_slots_count_t() noexcept {}
    explicit constexpr bht_slots_count_t(std::size_t elems) noexcept {
        if (elems == 0) return;
        std::size_t needed_slots = (elems * 4ul) / 3ul;
        // We calculate the bucket index with AND masks, so it must be a power of two.
        raw = roundup_to_pow2(needed_slots);
        raw = std::max(raw, bht_bucket_capacity_k);
    }
};

/**
 *  @brief Smart reference to a hash table slot with metadata accessors.
 *  @tparam key_type_ Key type stored in the hash table, can't be an mapping.
 */
template <typename key_type_>
struct bht_slot_ref {

    static_assert(!is_mapping<key_type_>(), "for associations use bht_slot_ref.");
    using key_t = key_type_;

  protected:
    key_t *key_ptr_;                 /**< Pointer to key entry. */
    bht_bucket_head_t *header_ptr_;  /**< Pointer to the entire buckets header. */
    std::uint32_t bit_mask_;         /**< Mask for accessing relevant parts of the header. */
    std::uint32_t offset_in_bucket_; /**< Offset of this slot within its bucket (0-31). */

    bht_bucket_head_t header_mask_() const noexcept {
        bht_bucket_head_t mask;
        mask.u32s.populations = bit_mask_;
        mask.u32s.deletions = bit_mask_;
        return mask;
    }

  public:
    constexpr void lock() noexcept {}
    constexpr void unlock() noexcept {}
    constexpr bool try_lock() noexcept { return true; }

    bht_bucket_head_t &header_ref() noexcept { return *header_ptr_; }

    bool is_freed() noexcept { return header_ref().u64 & header_mask_().u64 == 0; }

    // TODO: These may be wrong, as they are not checking the other 32-bit lane
    bool is_populated() noexcept; // { return header_ref().populations & bit_mask_; }
    bool is_deleted() noexcept;   // { return header_ref().deletions & bit_mask_; }

    void mark_freed() noexcept {
        header_ref().populations &= ~bit_mask_;
        header_ref().deletions &= ~bit_mask_;
    }
    void mark_populated() noexcept {
        header_ref().populations |= bit_mask_;
        header_ref().deletions &= ~bit_mask_;
    }
    void mark_deleted() noexcept {
        header_ref().populations &= ~bit_mask_;
        header_ref().deletions |= bit_mask_;
    }

    key_t &key_ref() const noexcept { return *key_ptr_; }
    key_t const &key() const noexcept { return *key_ptr_; }

    key_t &operator*() noexcept { return key(); }
    key_t const &operator*() const noexcept { return key(); }
};

/**
 *  @brief Smart reference to a hash table slot with metadata accessors.
 *  @tparam key_type_ Key type stored in the hash table, can't be an mapping.
 *  @tparam value_type_ Value type stored in the hash table, or @c void for hash sets.
 */
template <typename key_type_, typename value_type_>
struct bht_slot_ref<mapping<key_type_, value_type_>> {

    using key_t = key_type_;
    using value_t = value_type_;

  protected:
    key_t *key_ptr_;                 /**< Pointer to key entry. */
    value_t *value_ptr_;             /**< Pointer to value entry. */
    bht_bucket_head_t *header_ptr_;  /**< Pointer to the entire buckets header. */
    std::uint32_t bit_mask_;         /**< Mask for accessing relevant parts of the header. */
    std::uint32_t offset_in_bucket_; /**< Offset of this slot within its bucket (0-31). */
  public:
    constexpr void lock() noexcept {}
    constexpr void unlock() noexcept {}
    constexpr bool try_lock() noexcept { return true; }

    bht_bucket_head_t &header_ref() noexcept { return *header_ptr_; }
    bool is_freed() noexcept { return header_ref().u64 & header_mask_().u64 == 0; }

    // TODO: These may be wrong, as they are not checking the other 32-bit lane
    bool is_populated() noexcept; // { return header_ref().populations & bit_mask_; }
    bool is_deleted() noexcept;   // { return header_ref().deletions & bit_mask_; }

    void mark_freed() noexcept {
        header_ref().populations &= ~bit_mask_;
        header_ref().deletions &= ~bit_mask_;
    }
    void mark_populated() noexcept {
        header_ref().populations |= bit_mask_;
        header_ref().deletions &= ~bit_mask_;
    }
    void mark_deleted() noexcept {
        header_ref().populations &= ~bit_mask_;
        header_ref().deletions |= bit_mask_;
    }

    key_t &key_ref() const noexcept { return *key_ptr_; }
    key_t const &key() const noexcept { return *key_ptr_; }

    value_t &value_ref() const noexcept { return *value_ptr_; }
    value_t const &value() const noexcept { return *value_ptr_; }

    mapping<key_t const &, value_t &> operator*() noexcept { return {key(), value_ref()}; }
    mapping<key_t const &, value_t const &> operator*() const noexcept { return {key(), value()}; }
};

/**
 *  @brief Thread-safe smart reference to a hash table slot with metadata accessors.
 *  @tparam element_type_ Element type stored in the hash table, either key or mapping.
 *
 *  Ideally, we would want to avoid Compare-And-Swap @b (CAS) loops for locking individual slots.
 *  On the locking path, we can use a @c fetch_or atomic operation to set both bits (populated + deleted)
 *  simultaneously and transition to the locked state. If the previous value indicates the slot was
 *  already locked, we retry until we acquire the lock.
 *
 *  The intuition of using the @c fetch_and for the inverse operation, however, is wrong. When unlocking,
 *  we need to restore the previous state (populated/deleted) of the slot or transition to the new one,
 *  depending on the @c mark_populated()/mark_deleted()/mark_freed() calls made while the slot was locked.
 *
 *  The bits outside the active ones in each 32-bit word shouldn't be changed. The active ones may have
 *  to be flipped. Assuming the value of the relevant bits couldn't have changed, we can use @c fetch_xor
 *  to control the result in the same lock-free manner. @b XOR-is-all-you-need!
 *
 *  This doesn't resolve @b false-sharing issues native to such a densely packed design, but still results
 *  in very low contention if the duration of atomic operations under the lock is comparable to CPU's
 *  memory latency.
 */
template <typename element_type_>
class bht_atomic_slot_ref : public bht_element_ref<element_type_> {

    using base_t = bht_element_ref<element_type_>;
    using base_t::bit_mask_;

    bht_bucket_head_t mutable future_header_ {0ul};

  public:
    bht_bucket_head_t &header_ref() const noexcept { return future_header_; }
    bool is_freed() noexcept { return header_ref().u64 & header_mask_().u64 == 0; }

    // TODO: These may be wrong, as they are not checking the other 32-bit lane
    bool is_populated() noexcept; // { return header_ref().populations & bit_mask_; }
    bool is_deleted() noexcept;   // { return header_ref().deletions & bit_mask_; }

    void mark_freed() noexcept {
        header_ref().populations &= ~bit_mask_;
        header_ref().deletions &= ~bit_mask_;
    }
    void mark_populated() noexcept {
        header_ref().populations |= bit_mask_;
        header_ref().deletions &= ~bit_mask_;
    }
    void mark_deleted() noexcept {
        header_ref().populations &= ~bit_mask_;
        header_ref().deletions |= bit_mask_;
    }

    void lock() noexcept {
        bht_bucket_head_t header_mask = header_mask_();
        // Lock the object and make sure it wasn't locked before us:
    relock:
        std::atomic_ref<std::uint64_t> atomic_header(base_t::header_ref().u64);
        future_header_.u64 = atomic_header.fetch_or(header_mask.u64, std::memory_order_acquire) //
                             & header_mask.u64;
        if (future_header_.u64 == header_mask.u64) goto relock;
    }

    void unlock() noexcept {
        // The bits we care about are now set to 11 (locked).
        // After this procedure they must be set to either 00, 01, or 10, depending on the "future header".
        bht_bucket_head_t header_mask = header_mask_();
        assert(std::popcount(header_mask.u64) == 2 && "only 2 bits must be set in the mask.");

        // Let's compute the "differences" between the old locked state and the new desired state:
        // - for "freed" slots 00: 11 ^ 00 = 11
        // - for "deleted" slots 01: 11 ^ 01 = 10
        // - for "populated" slots 10: 11 ^ 10 = 01
        std::uint64_t header_differences = header_mask.u64 ^ future_header_.u64;
        assert(std::popcount(header_differences) >= 1 && std::popcount(header_differences) <= 2 &&
               "only 1 or 2 bits can form the difference.");

        // Now if we only XOR the differences:
        // - going from "locked" state to "freed" state: 11 ^ 11 = 00
        // - going from "locked" state to "deleted" state: 11 ^ 10 = 01
        // - going from "locked" state to "populated" state: 11 ^ 01 = 10
        std::atomic_ref<std::uint64_t> atomic_header(base_t::header_ref().u64);
        atomic_header.fetch_xor(header_differences, std::memory_order_release);
    }

    bool try_lock() noexcept {
        bht_bucket_head_t header_mask = header_mask_();
        // Lock the object and make sure it wasn't locked before us:
        std::atomic_ref<std::uint64_t> atomic_header(base_t::header_ref().u64);
        future_header_.u64 = atomic_header.fetch_or(header_mask.u64, std::memory_order_acquire) //
                             & header_mask.u64;
        return future_header_.u64 != header_mask.u64;
    }
};

/**
 *  @brief Metadata extraction template that derives type information for hash table elements.
 *    Computes element types, sizes, and alignment requirements from key/value/hasher template parameters.
 *    Uses the hasher's return type as @c offset_t for slot indexing (typically @c std::size_t).
 *
 *  @tparam key_type_ Key type stored in the hash table, references are stripped to value types.
 *  @tparam value_type_ Value type stored in the hash table, or @c void for hash sets.
 *  @tparam hasher_type_ Hash function object, must be callable with @c key_type_.
 *
 *  @note Provides three element views:
 *    - @c element_t: Mutable reference pair @code mapping<key const&, value&> @endcode for iteration
 *    - @c element_const_t: Immutable reference pair @code mapping<key const&, value const&> @endcode for lookups
 *    - @c element_copy_t: Owned value pair @code mapping<key, value> @endcode for extraction operations
 *
 *  @see https://en.cppreference.com/w/cpp/utility/hash
 */
template <typename element_type_, typename hasher_type_>
struct bht_layout_for {
    using key_t = std::remove_reference_t<element_type_>;
    using value_t = void;
    using element_t = key_t;
    using element_const_t = key_t;
    using element_copy_t = key_t;
    using hasher_t = hasher_type_;
    using offset_t = decltype(hasher_type_ {}(std::declval<key_t>()));
    using ref_t = bht_slot_ref<key_t>;

    inline static constexpr std::size_t bytes_for_keys_k =
        roundup_to_multiple<std::size_t, cache_line_bytes_k>(sizeof(key_t) * bht_bucket_capacity_k);
    inline static constexpr std::size_t bytes_for_values_k = 0;

    template <typename value_transform_>
    using transformed_value = void;

    static constexpr bool will_memcpy_keys() { return std::is_trivially_copy_constructible<key_t>(); }
    static constexpr bool will_memcpy_vals() { return false; }
};

template <typename key_type_, typename value_type_, typename hasher_type_>
struct bht_layout_for<mapping<key_type_, value_type_>, hasher_type_> {

    using key_t = std::remove_reference_t<key_type_>;
    using value_t = std::remove_reference_t<value_type_>;
    using element_t = mapping<key_t const &, value_t &>;
    using element_const_t = mapping<key_t const &, value_t const &>;
    using element_copy_t = mapping<key_t, value_t>;
    using hasher_t = hasher_type_;
    using offset_t = decltype(hasher_type_ {}(std::declval<key_t>()));
    using ref_t = bht_slot_ref<key_t, value_t>;

    inline static constexpr std::size_t bytes_for_keys_k =
        roundup_to_multiple<std::size_t, cache_line_bytes_k>(sizeof(key_t) * bht_bucket_capacity_k);
    inline static constexpr std::size_t bytes_for_values_k =
        roundup_to_multiple<std::size_t, cache_line_bytes_k>(sizeof(value_t) * bht_bucket_capacity_k);

    template <typename value_transform_>
    using transformed_value = decltype(value_transform_ {}(*reinterpret_cast<value_t const *>(NULL)));

    static constexpr bool will_memcpy_keys() { return std::is_trivially_copy_constructible<key_t>(); }
    static constexpr bool will_memcpy_vals() { return std::is_trivially_copy_constructible<value_t>(); }
};

static_assert(bht_layout_for<int, std::hash<int>>::will_memcpy_keys());
static_assert(bht_layout_for<mapping<int, int>, std::hash<int>>::will_memcpy_keys());

/**
 *  @brief Iterates over all populated slots in a bucket using optimized bit-scanning.
 *    More efficient than sequential iteration as it skips empty/deleted slots by analyzing
 *    the population bitmap using @c std::popcount and @c std::countl_zero intrinsics.
 *
 *  @see https://en.cppreference.com/w/cpp/algorithm/for_each
 *
 *  @param[in,out] addr Reference to bucket element. The @c slot_ field is modified
 *    during iteration to point to each populated slot sequentially.
 *  @param[in] callback Functor invoked for each populated slot, receiving @c bht_slot_ref.
 */
template <typename key_type_, typename value_type_, typename hasher_type_, typename callback_type_>
void bht_for_each_in_bucket_(bht_slot_ref<key_type_, value_type_, hasher_type_> &addr,
                             callback_type_ &&callback) noexcept {

    using offset_t = typename bht_slot_ref<key_type_, value_type_, hasher_type_>::offset_t;
    offset_t bucket_start = (addr.slot_ / bht_bucket_capacity_k) * bht_bucket_capacity_k;

    bht_bucket_mask_t populations_left = addr.header_ref().populations;
    int count_populated = std::popcount(populations_left);
    while (count_populated) {
        offset_t idx_in_bucket = std::countl_zero(populations_left);
        addr.slot_ = bucket_start + idx_in_bucket;
        callback(addr);
        populations_left &= ~addr.mask_in_bucket();
        --count_populated;
    }
}

/**
 *  @brief Iterates through all slots in a bucket, including empty and deleted ones.
 *    Unlike @c bht_for_each_in_bucket_, this performs sequential iteration without
 *    checking the population bitmap. Useful for low-level operations like bucket copying.
 *
 *  @see https://en.cppreference.com/w/cpp/algorithm/for_each
 *
 *  @param[in,out] addr Reference to bucket element. The @c slot_ field is modified
 *    to iterate through all 32 slots in the bucket.
 *  @param[in] callback Functor invoked for every slot, receiving @c bht_slot_ref.
 */
template <typename key_type_, typename value_type_, typename hasher_type_, typename callback_type_>
void bht_for_slots_in_bucket(bht_slot_ref<key_type_, value_type_, hasher_type_> &addr,
                             callback_type_ &&callback) noexcept {

    using offset_t = typename bht_slot_ref<key_type_, value_type_, hasher_type_>::offset_t;
    offset_t bucket_start = (addr.slot_ / bht_bucket_capacity_k) * bht_bucket_capacity_k;
    offset_t bucket_end = bucket_start + bht_bucket_capacity_k;

    for (addr.slot_ = bucket_start; addr.slot_ != bucket_end; ++addr.slot_) { callback(addr); }
}

/**
 *  @brief Searches for an element within a bucket using optimized bit-scanning.
 *    Stops iteration early when the predicate returns @c true. More efficient than
 *    @c bht_for_each_in_bucket_ for lookups as it can short-circuit.
 *
 *  @see https://en.cppreference.com/w/cpp/algorithm/find
 *
 *  @param[in,out] addr Reference to bucket element. The @c slot_ field is modified
 *    during iteration to point to each populated slot until a match is found.
 *  @param[in] predicate Functor invoked for each populated slot. Must return @c true if the
 *    element matches (terminating the search) or @c false to continue.
 *  @return True if a matching element was found, false otherwise.
 */
template <typename key_type_, typename value_type_, typename hasher_type_, typename predicate_type_>
bool bht_find_in_bucket_(bht_slot_ref<key_type_, value_type_, hasher_type_> &addr,
                         predicate_type_ &&predicate) noexcept {

    using offset_t = typename bht_slot_ref<key_type_, value_type_, hasher_type_>::offset_t;
    offset_t bucket_start = (addr.slot_ / bht_bucket_capacity_k) * bht_bucket_capacity_k;

    bht_bucket_mask_t populations_left = addr.header_ref().populations;
    int count_populated = std::popcount(populations_left);
    bool found_match = false;
    while ((count_populated != 0) & !found_match) {
        offset_t idx_in_bucket = std::countl_zero(populations_left);
        addr.slot_ = bucket_start + idx_in_bucket;
        found_match = predicate(addr);
        populations_left &= ~addr.mask_in_bucket();
        --count_populated;
    }
    return found_match;
}

/**
 *  @brief Maps elements from a source bucket to a target bucket, applying a transformation to values.
 *    Equivalent to bucket-level @c std::transform. Preserves the exact bucket layout including deleted
 *    slots to accelerate copy operations. Automatically uses @c std::memcpy for trivially copyable types.
 *
 *  @tparam key_type_ Source key type (const-qualified for immutable source).
 *  @tparam value_type_ Source value type (const-qualified for immutable source).
 *  @tparam transformed_value_type_ Target value type after transformation.
 *  @tparam hasher_type_ Hash function type.
 *  @tparam value_transform_ Transformation function type, defaults to @c identity_fn_t.
 *
 *  @param[in] src_addr Reference to source bucket containing elements to transform.
 *  @param[out] tgt_addr Reference to target bucket where transformed elements are written.
 *  @param[in] value_transform Functor applied to each value during transformation. Common patterns:
 *    @c identity_fn_t for copy-construction (uses @c std::memcpy when elements are trivially
 *    copy-constructible), or @c null_operator_t for exporting Hash-Maps to Hash-Sets by discarding values.
 *
 *  @note Deleted slots are not reused and entries are not rehashed. This preserves the source bucket's
 *    exact layout, accelerating copy-construction at the cost of potential inefficiency if the source
 *    bucket had many deleted entries.
 *
 *  @see https://en.cppreference.com/w/cpp/algorithm/transform
 */
template <typename key_type_, typename value_type_, typename transformed_value_type_, typename hasher_type_,
          typename value_transform_ = identity_fn_t>
void bht_transform_bucket(bht_slot_ref<key_type_ const, value_type_ const, hasher_type_> &src_addr,
                          bht_slot_ref<key_type_, transformed_value_type_, hasher_type_> &tgt_addr,
                          value_transform_ &&value_transform = {}) noexcept {

    // Check what can be done with a simple memcpy.
    // using new_val_t = typename layout_t::transformed_value<value_transform_>;
    using layout_t = bht_layout_for<key_type_ const, value_type_ const, hasher_type_>;
    constexpr bool outputs_vals_k = !std::is_same<transformed_value_type_, void>();
    constexpr bool changes_vals_k = std::is_same<value_transform_, identity_fn_t>();
    constexpr bool memcpy_keys_k = layout_t::will_memcpy_keys();
    constexpr bool memcpy_vals_k = !changes_vals_k && layout_t::will_memcpy_vals() && outputs_vals_k;
    static_assert(outputs_vals_k <= layout_t::has_values_k, "It's hard to output something that doesn't exist!");

    using offset_t = typename bht_slot_ref<key_type_, value_type_, hasher_type_>::offset_t;
    offset_t src_bucket_start = (src_addr.slot_ / bht_bucket_capacity_k) * bht_bucket_capacity_k;
    offset_t tgt_bucket_start = (tgt_addr.slot_ / bht_bucket_capacity_k) * bht_bucket_capacity_k;

    // Copy bucket header
    *tgt_addr.headers_ = *src_addr.headers_;

    // Trivially copy keys if possible
    constexpr std::size_t bytes_in_buckets_keys_k = bht_bucket_capacity_k * sizeof(key_type_);
    if constexpr (memcpy_keys_k)
        std::memcpy(&tgt_addr.keys_[tgt_bucket_start], &src_addr.keys_[src_bucket_start], bytes_in_buckets_keys_k);

    // Trivially copy values if possible
    constexpr std::size_t bytes_in_buckets_vals_k = bht_bucket_capacity_k * sizeof(value_type_);
    if constexpr (memcpy_vals_k)
        std::memcpy(&tgt_addr.values_[tgt_bucket_start], &src_addr.values_[src_bucket_start], bytes_in_buckets_vals_k);

    // Manually copy the rest
    if constexpr (!memcpy_keys_k || (!memcpy_vals_k && outputs_vals_k)) {
        bht_for_each_in_bucket_(src_addr, [&tgt_addr, &value_transform](auto const &src_addr) {
            tgt_addr.slot_ = src_addr.slot_;
            if constexpr (!memcpy_keys_k) new (&tgt_addr.key_ref()) key_type_(src_addr.key());
            if constexpr (!memcpy_vals_k && outputs_vals_k) {
                if constexpr (changes_vals_k)
                    new (&tgt_addr.value_ref()) transformed_value_type_(value_transform(src_addr.value()));
                else new (&tgt_addr.value_ref()) transformed_value_type_(src_addr.value());
            }
        });
    }
}

/**
 *  @brief Forward iterator for hash tables using simplified slot-based addressing.
 *    Inherits 4 pointers from @c bht_slot_ref plus @c slots_remaining counter.
 *    Implements @c std::forward_iterator_tag for sequential traversal of populated slots.
 *
 *  @section Memory Layout
 *  Inherits from @c bht_slot_ref (32 bytes on 64-bit):
 *  - @c keys_: Pointer to keys array (8 bytes)
 *  - @c values_: Pointer to values array (8 bytes)
 *  - @c headers_: Pointer to headers array (8 bytes)
 *  - @c slot_: Current absolute slot index (8 bytes)
 *  Plus iterator-specific:
 *  - @c slots_remaining: Count of unvisited slots (8 bytes)
 *
 *  When @c slots_remaining reaches zero, the iterator is at end.
 *
 *  @note This iterator is not thread-safe. For concurrent iteration, use @c for_each with
 *    appropriate synchronization or operate on a snapshot of the table.
 */
template <typename key_type_, typename value_type_, typename hasher_type_>
struct bht_iterator_gt : public bht_slot_ref<key_type_, value_type_, hasher_type_> {

    using base_t = bht_slot_ref<key_type_, value_type_, hasher_type_>;
    using element_t = typename base_t::element_t;
    using offset_t = typename base_t::offset_t;
    using base_t::keys_;
    using base_t::values_;
    using base_t::headers_;
    using base_t::slot_;
    using base_t::header_ref;
    using base_t::mask_in_bucket;

    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = element_t;
    using reference = element_t;
    using pointer = void;

    offset_t slots_remaining; /**< Number of slots left to visit (0 = end). */

    [[nodiscard]] bool operator==(end_sentinel_t) const noexcept { return is_end(); }
    [[nodiscard]] bool operator!=(end_sentinel_t) const noexcept { return isnt_end(); }
    [[nodiscard]] bool isnt_end() const noexcept { return slots_remaining != 0; }
    [[nodiscard]] bool is_end() const noexcept { return slots_remaining == 0; }

    bool operator==(bht_iterator_gt const &o) const noexcept { return (keys_ == o.keys_) & (slot_ == o.slot_); }
    bool operator!=(bht_iterator_gt const &o) const noexcept { return (keys_ != o.keys_) | (slot_ != o.slot_); }

    bht_iterator_gt &operator++() noexcept {
        advance();
        return *this;
    }

    bht_iterator_gt operator++(int) noexcept {
        bht_iterator_gt copy = *this;
        advance();
        return copy;
    }

    /**
     *  @brief Advances iterator to next populated slot.
     *    Simplified from bucket arithmetic to direct slot increment.
     *    Must not be called when @c is_end() is true.
     */
    void advance() noexcept {
        assert(!is_end() && "Going out of range!");
        --slots_remaining;
        ++slot_;
        skip_non_populated();
    }

    /**
     *  @brief Skips over empty and deleted slots until finding a populated one or reaching end.
     *    Uses bitmap checking via @c header_ref().populations for efficient skipping.
     */
    void skip_non_populated() noexcept {
        while ((slots_remaining != 0) && (~header_ref().populations & mask_in_bucket())) {
            --slots_remaining;
            ++slot_;
        }
    }
};

/**
 *  @brief Wraps @c std::hash default hasher that delays type resolution until invocation.
 */
struct lazy_std_hash_t {

    template <typename key_type_>
    std::size_t operator()(key_type_ const &key) const noexcept {
        return std::hash<key_type_> {}(key);
    }
};

/**
 *  @brief Lock-free concurrent hash table with linear probing and Structure-of-Arrays layout.
 *    Compatible with @c std::unordered_set interface. See file header for detailed design rationale.
 *
 *  @tparam key_type_ Hashable and equality-comparable key type.
 *  @tparam value_type_ Value type, or @c void for hash sets.
 *  @tparam hasher_type_ Hash function type. Must be copy-constructible. Defaults to a @c std::hash wrapper.
 *  @tparam equals_type_ Equality predicate supporting heterogeneous lookups. Must be copy-constructible.
 *    Defaults to @c std::equal_to.
 *  @tparam allocator_type_ Allocator for internal memory management. Defaults to @c std::allocator<std::byte>.
 *
 *  @see https://en.cppreference.com/w/cpp/container/unordered_set
 *  @see https://en.cppreference.com/w/cpp/container/unordered_map
 */
template <typename element_type_, typename hasher_type_ = lazy_std_hash_t, typename equals_type_ = std::equal_to<>,
          typename allocator_type_ = std::allocator<std::byte>>
class basic_hash_table {

    using layout_t = bht_layout_for<element_type_, hasher_type_>;
    using key_t = typename layout_t::key_t;
    using value_t = typename layout_t::value_t;
    using element_t = typename layout_t::element_t;
    using element_copy_t = typename layout_t::element_copy_t;
    using offset_t = typename layout_t::offset_t;

    inline static constexpr std::size_t bytes_for_keys_k = layout_t::bytes_for_keys_k;
    inline static constexpr std::size_t bytes_for_values_k = layout_t::bytes_for_values_k;
    inline static constexpr bool hasher_is_empty_k = std::is_empty<hasher_type_>::value;
    inline static constexpr bool equals_is_empty_k = std::is_empty<equals_type_>::value;
    inline static constexpr bool allocator_is_empty_k = std::is_empty<allocator_type_>::value;

    using hasher_t = hasher_type_;
    using equals_t = equals_type_;
    using allocator_t = allocator_type_;
    using set_t = basic_hash_table<key_t, hasher_t, equals_t, allocator_t>; // ? Drops the values
    using iterator_t = bht_iterator_gt<key_t, value_t, hasher_t>;
    using iterator_ct = bht_iterator_gt<key_t, value_t const, hasher_t>;
    using element_rt = bht_slot_ref<key_t, value_t, hasher_t>;
    using element_crt = bht_slot_ref<key_t const, value_t const, hasher_t>;
    using element_atref_t = bht_element_atomic_ref<key_t, value_t, hasher_t>;
    using element_catref_t = bht_element_atomic_ref<key_t const, value_t const, hasher_t>;

    static_assert(!std::is_reference<key_t>(), "Keys can't be references!");
    static_assert(!std::is_reference<value_t>(), "Values can't be references!");

    inline static constexpr bool destruct_keys_k = !std::is_trivially_destructible<key_t>();
    inline static constexpr bool destruct_vals_k = has_values_k && !std::is_trivially_destructible<value_t>();

    using hash_value_t = decltype(hasher_t {}(key_t {}));
    static_assert(std::is_unsigned<hash_value_t>(),
                  "Hash value must be an unsigned integer, like std::uint32_t or std::uint64_t!");

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

    struct search_result_t {
        iterator_t position;
        bool exists = false;
    };

    struct search_cresult_t {
        iterator_ct position;
        bool exists = false;
    };

    /*  STL-compatibility definitions, identical to that of `std::map`:
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map
     */
    using key_type = key_t;
    using mapped_typed = value_t;
    using result_type = insert_result_t;
    using value_type = element_t;
    using size_type = offset_t;
    using difference_type = ptrdiff_t;
    using key_compare = equals_t;
    using allocator_type = allocator_t;
    using reference = element_t;
    using const_reference = element_t;
    using iterator = iterator_t;
    using const_iterator = iterator_ct;
    using hasher = hasher_t;
    using key_equal = equals_t;

  private:
    /**
     *  @brief Single allocation containing three cache-aligned regions.
     *    Layout: [keys | values | headers] enabling zero-copy compaction.
     */
    std::byte *memory_; /**< Base pointer to single allocated buffer. */

    offset_t slots_count_;      /**< Total number of slots (power of two). */
    offset_t growth_threshold_; /**< Rehash trigger at 75% load factor. */
    offset_t populated_count_;  /**< Count of populated slots. */
    offset_t deleted_count_;    /**< Count of deleted (tombstone) slots. */

    [[no_unique_address]] hasher_t hasher_;
    [[no_unique_address]] equals_t equals_;
    [[no_unique_address]] allocator_t allocator_;

    key_t *keys() const noexcept { return reinterpret_cast<key_t *>(memory_); }
    value_t *values() const noexcept {
        if constexpr (has_values_k)
            return reinterpret_cast<value_t *>(memory_ + bytes_for_keys_k * slots_count_ / bht_bucket_capacity_k);
        else return nullptr;
    }
    bht_bucket_head_t *headers() const noexcept {
        if constexpr (has_values_k)
            return reinterpret_cast<bht_bucket_head_t *>(memory_ +
                                                         bytes_for_keys_k * slots_count_ / bht_bucket_capacity_k +
                                                         bytes_for_values_k * slots_count_ / bht_bucket_capacity_k);
        else reinterpret_cast<bht_bucket_head_t *>(memory_ + bytes_for_keys_k * slots_count_ / bht_bucket_capacity_k);
    }

  public:
#pragma mark - Constructors

    /**
     *  @brief Default constructor, avoiding memory allocations, accepting a pre-constructed hasher functor,
     *    an equality operator, and the allocator state.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/unordered_map
     */
    basic_hash_table(hasher_t h = {}, equals_t eq = {}, allocator_t alloc = {}) noexcept
        : memory_(nullptr), slots_count_(0), growth_threshold_(0), populated_count_(0), deleted_count_(0),
          hasher_(std::move(h)), equals_(std::move(eq)), allocator_(std::move(alloc)) {}

    inline basic_hash_table(basic_hash_table &&other) noexcept
        : memory_(nullptr), slots_count_(0), growth_threshold_(0), populated_count_(0), deleted_count_(0), hasher_(),
          equals_(), allocator_() {
        swap(other);
    }

    inline basic_hash_table &operator=(basic_hash_table &&other) noexcept {
        clear();
        swap(other);
        return *this;
    }

    inline basic_hash_table(basic_hash_table const &) noexcept = delete;
    inline basic_hash_table &operator=(basic_hash_table const &) noexcept = delete;
    inline basic_hash_table(offset_t, hasher_t = {}, equals_t = {}, allocator_t = {}) noexcept = delete;

    /**
     *  @brief Most commonly used @b constructor-like interface. Returns an optional hash table instance,
     *    containing a @c std::nullopt if the memory allocation has failed.
     */
    static std::optional<basic_hash_table> make(offset_t planned_elements, hasher_t h = {}, equals_t eq = {},
                                                allocator_t alloc = {}) noexcept {
        return basic_hash_table(bht_slots_count_t {planned_elements}, std::move(h), std::move(eq), std::move(alloc));
    }

    /**
     *  @brief Most commonly used @b constructor-like interface. Returns an optional hash table instance,
     *    containing a @c std::nullopt if the memory allocation has failed.
     */
    static std::optional<basic_hash_table> make(bht_slots_count_t slots, hasher_t h = {}, equals_t eq = {},
                                                allocator_t alloc = {}) noexcept {

        hasher_construct(std::move(h));
        equals_construct(std::move(eq));
        allocator_construct(std::move(alloc));

        std::size_t needed_bytes = memory_usage(slots);
        memory_ = nullptr;

        if (needed_bytes && slots.raw > 0) {
            memory_ = allocator_().allocate(needed_bytes);
            assert(memory_ && "Allocation has failed!");
            allocator_().clear(memory_, needed_bytes);
        }

        slots_count_ = static_cast<offset_t>(slots.raw);
        populated_count_ = 0;
        deleted_count_ = 0;
        growth_threshold_ = slots_count_ * 3ul / 4ul;
    }

    ~basic_hash_table() {
        if (memory_) {
            clear(assume_reserved_t {});
            deallocate(assume_reserved_t {});
        }
    }

    template <typename begin_iterator_type_, typename end_iterator_type_,
              typename std::enable_if<is_iterator<begin_iterator_type_>(), int>::type = 0>
    static std::optional<basic_hash_table> make(begin_iterator_type_ begin, end_iterator_type_ end) {
        basic_hash_table table;
        if (!table.reserve(static_cast<offset_t>(std::distance(begin, end)))) return std::nullopt;
        insert_reserved(begin, end);
        return table;
    }

#pragma mark - Metadata

    bool empty() const noexcept { return !populated_count_; }
    offset_t size() const noexcept { return populated_count_; }

    offset_t optimal_capacity() const noexcept { return growth_threshold_; }
    offset_t capacity() const noexcept { return std::max(optimal_capacity(), size()); }

    std::size_t size_bytes() const noexcept { return bucket_count() * bytes_in_bucket_k; }
    offset_t capacity_bytes() const noexcept { return size_bytes(); }

    bht_slots_count_t slots_count() const noexcept {
        bht_slots_count_t result;
        result.raw = slots_count_;
        return result;
    }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/bucket_count
     */
    offset_t bucket_count() const noexcept { return slots_count_ / bht_bucket_capacity_k; }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/max_bucket_count
     */
    offset_t max_bucket_count() const noexcept { return std::numeric_limits<offset_t>::max() / bht_bucket_capacity_k; }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/hash_function
     */
    decltype(auto) hash_function() const noexcept { return hasher_(); }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/key_eq
     */
    decltype(auto) key_eq() const noexcept { return equals_(); }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/get_allocator
     */
    decltype(auto) get_allocator() const noexcept { return allocator_(); }

#pragma mark - Search

    /**
     * @brief Search optimized for @c find: check is value is present.
     * If you have an intention to insert/upsert something, use another func.
     * Unlike the @c search_to_insert or @c search_to_upsert, doesn't
     * track "deleted" slots.
     *
     * @param wanted Hashable and comparable with key object.
     * @param call A callback receiving @c element_crt or
     * an atomic @c element_catref_t to an initialized matching object.
     *
     * @param tags Markers for special acceleration:
     * > assume_reserved_t: Avoids null checks.
     * > threadsafe_t: Enables lock-less concurrency.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    void search_to_find(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_...) const noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), element_catref_t, element_crt>;
        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (!populated_count_) [[unlikely]]
                return;

        offset_t const offset_mask = slots_count_ - 1;
        // offset_t const max_attempts = offset_mask / 2;
        offset_t const initial_offset = hasher_()(wanted) & offset_mask;
        [[maybe_unused]] offset_t const final_offset = initial_offset; // (initial_offset + max_attempts) & offset_mask;

        ref_t addr;
        offset_t off = initial_offset;

        while (true) {
            unsafe_retarget(addr, off);
            addr.lock();

            // Checking for equality isn't safe, if we operate on uninitialized
            // memory and call some complex comparison operators on them.
            // So instead of one runtime `if`, we have two nested `if`s.
            if (addr.is_populated()) {
                // A "populated" slot. Can't be "free" or "deleted".
                // Shouldn't be "locked" when used properly.
                if (equals_()(addr.key(), wanted)) {
                    call(addr);
                    // If we reached here without firing callback - nothing was found.
                    // Gracefully quit the loop!
                    addr.unlock();
                    break;
                }
                else {
                    addr.unlock();
                    off = (off + 1) & (slots_count_ - 1);
                    assert(off != final_offset && "Poor hash table usage!");

                    if constexpr (contains_type<first_match_t, tags_types_...>()) break;

                    // if (off == final_offset)
                    //     break;
                }
            }
            else if (addr.is_deleted()) {
                // A "deleted" slot.
                addr.unlock();
                off = (off + 1) & (slots_count_ - 1);
                assert(off != final_offset && "Poor hash table usage!");

                if constexpr (contains_type<first_match_t, tags_types_...>()) break;

                // if (off == final_offset)
                //     break;
            }
            else {
                // A "free" slot.
                addr.unlock();
                break;
            }
        }
    }

    /**
     * @brief Search optimized for @c find: check is value is present.
     * If you have an intention to insert/upsert something, use another func.
     * Unlike the @c search_to_insert or @c search_to_upsert, doesn't
     * track "deleted" slots.
     *
     * @param wanted Hashable and comparable with key object.
     * @param call A callback receiving @c element_crt or
     * an atomic @c element_catref_t to an initialized matching object.
     *
     * @param tags Markers for special acceleration:
     * > assume_reserved_t: Avoids null checks.
     * > threadsafe_t: Enables lock-less concurrency.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    void search_to_find(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_...) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), element_atref_t, element_rt>;
        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (!populated_count_) [[unlikely]]
                return;

        offset_t const offset_mask = slots_count_ - 1;
        // offset_t const max_attempts = offset_mask / 2;
        offset_t const initial_offset = hasher_()(wanted) & offset_mask;
        [[maybe_unused]] offset_t const final_offset = initial_offset; // (initial_offset + max_attempts) & offset_mask;

        ref_t addr;
        offset_t off = initial_offset;

        while (true) {
            unsafe_retarget(addr, off);
            addr.lock();

            // Checking for equality isn't safe, if we operate on uninitialized
            // memory and call some complex comparison operators on them.
            // So instead of one runtime `if`, we have two nested `if`s.
            if (addr.is_populated()) {
                // A "populated" slot. Can't be "free" or "deleted".
                // Shouldn't be "locked" when used properly.
                if (equals_()(addr.key(), wanted)) {
                    call(addr);
                    // If we reached here without firing callback - nothing was found.
                    // Gracefully quit the loop!
                    addr.unlock();
                    break;
                }
                else {
                    addr.unlock();
                    off = (off + 1) & (slots_count_ - 1);
                    assert(off != final_offset && "Poor hash table usage!");

                    if constexpr (contains_type<first_match_t, tags_types_...>()) break;

                    // if (off == final_offset)
                    //     break;
                }
            }
            else if (addr.is_deleted()) {
                // A "deleted" slot.
                addr.unlock();
                off = (off + 1) & (slots_count_ - 1);
                assert(off != final_offset && "Poor hash table usage!");

                if constexpr (contains_type<first_match_t, tags_types_...>()) break;

                // if (off == final_offset)
                //     break;
            }
            else {
                // A "free" slot.
                addr.unlock();
                break;
            }
        }
    }

    template <typename... tags_types_>
    inline offset_t advance(offset_t &count, offset_t offset, tags_types_...) const noexcept {
        if constexpr (contains_type<threadsafe_t, tags_types_...>()) return atomic_add_fetch(count, offset);
        else return count += offset;
    }

    template <typename... tags_types_>
    inline offset_t decrement(offset_t &count, offset_t offset, tags_types_...) const noexcept {
        if constexpr (contains_type<threadsafe_t, tags_types_...>()) return atomic_sub_fetch(count, offset);
        else return count -= offset;
    }

    /**
     * @brief Search optimized for @c insert of a uniquie key.
     * Unlike @c search_to_upsert avoids potentially expensive
     * equality comparisons, knowing that the incoming key
     * is different from all present members.
     *
     * @param wanted Hashable and comparable with key object.
     * @param call A callback receiving  @c element_rt or
     * an atomic @c element_atref_t to UN-initialized memory,
     * where an element should be built.
     *
     * @param tags Markers for special acceleration:
     * > assume_reserved_t: Avoids null checks.
     * > threadsafe_t: Enables lock-less concurrency.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    void search_to_insert(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), element_atref_t, element_rt>;
        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (!populated_count_) [[unlikely]]
                return;

        // We pre-increment the counter, before beginning the potentially slow
        // search. It helps avoiding premature cleanup from a different thread.
        get_type_or<return_new_size_t, black_hole_t>(tags...) = advance(populated_count_, 1, tags...);

        // Hash to determine the ideal slot.
        ref_t addr;
        offset_t off = hasher_()(wanted) & (slots_count_ - 1);
        bool did_find_deleted = false;

        // Loop until we find any "deleted" or "free" slot.
        while (true) {
            unsafe_retarget(addr, off);
            addr.lock();

            if ((~addr.header_ref().populations | addr.header_ref().deletions) & addr.mask_in_bucket()) {
                call(addr);

                // Update the cell state before unlocking.
                did_find_deleted = boolify(addr.header_ref().deletions & addr.mask_in_bucket());
                addr.mark_populated();
                addr.unlock();

                // Decrement this counter afterwards - in a relaxed manner.
                // Somebody else might be already searching this slot,
                // we don't want them to wait :)
                decrement(deleted_count_, did_find_deleted, tags...);
                break;
            }
            else {
                // A slot is "populated", definitely with a different key!
                addr.unlock();
                off = (off + 1) & (slots_count_ - 1);
            }
        }
    }

    /**
     * @brief Search optimized for potentially @c upserts.
     * It's more expensive than @c search_to_find and @c search_to_insert,
     * so pick this method wisely.
     *
     * @param wanted Hashable and comparable with key object.
     * @param call_unused A callback receiving @c element_rt or
     * an atomic @c element_atref_t to UN-initialized memory,
     * where an element should be built.
     * @param call_equal A callback receiving @c element_rt or
     * an atomic @c element_atref_t to initialized matching object.
     *
     * @param tags Markers for special acceleration:
     * > assume_reserved_t: Avoids null checks.
     * > threadsafe_t: Enables lock-less concurrency.
     */
    template <typename comparable_key_type_, typename callback_unused_at, typename callback_equal_at,
              typename... tags_types_>
    void search_to_upsert(comparable_key_type_ &&wanted, callback_unused_at &&call_unused,
                          callback_equal_at &&call_equal, tags_types_... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), element_atref_t, element_rt>;
        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (!populated_count_) [[unlikely]]
                return;

        offset_t const offset_mask = slots_count_ - 1;
        // offset_t const max_attempts = offset_mask / 2;
        offset_t const initial_offset = hasher_()(wanted) & offset_mask;
        [[maybe_unused]] offset_t const final_offset = initial_offset; // (initial_offset + max_attempts) & offset_mask;

        offset_t off = initial_offset;
        bool did_find_deleted = false;

        /// Contains two offsets, first for current target.
        /// And one for the first observed "deleted" slot, if any exists.
        /// The second slot is locked, if exists.
        ref_t tgts[2];

        // Retarget the "deleted" to avoid writing to uninitialized memory.
        // But don't lock!
        unsafe_retarget(tgts[1], off);

        while (true) {
            // Loop until we find the first ("populated" AND equal) OR "free" slot.
            unsafe_retarget(tgts[0], off);
            tgts[0].lock();

            if (tgts[0].is_populated()) {
                // We have found a filled slot and must check for match:
                // > If matches: unlock the deleted slot, call back, unlock current, exit.
                // > If not: unlock current and jump forward.
                if (equals_()(tgts[0].key(), wanted)) {
                    tgts[1].unlock();
                    call_equal(tgts[0]);
                    tgts[0].unlock();
                    break;
                }
                else {
                    if constexpr (contains_type<first_match_t, tags_types_...>()) {
                        call_unused(tgts[0]);
                        tgts[0].mark_populated();
                        tgts[0].unlock();
                        break;
                    }

                    tgts[0].unlock();
                    off = (off + 1) & (slots_count_ - 1);
                    assert(off != final_offset && "Poor hash table usage!");
                    continue;

                    // if (off != final_offset)
                    //     continue;

                    // if (!did_find_deleted) [[unlikely]]
                    //     break;
                }
            }
            else if (tgts[0].is_deleted()) {
                // Found a deleted slot:
                // > If its first deleted slot: keep it for future, don't unlock, jump forward.
                // > If we already saw deleted slots: unlock current, just jump forward.
                if constexpr (contains_type<first_match_t, tags_types_...>()) {
                    call_unused(tgts[0]);
                    tgts[0].mark_populated();
                    tgts[0].unlock();

                    // Change this counters afterwards - in a relaxed manner.
                    // Somebody else might be already searching for this slot,
                    // we don't want them to wait :)
                    get_type_or<return_new_size_t, black_hole_t>(tags...) = advance(populated_count_, 1, tags...);
                    decrement(deleted_count_, true, tags...);
                    break;
                }

                tgts[!did_find_deleted].unlock();
                tgts[1] = tgts[did_find_deleted];

                did_find_deleted = true;
                off = (off + 1) & (slots_count_ - 1);

                assert(off != final_offset && "Poor hash table usage!");
                continue;

                // if (off != final_offset)
                //     continue;
            }

            // In case of "free" slot:
            // > If we previously saw deleted slot: return it, unlock both, and decrement deleted counter.
            // > Otherwise: pseudo-unlock the deleted one, return unused slot, unlock the empty one.
            tgts[!did_find_deleted].unlock();
            auto &addr = tgts[did_find_deleted];
            call_unused(addr);
            addr.mark_populated();
            addr.unlock();

            // Change this counters afterwards - in a relaxed manner.
            // Somebody else might be already searching for this slot,
            // we don't want them to wait :)
            get_type_or<return_new_size_t, black_hole_t>(tags...) = advance(populated_count_, 1, tags...);
            decrement(deleted_count_, did_find_deleted, tags...);
            break;
        }
    }

    /**
     *  @brief Retargets an element reference to a specific slot using direct array indexing.
     *    Simplified from bucket arithmetic to direct slot assignment, enabling SIMD-friendly access.
     *
     *  @tparam similar_key_at Key type (may differ in const qualification).
     *  @tparam similar_val_at Value type (may differ in const qualification).
     *  @param addr Element reference to retarget with new pointers and slot index.
     *  @param slot Absolute slot index in [0, slots_count_).
     */
    template <typename similar_key_at, typename similar_val_at>
    void unsafe_retarget(bht_slot_ref<similar_key_at, similar_val_at, hasher_t> &addr, offset_t slot) const noexcept {
        addr.keys_ = const_cast<key_t *>(keys_);
        addr.values_ = const_cast<value_t *>(values_);
        addr.headers_ = const_cast<bht_bucket_head_t *>(headers_);
        addr.slot_ = slot;
    }

#pragma mark - Lookups

    /**
     *  @brief Calculates slots remaining after the current slot position.
     *    Simplified for slot-based addressing: just subtract current slot from total.
     *
     *  @param addr Element reference or iterator with current @c slot_ position.
     *  @return Number of slots from current position to end of table.
     */
    template <typename address_or_iterator_at>
    offset_t slots_remaining_after(address_or_iterator_at &&addr) const noexcept {
        return slots_count_ - addr.slot_;
    }

    template <typename comparable_key_type_, typename... tags_types_>
    bool contains(comparable_key_type_ &&wanted, tags_types_... tags) const noexcept {
        bool result = false;
        search_to_find(
            std::forward<comparable_key_type_>(wanted), [&result](element_crt const &) { result = true; }, tags...);
        return result;
    }

    /**
     * @brief Similar to STL, searches for element with the given key.
     * @return End iterator, iff element with such key is missing.
     */
    template <typename comparable_key_type_ = key_t const &>
    iterator_ct find(comparable_key_type_ &&wanted) const noexcept {
        iterator_ct it;
        it.keys_ = const_cast<key_t *>(keys_);
        it.values_ = const_cast<value_t *>(values_);
        it.headers_ = const_cast<bht_bucket_head_t *>(headers_);
        it.slot_ = slots_count_; // End position
        it.slots_remaining = 0;
        search_to_find(std::forward<comparable_key_type_>(wanted),
                       [&it](element_crt const &addr) { it.slot_ = addr.slot_; });
        it.slots_remaining = slots_remaining_after(it);
        return it;
    }

    /**
     * @brief Similar to STL, searches for element with the given key.
     * @return End iterator, iff element with such key is missing.
     */
    template <typename comparable_key_type_ = key_t const &>
    iterator_t find(comparable_key_type_ &&wanted) noexcept {
        iterator_t it;
        it.keys_ = keys_;
        it.values_ = values_;
        it.headers_ = headers_;
        it.slot_ = slots_count_; // End position
        it.slots_remaining = 0;
        search_to_find(std::forward<comparable_key_type_>(wanted),
                       [&it](element_rt const &addr) { it.slot_ = addr.slot_; });
        it.slots_remaining = slots_remaining_after(it);
        return it;
    }

    /**
     *  @brief Lock-free atomic lookup that invokes callback if key is found.
     *    Requires prior @c reserve() to prevent reallocations during concurrent access.
     *
     *  @tparam comparable_key_type_ Type comparable with @c key_t via @c equals_t.
     *  @tparam callback_type_ Functor type accepting @c element_catref_t.
     *  @param[in] wanted Key to search for, hashable and comparable.
     *  @param[in] callback Invoked with atomic reference to element if found.
     *  @return True if key was found and callback invoked, false otherwise.
     *
     *  @note This method uses atomic operations on bucket headers for lock-free concurrent reads.
     *    Do not mix with non-atomic operations on the same table instance.
     */
    template <typename comparable_key_type_, typename callback_type_>
    bool find_atomic(comparable_key_type_ &&wanted, callback_type_ &&callback) const noexcept {
        if (!populated_count_) return false;

        offset_t const offset_mask = slots_count_ - 1;
        offset_t const initial_offset = hasher_()(wanted) & offset_mask;

        element_catref_t addr;
        offset_t slot = initial_offset;
        bool found = false;

        while (true) {
            unsafe_retarget(addr, slot);
            addr.lock(); // Atomic lock via fetch_or

            if (addr.is_populated()) {
                if (equals_()(addr.key(), wanted)) {
                    callback(addr);
                    found = true;
                    addr.unlock();
                    break;
                }
                else {
                    addr.unlock();
                    slot = (slot + 1) & offset_mask;
                }
            }
            else if (addr.is_deleted()) {
                addr.unlock();
                slot = (slot + 1) & offset_mask;
            }
            else {
                // Free slot - key doesn't exist
                addr.unlock();
                break;
            }
        }

        return found;
    }

    /**
     *  @brief Lock-free atomic check for key existence.
     *    Lighter-weight than @c find_atomic() when you only need presence check.
     *
     *  @param[in] wanted Key to search for.
     *  @return True if key exists in table, false otherwise.
     */
    template <typename comparable_key_type_>
    bool contains_atomic(comparable_key_type_ &&wanted) const noexcept {
        return find_atomic(std::forward<comparable_key_type_>(wanted), [](element_catref_t const &) {});
    }

    /**
     * @brief Returns a reference to the mapped value of the element with key equivalent
     * to key. If no such element exists, an exception is thrown in @c DEBUG builds.
     * https://en.cppreference.com/w/cpp/container/unordered_map/at
     */
    template <typename comparable_key_type_ = key_t const &>
    decltype(auto) at(comparable_key_type_ &&key) noexcept {
        element_rt result;
        result.keys_ = keys_;
        result.values_ = values_;
        result.headers_ = headers_;
        result.slot_ = slots_count_; // Mark as "not found" initially
        search_to_find(std::forward<comparable_key_type_>(key), [&result](element_rt const &addr) { result = addr; });
        assert(result.slot_ != slots_count_ && "The object must be already present");
        return result.value_ref();
    }

    /**
     * @brief Returns a reference to the const mapped value of the element with key equivalent
     * to key. If no such element exists, an exception is thrown in @c DEBUG builds.
     * https://en.cppreference.com/w/cpp/container/unordered_map/at
     */
    template <typename comparable_key_type_ = key_t const &>
    decltype(auto) at(comparable_key_type_ &&key) const noexcept {
        element_crt result;
        result.keys_ = const_cast<key_t *>(keys_);
        result.values_ = const_cast<value_t *>(values_);
        result.headers_ = const_cast<bht_bucket_head_t *>(headers_);
        result.slot_ = slots_count_; // Mark as "not found" initially
        search_to_find(std::forward<comparable_key_type_>(key), [&result](element_crt const &addr) { result = addr; });
        assert(result.slot_ != slots_count_ && "The object must be already present");
        return result.value_ref();
    }

    /**
     * @brief A temporarily banned function, that is replaced by @b @c at().
     * https://en.cppreference.com/w/cpp/container/unordered_map/operator_at
     */
    template <typename comparable_key_type_ = key_t const &>
    void operator[](comparable_key_type_ &&) = delete;

    /**
     * @brief Returns the number of elements with key that compares equal to
     * the specified argument key, which is either 1 or 0 since this container
     * does not allow duplicates.
     * https://en.cppreference.com/w/cpp/container/unordered_map/count
     */
    template <typename comparable_key_type_ = key_t const &>
    std::size_t count(comparable_key_type_ &&key) const noexcept {
        return contains(key);
    }

#pragma mark - Scans

    iterator_t begin() noexcept {
        iterator_t it;
        it.keys_ = keys_;
        it.values_ = values_;
        it.headers_ = headers_;
        if (empty()) {
            it.slot_ = slots_count_;
            it.slots_remaining = 0;
        }
        else {
            it.slot_ = 0;
            it.slots_remaining = slots_count_;
            it.skip_non_populated();
        }
        return it;
    }

    iterator_ct cbegin() const noexcept {
        iterator_ct it;
        it.keys_ = const_cast<key_t *>(keys_);
        it.values_ = const_cast<value_t *>(values_);
        it.headers_ = const_cast<bht_bucket_head_t *>(headers_);
        if (empty()) {
            it.slot_ = slots_count_;
            it.slots_remaining = 0;
        }
        else {
            it.slot_ = 0;
            it.slots_remaining = slots_count_;
            it.skip_non_populated();
        }
        return it;
    }

    iterator_ct begin() const noexcept { return cbegin(); }
    end_sentinel_t end() const noexcept { return end_sentinel_t {}; }
    end_sentinel_t cend() const noexcept { return end_sentinel_t {}; }

#pragma mark - Set Algorithms

    /**
     * @brief Similar to @c std::set_intersection, but reports only
     * a boolean checking if there is any collision between any keys.
     * https://en.cppreference.com/w/cpp/algorithm/set_intersection
     *
     * @param small The small Hash-Table from the two.
     * @return bool True if any key intersects. False otherwise.
     */
    template <typename val_small_at, typename hasher_small_at, typename equals_small_at, typename allocator_small_at>
    bool intersection_exists(basic_hash_table<key_t, val_small_at, hasher_small_at, equals_small_at,
                                              allocator_small_at> const &small) const noexcept {

        auto const &big = *this;
        if (small.empty() | big.empty()) return false;
        if (small.size() > big.size()) return small.intersection_exists(big);

        return small.find_if(
            [&big](auto const &small_addr) { return big.contains(small_addr.key(), assume_reserved_t {}); });
    }

    /**
     * @brief Similar to @c std::set_intersection, but reports only
     * the number of matching keys between two containers.
     * https://en.cppreference.com/w/cpp/algorithm/set_intersection
     *
     * @param small The small Hash-Table from the two.
     * @return std::size_t The number of matches.
     */
    template <typename val_small_at, typename hasher_small_at, typename equals_small_at>
    std::size_t intersection_size(
        basic_hash_table<key_t, val_small_at, hasher_small_at, equals_small_at> const &small) const noexcept {

        auto const &big = *this;
        if (small.empty() | big.empty()) return 0;
        if (small.size() > big.size()) return big.intersection_size(small);

        std::size_t cnt = 0;
        small.for_each([&](auto const &addr) { cnt += big.contains(addr.key(), assume_reserved_t {}); });
        return cnt;
    }

    /**
     * @brief Similar to @c std::set_union, but reports only
     * the number of matching keys between two containers.
     * https://en.cppreference.com/w/cpp/algorithm/set_union
     *
     * @param small The small Hash-Table from the two.
     * @return std::size_t The number of unique elements in two combined Hash-Tables.
     */
    template <typename val_other_type_, typename hasher_other_type_, typename equals_other_type_>
    std::size_t merge_size(
        basic_hash_table<key_t, val_other_type_, hasher_other_type_, equals_other_type_> const &other) const noexcept {
        return other.size() + size() - intersection_size(other);
    }

#pragma mark - Insertions

    template <typename convertible_key_type_, typename convertible_value_type_>
    static constexpr bool can_use_map_emplace() {
        return has_values_k && std::is_constructible<key_t, convertible_key_type_ &&>() &&
               std::is_constructible<value_t, convertible_value_type_ &&>();
    }

    template <typename convertible_key_type_>
    static constexpr bool can_use_set_emplace() {
        return !has_values_k && std::is_constructible<key_t, convertible_key_type_ &&>();
    }

    /**
     * @brief Emplacing with a hint is banned in favor of lower-level
     * function @c search_to_upsert, with more customizable behaviour.
     * https://en.cppreference.com/w/cpp/container/unordered_map/emplace_hint
     */
    template <
        typename convertible_key_type_, typename convertible_value_type_, typename... tags_types_,
        typename std::enable_if<can_use_map_emplace<convertible_key_type_, convertible_value_type_>(), int>::type = 0>
    iterator_t emplace_hint(iterator_ct, convertible_key_type_ &&k, convertible_value_type_ &&v,
                            tags_types_... tags) = delete;

    /**
     * @brief Emplacing with a hint is banned in favor of lower-level
     * function @c search_to_upsert, with more customizable behaviour.
     * https://en.cppreference.com/w/cpp/container/unordered_map/emplace_hint
     */
    template <typename convertible_key_type_, typename... tags_types_,
              typename std::enable_if<can_use_set_emplace<convertible_key_type_>(), int>::type = 0>
    iterator_t emplace_hint(iterator_ct, convertible_key_type_ &&k, tags_types_... tags) = delete;

    template <typename... tags_types_>
    static constexpr bool emplace_returns() {
        return contains_type<return_position_t, tags_types_...>();
    }

    template <typename... tags_types_>
    using emplace_return = std::conditional_t<emplace_returns<tags_types_...>(), insert_result_t, void>;

    /**
     * @brief Main insertion method, that both finds the optimal location and
     * overwrites it. More performant than @c insert, as avoids temporary objects
     * construction. Compatiable with STL code.
     * https://en.cppreference.com/w/cpp/container/unordered_map/emplace
     *
     * @param tags Markers for special acceleration:
     * > assume_reserved_t: Would avoid reserving more memory.
     * > assume_unique_t: Would avoid potentially expensive equality comparisons.
     * > return_position_t: Avoids calculating the resulting iterator offsets.
     * > threadsafe_t: Would allow concurrent read & write operations.
     * > return_new_size_t: Atomically exports the updated @c size().
     *
     * @return Returns nothing unless the @c needs_result_tag_t is provided.
     * In latter case @c insert_result_t containing the iterator will be constructed.
     */
    template <
        typename convertible_key_type_, typename convertible_value_type_, typename... tags_types_,
        typename std::enable_if<can_use_map_emplace<convertible_key_type_, convertible_value_type_>(), int>::type = 0>
    emplace_return<tags_types_...> emplace(convertible_key_type_ &&key, convertible_value_type_ &&value,
                                           tags_types_... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), element_atref_t, element_rt>;
        if constexpr (contains_type<threadsafe_t, tags_types_...>()) {
            static_assert(contains_type<assume_reserved_t, tags_types_...>(),
                          "Atomic operations can't cause reallocations!");
            static_assert(!contains_type<return_position_t, tags_types_...>(),
                          "Atomic operations can't return a persistent iterator!");
        }

        if constexpr (!contains_type<assume_reserved_t, tags_types_...>()) reserve_more(1);

        constexpr bool return_k = emplace_returns<tags_types_...>();
        using result_t = std::conditional_t<return_k, insert_result_t, dummy_t>;

        result_t result;
        if constexpr (return_k) {
            result.position.keys_ = keys_;
            result.position.values_ = values_;
            result.position.headers_ = headers_;
            result.position.slot_ = 0;
            result.position.slots_remaining = 0;
        }

        auto callback_unused = [&](ref_t &addr_unused) {
            // Construct inplace:
            new (&addr_unused.key_ref()) key_t(std::forward<convertible_key_type_>(key));
            new (&addr_unused.value_ref()) value_t(std::forward<convertible_value_type_>(value));

            // Export results:
            if constexpr (emplace_returns<tags_types_...>()) {
                result.position.keys_ = addr_unused.keys_;
                result.position.values_ = addr_unused.values_;
                result.position.headers_ = addr_unused.headers_;
                result.position.slot_ = addr_unused.slot_;
                result.inserted = true;
            }
        };

        if constexpr (contains_type<assume_unique_t, tags_types_...>())
            search_to_insert(key, callback_unused, assume_reserved_t {}, tags...);
        else
            search_to_upsert(
                key, callback_unused,
                [&](ref_t &addr_equal) {
                    addr_equal.value_ref() = std::forward<convertible_value_type_>(value);
                    if constexpr (emplace_returns<tags_types_...>()) {
                        result.position.keys_ = addr_equal.keys_;
                        result.position.values_ = addr_equal.values_;
                        result.position.headers_ = addr_equal.headers_;
                        result.position.slot_ = addr_equal.slot_;
                    }
                },
                assume_reserved_t {}, tags...);

        if constexpr (return_k) {
            result.position.slots_remaining = slots_remaining_after(result.position);
            return result;
        }
    }

    /**
     * @brief Main insertion method, that both finds the optimal location and
     * overwrites it. More performant than @c insert, as avoids temporary objects
     * construction. Compatiable with STL code.
     * https://en.cppreference.com/w/cpp/container/unordered_map/emplace
     *
     * @param tags Markers for special acceleration:
     * > assume_reserved_t: Would avoid reserving more memory.
     * > assume_unique_t: Would avoid potentially expensive equality comparisons.
     * > return_position_t: Avoids calculating the resulting iterator offsets.
     * > threadsafe_t: Would allow concurrent read & write operations.
     * > return_new_size_t: Atomically exports the updated @c size().
     *
     * @return Returns nothing unless the @c needs_result_tag_t is provided.
     * In latter case @c insert_result_t containing the iterator will be constructed.
     */
    template <typename convertible_key_type_, typename... tags_types_,
              typename std::enable_if<can_use_set_emplace<convertible_key_type_>(), int>::type = 0>
    emplace_return<tags_types_...> emplace(convertible_key_type_ &&key, tags_types_... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<threadsafe_t, tags_types_...>(), element_atref_t, element_rt>;
        if constexpr (!contains_type<assume_reserved_t, tags_types_...>()) reserve_more(1);

        constexpr bool return_k = emplace_returns<tags_types_...>();
        using result_t = std::conditional_t<return_k, insert_result_t, dummy_t>;

        result_t result;
        if constexpr (return_k) {
            result.position.keys_ = keys_;
            result.position.values_ = values_;
            result.position.headers_ = headers_;
            result.position.slot_ = 0;
            result.position.slots_remaining = 0;
        }

        auto callback_unused = [&](ref_t &addr_unused) {
            // Construct inplace:
            new (&addr_unused.key_ref()) key_t(std::forward<convertible_key_type_>(key));

            // Export results:
            if constexpr (emplace_returns<tags_types_...>()) {
                result.position.keys_ = addr_unused.keys_;
                result.position.values_ = addr_unused.values_;
                result.position.headers_ = addr_unused.headers_;
                result.position.slot_ = addr_unused.slot_;
                result.inserted = true;
            }
        };

        if constexpr (contains_type<assume_unique_t, tags_types_...>())
            search_to_insert(key, callback_unused, assume_reserved_t {}, tags...);
        else if constexpr (return_k)
            search_to_upsert(
                key, callback_unused,
                [&](ref_t &addr_equal) {
                    result.position.keys_ = addr_equal.keys_;
                    result.position.values_ = addr_equal.values_;
                    result.position.headers_ = addr_equal.headers_;
                    result.position.slot_ = addr_equal.slot_;
                },
                assume_reserved_t {}, tags...);
        else search_to_upsert(key, callback_unused, null_operator_t {}, assume_reserved_t {}, tags...);

        if constexpr (return_k) {
            result.position.slots_remaining = slots_remaining_after(result.position);
            return result;
        }
    }

    /**
     * @brief Main insertion and upsertion method. Compatiable with STL code.
     * https://en.cppreference.com/w/cpp/container/unordered_map/insert
     *
     * ! Don't try to use it with @c std::initializer_lists. Emplace!
     * With maps, this will only work on objects that have @c first and @c second
     * members, like @c std::pair or @c mapping.
     *
     * @param tags Markers for special acceleration:
     * > assume_reserved_t: Would avoid reserving more memory.
     * > assume_unique_t: Would avoid potentially expensive equality comparisons.
     * > return_position_t: Avoids calculating the resulting iterator offsets.
     * > threadsafe_t: Would allow concurrent read & write operations.
     * > return_new_size_t: Atomically exports the updated @c size().
     *
     * @return Returns nothing unless the @c return_position_t is provided.
     * In latter case @c insert_result_t containing the iterator will be constructed.
     */
    template <typename temporary_pack_at, typename... tags_types_>
    emplace_return<tags_types_...> insert(temporary_pack_at &&copy, tags_types_... tags) noexcept {
        if constexpr (has_values_k)
            if constexpr (std::is_rvalue_reference<decltype(copy)>())
                return emplace(std::move(copy.first), std::move(copy.second), tags...);
            else return emplace(copy.first, copy.second, tags...);
        else return emplace(std::forward<temporary_pack_at>(copy), tags...);
    }

    /**
     * @brief Inserts elements from range [begin, end).
     * If multiple elements in the range have keys that
     * compare equivalent, only the first element is inserted.
     * https://en.cppreference.com/w/cpp/container/unordered_map/insert
     *
     * @param tags Markers for special acceleration:
     * > assume_reserved_t: Would avoid reserving more memory.
     * > assume_unique_t: Would avoid potentially expensive equality comparisons.
     * > threadsafe_t: Would allow concurrent read & write operations.
     * > return_new_size_t: Atomically exports the updated @c size().
     *
     * @return Nothing, just like STL!
     */
    template <typename begin_iterator_type_, typename end_iterator_type_, typename... tags_types_,
              typename std::enable_if<is_iterator<begin_iterator_type_>(), int>::type = 0>
    inline void insert(begin_iterator_type_ begin, end_iterator_type_ end, tags_types_... tags) {

        if constexpr (!contains_type<assume_reserved_t, tags_types_...>()) reserve_more(end - begin);

        for (; begin != end; ++begin) insert(*begin, assume_reserved_t {}, tags...);
    }

    /**
     * @brief An interface similar to @c std::unordered_map::merge,
     * that is currenly banned in favor of a manual @c insert(begin, end),
     * preceded by something line @c intersection_size(...).
     * https://en.cppreference.com/w/cpp/container/unordered_map/merge
     */
    template <typename other_type_ = take_second_t>
    void merge(other_type_ &&) = delete;

    /**
     *  @brief Lock-free atomic insertion without reallocation.
     *    Requires prior @c reserve() to ensure sufficient capacity.
     *
     *  @tparam convertible_key_type_ Type constructible to @c key_t.
     *  @tparam convertible_value_type_ Type constructible to @c value_t.
     *  @param[in] key Key to insert or update, forwarded to in-place construction.
     *  @param[in] value Value to insert or update, forwarded to in-place construction.
     *
     *  @note Uses atomic operations on bucket headers for lock-free concurrent writes.
     *    Do not mix with non-atomic operations. If key exists, updates the value.
     */
    template <
        typename convertible_key_type_, typename convertible_value_type_,
        typename std::enable_if<can_use_map_emplace<convertible_key_type_, convertible_value_type_>(), int>::type = 0>
    void emplace_atomic(convertible_key_type_ &&key, convertible_value_type_ &&value) noexcept {
        offset_t const offset_mask = slots_count_ - 1;
        offset_t const initial_offset = hasher_()(key) & offset_mask;

        element_atref_t addr;
        offset_t slot = initial_offset;
        bool found_deleted = false;
        offset_t deleted_slot = 0;

        while (true) {
            unsafe_retarget(addr, slot);
            addr.lock();

            if ((~addr.header_ref().populations | addr.header_ref().deletions) & addr.mask_in_bucket()) {
                // Found a free or deleted slot
                bool is_deleted = addr.header_ref().deletions & addr.mask_in_bucket();

                // Construct in place
                new (&addr.key_ref()) key_t(std::forward<convertible_key_type_>(key));
                new (&addr.value_ref()) value_t(std::forward<convertible_value_type_>(value));

                // Mark as populated
                addr.mark_populated();
                addr.unlock();

                // Update counters atomically
                std::atomic_ref<offset_t> atomic_pop(populated_count_);
                atomic_pop.fetch_add(1, std::memory_order_relaxed);
                if (is_deleted) {
                    std::atomic_ref<offset_t> atomic_del(deleted_count_);
                    atomic_del.fetch_sub(1, std::memory_order_relaxed);
                }
                break;
            }
            else if (addr.is_populated()) {
                // Check if key matches (update case)
                if (equals_()(addr.key(), key)) {
                    // Update existing value
                    addr.value_ref() = std::forward<convertible_value_type_>(value);
                    addr.unlock();
                    break;
                }
                else {
                    // Continue probing
                    addr.unlock();
                    slot = (slot + 1) & offset_mask;
                }
            }
        }
    }

    /**
     *  @brief Insertion without reallocation, skips capacity check.
     *    Assumes @c reserve() was called beforehand or sufficient space is guaranteed.
     *
     *  @param[in] key Key to insert or update.
     *  @param[in] value Value to insert or update.
     *  @return insert_result_t with iterator to inserted/existing element and success flag.
     */
    template <
        typename convertible_key_type_, typename convertible_value_type_,
        typename std::enable_if<can_use_map_emplace<convertible_key_type_, convertible_value_type_>(), int>::type = 0>
    insert_result_t emplace_reserved(convertible_key_type_ &&key, convertible_value_type_ &&value) noexcept {
        return emplace(std::forward<convertible_key_type_>(key), std::forward<convertible_value_type_>(value),
                       assume_reserved_t {}, return_position_t {});
    }

    /**
     *  @brief Lock-free atomic insertion for unique keys without equality check.
     *    Most efficient insertion path: skips both reallocation and key comparison.
     *
     *  @param[in] key Guaranteed-unique key, equality check skipped for performance.
     *  @param[in] value Value to insert.
     *
     *  @note Use only when you can guarantee the key doesn't exist. Violating this
     *    assumption leads to duplicate keys and undefined behavior during lookup.
     */
    template <
        typename convertible_key_type_, typename convertible_value_type_,
        typename std::enable_if<can_use_map_emplace<convertible_key_type_, convertible_value_type_>(), int>::type = 0>
    void emplace_unique_atomic_reserved(convertible_key_type_ &&key, convertible_value_type_ &&value) noexcept {
        emplace(std::forward<convertible_key_type_>(key), std::forward<convertible_value_type_>(value), threadsafe_t {},
                assume_reserved_t {}, assume_unique_t {});
    }

    /**
     *  @brief Lock-free atomic value update for existing key.
     *    If key doesn't exist, no action is taken.
     *
     *  @tparam comparable_key_type_ Type comparable with @c key_t.
     *  @tparam convertible_value_type_ Type assignable to @c value_t.
     *  @param[in] key Key to search for.
     *  @param[in] new_value New value to assign if key is found.
     *  @return True if key was found and value updated, false otherwise.
     */
    template <typename comparable_key_type_, typename convertible_value_type_>
    bool update_atomic(comparable_key_type_ &&key, convertible_value_type_ &&new_value) noexcept {
        static_assert(has_values_k, "update_atomic() only available for maps, not sets");
        return find_atomic(std::forward<comparable_key_type_>(key), [&](element_catref_t &addr) {
            // Cast away const for atomic update (safe because we're non-const method)
            const_cast<value_t &>(addr.value()) = std::forward<convertible_value_type_>(new_value);
        });
    }

#pragma mark - Erasures

    /**
     * @brief Removes element at specified location.
     * https://en.cppreference.com/w/cpp/container/unordered_map/erase
     *
     * ! The given input iterator must be valid! It can't be the end!
     * ! If you want to overwrite the freed slot, make sure, that new objects
     * ! hash fits exactly and doesn't corrupt the state of the hash-table.
     */
    void erase(element_rt addr) noexcept {

        // Call destructors
        if constexpr (destruct_keys_k) addr.key_ref().~key_t();
        if constexpr (destruct_vals_k) addr.value_ref().~value_t();

        // Update indicators & stats
        addr.mark_deleted();
        deleted_count_++;
        populated_count_--;
    }

    void erase(iterator_t addr) noexcept { return erase((element_rt)addr); }

    void erase(iterator_ct) = delete;
    iterator_ct erase(iterator_ct, return_position_t) = delete;
    iterator_t erase(iterator_t, return_position_t) = delete;

    /**
     * @brief Removes a value by key in the most efficient fashion.
     * If you already have the iterator, use the faster version, that
     * avoids search entirely!
     * https://en.cppreference.com/w/cpp/container/unordered_map/erase
     *
     * ! Erasures in this containers won't deallocate memory.
     * ! Memory will remain cluttered until the next @c force_resize().
     * So if you remove a lot of values from filled containers,
     * it makes sense to @c rehash() it before doing billions of lookups.
     *
     * @param tags Markers for special acceleration:
     * > assume_reserved_t: Means container contains at least one object.
     * > threadsafe_t: Would allow concurrent read & write operations.
     *
     * @return True if wanted key was found.
     */
    template <typename comparable_key_type_, typename... tags_types_>
    bool erase(comparable_key_type_ &&k, tags_types_... tags) {

        bool result = false;

        search_to_find(
            std::forward<comparable_key_type_>(k),
            // This can't call `erase` on the returned references,
            // as it allows even atomic erasures, which must be done within
            // a block.
            [&](auto &addr) {
                // Call destructors
                if constexpr (destruct_keys_k) addr.key_ref().~key_t();
                if constexpr (destruct_vals_k) addr.value_ref().~value_t();

                // Update indicators & stats
                addr.mark_deleted();
                advance(deleted_count_, 1, tags...);
                decrement(populated_count_, 1, tags...);
                result = true;
            },
            tags...);

        return result;
    }

    /**
     *  @brief Lock-free atomic erasure by key.
     *    Requires prior @c reserve() to prevent reallocations during concurrent access.
     *
     *  @tparam comparable_key_type_ Type comparable with @c key_t via @c equals_t.
     *  @param[in] key Key to erase, hashable and comparable.
     *  @return True if key was found and erased, false if key didn't exist.
     *
     *  @note Uses atomic operations on bucket headers. Destructors are called for non-trivial types.
     *    The slot is marked as deleted (tombstone) rather than freed, preserving probe sequences.
     */
    template <typename comparable_key_type_>
    bool erase_atomic(comparable_key_type_ &&key) noexcept {
        if (!populated_count_) return false;

        offset_t const offset_mask = slots_count_ - 1;
        offset_t const initial_offset = hasher_()(key) & offset_mask;

        element_atref_t addr;
        offset_t slot = initial_offset;
        bool erased = false;

        while (true) {
            unsafe_retarget(addr, slot);
            addr.lock();

            if (addr.is_populated()) {
                if (equals_()(addr.key(), key)) {
                    // Call destructors
                    if constexpr (destruct_keys_k) addr.key_ref().~key_t();
                    if constexpr (destruct_vals_k) addr.value_ref().~value_t();

                    // Mark as deleted
                    addr.mark_deleted();
                    addr.unlock();

                    // Update counters atomically
                    std::atomic_ref<offset_t> atomic_del(deleted_count_);
                    atomic_del.fetch_add(1, std::memory_order_relaxed);
                    std::atomic_ref<offset_t> atomic_pop(populated_count_);
                    atomic_pop.fetch_sub(1, std::memory_order_relaxed);

                    erased = true;
                    break;
                }
                else {
                    addr.unlock();
                    slot = (slot + 1) & offset_mask;
                }
            }
            else if (addr.is_deleted()) {
                addr.unlock();
                slot = (slot + 1) & offset_mask;
            }
            else {
                // Free slot - key doesn't exist
                addr.unlock();
                break;
            }
        }

        return erased;
    }

#pragma mark - Memory Management

    /**
     *  @brief Calculates total memory usage for three cache-aligned regions.
     *    Layout: [keys | values | headers] with padding between regions to prevent false sharing.
     *
     *  @param slots_cnt Total number of slots (must be power of two, multiple of 32).
     *  @return Total bytes needed for single allocation containing all three regions.
     *
     *  @note Keys and values regions are padded to cache-line boundaries (64 bytes) to prevent
     *    false sharing. Headers region is not padded (end of allocation).
     */
    static constexpr std::size_t memory_usage(bht_slots_count_t slots_count) noexcept {
        std::size_t slots = slots_count.raw;
        if (slots == 0) return 0;

        // Region 1: Keys array, cache-line aligned
        std::size_t keys_bytes = roundup_to_multiple<std::size_t, cache_line_bytes_k>(slots * sizeof(key_t));

        // Region 2: Values array, cache-line aligned (skip for sets where value_t is void-like)
        std::size_t values_bytes =
            has_values_k ? roundup_to_multiple<std::size_t, cache_line_bytes_k>(slots * sizeof(value_t)) : 0;

        // Region 3: Headers array, one 64-bit header per 32 slots (no padding needed, end of buffer)
        std::size_t header_bytes = (slots / bht_bucket_capacity_k) * sizeof(bht_bucket_head_t);

        return keys_bytes + values_bytes + header_bytes;
    }

    /**
     * @brief Returns the maximum number of slots that can fit a memory
     * region of given size. Is often used for caches in DB to calculate
     * the capacity of the Hash-Table before allocating one.
     */
    static constexpr bht_slots_count_t slots_fitting_bytes(std::size_t max_bytes) noexcept {
        auto buckets = max_bytes / bytes_in_bucket_k;
        bht_slots_count_t result;
        result.raw = buckets * bht_bucket_capacity_k;
        return result;
    }

    /**
     * @brief Allocates more memory to fit that many elements,
     * if needed. If current capacity is enough, no allocations
     * will happen.
     * https://en.cppreference.com/w/cpp/container/unordered_map/reserve
     *
     * @return True when the underlying capacity has changed.
     * After that the older iterators will be deleted.
     */
    inline bool reserve(offset_t planned_elements) {
        if (planned_elements <= growth_threshold_) return false;
        force_resize(bht_slots_count_t {planned_elements});
        return true;
    }

    /**
     * @brief The recommended function for bulk insertions.
     * Unlike the classical @c reserve(std::size_t), this adds the
     * number of already present objects to avoid calling @c size()
     * and includes the number "free" slots, to keep search time
     * constant.
     *
     * @return True when the underlying capacity has changed.
     * After that the older iterators will be deleted.
     */
    inline bool reserve_more(offset_t new_elements) {
        return reserve(new_elements + populated_count_ + deleted_count_);
    }

    /**
     * @brief Even if the new capacity is identical, this operation
     * will export the data into a new hash-set and then swap the contents.
     * It may be used to clean the garbage after deletions.
     *
     * ! The new capacity can't be less than @c size()!
     */
    void force_resize(bht_slots_count_t slots_cnt) {
        auto resized = move_to_new(slots_cnt);
        swap(resized);
    }

    /**
     * @brief A cheap copy-less swap mechanism used in move-constructors.
     * https://en.cppreference.com/w/cpp/container/unordered_map/swap
     */
    inline void swap(basic_hash_table &other) noexcept {
        std::swap(memory_, other.memory_);
        std::swap(keys_, other.keys_);
        std::swap(values_, other.values_);
        std::swap(headers_, other.headers_);
        std::swap(slots_count_, other.slots_count_);
        std::swap(growth_threshold_, other.growth_threshold_);
        std::swap(populated_count_, other.populated_count_);
        std::swap(deleted_count_, other.deleted_count_);

        if constexpr (!hasher_is_empty_k) std::swap(hasher_(), other.hasher_());
        if constexpr (!equals_is_empty_k) std::swap(equals_(), other.equals_());
        if constexpr (!allocator_is_empty_k) std::swap(allocator_(), other.allocator_());
    }

    /**
     * @brief Erases all elements from the container.
     * After this call, @c size() returns zero.
     * No deallocations will happen, for that call @c shrink_to_fit().
     * https://en.cppreference.com/w/cpp/container/unordered_map/clear
     *
     * ! If you know that container isn't empty pass the @c assume_reserved_t tag.
     */
    void clear() noexcept {
        if (populated_count_ | deleted_count_) clear(assume_reserved_t {});
        else allocator_().clear(memory_, size_bytes());
    }

    /**
     * @brief Erases all elements from the container.
     * After this call, @c size() returns zero.
     * No deallocations will happen, for that call @c shrink_to_fit().
     * https://en.cppreference.com/w/cpp/container/unordered_map/clear
     */
    void clear(assume_reserved_t) noexcept {

        if constexpr (destruct_keys_k || destruct_vals_k) {
            auto const e = end();
            for (auto it = begin(); it != e; it.advance()) {
                if constexpr (destruct_keys_k) it.key_ref().~key_t();
                if constexpr (destruct_vals_k) it.value_ref().~value_t();
            }
        }

        allocator_().clear(memory_, size_bytes());
        deleted_count_ = 0;
        populated_count_ = 0;
    }

    /**
     * @brief Brings the memory consumption of the contain to the minimum.
     * STLs associative containers have no such functionality, only arrays:
     * https://en.cppreference.com/w/cpp/container/vector/shrink_to_fit
     */
    void shrink_to_fit() {
        auto new_cap = bht_slots_count_t {size()};
        if (new_cap == slots_count()) return;
        return size() ? force_resize(new_cap) : deallocate();
    }

    /**
     * @brief Deallocates all the memory used by container.
     * ! Make sure to @c clear() to destruct all the non-trivial elements
     * ! before deallocating the memory. It's left to user.
     */
    [[attr_reinitializes_m]] void deallocate() {
        if (memory_) return deallocate(assume_reserved_t {});
    }

    /**
     * @brief Deallocates all the memory used by container.
     * ! Make sure to @c clear() to destruct all the non-trivial elements
     * ! before deallocating the memory. It's left to user.
     */
    void deallocate(assume_reserved_t) {
        allocator_().deallocate(memory_, size_bytes());
        unsafe_reset();
    }

    inline void unsafe_reset() noexcept {
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
     * @brief Identical to @c unordered_map, sets the number of buckets
     * to the desired value, causing rehashing in the process.
     * https://en.cppreference.com/w/cpp/container/unordered_map/rehash
     */
    inline void rehash(offset_t count_buckets) {
        bht_slots_count_t slots_cnt;
        slots_cnt.raw = count_buckets * bht_bucket_capacity_k;
        force_resize(slots_cnt);
    }

#pragma mark - Copies and Moves

    template <typename>
    struct packed_pairs_array_t {};

    /**
     * @brief A unique function, that reuses the internal buffer
     * of this Hash-Table to pack all the key-value pairs into array.
     * No order guarantees are provided.
     * Extra allocation ALMOST never appear!
     *
     * @note Currently disabled - requires external dbuffer_gt and darray_gt types.
     * TODO: Implement using std::vector or similar standard container.
     */
    // auto convert_to_array() {
    //     // Commented out - requires dbuffer_gt, darray_gt types not in current codebase
    //     return nullptr;
    // }

    template <typename callback_type_ = null_operator_gt<element_rt>>
    void for_each(callback_type_ &&callback) noexcept {
        element_rt addr;
        addr.keys_ = keys_;
        addr.values_ = values_;
        addr.headers_ = headers_;
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
            addr.slot_ = bucket_idx * bht_bucket_capacity_k;
            bht_for_each_in_bucket_(addr, callback);
        }
    }

    template <typename callback_type_ = null_operator_gt<element_crt>>
    void for_each(callback_type_ &&callback) const noexcept {
        element_crt addr;
        addr.keys_ = const_cast<key_t *>(keys_);
        addr.values_ = const_cast<value_t *>(values_);
        addr.headers_ = const_cast<bht_bucket_head_t *>(headers_);
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
            addr.slot_ = bucket_idx * bht_bucket_capacity_k;
            bht_for_each_in_bucket_(addr, callback);
        }
    }

    template <typename callback_type_ = null_operator_gt<element_rt>>
    void for_slots(callback_type_ &&callback) noexcept {
        element_rt addr;
        addr.keys_ = keys_;
        addr.values_ = values_;
        addr.headers_ = headers_;
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
            addr.slot_ = bucket_idx * bht_bucket_capacity_k;
            bht_for_slots_in_bucket(addr, callback);
        }
    }

    template <typename predicate_type_>
    bool find_if(predicate_type_ &&predicate) noexcept {
        element_rt addr;
        addr.keys_ = keys_;
        addr.values_ = values_;
        addr.headers_ = headers_;
        bool found_match = false;
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; (bucket_idx != buckets) & !found_match; ++bucket_idx) {
            addr.slot_ = bucket_idx * bht_bucket_capacity_k;
            found_match = bht_find_in_bucket_(addr, predicate);
        }
        return found_match;
    }

    template <typename predicate_type_>
    bool find_if(predicate_type_ &&predicate) const noexcept {
        element_crt addr;
        addr.keys_ = const_cast<key_t *>(keys_);
        addr.values_ = const_cast<value_t *>(values_);
        addr.headers_ = const_cast<bht_bucket_head_t *>(headers_);
        bool found_match = false;
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; (bucket_idx != buckets) & !found_match; ++bucket_idx) {
            addr.slot_ = bucket_idx * bht_bucket_capacity_k;
            found_match = bht_find_in_bucket_(addr, predicate);
        }
        return found_match;
    }

    /**
     * @brief Creates a new memory buffer of requested capacity
     * and rehashes all the present data into the new buckets.
     * Move-constructors will be used, avoiding copies.
     * TODO: Implement @c move_to_next_size using @c realloc.
     *
     * ! The new capacity can't be less than current @c size()!
     */
    basic_hash_table move_to_new(bht_slots_count_t slots_cnt) {

        basic_hash_table tgt(slots_cnt, hash_function(), key_eq(), allocator_());
        assert(tgt.capacity() >= size() && "Not enough space in the new Hash-Table!");

        for_each([&tgt](element_rt const &src_addr) {
            if constexpr (has_values_k)
                tgt.emplace(std::move(src_addr.key_ref()), std::move(src_addr.value_ref()), assume_reserved_t {},
                            assume_unique_t {});
            else tgt.emplace(std::move(src_addr.key_ref()), assume_reserved_t {}, assume_unique_t {});
        });

        assert(tgt.populated_count_ == populated_count_ && "Element counts must match!");
        deallocate();
        return tgt;
    }

    basic_hash_table copy_to_new() const {

        basic_hash_table const &src = *this;
        basic_hash_table tgt(slots_count(), hash_function(), key_eq(), allocator_());
        tgt.populated_count_ = src.populated_count_;
        tgt.deleted_count_ = src.deleted_count_;

        // If everything is trivially constructible, a single `memcpy` is enough!
        if constexpr (layout_t::will_memcpy_keys() && layout_t::will_memcpy_vals())
            allocator_().copy(src.memory_, tgt.memory_, src.size_bytes());
        else {
            element_crt src_addr;
            src_addr.keys_ = const_cast<key_t *>(src.keys_);
            src_addr.values_ = const_cast<value_t *>(src.values_);
            src_addr.headers_ = const_cast<bht_bucket_head_t *>(src.headers_);
            element_rt tgt_addr;
            tgt_addr.keys_ = tgt.keys_;
            tgt_addr.values_ = tgt.values_;
            tgt_addr.headers_ = tgt.headers_;
            offset_t const buckets = src.bucket_count();

            for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
                src_addr.slot_ = bucket_idx * bht_bucket_capacity_k;
                tgt_addr.slot_ = bucket_idx * bht_bucket_capacity_k;
                bht_transform_bucket(src_addr, tgt_addr, identity_fn_t {});
            }
        }

        return tgt;
    }

    basic_hash_table copy_to_new(bht_slots_count_t slots_cnt) const {
        basic_hash_table const &src = *this;
        basic_hash_table tgt(slots_cnt, hash_function(), key_eq(), allocator_());
        tgt.insert(src.begin(), src.end(), assume_reserved_t {});
        return tgt;
    }

    set_t copy_to_set() const {

        basic_hash_table const &src = *this;
        set_t tgt(src.slots_count(), hash_function(), key_eq(), allocator_());
        tgt.populated_count_ = src.populated_count_;
        tgt.deleted_count_ = src.deleted_count_;

        element_crt src_addr;
        src_addr.keys_ = const_cast<key_t *>(src.keys_);
        src_addr.values_ = const_cast<value_t *>(src.values_);
        src_addr.headers_ = const_cast<bht_bucket_head_t *>(src.headers_);
        typename set_t::element_rt tgt_addr;
        tgt_addr.keys_ = tgt.keys_;
        tgt_addr.values_ = tgt.values_;
        tgt_addr.headers_ = tgt.headers_;
        offset_t const buckets = src.bucket_count();

        for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
            src_addr.slot_ = bucket_idx * bht_bucket_capacity_k;
            tgt_addr.slot_ = bucket_idx * bht_bucket_capacity_k;
            bht_transform_bucket(src_addr, tgt_addr, null_operator_t {});
        }

        return tgt;
    }
};

#pragma mark - Aliases

template <typename key_type_, typename value_type_, typename hasher_type_ = std::hash<key_type_>,
          typename equals_type_ = std::equal_to<key_type_>, typename allocator_type_ = std::allocator<std::byte>>
using hash_map_gt = basic_hash_table<key_type_, value_type_, hasher_type_, equals_type_, allocator_type_>;

template <typename key_type_, typename hasher_type_ = std::hash<key_type_>,
          typename equals_type_ = std::equal_to<key_type_>, typename allocator_type_ = std::allocator<std::byte>>
using hash_set_gt = basic_hash_table<key_type_, void, hasher_type_, equals_type_, allocator_type_>;

/**
 *
 *
 *
 */
static_assert(sizeof(hash_set_gt<int>) >= 3 * sizeof(void *), "Hash-Table is too big!");

} // namespace ashvardanian::smashtable

#pragma GCC diagnostic pop

/**
 * @brief An overload of @c std::swap for faster sorting
 * and STL containers of such Hash-Tables.
 */
namespace std {

template <typename element_type_, typename hasher_type_, typename equals_type_, typename allocator_type_>
inline void swap(
    ashvardanian::smashtable::basic_hash_table<element_type_, hasher_type_, equals_type_, allocator_type_> &a,
    ashvardanian::smashtable::basic_hash_table<element_type_, hasher_type_, equals_type_, allocator_type_> &b) {
    a.swap(b);
}

} // namespace std
