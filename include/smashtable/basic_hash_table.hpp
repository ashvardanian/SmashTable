/**
 *  @brief  Thread-Safe Hash-Table implementation with flat layout and open addressing.
 *          Doesn't raise any exceptions unlike STL-based alternatives, like `std::unordered_map`
 *          and `std::unordered_set`.
 *
 *  @file   basic_hash_table.hpp
 *  @author Ash Vardanian
 *  @see    https://en.wikipedia.org/wiki/Hash_table
 */
#pragma once
#include <cassert>   // `assert`
#include <algorithm> // `std::max`
#include <memory>    // `std::allocator`
#include <optional>  // `std::optional`
#include <random>    // `std::uniform_int_distribution`
#include <utility>   // `std::exchange`

#include "status.hpp"

namespace ashvardanian::smashtable {

/**
 * @brief The length of a bucket, where that many keys are
 * followed by that many values.
 * This must be a power of two, to allow division-by-shifting.
 * This isn't 64, as we need two extra bits of information per
 * element and can't have 128-bit atomic operations on most
 * platforms.
 * Furthermore, picking a number too big would put too much
 * pressure on the stack size, when exporting hash-tables
 * into contiguous state.
 */
inline static constexpr size_t htx_slots_in_bucket_k = 32;
using htx_bucket_mask_t = u32_t;

/**
 * @brief Header for next `htx_slots_in_bucket_k` slots.
 * It contains 2x lanes: population and deletion states.
 * Two bits of indicators for every slot:
 *  - 00: "free" slot
 *  - 01: "deleted" slot
 *  - 10: "populated" slot
 *  - 11: "locked" slot
 */
union htx_bucket_head_t {
    struct {
        htx_bucket_mask_t populations;
        htx_bucket_mask_t deletions;
    };
    u64_t u64;
};

static_assert(sizeof(htx_bucket_head_t) == 2 * sizeof(htx_bucket_mask_t));

/**
 * @brief Scaling schema, that sugests the number of slots for requested
 * volume of useful data. Some of the slots must remain vacant, to avoid
 * endless loops of probing. Currently, we wait until HTs are 75% full.
 */
struct htx_slots_count_t {
    size_t raw = 0;

    inline_m operator size_t() const noexcept { return raw; }
    inline_m explicit constexpr htx_slots_count_t() noexcept {}
    inline_m explicit constexpr htx_slots_count_t(size_t elems) noexcept {
        if (elems == 0) return;
        size_t needed_slots = (elems * 4ul) / 3ul;
        // We calculate the bucket index with AND masks,
        // so it must be a power of two.
        raw = roundup_to_pow2(needed_slots);
        raw = std::max(raw, size_t(htx_slots_in_bucket_k));
    }
};

/**
 * @brief SFINAE to extract metadata about Hash-Tables
 * keys and values. Uses the return type of the hash-function
 * as the offset and counter for slots (`size_t` for `std::hash`).
 * https://en.cppreference.com/w/cpp/utility/hash
 */
template <typename key_at, typename val_at, typename hasher_at>
struct htx_element_resolver_gt {
    using key_t = std::remove_reference_t<key_at>;
    using val_t = std::remove_reference_t<val_at>;
    using element_t = key_value_pair_gt<key_t const &, val_t &>;
    using element_const_t = key_value_pair_gt<key_t const &, val_t const &>;
    using element_copy_t = key_value_pair_gt<key_t, val_t>;
    using hasher_t = hasher_at;
    using offset_t = decltype(hasher_at {}(std::declval<key_t>()));

    inline static constexpr bool_t has_values_k = true;
    inline static constexpr size_t usefull_bytes_in_bucket_k =
        sizeof(htx_bucket_head_t) + (sizeof(key_t) + sizeof(val_t)) * htx_slots_in_bucket_k;
    inline static constexpr size_t bytes_in_bucket_k =
        roundup_to_multiple<size_t, cache_line_bytes_k>(usefull_bytes_in_bucket_k);

    template <typename value_transform_at>
    using transformed_value_gt = decltype(value_transform_at {}(*reinterpret_cast<val_t const *>(NULL)));

    static constexpr bool_t will_memcpy_keys() { return std::is_trivially_copy_constructible<key_t>(); }
    static constexpr bool_t will_memcpy_vals() { return std::is_trivially_copy_constructible<val_t>(); }
};

template <typename key_at, typename hasher_at>
struct htx_element_resolver_gt<key_at, void, hasher_at> {
    using key_t = std::remove_reference_t<key_at>;
    using val_t = void;
    using element_t = key_t;
    using element_const_t = key_t;
    using element_copy_t = key_t;
    using hasher_t = hasher_at;
    using offset_t = decltype(hasher_at {}(std::declval<key_t>()));

    inline static constexpr bool_t has_values_k = false;
    inline static constexpr size_t usefull_bytes_in_bucket_k =
        sizeof(htx_bucket_head_t) + sizeof(key_t) * htx_slots_in_bucket_k;
    inline static constexpr size_t bytes_in_bucket_k =
        roundup_to_multiple<size_t, cache_line_bytes_k>(usefull_bytes_in_bucket_k);

    template <typename value_transform_at>
    using transformed_value_gt = void;

    static constexpr bool_t will_memcpy_keys() { return std::is_trivially_copy_constructible<key_t>(); }
    static constexpr bool_t will_memcpy_vals() { return false; }
};

template <typename key_at, typename hasher_at>
struct htx_element_resolver_gt<key_at, void const, hasher_at>
    : public htx_element_resolver_gt<key_at, void, hasher_at> {};

static_assert(htx_element_resolver_gt<int, void, hash_gt<int>>::will_memcpy_keys());
static_assert(htx_element_resolver_gt<int, int, hash_gt<int>>::will_memcpy_keys());

/**
 * @brief Reference for an element at certain index within a bucket.
 * It has a thread-safe alternative @b `htx_element_atomic_ref_gt`,
 * used for concurrent `tag_atomic_t` operations.
 *
 * @section Further Optimizations.
 * On both x86 and ARM specialized functions exist for faster bit-testing.
 * Those are often unavailable in a form of intrinsics and have a significant
 * development overhead. Furthermore, the main bottleneck
 */
template <typename key_at, typename val_at, typename hasher_at>
struct htx_element_ref_gt {

    using element_resolver_t = htx_element_resolver_gt<key_at, val_at, hasher_at>;
    using key_t = typename element_resolver_t::key_t;
    using val_t = typename element_resolver_t::val_t;
    using element_t = typename element_resolver_t::element_t;
    using element_const_t = typename element_resolver_t::element_const_t;
    using offset_t = typename element_resolver_t::offset_t;
    inline static constexpr size_t has_values_k = element_resolver_t::has_values_k;
    inline static constexpr size_t bytes_in_bucket_k = element_resolver_t::bytes_in_bucket_k;

    byte_t *head_bytes;
    offset_t idx_in_bucket;

    constexpr void lock() noexcept {}
    constexpr void unlock() noexcept {}
    constexpr bool try_lock() noexcept { return true; }

    inline_m htx_bucket_head_t &head_ref() const noexcept {
        return *reinterpret_cast<htx_bucket_head_t *>(__builtin_assume_aligned(head_bytes, cache_line_bytes_k));
    }

    inline_m htx_bucket_mask_t mask_in_bucket() const noexcept {
        return enabled_top_bit<htx_bucket_mask_t>() >> idx_in_bucket;
    }

    inline_m bool is_populated() noexcept { return head_ref().populations & mask_in_bucket(); }
    inline_m bool is_deleted() noexcept { return head_ref().deletions & mask_in_bucket(); }
    inline_m bool is_freed() noexcept { return ~(is_populated() | is_deleted()); }

    inline_m void mark_populated() noexcept {
        head_ref().populations |= mask_in_bucket();
        head_ref().deletions &= ~mask_in_bucket();
    }
    inline_m void mark_deleted() noexcept {
        head_ref().populations &= ~mask_in_bucket();
        head_ref().deletions |= mask_in_bucket();
    }
    inline_m void mark_freed() noexcept {
        head_ref().populations &= ~mask_in_bucket();
        head_ref().deletions &= mask_in_bucket();
    }

    inline_m key_t &key_ref() const noexcept {
        return *reinterpret_cast<key_t *>(head_bytes + sizeof(htx_bucket_head_t) + sizeof(key_t) * idx_in_bucket);
    }

    inline_m decltype(auto) val_ref() const noexcept {
        if constexpr (has_values_k)
            return (val_t &)*reinterpret_cast<val_t *>(head_bytes + sizeof(htx_bucket_head_t) +
                                                       sizeof(key_t) * htx_slots_in_bucket_k +
                                                       sizeof(val_t) * idx_in_bucket);
    }

    inline_m key_t const &key() const noexcept { return key_ref(); }
    inline_m decltype(auto) value() const noexcept { return val_ref(); }

    inline_m element_const_t operator*() const noexcept {
        if constexpr (has_values_k) return element_const_t {key(), value()};
        else
            return key();
    }

    inline_m element_t operator*() noexcept {
        if constexpr (has_values_k) return element_t {key(), val_ref()};
        else
            return key();
    }
};

/**
 * @brief The most efficient intra-bucket iteration variant.
 * https://en.cppreference.com/w/cpp/algorithm/for_each
 *
 * @param addr Address pointing to the first element in the bucket.
 * @param callback Receives the `htx_element_ref_gt` of "populated" slots.
 */
template <typename key_at, typename val_at, typename hasher_at, typename callback_at>
inline_m void htx_for_each_in_bucket(htx_element_ref_gt<key_at, val_at, hasher_at> &addr,
                                     callback_at &&callback) noexcept {

    // Classical iterator won't be as fast as the approach below.
    // for (addr.idx_in_bucket = 0; addr.idx_in_bucket != htx_slots_in_bucket_k; ++addr.idx_in_bucket)
    //     if (addr.head_ref().populations & addr.mask_in_bucket())
    //         callback(addr);

    htx_bucket_mask_t populations_left = addr.head_ref().populations;
    int count_populated = popcount(populations_left);
    while (count_populated) {
        addr.idx_in_bucket = clz(populations_left);
        callback(addr);
        populations_left &= ~addr.mask_in_bucket();
        --count_populated;
    }
}

/**
 * @brief Trivially iterates through all the slots in the bucket,
 * not just the populated ones, but also the free and deleted.
 * https://en.cppreference.com/w/cpp/algorithm/for_each
 *
 * @param addr Address pointing to the first element in the bucket.
 * @param callback Receives the `htx_element_ref_gt` of "populated" slots.
 */
template <typename key_at, typename val_at, typename hasher_at, typename callback_at>
inline_m void htx_for_slots_in_bucket(htx_element_ref_gt<key_at, val_at, hasher_at> &addr,
                                      callback_at &&callback) noexcept {

    for (addr.idx_in_bucket = 0; addr.idx_in_bucket != htx_slots_in_bucket_k; ++addr.idx_in_bucket) callback(addr);
}

/**
 * @brief The most efficient intra-bucket search variant.
 * https://en.cppreference.com/w/cpp/algorithm/find
 *
 * @param addr Address pointing to the first element in the bucket.
 * @param predicate Receives the `htx_element_ref_gt` of "populated" slots
 * and must return a boolean, if the object matches and further iteration
 * isn't needed.
 * @return True if match was found.
 */
template <typename key_at, typename val_at, typename hasher_at, typename predicate_at>
inline_m bool_t htx_find_in_bucket(htx_element_ref_gt<key_at, val_at, hasher_at> &addr,
                                   predicate_at &&predicate) noexcept {

    htx_bucket_mask_t populations_left = addr.head_ref().populations;
    int count_populated = popcount(populations_left);
    bool_t found_match = false;
    while ((count_populated != 0) & !found_match) {
        addr.idx_in_bucket = clz(populations_left);
        found_match = predicate(addr);
        populations_left &= ~addr.mask_in_bucket();
        --count_populated;
    }
    return found_match;
}

/**
 * @brief Maps elements of one bucket into another bucket.
 * In other words, it's a bucket-level `std::transform`.
 * https://en.cppreference.com/w/cpp/algorithm/transform
 *
 * @param value_transform The functor to be applied to values.
 * > With `identity_t` is used for copy-construction.
 *   Will use `memcpy` if elements are trivially copy-constructible.
 * > With `null_operator_t` or other `void` returning function
 *   exports Hash-Maps into Hash-Sets.
 *
 * ! The deleted slots won't be reused, no entries will be rehashed!
 * ! That is done to accelerate the copy-construction!
 */
template <typename key_at, typename val_at, typename transformed_val_at, typename hasher_at,
          typename value_transform_at = identity_t>
inline_m void htx_transform_bucket(htx_element_ref_gt<key_at const, val_at const, hasher_at> &src_addr,
                                   htx_element_ref_gt<key_at, transformed_val_at, hasher_at> &tgt_addr,
                                   value_transform_at &&value_transform = {}) noexcept {

    // Check what can be done with a simple memcpy.
    // using new_val_t = typename element_resolver_t::transformed_value_gt<value_transform_at>;
    using element_resolver_t = htx_element_resolver_gt<key_at const, val_at const, hasher_at>;
    constexpr bool_t outputs_vals_k = !std::is_same<transformed_val_at, void>();
    constexpr bool_t changes_vals_k = std::is_same<value_transform_at, identity_t>();
    constexpr bool_t memcpy_keys_k = element_resolver_t::will_memcpy_keys();
    constexpr bool_t memcpy_vals_k = !changes_vals_k && element_resolver_t::will_memcpy_vals() && outputs_vals_k;
    static_assert(outputs_vals_k <= element_resolver_t::has_values_k,
                  "It's hard to output something that doesn't exist!");

    // Trivially copy what is possible
    std::memcpy(tgt_addr.head_bytes, src_addr.head_bytes, sizeof(htx_bucket_head_t));

    constexpr size_t bytes_in_buckets_keys_k = htx_slots_in_bucket_k * sizeof(key_at);
    if constexpr (memcpy_keys_k)
        std::memcpy(tgt_addr.head_bytes + sizeof(htx_bucket_head_t), src_addr.head_bytes + sizeof(htx_bucket_head_t),
                    bytes_in_buckets_keys_k);

    if constexpr (memcpy_vals_k) {
        constexpr size_t bytes_in_buckets_vals_k = htx_slots_in_bucket_k * sizeof(val_at);
        std::memcpy(tgt_addr.head_bytes + sizeof(htx_bucket_head_t) + bytes_in_buckets_keys_k,
                    src_addr.head_bytes + sizeof(htx_bucket_head_t) + bytes_in_buckets_keys_k, bytes_in_buckets_vals_k);
    }

    // Manually copy the rest
    if constexpr (!memcpy_keys_k || (!memcpy_vals_k && outputs_vals_k)) {
        htx_for_each_in_bucket(src_addr, [&tgt_addr, &value_transform](auto const &src_addr) {
            tgt_addr.idx_in_bucket = src_addr.idx_in_bucket;
            if constexpr (!memcpy_keys_k) new (&tgt_addr.key_ref()) key_at(src_addr.key());
            if constexpr (!memcpy_vals_k && outputs_vals_k) {
                if constexpr (changes_vals_k)
                    new (&tgt_addr.val_ref()) transformed_val_at(value_transform(src_addr.value()));
                else
                    new (&tgt_addr.val_ref()) transformed_val_at(src_addr.value());
            }
        });
    }
}

template <typename key_at, typename val_at, typename hasher_at>
struct htx_element_atomic_ref_gt : public htx_element_ref_gt<key_at, val_at, hasher_at> {

    using base_t = htx_element_ref_gt<key_at, val_at, hasher_at>;
    using base_t::head_bytes;
    using base_t::idx_in_bucket;
    using base_t::key_ref;
    using base_t::mask_in_bucket;
    using base_t::val_ref;
    using base_t::operator*;

    htx_bucket_head_t bilane_mask {0ul};
    htx_bucket_head_t mutable replacement_head {0ul};

    inline_m htx_bucket_head_t &head_ref() const noexcept { return replacement_head; }

    inline_m bool is_populated() noexcept { return head_ref().populations & mask_in_bucket(); }
    inline_m bool is_deleted() noexcept { return head_ref().deletions & mask_in_bucket(); }
    inline_m bool is_freed() noexcept { return ~(is_populated() | is_deleted()); }

    inline_m void mark_populated() noexcept {
        head_ref().populations |= mask_in_bucket();
        head_ref().deletions &= ~mask_in_bucket();
    }
    inline_m void mark_deleted() noexcept {
        head_ref().populations &= ~mask_in_bucket();
        head_ref().deletions |= mask_in_bucket();
    }
    inline_m void mark_freed() noexcept {
        head_ref().populations &= ~mask_in_bucket();
        head_ref().deletions &= mask_in_bucket();
    }

    inline_m void lock() noexcept {
        // The unique aspect of this implementation is that we don't use
        // Compare-And-Swap intrinsics, just an atomic bitwise OR.
        bilane_mask.populations = mask_in_bucket();
        bilane_mask.deletions = mask_in_bucket();
        // Lock the object and make sure it wasn't locked before us:
        // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
    relock:
        replacement_head.u64 =
            __atomic_fetch_or(&base_t::head_ref().u64, bilane_mask.u64, __ATOMIC_ACQUIRE) & bilane_mask.u64;
        if (replacement_head.u64 == bilane_mask.u64) goto relock;
    }

    inline_m void unlock() noexcept {
        // Both bits are set after the lock.
        // We only need to disable one or two of them.
        __atomic_and_fetch(&base_t::head_ref().u64, ~bilane_mask.u64 | replacement_head.u64, __ATOMIC_RELEASE);
    }

    inline_m bool try_lock() noexcept {
        // The unique aspect of this implementation is that we don't use
        // Compare-And-Swap intrinsics, just an atomic bitwise OR.
        bilane_mask.populations = mask_in_bucket();
        bilane_mask.deletions = mask_in_bucket();
        // Lock the object and make sure it wasn't locked before us:
        // https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html
        replacement_head.u64 =
            __atomic_fetch_or(&base_t::head_ref().u64, bilane_mask.u64, __ATOMIC_ACQUIRE) & bilane_mask.u64;
        return replacement_head.u64 != bilane_mask.u64;
    }
};

/* Make sure these objects are lightweight, to avoid extra register pressure. */
static_assert(sizeof(htx_element_ref_gt<int, int, hash_gt<int>>) <= 2 * sizeof(void *));
static_assert(sizeof(htx_element_atomic_ref_gt<int, int, hash_gt<int>>) <= 4 * sizeof(void *));

/**
 * @brief A heavy iterator for Hash-Tables, occupies 3x8 = 24 bytes.
 * To make it a `std::bidirectional_iterator_tag` we would need one more pointer.
 * Instead of keeping the end pointer, we look for `hash_table_end_marker_k`.
 *
 * End state is reached when `head_bytes == end_bytes`.
 * The value of `idx_in_bucket` will be zero.
 *
 * ! This doesn't support atomic iteration.
 */
template <typename key_at, typename val_at, typename hasher_at>
struct htx_iterator_gt : public htx_element_ref_gt<key_at, val_at, hasher_at> {

    using base_t = htx_element_ref_gt<key_at, val_at, hasher_at>;
    using element_t = typename base_t::element_t;
    using offset_t = typename base_t::offset_t;
    using base_t::bytes_in_bucket_k;
    using base_t::head_bytes;
    using base_t::head_ref;
    using base_t::idx_in_bucket;
    using base_t::mask_in_bucket;

    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = element_t;
    using reference = element_t;
    using pointer = void;

    offset_t slots_remaining;

    [[nodiscard]] inline_m bool_t operator==(end_sentinel_t) const noexcept { return is_end(); }
    [[nodiscard]] inline_m bool_t operator!=(end_sentinel_t) const noexcept { return isnt_end(); }
    [[nodiscard]] inline_m bool_t isnt_end() const noexcept { return slots_remaining != 0; }
    [[nodiscard]] inline_m bool_t is_end() const noexcept { return slots_remaining == 0; }

    inline_m bool_t operator==(htx_iterator_gt const &o) const noexcept {
        return (head_bytes == o.head_bytes) & (idx_in_bucket == o.idx_in_bucket);
    }
    inline_m bool_t operator!=(htx_iterator_gt const &o) const noexcept {
        return (head_bytes != o.head_bytes) | (idx_in_bucket != o.idx_in_bucket);
    }

    operators_increment_m(htx_iterator_gt);

    /**
     * @brief Mostly branchless iterator increment.
     * ! Can only be called if `is_end() == false`.
     */
    inline_m void advance() noexcept_in_release_m {
        validate_m(!is_end(), "Going out of range!");
        --slots_remaining;
        idx_in_bucket = (idx_in_bucket + 1) & (htx_slots_in_bucket_k - 1);
        head_bytes += bytes_in_bucket_k * (idx_in_bucket == 0);
        skip_non_populated();
    }

    inline_m void skip_non_populated() noexcept {
        while ((slots_remaining != 0) && (~head_ref().populations & mask_in_bucket())) {
            --slots_remaining;
            idx_in_bucket = (idx_in_bucket + 1) & (htx_slots_in_bucket_k - 1);
            head_bytes += bytes_in_bucket_k * (idx_in_bucket == 0);
        }
    }

    /**
     * @brief DEPRECATED almost-brnachless approach to iteration.
     * Performs worse than brancheing approach. For much faster
     * internal iteration use the `hash_table.for_each(...)`
     * and the underlying `htx_for_each_in_bucket`.
     */
    inline_m void skip_non_populated_branchless(bool_t skip_current = false) noexcept {

        // The trick here is to temporarily forget about updating our `head_bytes`
        // pointer and forget that our `idx_in_bucket` can't be greater than
        // the `htx_slots_in_bucket_k`.
        auto new_idx = idx_in_bucket;
    increment_more:
        auto wrapped_idx_in_bucket = new_idx & (htx_slots_in_bucket_k - 1);
        auto mask_in_bucket = enabled_top_bit<htx_bucket_mask_t>() >> wrapped_idx_in_bucket;

        auto new_wrapped_idx_in_bucket = clz(*reinterpret_cast<htx_bucket_mask_t const *>(
                                                 head_bytes + (new_idx / htx_slots_in_bucket_k) * bytes_in_bucket_k) &
                                             ((mask_in_bucket | (mask_in_bucket - 1)) >> skip_current));

        new_idx += new_wrapped_idx_in_bucket - wrapped_idx_in_bucket;
        skip_current = false;

        if ((new_wrapped_idx_in_bucket == htx_slots_in_bucket_k) & (new_idx != slots_remaining)) goto increment_more;

        // Once we finish pushing the `idx_in_bucket`, we update the head,
        // as well as the `slots_count`.
        head_bytes += (new_idx / htx_slots_in_bucket_k) * bytes_in_bucket_k;
        slots_remaining -= (new_idx / htx_slots_in_bucket_k) * htx_slots_in_bucket_k;
        idx_in_bucket = new_idx & (htx_slots_in_bucket_k - 1);
    }
};

/**
 * @brief Open-Addressing Constant-Probing Hash-Table inspired by `google::dense_hash_map`.
 * A direct competitor of @c `std::unordered_map`: https://en.cppreference.com/w/cpp/container/unordered_map
 *
 * Unlike Google, uses constant "single step" probing, not the linear version.
 * Unlike Google, keeps keys and values separately, for denser packing.
 * Unlike Google, supports deletions without reserved "empty" values.
 * Unlike Google, supports concurrent operations out of the box.
 * Unlike Google, supports `merge`-like fast set operations.
 * Uses only 2 extra bits per key-value pair to indicate the validity of
 * the slots and atomic locks.
 *
 * @section Insert vs Emplace
 * We don't separate insert/update/emplace semantically, as that functionality is
 * rarely used. The only difference is that `insert(...)` receives a pre-constructed
 * key-value-pair, while `emplace` avoid temporary copies and constructs in-place.
 *
 * @section Memory Usage
 * Similar to Google we allows Hash-Tables to be upto 75% full,
 * meaning 1 in 4 slots will be UNUSED even in the best case.
 * It's still generally more efficient, than tree-like heap-allocating
 * containers, but obiously less efficient than `std::vector`. Overall:
 * > In best case scenario 3 in 4 slots will be "populated".
 * > In bad case, rights after growth, only 3 in 8 will be "populated".
 * > In worst case, after a big `reserve` under 1% can be "populated".
 * That's why it is strongly recommended to use `for_each` instead of iterators.
 * @see Low-level iteration via `htx_for_each_in_bucket` and `htx_transform_bucket`.
 *
 * @section Memory Layout
 * Stores all the elements in buckets with @b `htx_slots_in_bucket_k` slots in each.
 * Wihtin each bucket the content order is the following:
 * 1. 64-bit header;
 * 2. 32x keys;
 * 3. 32x values, or no values at all.
 *
 * @section Heterogeneous Lookups
 * When working with very hot data-paths, it's your responsibility to select
 * the optimal data-layout. Do you prefer to store (validities, keys, values) or
 * the denser (validities, {key,value}s)? When the size of the key is equal or
 * greater than the size of the value (like std::string + std::size_t), a set
 * should be used. Otherwise, a map is preferrable.
 *
 * @section Concurrency
 * Every operation, that doesn't require a memory allocation, can be performed
 * atomically, including insertions, lookups and erasures. Same is true for
 * operations on GPUs, making it the only major associative container besides
 * the "cuCollections" by Nvidia.
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
 * https://github.com/NVIDIA/cuCollections
 *
 * @section Further Reading
 * Abseil and Google teams have a number of hash-table implementations.
 * It's strongly recommended to watch the CppCon 2017 talk by Matt Kulukundis:
 * “Designing a Fast, Efficient, Cache-friendly Hash Table, Step by Step”.
 * Also, we have numerous benchmarks comparing associative containers:
 * http://gitlab.unum.am/hpc-lab/associativecontainers
 *
 * @tparam key_at A hashable and equality-comparable key type.
 * @tparam val_at Optional value type. Degenerates to Hash-Set with `void`.
 *
 * @tparam hasher_at In many applications it's dangerous to use
 * the `hash_gt`, which often simply maps integers to themselves.
 * If you need the identity function, just pass `identity_gt`.
 * ! Must be copy-constructible, but not necesserily default-constructible.
 *
 * @tparam equals_at By default the our equality operator is passed.
 * It's different from `std::equals_to`, as supports differing argument
 * types, which is crucial for heterogeneous lookups.
 * ! Must be copy-constructible, but not necesserily default-constructible.
 */
template <typename key_at, typename val_at = void, typename hasher_at = hash_gt<key_at>,
          typename equals_at = equals_gt<>, typename allocator_at = alloc::libc_t>
struct hash_table_gt {

    using element_resolver_t = htx_element_resolver_gt<key_at, val_at, hasher_at>;
    using key_t = typename element_resolver_t::key_t;
    using val_t = typename element_resolver_t::val_t;
    using element_t = typename element_resolver_t::element_t;
    using element_copy_t = typename element_resolver_t::element_copy_t;
    using offset_t = typename element_resolver_t::offset_t;
    inline static constexpr size_t has_values_k = element_resolver_t::has_values_k;
    inline static constexpr size_t bytes_in_bucket_k = element_resolver_t::bytes_in_bucket_k;

    using hasher_t = hasher_at;
    using equals_t = equals_at;
    using allocator_t = allocator_at;
    using set_t = hash_table_gt<key_t, void, hasher_t, equals_t, allocator_t>;
    using iterator_t = htx_iterator_gt<key_t, val_t, hasher_t>;
    using iterator_ct = htx_iterator_gt<key_t, val_t const, hasher_t>;
    using element_rt = htx_element_ref_gt<key_t, val_t, hasher_t>;
    using element_crt = htx_element_ref_gt<key_t const, val_t const, hasher_t>;
    using element_atref_t = htx_element_atomic_ref_gt<key_t, val_t, hasher_t>;
    using element_catref_t = htx_element_atomic_ref_gt<key_t const, val_t const, hasher_t>;

    static_assert(!std::is_reference<key_t>(), "Keys can't be references!");
    static_assert(!std::is_reference<val_t>(), "Values can't be references!");
    /**
     * If the element type isn't trivially constructible, but it's key is at least
     * zero-initializable, we can reduce branching in the most essential search operation.
     */
    inline static constexpr bool_t keys_default_to_zeros_k = is_trivially_zero_constructible<key_t>();
    inline static constexpr bool_t destruct_keys_k = !std::is_trivially_destructible<key_t>();
    inline static constexpr bool_t destruct_vals_k = has_values_k && !std::is_trivially_destructible<val_t>();

    using hash_value_t = decltype(hasher_t {}(key_t {}));
    static_assert(std::is_unsigned<hash_value_t>(), "Hash value must be an unsigned integer, like u32_t or u64_t!");

    /**
     * @brief A return type for the insert function,
     * @c position indicates the inserted or conflicted element, if none exists points to the end.
     * @c inserted true if the given element was inserted or updated.
     *
     * It's identical to the `result_type` in STL docs:
     * https://en.cppreference.com/w/cpp/container/unordered_map
     */
    struct insert_result_t {
        iterator_t position;
        bool_t inserted = false;
    };

    struct search_result_t {
        iterator_t position;
        bool_t exists = false;
    };

    struct search_cresult_t {
        iterator_ct position;
        bool_t exists = false;
    };

    /* STL-compatiability definitions, idetical to that of `std::map`:
     * https://en.cppreference.com/w/cpp/container/unordered_map
     */
    using key_type = key_t;
    using mapped_typed = val_t;
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

    /**
     * @brief A coninuous block of memory virtually split
     * into buckets for each `htx_slots_in_bucket_k` elements.
     */
    byte_t *memory_;
    offset_t slots_count_;
    offset_t growth_threashold_;
    offset_t populated_count_;
    offset_t deleted_count_;

    member_make_lite_m(hasher_t, hasher);
    member_make_lite_m(equals_t, equals);
    member_make_lite_m(allocator_t, allocator);

#pragma region Constructors

    /**
     * @brief Default constructor, avoids any allocations, accepts
     * a pre-constructed hasher functor and the equality operator.
     * https://en.cppreference.com/w/cpp/container/unordered_map/unordered_map
     */
    inline_m hash_table_gt(hasher_t h = {}, equals_t eq = {}, allocator_t alloc = {}) noexcept
        : memory_(nullptr), slots_count_(0), growth_threashold_(0), populated_count_(0), deleted_count_(0) {

        hasher_construct(std::move(h));
        equals_construct(std::move(eq));
        allocator_construct(std::move(alloc));
    }

    /**
     * @brief Most commonly used constructor interface.
     * https://en.cppreference.com/w/cpp/container/unordered_map/unordered_map
     */
    explicit hash_table_gt(offset_t planned_elements, hasher_t h = {}, equals_t eq = {}, allocator_t alloc = {})
        : hash_table_gt(htx_slots_count_t {planned_elements}, std::move(h), std::move(eq), std::move(alloc)) {}

    /**
     * @brief Main constructor.
     * https://en.cppreference.com/w/cpp/container/unordered_map/unordered_map
     */
    noinline_m hash_table_gt(htx_slots_count_t slots, hasher_t h = {}, equals_t eq = {}, allocator_t alloc = {}) {

        hasher_construct(std::move(h));
        equals_construct(std::move(eq));
        allocator_construct(std::move(alloc));

        size_t needed_bytes = memory_usage(slots);
        memory_ = nullptr;
        if (needed_bytes) {
            memory_ = allocator_().allocate(needed_bytes);
            throw_m(memory_, "Allocation has failed!");
            allocator_().clear(memory_, needed_bytes);
        }
        slots_count_ = static_cast<offset_t>(slots.raw);
        populated_count_ = 0;
        deleted_count_ = 0;
        growth_threashold_ = slots_count_ * 3ul / 4ul;
    }

    noinline_m ~hash_table_gt() {
        if (memory_) {
            clear(tag_preallocated_t {});
            deallocate(tag_preallocated_t {});
        }

        hasher_destruct();
        equals_destruct();
        allocator_destruct();
    }

    inline_host_m hash_table_gt(hash_table_gt &&other) noexcept
        : memory_(nullptr), slots_count_(0), growth_threashold_(0), populated_count_(0), deleted_count_(0) {
        allocator_construct(other.allocator_());
        swap(other);
    }

    inline_host_m hash_table_gt(hash_table_gt const &other) noexcept
        : memory_(nullptr), slots_count_(0), growth_threashold_(0), populated_count_(0), deleted_count_(0) {
        allocator_construct(other.allocator_());
        auto copy = other.copy_to_new();
        swap(copy);
    }

    inline_host_m hash_table_gt &operator=(hash_table_gt &&other) noexcept {
        clear();
        swap(other);
        return *this;
    }

    inline_host_m hash_table_gt &operator=(hash_table_gt const &other) noexcept {
        auto copy = other.copy_to_new();
        swap(copy);
        return *this;
    }

    template <typename begin_iterator_at, typename end_iterator_at, typename... tags_at,
              typename std::enable_if<is_iterator<begin_iterator_at>(), int>::type = 0>
    inline_host_m hash_table_gt(begin_iterator_at begin, end_iterator_at end, tags_at... tags)
        : memory_(nullptr), slots_count_(0), growth_threashold_(0), populated_count_(0), deleted_count_(0) {
        insert(begin, end, tags...);
    }

#pragma region Metadata

    inline_m bool_t empty() const noexcept { return !populated_count_; }
    inline_m offset_t size() const noexcept { return populated_count_; }

    inline_m offset_t optimal_capacity() const noexcept { return growth_threashold_; }
    inline_m offset_t capacity() const noexcept { return std::max(optimal_capacity(), size()); }

    inline_m size_t size_bytes() const noexcept { return bucket_count() * bytes_in_bucket_k; }
    inline_m offset_t capacity_bytes() const noexcept { return size_bytes(); }

    inline_m htx_slots_count_t slots_count() const noexcept {
        htx_slots_count_t result;
        result.raw = slots_count_;
        return result;
    }

    /**
     * @brief STL-compatiability function.
     * https://en.cppreference.com/w/cpp/container/unordered_map/bucket_count
     */
    offset_t bucket_count() const noexcept { return slots_count_ / htx_slots_in_bucket_k; }

    /**
     * @brief STL-compatiability function.
     * https://en.cppreference.com/w/cpp/container/unordered_map/max_bucket_count
     */
    offset_t max_bucket_count() const noexcept { return std::numeric_limits<offset_t>::max() / htx_slots_in_bucket_k; }

    /**
     * @brief STL-compatiability function.
     * https://en.cppreference.com/w/cpp/container/unordered_map/hash_function
     */
    decltype(auto) hash_function() const noexcept { return hasher_(); }

    /**
     * @brief STL-compatiability function.
     * https://en.cppreference.com/w/cpp/container/unordered_map/key_eq
     */
    decltype(auto) key_eq() const noexcept { return equals_(); }

    /**
     * @brief STL-compatiability function.
     * https://en.cppreference.com/w/cpp/container/unordered_map/get_allocator
     */
    decltype(auto) get_allocator() const noexcept { return allocator_(); }

    inline_m byte_t *data() const noexcept { return memory_; }
    inline_m byte_t *data_in_bucket(offset_t bucket_idx) const noexcept {
        return memory_ + bytes_in_bucket_k * bucket_idx;
    }

#pragma region Search

    /**
     * @brief Search optimized for `find`: check is value is present.
     * If you have an intention to insert/upsert something, use another func.
     * Unlike the `search_to_insert` or `search_to_upsert`, doesn't
     * track "deleted" slots.
     *
     * @param wanted Hashable and comparable with key object.
     * @param call A callback receiving `element_crt` or
     * an atomic `element_catref_t` to an initialized matching object.
     *
     * @param tags Markers for special acceleration:
     * > tag_preallocated_t: Avoids null checks.
     * > tag_atomic_t: Enables lock-less concurrency.
     */
    template <typename hetero_key_at, typename callback_at, typename... tags_at>
    inline_m void search_to_find(hetero_key_at &&wanted, callback_at &&call, tags_at...) const noexcept {

        using ref_t = std::conditional_t<contains_type<tag_atomic_t, tags_at...>(), element_catref_t, element_crt>;
        if constexpr (!contains_type<tag_preallocated_t, tags_at...>())
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
                    validate_m(off != final_offset, "Poor hash table usage!");

                    if constexpr (contains_type<tag_force_t, tags_at...>()) break;

                    // if (off == final_offset)
                    //     break;
                }
            }
            else if (addr.is_deleted()) {
                // A "deleted" slot.
                addr.unlock();
                off = (off + 1) & (slots_count_ - 1);
                validate_m(off != final_offset, "Poor hash table usage!");

                if constexpr (contains_type<tag_force_t, tags_at...>()) break;

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
     * @brief Search optimized for `find`: check is value is present.
     * If you have an intention to insert/upsert something, use another func.
     * Unlike the `search_to_insert` or `search_to_upsert`, doesn't
     * track "deleted" slots.
     *
     * @param wanted Hashable and comparable with key object.
     * @param call A callback receiving `element_crt` or
     * an atomic `element_catref_t` to an initialized matching object.
     *
     * @param tags Markers for special acceleration:
     * > tag_preallocated_t: Avoids null checks.
     * > tag_atomic_t: Enables lock-less concurrency.
     */
    template <typename hetero_key_at, typename callback_at, typename... tags_at>
    inline_m void search_to_find(hetero_key_at &&wanted, callback_at &&call, tags_at...) noexcept {

        using ref_t = std::conditional_t<contains_type<tag_atomic_t, tags_at...>(), element_atref_t, element_rt>;
        if constexpr (!contains_type<tag_preallocated_t, tags_at...>())
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
                    validate_m(off != final_offset, "Poor hash table usage!");

                    if constexpr (contains_type<tag_force_t, tags_at...>()) break;

                    // if (off == final_offset)
                    //     break;
                }
            }
            else if (addr.is_deleted()) {
                // A "deleted" slot.
                addr.unlock();
                off = (off + 1) & (slots_count_ - 1);
                validate_m(off != final_offset, "Poor hash table usage!");

                if constexpr (contains_type<tag_force_t, tags_at...>()) break;

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

    template <typename... tags_at>
    inline_host_m offset_t advance(offset_t &count, offset_t offset, tags_at...) const noexcept {
        if constexpr (contains_type<tag_atomic_t, tags_at...>()) return atomic_add_fetch(count, offset);
        else
            return count += offset;
    }

    template <typename... tags_at>
    inline_host_m offset_t decrement(offset_t &count, offset_t offset, tags_at...) const noexcept {
        if constexpr (contains_type<tag_atomic_t, tags_at...>()) return atomic_sub_fetch(count, offset);
        else
            return count -= offset;
    }

    /**
     * @brief Search optimized for `insert` of a uniquie key.
     * Unlike `search_to_upsert` avoids potentially expensive
     * equality comparisons, knowing that the incoming key
     * is different from all present members.
     *
     * @param wanted Hashable and comparable with key object.
     * @param call A callback receiving  `element_rt` or
     * an atomic `element_atref_t` to UN-initialized memory,
     * where an element should be built.
     *
     * @param tags Markers for special acceleration:
     * > tag_preallocated_t: Avoids null checks.
     * > tag_atomic_t: Enables lock-less concurrency.
     */
    template <typename hetero_key_at, typename callback_at, typename... tags_at>
    inline_m void search_to_insert(hetero_key_at &&wanted, callback_at &&call, tags_at... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<tag_atomic_t, tags_at...>(), element_atref_t, element_rt>;
        if constexpr (!contains_type<tag_preallocated_t, tags_at...>())
            if (!populated_count_) [[unlikely]]
                return;

        // We pre-increment the counter, before beginning the potentially slow
        // search. It helps avoiding premature cleanup from a different thread.
        get_type_or<tag_pull_new_size_t, black_hole_t>(tags...) = advance(populated_count_, 1, tags...);

        // Hash to determine the ideal slot.
        ref_t addr;
        offset_t off = hasher_()(wanted) & (slots_count_ - 1);
        bool_t did_find_deleted = false;

        // Loop until we find any "deleted" or "free" slot.
        while (true) {
            unsafe_retarget(addr, off);
            addr.lock();

            if ((~addr.head_ref().populations | addr.head_ref().deletions) & addr.mask_in_bucket()) {
                call(addr);

                // Update the cell state before unlocking.
                did_find_deleted = boolify(addr.head_ref().deletions & addr.mask_in_bucket());
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
     * @brief Search optimized for potentially `upsert`s.
     * It's more expensive than `search_to_find` and `search_to_insert`,
     * so pick this method wisely.
     *
     * @param wanted Hashable and comparable with key object.
     * @param call_unused A callback receiving `element_rt` or
     * an atomic `element_atref_t` to UN-initialized memory,
     * where an element should be built.
     * @param call_equal A callback receiving `element_rt` or
     * an atomic `element_atref_t` to initialized matching object.
     *
     * @param tags Markers for special acceleration:
     * > tag_preallocated_t: Avoids null checks.
     * > tag_atomic_t: Enables lock-less concurrency.
     */
    template <typename hetero_key_at, typename callback_unused_at, typename callback_equal_at, typename... tags_at>
    inline_m void search_to_upsert(hetero_key_at &&wanted, callback_unused_at &&call_unused,
                                   callback_equal_at &&call_equal, tags_at... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<tag_atomic_t, tags_at...>(), element_atref_t, element_rt>;
        if constexpr (!contains_type<tag_preallocated_t, tags_at...>())
            if (!populated_count_) [[unlikely]]
                return;

        offset_t const offset_mask = slots_count_ - 1;
        // offset_t const max_attempts = offset_mask / 2;
        offset_t const initial_offset = hasher_()(wanted) & offset_mask;
        [[maybe_unused]] offset_t const final_offset = initial_offset; // (initial_offset + max_attempts) & offset_mask;

        offset_t off = initial_offset;
        bool_t did_find_deleted = false;

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
                    if constexpr (contains_type<tag_force_t, tags_at...>()) {
                        call_unused(tgts[0]);
                        tgts[0].mark_populated();
                        tgts[0].unlock();
                        break;
                    }

                    tgts[0].unlock();
                    off = (off + 1) & (slots_count_ - 1);
                    validate_m(off != final_offset, "Poor hash table usage!");
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
                if constexpr (contains_type<tag_force_t, tags_at...>()) {
                    call_unused(tgts[0]);
                    tgts[0].mark_populated();
                    tgts[0].unlock();

                    // Change this counters afterwards - in a relaxed manner.
                    // Somebody else might be already searching for this slot,
                    // we don't want them to wait :)
                    get_type_or<tag_pull_new_size_t, black_hole_t>(tags...) = advance(populated_count_, 1, tags...);
                    decrement(deleted_count_, true, tags...);
                    break;
                }

                tgts[!did_find_deleted].unlock();
                tgts[1] = tgts[did_find_deleted];

                did_find_deleted = true;
                off = (off + 1) & (slots_count_ - 1);

                validate_m(off != final_offset, "Poor hash table usage!");
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
            get_type_or<tag_pull_new_size_t, black_hole_t>(tags...) = advance(populated_count_, 1, tags...);
            decrement(deleted_count_, did_find_deleted, tags...);
            break;
        }
    }

    template <typename similar_key_at, typename similar_val_at>
    inline_m void unsafe_retarget(htx_element_ref_gt<similar_key_at, similar_val_at, hasher_t> &addr,
                                  offset_t off) const noexcept {

        addr.head_bytes = const_cast<byte_t *>(memory_) + (off / htx_slots_in_bucket_k) * bytes_in_bucket_k;
        addr.idx_in_bucket = off & (htx_slots_in_bucket_k - 1);
    }

#pragma region Lookups

    template <typename address_or_iterator_at>
    inline_m offset_t slots_remaining_after(address_or_iterator_at &&addr) const noexcept {
        return slots_count_ - (addr.head_bytes - memory_) * htx_slots_in_bucket_k / bytes_in_bucket_k -
               addr.idx_in_bucket;
    }

    template <typename hetero_key_at, typename... tags_at>
    inline_m bool_t contains(hetero_key_at &&wanted, tags_at... tags) const noexcept {
        bool_t result = false;
        search_to_find(std::forward<hetero_key_at>(wanted), [&result](element_crt const &) { result = true; }, tags...);
        return result;
    }

    /**
     * @brief Similar to STL, searches for element with the given key.
     * @return End iterator, iff element with such key is missing.
     */
    template <typename hetero_key_at = key_t const &>
    inline_m iterator_ct find(hetero_key_at &&wanted) const noexcept {
        iterator_ct it;
        it.head_bytes = const_cast<byte_t *>(memory_) + size_bytes();
        it.idx_in_bucket = 0;
        it.slots_remaining = 0;
        search_to_find(std::forward<hetero_key_at>(wanted), [&it](element_crt const &addr) {
            it.head_bytes = addr.head_bytes;
            it.idx_in_bucket = addr.idx_in_bucket;
        });
        it.slots_remaining = slots_remaining_after(it);
        return it;
    }

    /**
     * @brief Similar to STL, searches for element with the given key.
     * @return End iterator, iff element with such key is missing.
     */
    template <typename hetero_key_at = key_t const &>
    inline_m iterator_t find(hetero_key_at &&wanted) noexcept {
        iterator_t it;
        it.head_bytes = const_cast<byte_t *>(memory_) + size_bytes();
        it.idx_in_bucket = 0;
        it.slots_remaining = 0;
        search_to_find(std::forward<hetero_key_at>(wanted), [&it](element_rt const &addr) {
            it.head_bytes = addr.head_bytes;
            it.idx_in_bucket = addr.idx_in_bucket;
        });
        it.slots_remaining = slots_remaining_after(it);
        return it;
    }

    /**
     * @brief Returns a reference to the mapped value of the element with key equivalent
     * to key. If no such element exists, an exception is thrown in `DEBUG` builds.
     * https://en.cppreference.com/w/cpp/container/unordered_map/at
     */
    template <typename hetero_key_at = key_t const &>
    inline_m decltype(auto) at(hetero_key_at &&key) noexcept_in_release_m {
        element_rt result;
        result.head_bytes = nullptr;
        search_to_find(std::forward<hetero_key_at>(key), [&result](element_rt const &addr) { result = addr; });
        validate_m(result.head_bytes != nullptr, "The object must be already present");
        return result.val_ref();
    }

    /**
     * @brief Returns a reference to the const mapped value of the element with key equivalent
     * to key. If no such element exists, an exception is thrown in `DEBUG` builds.
     * https://en.cppreference.com/w/cpp/container/unordered_map/at
     */
    template <typename hetero_key_at = key_t const &>
    inline_m decltype(auto) at(hetero_key_at &&key) const noexcept_in_release_m {
        element_crt result;
        result.head_bytes = nullptr;
        search_to_find(std::forward<hetero_key_at>(key), [&result](element_crt const &addr) { result = addr; });
        validate_m(result.head_bytes != nullptr, "The object must be already present");
        return result.val_ref();
    }

    /**
     * @brief A temporarily banned function, that is replaced by @b `at()`.
     * https://en.cppreference.com/w/cpp/container/unordered_map/operator_at
     */
    template <typename hetero_key_at = key_t const &>
    void operator[](hetero_key_at &&) = delete;

    /**
     * @brief Returns the number of elements with key that compares equal to
     * the specified argument key, which is either 1 or 0 since this container
     * does not allow duplicates.
     * https://en.cppreference.com/w/cpp/container/unordered_map/count
     */
    template <typename hetero_key_at = key_t const &>
    size_t count(hetero_key_at &&key) const noexcept {
        return contains(key);
    }

#pragma region Scans

    inline_m iterator_t begin() noexcept {
        iterator_t it;
        if (empty()) {
            it.head_bytes = const_cast<byte_t *>(memory_) + size_bytes();
            it.idx_in_bucket = 0;
            it.slots_remaining = 0;
        }
        else {
            it.head_bytes = const_cast<byte_t *>(memory_);
            it.idx_in_bucket = 0;
            it.slots_remaining = slots_count_;
            it.skip_non_populated();
        }
        return it;
    }

    inline_m iterator_ct cbegin() const noexcept {
        iterator_ct it;
        if (empty()) {
            it.head_bytes = const_cast<byte_t *>(memory_) + size_bytes();
            it.idx_in_bucket = 0;
            it.slots_remaining = 0;
        }
        else {
            it.head_bytes = const_cast<byte_t *>(memory_);
            it.idx_in_bucket = 0;
            it.slots_remaining = slots_count_;
            it.skip_non_populated();
        }
        return it;
    }

    inline_m iterator_ct begin() const noexcept { return cbegin(); }
    inline_m end_sentinel_t end() const noexcept { return end_sentinel_t {}; }
    inline_m end_sentinel_t cend() const noexcept { return end_sentinel_t {}; }

#pragma region Set Algorithms

    /**
     * @brief Similar to `std::set_intersection`, but reports only
     * a boolean checking if there is any collision between any keys.
     * https://en.cppreference.com/w/cpp/algorithm/set_intersection
     *
     * @param small The small Hash-Table from the two.
     * @return bool_t True if any key intersects. False otherwise.
     */
    template <typename val_small_at, typename hasher_small_at, typename equals_small_at, typename allocator_small_at>
    bool_t intersection_exists(hash_table_gt<key_t, val_small_at, hasher_small_at, equals_small_at,
                                             allocator_small_at> const &small) const noexcept {

        auto const &big = *this;
        if (small.empty() | big.empty()) return false;
        if (small.size() > big.size()) return small.intersection_exists(big);

        return small.find_if(
            [&big](auto const &small_addr) { return big.contains(small_addr.key(), tag_preallocated_t {}); });
    }

    /**
     * @brief Similar to `std::set_intersection`, but reports only
     * the number of matching keys between two containers.
     * https://en.cppreference.com/w/cpp/algorithm/set_intersection
     *
     * @param small The small Hash-Table from the two.
     * @return size_t The number of matches.
     */
    template <typename val_small_at, typename hasher_small_at, typename equals_small_at>
    size_t intersection_size(
        hash_table_gt<key_t, val_small_at, hasher_small_at, equals_small_at> const &small) const noexcept {

        auto const &big = *this;
        if (small.empty() | big.empty()) return 0;
        if (small.size() > big.size()) return big.intersection_size(small);

        size_t cnt = 0;
        small.for_each([&](auto const &addr) { cnt += big.contains(addr.key(), tag_preallocated_t {}); });
        return cnt;
    }

    /**
     * @brief Similar to `std::set_union`, but reports only
     * the number of matching keys between two containers.
     * https://en.cppreference.com/w/cpp/algorithm/set_union
     *
     * @param small The small Hash-Table from the two.
     * @return size_t The number of unique elements in two combined Hash-Tables.
     */
    template <typename val_other_at, typename hasher_other_at, typename equals_other_at>
    size_t merge_size(
        hash_table_gt<key_t, val_other_at, hasher_other_at, equals_other_at> const &other) const noexcept {
        return other.size() + size() - intersection_size(other);
    }

#pragma region Insertions

    template <typename convertible_key_at, typename convertible_val_at>
    static constexpr bool_t can_use_map_emplace() {
        return has_values_k && std::is_constructible<key_t, convertible_key_at &&>() &&
               std::is_constructible<val_t, convertible_val_at &&>();
    }

    template <typename convertible_key_at>
    static constexpr bool_t can_use_set_emplace() {
        return !has_values_k && std::is_constructible<key_t, convertible_key_at &&>();
    }

    /**
     * @brief Emplacing with a hint is banned in favor of lower-level
     * function `search_to_upsert`, with more customizable behaviour.
     * https://en.cppreference.com/w/cpp/container/unordered_map/emplace_hint
     */
    template <typename convertible_key_at, typename convertible_val_at, typename... tags_at,
              typename std::enable_if<can_use_map_emplace<convertible_key_at, convertible_val_at>(), int>::type = 0>
    iterator_t emplace_hint(iterator_ct, convertible_key_at &&k, convertible_val_at &&v, tags_at... tags) = delete;

    /**
     * @brief Emplacing with a hint is banned in favor of lower-level
     * function `search_to_upsert`, with more customizable behaviour.
     * https://en.cppreference.com/w/cpp/container/unordered_map/emplace_hint
     */
    template <typename convertible_key_at, typename... tags_at,
              typename std::enable_if<can_use_set_emplace<convertible_key_at>(), int>::type = 0>
    iterator_t emplace_hint(iterator_ct, convertible_key_at &&k, tags_at... tags) = delete;

    template <typename... tags_at>
    static constexpr bool_t emplace_returns() {
        return contains_type<tag_needs_result_t, tags_at...>();
    }

    template <typename... tags_at>
    using emplace_return_gt = std::conditional_t<emplace_returns<tags_at...>(), insert_result_t, void>;

    /**
     * @brief Main insertion method, that both finds the optimal location and
     * overwrites it. More performant than `insert`, as avoids temporary objects
     * construction. Compatiable with STL code.
     * https://en.cppreference.com/w/cpp/container/unordered_map/emplace
     *
     * @param tags Markers for special acceleration:
     * > tag_preallocated_t: Would avoid reserving more memory.
     * > tag_unique_element_t: Would avoid potentially expensive equality comparisons.
     * > tag_needs_result_t: Avoids calculating the resulting iterator offsets.
     * > tag_atomic_t: Would allow concurrent read & write operations.
     * > tag_pull_new_size_t: Atomically exports the updated `size()`.
     *
     * @return Returns nothing unless the `needs_result_tag_t` is provided.
     * In latter case `insert_result_t` containing the iterator will be constructed.
     */
    template <typename convertible_key_at, typename convertible_val_at, typename... tags_at,
              typename std::enable_if<can_use_map_emplace<convertible_key_at, convertible_val_at>(), int>::type = 0>
    inline_m emplace_return_gt<tags_at...> emplace(convertible_key_at &&key, convertible_val_at &&value,
                                                   tags_at... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<tag_atomic_t, tags_at...>(), element_atref_t, element_rt>;
        if constexpr (contains_type<tag_atomic_t, tags_at...>()) {
            static_assert(contains_type<tag_preallocated_t, tags_at...>(),
                          "Atomic operations can't cause reallocations!");
            static_assert(!contains_type<tag_needs_result_t, tags_at...>(),
                          "Atomic operations can't return a persistent iterator!");
        }

        if constexpr (!contains_type<tag_preallocated_t, tags_at...>()) reserve_more(1);

        constexpr bool_t return_k = emplace_returns<tags_at...>();
        using result_t = std::conditional_t<return_k, insert_result_t, dummy_t>;

        result_t result;
        if constexpr (return_k) {
            result.position.head_bytes = memory_;
            result.position.idx_in_bucket = 0;
            result.position.slots_remaining = 0;
        }

        auto callback_unused = [&](ref_t &addr_unused) {
            // Construct inplace:
            new (&addr_unused.key_ref()) key_t(std::forward<convertible_key_at>(key));
            new (&addr_unused.val_ref()) val_t(std::forward<convertible_val_at>(value));

            // Export results:
            if constexpr (emplace_returns<tags_at...>()) {
                result.position.head_bytes = addr_unused.head_bytes;
                result.position.idx_in_bucket = addr_unused.idx_in_bucket;
                result.inserted = true;
            }
        };

        if constexpr (contains_type<tag_unique_element_t, tags_at...>())
            search_to_insert(key, callback_unused, tag_preallocated_t {}, tags...);
        else
            search_to_upsert(
                key, callback_unused,
                [&](ref_t &addr_equal) {
                    addr_equal.val_ref() = std::forward<convertible_val_at>(value);
                    if constexpr (emplace_returns<tags_at...>()) {
                        result.position.head_bytes = addr_equal.head_bytes;
                        result.position.idx_in_bucket = addr_equal.idx_in_bucket;
                    }
                },
                tag_preallocated_t {}, tags...);

        if constexpr (return_k) {
            result.position.slots_remaining = slots_remaining_after(result.position);
            return result;
        }
    }

    /**
     * @brief Main insertion method, that both finds the optimal location and
     * overwrites it. More performant than `insert`, as avoids temporary objects
     * construction. Compatiable with STL code.
     * https://en.cppreference.com/w/cpp/container/unordered_map/emplace
     *
     * @param tags Markers for special acceleration:
     * > tag_preallocated_t: Would avoid reserving more memory.
     * > tag_unique_element_t: Would avoid potentially expensive equality comparisons.
     * > tag_needs_result_t: Avoids calculating the resulting iterator offsets.
     * > tag_atomic_t: Would allow concurrent read & write operations.
     * > tag_pull_new_size_t: Atomically exports the updated `size()`.
     *
     * @return Returns nothing unless the `needs_result_tag_t` is provided.
     * In latter case `insert_result_t` containing the iterator will be constructed.
     */
    template <typename convertible_key_at, typename... tags_at,
              typename std::enable_if<can_use_set_emplace<convertible_key_at>(), int>::type = 0>
    inline_m emplace_return_gt<tags_at...> emplace(convertible_key_at &&key, tags_at... tags) noexcept {

        using ref_t = std::conditional_t<contains_type<tag_atomic_t, tags_at...>(), element_atref_t, element_rt>;
        if constexpr (!contains_type<tag_preallocated_t, tags_at...>()) reserve_more(1);

        constexpr bool_t return_k = emplace_returns<tags_at...>();
        using result_t = std::conditional_t<return_k, insert_result_t, dummy_t>;

        result_t result;
        if constexpr (return_k) {
            result.position.head_bytes = memory_;
            result.position.idx_in_bucket = 0;
            result.position.slots_remaining = 0;
        }

        auto callback_unused = [&](ref_t &addr_unused) {
            // Construct inplace:
            new (&addr_unused.key_ref()) key_t(std::forward<convertible_key_at>(key));

            // Export results:
            if constexpr (emplace_returns<tags_at...>()) {
                result.position.head_bytes = addr_unused.head_bytes;
                result.position.idx_in_bucket = addr_unused.idx_in_bucket;
                result.inserted = true;
            }
        };

        if constexpr (contains_type<tag_unique_element_t, tags_at...>())
            search_to_insert(key, callback_unused, tag_preallocated_t {}, tags...);
        else if constexpr (return_k)
            search_to_upsert(
                key, callback_unused,
                [&](ref_t &addr_equal) {
                    result.position.head_bytes = addr_equal.head_bytes;
                    result.position.idx_in_bucket = addr_equal.idx_in_bucket;
                },
                tag_preallocated_t {}, tags...);
        else
            search_to_upsert(key, callback_unused, null_operator_t {}, tag_preallocated_t {}, tags...);

        if constexpr (return_k) {
            result.position.slots_remaining = slots_remaining_after(result.position);
            return result;
        }
    }

    /**
     * @brief Main insertion and upsertion method. Compatiable with STL code.
     * https://en.cppreference.com/w/cpp/container/unordered_map/insert
     *
     * ! Don't try to use it with `std::initializer_list`s. Emplace!
     * With maps, this will only work on objects that have `first` and `second`
     * members, like `std::pair` or `key_value_pair_gt`.
     *
     * @param tags Markers for special acceleration:
     * > tag_preallocated_t: Would avoid reserving more memory.
     * > tag_unique_element_t: Would avoid potentially expensive equality comparisons.
     * > tag_needs_result_t: Avoids calculating the resulting iterator offsets.
     * > tag_atomic_t: Would allow concurrent read & write operations.
     * > tag_pull_new_size_t: Atomically exports the updated `size()`.
     *
     * @return Returns nothing unless the `tag_needs_result_t` is provided.
     * In latter case `insert_result_t` containing the iterator will be constructed.
     */
    template <typename temporary_pack_at, typename... tags_at>
    inline_m emplace_return_gt<tags_at...> insert(temporary_pack_at &&copy, tags_at... tags) noexcept {
        if constexpr (has_values_k)
            if constexpr (std::is_rvalue_reference<decltype(copy)>())
                return emplace(std::move(copy.first), std::move(copy.second), tags...);
            else
                return emplace(copy.first, copy.second, tags...);
        else
            return emplace(std::forward<temporary_pack_at>(copy), tags...);
    }

    /**
     * @brief Inserts elements from range [begin, end).
     * If multiple elements in the range have keys that
     * compare equivalent, only the first element is inserted.
     * https://en.cppreference.com/w/cpp/container/unordered_map/insert
     *
     * @param tags Markers for special acceleration:
     * > tag_preallocated_t: Would avoid reserving more memory.
     * > tag_unique_element_t: Would avoid potentially expensive equality comparisons.
     * > tag_atomic_t: Would allow concurrent read & write operations.
     * > tag_pull_new_size_t: Atomically exports the updated `size()`.
     *
     * @return Nothing, just like STL!
     */
    template <typename begin_iterator_at, typename end_iterator_at, typename... tags_at,
              typename std::enable_if<is_iterator<begin_iterator_at>(), int>::type = 0>
    inline_host_m void insert(begin_iterator_at begin, end_iterator_at end, tags_at... tags) {

        if constexpr (!contains_type<tag_preallocated_t, tags_at...>()) reserve_more(end - begin);

        for (; begin != end; ++begin) insert(*begin, tag_preallocated_t {}, tags...);
    }

    /**
     * @brief An interface similar to `std::unordered_map::merge`,
     * that is currenly banned in favor of a manual `insert(begin, end)`,
     * preceded by something line `intersection_size(...)`.
     * https://en.cppreference.com/w/cpp/container/unordered_map/merge
     */
    template <typename other_at = take_second_t>
    void merge(other_at &&) = delete;

#pragma region Erasures

    /**
     * @brief Removes element at specified location.
     * https://en.cppreference.com/w/cpp/container/unordered_map/erase
     *
     * ! The given input iterator must be valid! It can't be the end!
     * ! If you want to overwrite the freed slot, make sure, that new objects
     * ! hash fits exactly and doesn't corrupt the state of the hash-table.
     */
    inline_m void erase(element_rt addr) noexcept {

        // Call destructors
        if constexpr (destruct_keys_k) addr.key_ref().~key_t();
        if constexpr (destruct_vals_k) addr.val_ref().~val_t();

        // Update indicators & stats
        addr.mark_deleted();
        deleted_count_++;
        populated_count_--;
    }

    inline_m void erase(iterator_t addr) noexcept { return erase((element_rt)addr); }

    void erase(iterator_ct) = delete;
    iterator_ct erase(iterator_ct, tag_needs_result_t) = delete;
    iterator_t erase(iterator_t, tag_needs_result_t) = delete;

    /**
     * @brief Removes a value by key in the most efficient fashion.
     * If you already have the iterator, use the faster version, that
     * avoids search entirely!
     * https://en.cppreference.com/w/cpp/container/unordered_map/erase
     *
     * ! Erasures in this containers won't deallocate memory.
     * ! Memory will remain cluttered until the next `force_resize()`.
     * So if you remove a lot of values from filled containers,
     * it makes sense to `rehash()` it before doing billions of lookups.
     *
     * @param tags Markers for special acceleration:
     * > tag_preallocated_t: Means container contains at least one object.
     * > tag_atomic_t: Would allow concurrent read & write operations.
     *
     * @return True if wanted key was found.
     */
    template <typename hetero_key_at, typename... tags_at>
    inline_m bool_t erase(hetero_key_at &&k, tags_at... tags) {

        bool_t result = false;

        search_to_find(
            std::forward<hetero_key_at>(k),
            // This can't call `erase` on the returned references,
            // as it allows even atomic erasures, which must be done within
            // a block.
            [&](auto &addr) {
                // Call destructors
                if constexpr (destruct_keys_k) addr.key_ref().~key_t();
                if constexpr (destruct_vals_k) addr.val_ref().~val_t();

                // Update indicators & stats
                addr.mark_deleted();
                advance(deleted_count_, 1, tags...);
                decrement(populated_count_, 1, tags...);
                result = true;
            },
            tags...);

        return result;
    }

#pragma region Memory Management

    /**
     * @brief Memory usage calculator for Hash-Tables.
     * We align bucket size to cache-line size, to avoid split loads.
     */
    static constexpr size_t memory_usage(htx_slots_count_t slots_cnt) noexcept {
        auto buckets = size_t(slots_cnt) / htx_slots_in_bucket_k;
        return buckets * bytes_in_bucket_k;
    }

    /**
     * @brief Returns the maximum number of slots that can fit a memory
     * region of given size. Is often used for caches in DB to calculate
     * the capacity of the Hash-Table before allocating one.
     */
    static constexpr htx_slots_count_t slots_fitting_bytes(size_t max_bytes) noexcept {
        auto buckets = max_bytes / bytes_in_bucket_k;
        htx_slots_count_t result;
        result.raw = buckets * htx_slots_in_bucket_k;
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
    inline_host_m bool_t reserve(offset_t planned_elements) {
        if (planned_elements <= growth_threashold_) return false;
        force_resize(htx_slots_count_t {planned_elements});
        return true;
    }

    /**
     * @brief The recommended function for bulk insertions.
     * Unlike the classical `reserve(size_t)`, this adds the
     * number of already present objects to avoid calling `size()`
     * and includes the number "free" slots, to keep search time
     * constant.
     *
     * @return True when the underlying capacity has changed.
     * After that the older iterators will be deleted.
     */
    inline_host_m bool_t reserve_more(offset_t new_elements) {
        return reserve(new_elements + populated_count_ + deleted_count_);
    }

    /**
     * @brief Even if the new capacity is identical, this operation
     * will export the data into a new hash-set and then swap the contents.
     * It may be used to clean the garbage after deletions.
     *
     * ! The new capacity can't be less than `size()`!
     */
    void force_resize(htx_slots_count_t slots_cnt) {
        auto resized = move_to_new(slots_cnt);
        swap(resized);
    }

    /**
     * @brief A cheap copy-less swap mechanism used in move-constructors.
     * https://en.cppreference.com/w/cpp/container/unordered_map/swap
     */
    inline_host_m void swap(hash_table_gt &other) noexcept {
        std::swap(memory_, other.memory_);
        std::swap(slots_count_, other.slots_count_);
        std::swap(growth_threashold_, other.growth_threashold_);
        std::swap(populated_count_, other.populated_count_);
        std::swap(deleted_count_, other.deleted_count_);

        if constexpr (!hasher_is_empty_k) std::swap(hasher_(), other.hasher_());
        if constexpr (!equals_is_empty_k) std::swap(equals_(), other.equals_());
        if constexpr (!allocator_is_empty_k) std::swap(allocator_(), other.allocator_());
    }

    /**
     * @brief Erases all elements from the container.
     * After this call, `size()` returns zero.
     * No deallocations will happen, for that call `shrink_to_fit()`.
     * https://en.cppreference.com/w/cpp/container/unordered_map/clear
     *
     * ! If you know that container isn't empty pass the `tag_preallocated_t` tag.
     */
    void clear() noexcept {
        if (populated_count_ | deleted_count_) clear(tag_preallocated_t {});
        else
            allocator_().clear(memory_, size_bytes());
    }

    /**
     * @brief Erases all elements from the container.
     * After this call, `size()` returns zero.
     * No deallocations will happen, for that call `shrink_to_fit()`.
     * https://en.cppreference.com/w/cpp/container/unordered_map/clear
     */
    noinline_m void clear(tag_preallocated_t) noexcept {

        if constexpr (destruct_keys_k || destruct_vals_k) {
            auto const e = end();
            for (auto it = begin(); it != e; it.advance()) {
                if constexpr (destruct_keys_k) it.key_ref().~key_t();
                if constexpr (destruct_vals_k) it.val_ref().~val_t();
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
    noinline_m void shrink_to_fit() {
        auto new_cap = htx_slots_count_t {size()};
        if (new_cap == slots_count()) return;
        return size() ? force_resize(new_cap) : deallocate();
    }

    /**
     * @brief Deallocates all the memory used by container.
     * ! Make sure to `clear()` to destruct all the non-trivial elements
     * ! before deallocating the memory. It's left to user.
     */
    [[attr_reinitializes_m]] noinline_m void deallocate() {
        if (memory_) return deallocate(tag_preallocated_t {});
    }

    /**
     * @brief Deallocates all the memory used by container.
     * ! Make sure to `clear()` to destruct all the non-trivial elements
     * ! before deallocating the memory. It's left to user.
     */
    noinline_m void deallocate(tag_preallocated_t) {
        allocator_().deallocate(memory_, size_bytes());
        unsafe_reset();
    }

    inline_host_m void unsafe_reset() noexcept {
        memory_ = nullptr;
        slots_count_ = 0;
        growth_threashold_ = 0;
        populated_count_ = 0;
        deleted_count_ = 0;
    }

    /**
     * @brief Identical to `unordered_map`, sets the number of buckets
     * to the desired value, causing rehashing in the process.
     * https://en.cppreference.com/w/cpp/container/unordered_map/rehash
     */
    inline_host_m void rehash(offset_t count_buckets) {
        htx_slots_count_t slots_cnt;
        slots_cnt.raw = count_buckets * htx_slots_in_bucket_k;
        force_resize(slots_cnt);
    }

#pragma region Copies and Moves

    template <typename>
    struct packed_pairs_array_t {};

    /**
     * @brief A unique function, that reuses the internal `dbuffer_t`
     * of this Hash-Table to pack all the key-value pairs into array.
     * No order guarantees are provided.
     * Extra allocation ALMOST never appear!
     */
    noinline_m auto convert_to_array() {

        if constexpr (!has_values_k && element_resolver_t::will_memcpy_keys()) {
            byte_t *left_ptr = memory_;
            if constexpr (sizeof(element_copy_t) > sizeof(htx_bucket_head_t)) {
                element_crt bucket_addr;
                byte_t tmp_element[sizeof(element_copy_t)];
                for (offset_t bucket_idx = 0; bucket_idx != bucket_count(); ++bucket_idx) {
                    bucket_addr.head_bytes = data_in_bucket(bucket_idx);
                    size_t const free_distance = bucket_addr.head_bytes - left_ptr;
                    if (free_distance >= sizeof(htx_bucket_head_t)) break;

                    htx_for_each_in_bucket(bucket_addr, [&left_ptr, &tmp_element](element_crt element) {
                        memcpy(tmp_element, &element.key(), sizeof(element_copy_t));
                        memcpy(left_ptr, tmp_element, sizeof(element_copy_t));
                        left_ptr += sizeof(element_copy_t);
                    });
                }
            }
            for_each([&left_ptr](element_rt element) {
                memcpy(left_ptr, &element.key(), sizeof(element_copy_t));
                left_ptr += sizeof(element_copy_t);
            });

            size_t elements_count = (left_ptr - memory_) / sizeof(element_copy_t);
            validate_m(elements_count == size(), "Invalid hash table to array convertions!");

            constexpr size_t align_k = std::alignment_of_v<element_copy_t>;
            using alloc_t = typename allocator_t::template rebind<align_k>::type;
            using buffer_t = dbuffer_gt<alloc_t>;
            using array_t = darray_gt<element_copy_t, alloc_t>;
            buffer_t squeezed_buffer = buffer_t::preconstructed({memory_, size_bytes()});
            array_t squeezed_array = array_t::preconstructed(std::move(squeezed_buffer), elements_count);

            unsafe_reset();
            return squeezed_array;
        }
        else {
            // TODO: Need to implement.
            return nullptr;
        }
    }

    template <typename callback_at = null_operator_gt<element_rt>>
    noinline_m void for_each(callback_at &&callback) noexcept {
        element_rt addr;
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
            addr.head_bytes = data_in_bucket(bucket_idx);
            htx_for_each_in_bucket(addr, callback);
        }
    }

    template <typename callback_at = null_operator_gt<element_crt>>
    noinline_m void for_each(callback_at &&callback) const noexcept {
        element_crt addr;
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
            addr.head_bytes = data_in_bucket(bucket_idx);
            htx_for_each_in_bucket(addr, callback);
        }
    }

    template <typename callback_at = null_operator_gt<element_rt>>
    noinline_m void for_slots(callback_at &&callback) noexcept {
        element_rt addr;
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
            addr.head_bytes = data_in_bucket(bucket_idx);
            htx_for_slots_in_bucket(addr, callback);
        }
    }

    template <typename predicate_at>
    noinline_m bool_t find_if(predicate_at &&predicate) noexcept {
        element_rt addr;
        bool_t found_match = false;
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; (bucket_idx != buckets) & !found_match; ++bucket_idx) {
            addr.head_bytes = data_in_bucket(bucket_idx);
            found_match = htx_find_in_bucket(addr, predicate);
        }
        return found_match;
    }

    template <typename predicate_at>
    noinline_m bool_t find_if(predicate_at &&predicate) const noexcept {
        element_crt addr;
        bool_t found_match = false;
        offset_t const buckets = bucket_count();
        for (offset_t bucket_idx = 0; (bucket_idx != buckets) & !found_match; ++bucket_idx) {
            addr.head_bytes = data_in_bucket(bucket_idx);
            found_match = htx_find_in_bucket(addr, predicate);
        }
        return found_match;
    }

    /**
     * @brief Creates a new memory buffer of requested capacity
     * and rehashes all the present data into the new buckets.
     * Move-constructors will be used, avoiding copies.
     * TODO: Implement `move_to_next_size` using `realloc`.
     *
     * ! The new capacity can't be less than current `size()`!
     */
    hash_table_gt move_to_new(htx_slots_count_t slots_cnt) {

        hash_table_gt tgt(slots_cnt, hash_function(), key_eq(), allocator_());
        validate_m(tgt.capacity() >= size(), "Not enough space in the new Hash-Table!");

        for_each([&tgt](element_rt const &src_addr) {
            if constexpr (has_values_k)
                tgt.emplace(std::move(src_addr.key_ref()), std::move(src_addr.val_ref()), tag_preallocated_t {},
                            tag_unique_element_t {});
            else
                tgt.emplace(std::move(src_addr.key_ref()), tag_preallocated_t {}, tag_unique_element_t {});
        });

        validate_m(tgt.populated_count_ == populated_count_, "Element counts must match!");
        deallocate();
        return tgt;
    }

    hash_table_gt copy_to_new() const {

        hash_table_gt const &src = *this;
        hash_table_gt tgt(slots_count(), hash_function(), key_eq(), allocator_());
        tgt.populated_count_ = src.populated_count_;
        tgt.deleted_count_ = src.deleted_count_;

        // If everything is trvially constructible, a single `memcpy` is enough!
        if constexpr (element_resolver_t::will_memcpy_keys() && element_resolver_t::will_memcpy_vals())
            allocator_().copy(src.memory_, tgt.memory_, src.size_bytes());
        else {
            element_crt src_addr;
            element_rt tgt_addr;
            offset_t const buckets = src.bucket_count();

            for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
                src_addr.head_bytes = src.data_in_bucket(bucket_idx);
                tgt_addr.head_bytes = tgt.data_in_bucket(bucket_idx);
                htx_transform_bucket(src_addr, tgt_addr, identity_t {});
            }
        }

        return tgt;
    }

    hash_table_gt copy_to_new(htx_slots_count_t slots_cnt) const {
        hash_table_gt const &src = *this;
        hash_table_gt tgt(slots_cnt, hash_function(), key_eq(), allocator_());
        tgt.insert(src.begin(), src.end(), tag_preallocated_t {});
        return tgt;
    }

    set_t copy_to_set() const {

        hash_table_gt const &src = *this;
        set_t tgt(src.slots_count(), hash_function(), key_eq(), allocator_());
        tgt.populated_count_ = src.populated_count_;
        tgt.deleted_count_ = src.deleted_count_;

        element_crt src_addr;
        typename set_t::element_rt tgt_addr;
        offset_t const buckets = src.bucket_count();

        for (offset_t bucket_idx = 0; bucket_idx != buckets; ++bucket_idx) {
            src_addr.head_bytes = src.data_in_bucket(bucket_idx);
            tgt_addr.head_bytes = tgt.data_in_bucket(bucket_idx);
            htx_transform_bucket(src_addr, tgt_addr, null_operator_t {});
        }

        return tgt;
    }
};

#pragma mark - Aliases

template <typename key_at, typename val_at, typename hasher_at = hash_gt<key_at>, typename equals_at = equals_gt<>,
          typename allocator_at = alloc::libc_t>
using hash_map_gt = hash_table_gt<key_at, val_at, hasher_at, equals_at, allocator_at>;

template <typename key_at, typename hasher_at = hash_gt<key_at>, typename equals_at = equals_gt<>,
          typename allocator_at = alloc::libc_t>
using hash_set_gt = hash_table_gt<key_at, void, hasher_at, equals_at, allocator_at>;

/**
 *
 *
 *
 */
static_assert(sizeof(hash_set_gt<int>) >= 3 * sizeof(void *), "Hash-Table is too big!");

} // namespace ashvardanian::smashtable

#pragma GCC diagnostic pop

/**
 * @brief An overload of `std::swap` for faster sorting
 * and STL containers of such Hash-Tables.
 */
template <typename key_at, typename val_at, typename hasher_at, typename equals_at>
inline_host_m void std::swap(unum::hash_table_gt<key_at, val_at, hasher_at, equals_at> &a,
                             unum::hash_table_gt<key_at, val_at, hasher_at, equals_at> &b) {
    a.swap(b);
}

#pragma mark - Serialization
#include "primitive/sfinae/serializer.hpp"

namespace unum::prim::sfinae {

/**
 * @brief Hash-Tables serialized, that packs and reconstructs objects
 * without any gaps. Reads/writes one key-value pair at once.
 */
template <typename key_at, typename val_at, typename hasher_at, typename equals_at, typename at>
struct serializer_gt<hash_table_gt<key_at, val_at, hasher_at, equals_at>, at> {

    using container_t = hash_table_gt<key_at, val_at, hasher_at, equals_at>;
    using element_copy_t = typename container_t::element_copy_t;

    inline_host_m size_t serialized_bytes(container_t const &cont) const noexcept {

        size_t elements_length = 0;
        if constexpr (!std::is_trivially_copy_constructible<element_copy_t>()) {
            for (auto const &element : cont) elements_length += sfinae::serialized_bytes(element);
        }
        else
            elements_length = sizeof(element_copy_t) * cont.size();

        return sizeof(u64_t) + elements_length;
    }

    inline_host_m size_t save(container_t const &cont, span_bytes_t bytes) const {
        auto start_point = bytes.data();
        auto size = static_cast<u64_t>(cont.size());
        bytes.first_ptr_ += sfinae::save(size, bytes);

        for (auto const &element : cont) bytes.first_ptr_ += sfinae::save(element, bytes);
        return bytes.data() - start_point;
    }

    inline_host_m size_t load(spanc_bytes_t bytes, container_t &cont) const {
        auto start_point = bytes.data();

        u64_t size = 0;
        bytes.first_ptr_ += sfinae::load(bytes, size);
        cont.reserve(size);

        element_copy_t element;
        for (u64_t i = 0; i != size; i++) {
            bytes.first_ptr_ += sfinae::load(bytes, element);
            if constexpr (container_t::has_values_k)
                cont.emplace(element.first, element.second, tag_preallocated_t {}, tag_unique_element_t {});
            else
                cont.emplace(element, tag_preallocated_t {}, tag_unique_element_t {});
        }

        return bytes.data() - start_point;
    }
};

} // namespace unum::prim::sfinae