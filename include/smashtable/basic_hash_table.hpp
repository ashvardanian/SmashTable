/**
 *  @brief Growable open-addressing hash table optimized for CPU and GPU workloads.
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
 *  - Uses only 2 bits of metadata per slot (vs 1 byte in Google's SwissTable)
 *  - Organizes data in Structure-of-Arrays layout for better cache efficiency
 *  - Being SIMD-friendly and GPU-friendly with minimal modifications
 *
 *  @section basic_hash_table_core_design_goals Core Design Goals
 *
 *  @b Growth: this table grows, and growing repoints the regions every live slot reference has
 *  cached, so it is single-threaded by construction. Handing @c release() to a pinned table is how
 *  the same allocation becomes concurrently readable, and @c adopt takes it back for compaction.
 *
 *  @b Compactness: Each slot requires only 2 bits of overhead stored in a shared bucket header.
 *  This is 4× more space-efficient than SwissTable's 1 byte per slot. We colocate 32× such slots,
 *  cause single- and dual-bit atomic operations aren't possible, but 64-bit ones are!
 *
 *  @b Cache-Efficiency: Keys and values are stored separately (Structure-of-Arrays) rather
 *  than interleaved (Array-of-Structures). During lookups that don't need values, only keys are
 *  loaded into cache, halving memory traffic for failed lookups.
 *
 *  @b Performance: Optimized for bulk operations common in analytical and database workloads.
 *  The promises a call carries - @c assume_reserved_t, @c assume_unique_t - select the specialized
 *  code path at compile time, so no branch survives into the emitted code. Each is a claim about the
 *  argument or the container's state that the callee cannot check for itself, never a mode switch.
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
 *  A single allocation carved into three regions, owned by @c hash_storage and described in
 *  @c hash_layout.hpp. The ordering enables @b compaction: when squeezing the hash table into a
 *  dense array, the keys and values are already contiguous at the start of the buffer, and the
 *  headers at the end can be discarded.
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
 *  @par Small String Optimization
 *  Inline storage for keys under 16 bytes can eliminate heap allocations and enable efficient
 *  vectorized hashing of fixed-size data. Two approaches possible:
 *  - Dual tables: separate @c basic_hash_table<array<char,16>, V> for short strings
 *  - Variant storage: @c union of inline buffer and pointer (standard SSO pattern)
 *
 *  @see https://en.wikipedia.org/wiki/Hash_table
 *  @see Google's SwissTable: https://abseil.io/docs/cpp/guides/container
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::uint8_t`
#include <cstdlib> // `std::abort`
#include <cstring> // `std::memcpy`

#include <limits>      // `std::numeric_limits`
#include <type_traits> // `std::is_same`, `std::enable_if`
#include <utility>     // `std::move`, `std::swap`

#include "hash_layout.hpp"

namespace ashvardanian::smashtable {

#pragma region Iterators

/** @brief Detects iterators by the presence of an @c iterator_category, without hard-erroring on other types. */
template <typename type_, typename = void>
struct is_iterator_type : std::false_type {};

template <typename type_>
struct is_iterator_type<type_, std::void_t<decltype(*std::declval<type_ &>()), decltype(++std::declval<type_ &>())>>
    : std::true_type {};

/** @brief Type trait to check if a type is an iterator. */
template <typename type_>
constexpr bool is_iterator() {
    return is_iterator_type<std::remove_cvref_t<type_>>::value;
}

/**
 *  @brief Forward iterator for hash tables using simplified slot-based addressing.
 *    Inherits the three region pointers and the slot index from @c hash_slot_ref, and adds a
 *    counter of unvisited slots. Implements @c std::forward_iterator_tag for sequential traversal.
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

#pragma endregion Iterators

/** @brief What the callback tells the probe walk to do next, returned from each visit. */
enum class probe_control_t : std::uint8_t {
    /** @brief The callback wants the remaining matches of the same probe run. */
    resume_k,
    /** @brief The callback is done, and the walk stops before the next slot. */
    halt_k,
};

/**
 *  @brief Growable hash table with linear probing and Structure-of-Arrays layout.
 *    Compatible with @c std::unordered_set interface. See file header for detailed design rationale.
 *
 *  @tparam element_type_ Hashable and equality-comparable key type, or @c mapping<K,V> for maps.
 *  @tparam hasher_type_ Hash function type. Must be copy-constructible. Defaults to a @c std::hash wrapper.
 *  @tparam equals_type_ Equality predicate supporting heterogeneous lookups. Must be copy-constructible.
 *    Defaults to a transparent @c std::equal_to.
 *  @tparam allocator_type_ Allocator for internal memory management. Defaults to @c default_allocator<std::byte>.
 *
 *  @see https://en.cppreference.com/w/cpp/container/unordered_set
 *  @see https://en.cppreference.com/w/cpp/container/unordered_map
 */
template <typename element_type_, typename hasher_type_ = default_hash_t, typename equals_type_ = equal_to_t,
          typename allocator_type_ = default_allocator<std::byte>>
class basic_hash_table {

    using layout_t = hash_layout_for<element_type_, hasher_type_>;
    using key_t = typename layout_t::key_t;
    using value_t = typename layout_t::value_t;
    using value_storage_t = typename layout_t::value_storage_t;
    using element_t = typename layout_t::element_t;
    using offset_t = typename layout_t::offset_t;

    /** @brief Whether the element carries a mapped value, making this table a map rather than a set. */
    inline static constexpr bool has_values_k = layout_t::has_values_k;
    /** @brief Bytes one bucket occupies across all three regions, header included. */
    inline static constexpr std::size_t bytes_in_bucket_k = layout_t::bytes_in_bucket_k;
    /** @brief Whether the hasher is stateless, so swapping tables need not exchange it. */
    inline static constexpr bool hasher_is_empty_k = std::is_empty<hasher_type_>::value;
    /** @brief Whether the equality predicate is stateless, so swapping tables need not exchange it. */
    inline static constexpr bool equals_is_empty_k = std::is_empty<equals_type_>::value;

    using hasher_t = hasher_type_;
    using equals_t = equals_type_;
    using allocator_t = allocator_type_;
    using storage_t = hash_storage<element_type_, hasher_type_, allocator_type_>;
    using iterator_t = hash_table_iterator<element_type_, hasher_t>;
    using const_iterator_t = hash_table_iterator<element_type_ const, hasher_t>;
    using slot_ref_t = hash_slot_ref<element_type_, hasher_t>;
    using const_slot_ref_t = hash_slot_ref<element_type_ const, hasher_t>;

    static_assert(!std::is_reference<key_t>(), "Keys can't be references!");
    static_assert(!std::is_reference<value_t>(), "Values can't be references!");
    static_assert(std::is_unsigned<offset_t>(),
                  "Hash value must be an unsigned integer, like std::uint32_t or std::uint64_t!");

    /** @brief Whether erasure and teardown must run a key destructor. */
    inline static constexpr bool destruct_keys_k = !std::is_trivially_destructible<key_t>();
    /** @brief Whether erasure and teardown must run a value destructor. */
    inline static constexpr bool destruct_values_k = has_values_k && !std::is_trivially_destructible<value_storage_t>();

  public:
    /** @brief What a probe that may store an element managed to do. */
    enum class upsert_result_t : std::uint8_t {
        /** @brief A free or tombstoned slot took the new element. */
        inserted_k,
        /** @brief An equal key was already there, and the callback saw it. */
        updated_k,
        /** @brief Every slot along the probe sequence was taken by a different key. */
        no_slot_k,
        /** @brief A new key needed a larger table, and the allocator refused to supply one. */
        no_memory_k,
    };

    /**
     *  @brief A return type for the insert function, similar to @c insert_return_type in STL.
     *  @c position names the stored or the conflicting element, and the end when nothing was stored.
     *  @c outcome says which of the four happened, so a key already taken, a probe with no slot left
     *  and an allocation the table could not make are all distinguishable from one another, without
     *  comparing the iterator against @c end().
     */
    struct insert_result_t {
        iterator_t position;
        upsert_result_t outcome = upsert_result_t::no_slot_k;

        /** @brief Whether nothing was stored, which is the only way an insertion fails. */
        bool failed() const noexcept {
            return outcome != upsert_result_t::inserted_k && outcome != upsert_result_t::updated_k;
        }
    };

    /** @brief What a growth attempt did to the table. */
    enum class reserve_result_t : std::uint8_t {
        /** @brief The capacity already covered the request, so every iterator stays valid. */
        unchanged_k,
        /** @brief A larger buffer replaced the old one, invalidating every iterator. */
        reallocated_k,
        /** @brief The allocator refused, leaving the table exactly as it was. */
        failed_k,
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
    using storage_type = storage_t;

    // Traits the transactional adapters dispatch on.
    using is_associative = std::bool_constant<has_values_k>;
    using is_transactional = std::false_type;

    /**
     *  @brief Restates this table over a different element type, hasher and equality.
     *    A store that decorates its elements - wrapping them in version metadata, say - needs the
     *    equality restated alongside the hasher, since both must address the decorated shape.
     */
    template <typename other_element_type_, typename other_hasher_type_, typename other_equals_type_ = equals_t>
    using rebind = basic_hash_table<other_element_type_, other_hasher_type_, other_equals_type_, allocator_t>;

  private:
    /** @brief The single allocation, the three regions carved from it, and the counters. */
    storage_t storage_;

    /** @brief Hashes a key down to its initial probe offset. */
    ST_NO_UNIQUE_ADDRESS_ hasher_t hasher_ {};
    /** @brief Decides whether a probed key matches the wanted one. */
    ST_NO_UNIQUE_ADDRESS_ equals_t equals_ {};

    /**
     *  @brief Allocates and zeroes a table of the requested slot count.
     *    On allocation failure the table is left empty, which the factories report as an error.
     */
    basic_hash_table(hash_slots_count_t slots, hasher_t hasher, equals_t equals, allocator_t allocator) noexcept
        : storage_(storage_t::make(slots, std::move(allocator))), hasher_(std::move(hasher)),
          equals_(std::move(equals)) {}

    /** @brief What an insertion that stored nothing reports: the end position, and why it stored nothing. */
    insert_result_t missed_position_(upsert_result_t outcome) noexcept {
        insert_result_t result;
        unsafe_retarget(result.position, storage_.slots_count);
        result.position.slots_remaining = 0;
        result.outcome = outcome;
        return result;
    }

    /**
     *  @brief Makes room for one more key, unless @p wanted is already stored and needs no room.
     *    An element the table already holds is overwritten in place, so asking the allocator for a
     *    slot before the probe has established the key is absent turns a plain overwrite into an
     *    allocation failure the caller has no way to satisfy.
     *  @param[in] tags Markers for special acceleration: @c assume_unique_t has already answered
     *    the question the presence probe would ask, so the growth is decided without one.
     *  @return @c unchanged_k when nothing had to move, @c reallocated_k after a growth, and
     *    @c failed_k when a growth the table genuinely needed was refused.
     */
    template <typename comparable_key_type_, typename... tags_types_>
    reserve_result_t reserve_for_absent_key_(comparable_key_type_ const &wanted, tags_types_...) noexcept {
        offset_t const planned_elements = 1 + storage_.populated_count + storage_.deleted_count;
        if (planned_elements <= storage_.growth_threshold) return reserve_result_t::unchanged_k;
        if constexpr (!contains_type<assume_unique_t, tags_types_...>()) {
            bool key_is_present = false;
            probe_to_find(wanted, [&](auto const &) noexcept { key_is_present = true; });
            if (key_is_present) return reserve_result_t::unchanged_k;
        }
        return reserve(planned_elements);
    }

  public:
#pragma region Constructors

    /**
     *  @brief Default constructor, avoiding memory allocations, accepting a pre-constructed hasher functor,
     *    an equality operator, and the allocator state.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/unordered_map
     */
    basic_hash_table(hasher_t hasher = {}, equals_t equals = {}, allocator_t allocator = {}) noexcept
        : storage_(std::move(allocator)), hasher_(std::move(hasher)), equals_(std::move(equals)) {}

    basic_hash_table(basic_hash_table &&other) noexcept { swap(other); }

    basic_hash_table &operator=(basic_hash_table &&other) noexcept {
        swap(other);
        return *this;
    }

    basic_hash_table(basic_hash_table const &) noexcept = delete;
    basic_hash_table &operator=(basic_hash_table const &) noexcept = delete;
    basic_hash_table(offset_t, hasher_t = {}, equals_t = {}, allocator_t = {}) noexcept = delete;

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
        if (slots.raw && !table.storage_.is_allocated()) return expected<basic_hash_table>(out_of_memory_heap_k);
        return expected<basic_hash_table>(std::move(table), success_k);
    }

    /**
     *  @brief Builds a table pre-sized for the range and fills it.
     *  @return An empty-status @c expected if the allocation has failed.
     */
    template <typename begin_iterator_type_, typename end_iterator_type_,
              typename std::enable_if<is_iterator<begin_iterator_type_>(), int>::type = 0>
    [[nodiscard]] static expected<basic_hash_table> make(begin_iterator_type_ begin, end_iterator_type_ end) noexcept {
        auto table = make(hash_slots_count_t {static_cast<std::size_t>(distance_between(begin, end))});
        if (!table) return table;
        table->insert(begin, end, assume_reserved_t {});
        return table;
    }

    /** @brief Surrenders the allocation, leaving this table empty. */
    [[nodiscard]] storage_t release() && noexcept { return std::move(storage_); }

    /** @brief Takes ownership of an allocation another table built. */
    [[nodiscard]] static basic_hash_table adopt(storage_t &&storage, hasher_t hasher = {},
                                                equals_t equals = {}) noexcept {
        basic_hash_table table(std::move(hasher), std::move(equals));
        table.storage_ = std::move(storage);
        return table;
    }

#pragma endregion Constructors

#pragma region Metadata

    bool empty() const noexcept { return !storage_.populated_count; }
    offset_t size() const noexcept { return storage_.populated_count; }

    /** @brief Tombstones left behind by erasures, which only a rehash reclaims. */
    offset_t deleted_count() const noexcept { return storage_.deleted_count; }

    offset_t optimal_capacity() const noexcept { return storage_.growth_threshold; }
    offset_t capacity() const noexcept { return larger_of(optimal_capacity(), size()); }

    std::size_t size_bytes() const noexcept { return storage_.size_bytes(); }

    hash_slots_count_t slots_count() const noexcept {
        hash_slots_count_t result;
        result.raw = storage_.slots_count;
        return result;
    }

    /**
     *  @brief STL-compatibility function.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/bucket_count
     */
    offset_t bucket_count() const noexcept { return storage_.bucket_count(); }

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
    allocator_t get_allocator() const noexcept { return storage_.allocator; }

#pragma endregion Metadata

#pragma region Probes

    /**
     *  @brief Probe optimized for @c find: checks if a value is present.
     *    Unlike @c probe_to_insert or @c probe_to_upsert, doesn't track "deleted" slots.
     *
     *  @param[in] wanted Hashable and comparable with key object.
     *  @param[in] call A callback receiving a @c const_slot_ref_t to an initialized matching object.
     *  @param[in] tags Markers for special acceleration: @c assume_reserved_t avoids null checks.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    void probe_to_find(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_...) const noexcept {

        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (!storage_.populated_count) [[unlikely]]
                return;

        offset_t const offset_mask = storage_.slots_count - 1;
        offset_t const initial_offset = hasher_(wanted) & offset_mask;
        offset_t offset = initial_offset;
        const_slot_ref_t slot;

        while (true) {
            unsafe_retarget(slot, offset);

            // Checking for equality isn't safe, if we operate on uninitialized
            // memory and call some complex comparison operators on them.
            // So instead of one runtime `if`, we have two nested `if`s.
            if (slot.is_populated()) {
                if (equals_(slot.key(), wanted)) {
                    call(slot);
                    break;
                }
                offset = (offset + 1) & offset_mask;
                if (offset == initial_offset) break;
            }
            else if (slot.is_deleted()) {
                offset = (offset + 1) & offset_mask;
                if (offset == initial_offset) break;
            }
            else {
                // A "free" slot ends the probe sequence, and a locked one would truncate it, so
                // the state is checked rather than inferred from "not populated, not deleted".
                assert(slot.is_free() && "A growable table is single-threaded, so no slot can be locked");
                break;
            }
        }
    }

    /**
     *  @brief Visits @b every populated slot matching @p wanted, not just the first.
     *    A table whose equality separates versions of one key files them all in one probe run, because
     *    the hasher still peels to the bare key. Finding one of them is not finding the newest one.
     *
     *  @param[in] wanted Hashable and comparable with key object.
     *  @param[in] call Receives each @c const_slot_ref_t and returns @c probe_control_t. Must be @c noexcept.
     *  @param[in] tags Markers for special acceleration: @c assume_reserved_t avoids null checks.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    void probe_to_visit(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_...) const noexcept {

        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (!storage_.populated_count) [[unlikely]]
                return;

        offset_t const offset_mask = storage_.slots_count - 1;
        offset_t const initial_offset = hasher_(wanted) & offset_mask;
        offset_t offset = initial_offset;
        const_slot_ref_t slot;

        while (true) {
            unsafe_retarget(slot, offset);

            // Checking for equality isn't safe, if we operate on uninitialized
            // memory and call some complex comparison operators on them.
            // So instead of one runtime `if`, we have two nested `if`s.
            if (slot.is_populated()) {
                if (equals_(slot.key(), wanted))
                    if (call(slot) == probe_control_t::halt_k) break;
                offset = (offset + 1) & offset_mask;
                if (offset == initial_offset) break;
            }
            else if (slot.is_deleted()) {
                // A tombstone only interrupts the run for a reader that stops at the first match,
                // and this walk wants the ones behind it.
                offset = (offset + 1) & offset_mask;
                if (offset == initial_offset) break;
            }
            else {
                // A "free" slot ends the probe sequence, and a locked one would truncate it, so
                // the state is checked rather than inferred from "not populated, not deleted".
                assert(slot.is_free() && "A growable table is single-threaded, so no slot can be locked");
                break;
            }
        }
    }

    /**
     *  @brief Probe optimized for @c find: checks if a value is present.
     *    Unlike @c probe_to_insert or @c probe_to_upsert, doesn't track "deleted" slots.
     *
     *  @param[in] wanted Hashable and comparable with key object.
     *  @param[in] call A callback receiving a @c slot_ref_t to an initialized matching object.
     *  @param[in] tags Markers for special acceleration: @c assume_reserved_t avoids null checks.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    void probe_to_find(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_...) noexcept {

        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (!storage_.populated_count) [[unlikely]]
                return;

        offset_t const offset_mask = storage_.slots_count - 1;
        offset_t const initial_offset = hasher_(wanted) & offset_mask;
        offset_t offset = initial_offset;
        slot_ref_t slot;

        while (true) {
            unsafe_retarget(slot, offset);

            if (slot.is_populated()) {
                if (equals_(slot.key(), wanted)) {
                    call(slot);
                    break;
                }
                offset = (offset + 1) & offset_mask;
                if (offset == initial_offset) break;
            }
            else if (slot.is_deleted()) {
                offset = (offset + 1) & offset_mask;
                if (offset == initial_offset) break;
            }
            else {
                assert(slot.is_free() && "A growable table is single-threaded, so no slot can be locked");
                break;
            }
        }
    }

    /**
     *  @brief Probe optimized for @c insert of a unique key.
     *    Unlike @c probe_to_upsert avoids potentially expensive equality comparisons,
     *    knowing that the incoming key is different from all present members.
     *
     *  @param[in] wanted Hashable and comparable with key object.
     *  @param[in] call A callback receiving a @c slot_ref_t to UN-initialized memory, where an
     *    element should be built.
     *  @return Whether a slot took the element; the counters only move when one did.
     */
    template <typename comparable_key_type_, typename callback_type_, typename... tags_types_>
    upsert_result_t probe_to_insert(comparable_key_type_ &&wanted, callback_type_ &&call, tags_types_...) noexcept {

        offset_t const offset_mask = storage_.slots_count - 1;
        offset_t offset = hasher_(wanted) & offset_mask;
        slot_ref_t slot;

        // Bounded by the table itself. A caller promising @c assume_reserved_t promises a free slot
        // exists, and probing forever is the wrong way to discover the promise was false - it hangs
        // with nothing to show, in a build where an assertion would be gone.
        for (offset_t probes = 0; probes != storage_.slots_count; ++probes) {
            unsafe_retarget(slot, offset);

            if (slot.is_free() || slot.is_deleted()) {
                bool const reused_tombstone = slot.is_deleted();
                call(slot);
                slot.mark_populated();
                ++storage_.populated_count;
                if (reused_tombstone) --storage_.deleted_count;
                return upsert_result_t::inserted_k;
            }

            // A slot is "populated", definitely with a different key!
            offset = (offset + 1) & offset_mask;
        }

        return upsert_result_t::no_slot_k;
    }

    /**
     *  @brief Probe optimized for potential @c upserts.
     *    More expensive than @c probe_to_find and @c probe_to_insert, so pick this method wisely.
     *
     *  @param[in] wanted Hashable and comparable with key object.
     *  @param[in] call_unused A callback receiving a @c slot_ref_t to UN-initialized memory, where
     *    an element should be built.
     *  @param[in] call_equal A callback receiving a @c slot_ref_t to an initialized matching object.
     *  @return What the probe managed to do; @c no_slot_k when every slot holds a different key.
     */
    template <typename comparable_key_type_, typename callback_unused_type_, typename callback_equal_type_,
              typename... tags_types_>
    upsert_result_t probe_to_upsert(comparable_key_type_ &&wanted, callback_unused_type_ &&call_unused,
                                    callback_equal_type_ &&call_equal, tags_types_...) noexcept {

        offset_t const offset_mask = storage_.slots_count - 1;
        offset_t offset = hasher_(wanted) & offset_mask;

        // The slot currently under inspection, and the first observed "deleted" slot.
        slot_ref_t current, reusable;
        bool did_find_deleted = false;

        // Bounded by the table itself: a table with no free slot left has nothing to end the probe
        // sequence on, and spinning is the wrong way to discover that - it hangs with nothing to
        // show, in a build where an assertion would be gone.
        for (offset_t probes = 0; probes != storage_.slots_count; ++probes) {
            unsafe_retarget(current, offset);

            if (current.is_populated()) {
                // We have found a filled slot and must check for match.
                if (equals_(current.key(), wanted)) {
                    call_equal(current);
                    return upsert_result_t::updated_k;
                }
                offset = (offset + 1) & offset_mask;
                continue;
            }

            if (current.is_deleted()) {
                // The first deleted slot is remembered, in case the probe never finds the key.
                if (!did_find_deleted) {
                    reusable = current;
                    did_find_deleted = true;
                }
                offset = (offset + 1) & offset_mask;
                continue;
            }

            // In case of a "free" slot:
            // > If a deleted slot was seen earlier in this probe: reuse it, drop the free one.
            // > Otherwise: take the free slot itself.
            assert(current.is_free() && "A growable table is single-threaded, so no slot can be locked");
            slot_ref_t &chosen = did_find_deleted ? reusable : current;
            call_unused(chosen);
            chosen.mark_populated();

            ++storage_.populated_count;
            if (did_find_deleted) --storage_.deleted_count;
            return upsert_result_t::inserted_k;
        }

        // The walk came back to where it started without meeting a free slot. A tombstone seen along
        // the way is still ours to take; without one there is nowhere left to store.
        if (!did_find_deleted) return upsert_result_t::no_slot_k;
        call_unused(reusable);
        reusable.mark_populated();
        ++storage_.populated_count;
        --storage_.deleted_count;
        return upsert_result_t::inserted_k;
    }

    /**
     *  @brief Retargets an element reference to a specific slot using direct array indexing.
     *  @param[out] slot Slot reference to retarget with new region pointers and a new index.
     *  @param[in] slot_index Absolute slot index in [0, slots_count).
     */
    template <typename element_ref_type_>
    void unsafe_retarget(element_ref_type_ &slot, offset_t slot_index) const noexcept {
        storage_.retarget_slot(slot, slot_index);
    }

#pragma endregion Probes

#pragma region Lookups

    /** @brief Number of slots between the reference's position and the end of the table. */
    template <typename address_or_iterator_type_>
    offset_t slots_remaining_after(address_or_iterator_type_ &&slot) const noexcept {
        return storage_.slots_count - slot.slot_;
    }

    /**
     *  @brief Reports whether a key equivalent to @p wanted is present.
     *  @param[in] wanted Key to probe for.
     *  @param[in] tags Dispatch tags the probe understands.
     *  @return Whether the key is present. A probe over owned slots allocates nothing, so it never refuses.
     */
    template <typename comparable_key_type_, typename... tags_types_>
    [[nodiscard]] expected<bool> contains(comparable_key_type_ &&wanted, tags_types_... tags) const noexcept {
        bool present = false;
        probe_to_find(
            std::forward<comparable_key_type_>(wanted), [&present](auto const &) noexcept { present = true; }, tags...);
        return present;
    }

    /**
     *  @brief Similar to STL, searches for the element with the given key.
     *  @return End iterator, iff element with such key is missing.
     */
    template <typename comparable_key_type_ = key_t const &>
    const_iterator_t find(comparable_key_type_ &&wanted) const noexcept {
        const_iterator_t it;
        unsafe_retarget(it, storage_.slots_count);
        it.slots_remaining = 0;
        probe_to_find(std::forward<comparable_key_type_>(wanted),
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
        unsafe_retarget(it, storage_.slots_count);
        it.slots_remaining = 0;
        probe_to_find(std::forward<comparable_key_type_>(wanted),
                      [&it](auto const &slot) noexcept { it.slot_ = slot.slot_; });
        it.slots_remaining = slots_remaining_after(it);
        return it;
    }

    /**
     *  @brief Returns a reference to the mapped value of the element with the given key.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/at
     *  @warning A missing key has no value to name and this library does not throw, so the call
     *    aborts rather than hand back a reference one past the values region.
     */
    template <typename comparable_key_type_ = key_t const &>
    value_storage_t &at(comparable_key_type_ &&key) noexcept {
        static_assert(has_values_k, "at() is only available for maps, not sets");
        slot_ref_t result;
        unsafe_retarget(result, 0);
        bool found = false;
        probe_to_find(std::forward<comparable_key_type_>(key), [&](slot_ref_t const &slot) noexcept {
            result.slot_ = slot.slot_;
            found = true;
        });
        assert(found && "The object must be already present");
        if (!found) std::abort();
        return result.value_ref();
    }

    /**
     *  @brief Returns a const reference to the mapped value of the element with the given key.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/at
     *  @warning A missing key has no value to name and this library does not throw, so the call
     *    aborts rather than hand back a reference one past the values region.
     */
    template <typename comparable_key_type_ = key_t const &>
    value_storage_t const &at(comparable_key_type_ &&key) const noexcept {
        static_assert(has_values_k, "at() is only available for maps, not sets");
        const_slot_ref_t result;
        unsafe_retarget(result, 0);
        bool found = false;
        probe_to_find(std::forward<comparable_key_type_>(key), [&](const_slot_ref_t const &slot) noexcept {
            result.slot_ = slot.slot_;
            found = true;
        });
        assert(found && "The object must be already present");
        if (!found) std::abort();
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
    [[nodiscard]] expected<std::size_t> count(comparable_key_type_ &&key) const noexcept {
        expected<bool> const present = contains(std::forward<comparable_key_type_>(key));
        if (!present) return present.status();
        return *present ? std::size_t {1} : std::size_t {0};
    }

#pragma endregion Lookups

#pragma region Scans

    iterator_t begin() noexcept {
        iterator_t it;
        unsafe_retarget(it, 0);
        if (empty()) {
            it.slot_ = storage_.slots_count;
            it.slots_remaining = 0;
        }
        else {
            it.slots_remaining = storage_.slots_count;
            it.skip_non_populated();
        }
        return it;
    }

    const_iterator_t cbegin() const noexcept {
        const_iterator_t it;
        unsafe_retarget(it, 0);
        if (empty()) {
            it.slot_ = storage_.slots_count;
            it.slots_remaining = 0;
        }
        else {
            it.slots_remaining = storage_.slots_count;
            it.skip_non_populated();
        }
        return it;
    }

    const_iterator_t begin() const noexcept { return cbegin(); }
    end_sentinel_t end() const noexcept { return end_sentinel_t {}; }
    end_sentinel_t cend() const noexcept { return end_sentinel_t {}; }

    /** @brief Invokes the callback for every populated slot, bucket by bucket. */
    template <typename callback_type_ = no_op<slot_ref_t>>
    [[nodiscard]] status_t for_each(callback_type_ &&callback) noexcept {
        slot_ref_t slot;
        unsafe_retarget(slot, 0);
        offset_t const buckets = bucket_count();
        for (offset_t bucket_index = 0; bucket_index != buckets; ++bucket_index) {
            slot.slot_ = bucket_index * hash_bucket_capacity_k;
            for_each_in_hash_bucket(slot, callback);
        }
        return success_k;
    }

    /** @brief Invokes the callback for every populated slot, bucket by bucket. */
    template <typename callback_type_ = no_op<const_slot_ref_t>>
    [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept {
        const_slot_ref_t slot;
        unsafe_retarget(slot, 0);
        offset_t const buckets = bucket_count();
        for (offset_t bucket_index = 0; bucket_index != buckets; ++bucket_index) {
            slot.slot_ = bucket_index * hash_bucket_capacity_k;
            for_each_in_hash_bucket(slot, callback);
        }
        return success_k;
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

    /** @brief What an insertion hands back: nothing at all, or where the element landed and how. */
    enum class emplace_report_t : bool { no_result_k, insert_result_k };

    /**
     *  @brief Deleted: hint-based emplace is not supported.
     *    A hint names a position, and an open-addressed table decides position by hashing the key,
     *    so there is nothing a caller could usefully hint at. Declaring it deleted rather than
     *    omitting it turns a port from @c std::unordered_map into a signposted error instead of an
     *    unexplained missing member. Use @c emplace, or @c probe_to_upsert for finer control.
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
     *    @c assume_reserved_t avoids reserving more memory, @c assume_unique_t skips equality comparisons.
     */
    template <
        emplace_report_t report_ = emplace_report_t::no_result_k, typename convertible_key_type_,
        typename convertible_value_type_, typename... tags_types_,
        typename std::enable_if<can_use_map_emplace<convertible_key_type_, convertible_value_type_>(), int>::type = 0>
    std::conditional_t<report_ == emplace_report_t::insert_result_k, insert_result_t, void> emplace(
        convertible_key_type_ &&key, convertible_value_type_ &&value, tags_types_... tags) noexcept {

        // A refused allocation must stop the insertion here: probing a table that could not grow
        // fills it to the last slot and then has nowhere left to go. A key already present takes no
        // slot, so the allocator is only consulted once the probe has shown this one is new.
        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (reserve_for_absent_key_(key, tags...) == reserve_result_t::failed_k) {
                if constexpr (report_ == emplace_report_t::insert_result_k)
                    return missed_position_(upsert_result_t::no_memory_k);
                else return;
            }

        constexpr bool return_k = report_ == emplace_report_t::insert_result_k;
        using result_t = std::conditional_t<return_k, insert_result_t, placeholder_t>;

        [[maybe_unused]] result_t result;
        if constexpr (return_k) {
            unsafe_retarget(result.position, storage_.slots_count);
            result.position.slots_remaining = 0;
        }

        auto callback_unused = [&](slot_ref_t &unused_slot) noexcept {
            new (&unused_slot.key_ref()) key_t(std::forward<convertible_key_type_>(key));
            new (&unused_slot.value_ref()) value_storage_t(std::forward<convertible_value_type_>(value));
            if constexpr (return_k) result.position.slot_ = unused_slot.slot_;
        };

        [[maybe_unused]] upsert_result_t stored = upsert_result_t::no_slot_k;
        if constexpr (contains_type<assume_unique_t, tags_types_...>())
            stored = probe_to_insert(key, callback_unused, assume_reserved_t {}, tags...);
        else
            stored = probe_to_upsert(
                key, callback_unused,
                [&](slot_ref_t &equal_slot) noexcept {
                    equal_slot.value_ref() = std::forward<convertible_value_type_>(value);
                    if constexpr (return_k) result.position.slot_ = equal_slot.slot_;
                },
                assume_reserved_t {}, tags...);

        if constexpr (return_k) {
            result.outcome = stored;
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
     *    @c assume_reserved_t avoids reserving more memory, @c assume_unique_t skips equality comparisons.
     */
    template <emplace_report_t report_ = emplace_report_t::no_result_k, typename convertible_key_type_,
              typename... tags_types_,
              typename std::enable_if<can_use_set_emplace<convertible_key_type_>(), int>::type = 0>
    std::conditional_t<report_ == emplace_report_t::insert_result_k, insert_result_t, void> emplace(
        convertible_key_type_ &&key, tags_types_... tags) noexcept {

        // A refused allocation must stop the insertion here: probing a table that could not grow
        // fills it to the last slot and then has nowhere left to go. A key already present takes no
        // slot, so the allocator is only consulted once the probe has shown this one is new.
        if constexpr (!contains_type<assume_reserved_t, tags_types_...>())
            if (reserve_for_absent_key_(key, tags...) == reserve_result_t::failed_k) {
                if constexpr (report_ == emplace_report_t::insert_result_k)
                    return missed_position_(upsert_result_t::no_memory_k);
                else return;
            }

        constexpr bool return_k = report_ == emplace_report_t::insert_result_k;
        using result_t = std::conditional_t<return_k, insert_result_t, placeholder_t>;

        [[maybe_unused]] result_t result;
        if constexpr (return_k) {
            unsafe_retarget(result.position, storage_.slots_count);
            result.position.slots_remaining = 0;
        }

        auto callback_unused = [&](slot_ref_t &unused_slot) noexcept {
            new (&unused_slot.key_ref()) key_t(std::forward<convertible_key_type_>(key));
            if constexpr (return_k) result.position.slot_ = unused_slot.slot_;
        };

        [[maybe_unused]] upsert_result_t stored = upsert_result_t::no_slot_k;
        if constexpr (contains_type<assume_unique_t, tags_types_...>())
            stored = probe_to_insert(key, callback_unused, assume_reserved_t {}, tags...);
        else if constexpr (return_k)
            stored = probe_to_upsert(
                key, callback_unused,
                [&](slot_ref_t &equal_slot) noexcept { result.position.slot_ = equal_slot.slot_; },
                assume_reserved_t {}, tags...);
        else stored = probe_to_upsert(key, callback_unused, no_op_t {}, assume_reserved_t {}, tags...);

        if constexpr (return_k) {
            result.outcome = stored;
            result.position.slots_remaining = slots_remaining_after(result.position);
            return result;
        }
    }

    /**
     *  @brief Main insertion and upsertion method, compatible with STL code.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/insert
     *
     *  @note With maps this expects a @c mapping, exposing @c key and @c mapped members.
     */
    template <emplace_report_t report_ = emplace_report_t::no_result_k, typename temporary_pack_type_,
              typename... tags_types_, typename std::enable_if<!is_iterator<temporary_pack_type_>(), int>::type = 0>
    std::conditional_t<report_ == emplace_report_t::insert_result_k, insert_result_t, void> insert(
        temporary_pack_type_ &&pack, tags_types_... tags) noexcept {
        if constexpr (has_values_k)
            return emplace<report_>(std::forward<temporary_pack_type_>(pack).key,
                                    std::forward<temporary_pack_type_>(pack).mapped, tags...);
        else return emplace<report_>(std::forward<temporary_pack_type_>(pack), tags...);
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
     *  @return @c out_of_memory_heap_k when a @b new key needed a larger table and the allocator
     *    refused, @c capacity_exhausted_k when the bounded probe found no slot for it and only a
     *    rehash can make one, and @c success_k otherwise - including the overwrite of a key already
     *    stored, which needs no room and therefore never asks the allocator for any.
     */
    template <typename convertible_element_type_>
    [[nodiscard]] status_t upsert(convertible_element_type_ &&element) noexcept {
        // The probe below overwrites an equal key in place, taking no slot of its own, so the
        // allocator is only consulted once the probe has shown this key is not already stored.
        reserve_result_t reserved = reserve_result_t::unchanged_k;
        if constexpr (has_values_k) reserved = reserve_for_absent_key_(element.key);
        else reserved = reserve_for_absent_key_(element);
        if (reserved == reserve_result_t::failed_k) return out_of_memory_heap_k;

        upsert_result_t stored = upsert_result_t::no_slot_k;
        if constexpr (has_values_k)
            stored = probe_to_upsert(
                element.key,
                [&](slot_ref_t &unused_slot) noexcept {
                    new (&unused_slot.key_ref()) key_t(std::move(element.key));
                    new (&unused_slot.value_ref()) value_storage_t(std::move(element.mapped));
                },
                [&](slot_ref_t &equal_slot) noexcept { equal_slot.value_ref() = std::move(element.mapped); },
                assume_reserved_t {});
        else
            stored = probe_to_upsert(
                element,
                [&](slot_ref_t &unused_slot) noexcept { new (&unused_slot.key_ref()) key_t(std::move(element)); },
                [&](slot_ref_t &equal_slot) noexcept { equal_slot.key_ref() = std::move(element); },
                assume_reserved_t {});
        return stored == upsert_result_t::no_slot_k ? capacity_exhausted_k : success_k;
    }

    /**
     *  @brief An interface similar to @c std::unordered_map::merge, banned in favor of
     *    a manual @c insert(begin, end).
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/merge
     */
    template <typename other_type_>
    void merge(other_type_ &&) = delete;

#pragma endregion Insertions

#pragma region Erasures

    /**
     *  @brief Removes the element the reference points to, leaving a tombstone behind.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/erase
     *  @warning The reference must point at a populated slot.
     */
    void erase(slot_ref_t slot) noexcept {
        if constexpr (destruct_keys_k) slot.key_ref().~key_t();
        if constexpr (destruct_values_k) slot.value_ref().~value_storage_t();
        slot.mark_deleted();
        ++storage_.deleted_count;
        --storage_.populated_count;
    }

    void erase(iterator_t slot) noexcept { erase(static_cast<slot_ref_t>(slot)); }

    void erase(const_iterator_t) = delete;

    /**
     *  @brief Removes a value by key in the most efficient fashion.
     *    If you already have the iterator, use the faster overload that avoids the search entirely.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/erase
     *
     *  @param[in] tags Markers for special acceleration: @c assume_reserved_t assumes a non-empty table.
     *  @return True if the wanted key was found.
     *
     *  @note Erasures never deallocate; memory stays cluttered until the next @c force_resize().
     */
    template <typename comparable_key_type_, typename... tags_types_>
    bool erase(comparable_key_type_ &&key, tags_types_... tags) noexcept {

        bool result = false;
        probe_to_find(
            std::forward<comparable_key_type_>(key),
            [&](slot_ref_t const &slot) noexcept {
                if constexpr (destruct_keys_k) slot.key_ref().~key_t();
                if constexpr (destruct_values_k) slot.value_ref().~value_storage_t();
                slot.mark_deleted();
                ++storage_.deleted_count;
                --storage_.populated_count;
                result = true;
            },
            tags...);

        return result;
    }

#pragma endregion Erasures

#pragma region Memory Management

    /**
     *  @brief Total memory needed for the three regions of a table this size.
     *  @param[in] slots_count Total number of slots, a power of two and a multiple of 32.
     *  @return Total bytes needed for the single allocation containing all three regions.
     */
    static constexpr std::size_t memory_usage(hash_slots_count_t slots_count) noexcept {
        return storage_t::memory_usage(slots_count);
    }

    /**
     *  @brief Grows the table to fit that many elements, if the current capacity is short.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/reserve
     *  @return Whether the layout moved, invalidating older iterators, or the allocation failed.
     */
    reserve_result_t reserve(offset_t planned_elements) noexcept {
        if (planned_elements <= storage_.growth_threshold) return reserve_result_t::unchanged_k;
        return force_resize(hash_slots_count_t {planned_elements});
    }

    /**
     *  @brief The recommended entry point for bulk insertions, sizing for the elements already present.
     *  @return Whether the layout moved, invalidating older iterators, or the allocation failed.
     */
    reserve_result_t reserve_more(offset_t new_elements) noexcept {
        return reserve(new_elements + storage_.populated_count + storage_.deleted_count);
    }

    /**
     *  @brief Exports the data into a fresh table of the requested size and swaps the contents.
     *    Also used to clean the tombstones left behind by deletions.
     *  @return @c failed_k when the request cannot hold what is stored, or the new buffer could not
     *    be allocated, leaving this table untouched either way.
     */
    reserve_result_t force_resize(hash_slots_count_t slots_count) noexcept {
        if (slots_count.raw < size()) return reserve_result_t::failed_k;
        basic_hash_table resized = move_to_new(slots_count);
        if (slots_count.raw && !resized.storage_.is_allocated()) return reserve_result_t::failed_k;
        swap(resized);
        return reserve_result_t::reallocated_k;
    }

    /**
     *  @brief A cheap copy-less swap mechanism used in move-constructors.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/swap
     */
    void swap(basic_hash_table &other) noexcept {
        storage_.swap(other.storage_);
        if constexpr (!hasher_is_empty_k) std::swap(hasher_, other.hasher_);
        if constexpr (!equals_is_empty_k) std::swap(equals_, other.equals_);
    }

    /**
     *  @brief Erases all elements, keeping the memory. For deallocation call @c shrink_to_fit().
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/clear
     */
    void clear() noexcept { storage_.clear(); }

    /**
     *  @brief Brings the memory consumption of the container to the minimum.
     *  @see https://en.cppreference.com/w/cpp/container/vector/shrink_to_fit
     *  @note Best-effort, matching the @c void shape of the standard containers: a refused
     *    allocation leaves the table exactly as it was, with every element still reachable.
     *    Call @c force_resize when the outcome has to be known.
     */
    void shrink_to_fit() noexcept {
        hash_slots_count_t const new_capacity {size()};
        if (new_capacity.raw == storage_.slots_count) return;
        if (size()) { [[maybe_unused]] reserve_result_t const resized = force_resize(new_capacity); }
        else deallocate();
    }

    /**
     *  @brief Deallocates all the memory used by the container.
     *  @warning Call @c clear() first to destruct all the non-trivial elements.
     */
    void deallocate() noexcept { storage_.deallocate(); }

    /**
     *  @brief Sets the number of buckets to the desired value, rehashing in the process.
     *  @see https://en.cppreference.com/w/cpp/container/unordered_map/rehash
     *  @return Whether the layout moved, invalidating older iterators, or the allocation failed.
     *  @note A request too small for what is stored is raised to what the current size needs, so
     *    @c rehash(0) compacts rather than emptying the table into a layout with no slots at all.
     */
    reserve_result_t rehash(offset_t count_buckets) noexcept {
        hash_slots_count_t const wanted = hash_slots_count_t::from_slots(count_buckets * hash_bucket_capacity_k);
        hash_slots_count_t const needed {size()};
        return force_resize(wanted.raw >= needed.raw ? wanted : needed);
    }

#pragma endregion Memory Management

#pragma region Copies and Moves

    /**
     *  @brief Creates a new buffer of the requested capacity and rehashes the present data into it.
     *    Move-constructors are used, avoiding copies, and this table is left deallocated.
     *  @return An unallocated table when the allocator refused or nothing was asked for, leaving
     *    this one untouched.
     *  @warning The new capacity can't be less than the current @c size().
     */
    basic_hash_table move_to_new(hash_slots_count_t slots_count) noexcept {

        // A zero-slot request allocates nothing, and emplacing into an unallocated table would walk
        // a null headers region, so it leaves through the same door an allocation failure does.
        basic_hash_table target(slots_count, hasher_, equals_, get_allocator());
        if (!target.storage_.is_allocated()) return target;
        assert(target.capacity() >= size() && "Not enough space in the new Hash-Table!");

        [[maybe_unused]] status_t const visited = for_each([&target](slot_ref_t const &source_slot) noexcept {
            if constexpr (has_values_k)
                target.emplace(std::move(source_slot.key_ref()), std::move(source_slot.value_ref()),
                               assume_reserved_t {}, assume_unique_t {});
            else target.emplace(std::move(source_slot.key_ref()), assume_reserved_t {}, assume_unique_t {});
            if constexpr (destruct_keys_k) source_slot.key_ref().~key_t();
            if constexpr (destruct_values_k) source_slot.value_ref().~value_storage_t();
        });

        assert(target.size() == size() && "Element counts must match!");
        deallocate();
        return target;
    }

    /** @brief Duplicates the table, reusing the exact bucket layout when the elements are trivial. */
    basic_hash_table copy_to_new() const noexcept {

        basic_hash_table target(slots_count(), hasher_, equals_, get_allocator());
        if (!target.storage_.is_allocated()) return target;

        if constexpr (layout_t::will_memcpy_keys() && (!has_values_k || layout_t::will_memcpy_vals())) {
            std::memcpy(target.storage_.memory, storage_.memory, size_bytes());
            target.storage_.populated_count = storage_.populated_count;
            target.storage_.deleted_count = storage_.deleted_count;
        }
        else [[maybe_unused]]
            status_t const visited = for_each([&target](const_slot_ref_t const &source_slot) noexcept {
                if constexpr (has_values_k)
                    target.emplace(source_slot.key(), *source_slot, assume_reserved_t {}, assume_unique_t {});
                else target.emplace(source_slot.key(), assume_reserved_t {}, assume_unique_t {});
            });

        return target;
    }

    /** @brief Duplicates the table into a differently sized one, rehashing every element. */
    basic_hash_table copy_to_new(hash_slots_count_t slots_count) const noexcept {
        basic_hash_table target(slots_count, hasher_, equals_, get_allocator());
        if (!target.storage_.is_allocated()) return target;
        [[maybe_unused]] status_t const visited = for_each([&target](const_slot_ref_t const &source_slot) noexcept {
            if constexpr (has_values_k)
                target.emplace(source_slot.key(), *source_slot, assume_reserved_t {}, assume_unique_t {});
            else target.emplace(source_slot.key(), assume_reserved_t {}, assume_unique_t {});
        });
        return target;
    }

#pragma endregion Copies and Moves
};

#pragma region Aliases

template <typename key_type_, typename value_type_, typename hasher_type_ = default_hash_t,
          typename equals_type_ = equal_to_t, typename allocator_type_ = default_allocator<std::byte>>
using hash_map = basic_hash_table<mapping<key_type_, value_type_>, hasher_type_, equals_type_, allocator_type_>;

template <typename key_type_, typename hasher_type_ = default_hash_t, typename equals_type_ = equal_to_t,
          typename allocator_type_ = default_allocator<std::byte>>
using hash_set = basic_hash_table<key_type_, hasher_type_, equals_type_, allocator_type_>;

static_assert(sizeof(hash_set<int>) >= 3 * sizeof(void *), "Hash-Table is too small!");

#pragma endregion Aliases

} // namespace ashvardanian::smashtable
