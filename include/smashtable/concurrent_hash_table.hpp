/**
 *  @brief A pinned open-addressed hash table: fixed capacity, lock-free per-operation atomicity, no iterators.
 *  @author Ash Vardanian
 *  @file include/smashtable/concurrent_hash_table.hpp
 *  @date December 21, 2021
 *
 *  @section concurrent_hash_table_why_a_second_type Why a Second Type
 *
 *  A table that can grow can never be read concurrently: growing repoints the key, value and header
 *  regions that every live slot reference has already cached. This table cannot grow, which is what
 *  makes every operation on it sound from any number of threads, and it is a separate type so that
 *  nothing in the growable table's surface - @c reserve, @c rehash, iterators - is reachable here.
 *
 *  The two share only @c hash_layout.hpp. An allocation travels between them as a @c hash_storage:
 *  @c adopt takes one over, @c release hands it back, and neither type names the other.
 *
 *  @section concurrent_hash_table_what_is_absent What Is Absent, and Why
 *
 *  No @c begin, no @c end, no @c for_each: a snapshot of a table other threads are mutating is not a
 *  snapshot. No @c at: it hands out a reference that a concurrent erase invalidates. No @c reserve,
 *  @c rehash or @c shrink_to_fit: reallocation is the thing pinning forbids. Reads take a callback
 *  instead, which is the shape every other collection in this library already uses.
 *
 *  Tombstones only accumulate while pinned, since compaction needs a rehash. @c deleted_count is
 *  what tells a caller it is time to hand the storage back to a growable table and compact it.
 *
 *  @section concurrent_hash_table_gpu Portability to GPUs
 *
 *  The slot protocol underneath is one @c fetch_or to take a slot and one @c fetch_xor to release it,
 *  over a 64-bit bucket header shared by 32 slots - one warp. Those reach a device through the
 *  @c atomic_ref alias in @c shared.hpp, and the fixed capacity is the same constraint device code
 *  wants anyway, so this type is the one that ports.
 *
 *  A kernel calls this table unchanged when built with @c nvcc @c --expt-relaxed-constexpr, which is
 *  what makes the @c constexpr surface here device-callable. What the caller must supply is a hasher,
 *  an equality and an allocator that are themselves device-usable: the defaults reach @c std::hash
 *  and @c std::allocator, neither of which exists on a device. The table object and its storage have
 *  to sit in memory the device can address, which in practice means @c cudaMallocManaged.
 *
 *  Independent thread scheduling is required, so @c sm_70 and newer. A warp probing one bucket
 *  serializes through @c lock, and without it the spin livelocks the warp.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::byte`, `std::size_t`

#include <type_traits> // `std::is_trivially_destructible`
#include <utility>     // `std::move`, `std::forward`

#include "hash_layout.hpp"

namespace ashvardanian::smashtable {

#pragma region Concurrent Table

/**
 *  @brief A fixed-capacity open-addressed table whose every operation is lock-free.
 *
 *  Built by adopting an allocation someone else sized, which is what guarantees the capacity was
 *  reserved before any thread could observe the table.
 *
 *  @tparam element_type_ Stored element - a bare key for a set, a @c mapping for a map.
 *  @tparam hasher_type_ Hashes a key to a slot index.
 *  @tparam equals_type_ Compares two keys for equality.
 *  @tparam allocator_type_ Supplies the single byte buffer the table is carved from.
 */
template <typename element_type_, typename hasher_type_ = default_hash_t, typename equals_type_ = equal_to_t,
          typename allocator_type_ = default_allocator<std::byte>>
class concurrent_hash_table {

    using layout_t = hash_layout_for<element_type_, hasher_type_>;
    using key_t = typename layout_t::key_t;
    using value_t = typename layout_t::value_t;
    using value_storage_t = typename layout_t::value_storage_t;
    using element_t = typename layout_t::element_t;
    using offset_t = typename layout_t::offset_t;

    /** @brief Whether the element carries a mapped value, making this table a map rather than a set. */
    inline static constexpr bool has_values_k = layout_t::has_values_k;
    /** @brief Whether erasure and teardown must run a key destructor. */
    inline static constexpr bool destruct_keys_k = !std::is_trivially_destructible<key_t>();
    /** @brief Whether erasure and teardown must run a value destructor. */
    inline static constexpr bool destruct_vals_k = has_values_k && !std::is_trivially_destructible<value_storage_t>();

    using hasher_t = hasher_type_;
    using equals_t = equals_type_;
    using allocator_t = allocator_type_;
    using storage_t = hash_storage<element_type_, hasher_type_, allocator_type_>;
    using slot_ref_t = hash_atomic_slot_ref<element_type_, hasher_t>;
    using const_slot_ref_t = hash_atomic_slot_ref<element_type_ const, hasher_t>;

    static_assert(std::is_unsigned<offset_t>(),
                  "Hash value must be an unsigned integer, like std::uint32_t or std::uint64_t!");

  public:
    // STL-compatibility definitions, narrowed to what a pinned table can honestly provide.
    using key_type = key_t;
    using mapped_type = value_t;
    using value_type = element_t;
    using owned_value_type = typename layout_t::element_copy_t;
    using size_type = offset_t;
    using hasher = hasher_t;
    using key_equal = equals_t;
    using allocator_type = allocator_t;
    using storage_type = storage_t;

    // Traits the transactional adapters dispatch on.
    using is_associative = std::bool_constant<has_values_k>;
    using callback_reads = std::true_type;
    using is_transactional = std::false_type;

  private:
    /** @brief The single allocation, the three regions carved from it, and the counters. */
    storage_t storage_;

    /** @brief Hashes a key down to its initial probe offset. */
    [[no_unique_address]] hasher_t hasher_ {};
    /** @brief Decides whether a probed key matches the wanted one. */
    [[no_unique_address]] equals_t equals_ {};

  public:
    concurrent_hash_table(hasher_t hasher = {}, equals_t equals = {}) noexcept
        : hasher_(std::move(hasher)), equals_(std::move(equals)) {}

    concurrent_hash_table(concurrent_hash_table &&other) noexcept { swap(other); }
    concurrent_hash_table &operator=(concurrent_hash_table &&other) noexcept {
        swap(other);
        return *this;
    }
    concurrent_hash_table(concurrent_hash_table const &) = delete;
    concurrent_hash_table &operator=(concurrent_hash_table const &) = delete;

    /** @brief Surrenders the allocation, leaving this table empty. */
    [[nodiscard]] storage_t release() && noexcept { return std::move(storage_); }

    /** @brief Takes ownership of an allocation another table built. */
    [[nodiscard]] static concurrent_hash_table adopt(storage_t &&storage, hasher_t hasher = {},
                                                     equals_t equals = {}) noexcept {
        concurrent_hash_table table(std::move(hasher), std::move(equals));
        table.storage_ = std::move(storage);
        return table;
    }

    /** @brief A cheap copy-less exchange of the allocation and the two functors. */
    void swap(concurrent_hash_table &other) noexcept {
        storage_.swap(other.storage_);
        if constexpr (!std::is_empty<hasher_t>::value) std::swap(hasher_, other.hasher_);
        if constexpr (!std::is_empty<equals_t>::value) std::swap(equals_, other.equals_);
    }

#pragma region Metadata

    /**
     *  @brief Live elements. Read atomically but relaxed, so it may lag a concurrent writer - the
     *    count is a statistic here, and other threads are moving it while this returns.
     */
    constexpr offset_t size() const noexcept { return atomic_load(storage_.populated_count); }
    constexpr bool empty() const noexcept { return size() == 0; }

    /** @brief Slots this table will never exceed, since it cannot grow. */
    constexpr offset_t capacity() const noexcept { return larger_of(storage_.growth_threshold, size()); }
    constexpr offset_t slots_count() const noexcept { return storage_.slots_count; }

    /**
     *  @brief Tombstones left by @c erase, which only ever grow while the table is pinned.
     *    Compaction needs a rehash, so this is the signal to hand the storage to a growable table.
     */
    constexpr offset_t deleted_count() const noexcept { return atomic_load(storage_.deleted_count); }

    hasher hash_function() const noexcept { return hasher_; }
    key_equal key_eq() const noexcept { return equals_; }
    allocator_type get_allocator() const noexcept { return storage_.allocator; }

#pragma endregion Metadata

#pragma region Lookups

    /**
     *  @brief Invokes @p callback with the matching slot, under that slot's lock.
     *  @return Whether the key was present, and therefore whether @p callback ran.
     */
    template <typename comparable_key_type_, typename callback_type_>
    constexpr bool find(comparable_key_type_ &&wanted, callback_type_ &&callback) const noexcept {
        return probe_to_find_<const_slot_ref_t>(std::forward<comparable_key_type_>(wanted),
                                                std::forward<callback_type_>(callback));
    }

    template <typename comparable_key_type_>
    constexpr bool contains(comparable_key_type_ &&wanted) const noexcept {
        return probe_to_find_<const_slot_ref_t>(std::forward<comparable_key_type_>(wanted), no_op_fn_t {});
    }

    /** @brief Deleted: a reference into a slot is invalid the moment another thread erases it. */
    template <typename comparable_key_type_ = key_t const &>
    void at(comparable_key_type_ &&) = delete;

    /** @brief Deleted: a snapshot of a table other threads are mutating is not a snapshot. */
    void begin() = delete;
    void end() = delete;

#pragma endregion Lookups

#pragma region Modifications

    /**
     *  @brief Inserts, or overwrites the value when the key is already present.
     *  @return @c success_k, or @c out_of_memory_heap_k when every slot is taken.
     *  @note A pinned table can genuinely fill up, so this reports rather than asserts.
     */
    template <typename convertible_key_type_, typename convertible_value_type_>
    [[nodiscard]] constexpr status_t emplace(convertible_key_type_ &&key, convertible_value_type_ &&value) noexcept {
        static_assert(has_values_k, "A two-argument emplace is only available for maps");
        if (size() + deleted_count() >= slots_count()) return out_of_memory_heap_k;
        probe_to_upsert_(
            key,
            [&](slot_ref_t &unused_slot) noexcept {
                new (&unused_slot.key_ref()) key_t(std::forward<convertible_key_type_>(key));
                new (&unused_slot.value_ref()) value_storage_t(std::forward<convertible_value_type_>(value));
            },
            [&](slot_ref_t &equal_slot) noexcept {
                equal_slot.value_ref() = std::forward<convertible_value_type_>(value);
            });
        return success_k;
    }

    /**
     *  @brief Inserts the key, doing nothing when an equal one is already present.
     *  @return @c success_k, or @c out_of_memory_heap_k when every slot is taken.
     */
    template <typename convertible_key_type_>
    [[nodiscard]] constexpr status_t emplace(convertible_key_type_ &&key) noexcept {
        static_assert(!has_values_k, "A one-argument emplace is only available for sets");
        if (size() + deleted_count() >= slots_count()) return out_of_memory_heap_k;
        probe_to_upsert_(
            key,
            [&](slot_ref_t &unused_slot) noexcept {
                new (&unused_slot.key_ref()) key_t(std::forward<convertible_key_type_>(key));
            },
            no_op_fn_t {});
        return success_k;
    }

    /**
     *  @brief Overwrites the value of an existing key, doing nothing when it is absent.
     *  @return Whether the key was there.
     */
    template <typename comparable_key_type_, typename convertible_value_type_>
    constexpr bool update(comparable_key_type_ &&key, convertible_value_type_ &&new_value) noexcept {
        static_assert(has_values_k, "update() is only available for maps, not sets");
        return probe_to_find_<slot_ref_t>(std::forward<comparable_key_type_>(key),
                                          [&](slot_ref_t const &slot) noexcept {
                                              slot.value_ref() = std::forward<convertible_value_type_>(new_value);
                                          });
    }

    /**
     *  @brief Leaves a tombstone rather than freeing the slot, so probe sequences stay intact.
     *  @return Whether the key was there.
     */
    template <typename comparable_key_type_>
    constexpr bool erase(comparable_key_type_ &&key) noexcept {
        return probe_to_find_<slot_ref_t>(std::forward<comparable_key_type_>(key),
                                          [&](slot_ref_t const &slot) noexcept {
                                              if constexpr (destruct_keys_k) slot.key_ref().~key_t();
                                              if constexpr (destruct_vals_k) slot.value_ref().~value_storage_t();
                                              slot.mark_deleted();
                                              atomic_add_fetch<offset_t>(storage_.deleted_count, 1);
                                              atomic_sub_fetch<offset_t>(storage_.populated_count, 1);
                                          });
    }

#pragma endregion Modifications

  private:
#pragma region Probes

    /**
     *  @brief Walks the probe sequence of @p wanted, invoking @p callback under the matching slot's lock.
     *  @tparam slot_ref_type_ The mutable or the read-only atomic reference, depending on the caller.
     *  @return Whether a match was found. A free slot ends the sequence, a tombstone continues it.
     */
    template <typename slot_ref_type_, typename comparable_key_type_, typename callback_type_>
    constexpr bool probe_to_find_(comparable_key_type_ &&wanted, callback_type_ &&callback) const noexcept {

        if (!storage_.slots_count) [[unlikely]]
            return false;

        offset_t const offset_mask = storage_.slots_count - 1;
        offset_t const initial_offset = hasher_(wanted) & offset_mask;
        offset_t offset = initial_offset;
        slot_ref_type_ slot;
        bool found = false;

        while (true) {
            storage_.retarget_slot(slot, offset);
            slot.lock();

            // Checking for equality isn't safe, if we operate on uninitialized
            // memory and call some complex comparison operators on them.
            // So instead of one runtime `if`, we have two nested `if`s.
            if (slot.is_populated()) {
                if (equals_(slot.key(), wanted)) {
                    callback(slot);
                    slot.unlock();
                    found = true;
                    break;
                }
                slot.unlock();
            }
            else if (slot.is_deleted()) { slot.unlock(); }
            else {
                // A "free" slot ends the probe sequence.
                slot.unlock();
                break;
            }

            offset = (offset + 1) & offset_mask;
            if (offset == initial_offset) break;
        }

        return found;
    }

    /**
     *  @brief Walks the probe sequence of @p wanted, building an element or overwriting the equal one.
     *    The first tombstone met is kept locked until the sequence commits, so no other thread can
     *    claim the slot this one intends to reuse.
     */
    template <typename comparable_key_type_, typename callback_unused_type_, typename callback_equal_type_>
    constexpr void probe_to_upsert_(comparable_key_type_ &&wanted, callback_unused_type_ &&call_unused,
                                    callback_equal_type_ &&call_equal) noexcept {

        offset_t const offset_mask = storage_.slots_count - 1;
        offset_t const initial_offset = hasher_(wanted) & offset_mask;
        offset_t offset = initial_offset;

        // The slot currently under inspection, and the first observed "deleted" slot.
        // The latter stays locked from the moment it is remembered until we commit.
        slot_ref_t current, reusable;
        bool did_find_deleted = false;

        while (true) {
            storage_.retarget_slot(current, offset);
            current.lock();

            if (current.is_populated()) {
                // We have found a filled slot and must check for match:
                // > If matches: unlock the remembered slot, call back, unlock current, exit.
                // > If not: unlock current and jump forward.
                if (equals_(current.key(), wanted)) {
                    if (did_find_deleted) reusable.unlock();
                    call_equal(current);
                    current.unlock();
                    return;
                }
                current.unlock();
                offset = (offset + 1) & offset_mask;
                assert(offset != initial_offset && "Poor hash table usage!");
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
                offset = (offset + 1) & offset_mask;
                assert(offset != initial_offset && "Poor hash table usage!");
                continue;
            }

            // In case of a "free" slot:
            // > If a deleted slot was seen earlier in this probe: reuse it, drop the free one.
            // > Otherwise: take the free slot itself.
            if (did_find_deleted) current.unlock();
            slot_ref_t &chosen = did_find_deleted ? reusable : current;
            call_unused(chosen);
            chosen.mark_populated();
            chosen.unlock();

            // Change these counters afterwards - in a relaxed manner.
            // Somebody else might be already searching for this slot,
            // we don't want them to wait :)
            atomic_add_fetch<offset_t>(storage_.populated_count, 1);
            if (did_find_deleted) atomic_sub_fetch<offset_t>(storage_.deleted_count, 1);
            return;
        }
    }

#pragma endregion Probes
};

#pragma endregion Concurrent Table

#pragma region Aliases

template <typename key_type_, typename value_type_, typename hasher_type_ = default_hash_t,
          typename equals_type_ = equal_to_t, typename allocator_type_ = default_allocator<std::byte>>
using concurrent_hash_map =
    concurrent_hash_table<mapping<key_type_, value_type_>, hasher_type_, equals_type_, allocator_type_>;

template <typename key_type_, typename hasher_type_ = default_hash_t, typename equals_type_ = equal_to_t,
          typename allocator_type_ = default_allocator<std::byte>>
using concurrent_hash_set = concurrent_hash_table<key_type_, hasher_type_, equals_type_, allocator_type_>;

#pragma endregion Aliases

} // namespace ashvardanian::smashtable
