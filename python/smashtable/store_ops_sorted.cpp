/**
 *  @brief The @c store_ops_t tables for the ordered cores, and the resolver every constructor goes through.
 *  @author Ash Vardanian
 *  @file python/smashtable/store_ops_sorted.cpp
 *  @date August 18, 2026
 *
 *  One translation unit per core, because the cores are what pull in the heavy templates: an isolation
 *  level and a sharing strategy each double the instantiation count, and keeping both cores in one file
 *  would make every edit to the binding pay for all of them.
 */
#include "store_ops.hpp"

namespace ashvardanian::smashtable::py {

#pragma region Instantiations

/** @brief The ordered core every class here is built on, once per element shape. */
template <typename value_type_>
using sorted_core_t = basic_avl_tree<value_type_, key_less_t, std::allocator<value_type_>>;

/** @brief One isolation level over that core, then one sharing strategy over that. */
template <typename value_type_>
using sorted_monotonic_t = monotonic_store<sorted_core_t<value_type_>>;
template <typename value_type_>
using sorted_snapshot_t = snapshot_store<sorted_core_t<value_type_>>;

template <typename store_type_>
using shared_by_lock_t = locked_store<store_type_>;
template <typename store_type_>
using shared_by_partition_t = partitioned_store<store_type_, key_hash_t>;

/**
 *  @brief Every table this build carries for the ordered cores.
 *
 *  Spelled out one line per configuration rather than assembled from a loop over knobs, so a
 *  combination that does not exist cannot be named, and so each table is a distinct symbol a debugger
 *  and a profiler can tell apart.
 */
constexpr store_ops_t sorted_map_monotonic_locked = store_bridge<shared_by_lock_t<sorted_monotonic_t<entry_t>>>::table();
constexpr store_ops_t sorted_map_monotonic_partitioned =
    store_bridge<shared_by_partition_t<sorted_monotonic_t<entry_t>>>::table();
constexpr store_ops_t sorted_map_snapshot_locked = store_bridge<shared_by_lock_t<sorted_snapshot_t<entry_t>>>::table();
constexpr store_ops_t sorted_map_snapshot_partitioned =
    store_bridge<shared_by_partition_t<sorted_snapshot_t<entry_t>>>::table();

constexpr store_ops_t sorted_set_monotonic_locked =
    store_bridge<shared_by_lock_t<sorted_monotonic_t<key_variant_t>>>::table();
constexpr store_ops_t sorted_set_monotonic_partitioned =
    store_bridge<shared_by_partition_t<sorted_monotonic_t<key_variant_t>>>::table();
constexpr store_ops_t sorted_set_snapshot_locked =
    store_bridge<shared_by_lock_t<sorted_snapshot_t<key_variant_t>>>::table();
constexpr store_ops_t sorted_set_snapshot_partitioned =
    store_bridge<shared_by_partition_t<sorted_snapshot_t<key_variant_t>>>::table();

#pragma endregion Instantiations

#pragma region Resolution

store_ops_t const *sorted_store_ops_for(isolation_choice_t isolation, sharing_choice_t sharing,
                                        bool associative) noexcept {
    bool const snapshot = isolation == isolation_choice_t::snapshot_k;
    bool const partitioned = sharing == sharing_choice_t::partitioned_k;
    if (associative) {
        if (snapshot) return partitioned ? &sorted_map_snapshot_partitioned : &sorted_map_snapshot_locked;
        return partitioned ? &sorted_map_monotonic_partitioned : &sorted_map_monotonic_locked;
    }
    if (snapshot) return partitioned ? &sorted_set_snapshot_partitioned : &sorted_set_snapshot_locked;
    return partitioned ? &sorted_set_monotonic_partitioned : &sorted_set_monotonic_locked;
}

#pragma endregion Resolution

} // namespace ashvardanian::smashtable::py
