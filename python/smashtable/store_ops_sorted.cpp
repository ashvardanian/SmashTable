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
using sorted_core = basic_avl_tree<value_type_, key_less_t, std::allocator<value_type_>>;

/** @brief One isolation level over that core, then one sharing strategy over that. */
template <typename value_type_>
using sorted_monotonic = monotonic_store<sorted_core<value_type_>>;
template <typename value_type_>
using sorted_snapshot = snapshot_store<sorted_core<value_type_>>;
template <typename value_type_>
using sorted_serializable = serializable_store<sorted_core<value_type_>>;

template <typename store_type_>
using shared_by_lock = locked_store<store_type_>;
template <typename store_type_>
using shared_by_partition = partitioned_store<store_type_, key_hash_t>;

/**
 *  @brief Every table this build carries for the ordered cores.
 *
 *  Spelled out one line per configuration rather than assembled from a loop over knobs, so a
 *  combination that does not exist cannot be named, and so each table is a distinct symbol a debugger
 *  and a profiler can tell apart.
 */
constexpr store_ops_t sorted_map_monotonic_locked = store_bridge<shared_by_lock<sorted_monotonic<entry_t>>>::table();
constexpr store_ops_t sorted_map_monotonic_partitioned =
    store_bridge<shared_by_partition<sorted_monotonic<entry_t>>>::table();
constexpr store_ops_t sorted_map_snapshot_locked = store_bridge<shared_by_lock<sorted_snapshot<entry_t>>>::table();
constexpr store_ops_t sorted_map_snapshot_partitioned =
    store_bridge<shared_by_partition<sorted_snapshot<entry_t>>>::table();

constexpr store_ops_t sorted_set_monotonic_locked =
    store_bridge<shared_by_lock<sorted_monotonic<key_variant_t>>>::table();
constexpr store_ops_t sorted_set_monotonic_partitioned =
    store_bridge<shared_by_partition<sorted_monotonic<key_variant_t>>>::table();
constexpr store_ops_t sorted_set_snapshot_locked =
    store_bridge<shared_by_lock<sorted_snapshot<key_variant_t>>>::table();
constexpr store_ops_t sorted_set_snapshot_partitioned =
    store_bridge<shared_by_partition<sorted_snapshot<key_variant_t>>>::table();

constexpr store_ops_t sorted_map_serializable_locked =
    store_bridge<shared_by_lock<sorted_serializable<entry_t>>>::table();
constexpr store_ops_t sorted_map_serializable_partitioned =
    store_bridge<shared_by_partition<sorted_serializable<entry_t>>>::table();
constexpr store_ops_t sorted_set_serializable_locked =
    store_bridge<shared_by_lock<sorted_serializable<key_variant_t>>>::table();
constexpr store_ops_t sorted_set_serializable_partitioned =
    store_bridge<shared_by_partition<sorted_serializable<key_variant_t>>>::table();

#pragma endregion Instantiations

#pragma region Resolution

store_ops_t const *sorted_store_ops_for(isolation_choice_t isolation, sharing_choice_t sharing,
                                        bool associative) noexcept {
    bool const partitioned = sharing == sharing_choice_t::partitioned_k;
    // A switch rather than a pair of flags: three levels do not fit in one boolean, and naming
    // each arm keeps a configuration that does not exist from being spelled by accident.
    switch (isolation) {
    case isolation_choice_t::serializable_k:
        if (associative) return partitioned ? &sorted_map_serializable_partitioned : &sorted_map_serializable_locked;
        return partitioned ? &sorted_set_serializable_partitioned : &sorted_set_serializable_locked;
    case isolation_choice_t::snapshot_k:
        if (associative) return partitioned ? &sorted_map_snapshot_partitioned : &sorted_map_snapshot_locked;
        return partitioned ? &sorted_set_snapshot_partitioned : &sorted_set_snapshot_locked;
    case isolation_choice_t::monotonic_k:
        if (associative) return partitioned ? &sorted_map_monotonic_partitioned : &sorted_map_monotonic_locked;
        return partitioned ? &sorted_set_monotonic_partitioned : &sorted_set_monotonic_locked;
    }
    return nullptr;
}

#pragma endregion Resolution

} // namespace ashvardanian::smashtable::py
