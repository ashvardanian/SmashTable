/**
 *  @brief The @c store_ops_t tables for the unordered core, which supplies no ordering at all.
 *  @author Ash Vardanian
 *  @file python/store_ops_hashed.cpp
 *  @date August 18, 2026
 *
 *  A separate translation unit from the ordered core because the cores are what pull in the heavy
 *  templates, and because nothing here shares an instantiation with that file.
 *
 *  Every ordered slot in these tables is null. That is not a refusal invented here - the bridge
 *  probes for the member and the core does not have one, so the class built on these tables installs
 *  no iteration, no scan and no range erase, and reaching for one is an @c AttributeError.
 */
#include "store_ops.hpp"

namespace ashvardanian::smashtable::py {

#pragma region Instantiations

/** @brief The unordered core every class here is built on, once per element shape. */
template <typename value_type_>
using hashed_core = basic_hash_table<value_type_, key_variant_hash_t, key_variant_equal_t, std::allocator<std::byte>>;

template <typename value_type_>
using hashed_monotonic = monotonic_store<hashed_core<value_type_>>;
template <typename value_type_>
using hashed_snapshot = snapshot_store<hashed_core<value_type_>>;
template <typename value_type_>
using hashed_serializable = serializable_store<hashed_core<value_type_>>;
template <typename value_type_>
using hashed_strict_serializable = strict_serializable_store<hashed_core<value_type_>>;

template <typename store_type_>
using shared_by_lock = locked_store<store_type_>;
template <typename store_type_>
using shared_by_partition = partitioned_store<store_type_, key_hash_t>;

constexpr store_ops_t hash_map_monotonic_locked = store_bridge<shared_by_lock<hashed_monotonic<entry_t>>>::table();
constexpr store_ops_t hash_map_monotonic_partitioned =
    store_bridge<shared_by_partition<hashed_monotonic<entry_t>>>::table();
constexpr store_ops_t hash_map_snapshot_locked = store_bridge<shared_by_lock<hashed_snapshot<entry_t>>>::table();
constexpr store_ops_t hash_map_snapshot_partitioned =
    store_bridge<shared_by_partition<hashed_snapshot<entry_t>>>::table();

constexpr store_ops_t hash_set_monotonic_locked =
    store_bridge<shared_by_lock<hashed_monotonic<key_variant_t>>>::table();
constexpr store_ops_t hash_set_monotonic_partitioned =
    store_bridge<shared_by_partition<hashed_monotonic<key_variant_t>>>::table();
constexpr store_ops_t hash_set_snapshot_locked = store_bridge<shared_by_lock<hashed_snapshot<key_variant_t>>>::table();
constexpr store_ops_t hash_set_snapshot_partitioned =
    store_bridge<shared_by_partition<hashed_snapshot<key_variant_t>>>::table();

constexpr store_ops_t hash_map_serializable_locked =
    store_bridge<shared_by_lock<hashed_serializable<entry_t>>>::table();
constexpr store_ops_t hash_map_serializable_partitioned =
    store_bridge<shared_by_partition<hashed_serializable<entry_t>>>::table();
constexpr store_ops_t hash_set_serializable_locked =
    store_bridge<shared_by_lock<hashed_serializable<key_variant_t>>>::table();
constexpr store_ops_t hash_set_serializable_partitioned =
    store_bridge<shared_by_partition<hashed_serializable<key_variant_t>>>::table();

constexpr store_ops_t hash_map_strict_serializable_locked =
    store_bridge<shared_by_lock<hashed_strict_serializable<entry_t>>>::table();
constexpr store_ops_t hash_map_strict_serializable_partitioned =
    store_bridge<shared_by_partition<hashed_strict_serializable<entry_t>>>::table();
constexpr store_ops_t hash_set_strict_serializable_locked =
    store_bridge<shared_by_lock<hashed_strict_serializable<key_variant_t>>>::table();
constexpr store_ops_t hash_set_strict_serializable_partitioned =
    store_bridge<shared_by_partition<hashed_strict_serializable<key_variant_t>>>::table();

#pragma endregion Instantiations

#pragma region Resolution

store_ops_t const *hashed_store_ops_for(isolation_choice_t isolation, sharing_choice_t sharing,
                                        bool associative) noexcept {
    bool const partitioned = sharing == sharing_choice_t::partitioned_k;
    // A switch rather than a pair of flags: four levels do not fit in one boolean, and naming
    // each arm keeps a configuration that does not exist from being spelled by accident.
    switch (isolation) {
    case isolation_choice_t::strict_serializable_k:
        if (associative)
            return partitioned ? &hash_map_strict_serializable_partitioned : &hash_map_strict_serializable_locked;
        return partitioned ? &hash_set_strict_serializable_partitioned : &hash_set_strict_serializable_locked;
    case isolation_choice_t::serializable_k:
        if (associative) return partitioned ? &hash_map_serializable_partitioned : &hash_map_serializable_locked;
        return partitioned ? &hash_set_serializable_partitioned : &hash_set_serializable_locked;
    case isolation_choice_t::snapshot_k:
        if (associative) return partitioned ? &hash_map_snapshot_partitioned : &hash_map_snapshot_locked;
        return partitioned ? &hash_set_snapshot_partitioned : &hash_set_snapshot_locked;
    case isolation_choice_t::monotonic_k:
        if (associative) return partitioned ? &hash_map_monotonic_partitioned : &hash_map_monotonic_locked;
        return partitioned ? &hash_set_monotonic_partitioned : &hash_set_monotonic_locked;
    }
    return nullptr;
}

#pragma endregion Resolution

} // namespace ashvardanian::smashtable::py
