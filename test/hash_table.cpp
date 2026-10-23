/**
 *  @file test/hash_table.cpp
 *  @author Ash Vardanian
 *  @date August 16, 2026
 *  @brief Test instantiations for the open-addressing hash table. Covers sets and maps over trivial
 *      and heap-allocating key and value types, and per-slot atomics exercised from many threads.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <smashtable/basic_hash_table.hpp>
#include <smashtable/atomic_hash_table.hpp>

#include "harness.hpp"
#include "fixture_coverage.hpp"
#include "surfaces.hpp"
#include "unordered.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Type Aliases

/**
 *  Tests: Baseline probing, growth and tombstone reuse on the cheapest possible key.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Values: ✗.
 */
using trivial_set_t = hash_set<std::size_t>;

/**
 *  Tests: A strongly-typed key reaching the table through its @c std::hash specialization.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Values: ✗.
 */
using strong_set_t = hash_set<trivial_key_t>;

/**
 *  Tests: Baseline map operations, value overwrites, the whole atomic surface.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Values: @c std::size_t.
 */
using trivial_map_t = hash_map<std::size_t, std::size_t>;

/**
 *  Tests: Value construction, destruction and move paths under a lifecycle-checking payload.
 *  Heterogeneous lookup: ✗.
 *  Copy: Non-trivial value.
 *  Memory: Stack.
 *  Values: @c guarded_payload_t.
 */
using guarded_map_t = hash_map<std::size_t, guarded_payload_t>;

/**
 *  Tests: Non-trivial key destructors, the non-memcpy rehash path.
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: Heap.
 *  Memory: Heap.
 *  Values: ✗.
 */
using string_set_t = hash_set<std::string>;

/**
 *  Tests: Dual-heap lifetimes, heterogeneous lookup, atomics over non-trivial elements.
 *  Heterogeneous lookup: ✓, over @c string_view.
 *  Copy: Heap.
 *  Memory: Heap.
 *  Values: @c std::string.
 */
using string_map_t = hash_map<std::string, std::string>;

/**
 *  Tests: Insertion into a table whose allocator refuses to let it grow.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Values: ✗.
 */
using capped_set_t = hash_set<std::size_t, default_hash_t, equal_to_t, stateful_allocator<std::byte>>;

/**
 *  Tests: The same refusal path with a mapped value to place alongside the key.
 *  Heterogeneous lookup: ✗.
 *  Copy: Trivial.
 *  Memory: Stack.
 *  Values: @c std::size_t.
 */
using capped_map_t = hash_map<std::size_t, std::size_t, default_hash_t, equal_to_t, stateful_allocator<std::byte>>;

#pragma endregion Type Aliases

#pragma region Basic Operations Tests

/** Tests that reads and teardown tolerate an empty container */
static void unordered_ops_empty_container_operations() {
    test_unordered_empty_container_operations<trivial_set_t>();
    test_unordered_empty_container_operations<strong_set_t>();
    test_unordered_empty_container_operations<trivial_map_t>();
    test_unordered_empty_container_operations<guarded_map_t>();
    test_unordered_empty_container_operations<string_set_t>();
    test_unordered_empty_container_operations<string_map_t>();
}

/** Tests the insert, lookup, overwrite and erase cycle on a single element */
static void unordered_ops_single_element_operations() {
    test_unordered_single_element_operations<trivial_set_t>();
    test_unordered_single_element_operations<strong_set_t>();
    test_unordered_single_element_operations<trivial_map_t>();
    test_unordered_single_element_operations<guarded_map_t>();
    test_unordered_single_element_operations<string_set_t>();
    test_unordered_single_element_operations<string_map_t>();
}

/** Tests that growth from an unallocated table preserves every element across many rehashes */
static void unordered_ops_growth_through_rehashes() {
    test_unordered_growth_through_rehashes<trivial_set_t>();
    test_unordered_growth_through_rehashes<strong_set_t>();
    test_unordered_growth_through_rehashes<trivial_map_t>();
    test_unordered_growth_through_rehashes<guarded_map_t>();
    // ! The string configurations rehash whole heap-allocated keys, so they run a smaller sweep
    test_unordered_growth_through_rehashes<string_set_t>(1500);
    test_unordered_growth_through_rehashes<string_map_t>(1500);
}

/** Tests that a tombstoned table still finds its survivors, then refills the tombstones */
static void unordered_ops_tombstone_reuse() {
    test_unordered_tombstone_reuse<trivial_set_t>();
    test_unordered_tombstone_reuse<strong_set_t>();
    test_unordered_tombstone_reuse<trivial_map_t>();
    test_unordered_tombstone_reuse<guarded_map_t>();
    test_unordered_tombstone_reuse<string_set_t>(800);
    test_unordered_tombstone_reuse<string_map_t>(800);
}

/** Tests that iteration and for_each each visit every live element exactly once */
static void unordered_ops_full_iteration() {
    test_unordered_full_iteration<trivial_set_t>();
    test_unordered_full_iteration<strong_set_t>();
    test_unordered_full_iteration<trivial_map_t>();
    test_unordered_full_iteration<guarded_map_t>();
    test_unordered_full_iteration<string_set_t>(800);
    test_unordered_full_iteration<string_map_t>(800);
}

/** Tests move construction, move assignment and swap */
static void unordered_ops_moves_and_swaps() {
    test_unordered_moves_and_swaps<trivial_set_t>();
    test_unordered_moves_and_swaps<strong_set_t>();
    test_unordered_moves_and_swaps<trivial_map_t>();
    test_unordered_moves_and_swaps<guarded_map_t>();
    test_unordered_moves_and_swaps<string_set_t>();
    test_unordered_moves_and_swaps<string_map_t>();
}

/** Tests reserve idempotence, explicit rehash, clear and shrink_to_fit */
static void unordered_ops_capacity_management() {
    test_unordered_capacity_management<trivial_set_t>();
    test_unordered_capacity_management<strong_set_t>();
    test_unordered_capacity_management<trivial_map_t>();
    test_unordered_capacity_management<guarded_map_t>();
    test_unordered_capacity_management<string_set_t>();
    test_unordered_capacity_management<string_map_t>();
}

/** Tests that the load factor stays under the documented cap and the counters stay honest */
static void unordered_ops_load_factor_consistency() {
    test_unordered_load_factor_consistency<trivial_set_t>();
    test_unordered_load_factor_consistency<strong_set_t>();
    test_unordered_load_factor_consistency<trivial_map_t>();
    test_unordered_load_factor_consistency<guarded_map_t>();
    test_unordered_load_factor_consistency<string_set_t>(1000);
    test_unordered_load_factor_consistency<string_map_t>(1000);
}

/** Tests lookup by string_view, which only the string-keyed configurations support */
static void unordered_ops_heterogeneous_lookups() {
    test_unordered_heterogeneous_string_view_lookup<string_set_t>();
    test_unordered_heterogeneous_string_view_lookup<string_map_t>();
}

#pragma endregion Basic Operations Tests

#pragma region Saturation and Refusal Tests

/** Tests that a table whose allocator stops supplying memory stops storing instead of spinning */
static void unordered_ops_exhausted_allocator_insertions() {
    test_unordered_exhausted_allocator_insertions<capped_set_t>();
    test_unordered_exhausted_allocator_insertions<capped_map_t>();
}

/** Tests that a reporting insertion separates a fresh key from one already present */
static void unordered_ops_insert_reports_outcome() {
    test_unordered_insert_reports_outcome<trivial_set_t>();
    test_unordered_insert_reports_outcome<strong_set_t>();
    test_unordered_insert_reports_outcome<trivial_map_t>();
    test_unordered_insert_reports_outcome<guarded_map_t>();
    test_unordered_insert_reports_outcome<string_set_t>();
    test_unordered_insert_reports_outcome<string_map_t>();
}

/** Tests that an insertion with nowhere to go is distinguishable from one that met a duplicate */
static void unordered_ops_insert_reports_refusal() {
    test_unordered_insert_reports_refusal<capped_set_t>();
    test_unordered_insert_reports_refusal<capped_map_t>();
}

/** Tests that overwriting a key the table already holds never consults the allocator */
static void unordered_ops_present_key_needs_no_room() {
    test_unordered_present_key_needs_no_room<capped_set_t>();
    test_unordered_present_key_needs_no_room<capped_map_t>();
}

/** Tests that a pinned table of tombstones blames the probe rather than the heap */
static void unordered_ops_pinned_tombstone_saturation() {
    test_unordered_pinned_tombstone_saturation<trivial_map_t>();
    test_unordered_pinned_tombstone_saturation<guarded_map_t>();
    test_unordered_pinned_tombstone_saturation<string_map_t>();
}

/** Tests that the pinned table reports a missing key as a status and through a callback */
static void unordered_ops_pinned_reports_status() {
    test_unordered_pinned_reports_status<trivial_map_t>();
    test_unordered_pinned_reports_status<guarded_map_t>();
    test_unordered_pinned_reports_status<string_map_t>();
}

/** Tests that a table with every slot taken refuses further keys without inflating its size */
static void unordered_ops_full_table_refusals() {
    test_unordered_full_table_refusals<trivial_set_t>();
    test_unordered_full_table_refusals<strong_set_t>();
    test_unordered_full_table_refusals<trivial_map_t>();
    test_unordered_full_table_refusals<guarded_map_t>();
    test_unordered_full_table_refusals<string_set_t>();
    test_unordered_full_table_refusals<string_map_t>();
}

/** Tests that a pinned table with no free slot reports the refusal instead of probing forever */
static void unordered_ops_pinned_saturation() {
    test_unordered_pinned_saturation<trivial_map_t>();
    test_unordered_pinned_saturation<guarded_map_t>();
    test_unordered_pinned_saturation<string_map_t>();
}

/** Tests that a rehash asking for nothing compacts rather than dropping every slot */
static void unordered_ops_rehash_to_nothing() {
    test_unordered_rehash_to_nothing<trivial_set_t>();
    test_unordered_rehash_to_nothing<strong_set_t>();
    test_unordered_rehash_to_nothing<trivial_map_t>();
    test_unordered_rehash_to_nothing<guarded_map_t>();
    test_unordered_rehash_to_nothing<string_set_t>();
    test_unordered_rehash_to_nothing<string_map_t>();
}

/** Tests that an element count no power of two can cover fails instead of yielding one bucket */
static void unordered_ops_unrepresentable_capacity() {
    test_unordered_unrepresentable_capacity<trivial_set_t>();
    test_unordered_unrepresentable_capacity<trivial_map_t>();
    test_unordered_unrepresentable_capacity<string_map_t>();
}

/** Tests that a move into fewer slots than elements refuses up front and leaves the source whole */
static void unordered_ops_move_into_too_few_slots() {
    trivial_set_t set;
    for (std::size_t key = 0; key != 100; ++key) st_verify_(set.upsert(std::size_t {key}));
    trivial_set_t const moved = set.move_to_new(hash_slots_count_t::from_slots(32));
    st_verify_eq_(moved.bucket_count(), 0u);
    st_verify_eq_(set.size(), 100u);
    for (std::size_t key = 0; key != 100; ++key) st_verify_eq_(set.contains(key), true);
}

#pragma endregion Saturation and Refusal Tests

#pragma region Kit Window Tests

/** Both ends of each word and the sign boundaries, as keys of @p key_type_, deduplicated. */
template <typename key_type_>
std::vector<key_type_> extreme_keys() {
    std::uint64_t const words[] = {0,          1,    0x7FFFFFFFull, 0x80000000ull, 0xFFFFFFFFull, 0x7FFFFFFFFFFFFFFFull,
                                   1ull << 63, ~0ull};
    std::vector<key_type_> keys;
    for (std::uint64_t const high : words) {
        if constexpr (std::is_same_v<key_type_, key128_t>)
            for (std::uint64_t const low : words) keys.push_back(key128_t {high, low});
        else keys.push_back(static_cast<key_type_>(high));
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

/** Keys at both ends of the range survive a find, an erase, an upsert into the tombstone it left,
 *  and a @c clear that leaves their bytes behind in the key region. */
template <typename map_type_>
void verify_extreme_keys() {
    using key_t = typename map_type_::key_type;
    using owned_t = typename map_type_::owned_value_type;
    std::vector<key_t> const keys = extreme_keys<key_t>();
    auto const verify_values = [&](map_type_ const &map, std::size_t offset, std::size_t erased_parity) {
        for (std::size_t index = 0; index != keys.size(); ++index) {
            bool const erased = index % 2 == erased_parity;
            st_verify_eq_(map.contains(keys[index]), !erased);
            if (!erased) st_verify_eq_(map.at(keys[index]), index + offset);
        }
    };

    map_type_ map;
    // Reserved up front, so no rehash sweeps the tombstones away before the upserts can reuse them.
    st_verify_(map.reserve(keys.size() * 2) != map_type_::reserve_result_t::failed_k);
    for (std::size_t index = 0; index != keys.size(); ++index) st_verify_(map.upsert(owned_t {keys[index], index}));
    verify_values(map, 0, 2);

    for (std::size_t index = 0; index < keys.size(); index += 2) st_verify_(map.erase(keys[index]));
    st_verify_eq_(map.deleted_count(), (keys.size() + 1) / 2);
    verify_values(map, 0, 0);

    for (std::size_t index = 0; index != keys.size(); ++index)
        st_verify_(map.upsert(owned_t {keys[index], index + keys.size()}));
    st_verify_eq_(map.deleted_count(), 0u, "every re-inserted key must land in a tombstone of its run");
    st_verify_eq_(map.size(), keys.size());
    verify_values(map, keys.size(), 2);

    map.clear();
    st_verify_eq_(map.size(), 0u);
    for (key_t const &key : keys) st_verify_eq_(map.contains(key), false, "a cleared slot must not match its old key");
    for (std::size_t index = 1; index < keys.size(); index += 2) st_verify_(map.upsert(owned_t {keys[index], index}));
    verify_values(map, 0, 0);
}

/** Window lookups over 4-, 8- and 16-byte keys through growth, tombstones and full tables. */
template <typename kit_type_>
void verify_kit(kit_type_) {
    using narrow_set_t = basic_hash_table<std::uint32_t, default_hash_t, equal_to_t, default_allocator_t, kit_type_>;
    using wide_set_t = basic_hash_table<std::uint64_t, default_hash_t, equal_to_t, default_allocator_t, kit_type_>;
    using key128_set_t = basic_hash_table<key128_t, default_hash_t, equal_to_t, default_allocator_t, kit_type_>;
    test_unordered_growth_through_rehashes<narrow_set_t>();
    test_unordered_growth_through_rehashes<wide_set_t>();
    test_unordered_growth_through_rehashes<key128_set_t>();
    test_unordered_tombstone_reuse<narrow_set_t>();
    test_unordered_tombstone_reuse<wide_set_t>();
    test_unordered_tombstone_reuse<key128_set_t>();
    test_unordered_full_table_refusals<narrow_set_t>();
    test_unordered_full_table_refusals<wide_set_t>();
    test_unordered_full_table_refusals<key128_set_t>();
    test_unordered_capacity_management<key128_set_t>();

    verify_extreme_keys<basic_hash_table<mapping<std::uint32_t, std::size_t>, default_hash_t, equal_to_t,
                                         default_allocator_t, kit_type_>>();
    verify_extreme_keys<basic_hash_table<mapping<std::uint64_t, std::size_t>, default_hash_t, equal_to_t,
                                         default_allocator_t, kit_type_>>();
    verify_extreme_keys<
        basic_hash_table<mapping<key128_t, std::size_t>, default_hash_t, equal_to_t, default_allocator_t, kit_type_>>();
}

void verify_through(row_kit_t kit) {
    if (row_kit_supported(kit))
        visit_row_kit(kit, [](row_kit auto kit_instance) noexcept { verify_kit(kit_instance); });
}

static void hash_table_serial_kit() { verify_through(row_kit_t::serial_k); }
static void hash_table_haswell_kit() { verify_through(row_kit_t::haswell_k); }
static void hash_table_skylake_kit() { verify_through(row_kit_t::skylake_k); }
static void hash_table_icelake_kit() { verify_through(row_kit_t::icelake_k); }
static void hash_table_neon_kit() { verify_through(row_kit_t::neon_k); }
static void hash_table_sve_kit() { verify_through(row_kit_t::sve_k); }
static void hash_table_rvv_kit() { verify_through(row_kit_t::rvv_k); }
static void hash_table_v128_kit() { verify_through(row_kit_t::v128_k); }

#pragma endregion Kit Window Tests

#pragma region Multi Match Probe Walk Tests

/** Tests that the walk reaches every version of one key in a shared run, and no foreign key */
static void unordered_visit_every_match() { test_unordered_visit_every_match(); }

/** Tests that a halting callback stops the walk immediately */
static void unordered_visit_early_exit() { test_unordered_visit_early_exit(); }

/** Tests that tombstones between matches don't truncate the walk */
static void unordered_visit_across_tombstones() { test_unordered_visit_across_tombstones(); }

/** Tests that an empty table and an absent key cost zero visits */
static void unordered_visit_empty_table() { test_unordered_visit_empty_table(); }

/** Tests that a run homed in the last slot wraps to the front */
static void unordered_visit_wraparound() { test_unordered_visit_wraparound(); }

/** Tests that a table with no free slot terminates the walk instead of circling forever */
static void unordered_visit_full_table() { test_unordered_visit_full_table(); }

#pragma endregion Multi Match Probe Walk Tests

#pragma region Concurrency Tests

/** Tests concurrent emplace on a pinned table, then concurrent find and contains */
static void unordered_concurrency_emplace_and_find() {
    test_unordered_concurrent_emplace_and_find<trivial_map_t>();
    test_unordered_concurrent_emplace_and_find<guarded_map_t>();
    test_unordered_concurrent_emplace_and_find<string_map_t>(500);
}

/** Tests concurrent update over existing keys, then concurrent erase */
static void unordered_concurrency_update_and_erase() {
    test_unordered_concurrent_update_and_erase<trivial_map_t>();
    test_unordered_concurrent_update_and_erase<guarded_map_t>();
    test_unordered_concurrent_update_and_erase<string_map_t>(500);
}

#pragma endregion Concurrency Tests

#pragma region Batch Atomicity Tests

/** Every range modifier refused at every point it asks for memory or copies an element leaves the
 *  table as it was. */
void hash_table_batch_is_all_or_nothing() {
    using ledger_set_t = hash_set<trivial_key_t, default_hash_t, equal_to_t, stateful_allocator_t>;
    test_every_offered_surface<ledger_set_t>([](allocation_ledger_t &ledger) noexcept {
        return ledger_set_t(default_hash_t {}, equal_to_t {}, stateful_allocator_t(1, ledger));
    });
    test_batch_refuses_copies_whole<hash_set<budgeted_key_t>>();
}

#pragma endregion Batch Atomicity Tests

using counting_set_t = hash_set<trivial_key_t, counting_hash_t, counting_equals_t>;

static void fixture_coverage_hash_lookup_cost_is_bounded() { test_hash_lookup_cost_is_bounded<counting_set_t>(); }

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

int main(int, char **arguments) {
    settings_t const settings = read_settings(arguments[0]);
    install_test_signal_handlers();
    print(probe_machine());
    print(settings);
    test_tally_t tally;

    tally += run_test(settings, "unordered_ops.empty_container_operations", unordered_ops_empty_container_operations);
    tally += run_test(settings, "unordered_ops.single_element_operations", unordered_ops_single_element_operations);
    tally += run_test(settings, "unordered_ops.growth_through_rehashes", unordered_ops_growth_through_rehashes);
    tally += run_test(settings, "unordered_ops.tombstone_reuse", unordered_ops_tombstone_reuse);
    tally += run_test(settings, "unordered_ops.full_iteration", unordered_ops_full_iteration);
    tally += run_test(settings, "unordered_ops.moves_and_swaps", unordered_ops_moves_and_swaps);
    tally += run_test(settings, "unordered_ops.capacity_management", unordered_ops_capacity_management);
    tally += run_test(settings, "unordered_ops.load_factor_consistency", unordered_ops_load_factor_consistency);
    tally += run_test(settings, "unordered_ops.heterogeneous_lookups", unordered_ops_heterogeneous_lookups);

    tally += run_test(settings, "unordered_ops.exhausted_allocator_insertions",
                      unordered_ops_exhausted_allocator_insertions);
    tally += run_test(settings, "unordered_ops.insert_reports_outcome", unordered_ops_insert_reports_outcome);
    tally += run_test(settings, "unordered_ops.insert_reports_refusal", unordered_ops_insert_reports_refusal);
    tally += run_test(settings, "unordered_ops.full_table_refusals", unordered_ops_full_table_refusals);
    tally += run_test(settings, "unordered_ops.present_key_needs_no_room", unordered_ops_present_key_needs_no_room);
    tally += run_test(settings, "unordered_ops.pinned_saturation", unordered_ops_pinned_saturation);
    tally += run_test(settings, "unordered_ops.pinned_tombstone_saturation", unordered_ops_pinned_tombstone_saturation);
    tally += run_test(settings, "unordered_ops.pinned_reports_status", unordered_ops_pinned_reports_status);
    tally += run_test(settings, "unordered_ops.rehash_to_nothing", unordered_ops_rehash_to_nothing);
    tally += run_test(settings, "unordered_ops.unrepresentable_capacity", unordered_ops_unrepresentable_capacity);
    tally += run_test(settings, "unordered_ops.move_into_too_few_slots", unordered_ops_move_into_too_few_slots);
    tally += run_test(settings, "hash_table.serial_kit", hash_table_serial_kit);
    tally += run_test(settings, "hash_table.haswell_kit", hash_table_haswell_kit);
    tally += run_test(settings, "hash_table.skylake_kit", hash_table_skylake_kit);
    tally += run_test(settings, "hash_table.icelake_kit", hash_table_icelake_kit);
    tally += run_test(settings, "hash_table.neon_kit", hash_table_neon_kit);
    tally += run_test(settings, "hash_table.sve_kit", hash_table_sve_kit);
    tally += run_test(settings, "hash_table.rvv_kit", hash_table_rvv_kit);
    tally += run_test(settings, "hash_table.v128_kit", hash_table_v128_kit);

    tally += run_test(settings, "unordered_visit.every_match", unordered_visit_every_match);
    tally += run_test(settings, "unordered_visit.early_exit", unordered_visit_early_exit);
    tally += run_test(settings, "unordered_visit.across_tombstones", unordered_visit_across_tombstones);
    tally += run_test(settings, "unordered_visit.empty_table", unordered_visit_empty_table);
    tally += run_test(settings, "unordered_visit.wraparound", unordered_visit_wraparound);
    tally += run_test(settings, "unordered_visit.full_table", unordered_visit_full_table);

    tally += run_test(settings, "unordered_concurrency.emplace_and_find", unordered_concurrency_emplace_and_find);
    tally += run_test(settings, "unordered_concurrency.update_and_erase", unordered_concurrency_update_and_erase);

    tally += run_test(settings, "fixture_coverage.hash_lookup_cost_is_bounded",
                      fixture_coverage_hash_lookup_cost_is_bounded);

    tally += run_test(settings, "hash_table.batch_is_all_or_nothing", hash_table_batch_is_all_or_nothing);

    return report_test_failures(settings, tally);
}
