/**
 *  @brief Test instantiations for the thread-safety wrappers. Both wrappers forward to a transactional AVL tree, so the
 *      same suites apply unchanged.
 *  @author Ash Vardanian
 *  @file scripts/test_partitioned.cpp
 *  @date August 16, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define ST_STRICT_CALLBACK_CHECKS_ 1

#include <shared_mutex> // `std::shared_mutex`, to keep the substitution path covered

#include <smashtable/basic_avl_tree.hpp>
#include <smashtable/locked_store.hpp>
#include <smashtable/partitioned_store.hpp>
#include <smashtable/monotonic_store.hpp>
#include <smashtable/snapshot_store.hpp>

#include "test.hpp"
#include "test_basic.hpp"
#include "test_surface_parity.hpp"
#include "test_consistency.hpp"
#include "test_sharded_concurrency.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

namespace {

#pragma region Type Aliases

using tree_trivial_set_t = monotonic_avl_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;
using tree_composite_set_t =
    monotonic_avl_set<composite_key_t, composite_key_compare_t, std::allocator<composite_key_t>>;
using tree_trivial_map_t =
    monotonic_avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;
using tree_composite_map_t = monotonic_avl_map<composite_key_t, guarded_payload_t, composite_key_compare_t,
                                               std::allocator<mapping<composite_key_t, guarded_payload_t>>>;

/** Sharded across sixteen independently locked partitions. */
using transactional_trivial_set_t = partitioned_set<tree_trivial_set_t>;
using transactional_composite_set_t = partitioned_store<tree_composite_set_t>;
using transactional_trivial_map_t = partitioned_map<tree_trivial_map_t>;

using refusing_tree_map_t = refusing_store<tree_trivial_map_t>;
using sharded_refusing_map_t = partitioned_store<refusing_tree_map_t>;
using transactional_composite_map_t = partitioned_store<tree_composite_map_t>;

/** Sharded, and built around a comparator instance rather than a default-constructed one. */
using tree_tracking_set_t = monotonic_avl_set<trivial_key_t, stateful_comparator_t, stateful_allocator_t>;
using self_tracking_set_t = partitioned_store<tree_tracking_set_t>;

/**
 *  Sharded on @c std::shared_mutex rather than the library's own.
 *
 *  The mutex is a template parameter, so the default is a choice and not the only thing that fits;
 *  running the concurrency suites against the standard one is what keeps the substitution honest.
 */
using standard_mutex_set_t = partitioned_store<tree_composite_set_t, hash<composite_key_t>, std::shared_mutex, 16>;
using standard_mutex_map_t = partitioned_store<tree_composite_map_t, hash<composite_key_t>, std::shared_mutex, 16>;

/**
 *  Wrapped around a hash-backed store, which offers no ordering at all.
 *
 *  The wrappers forward the ordered surface, so an unordered inner store must lose it at overload
 *  resolution rather than inside an instantiation of a body that cannot compile.
 */
using hash_store_t = monotonic_hash_set<trivial_key_t, hash<trivial_key_t>, equal_to_t, std::allocator<std::byte>>;

static_assert(offers_ordered_surface<tree_trivial_set_t>, "the tree-backed store is ordered");
static_assert(offers_ordered_surface<locked_store<tree_trivial_set_t>>,
              "wrapping an ordered store must keep the ordered surface");
static_assert(offers_ordered_surface<partitioned_store<tree_trivial_set_t>>,
              "sharding an ordered store must keep the ordered surface");

static_assert(!offers_ordered_surface<hash_store_t>, "a hash-backed store has no ordering to offer");
static_assert(!offers_ordered_surface<locked_store<hash_store_t>>,
              "the lock wrapper must not claim an ordering its inner store denies");
static_assert(!offers_ordered_surface<partitioned_store<hash_store_t>>,
              "the partitioned wrapper must not claim an ordering its inner store denies");

/** Backed by a weight-balanced core, the only one that sums the subtree counts @c select descends on. */
using ranked_set_t = snapshot_wb_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;
using monotonic_ranked_set_t = monotonic_wb_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;

/**
 *  @brief A store whose enumeration is spelled as a walk of the whole keyspace.
 *
 *  The wrapper suites walk this rather than a shipped store, so what they exercise is the forwarding
 *  and the locking around an enumeration, not the visibility rules a store applies inside its own.
 */
struct enumerable_set_t : tree_trivial_set_t {
    using base_t = tree_trivial_set_t;

    /** @brief One past the largest identifier any suite here builds a key from. */
    static constexpr trivial_id_t keyspace_k = 1u << 20;

    enumerable_set_t() noexcept = default;
    enumerable_set_t(base_t &&other) noexcept : base_t(std::move(other)) {}
    enumerable_set_t(enumerable_set_t &&) noexcept = default;
    enumerable_set_t &operator=(enumerable_set_t &&) noexcept = default;

    [[nodiscard]] static expected<enumerable_set_t> make() noexcept {
        expected<base_t> built = base_t::make();
        if (!built) return built.status();
        return enumerable_set_t {std::move(*built)};
    }

    template <typename callback_type_>
    [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept {
        return base_t::range(trivial_key_t {0}, trivial_key_t {keyspace_k}, std::forward<callback_type_>(callback));
    }
};

static_assert(offers_order_statistics<ranked_set_t>, "the weight-balanced store answers by ordinal");
static_assert(offers_order_statistics<locked_store<monotonic_ranked_set_t>>,
              "the ordinal surface travels through the wrapper whichever store family carries it");
static_assert(offers_order_statistics<partitioned_store<monotonic_ranked_set_t>>,
              "the ordinal surface travels through the wrapper whichever store family carries it");
static_assert(offers_order_statistics<locked_store<ranked_set_t>>,
              "wrapping an order-statistics store must keep the ordinal surface");
static_assert(offers_order_statistics<partitioned_store<ranked_set_t>>,
              "sharding an order-statistics store must keep the ordinal surface");

static_assert(!offers_order_statistics<tree_trivial_set_t>, "the AVL core sums no subtree counts");
static_assert(!offers_order_statistics<locked_store<tree_trivial_set_t>>,
              "the lock wrapper must not claim ordinals its inner core cannot answer");
static_assert(!offers_order_statistics<partitioned_store<tree_trivial_set_t>>,
              "the partitioned wrapper must not claim ordinals its inner core cannot answer");
static_assert(!offers_order_statistics<locked_store<hash_store_t>>, "an unordered core has no ordinals at all");
static_assert(!offers_order_statistics<partitioned_store<hash_store_t>>, "an unordered core has no ordinals at all");

static_assert(offers_enumeration<enumerable_set_t>, "the fixture store enumerates");
static_assert(offers_enumeration<locked_store<enumerable_set_t>>,
              "wrapping an enumerable store must keep the enumeration");
static_assert(offers_enumeration<partitioned_store<enumerable_set_t>>,
              "sharding an enumerable store must keep the enumeration");

static_assert(offers_enumeration<tree_trivial_set_t>, "a transactional store walks its own members");
static_assert(offers_enumeration<locked_store<tree_trivial_set_t>>,
              "the lock wrapper forwards the enumeration its inner store offers");
static_assert(offers_enumeration<partitioned_store<tree_trivial_set_t>>,
              "the partitioned wrapper forwards the enumeration its inner store offers");

/** Both wrappers over the enumerable fixture, which is what the enumeration suites walk. */
using locked_enumerable_set_t = locked_store<enumerable_set_t>;
using partitioned_enumerable_set_t = partitioned_store<enumerable_set_t>;

/** One shared mutex over the whole collection, reached through the shape-naming aliases. */
using transactional_tracking_set_t = locked_set<tree_trivial_set_t>;
using transactional_tracking_map_t = locked_map<tree_trivial_map_t>;

/**
 *  The shape-naming aliases wrap one class each, so what distinguishes them is which stores they
 *  accept: an alias that took every store would name the same type as its sibling and catch nothing.
 */
template <typename store_type_>
constexpr bool names_locked_set = requires { typename locked_set<store_type_>; };
template <typename store_type_>
constexpr bool names_locked_map = requires { typename locked_map<store_type_>; };
template <typename store_type_>
constexpr bool names_partitioned_set = requires { typename partitioned_set<store_type_>; };
template <typename store_type_>
constexpr bool names_partitioned_map = requires { typename partitioned_map<store_type_>; };

static_assert(names_locked_set<tree_trivial_set_t>, "the set alias takes a store of plain keys");
static_assert(names_locked_map<tree_trivial_map_t>, "the map alias takes a store of mappings");
static_assert(!names_locked_map<tree_trivial_set_t>, "the map alias must refuse a set-shaped store");
static_assert(!names_locked_set<tree_trivial_map_t>, "the set alias must refuse a map-shaped store");
static_assert(names_partitioned_set<tree_trivial_set_t>, "the set alias takes a store of plain keys");
static_assert(names_partitioned_map<tree_trivial_map_t>, "the map alias takes a store of mappings");
static_assert(!names_partitioned_map<tree_trivial_set_t>, "the map alias must refuse a set-shaped store");
static_assert(!names_partitioned_set<tree_trivial_map_t>, "the set alias must refuse a map-shaped store");

/**
 *  A wrapper names itself @c store_t, which is what every engine calls its own self-alias, so one name
 *  means one thing whichever kind of store a generic caller is handed.
 */
static_assert(std::is_same<typename locked_store<tree_trivial_set_t>::store_t, locked_store<tree_trivial_set_t>>(),
              "a store names itself");
static_assert(
    std::is_same<typename partitioned_store<tree_trivial_set_t>::store_t, partitioned_store<tree_trivial_set_t>>(),
    "a store names itself");

#pragma endregion Type Aliases

#pragma region Surface Parity

/** The snapshot-isolated families, which carry the surfaces the monotonic ones never had. */
using snapshot_trivial_set_t = snapshot_avl_set<trivial_key_t, std::less<trivial_key_t>, std::allocator<trivial_key_t>>;
using snapshot_trivial_map_t =
    snapshot_avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;
using snapshot_hash_store_t =
    snapshot_hash_set<trivial_key_t, hash<trivial_key_t>, equal_to_t, std::allocator<std::byte>>;

/** Sharded and stamped, so sixteen partitions draw one snapshot and publish under one stamp. */
using sharded_snapshot_map_t = partitioned_store<snapshot_trivial_map_t>;

/**
 *  Every public surface of every shipped store, against every wrapper nesting.
 *
 *  One line per store: the fold names the surface, the wrapper and the store in the diagnostic, so a
 *  forward that goes missing fails here rather than at whatever call site happened to want it.
 */
static_assert(every_wrapper_keeps_surfaces<tree_trivial_set_t>, "a wrapper must keep what its store offers");
static_assert(every_wrapper_keeps_surfaces<tree_trivial_map_t>, "a wrapper must keep what its store offers");
static_assert(every_wrapper_keeps_surfaces<monotonic_ranked_set_t>, "a wrapper must keep what its store offers");
static_assert(every_wrapper_keeps_surfaces<hash_store_t>, "a wrapper must keep what its store offers");
static_assert(every_wrapper_keeps_surfaces<enumerable_set_t>, "a wrapper must keep what its store offers");
static_assert(every_wrapper_keeps_surfaces<snapshot_trivial_set_t>, "a wrapper must keep what its store offers");
static_assert(every_wrapper_keeps_surfaces<snapshot_trivial_map_t>, "a wrapper must keep what its store offers");
static_assert(every_wrapper_keeps_surfaces<ranked_set_t>, "a wrapper must keep what its store offers");
static_assert(every_wrapper_keeps_surfaces<snapshot_hash_store_t>, "a wrapper must keep what its store offers");

/** A wrapper meant to be invisible must not change what an outer wrapper concludes about isolation. */
static_assert(nesting_preserves_isolation<tree_trivial_set_t>, "a transparent wrapper decides nothing");
static_assert(nesting_preserves_isolation<snapshot_trivial_set_t>, "a transparent wrapper decides nothing");
static_assert(nesting_preserves_isolation<snapshot_trivial_map_t>, "a transparent wrapper decides nothing");
static_assert(nesting_preserves_isolation<ranked_set_t>, "a transparent wrapper decides nothing");

/** Sharding a stamped store keeps its promise whole, and a wrapper between the two takes nothing from it. */
static_assert(partitioned_store<snapshot_trivial_map_t>::isolation_k == snapshot_trivial_map_t::isolation_k,
              "one clock across partitions carries the inner store's isolation");
static_assert(partitioned_store<locked_store<snapshot_trivial_map_t>>::isolation_k ==
                  snapshot_trivial_map_t::isolation_k,
              "one clock across partitions carries the inner store's isolation");

/** A transaction has to survive being stored, which is what a deleted move assignment takes away. */
static_assert(transaction_moves_as_a_value<locked_store<tree_trivial_set_t>>, "a transaction moves as a value");
static_assert(transaction_moves_as_a_value<partitioned_store<tree_trivial_set_t>>, "a transaction moves as a value");
static_assert(transaction_moves_as_a_value<partitioned_store<locked_store<snapshot_trivial_map_t>>>,
              "a transaction moves as a value");

/** The wrappers stay in the group the concept describes, which now also asks how a transaction moves. */
static_assert(optimistically_concurrent_store<locked_store<tree_trivial_set_t>>, "the lock wrapper stays in the group");
static_assert(optimistically_concurrent_store<partitioned_store<tree_trivial_set_t>>,
              "the partitioned wrapper stays in the group");
static_assert(optimistically_concurrent_store<partitioned_store<locked_store<snapshot_trivial_map_t>>>,
              "the nesting stays in the group");

#pragma endregion Surface Parity

#pragma region Suites

/** @brief Walkers crossing a sharded map while an eraser churns it, with heap-owning keys. */
static void sharded_concurrency_walks_never_race_erasures() {
    test_sharded_walks_never_race_erasures<transactional_composite_map_t>();
    test_sharded_walks_never_race_erasures<standard_mutex_map_t>();
}

/** @brief The same suites against @c std::shared_mutex, so the substituted lock stays exercised. */
static void sharded_concurrency_standard_mutex_substitutes() {
    test_empty_container_operations<standard_mutex_set_t>();
    test_single_element_operations<standard_mutex_set_t>();
    test_sharded_stage_unwinds_on_partial_failure<standard_mutex_map_t>();
}

/** @brief Concurrent transaction opens must never share a generation. */
static void sharded_concurrency_distinct_generations() {
    test_concurrent_transactions_get_distinct_generations<tree_trivial_map_t>();
    test_concurrent_transactions_get_distinct_generations<tree_trivial_set_t>();
}

/** @brief A refused stage must leave no partition holding an unpublishable reservation. */
static void sharded_concurrency_stage_unwinds_on_partial_failure() {
    test_sharded_stage_unwinds_on_partial_failure<transactional_trivial_map_t>();
    test_sharded_stage_unwinds_on_partial_failure<sharded_refusing_map_t>();
}

/**
 *  @brief What a commit spanning partitions looks like to a reader, at both levels a shard set reaches.
 *
 *  A stamped part keeps its promise whole through the sharding, and a part without a stamp is capped
 *  at @c read_committed_k - so the same walk must tear against one and never against the other.
 */
static void sharded_concurrency_commit_spans_partitions() {
    test_commit_spans_partitions_matches_isolation<transactional_trivial_map_t>();
    test_commit_spans_partitions_matches_isolation<sharded_snapshot_map_t>();
}

/** @brief Tests operations on empty container don't crash */
static void basic_ops_empty_container_operations() {
    test_empty_container_operations<transactional_trivial_set_t>();
    test_empty_container_operations<transactional_tracking_set_t>();
    test_empty_container_operations<transactional_composite_set_t>();
    test_empty_container_operations<transactional_trivial_map_t>();
    test_empty_container_operations<transactional_tracking_map_t>();
    test_empty_container_operations<transactional_composite_map_t>();
}

/** @brief Tests operations on single-element container */
static void basic_ops_single_element_operations() {
    test_single_element_operations<transactional_trivial_set_t>();
    test_single_element_operations<transactional_tracking_set_t>();
    test_single_element_operations<transactional_composite_set_t>();
    test_single_element_operations<transactional_trivial_map_t>();
    test_single_element_operations<transactional_tracking_map_t>();
    test_single_element_operations<transactional_composite_map_t>();
}

/** @brief Tests insertion patterns (ascending, descending, random) for all AVL containers */
static void basic_ops_insertion_patterns() {
    test_basic_insertion_patterns<transactional_trivial_set_t>();
    test_basic_insertion_patterns<transactional_tracking_set_t>();
    test_basic_insertion_patterns<transactional_composite_set_t>();
    test_basic_insertion_patterns<transactional_trivial_map_t>();
    test_basic_insertion_patterns<transactional_tracking_map_t>();
    test_basic_insertion_patterns<transactional_composite_map_t>();
}

/** @brief Tests bulk insertion from iterators for all AVL containers */
static void basic_ops_bulk_insertion_iterators() {
    test_bulk_insertion_from_iterators<transactional_trivial_set_t>();
    test_bulk_insertion_from_iterators<transactional_tracking_set_t>();
    test_bulk_insertion_from_iterators<transactional_composite_set_t>();
    test_bulk_insertion_from_iterators<transactional_trivial_map_t>();
    test_bulk_insertion_from_iterators<transactional_tracking_map_t>();
    test_bulk_insertion_from_iterators<transactional_composite_map_t>();
}

/** @brief Tests bulk upsert correctly overwrites duplicate keys */
static void basic_ops_bulk_upsert_with_duplicate_pairs() {
    test_bulk_upsert_with_duplicates<transactional_trivial_map_t>();
    test_bulk_upsert_with_duplicates<transactional_tracking_map_t>();
}

/** @brief Tests range queries on committed HEAD state */
static void basic_ops_range_query_head_state() {
    test_range_query_head_state<transactional_trivial_set_t>();
    test_range_query_head_state<transactional_tracking_set_t>();
    test_range_query_head_state<transactional_composite_set_t>();
    test_range_query_head_state<transactional_trivial_map_t>();
    test_range_query_head_state<transactional_tracking_map_t>();
    test_range_query_head_state<transactional_composite_map_t>();
}

/** @brief Tests erase_range on committed HEAD state */
static void basic_ops_erase_range_head_state() {
    test_erase_range_head_state<transactional_trivial_set_t>();
    test_erase_range_head_state<transactional_tracking_set_t>();
    test_erase_range_head_state<transactional_composite_set_t>();
    test_erase_range_head_state<transactional_trivial_map_t>();
    test_erase_range_head_state<transactional_tracking_map_t>();
    test_erase_range_head_state<transactional_composite_map_t>();
}

/** @brief Tests heterogeneous lookup for composite and heavy key types */
static void basic_ops_heterogeneous_lookups() {
    test_heterogeneous_composite_find<transactional_composite_set_t>();
    test_heterogeneous_composite_find<transactional_composite_map_t>();
}

static void transactional_consistency_empty_transaction_commit() {
    test_empty_transaction_commit<transactional_trivial_set_t>();
    test_empty_transaction_commit<transactional_tracking_set_t>();
    test_empty_transaction_commit<transactional_composite_set_t>();
    test_empty_transaction_commit<transactional_trivial_map_t>();
    test_empty_transaction_commit<transactional_tracking_map_t>();
    test_empty_transaction_commit<transactional_composite_map_t>();
}

static void transactional_consistency_no_dirty_reads_multi_key() {
    test_no_dirty_reads_multi_key<transactional_trivial_map_t>();
    test_no_dirty_reads_multi_key<transactional_tracking_map_t>();
    test_no_dirty_reads_multi_key<transactional_composite_map_t>();
}

static void transactional_consistency_new_transaction_sees_nothing_staged() {
    test_new_transaction_sees_nothing_staged<transactional_trivial_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_tracking_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_set_t>();
    test_new_transaction_sees_nothing_staged<transactional_trivial_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_tracking_map_t>();
    test_new_transaction_sees_nothing_staged<transactional_composite_map_t>();
}

static void transactional_consistency_committed_immediately_visible() {
    test_committed_immediately_visible<transactional_trivial_map_t>();
    test_committed_immediately_visible<transactional_tracking_map_t>();
    test_committed_immediately_visible<transactional_composite_map_t>();
}

static void transactional_consistency_multi_key_atomicity_10_keys() {
    test_multi_key_atomicity_10_keys<transactional_trivial_map_t>();
    test_multi_key_atomicity_10_keys<transactional_tracking_map_t>();
    test_multi_key_atomicity_10_keys<transactional_composite_map_t>();
}

static void transactional_consistency_rollback_makes_all_invisible() {
    test_rollback_makes_all_invisible<transactional_trivial_map_t>();
    test_rollback_makes_all_invisible<transactional_tracking_map_t>();
    test_rollback_makes_all_invisible<transactional_composite_map_t>();
}

static void transactional_consistency_range_query_sees_atomic_boundaries() {
    test_range_query_sees_atomic_boundaries<transactional_trivial_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_tracking_set_t>();
    test_range_query_sees_atomic_boundaries<transactional_composite_set_t>();
}

static void transactional_consistency_fractured_read_prevention() {
    test_fractured_read_prevention<transactional_trivial_map_t>();
    test_fractured_read_prevention<transactional_tracking_map_t>();
    test_fractured_read_prevention<transactional_composite_map_t>();
}

static void transactional_consistency_sequential_updates_never_regress() {
    test_sequential_updates_never_regress<transactional_trivial_map_t>();
    test_sequential_updates_never_regress<transactional_tracking_map_t>();
}

/** @brief Ordering of mapped values is only meaningful where they are numbers, so @c int maps only. */
static void transactional_consistency_transaction_commits_maintain_order() {
    test_transaction_commits_maintain_order<transactional_trivial_map_t>();
    test_transaction_commits_maintain_order<transactional_tracking_map_t>();
}

static void transactional_consistency_concurrent_transactions_on_same_key() {
    test_concurrent_transactions_on_same_key<transactional_trivial_map_t>();
    test_concurrent_transactions_on_same_key<transactional_tracking_map_t>();
    test_concurrent_transactions_on_same_key<transactional_composite_map_t>();
}

static void transactional_consistency_multi_key_conflict_any_key_fails() {
    test_multi_key_conflict_any_key_fails<transactional_trivial_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_tracking_map_t>();
    test_multi_key_conflict_any_key_fails<transactional_composite_map_t>();
}

static void transactional_consistency_watch_detects_external_direct_modification() {
    test_watch_detects_external_direct_modification<transactional_trivial_map_t>();
    test_watch_detects_external_direct_modification<transactional_tracking_map_t>();
    test_watch_detects_external_direct_modification<transactional_composite_map_t>();
}

static void transactional_consistency_watch_detects_staged_invisible_writes() {
    test_watch_detects_staged_invisible_writes<transactional_trivial_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_tracking_map_t>();
    test_watch_detects_staged_invisible_writes<transactional_composite_map_t>();
}

static void transactional_consistency_watch_detects_staged_writes_of_older_generation() {
    test_watch_detects_staged_writes_of_older_generation<transactional_trivial_map_t>();
    test_watch_detects_staged_writes_of_older_generation<transactional_tracking_map_t>();
    test_watch_detects_staged_writes_of_older_generation<transactional_composite_map_t>();
}

static void transactional_consistency_disjoint_keys_both_succeed() {
    test_disjoint_keys_both_succeed<transactional_trivial_map_t>();
    test_disjoint_keys_both_succeed<transactional_tracking_map_t>();
    test_disjoint_keys_both_succeed<transactional_composite_map_t>();
}

static void transactional_consistency_lost_update_matches_isolation() {
    test_lost_update_matches_isolation<transactional_trivial_map_t>();
    test_lost_update_matches_isolation<transactional_tracking_map_t>();
    test_lost_update_matches_isolation<transactional_composite_map_t>();
}

static void transactional_consistency_write_skew_matches_isolation() {
    test_write_skew_matches_isolation<transactional_trivial_map_t>();
    test_write_skew_matches_isolation<transactional_tracking_map_t>();
}

static void transactional_consistency_read_conflict_matches_isolation() {
    test_read_conflict_matches_isolation<transactional_trivial_map_t>();
    test_read_conflict_matches_isolation<transactional_tracking_map_t>();
}

static void transactional_consistency_repeated_read_matches_isolation() {
    test_repeated_read_matches_isolation<transactional_trivial_map_t>();
    test_repeated_read_matches_isolation<transactional_tracking_map_t>();
    test_repeated_read_matches_isolation<transactional_composite_map_t>();
}

static void transactional_consistency_repeated_range_matches_isolation() {
    test_repeated_range_matches_isolation<transactional_trivial_set_t>();
    test_repeated_range_matches_isolation<transactional_tracking_set_t>();
    test_repeated_range_matches_isolation<transactional_composite_set_t>();
    test_repeated_range_matches_isolation<transactional_trivial_map_t>();
    test_repeated_range_matches_isolation<transactional_tracking_map_t>();
    test_repeated_range_matches_isolation<transactional_composite_map_t>();
}

static void transactional_consistency_delete_visibility() {
    test_delete_visibility<transactional_trivial_set_t>();
    test_delete_visibility<transactional_tracking_set_t>();
    test_delete_visibility<transactional_composite_set_t>();
}

static void transactional_consistency_reset_clears_transaction_state() {
    test_reset_clears_transaction_state<transactional_trivial_map_t>();
    test_reset_clears_transaction_state<transactional_tracking_map_t>();
    test_reset_clears_transaction_state<transactional_composite_map_t>();
}

static void transactional_consistency_stateful_comparator_is_consulted() {
    test_stateful_comparator_is_consulted<self_tracking_set_t>();
}

static void transactional_consistency_group_commits_participants_together() {
    test_group_commits_participants_together<transactional_trivial_map_t>();
    test_group_commits_participants_together<transactional_composite_map_t>();
}

static void transactional_consistency_group_unwinds_every_participant_on_conflict() {
    test_group_unwinds_every_participant_on_conflict<transactional_trivial_map_t>();
    test_group_unwinds_every_participant_on_conflict<transactional_composite_map_t>();
}

static void transactional_consistency_find_does_not_watch() {
    test_find_does_not_watch<transactional_trivial_map_t>();
    test_find_does_not_watch<transactional_composite_map_t>();
}

#pragma endregion Suites

#pragma region Forwarded Surface

/** @brief Builds a store through its own @c make, since a wrapper has no other public constructor. */
template <typename store_type_>
static expected<store_type_> built_store() {
    expected<store_type_> made = store_type_::make();
    st_verify_((made) && "the store under test must build");
    return made;
}

/**
 *  @brief A strict insert through a wrapper must refuse an occupied key with the bare store's status.
 *
 *  The status is read off the bare store rather than spelled out, so the suite stays true whichever
 *  refusal a store family picks - what is under test is that the wrapper hands the same one back.
 */
template <typename wrapper_type_, typename inner_type_>
static void test_forwarded_strict_insert() {

    using member_t = typename wrapper_type_::value_type;

    expected<inner_type_> bare_made = built_store<inner_type_>();
    inner_type_ &bare = *bare_made;
    st_verify_(bare.upsert(trivial_id_to_member<member_t>(7)));
    status_t const bare_refusal = bare.insert(trivial_id_to_member<member_t>(7));
    st_verify_eq_(bare_refusal, status_t::key_already_exists_k);

    expected<wrapper_type_> wrapped_made = built_store<wrapper_type_>();
    wrapper_type_ &wrapped = *wrapped_made;
    st_verify_(wrapped.upsert(trivial_id_to_member<member_t>(7)));
    st_verify_eq_(wrapped.insert(trivial_id_to_member<member_t>(7)), bare_refusal);
    st_verify_eq_(wrapped.size(), 1u);
    st_verify_(wrapped.insert(trivial_id_to_member<member_t>(8)));
    st_verify_eq_(wrapped.size(), 2u);

    // The same refusal staged rather than published, which is the level a binding reaches for.
    expected<typename inner_type_::transaction_t> bare_writer = bare.transaction();
    st_verify_((bare_writer) && "the bare store must open a transaction");
    status_t const bare_staged_refusal = bare_writer->insert(trivial_id_to_member<member_t>(7));
    st_verify_eq_(bare_staged_refusal, status_t::key_already_exists_k);

    expected<typename wrapper_type_::transaction_t> wrapped_writer = wrapped.transaction();
    st_verify_((wrapped_writer) && "the wrapped store must open a transaction");
    st_verify_eq_(wrapped_writer->insert(trivial_id_to_member<member_t>(7)), bare_staged_refusal);
    st_verify_(wrapped_writer->insert(trivial_id_to_member<member_t>(9)));
    st_verify_(wrapped_writer->stage());
    st_verify_(wrapped_writer->commit());
    st_verify_eq_(wrapped.size(), 3u);
}

/** @brief A strict update through a wrapper must refuse an absent key with the bare store's status. */
template <typename wrapper_type_, typename inner_type_>
static void test_forwarded_strict_update() {

    using member_t = typename wrapper_type_::value_type;

    expected<inner_type_> bare_made = built_store<inner_type_>();
    inner_type_ &bare = *bare_made;
    status_t const bare_refusal = bare.update(trivial_id_to_member<member_t>(3));
    st_verify_eq_(bare_refusal, status_t::key_not_found_k);

    expected<wrapper_type_> wrapped_made = built_store<wrapper_type_>();
    wrapper_type_ &wrapped = *wrapped_made;
    st_verify_eq_(wrapped.update(trivial_id_to_member<member_t>(3)), bare_refusal);
    st_verify_eq_(wrapped.size(), 0u);

    st_verify_(wrapped.upsert(trivial_id_to_member<member_t>(3)));
    st_verify_(wrapped.update(trivial_id_to_member<member_t>(3)));
    st_verify_eq_(wrapped.size(), 1u);

    expected<typename wrapper_type_::transaction_t> writer = wrapped.transaction();
    st_verify_((writer) && "the wrapped store must open a transaction");
    st_verify_eq_(writer->update(trivial_id_to_member<member_t>(4)), status_t::key_not_found_k);
    st_verify_(writer->update(trivial_id_to_member<member_t>(3)));
}

/** @brief Ordinals through a wrapper must name the same elements the merged order does. */
template <typename wrapper_type_>
static void test_forwarded_order_statistics(std::size_t count = 64) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    for (std::size_t identifier = 0; identifier != count; ++identifier) {
        member_t selected {};
        bool selected_one = false;
        st_verify_(container.select(
            identifier,
            [&](member_t const &member) noexcept {
                selected = member;
                selected_one = true;
            },
            []() noexcept {}));
        st_verify_((selected_one) && "every ordinal below the size must name an element");
        st_verify_(selected == trivial_id_to_key<member_t>(identifier));

        std::size_t position = count;
        st_verify_(container.rank(
            trivial_id_to_key<member_t>(identifier), [&](std::size_t found) noexcept { position = found; },
            []() noexcept {}));
        st_verify_eq_(position, identifier);
    }

    bool missed = false;
    st_verify_(container.select(count, [](member_t const &) noexcept {}, [&]() noexcept { missed = true; }));
    st_verify_((missed) && "an ordinal at the size must find nothing");

    missed = false;
    st_verify_(container.rank(
        trivial_id_to_key<member_t>(count + 1), [](std::size_t) noexcept {}, [&]() noexcept { missed = true; }));
    st_verify_((missed) && "a key that is not there has no rank");
}

/** @brief A sweep through a wrapper must answer with a count and leave every survivor readable. */
template <typename wrapper_type_>
static void test_forwarded_vacuum(std::size_t count = 32) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));
    for (std::size_t identifier = 0; identifier < count; identifier += 2)
        st_verify_(container.erase(trivial_id_to_key<member_t>(identifier)));

    expected<std::size_t> const reclaimed = container.vacuum();
    st_verify_((reclaimed) && "a sweep that cannot refuse must still answer with a count");
    st_verify_eq_(container.size(), count / 2);
    for (std::size_t identifier = 1; identifier < count; identifier += 2) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(identifier)), true);
    }
    for (std::size_t identifier = 0; identifier < count; identifier += 2) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(identifier)), false);
    }
}

/** @brief A windowed sweep must reclaim its window and leave the rest of the keyspace readable. */
template <typename wrapper_type_>
static void test_forwarded_windowed_vacuum(std::size_t count = 32) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(container.erase(trivial_id_to_key<member_t>(identifier)));

    expected<std::size_t> const reclaimed =
        container.vacuum(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(count / 2));
    st_verify_((reclaimed) && "a windowed sweep must answer with a count");
    st_verify_eq_(container.size(), 0u);
}

/** @brief A range rewrite through a wrapper must touch the window and nothing beside it. */
template <typename wrapper_type_>
static void test_forwarded_update_range(std::size_t count = 32) {

    using member_t = typename wrapper_type_::value_type;
    using key_t = typename member_t::key_type;
    using mapped_t = typename member_t::mapped_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier, identifier)));

    std::size_t const window_begin = count / 4;
    std::size_t const window_end = count / 2;
    st_verify_(container.update_range(
        trivial_id_to_key<member_t>(window_begin), trivial_id_to_key<member_t>(window_end),
        [](key_t const &, mapped_t &mapped) noexcept { mapped = static_cast<mapped_t>(mapped + 100); }));

    for (std::size_t identifier = 0; identifier != count; ++identifier) {
        std::size_t const bonus = identifier >= window_begin && identifier < window_end ? 100 : 0;
        std::size_t observed = count + 1000;
        st_verify_(container.find(
            trivial_id_to_key<member_t>(identifier),
            [&](member_t const &member) noexcept { observed = static_cast<std::size_t>(member.mapped); },
            []() noexcept {}));
        st_verify_eq_(observed, identifier + bonus);
    }
}

/** @brief A wrapper names the least member of the whole store, not of one partition. */
template <typename wrapper_type_>
static void test_forwarded_smallest(std::size_t count = 64) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;

    bool empty_answered = false;
    st_verify_(container.smallest(no_op_t {}, [&]() noexcept { empty_answered = true; }));
    st_verify_((empty_answered) && "an empty store reports its emptiness rather than a member");

    // Inserted back to front, so a wrapper answering from insertion order rather than key order fails.
    for (std::size_t identifier = count; identifier != 0; --identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    trivial_id_t least = 0;
    st_verify_(container.smallest([&](member_t const &member) noexcept { least = member.unique_id; }));
    st_verify_eq_(least, trivial_id_t {1});
}

/** @brief Popping the least member removes exactly it, and drains the store in ascending order. */
template <typename wrapper_type_>
static void test_forwarded_pop_smallest(std::size_t count = 64) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;

    bool empty_answered = false;
    st_verify_eq_(container.pop_smallest(no_op_t {}, [&]() noexcept { empty_answered = true; }),
                  status_t::key_not_found_k);
    st_verify_((empty_answered) && "an empty store refuses rather than handing over a member");

    for (std::size_t identifier = count; identifier != 0; --identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    for (std::size_t expected_id = 1; expected_id <= count; ++expected_id) {
        trivial_id_t popped = 0;
        st_verify_(container.pop_smallest([&](member_t const &member) noexcept { popped = member.unique_id; }));
        st_verify_eq_(popped, static_cast<trivial_id_t>(expected_id));
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(expected_id)), false);
        st_verify_eq_(container.size(), count - expected_id);
    }
    st_verify_eq_(container.pop_smallest(), status_t::key_not_found_k);
}

/** @brief The open-ended erasures must each take their own half of the keyspace and no more. */
template <typename wrapper_type_>
static void test_forwarded_open_ended_erase(std::size_t count = 32) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    std::size_t erased = 0;
    st_verify_(
        container.erase_from(trivial_id_to_key<member_t>(count / 2), [&](member_t const &) noexcept { ++erased; }));
    st_verify_eq_(erased, count - count / 2);
    st_verify_eq_(container.size(), count / 2);
    st_verify_eq_(container.contains(trivial_id_to_key<member_t>(count / 2 - 1)), true);
    st_verify_eq_(container.contains(trivial_id_to_key<member_t>(count / 2)), false);

    erased = 0;
    st_verify_(
        container.erase_up_to(trivial_id_to_key<member_t>(count / 4), [&](member_t const &) noexcept { ++erased; }));
    st_verify_eq_(erased, count / 4);
    st_verify_eq_(container.size(), count / 2 - count / 4);
    st_verify_eq_(container.contains(trivial_id_to_key<member_t>(0)), false);
    st_verify_eq_(container.contains(trivial_id_to_key<member_t>(count / 4)), true);
}

/** @brief An enumeration through a wrapper must visit every member exactly once. */
template <typename wrapper_type_>
static void test_forwarded_for_each(std::size_t count = 200) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    std::vector<std::size_t> visits(count, 0);
    std::size_t total = 0;
    st_verify_(container.for_each([&](member_t const &member) noexcept {
        ++total;
        std::size_t const identifier = static_cast<std::size_t>(member.unique_id);
        if (identifier < visits.size()) ++visits[identifier];
    }));

    st_verify_eq_(total, count);
    for (std::size_t identifier = 0; identifier != count; ++identifier) st_verify_eq_(visits[identifier], 1u);

    st_verify_(container.erase(trivial_id_to_key<member_t>(count / 2)));
    total = 0;
    st_verify_(container.for_each([&](member_t const &) noexcept { ++total; }));
    st_verify_eq_(total, count - 1);
}

/**
 *  @brief A transaction has to survive being stored, moved and rearranged inside a container.
 *
 *  A defaulted move assignment over a reference member is silently deleted, which no declaration
 *  reports and no bare store ever hit, so the wrappers were the only ones that failed - and only at
 *  the call site that tried to rearrange one.
 */
template <typename wrapper_type_>
static void test_forwarded_transaction_moves_as_a_value() {

    using transaction_t = typename wrapper_type_::transaction_t;
    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &store = *made;

    basic_vector<transaction_t> writers;
    st_verify_(writers.reserve(2));
    for (std::size_t opened = 0; opened != 2; ++opened) {
        expected<transaction_t> writer = store.transaction();
        st_verify_((writer) && "the wrapped store must open a transaction");
        st_verify_(writers.push_back(std::move(*writer)));
    }
    st_verify_eq_(writers.size(), 2u);

    // The swap is the whole point: it is three move assignments, and a deleted one refuses here.
    auto const first_generation = writers[0].generation();
    auto const second_generation = writers[1].generation();
    transaction_t held = std::move(writers[0]);
    writers[0] = std::move(writers[1]);
    writers[1] = std::move(held);
    st_verify_eq_(writers[0].generation(), second_generation);
    st_verify_eq_(writers[1].generation(), first_generation);

    // And a transaction that travelled still reaches the store it was opened on.
    st_verify_(writers[0].upsert(trivial_id_to_member<member_t>(11)));
    st_verify_(writers[0].stage());
    st_verify_(writers[0].commit());
    st_verify_eq_(store.size(), 1u);
    st_verify_(writers[1].upsert(trivial_id_to_member<member_t>(12)));
    st_verify_(writers[1].reset());
}

/** @brief Every member equal to a key, which one partition owns outright and one lock covers. */
template <typename wrapper_type_>
static void test_forwarded_equal_range() {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &store = *made;
    for (std::size_t identifier = 0; identifier != 8; ++identifier)
        st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier)));

    std::size_t matched = 0;
    st_verify_(store.equal_range(trivial_id_to_key<member_t>(3), [&](member_t const &) noexcept { ++matched; }));
    st_verify_eq_(matched, 1u);

    matched = 0;
    st_verify_(store.equal_range(trivial_id_to_key<member_t>(99), [&](member_t const &) noexcept { ++matched; }));
    st_verify_eq_(matched, 0u);

    expected<typename wrapper_type_::transaction_t> writer = store.transaction();
    st_verify_((writer) && "the wrapped store must open a transaction");
    matched = 0;
    st_verify_(writer->equal_range(trivial_id_to_key<member_t>(3), [&](member_t const &) noexcept { ++matched; }));
    st_verify_eq_(matched, 1u);
}

/**
 *  @brief An erase through a wrapper must take whatever the inner store compares against.
 *    A composite key is looked up by its identifier alone, which a wrapper narrowed to
 *    @c identifier_t refuses before the store is ever asked.
 */
template <typename wrapper_type_>
static void test_forwarded_heterogeneous_erase() {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &store = *made;
    for (std::size_t identifier = 0; identifier != 6; ++identifier)
        st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier)));

    std::size_t removed = 0;
    st_verify_(store.erase(static_cast<trivial_id_t>(2), [&](member_t const &) noexcept { ++removed; }));
    st_verify_eq_(removed, 1u);
    st_verify_eq_(store.size(), 5u);
    st_verify_eq_(store.contains(static_cast<trivial_id_t>(2)), false);

    std::size_t missed = 0;
    st_verify_eq_(store.erase(
                      static_cast<trivial_id_t>(2), [](member_t const &) noexcept {}, [&]() noexcept { ++missed; }),
                  status_t::key_not_found_k);
    st_verify_eq_(missed, 1u);
}

/** @brief What a transaction has staged, and the walks that show it, must survive both wrappers. */
template <typename wrapper_type_>
static void test_forwarded_transaction_staged_surface() {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &store = *made;
    st_verify_(store.upsert(trivial_id_to_member<member_t>(1)));

    expected<typename wrapper_type_::transaction_t> writer = store.transaction();
    st_verify_((writer) && "the wrapped store must open a transaction");
    st_verify_(!writer->has_changes());
    st_verify_eq_(writer->changes_count(), 0u);

    st_verify_(writer->reserve(4));
    st_verify_(writer->insert_if_missing(trivial_id_to_member<member_t>(2)));
    st_verify_(writer->insert_if_missing(trivial_id_to_member<member_t>(1)));
    st_verify_(writer->has_changes());
    st_verify_eq_(writer->changes_count(), 1u);

    std::size_t walked = 0;
    st_verify_(writer->for_each([&](member_t const &) noexcept { ++walked; }));
    st_verify_eq_(walked, 2u);

    std::size_t within = 0;
    st_verify_(writer->range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(2),
                             [&](member_t const &) noexcept { ++within; }));
    st_verify_eq_(within, 1u);

    st_verify_(writer->stage());
    st_verify_(writer->commit());
    st_verify_eq_(store.size(), 2u);
}

/** @brief The version bookkeeping a snapshot store keeps must be readable through both wrappers. */
template <typename wrapper_type_>
static void test_forwarded_version_bookkeeping() {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &store = *made;
    for (std::size_t identifier = 0; identifier != 4; ++identifier)
        st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier)));
    st_verify_(store.upsert(trivial_id_to_member<member_t>(0)));

    st_verify_ge_(store.versions_count(), 4u);
    st_verify_ge_(store.versions_count(trivial_id_to_key<member_t>(0)), 1u);
    [[maybe_unused]] auto const mark = store.low_water_mark();

    [[maybe_unused]] expected<std::size_t> const reclaimed = store.vacuum();
    st_verify_eq_(store.size(), 4u);
}

#pragma endregion Forwarded Surface

static void forwarded_surface_strict_insert() {
    test_forwarded_strict_insert<locked_store<tree_trivial_set_t>, tree_trivial_set_t>();
    test_forwarded_strict_insert<partitioned_store<tree_trivial_set_t>, tree_trivial_set_t>();
    test_forwarded_strict_insert<locked_store<tree_trivial_map_t>, tree_trivial_map_t>();
    test_forwarded_strict_insert<partitioned_store<tree_trivial_map_t>, tree_trivial_map_t>();
    test_forwarded_strict_insert<locked_store<hash_store_t>, hash_store_t>();
    test_forwarded_strict_insert<partitioned_store<hash_store_t>, hash_store_t>();
}

static void forwarded_surface_strict_update() {
    test_forwarded_strict_update<locked_store<tree_trivial_set_t>, tree_trivial_set_t>();
    test_forwarded_strict_update<partitioned_store<tree_trivial_set_t>, tree_trivial_set_t>();
    test_forwarded_strict_update<locked_store<hash_store_t>, hash_store_t>();
    test_forwarded_strict_update<partitioned_store<hash_store_t>, hash_store_t>();
}

static void forwarded_surface_order_statistics() {
    test_forwarded_order_statistics<locked_store<ranked_set_t>>();
    test_forwarded_order_statistics<partitioned_store<ranked_set_t>>();
}

static void forwarded_surface_vacuum() {
    test_forwarded_vacuum<locked_store<tree_trivial_set_t>>();
    test_forwarded_vacuum<partitioned_store<tree_trivial_set_t>>();
    test_forwarded_vacuum<locked_store<hash_store_t>>();
    test_forwarded_vacuum<partitioned_store<hash_store_t>>();
    test_forwarded_windowed_vacuum<locked_store<tree_trivial_set_t>>();
    test_forwarded_windowed_vacuum<partitioned_store<tree_trivial_set_t>>();
}

static void forwarded_surface_update_range() {
    test_forwarded_update_range<locked_store<tree_trivial_map_t>>();
    test_forwarded_update_range<partitioned_store<tree_trivial_map_t>>();
}

static void forwarded_surface_open_ended_erase() {
    test_forwarded_open_ended_erase<locked_store<tree_trivial_set_t>>();
    test_forwarded_open_ended_erase<partitioned_store<tree_trivial_set_t>>();
}

static void forwarded_surface_smallest() {
    test_forwarded_smallest<locked_store<tree_trivial_set_t>>();
    test_forwarded_smallest<partitioned_store<tree_trivial_set_t>>();
    test_forwarded_smallest<partitioned_store<locked_store<tree_trivial_set_t>>>();
}

static void forwarded_surface_pop_smallest() {
    test_forwarded_pop_smallest<locked_store<tree_trivial_set_t>>();
    test_forwarded_pop_smallest<partitioned_store<tree_trivial_set_t>>();
    test_forwarded_pop_smallest<partitioned_store<locked_store<tree_trivial_set_t>>>();
}

static void forwarded_surface_for_each() {
    test_forwarded_for_each<locked_enumerable_set_t>();
    test_forwarded_for_each<partitioned_enumerable_set_t>();
}

/** @brief A transaction must survive being stored in a container and rearranged inside it. */
static void forwarded_surface_transaction_moves_as_a_value() {
    test_forwarded_transaction_moves_as_a_value<tree_trivial_set_t>();
    test_forwarded_transaction_moves_as_a_value<locked_store<tree_trivial_set_t>>();
    test_forwarded_transaction_moves_as_a_value<partitioned_store<tree_trivial_set_t>>();
    test_forwarded_transaction_moves_as_a_value<partitioned_store<locked_store<tree_trivial_set_t>>>();
    test_forwarded_transaction_moves_as_a_value<partitioned_store<locked_store<snapshot_trivial_map_t>>>();
}

static void forwarded_surface_equal_range() {
    test_forwarded_equal_range<locked_store<tree_trivial_set_t>>();
    test_forwarded_equal_range<partitioned_store<tree_trivial_set_t>>();
    test_forwarded_equal_range<locked_store<tree_trivial_map_t>>();
    test_forwarded_equal_range<partitioned_store<tree_trivial_map_t>>();
    test_forwarded_equal_range<partitioned_store<locked_store<snapshot_trivial_map_t>>>();
}

static void forwarded_surface_heterogeneous_erase() {
    test_forwarded_heterogeneous_erase<locked_store<tree_composite_set_t>>();
    test_forwarded_heterogeneous_erase<partitioned_store<tree_composite_set_t>>();
}

static void forwarded_surface_transaction_staged_surface() {
    test_forwarded_transaction_staged_surface<locked_store<tree_trivial_set_t>>();
    test_forwarded_transaction_staged_surface<partitioned_store<tree_trivial_set_t>>();
    test_forwarded_transaction_staged_surface<partitioned_store<locked_store<tree_trivial_set_t>>>();
}

static void forwarded_surface_version_bookkeeping() {
    test_forwarded_version_bookkeeping<locked_store<snapshot_trivial_set_t>>();
    test_forwarded_version_bookkeeping<partitioned_store<snapshot_trivial_set_t>>();
    test_forwarded_version_bookkeeping<partitioned_store<locked_store<snapshot_trivial_set_t>>>();
}

/** @brief The nesting a transparent wrapper creates has to behave like the shard set without it. */
static void forwarded_surface_nested_wrapper_behaves() {
    test_empty_container_operations<partitioned_store<locked_store<snapshot_trivial_set_t>>>();
    test_single_element_operations<partitioned_store<locked_store<snapshot_trivial_set_t>>>();
}

/** @brief An enumeration crossing a writer must still see everything that stayed put, exactly once. */
static void forwarded_surface_for_each_sees_every_stable_element() {
    test_sharded_enumeration_sees_every_stable_element<partitioned_enumerable_set_t>();
    test_sharded_enumeration_sees_every_stable_element<locked_enumerable_set_t>();
}

static void failure_policy_distinct_causes() {
    test_store_maps_causes_to_distinct_statuses<transactional_trivial_map_t>();
}
static void failure_policy_relayed_causes() { test_wrapper_relays_every_cause<sharded_refusing_map_t>(); }
static void failure_policy_bulk_methods() { test_bulk_methods_match_declared_policy<sharded_refusing_map_t>(); }

#pragma region Merged Order

/** @brief A store holding @p size members, keyed by identifier, spread across every partition. */
static transactional_trivial_set_t seeded_sharded_set(std::size_t size) {
    auto built = transactional_trivial_set_t::make();
    st_verify_((built) && "the sharded set must build");
    transactional_trivial_set_t store = std::move(*built);
    for (std::size_t identifier = 0; identifier != size; ++identifier)
        st_verify_(store.upsert(trivial_id_to_member<trivial_key_t>(identifier)));
    return store;
}

/**
 *  @brief A range over an ordered container answers in one ascending order, not sixteen sorted runs.
 *
 *  Concatenating each partition's run passes every membership check ever written for @c range, which
 *  is how it survived: only asking whether the sequence rises catches it.
 */
static void merged_order_range_ascends() {
    transactional_trivial_set_t store = seeded_sharded_set(256);

    std::size_t seen = 0;
    trivial_id_t previous = 0;
    bool first = true;
    st_verify_(store.range(trivial_id_to_key<trivial_key_t>(0), trivial_id_to_key<trivial_key_t>(256),
                           [&](trivial_key_t const &member) noexcept {
                               trivial_id_t const current = member.unique_id;
                               st_verify_((first || previous < current) && "a merged range must rise at every step");
                               previous = current;
                               first = false;
                               ++seen;
                           }));
    st_verify_eq_(seen, 256u);
}

/** @brief The merged range and repeated exclusive bounds answer with the same sequence. */
static void merged_order_range_matches_stepping() {
    transactional_trivial_set_t store = seeded_sharded_set(128);

    std::vector<trivial_id_t> walked;
    st_verify_(store.range(trivial_id_to_key<trivial_key_t>(0), trivial_id_to_key<trivial_key_t>(128),
                           [&](trivial_key_t const &member) noexcept { walked.push_back(member.unique_id); }));

    std::vector<trivial_id_t> stepped;
    trivial_key_t at = trivial_id_to_key<trivial_key_t>(0);
    st_verify_(store.lower_bound(at, [&](trivial_key_t const &member) noexcept {
        stepped.push_back(member.unique_id);
        at = member;
    }));
    while (true) {
        bool advanced = false;
        st_verify_(store.upper_bound(at, [&](trivial_key_t const &member) noexcept {
            stepped.push_back(member.unique_id);
            at = member;
            advanced = true;
        }));
        if (!advanced) break;
    }
    st_verify_eq_(walked.size(), stepped.size());
    for (std::size_t position = 0; position != walked.size(); ++position)
        st_verify_eq_(walked[position], stepped[position]);
}

/** @brief An inclusive bound answers with the key itself when it is there, and its successor when it is not. */
static void merged_order_inclusive_bound_is_one_probe() {
    transactional_trivial_set_t store = seeded_sharded_set(64);

    bool answered = false;
    st_verify_(store.lower_bound(trivial_id_to_key<trivial_key_t>(17), [&](trivial_key_t const &member) noexcept {
        st_verify_eq_(member.unique_id, 17u);
        answered = true;
    }));
    st_verify_((answered) && "an inclusive bound must answer with the key itself");

    st_verify_(store.erase(trivial_id_to_key<trivial_key_t>(17)));
    answered = false;
    st_verify_(store.lower_bound(trivial_id_to_key<trivial_key_t>(17), [&](trivial_key_t const &member) noexcept {
        st_verify_eq_(member.unique_id, 18u);
        answered = true;
    }));
    st_verify_((answered) && "an erased key must be answered by its successor");
}

#pragma endregion Merged Order

#pragma region Ordered Cursor

/** @brief Every key the cursor hands over from @p from onward, in the order it handed them over. */
static std::vector<trivial_id_t> drain_cursor(transactional_trivial_set_t const &store, trivial_id_t from,
                                              std::size_t limit) {
    std::vector<trivial_id_t> walked;
    auto walking = store.cursor_from(trivial_id_to_key<trivial_key_t>(from));
    for (std::size_t step = 0; step != limit; ++step) {
        bool handed = false;
        walking.next([&](trivial_key_t const &member) noexcept {
            walked.push_back(member.unique_id);
            handed = true;
        });
        if (!handed) break;
    }
    return walked;
}

/** @brief A cursor walks the same ascending sequence a merged range does, and stops when it runs out. */
static void ordered_cursor_matches_the_range() {
    transactional_trivial_set_t store = seeded_sharded_set(200);

    std::vector<trivial_id_t> ranged;
    st_verify_(store.range(trivial_id_to_key<trivial_key_t>(0), trivial_id_to_key<trivial_key_t>(200),
                           [&](trivial_key_t const &member) noexcept { ranged.push_back(member.unique_id); }));

    std::vector<trivial_id_t> const walked = drain_cursor(store, 0, 400);
    st_verify_eq_(walked.size(), ranged.size());
    for (std::size_t position = 0; position != walked.size(); ++position)
        st_verify_eq_(walked[position], ranged[position]);
}

/**
 *  @brief A cursor begun at a bound starts there, rather than at the smallest key of each partition.
 *
 *  Every front is read against the bound when the cursor settles, so the bound has to be in place
 *  before that happens - a walk seeded from a default key hands over members ordered below it.
 */
static void ordered_cursor_begins_at_the_bound_it_was_given() {
    transactional_trivial_set_t store = seeded_sharded_set(200);

    trivial_id_t const from = 137;
    std::vector<trivial_id_t> const walked = drain_cursor(store, from, 400);

    st_verify_eq_(walked.size(), std::size_t {200} - from);
    for (trivial_id_t identifier : walked) st_verify_ge_(identifier, from, "no key below the bound is handed over");
    st_verify_eq_(walked.front(), from);
}

/** @brief A bounded cursor stops before the key it was given, on the same half-open terms as a range. */
template <typename wrapper_type_>
static void test_cursor_stops_at_its_bound(std::size_t count = 64) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    std::vector<trivial_id_t> walked;
    for (auto walking = container.cursor_range(trivial_id_to_key<member_t>(8), trivial_id_to_key<member_t>(20));
         !walking.exhausted();)
        walking.next([&](member_t const &member) noexcept { walked.push_back(member.unique_id); });
    st_verify_eq_(walked.size(), std::size_t {12});
    st_verify_eq_(walked.front(), trivial_id_t {8});
    st_verify_eq_(walked.back(), trivial_id_t {19});

    walked.clear();
    for (auto walking = container.cursor_up_to(trivial_id_to_key<member_t>(5)); !walking.exhausted();)
        walking.next([&](member_t const &member) noexcept { walked.push_back(member.unique_id); });
    st_verify_eq_(walked.size(), std::size_t {5});
    st_verify_eq_(walked.back(), trivial_id_t {4});
}

/** @brief The bounded and unbounded walks read the same on both wrappers, which is what one shape means. */
static void ordered_cursor_stops_at_its_bound() {
    test_cursor_stops_at_its_bound<locked_store<tree_trivial_set_t>>();
    test_cursor_stops_at_its_bound<partitioned_store<tree_trivial_set_t>>();
    test_cursor_stops_at_its_bound<partitioned_store<locked_store<tree_trivial_set_t>>>();
}

/** @brief Erasing the key a cursor stands on hands over the successor rather than losing the walk. */
static void ordered_cursor_survives_its_own_key_erased() {
    transactional_trivial_set_t store = seeded_sharded_set(64);

    auto walking = store.cursor_from(trivial_id_to_key<trivial_key_t>(0));
    trivial_id_t standing = 0;
    walking.next([&](trivial_key_t const &member) noexcept { standing = member.unique_id; });

    st_verify_(store.erase(trivial_id_to_key<trivial_key_t>(standing)));

    bool handed = false;
    walking.next([&](trivial_key_t const &member) noexcept {
        st_verify_gt_(member.unique_id, standing, "a cursor must never hand back a key it already gave");
        handed = true;
    });
    st_verify_((handed) && "erasing the key underneath a cursor must not end its walk");
}

/**
 *  @brief A key inserted ahead of a live cursor is still handed over.
 *
 *  This is the property a cached front can quietly lose: the front was read before the insert, so
 *  nothing but the partition's write count tells the cursor to look again. Reverting the count leaves
 *  every other test here passing and this one failing.
 */
static void ordered_cursor_sees_a_key_inserted_ahead() {
    auto built = transactional_trivial_set_t::make();
    st_verify_((built) && "the sharded set must build");
    transactional_trivial_set_t store = std::move(*built);
    for (std::size_t identifier = 0; identifier != 64; identifier += 2)
        st_verify_(store.upsert(trivial_id_to_member<trivial_key_t>(identifier)));

    auto walking = store.cursor_from(trivial_id_to_key<trivial_key_t>(0));
    bool handed = false;
    walking.next([&](trivial_key_t const &member) noexcept {
        st_verify_eq_(member.unique_id, 0u);
        handed = true;
    });
    st_verify_(handed);

    // Slotted between the key just handed over and the front every partition is holding.
    st_verify_(store.upsert(trivial_id_to_member<trivial_key_t>(1)));

    handed = false;
    walking.next([&](trivial_key_t const &member) noexcept {
        st_verify_eq_(member.unique_id, 1u);
        handed = true;
    });
    st_verify_((handed) && "a key inserted ahead of a live cursor must still be handed over");
}

/** @brief A cursor hands over every key exactly once, whatever partition each of them hashed into. */
static void ordered_cursor_hands_every_key_once() {
    transactional_trivial_set_t store = seeded_sharded_set(300);

    std::vector<std::size_t> handed(300, 0);
    std::vector<trivial_id_t> const walked = drain_cursor(store, 0, 600);
    for (trivial_id_t identifier : walked) {
        st_verify_lt_(identifier, 300, "a cursor must hand over only keys the store holds");
        ++handed[identifier];
    }
    for (std::size_t identifier = 0; identifier != 300; ++identifier) st_verify_eq_(handed[identifier], 1u);
}

#pragma endregion Ordered Cursor

#pragma region Lock Cost

/**
 *  @brief A shared mutex that counts how many times it was taken, so a walk's cost can be asserted.
 *
 *  The count lives here rather than in the store because the store is already parameterized on its
 *  mutex: what a walk costs is measurable from outside without the library carrying a tally it would
 *  only ever use in a test.
 */
class counting_mutex_t {
    spin_shared_mutex_t held_;

  public:
    static inline std::atomic<std::size_t> acquisitions {0};

    static void reset() noexcept { acquisitions.store(0, std::memory_order_relaxed); }
    static std::size_t taken() noexcept { return acquisitions.load(std::memory_order_relaxed); }

    void lock() noexcept {
        acquisitions.fetch_add(1, std::memory_order_relaxed);
        held_.lock();
    }
    void unlock() noexcept { held_.unlock(); }
    void lock_shared() noexcept {
        acquisitions.fetch_add(1, std::memory_order_relaxed);
        held_.lock_shared();
    }
    void unlock_shared() noexcept { held_.unlock_shared(); }
};

using counted_sharded_set_t = partitioned_store<tree_trivial_set_t, hash<trivial_key_t>, counting_mutex_t, 16>;

/** @brief How many partitions a sharded store was built with, for a cost a test states in those terms. */
template <typename store_type_>
constexpr std::size_t partitions_of_v = store_type_::partitions_k;

/**
 *  @brief A cursor costs one partition acquisition per element, where stepping a bound costs all sixteen.
 *
 *  Asserted rather than measured: the whole reason the cursor caches a front per partition is that
 *  re-probing every partition per element is what an ordered walk used to cost, and a cache that
 *  quietly stopped working would still pass every correctness test above it.
 */
static void lock_cost_cursor_beats_stepping() {
    auto built = counted_sharded_set_t::make();
    st_verify_((built) && "the counted set must build");
    counted_sharded_set_t store = std::move(*built);

    constexpr std::size_t size_k = 256;
    for (std::size_t identifier = 0; identifier != size_k; ++identifier)
        st_verify_(store.upsert(trivial_id_to_member<trivial_key_t>(identifier)));

    // Stepping by exclusive bound: every element asks every partition.
    counting_mutex_t::reset();
    std::size_t stepped = 0;
    trivial_key_t at = trivial_id_to_key<trivial_key_t>(0);
    while (true) {
        bool advanced = false;
        st_verify_(store.upper_bound(at, [&](trivial_key_t const &member) noexcept {
            at = member;
            advanced = true;
        }));
        if (!advanced) break;
        ++stepped;
    }
    std::size_t const stepping_locks = counting_mutex_t::taken();

    // The cursor: one partition per element, plus the sixteen it seeded from.
    counting_mutex_t::reset();
    std::size_t walked = 0;
    auto walking = store.cursor_from(trivial_id_to_key<trivial_key_t>(0));
    while (true) {
        bool handed = false;
        walking.next([&](trivial_key_t const &) noexcept { handed = true; });
        if (!handed) break;
        ++walked;
    }
    std::size_t const cursor_locks = counting_mutex_t::taken();

    st_verify_eq_(walked, size_k);
    st_verify_ge_(stepping_locks, stepped * 16, "stepping a bound must ask every partition per element");
    // One acquisition per element handed over, plus the sixteen the first step seeds from.
    st_verify_le_(cursor_locks, walked + partitions_of_v<counted_sharded_set_t>,
                  "a cursor over a quiet store costs one partition acquisition per element");
    st_verify_lt_(cursor_locks * 8, stepping_locks, "the cursor must cost a fraction of stepping a bound");
}

#pragma endregion Lock Cost

/**
 *  @brief A key that refuses to be copied still reaches a partition.
 *
 *  Choosing a partition is a hash, and a hash needs to read a key rather than own one. Materializing an
 *  identifier to feed the hasher made every point operation demand a copy, so a move-only key - which
 *  every unsharded store accepts - could not be sharded at all. The merged walks still need a copy,
 *  because they remember one key per partition between steps, and they say so in their own assertion.
 */
static void sharded_ops_move_only_key_reaches_a_partition() {
    using heavy_set_t = monotonic_avl_set<heavy_key_t, std::less<void>, std::allocator<heavy_key_t>>;
    using sharded_t = partitioned_store<heavy_set_t, hash<heavy_key_t>>;

    auto made = sharded_t::make(std::less<void> {}, hash<heavy_key_t> {});
    st_verify_(made);
    sharded_t &store = *made;

    for (trivial_id_t identifier = 0; identifier != 64; ++identifier) {
        auto key = heavy_key_t::make(identifier);
        st_verify_(key);
        st_verify_(store.upsert(std::move(*key)));
    }
    st_verify_eq_(store.size(), 64u);

    for (trivial_id_t identifier = 0; identifier != 64; ++identifier) {
        auto probe = heavy_key_t::make(identifier);
        st_verify_(probe);
        bool seen = false;
        st_verify_(store.find(*probe, [&](heavy_key_t const &) noexcept { seen = true; }, no_op_t {}));
        st_verify_((seen) && "a key that reached a partition must be findable in it");
    }

    auto doomed = heavy_key_t::make(7);
    st_verify_(doomed);
    st_verify_(store.erase(*doomed));
    st_verify_eq_(store.size(), 63u);
}

/**
 *  @brief A commit spanning partitions publishes all of them or none, whatever a watch answers.
 *
 *  Each engine behind the unstamped path re-checks its watches when it commits, because another
 *  transaction may have published over a watched key while this one sat staged. Asking one partition
 *  at a time meant a later refusal arrived over writes an earlier partition had already made visible -
 *  a reader could name values from a transaction that told its caller it had not committed, which is
 *  weaker than the level this configuration reports.
 */
static void sharded_ops_commit_publishes_all_or_nothing() {
    using store_t = transactional_trivial_map_t;
    using member_t = typename store_t::value_type;

    store_t store;

    // Two keys the hash sends to different partitions, so one can refuse after the other would publish.
    trivial_id_t written = 0, watched = 0;
    hash<trivial_key_t> const hasher;
    std::size_t const parts = partitions_of_v<store_t>;
    for (trivial_id_t candidate = 1; candidate != 512 && !watched; ++candidate) {
        std::size_t const part = hasher(trivial_id_to_key<member_t>(candidate)) % parts;
        if (!written) written = candidate;
        else if (part != hasher(trivial_id_to_key<member_t>(written)) % parts) watched = candidate;
    }
    st_verify_((written && watched) && "the fixture needs two keys in different partitions");

    st_verify_(store.upsert(trivial_id_to_member<member_t>(watched, 1)));

    auto writer = store.transaction();
    st_verify_(writer);
    st_verify_(writer->watch(trivial_id_to_key<member_t>(watched)));
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(written, 42)));
    st_verify_(writer->stage());

    // The watch is invalidated after staging, which is the window the engines re-check for.
    st_verify_(store.upsert(trivial_id_to_member<member_t>(watched, 2)));

    st_verify_eq_(writer->commit(), status_t::read_conflict_k);

    // Nothing this transaction wrote may be readable, since its caller was told it did not commit.
    expected<bool> const landed = store.contains(trivial_id_to_key<member_t>(written));
    st_verify_(landed);
    st_verify_eq_(*landed, false, "a refused commit must leave none of its writes readable");
}

/**
 *  @brief A store that refuses to open a transaction, naming a reason a wrapper must carry outward.
 *
 *  Only @c reference_store can refuse this for real, and only when its own allocation throws, which is
 *  not schedulable from a test. Refusing on demand is what makes the wrapper's relay observable.
 */
struct transaction_refusing_set_t : tree_trivial_set_t {
    using base_t = tree_trivial_set_t;
    using transaction_t = typename base_t::transaction_t;

    transaction_refusing_set_t() noexcept = default;
    transaction_refusing_set_t(base_t &&other) noexcept : base_t(std::move(other)) {}
    transaction_refusing_set_t(transaction_refusing_set_t &&) noexcept = default;
    transaction_refusing_set_t &operator=(transaction_refusing_set_t &&) noexcept = default;

    [[nodiscard]] static expected<transaction_refusing_set_t> make() noexcept {
        expected<base_t> built = base_t::make();
        if (!built) return built.status();
        return transaction_refusing_set_t {std::move(*built)};
    }

    [[nodiscard]] expected<transaction_t> transaction() noexcept { return status_t::out_of_memory_heap_k; }
};

/** @brief A store that cannot be built at all, so a wrapper's factory has a reason to carry outward. */
struct construction_refusing_set_t : tree_trivial_set_t {
    using base_t = tree_trivial_set_t;
    using transaction_t = typename base_t::transaction_t;

    construction_refusing_set_t() noexcept = default;
    construction_refusing_set_t(construction_refusing_set_t &&) noexcept = default;
    construction_refusing_set_t &operator=(construction_refusing_set_t &&) noexcept = default;

    [[nodiscard]] static expected<construction_refusing_set_t> make() noexcept {
        return status_t::capacity_exhausted_k;
    }
};

/** @brief A wrapper that cannot build its store reports why, rather than a default-constructed reason. */
template <typename wrapper_type_>
static void test_make_reports_why_it_could_not_build() {
    expected<wrapper_type_> made = wrapper_type_::make();
    st_verify_((!made) && "the inner store refused to be built, so the wrapper must refuse too");
    st_verify_eq_(made.status(), status_t::capacity_exhausted_k,
                  "the wrapper must relay the reason rather than lose it to a default");
}

/** @brief A wrapper that cannot open a transaction reports why, rather than a default-constructed reason. */
template <typename wrapper_type_>
static void test_transaction_reports_why_it_could_not_open() {
    expected<wrapper_type_> made = wrapper_type_::make();
    st_verify_(made);
    expected<typename wrapper_type_::transaction_t> opened = made->transaction();
    st_verify_((!opened) && "the inner store refused, so the wrapper must refuse too");
    st_verify_eq_(opened.status(), status_t::out_of_memory_heap_k,
                  "the wrapper must relay the reason rather than lose it to a default");
}

static void sharded_ops_transaction_reports_its_reason() {
    test_transaction_reports_why_it_could_not_open<partitioned_store<transaction_refusing_set_t>>();
    test_transaction_reports_why_it_could_not_open<locked_store<transaction_refusing_set_t>>();
    test_make_reports_why_it_could_not_build<partitioned_store<construction_refusing_set_t>>();
    test_make_reports_why_it_could_not_build<locked_store<construction_refusing_set_t>>();
}

/**
 *  @brief A wrapper hands back the reason a read could not be recorded rather than answering success.
 *
 *  A validated read is written down before it can be validated, and writing it down allocates. The
 *  wrapper sits between the caller and the engine that lost the record, so a wrapper answering
 *  @c success_k over a refusal is the same defect one frame higher.
 */
static void sharded_ops_read_reports_what_it_could_not_record() {
    using inner_t = serializable_avl_map<trivial_key_t, int, std::less<trivial_key_t>,
                                         stateful_allocator<mapping<trivial_key_t, int>>>;
    using store_t = locked_store<inner_t>;
    using member_t = typename store_t::value_type;

    allocation_ledger_t ledger;
    {
        expected<store_t> made = store_t::make(typename inner_t::allocator_t(ledger));
        st_verify_(made);
        store_t &store = *made;
        st_verify_(store.upsert(trivial_id_to_member<member_t>(1, 1)));

        auto reader = store.transaction();
        st_verify_(reader);

        // Everything the transaction needed is already allocated, so the next request is the read set's.
        ledger.refuse_everything();
        status_t const answered = reader->find(trivial_key_t {1}, no_op_t {}, no_op_t {});
        st_verify_eq_(answered, status_t::out_of_memory_heap_k, "a wrapper must relay what the read could not record");
        ledger.allow(unlimited_budget_k);
        st_verify_(reader->reset());
    }
    ledger.verify_balanced();
}

static void sharded_concurrency_commit_is_visible_to_what_opens_after_it() {
    using gated_map_t = mapping<std::int64_t, std::int64_t>;
    // Pinned, because the test picks what it asserts from this level: a wrapper that dropped the rung
    // would leave the strict instantiation asserting nothing and still reporting a pass.
    static_assert(partitioned_store<strict_serializable_store<basic_avl_tree<gated_map_t, less_t>>,
                                    hash<std::int64_t>>::isolation_k == isolation_t::strict_serializable_k,
                  "a shard set over a strict store is strict");
    using serializable_gated_t =
        serializable_store<basic_avl_tree<gated_map_t, gated_less_t, std::allocator<gated_map_t>>>;
    using strict_gated_t =
        strict_serializable_store<basic_avl_tree<gated_map_t, gated_less_t, std::allocator<gated_map_t>>>;
    test_commit_is_visible_to_what_opens_after_it<partitioned_store<serializable_gated_t, hash<std::int64_t>>>();
    test_commit_is_visible_to_what_opens_after_it<partitioned_store<strict_gated_t, hash<std::int64_t>>>();
}

} // namespace

int main() {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "lock_cost.cursor_beats_stepping", lock_cost_cursor_beats_stepping);
    failures += run_test(filter, "merged_order.range_ascends", merged_order_range_ascends);
    failures += run_test(filter, "merged_order.range_matches_stepping", merged_order_range_matches_stepping);
    failures +=
        run_test(filter, "merged_order.inclusive_bound_is_one_probe", merged_order_inclusive_bound_is_one_probe);
    failures += run_test(filter, "ordered_cursor.matches_the_range", ordered_cursor_matches_the_range);
    failures += run_test(filter, "ordered_cursor.stops_at_its_bound", ordered_cursor_stops_at_its_bound);
    failures += run_test(filter, "ordered_cursor.begins_at_the_bound_it_was_given",
                         ordered_cursor_begins_at_the_bound_it_was_given);
    failures +=
        run_test(filter, "ordered_cursor.survives_its_own_key_erased", ordered_cursor_survives_its_own_key_erased);
    failures += run_test(filter, "ordered_cursor.sees_a_key_inserted_ahead", ordered_cursor_sees_a_key_inserted_ahead);
    failures += run_test(filter, "ordered_cursor.hands_every_key_once", ordered_cursor_hands_every_key_once);
    failures += run_test(filter, "failure_policy.distinct_causes", failure_policy_distinct_causes);
    failures += run_test(filter, "failure_policy.relayed_causes", failure_policy_relayed_causes);
    failures +=
        run_test(filter, "sharded_ops.transaction_reports_its_reason", sharded_ops_transaction_reports_its_reason);
    failures += run_test(filter, "sharded_ops.read_reports_what_it_could_not_record",
                         sharded_ops_read_reports_what_it_could_not_record);
    failures +=
        run_test(filter, "sharded_ops.commit_publishes_all_or_nothing", sharded_ops_commit_publishes_all_or_nothing);
    failures += run_test(filter, "failure_policy.bulk_methods", failure_policy_bulk_methods);
    failures += run_test(filter, "basic_ops.empty_container_operations", basic_ops_empty_container_operations);
    failures += run_test(filter, "sharded_concurrency.walks_never_race_erasures",
                         sharded_concurrency_walks_never_race_erasures);
    failures += run_test(filter, "sharded_ops.move_only_key_reaches_a_partition",
                         sharded_ops_move_only_key_reaches_a_partition);
    failures += run_test(filter, "sharded_concurrency.distinct_generations", sharded_concurrency_distinct_generations);
    failures += run_test(filter, "sharded_concurrency.commit_is_visible_to_what_opens_after_it",
                         sharded_concurrency_commit_is_visible_to_what_opens_after_it);
    failures += run_test(filter, "sharded_concurrency.standard_mutex_substitutes",
                         sharded_concurrency_standard_mutex_substitutes);
    failures += run_test(filter, "sharded_concurrency.stage_unwinds_on_partial_failure",
                         sharded_concurrency_stage_unwinds_on_partial_failure);
    failures +=
        run_test(filter, "sharded_concurrency.commit_spans_partitions", sharded_concurrency_commit_spans_partitions);
    failures += run_test(filter, "basic_ops.single_element_operations", basic_ops_single_element_operations);
    failures += run_test(filter, "basic_ops.insertion_patterns", basic_ops_insertion_patterns);
    failures += run_test(filter, "basic_ops.bulk_insertion_iterators", basic_ops_bulk_insertion_iterators);
    failures +=
        run_test(filter, "basic_ops.bulk_upsert_with_duplicate_pairs", basic_ops_bulk_upsert_with_duplicate_pairs);
    failures += run_test(filter, "basic_ops.range_query_head_state", basic_ops_range_query_head_state);
    failures += run_test(filter, "basic_ops.erase_range_head_state", basic_ops_erase_range_head_state);
    failures += run_test(filter, "basic_ops.heterogeneous_lookups", basic_ops_heterogeneous_lookups);
    failures += run_test(filter, "forwarded_surface.transaction_moves_as_a_value",
                         forwarded_surface_transaction_moves_as_a_value);
    failures += run_test(filter, "forwarded_surface.equal_range", forwarded_surface_equal_range);
    failures += run_test(filter, "forwarded_surface.heterogeneous_erase", forwarded_surface_heterogeneous_erase);
    failures +=
        run_test(filter, "forwarded_surface.transaction_staged_surface", forwarded_surface_transaction_staged_surface);
    failures += run_test(filter, "forwarded_surface.version_bookkeeping", forwarded_surface_version_bookkeeping);
    failures += run_test(filter, "forwarded_surface.nested_wrapper_behaves", forwarded_surface_nested_wrapper_behaves);

    failures += run_test(filter, "transactional_consistency.empty_transaction_commit",
                         transactional_consistency_empty_transaction_commit);
    failures += run_test(filter, "transactional_consistency.no_dirty_reads_multi_key",
                         transactional_consistency_no_dirty_reads_multi_key);
    failures += run_test(filter, "transactional_consistency.new_transaction_sees_nothing_staged",
                         transactional_consistency_new_transaction_sees_nothing_staged);
    failures += run_test(filter, "transactional_consistency.committed_immediately_visible",
                         transactional_consistency_committed_immediately_visible);
    failures += run_test(filter, "transactional_consistency.multi_key_atomicity_10_keys",
                         transactional_consistency_multi_key_atomicity_10_keys);
    failures += run_test(filter, "transactional_consistency.rollback_makes_all_invisible",
                         transactional_consistency_rollback_makes_all_invisible);
    failures += run_test(filter, "transactional_consistency.range_query_sees_atomic_boundaries",
                         transactional_consistency_range_query_sees_atomic_boundaries);
    failures += run_test(filter, "transactional_consistency.fractured_read_prevention",
                         transactional_consistency_fractured_read_prevention);
    failures += run_test(filter, "transactional_consistency.sequential_updates_never_regress",
                         transactional_consistency_sequential_updates_never_regress);
    failures += run_test(filter, "transactional_consistency.transaction_commits_maintain_order",
                         transactional_consistency_transaction_commits_maintain_order);
    failures += run_test(filter, "transactional_consistency.concurrent_transactions_on_same_key",
                         transactional_consistency_concurrent_transactions_on_same_key);
    failures += run_test(filter, "transactional_consistency.multi_key_conflict_any_key_fails",
                         transactional_consistency_multi_key_conflict_any_key_fails);
    failures += run_test(filter, "transactional_consistency.watch_detects_external_direct_modification",
                         transactional_consistency_watch_detects_external_direct_modification);
    failures += run_test(filter, "transactional_consistency.watch_detects_staged_invisible_writes",
                         transactional_consistency_watch_detects_staged_invisible_writes);
    failures += run_test(filter, "transactional_consistency.watch_detects_staged_writes_of_older_generation",
                         transactional_consistency_watch_detects_staged_writes_of_older_generation);
    failures += run_test(filter, "transactional_consistency.disjoint_keys_both_succeed",
                         transactional_consistency_disjoint_keys_both_succeed);
    failures += run_test(filter, "transactional_consistency.lost_update_matches_isolation",
                         transactional_consistency_lost_update_matches_isolation);
    failures += run_test(filter, "transactional_consistency.repeated_read_matches_isolation",
                         transactional_consistency_repeated_read_matches_isolation);
    failures += run_test(filter, "transactional_consistency.write_skew_matches_isolation",
                         transactional_consistency_write_skew_matches_isolation);
    failures += run_test(filter, "transactional_consistency.read_conflict_matches_isolation",
                         transactional_consistency_read_conflict_matches_isolation);
    failures += run_test(filter, "transactional_consistency.repeated_range_matches_isolation",
                         transactional_consistency_repeated_range_matches_isolation);
    failures +=
        run_test(filter, "transactional_consistency.delete_visibility", transactional_consistency_delete_visibility);
    failures += run_test(filter, "transactional_consistency.reset_clears_transaction_state",
                         transactional_consistency_reset_clears_transaction_state);

    failures += run_test(filter, "transactional_consistency.stateful_comparator_is_consulted",
                         transactional_consistency_stateful_comparator_is_consulted);

    failures += run_test(filter, "transactional_consistency.group_commits_participants_together",
                         transactional_consistency_group_commits_participants_together);
    failures += run_test(filter, "transactional_consistency.group_unwinds_every_participant_on_conflict",
                         transactional_consistency_group_unwinds_every_participant_on_conflict);

    failures += run_test(filter, "transactional_consistency.find_does_not_watch",
                         transactional_consistency_find_does_not_watch);

    failures += run_test(filter, "sharded_concurrency.locked_store_forwards_construction_and_writes",
                         test_locked_store_forwards_construction_and_writes);

    failures += run_test(filter, "forwarded_surface.strict_insert", forwarded_surface_strict_insert);
    failures += run_test(filter, "forwarded_surface.strict_update", forwarded_surface_strict_update);
    failures += run_test(filter, "forwarded_surface.order_statistics", forwarded_surface_order_statistics);
    failures += run_test(filter, "forwarded_surface.vacuum", forwarded_surface_vacuum);
    failures += run_test(filter, "forwarded_surface.update_range", forwarded_surface_update_range);
    failures += run_test(filter, "forwarded_surface.open_ended_erase", forwarded_surface_open_ended_erase);
    failures += run_test(filter, "forwarded_surface.smallest", forwarded_surface_smallest);
    failures += run_test(filter, "forwarded_surface.pop_smallest", forwarded_surface_pop_smallest);
    failures += run_test(filter, "forwarded_surface.for_each", forwarded_surface_for_each);
    failures += run_test(filter, "forwarded_surface.for_each_sees_every_stable_element",
                         forwarded_surface_for_each_sees_every_stable_element);

    return report_test_failures(failures);
}
