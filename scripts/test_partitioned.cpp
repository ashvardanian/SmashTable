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
#include "test_consistency.hpp"
#include "test_sharded_concurrency.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

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

template <typename store_type_>
constexpr bool offers_ordered_surface_k =
    requires(store_type_ &store, typename store_type_::identifier_t const &key, no_op_t callback) {
        store.lower_bound(key, callback, callback);
        store.upper_bound(key, callback, callback);
        store.range(key, key, callback);
        store.erase_range(key, key, callback);
    };

static_assert(offers_ordered_surface_k<tree_trivial_set_t>, "the tree-backed store is ordered");
static_assert(offers_ordered_surface_k<locked_store<tree_trivial_set_t>>,
              "wrapping an ordered store must keep the ordered surface");
static_assert(offers_ordered_surface_k<partitioned_store<tree_trivial_set_t>>,
              "sharding an ordered store must keep the ordered surface");

static_assert(!offers_ordered_surface_k<hash_store_t>, "a hash-backed store has no ordering to offer");
static_assert(!offers_ordered_surface_k<locked_store<hash_store_t>>,
              "the lock wrapper must not claim an ordering its inner store denies");
static_assert(!offers_ordered_surface_k<partitioned_store<hash_store_t>>,
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
    void for_each(callback_type_ &&callback) const noexcept {
        base_t::range(trivial_key_t {0}, trivial_key_t {keyspace_k}, std::forward<callback_type_>(callback));
    }
};

template <typename store_type_>
constexpr bool offers_order_statistics_k = requires(
    store_type_ const &store, typename store_type_::identifier_t const &key, std::size_t ordinal, no_op_t callback) {
    store.select(ordinal, callback, callback);
    store.rank(key, callback, callback);
};

static_assert(offers_order_statistics_k<ranked_set_t>, "the weight-balanced store answers by ordinal");
static_assert(offers_order_statistics_k<locked_store<monotonic_ranked_set_t>>,
              "the ordinal surface travels through the wrapper whichever store family carries it");
static_assert(offers_order_statistics_k<partitioned_store<monotonic_ranked_set_t>>,
              "the ordinal surface travels through the wrapper whichever store family carries it");
static_assert(offers_order_statistics_k<locked_store<ranked_set_t>>,
              "wrapping an order-statistics store must keep the ordinal surface");
static_assert(offers_order_statistics_k<partitioned_store<ranked_set_t>>,
              "sharding an order-statistics store must keep the ordinal surface");

static_assert(!offers_order_statistics_k<tree_trivial_set_t>, "the AVL core sums no subtree counts");
static_assert(!offers_order_statistics_k<locked_store<tree_trivial_set_t>>,
              "the lock wrapper must not claim ordinals its inner core cannot answer");
static_assert(!offers_order_statistics_k<partitioned_store<tree_trivial_set_t>>,
              "the partitioned wrapper must not claim ordinals its inner core cannot answer");
static_assert(!offers_order_statistics_k<locked_store<hash_store_t>>, "an unordered core has no ordinals at all");
static_assert(!offers_order_statistics_k<partitioned_store<hash_store_t>>, "an unordered core has no ordinals at all");

template <typename store_type_>
constexpr bool offers_enumeration_k =
    requires(store_type_ const &store, no_op_t callback) { store.for_each(callback); };

static_assert(offers_enumeration_k<enumerable_set_t>, "the fixture store enumerates");
static_assert(offers_enumeration_k<locked_store<enumerable_set_t>>,
              "wrapping an enumerable store must keep the enumeration");
static_assert(offers_enumeration_k<partitioned_store<enumerable_set_t>>,
              "sharding an enumerable store must keep the enumeration");

static_assert(offers_enumeration_k<tree_trivial_set_t>, "a transactional store walks its own members");
static_assert(offers_enumeration_k<locked_store<tree_trivial_set_t>>,
              "the lock wrapper forwards the enumeration its inner store offers");
static_assert(offers_enumeration_k<partitioned_store<tree_trivial_set_t>>,
              "the partitioned wrapper forwards the enumeration its inner store offers");

/** Both wrappers over the enumerable fixture, which is what the enumeration suites walk. */
using locked_enumerable_set_t = locked_store<enumerable_set_t>;
using partitioned_enumerable_set_t = partitioned_store<enumerable_set_t>;

/** One shared mutex over the whole collection, reached through the shape-naming aliases. */
using transactional_tracking_set_t = locked_set<tree_trivial_set_t>;
using transactional_tracking_map_t = locked_map<tree_trivial_map_t>;

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

#pragma endregion Type Aliases

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
    st_verify_(succeeded(bare.upsert(trivial_id_to_member<member_t>(7))));
    status_t const bare_refusal = bare.insert(trivial_id_to_member<member_t>(7));
    st_verify_(failed(bare_refusal) && "a bare store must refuse an occupied key");

    expected<wrapper_type_> wrapped_made = built_store<wrapper_type_>();
    wrapper_type_ &wrapped = *wrapped_made;
    st_verify_(succeeded(wrapped.upsert(trivial_id_to_member<member_t>(7))));
    st_verify_eq_(wrapped.insert(trivial_id_to_member<member_t>(7)), bare_refusal);
    st_verify_eq_(wrapped.size(), 1u);
    st_verify_(succeeded(wrapped.insert(trivial_id_to_member<member_t>(8))));
    st_verify_eq_(wrapped.size(), 2u);

    // The same refusal staged rather than published, which is the level a binding reaches for.
    expected<typename inner_type_::transaction_t> bare_writer = bare.transaction();
    st_verify_((bare_writer) && "the bare store must open a transaction");
    status_t const bare_staged_refusal = bare_writer->insert(trivial_id_to_member<member_t>(7));
    st_verify_(failed(bare_staged_refusal) && "a bare transaction must refuse an occupied key");

    expected<typename wrapper_type_::transaction_t> wrapped_writer = wrapped.transaction();
    st_verify_((wrapped_writer) && "the wrapped store must open a transaction");
    st_verify_eq_(wrapped_writer->insert(trivial_id_to_member<member_t>(7)), bare_staged_refusal);
    st_verify_(succeeded(wrapped_writer->insert(trivial_id_to_member<member_t>(9))));
    st_verify_(succeeded(wrapped_writer->stage()));
    st_verify_(succeeded(wrapped_writer->commit()));
    st_verify_eq_(wrapped.size(), 3u);
}

/** @brief A strict update through a wrapper must refuse an absent key with the bare store's status. */
template <typename wrapper_type_, typename inner_type_>
static void test_forwarded_strict_update() {

    using member_t = typename wrapper_type_::value_type;

    expected<inner_type_> bare_made = built_store<inner_type_>();
    inner_type_ &bare = *bare_made;
    status_t const bare_refusal = bare.update(trivial_id_to_member<member_t>(3));
    st_verify_(failed(bare_refusal) && "a bare store must refuse an absent key");

    expected<wrapper_type_> wrapped_made = built_store<wrapper_type_>();
    wrapper_type_ &wrapped = *wrapped_made;
    st_verify_eq_(wrapped.update(trivial_id_to_member<member_t>(3)), bare_refusal);
    st_verify_eq_(wrapped.size(), 0u);

    st_verify_(succeeded(wrapped.upsert(trivial_id_to_member<member_t>(3))));
    st_verify_(succeeded(wrapped.update(trivial_id_to_member<member_t>(3))));
    st_verify_eq_(wrapped.size(), 1u);

    expected<typename wrapper_type_::transaction_t> writer = wrapped.transaction();
    st_verify_((writer) && "the wrapped store must open a transaction");
    st_verify_(failed(writer->update(trivial_id_to_member<member_t>(4))));
    st_verify_(succeeded(writer->update(trivial_id_to_member<member_t>(3))));
}

/** @brief Ordinals through a wrapper must name the same elements the merged order does. */
template <typename wrapper_type_>
static void test_forwarded_order_statistics(std::size_t count = 64) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(identifier))));

    for (std::size_t identifier = 0; identifier != count; ++identifier) {
        member_t selected {};
        bool selected_one = false;
        container.select(
            identifier,
            [&](member_t const &member) noexcept {
                selected = member;
                selected_one = true;
            },
            []() noexcept {});
        st_verify_((selected_one) && "every ordinal below the size must name an element");
        st_verify_(selected == trivial_id_to_key<member_t>(identifier));

        std::size_t position = count;
        container.rank(
            trivial_id_to_key<member_t>(identifier), [&](std::size_t found) noexcept { position = found; },
            []() noexcept {});
        st_verify_eq_(position, identifier);
    }

    bool missed = false;
    container.select(count, [](member_t const &) noexcept {}, [&]() noexcept { missed = true; });
    st_verify_((missed) && "an ordinal at the size must find nothing");

    missed = false;
    container.rank(
        trivial_id_to_key<member_t>(count + 1), [](std::size_t) noexcept {}, [&]() noexcept { missed = true; });
    st_verify_((missed) && "a key that is not there has no rank");
}

/** @brief A sweep through a wrapper must answer with a count and leave every survivor readable. */
template <typename wrapper_type_>
static void test_forwarded_vacuum(std::size_t count = 32) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(identifier))));
    for (std::size_t identifier = 0; identifier < count; identifier += 2)
        st_verify_(succeeded(container.erase(trivial_id_to_key<member_t>(identifier))));

    expected<std::size_t> const reclaimed = container.vacuum();
    st_verify_((reclaimed) && "a sweep that cannot refuse must still answer with a count");
    st_verify_eq_(container.size(), count / 2);
    for (std::size_t identifier = 1; identifier < count; identifier += 2)
        st_verify_(container.contains(trivial_id_to_key<member_t>(identifier)));
    for (std::size_t identifier = 0; identifier < count; identifier += 2)
        st_verify_(!container.contains(trivial_id_to_key<member_t>(identifier)));
}

/** @brief A windowed sweep must reclaim its window and leave the rest of the keyspace readable. */
template <typename wrapper_type_>
static void test_forwarded_windowed_vacuum(std::size_t count = 32) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(identifier))));
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(succeeded(container.erase(trivial_id_to_key<member_t>(identifier))));

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
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(identifier, identifier))));

    std::size_t const window_begin = count / 4;
    std::size_t const window_end = count / 2;
    st_verify_(succeeded(container.update_range(
        trivial_id_to_key<member_t>(window_begin), trivial_id_to_key<member_t>(window_end),
        [](key_t const &, mapped_t &mapped) noexcept { mapped = static_cast<mapped_t>(mapped + 100); })));

    for (std::size_t identifier = 0; identifier != count; ++identifier) {
        std::size_t const bonus = identifier >= window_begin && identifier < window_end ? 100 : 0;
        std::size_t observed = count + 1000;
        container.find(
            trivial_id_to_key<member_t>(identifier),
            [&](member_t const &member) noexcept { observed = static_cast<std::size_t>(member.mapped); },
            []() noexcept {});
        st_verify_eq_(observed, identifier + bonus);
    }
}

/** @brief The open-ended erasures must each take their own half of the keyspace and no more. */
template <typename wrapper_type_>
static void test_forwarded_open_ended_erase(std::size_t count = 32) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(identifier))));

    std::size_t erased = 0;
    st_verify_(succeeded(
        container.erase_from(trivial_id_to_key<member_t>(count / 2), [&](member_t const &) noexcept { ++erased; })));
    st_verify_eq_(erased, count - count / 2);
    st_verify_eq_(container.size(), count / 2);
    st_verify_(container.contains(trivial_id_to_key<member_t>(count / 2 - 1)));
    st_verify_(!container.contains(trivial_id_to_key<member_t>(count / 2)));

    erased = 0;
    st_verify_(succeeded(
        container.erase_up_to(trivial_id_to_key<member_t>(count / 4), [&](member_t const &) noexcept { ++erased; })));
    st_verify_eq_(erased, count / 4);
    st_verify_eq_(container.size(), count / 2 - count / 4);
    st_verify_(!container.contains(trivial_id_to_key<member_t>(0)));
    st_verify_(container.contains(trivial_id_to_key<member_t>(count / 4)));
}

/** @brief An enumeration through a wrapper must visit every member exactly once. */
template <typename wrapper_type_>
static void test_forwarded_for_each(std::size_t count = 200) {

    using member_t = typename wrapper_type_::value_type;

    expected<wrapper_type_> made = built_store<wrapper_type_>();
    wrapper_type_ &container = *made;
    for (std::size_t identifier = 0; identifier != count; ++identifier)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(identifier))));

    std::vector<std::size_t> visits(count, 0);
    std::size_t total = 0;
    container.for_each([&](member_t const &member) noexcept {
        ++total;
        std::size_t const identifier = static_cast<std::size_t>(member.unique_id);
        if (identifier < visits.size()) ++visits[identifier];
    });

    st_verify_eq_(total, count);
    for (std::size_t identifier = 0; identifier != count; ++identifier) st_verify_eq_(visits[identifier], 1u);

    st_verify_(succeeded(container.erase(trivial_id_to_key<member_t>(count / 2))));
    total = 0;
    container.for_each([&](member_t const &) noexcept { ++total; });
    st_verify_eq_(total, count - 1);
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

static void forwarded_surface_for_each() {
    test_forwarded_for_each<locked_enumerable_set_t>();
    test_forwarded_for_each<partitioned_enumerable_set_t>();
}

/** @brief An enumeration crossing a writer must still see everything that stayed put, exactly once. */
static void forwarded_surface_for_each_sees_every_stable_element() {
    test_sharded_enumeration_sees_every_stable_element<partitioned_enumerable_set_t>();
    test_sharded_enumeration_sees_every_stable_element<locked_enumerable_set_t>();
}

int main() {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "basic_ops.empty_container_operations", basic_ops_empty_container_operations);
    failures += run_test(filter, "sharded_concurrency.walks_never_race_erasures",
                         sharded_concurrency_walks_never_race_erasures);
    failures += run_test(filter, "sharded_concurrency.distinct_generations", sharded_concurrency_distinct_generations);
    failures += run_test(filter, "sharded_concurrency.standard_mutex_substitutes",
                         sharded_concurrency_standard_mutex_substitutes);
    failures += run_test(filter, "sharded_concurrency.stage_unwinds_on_partial_failure",
                         sharded_concurrency_stage_unwinds_on_partial_failure);
    failures += run_test(filter, "basic_ops.single_element_operations", basic_ops_single_element_operations);
    failures += run_test(filter, "basic_ops.insertion_patterns", basic_ops_insertion_patterns);
    failures += run_test(filter, "basic_ops.bulk_insertion_iterators", basic_ops_bulk_insertion_iterators);
    failures +=
        run_test(filter, "basic_ops.bulk_upsert_with_duplicate_pairs", basic_ops_bulk_upsert_with_duplicate_pairs);
    failures += run_test(filter, "basic_ops.range_query_head_state", basic_ops_range_query_head_state);
    failures += run_test(filter, "basic_ops.erase_range_head_state", basic_ops_erase_range_head_state);
    failures += run_test(filter, "basic_ops.heterogeneous_lookups", basic_ops_heterogeneous_lookups);

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
    failures += run_test(filter, "transactional_consistency.repeated_read_matches_isolation",
                         transactional_consistency_repeated_read_matches_isolation);
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
    failures += run_test(filter, "forwarded_surface.for_each", forwarded_surface_for_each);
    failures += run_test(filter, "forwarded_surface.for_each_sees_every_stable_element",
                         forwarded_surface_for_each_sees_every_stable_element);

    return report_test_failures(failures);
}
