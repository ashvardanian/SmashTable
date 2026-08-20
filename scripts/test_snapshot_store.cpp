/**
 *  @brief Test instantiations for @c snapshot_store. Covers the promise that separates it
 *      from @c monotonic_store - a read answered at the stamp the transaction opened on - over both
 *      an ordered core and the open-addressed table, since the composite-key representation is uniform.
 *  @author Ash Vardanian
 *  @file scripts/test_snapshot_store.cpp
 *  @date August 17, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define ST_STRICT_CALLBACK_CHECKS_ 1

#include <cstddef> // `std::size_t`

#include <algorithm> // `std::sort`, `std::next_permutation`
#include <optional>  // `std::optional`
#include <random>    // `std::mt19937`
#include <vector>    // `std::vector`

#include <smashtable/snapshot_store.hpp>
#include <smashtable/monotonic_store.hpp>
#include <smashtable/partitioned_store.hpp>
#include <smashtable/reference_store.hpp>

#include "test.hpp"
#include "test_basic.hpp"
#include "test_consistency.hpp"
#include "test_sharded_concurrency.hpp"
#include "test_surface_parity.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

namespace {

#pragma region Type Aliases

/**
 *  Ordering: ✓ | Copy: Trivial (key & value) | Memory: Stack
 *  Tests: Snapshot reads, phantom-free ranges, run-based reclamation on a linked core
 */
using snapshot_avl_map_t =
    snapshot_avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/** The same core and members at the serializable level, so a suite can compare the two directly. */
using serializable_avl_map_t =
    serializable_avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/** @brief Serializable over an allocator that can be told to refuse, so a read can be made to fail. */
using budgeted_serializable_avl_map_t =
    serializable_avl_map<trivial_key_t, int, std::less<trivial_key_t>, stateful_allocator<mapping<trivial_key_t, int>>>;

/** @brief The same members one rung down, where nothing is recorded and so nothing can fail to be. */
using budgeted_snapshot_avl_map_t =
    snapshot_avl_map<trivial_key_t, int, std::less<trivial_key_t>, stateful_allocator<mapping<trivial_key_t, int>>>;

/** @brief Serializable and real-time ordered, so it must refuse everything the rung below refuses. */
using strict_serializable_avl_map_t = strict_serializable_avl_map<trivial_key_t, int, std::less<trivial_key_t>,
                                                                  std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Ordering: ✓ | Copy: Trivial (key & value) | Memory: Stack
 *  Tests: The same suites over the weight-balanced core
 */
using snapshot_wb_map_t =
    snapshot_wb_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Ordering: ✓ | Copy: Trivial (key & value) | Memory: Stack
 *  Tests: What an order statistic costs, since every comparison this comparator makes is counted
 */
using snapshot_ranked_map_t =
    snapshot_wb_map<trivial_key_t, int, counting_comparator_t, std::allocator<mapping<trivial_key_t, int>>>;

/**
 *  Ordering: ✗ | Copy: Trivial (key & value) | Memory: Stack
 *  Tests: Composite keys on the open-addressed core, where equality separates versions
 */
using snapshot_hash_map_t = snapshot_hash_map<trivial_key_t, int>;

/**
 *  Ordering: ✗ | Copy: .copy() → expected<T> | Memory: Heap
 *  Tests: Staging and reclamation with a key that allocates
 */
using snapshot_heavy_set_t = snapshot_hash_set<heavy_key_t>;

/**
 *  Ordering: ✓ | Copy: Trivial (key & value) | Memory: Stack
 *  Tests: One snapshot and one stamp shared by sixteen independently locked partitions
 */
using sharded_snapshot_map_t = partitioned_store<snapshot_avl_map_t>;

/** @brief The same map behind one mutex, where sharding has nothing to weaken. */
using monotonic_avl_map_t =
    monotonic_avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/** @brief A part that decides visibility by what is published rather than by a stamp, so sharding caps it. */
using sharded_monotonic_map_t = partitioned_store<monotonic_avl_map_t>;

/** @brief The @c std::set-backed oracle, which the range surfaces are checked against. */
using reference_avl_map_t =
    reference_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/** @brief The weight-balanced part of the lower isolation level, which is where its ordinals live. */
using monotonic_wb_map_t =
    monotonic_wb_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/** @brief The lower isolation level over the open-addressed core, where only an unordered walk exists. */
using monotonic_hash_map_t = monotonic_hash_map<trivial_key_t, int>;

/** @brief Whether a store offers the open-ended range erases, which only an ordered core can answer. */
template <typename store_type_>
concept erases_open_ended = requires(store_type_ &store) {
    store.erase_from(trivial_key_t {0});
    store.erase_up_to(trivial_key_t {0});
};

static_assert(erases_open_ended<snapshot_avl_map_t>, "an ordered core walks the keyspace from either end");
static_assert(erases_open_ended<monotonic_avl_map_t>, "the same surface, one isolation level down");
static_assert(erases_open_ended<reference_avl_map_t>, "the oracle is ordered by construction");
static_assert(!erases_open_ended<snapshot_hash_map_t>,
              "an unordered core refuses these exactly as it refuses `erase_range`");

static_assert(sharded_snapshot_map_t::isolation_k == isolation_t::snapshot_k,
              "sixteen partitions sharing one clock keep the promise the part makes alone");
static_assert(partitioned_store<snapshot_avl_map_t, hash<trivial_key_t>, spin_shared_mutex_t, 1>::isolation_k ==
                  isolation_t::snapshot_k,
              "a single partition never had anything to weaken");
static_assert(sharded_monotonic_map_t::isolation_k == isolation_t::read_committed_k,
              "a part without a stamp is still caught between two partition locks");
static_assert(partitioned_store<monotonic_avl_map_t, hash<trivial_key_t>, spin_shared_mutex_t, 1>::isolation_k ==
                  monotonic_avl_map_t::isolation_k,
              "one partition of an unstamped part is exactly the part");

#pragma endregion Type Aliases

#pragma region Helpers

/** @brief The value stored under @p identifier, or @c -1 when the key is not readable. */
template <typename readable_type_>
static int mapped_or_absent(readable_type_ const &readable, trivial_id_t identifier) noexcept {
    int observed = -1;
    st_verify_(readable.find(
        trivial_key_t {identifier}, [&](auto const &member) noexcept { observed = member.mapped; }, []() noexcept {}));
    return observed;
}

/** @brief How many versions a sweep reclaimed, insisting the sweep could run at all. */
template <typename store_type_, typename... bounds_types_>
[[nodiscard]] static std::size_t vacuumed(store_type_ &store, bounds_types_ &&...bounds) {
    expected<std::size_t> const reclaimed = store.vacuum(std::forward<bounds_types_>(bounds)...);
    st_verify_(reclaimed.has_value());
    return *reclaimed;
}

/** @brief Publishes @p value under @p identifier through a committed transaction. */
template <typename store_type_>
static void commit_write(store_type_ &store, trivial_id_t identifier, int value) {
    using member_t = typename store_type_::value_type;
    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(identifier, value)));
    st_verify_(writer->stage());
    st_verify_(writer->commit());
}

/** @brief Erases @p identifier through a committed transaction. */
template <typename store_type_>
static void commit_erase(store_type_ &store, trivial_id_t identifier) {
    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(writer->erase(trivial_key_t {identifier}));
    st_verify_(writer->stage());
    st_verify_(writer->commit());
}

#pragma endregion Helpers

#pragma region Isolation Tests

/** @brief Tests that the store advertises Snapshot Isolation and meets the optimistic store contract */
template <typename store_type_>
static void test_isolation_traits() {
    static_assert(store_type_::isolation_k == isolation_t::snapshot_k,
                  "the store exists to promise Snapshot Isolation");
    static_assert(at_least(store_type_::isolation_k, isolation_t::monotonic_atomic_view_k),
                  "Snapshot Isolation subsumes what `monotonic_store` promises");
    static_assert(optimistically_concurrent_store<store_type_>,
                  "the store stages, validates and commits like every other optimistic store");
    static_assert(store_type_::is_transactional::value);

    store_type_ store;
    st_verify_(store.empty());
    st_verify_eq_(store.versions_count(), 0);
}

/** @brief Tests that a read repeated inside one transaction survives an external commit */
template <typename store_type_>
static void test_repeated_read_is_stable() {

    store_type_ store;
    commit_write(store, 1, 100);
    commit_write(store, 2, 200);

    auto reader = store.transaction();
    st_verify_(reader.has_value());
    st_verify_eq_(mapped_or_absent(*reader, 1), 100);

    commit_write(store, 1, 111);
    commit_erase(store, 2);
    st_verify_eq_(mapped_or_absent(store, 1), 111);
    st_verify_eq_(mapped_or_absent(store, 2), -1);

    // The snapshot was fixed when the reader opened, so neither commit reaches it.
    st_verify_eq_(mapped_or_absent(*reader, 1), 100);
    st_verify_eq_(mapped_or_absent(*reader, 2), 200);
    st_verify_eq_(reader->contains(trivial_key_t {2}), true);
}

/** @brief Tests that a transaction reads its own writes before and after they are published */
template <typename store_type_>
static void test_reads_own_writes() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    commit_write(store, 1, 100);

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(1, 101)));
    st_verify_eq_(mapped_or_absent(*writer, 1), 101);
    st_verify_eq_(mapped_or_absent(store, 1), 100);

    st_verify_(writer->stage());
    st_verify_(writer->commit());
    st_verify_eq_(mapped_or_absent(store, 1), 101);
    st_verify_eq_(mapped_or_absent(*writer, 1), 101);
}

/** @brief Tests that a range walked twice inside one transaction admits no phantom */
template <typename store_type_>
static void test_range_admits_no_phantoms() {
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 10; identifier += 2) commit_write(store, identifier, 10);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    auto collect = [&]() {
        std::vector<trivial_id_t> seen;
        st_verify_(reader->range(trivial_key_t {0}, trivial_key_t {10},
                                 [&](auto const &member) noexcept { seen.push_back(member.key.unique_id); }));
        return seen;
    };

    std::vector<trivial_id_t> const first = collect();
    st_verify_eq_(first.size(), 5);

    // Every shape of external change at once: an insertion inside the window, an overwrite, an erase.
    commit_write(store, 3, 33);
    commit_write(store, 4, 44);
    commit_erase(store, 6);

    std::vector<trivial_id_t> const second = collect();
    st_verify_(first == second);
    st_verify_eq_(mapped_or_absent(*reader, 4), 10);

    std::vector<trivial_id_t> outside;
    st_verify_(store.range(trivial_key_t {0}, trivial_key_t {10},
                           [&](auto const &member) noexcept { outside.push_back(member.key.unique_id); }));
    st_verify_eq_(outside.size(), 5);
}

/** @brief Tests that the bounds a transaction reports also come from its own snapshot */
template <typename store_type_>
static void test_bounds_follow_the_snapshot() {
    store_type_ store;
    commit_write(store, 2, 20);
    commit_write(store, 6, 60);

    auto reader = store.transaction();
    st_verify_(reader.has_value());
    commit_write(store, 4, 40);

    trivial_id_t after_two = 0;
    st_verify_(reader->upper_bound(
        trivial_key_t {2}, [&](auto const &member) noexcept { after_two = member.key.unique_id; },
        []() noexcept { st_verify_(false && "the reader's snapshot still holds key 6"); }));
    st_verify_eq_(after_two, 6);

    trivial_id_t from_three = 0;
    st_verify_(reader->lower_bound(
        trivial_key_t {3}, [&](auto const &member) noexcept { from_three = member.key.unique_id; },
        []() noexcept { st_verify_(false && "the reader's snapshot still holds key 6"); }));
    st_verify_eq_(from_three, 6);

    trivial_id_t published = 0;
    st_verify_(store.lower_bound(
        trivial_key_t {3}, [&](auto const &member) noexcept { published = member.key.unique_id; },
        []() noexcept { st_verify_(false && "key 4 is published"); }));
    st_verify_eq_(published, 4);
}

#pragma endregion Isolation Tests

#pragma region Conflict Tests

/** @brief Tests that the first of two transactions writing one key keeps it */
template <typename store_type_>
static void test_first_committer_wins() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    commit_write(store, 1, 100);

    auto early = store.transaction();
    auto late = store.transaction();
    st_verify_(early.has_value() && late.has_value());

    st_verify_(early->upsert(trivial_id_to_member<member_t>(1, 111)));
    st_verify_(late->upsert(trivial_id_to_member<member_t>(1, 222)));

    st_verify_(early->stage());
    st_verify_(early->commit());

    // The loser is turned away rather than overwriting a version it never read.
    st_verify_eq_(late->stage(), status_t::write_conflict_k);
    st_verify_eq_(mapped_or_absent(store, 1), 111);
}

/** @brief Tests that a conflict is caught at commit when both transactions staged before either published */
template <typename store_type_>
static void test_conflict_caught_after_staging() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    commit_write(store, 1, 100);

    auto early = store.transaction();
    auto late = store.transaction();
    st_verify_(early.has_value() && late.has_value());

    st_verify_(early->upsert(trivial_id_to_member<member_t>(1, 111)));
    st_verify_(late->upsert(trivial_id_to_member<member_t>(1, 222)));

    // Staged versions carry no stamp, so neither transaction sees the other yet.
    st_verify_(early->stage());
    st_verify_(late->stage());
    st_verify_(early->commit());

    st_verify_eq_(late->commit(), status_t::write_conflict_k);
    st_verify_eq_(mapped_or_absent(store, 1), 111);
    st_verify_(late->rollback());
}

/** @brief Tests that a watch on a key someone else published over refuses the commit */
template <typename store_type_>
static void test_watch_refuses_lost_update() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    commit_write(store, 1, 100);
    commit_write(store, 2, 200);

    auto reader = store.transaction();
    st_verify_(reader.has_value());
    int observed = -1;
    st_verify_(reader->find_and_watch(
        trivial_key_t {1}, [&](auto const &member) noexcept { observed = member.mapped; }, []() noexcept {}));
    st_verify_eq_(observed, 100);
    st_verify_(reader->upsert(trivial_id_to_member<member_t>(2, 222)));

    commit_write(store, 1, 111);
    st_verify_eq_(reader->stage(), status_t::read_conflict_k);
    st_verify_eq_(mapped_or_absent(store, 2), 200);
}

/** @brief Tests that a watch on an absent key refuses a commit that raced an insert */
template <typename store_type_>
static void test_watch_records_absence() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    auto reader = store.transaction();
    st_verify_(reader.has_value());
    bool missed = false;
    st_verify_(
        reader->find_and_watch(trivial_key_t {1}, [](auto const &) noexcept {}, [&]() noexcept { missed = true; }));
    st_verify_(missed);
    st_verify_(reader->upsert(trivial_id_to_member<member_t>(2, 222)));

    commit_write(store, 1, 111);
    st_verify_eq_(reader->stage(), status_t::read_conflict_k);
}

/** @brief Tests that a watch on an absent key refuses a commit that inserted and erased it since */
template <typename store_type_>
static void test_watch_spans_insert_then_erase() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    auto reader = store.transaction();
    st_verify_(reader.has_value());
    bool missed = false;
    st_verify_(
        reader->find_and_watch(trivial_key_t {1}, [](auto const &) noexcept {}, [&]() noexcept { missed = true; }));
    st_verify_(missed);
    st_verify_(reader->upsert(trivial_id_to_member<member_t>(2, 222)));

    // The key is back to missing, and two commits the reader never saw sit between the two absences,
    // so resolving to the same shape is not the key having stood still.
    commit_write(store, 1, 111);
    commit_erase(store, 1);
    st_verify_eq_(mapped_or_absent(store, 1), -1);
    st_verify_eq_(reader->stage(), status_t::read_conflict_k);
}

/** @brief Tests that rollback hands staged writes back and lets a retry commit them */
template <typename store_type_>
static void test_rollback_returns_writes() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(1, 100)));
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(2, 200)));
    st_verify_(writer->stage());
    st_verify_eq_(writer->changes_count(), 0);

    st_verify_(writer->rollback());
    st_verify_eq_(writer->changes_count(), 2);
    st_verify_eq_(store.versions_count(), 0);
    st_verify_(store.empty());

    st_verify_(writer->stage());
    st_verify_(writer->commit());
    st_verify_eq_(store.size(), 2);
    st_verify_eq_(mapped_or_absent(store, 2), 200);
}

/** @brief Tests that an abandoned transaction leaves nothing behind and unpins the mark */
template <typename store_type_>
static void test_abandoned_transaction_leaves_nothing() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    commit_write(store, 1, 100);

    {
        auto abandoned = store.transaction();
        st_verify_(abandoned.has_value());
        st_verify_(abandoned->upsert(trivial_id_to_member<member_t>(1, 111)));
        st_verify_(abandoned->stage());
        st_verify_eq_(store.versions_count(trivial_key_t {1}), 2);
        st_verify_eq_(store.open_snapshots(), 1);
    }

    st_verify_eq_(store.open_snapshots(), 0);
    st_verify_eq_(store.versions_count(trivial_key_t {1}), 1);
    st_verify_eq_(mapped_or_absent(store, 1), 100);
}

#pragma endregion Conflict Tests

#pragma region Reclamation Tests

/** @brief Tests that with nothing open a key falls back to a single version on the next commit */
template <typename store_type_>
static void test_version_tail_collapses() {
    store_type_ store;
    for (int value = 0; value != 8; ++value) {
        commit_write(store, 1, value);
        st_verify_eq_(store.open_snapshots(), 0);
        st_verify_eq_(store.versions_count(trivial_key_t {1}), 1);
        st_verify_eq_(store.versions_count(), 1);
    }
    st_verify_eq_(mapped_or_absent(store, 1), 7);
    st_verify_eq_(vacuumed(store), 0);
}

/** @brief Tests that an open transaction pins the versions its snapshot still reaches */
template <typename store_type_>
static void test_open_transaction_pins_versions() {
    store_type_ store;
    commit_write(store, 1, 100);

    {
        auto reader = store.transaction();
        st_verify_(reader.has_value());
        commit_write(store, 1, 111);
        commit_write(store, 1, 222);

        // The mark cannot pass the reader, so the version it opened on stays reachable.
        st_verify_eq_(store.versions_count(trivial_key_t {1}), 3);
        st_verify_eq_(vacuumed(store), 0);
        st_verify_eq_(store.versions_count(trivial_key_t {1}), 3);
        st_verify_eq_(mapped_or_absent(*reader, 1), 100);
    }

    st_verify_eq_(store.open_snapshots(), 0);
    st_verify_eq_(vacuumed(store), 2);
    st_verify_eq_(store.versions_count(trivial_key_t {1}), 1);
    st_verify_eq_(mapped_or_absent(store, 1), 222);
}

/** @brief Tests that a committed tombstone is reclaimed once the mark passes the erasure */
template <typename store_type_>
static void test_tombstones_are_reclaimed() {
    store_type_ store;
    commit_write(store, 1, 100);
    commit_write(store, 2, 200);

    {
        auto reader = store.transaction();
        st_verify_(reader.has_value());
        commit_erase(store, 1);

        // The erasure hides the key from every reader that can see it, and pins it for the one that cannot.
        st_verify_eq_(mapped_or_absent(store, 1), -1);
        st_verify_eq_(mapped_or_absent(*reader, 1), 100);
        st_verify_eq_(store.size(), 1);
        st_verify_eq_(store.versions_count(trivial_key_t {1}), 2);
        st_verify_eq_(vacuumed(store), 0);
    }

    st_verify_eq_(vacuumed(store), 2);
    st_verify_eq_(store.versions_count(trivial_key_t {1}), 0);
    st_verify_eq_(store.versions_count(), 1);
    st_verify_eq_(store.size(), 1);
    st_verify_eq_(mapped_or_absent(store, 2), 200);
}

/** @brief Tests that an erase committed with nothing open costs the store no entry at all */
template <typename store_type_>
static void test_erase_with_nothing_open_costs_nothing() {
    store_type_ store;
    commit_write(store, 1, 100);
    commit_erase(store, 1);

    st_verify_eq_(store.versions_count(trivial_key_t {1}), 0);
    st_verify_eq_(store.versions_count(), 0);
    st_verify_(store.empty());
    st_verify_eq_(vacuumed(store), 0);
}

/** @brief Tests that a direct write outside any transaction publishes and prunes on the spot */
template <typename store_type_>
static void test_direct_writes_prune_themselves() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    for (int value = 0; value != 8; ++value) st_verify_(store.upsert(trivial_id_to_member<member_t>(1, value)));
    st_verify_eq_(store.versions_count(), 1);
    st_verify_eq_(mapped_or_absent(store, 1), 7);

    st_verify_(store.erase(trivial_key_t {1}));
    st_verify_eq_(store.versions_count(), 0);
    st_verify_eq_(store.erase(trivial_key_t {1}), status_t::key_not_found_k);
    st_verify_(store.empty());
}

/** @brief Tests that neither a direct write nor a sweep touches a version an open transaction staged */
template <typename store_type_>
static void test_pruning_spares_staged_versions() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    commit_write(store, 1, 100);

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(1, 111)));
    st_verify_(writer->stage());
    st_verify_eq_(store.versions_count(trivial_key_t {1}), 2);

    // A staged version carries no stamp, so no mark can reach past it.
    st_verify_(store.upsert(trivial_id_to_member<member_t>(1, 999)));
    st_verify_eq_(vacuumed(store), 0);
    st_verify_eq_(store.versions_count(trivial_key_t {1}), 3);
    st_verify_eq_(mapped_or_absent(store, 1), 999);

    st_verify_(writer->rollback());
    st_verify_eq_(writer->changes_count(), 1);
    st_verify_eq_(mapped_or_absent(store, 1), 999);
}

/** @brief Tests that many keys and many versions survive a single sweep intact */
template <typename store_type_>
static void test_bulk_sweep_keeps_every_reader_whole() {
    using member_t = typename store_type_::value_type;
    constexpr trivial_id_t keys_k = 64;
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier) commit_write(store, identifier, 0);

    {
        auto reader = store.transaction();
        st_verify_(reader.has_value());
        for (int round = 1; round != 4; ++round) {
            auto writer = store.transaction();
            st_verify_(writer.has_value());
            for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier)
                st_verify_(writer->upsert(trivial_id_to_member<member_t>(identifier, round)));
            st_verify_(writer->stage());
            st_verify_(writer->commit());
        }

        for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier) {
            st_verify_eq_(mapped_or_absent(*reader, identifier), 0);
            st_verify_eq_(mapped_or_absent(store, identifier), 3);
        }
        st_verify_eq_(store.versions_count(), keys_k * 4);
    }

    st_verify_eq_(vacuumed(store), keys_k * 3);
    st_verify_eq_(store.versions_count(), keys_k);
    st_verify_eq_(store.size(), keys_k);
}

/** @brief The oldest snapshot any open reader answers at, which is what retention may not pass. */
template <typename store_type_>
static generation_t oldest_open_snapshot(
    store_type_ const &store, std::vector<std::optional<typename store_type_::transaction_t>> const &readers) {
    generation_t oldest = store.published_stamp();
    bool anybody = false;
    for (auto const &reader : readers) {
        if (!reader) continue;
        if (!anybody || reader->snapshot() < oldest) oldest = reader->snapshot();
        anybody = true;
    }
    return oldest;
}

/** @brief Tests that the mark equals the oldest open snapshot after every arrival and every departure */
template <typename store_type_>
static void test_low_water_mark_tracks_the_oldest_reader() {
    using transaction_t = typename store_type_::transaction_t;
    constexpr std::size_t readers_k = 4;

    std::vector<std::size_t> closing_order {0, 1, 2, 3};
    do {
        store_type_ store;
        std::vector<std::optional<transaction_t>> readers(readers_k);

        for (std::size_t index = 0; index != readers_k; ++index) {
            commit_write(store, 1, static_cast<int>(index));
            auto opened = store.transaction();
            st_verify_(opened.has_value());
            readers[index].emplace(std::move(*opened));
            st_verify_eq_(store.open_snapshots(), index + 1);
            st_verify_eq_(store.low_water_mark(), oldest_open_snapshot(store, readers));
        }

        for (std::size_t position = 0; position != readers_k; ++position) {
            readers[closing_order[position]].reset();
            st_verify_eq_(store.open_snapshots(), readers_k - position - 1);
            st_verify_eq_(store.low_water_mark(), oldest_open_snapshot(store, readers));
            commit_write(store, 2, static_cast<int>(position));
            st_verify_eq_(store.low_water_mark(), oldest_open_snapshot(store, readers));
        }

        st_verify_eq_(store.open_snapshots(), 0);
        st_verify_eq_(store.low_water_mark(), store.published_stamp());
    } while (std::next_permutation(closing_order.begin(), closing_order.end()));
}

/** @brief Tests that a census that never empties still lets the mark follow its oldest reader */
template <typename store_type_>
static void test_rolling_readers_keep_reclamation_moving() {
    using transaction_t = typename store_type_::transaction_t;
    store_type_ store;
    commit_write(store, 1, 0);

    std::optional<transaction_t> holding;
    auto first = store.transaction();
    st_verify_(first.has_value());
    holding.emplace(std::move(*first));

    for (int round = 1; round != 64; ++round) {
        commit_write(store, 1, round);

        // The next reader arrives before the one before it leaves, so the census is never empty -
        // which is the shape a service with one perpetually open transaction has.
        auto opened = store.transaction();
        st_verify_(opened.has_value());
        std::optional<transaction_t> next;
        next.emplace(std::move(*opened));
        st_verify_eq_(store.open_snapshots(), 2);
        holding = std::move(next);

        st_verify_eq_(store.open_snapshots(), 1);
        st_verify_eq_(store.low_water_mark(), holding->snapshot());
        st_verify_le_(store.versions_count(trivial_key_t {1}), 3);
    }

    holding.reset();
    st_verify_eq_(store.open_snapshots(), 0);
    st_verify_eq_(mapped_or_absent(store, 1), 63);
}

/** @brief Tests that a reader that stays holds retention at its own snapshot once older ones leave */
template <typename store_type_>
static void test_long_lived_reader_holds_its_own_snapshot() {
    using transaction_t = typename store_type_::transaction_t;
    store_type_ store;
    commit_write(store, 1, 0);

    std::optional<transaction_t> early;
    auto opened_early = store.transaction();
    st_verify_(opened_early.has_value());
    early.emplace(std::move(*opened_early));

    commit_write(store, 1, 1);
    auto keeper = store.transaction();
    st_verify_(keeper.has_value());
    st_verify_lt_(early->snapshot(), keeper->snapshot());

    early.reset();
    st_verify_eq_(store.low_water_mark(), keeper->snapshot());

    for (int round = 2; round != 16; ++round) {
        commit_write(store, 1, round);
        {
            auto churning = store.transaction();
            st_verify_(churning.has_value());
            st_verify_eq_(store.open_snapshots(), 2);
            st_verify_eq_(store.low_water_mark(), keeper->snapshot());
        }
        st_verify_eq_(store.open_snapshots(), 1);
        st_verify_eq_(store.low_water_mark(), keeper->snapshot());
        st_verify_eq_(mapped_or_absent(*keeper, 1), 1);
    }
}

#pragma endregion Reclamation Tests

#pragma region Point Access Tests

/** @brief Tests that the four insert strategies differ exactly as documented */
template <typename store_type_>
static void test_point_insert_strategies() {
    using member_t = typename store_type_::value_type;

    store_type_ store;

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(writer->insert(trivial_id_to_member<member_t>(1)));
    st_verify_eq_(writer->insert(trivial_id_to_member<member_t>(1)), status_t::key_already_exists_k);
    st_verify_(writer->insert_if_missing(trivial_id_to_member<member_t>(1)));
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(1)));
    st_verify_(writer->update(trivial_id_to_member<member_t>(1)));
    st_verify_eq_(writer->update(trivial_id_to_member<member_t>(2)), status_t::key_not_found_k);
    st_verify_(writer->stage());
    st_verify_(writer->commit());
    st_verify_eq_(store.size(), 1);

    st_verify_eq_(store.insert(trivial_id_to_member<member_t>(1)), status_t::key_already_exists_k);
    st_verify_(store.insert_if_missing(trivial_id_to_member<member_t>(1)));
    st_verify_(store.insert(trivial_id_to_member<member_t>(2)));
    st_verify_eq_(store.update(trivial_id_to_member<member_t>(3)), status_t::key_not_found_k);
    st_verify_eq_(store.size(), 2);
    st_verify_eq_(store.count(trivial_key_t {2}), std::size_t {1});
}

/** @brief Tests that a key allocating on every copy stages, commits and reclaims like a trivial one */
static void test_heavy_keys_round_trip() {

    snapshot_heavy_set_t store;

    {
        auto writer = store.transaction();
        st_verify_(writer.has_value());
        for (trivial_id_t identifier = 0; identifier != 16; ++identifier) {
            auto key = heavy_key_t::make(identifier);
            st_verify_(key.has_value());
            st_verify_(writer->upsert(std::move(*key)));
        }
        st_verify_(writer->stage());
        st_verify_(writer->commit());
    }
    st_verify_eq_(store.size(), 16);

    auto probe = heavy_key_t::make(trivial_id_t {3});
    st_verify_(probe.has_value());
    st_verify_eq_(store.contains(*probe), true);
    st_verify_(store.erase(*probe));
    st_verify_eq_(store.contains(*probe), false);
    st_verify_eq_(store.versions_count(), 15);
}

/** @brief Tests that a group spanning a snapshot store and a monotonic one commits both together */
static void test_group_spans_both_stores() {
    using snapshot_t = snapshot_avl_map_t;
    using monotonic_t =
        monotonic_avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;
    using snapshot_member_t = typename snapshot_t::value_type;
    using monotonic_member_t = typename monotonic_t::value_type;

    snapshot_t snapshots;
    monotonic_t monotonics;

    auto group = make_transaction_group(snapshots, monotonics);
    st_verify_(group.has_value());
    st_verify_(group->template participant<0>().upsert(trivial_id_to_member<snapshot_member_t>(1, 100)));
    st_verify_(group->template participant<1>().upsert(trivial_id_to_member<monotonic_member_t>(1, 100)));

    st_verify_(group->stage());
    st_verify_eq_(snapshots.size(), 0);
    st_verify_eq_(monotonics.size(), 0);

    st_verify_(group->commit());
    st_verify_eq_(snapshots.size(), 1);
    st_verify_eq_(monotonics.size(), 1);
    st_verify_eq_(mapped_or_absent(snapshots, 1), 100);
    st_verify_eq_(mapped_or_absent(monotonics, 1), 100);
}

#pragma endregion Point Access Tests

#pragma region Cursor Tests

/** @brief Collects every key @p cursor still has, stepping it to exhaustion. */
template <typename cursor_type_>
static std::vector<trivial_id_t> drain_cursor(cursor_type_ cursor) {
    std::vector<trivial_id_t> seen;
    for (; !cursor.exhausted(); cursor.advance())
        cursor.peek([&](auto const &member) noexcept { seen.push_back(member.key.unique_id); },
                    []() noexcept { st_verify_(false && "a settled cursor always stands on a key"); });
    return seen;
}

/** @brief Tests that the cursor yields every visible key once, over tombstones and version tails alike */
template <typename store_type_>
static void test_cursor_yields_each_key_once() {
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 10; ++identifier) commit_write(store, identifier, 0);

    // An open reader pins the mark, so nothing below it is reclaimed and several versions of one key
    // stay visible at the published stamp at once - which is what the cursor must collapse to one key.
    auto pin = store.transaction();
    st_verify_(pin.has_value());
    for (int round = 1; round != 4; ++round)
        for (trivial_id_t identifier : {trivial_id_t {0}, trivial_id_t {3}, trivial_id_t {7}})
            commit_write(store, identifier, round);
    commit_erase(store, 5);
    commit_erase(store, 8);

    st_verify_eq_(store.versions_count(trivial_key_t {3}), 4);
    st_verify_eq_(store.versions_count(trivial_key_t {5}), 2);

    std::vector<trivial_id_t> const expected {0, 1, 2, 3, 4, 6, 7, 9};
    st_verify_(drain_cursor(store.visible_keys()) == expected);

    // The push-style walk and the resumable one must never disagree.
    std::vector<trivial_id_t> pushed;
    st_verify_(store.range(trivial_key_t {0}, trivial_key_t {10},
                           [&](auto const &member) noexcept { pushed.push_back(member.key.unique_id); }));
    st_verify_(pushed == expected);

    std::vector<trivial_id_t> const from_four {4, 6, 7, 9};
    st_verify_(drain_cursor(store.visible_keys_from(trivial_key_t {4})) == from_four);
    // Key 5 is a tombstone that still holds entries, so the cursor must step over its whole run.
    std::vector<trivial_id_t> const from_five {6, 7, 9};
    st_verify_(drain_cursor(store.visible_keys_from(trivial_key_t {5})) == from_five);

    auto standing = store.visible_keys_from(trivial_key_t {6});
    auto copied = standing.peek_copy();
    st_verify_(copied.has_value());
    st_verify_eq_(copied->key.unique_id, 6);
    st_verify_eq_(standing.snapshot(), store.published_stamp());

    // A cursor past the last key is exhausted rather than standing on nothing.
    auto beyond = store.visible_keys_from(trivial_key_t {10});
    st_verify_(beyond.exhausted());
    st_verify_(!store.visible_keys().exhausted());
}

/** @brief Tests that a transaction's own range is sorted where staged and committed keys interleave */
template <typename store_type_>
static void test_transaction_range_is_sorted() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 10; identifier += 2) commit_write(store, identifier, 10);

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    for (trivial_id_t identifier : {trivial_id_t {1}, trivial_id_t {3}, trivial_id_t {9}})
        st_verify_(writer->upsert(trivial_id_to_member<member_t>(identifier, 20)));
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(2, 22)));
    st_verify_(writer->erase(trivial_key_t {6}));

    std::vector<trivial_id_t> seen;
    std::vector<int> values;
    st_verify_(writer->range(trivial_key_t {0}, trivial_key_t {10}, [&](auto const &member) noexcept {
        seen.push_back(member.key.unique_id);
        values.push_back(member.mapped);
    }));

    std::vector<trivial_id_t> const expected {0, 1, 2, 3, 4, 8, 9};
    st_verify_(seen == expected);
    // The staged overwrite of key 2 speaks for it, and the staged tombstone hides key 6 entirely.
    std::vector<int> const expected_values {10, 20, 22, 20, 10, 10, 20};
    st_verify_(values == expected_values);

    // A window that opens on a staged key and closes inside the committed ones stays sorted too.
    std::vector<trivial_id_t> window;
    st_verify_(writer->range(trivial_key_t {1}, trivial_key_t {5},
                             [&](auto const &member) noexcept { window.push_back(member.key.unique_id); }));
    std::vector<trivial_id_t> const expected_window {1, 2, 3, 4};
    st_verify_(window == expected_window);
}

#pragma endregion Cursor Tests

#pragma region Publication Tests

/** @brief Tests that a many-key publication is observed all-or-nothing, never as a prefix */
template <typename store_type_>
static void test_publication_is_all_or_nothing() {
    using member_t = typename store_type_::value_type;
    constexpr trivial_id_t keys_k = 8;
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier) commit_write(store, identifier, 0);

    auto earlier = store.transaction();
    st_verify_(earlier.has_value());
    generation_t const before = store.published_stamp();

    auto group = store.publication();
    st_verify_(group.has_value());
    st_verify_(group->reserve(keys_k));
    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier)
        st_verify_(group->upsert(trivial_id_to_member<member_t>(identifier, 1)));
    st_verify_eq_(group->staged_count(), keys_k);

    // Filed and unstamped, so neither the store nor any reader can see any of it.
    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier)
        st_verify_eq_(mapped_or_absent(store, identifier), 0);
    st_verify_eq_(store.published_stamp(), before);

    st_verify_(group->publish());
    st_verify_eq_(group->staged_count(), 0);

    // One stamp for the whole group, so a snapshot names all of it or none of it.
    st_verify_eq_(store.published_stamp(), before + 1);
    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier) {
        st_verify_eq_(mapped_or_absent(store, identifier), 1);
        st_verify_eq_(mapped_or_absent(*earlier, identifier), 0);
    }

    auto later = store.transaction();
    st_verify_(later.has_value());
    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier)
        st_verify_eq_(mapped_or_absent(*later, identifier), 1);
}

/** @brief Tests that a publication erases through tombstones and undoes itself when discarded */
template <typename store_type_>
static void test_publication_erases_and_discards() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 4; ++identifier) commit_write(store, identifier, 10);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    {
        auto group = store.publication();
        st_verify_(group.has_value());
        st_verify_(group->erase(trivial_key_t {1}));
        st_verify_(group->erase(trivial_key_t {2}));
        st_verify_(group->publish());
    }
    st_verify_eq_(store.size(), 2);
    st_verify_eq_(mapped_or_absent(store, 1), -1);
    // The erasure is a tombstone, so the snapshot that opened before it still reads the key.
    st_verify_eq_(mapped_or_absent(*reader, 1), 10);
    st_verify_eq_(mapped_or_absent(*reader, 2), 10);

    {
        auto abandoned = store.publication();
        st_verify_(abandoned.has_value());
        st_verify_(abandoned->upsert(trivial_id_to_member<member_t>(0, 99)));
        st_verify_(abandoned->upsert(trivial_id_to_member<member_t>(3, 99)));
        st_verify_eq_(store.versions_count(trivial_key_t {0}), 2);
    }
    st_verify_eq_(mapped_or_absent(store, 0), 10);
    st_verify_eq_(store.versions_count(trivial_key_t {0}), 1);
    st_verify_eq_(store.size(), 2);
}

/** @brief Tests that clearing is refused while a transaction still holds a snapshot */
template <typename store_type_>
static void test_clear_refuses_open_readers() {
    store_type_ store;
    commit_write(store, 1, 100);

    {
        auto reader = store.transaction();
        st_verify_(reader.has_value());
        st_verify_eq_(store.clear(), status_t::operation_not_permitted_k);
        st_verify_eq_(mapped_or_absent(*reader, 1), 100);
    }

    st_verify_eq_(store.open_snapshots(), 0);
    st_verify_(store.clear());
    st_verify_(store.empty());
    st_verify_eq_(store.versions_count(), 0);
}

#pragma endregion Publication Tests

#pragma region Range Operation Tests

/** @brief Every key @p readable shows in [ @p lower, @p upper ), in order. */
template <typename readable_type_>
static std::vector<trivial_id_t> keys_in_range(readable_type_ const &readable, trivial_id_t lower, trivial_id_t upper) {
    std::vector<trivial_id_t> seen;
    st_verify_(readable.range(trivial_key_t {lower}, trivial_key_t {upper},
                              [&](auto const &member) noexcept { seen.push_back(member.key.unique_id); }));
    return seen;
}

/** @brief Tests that a single-key lookup answers through the range surface as well */
template <typename store_type_>
static void test_equal_range_collapses_to_find() {
    store_type_ store;
    commit_write(store, 1, 100);
    commit_write(store, 2, 200);
    commit_erase(store, 2);

    std::vector<int> found;
    st_verify_(
        store.equal_range(trivial_key_t {1}, [&](auto const &member) noexcept { found.push_back(member.mapped); }));
    st_verify_eq_(found.size(), 1);
    st_verify_eq_(found[0], 100);

    found.clear();
    st_verify_(
        store.equal_range(trivial_key_t {2}, [&](auto const &member) noexcept { found.push_back(member.mapped); }));
    st_verify_(found.empty());

    auto reader = store.transaction();
    st_verify_(reader.has_value());
    st_verify_(reader->upsert(trivial_id_to_member<typename store_type_::value_type>(2, 222)));
    found.clear();
    st_verify_(
        reader->equal_range(trivial_key_t {2}, [&](auto const &member) noexcept { found.push_back(member.mapped); }));
    st_verify_eq_(found.size(), 1);
    st_verify_eq_(found[0], 222);
}

/** @brief Tests that a range erase is all-or-nothing to a reader that opened before it */
template <typename store_type_>
static void test_erase_range_is_all_or_nothing() {
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 10; ++identifier) commit_write(store, identifier, 0);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    generation_t const before = store.published_stamp();
    std::vector<trivial_id_t> retired;
    st_verify_(store.erase_range(trivial_key_t {2}, trivial_key_t {8},
                                 [&](auto const &member) noexcept { retired.push_back(member.key.unique_id); }));

    // One stamp for the whole window is what makes it indivisible: a reader either names that stamp or
    // does not, so there is no observable moment where half the keys are gone.
    st_verify_eq_(store.published_stamp(), before + 1);
    st_verify_eq_(retired.size(), 6);
    st_verify_eq_(store.size(), 4);

    std::vector<trivial_id_t> const remaining = keys_in_range(store, 0, 10);
    st_verify_eq_(remaining.size(), 4);
    st_verify_eq_(remaining[0], 0);
    st_verify_eq_(remaining[3], 9);

    // The older snapshot keeps every key it opened on, all six of them, not a prefix.
    std::vector<trivial_id_t> const observed = keys_in_range(*reader, 0, 10);
    st_verify_eq_(observed.size(), 10);
    for (trivial_id_t identifier = 0; identifier != 10; ++identifier) st_verify_eq_(observed[identifier], identifier);

    st_verify_(store.erase_range(trivial_key_t {0}, trivial_key_t {0}));
    st_verify_eq_(store.size(), 4);
}

/** @brief Every key @p store shows, in order, which the open-ended erases are checked against. */
template <typename store_type_>
static std::vector<trivial_id_t> every_key(store_type_ const &store) {
    return keys_in_range(store, 0, 1000);
}

/** @brief Publishes the keys 10, 20, 30 and 40, so a bound can also fall between two of them. */
template <typename store_type_>
static void fill_decades(store_type_ &store) {
    using member_t = typename store_type_::value_type;
    for (trivial_id_t identifier = 10; identifier <= 40; identifier += 10)
        st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier, static_cast<int>(identifier))));
}

/** @brief Tests that an erase with no upper bound takes its lower bound with it */
template <typename store_type_>
static void test_erase_from_includes_its_bound() {
    store_type_ empty;
    st_verify_(empty.erase_from(trivial_key_t {7}));
    st_verify_eq_(empty.size(), 0);

    // A bound falling between two keys takes everything after the gap.
    store_type_ between;
    fill_decades(between);
    std::vector<trivial_id_t> retired;
    st_verify_(between.erase_from(trivial_key_t {25},
                                  [&](auto const &member) noexcept { retired.push_back(member.key.unique_id); }));
    st_verify_eq_(retired.size(), 2);
    st_verify_eq_(retired[0], 30);
    st_verify_eq_(retired[1], 40);
    std::vector<trivial_id_t> const kept = every_key(between);
    st_verify_eq_(kept.size(), 2);
    st_verify_eq_(kept[0], 10);
    st_verify_eq_(kept[1], 20);

    // The bound itself goes, which is the end `erase_range` includes.
    store_type_ on_key;
    fill_decades(on_key);
    st_verify_(on_key.erase_from(trivial_key_t {30}));
    std::vector<trivial_id_t> const survivors = every_key(on_key);
    st_verify_eq_(survivors.size(), 2);
    st_verify_eq_(survivors[0], 10);
    st_verify_eq_(survivors[1], 20);

    // Below every key the store empties, above every key nothing moves.
    store_type_ below;
    fill_decades(below);
    st_verify_(below.erase_from(trivial_key_t {5}));
    st_verify_eq_(below.size(), 0);

    store_type_ above;
    fill_decades(above);
    st_verify_(above.erase_from(trivial_key_t {45}));
    st_verify_eq_(above.size(), 4);
    st_verify_eq_(every_key(above).size(), 4);
}

/** @brief Tests that an erase with no lower bound stops short of its upper bound */
template <typename store_type_>
static void test_erase_up_to_excludes_its_bound() {
    store_type_ empty;
    st_verify_(empty.erase_up_to(trivial_key_t {7}));
    st_verify_eq_(empty.size(), 0);

    store_type_ between;
    fill_decades(between);
    std::vector<trivial_id_t> retired;
    st_verify_(between.erase_up_to(trivial_key_t {25},
                                   [&](auto const &member) noexcept { retired.push_back(member.key.unique_id); }));
    st_verify_eq_(retired.size(), 2);
    st_verify_eq_(retired[0], 10);
    st_verify_eq_(retired[1], 20);
    std::vector<trivial_id_t> const kept = every_key(between);
    st_verify_eq_(kept.size(), 2);
    st_verify_eq_(kept[0], 30);
    st_verify_eq_(kept[1], 40);

    // The bound itself stays, which is the end `erase_range` excludes.
    store_type_ on_key;
    fill_decades(on_key);
    st_verify_(on_key.erase_up_to(trivial_key_t {30}));
    std::vector<trivial_id_t> const survivors = every_key(on_key);
    st_verify_eq_(survivors.size(), 2);
    st_verify_eq_(survivors[0], 30);
    st_verify_eq_(survivors[1], 40);

    store_type_ below;
    fill_decades(below);
    st_verify_(below.erase_up_to(trivial_key_t {5}));
    st_verify_eq_(below.size(), 4);

    store_type_ above;
    fill_decades(above);
    st_verify_(above.erase_up_to(trivial_key_t {45}));
    st_verify_eq_(above.size(), 0);
}

/** @brief Tests that the reporting @c insert_if_missing fires the same callback on the same outcome */
template <typename store_type_>
static void test_insert_if_missing_reports_outcome() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    int inserted_mapped = -1;
    std::size_t existing_calls = 0;
    st_verify_(store.insert_if_missing(
        trivial_id_to_member<member_t>(1, 100), [&](auto const &member) noexcept { inserted_mapped = member.mapped; },
        [&](auto const &) noexcept { ++existing_calls; }));
    st_verify_eq_(inserted_mapped, 100);
    st_verify_eq_(existing_calls, 0);

    // The second call declines, which is still a success, so only the callback separates the cases.
    int existing_mapped = -1;
    std::size_t inserted_calls = 0;
    st_verify_(store.insert_if_missing(
        trivial_id_to_member<member_t>(1, 999), [&](auto const &) noexcept { ++inserted_calls; },
        [&](auto const &member) noexcept { existing_mapped = member.mapped; }));
    st_verify_eq_(existing_mapped, 100);
    st_verify_eq_(inserted_calls, 0);
    st_verify_eq_(mapped_or_absent(store, 1), 100);
}

/** @brief Tests that a range update publishes new versions rather than rewriting what readers hold */
template <typename store_type_>
static void test_update_range_leaves_readers_alone() {
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 6; ++identifier) commit_write(store, identifier, 10);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    generation_t const before = store.published_stamp();
    st_verify_(store.update_range(trivial_key_t {1}, trivial_key_t {4},
                                  [](auto const &, int &mapped) noexcept { mapped += 1; }));
    st_verify_eq_(store.published_stamp(), before + 1);

    st_verify_eq_(mapped_or_absent(store, 0), 10);
    st_verify_eq_(mapped_or_absent(store, 1), 11);
    st_verify_eq_(mapped_or_absent(store, 3), 11);
    st_verify_eq_(mapped_or_absent(store, 4), 10);
    st_verify_eq_(store.size(), 6);

    for (trivial_id_t identifier = 0; identifier != 6; ++identifier)
        st_verify_eq_(mapped_or_absent(*reader, identifier), 10);
}

/** @brief Tests that a bounded vacuum reclaims only the runs inside its window */
template <typename store_type_>
static void test_bounded_vacuum_sweeps_whole_runs() {
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 6; ++identifier) {
        auto reader = store.transaction();
        st_verify_(reader.has_value());
        commit_write(store, identifier, 1);
        commit_write(store, identifier, 2);
    }

    st_verify_eq_(store.open_snapshots(), 0);
    std::size_t const carried = store.versions_count();
    st_verify_gt_(carried, 6);

    [[maybe_unused]] std::size_t const swept = vacuumed(store, trivial_key_t {0}, trivial_key_t {3});
    for (trivial_id_t identifier = 0; identifier != 3; ++identifier)
        st_verify_eq_(store.versions_count(trivial_key_t {identifier}), 1);
    st_verify_eq_(store.size(), 6);

    [[maybe_unused]] std::size_t const rest = vacuumed(store, trivial_key_t {3}, trivial_key_t {6});
    st_verify_eq_(store.versions_count(), 6);
    st_verify_eq_(store.size(), 6);
    for (trivial_id_t identifier = 0; identifier != 6; ++identifier)
        st_verify_eq_(mapped_or_absent(store, identifier), 2);
}

/** @brief Tests that reservoir sampling draws only readable keys, once each per pass */
template <typename store_type_>
static void test_sample_reservoir_draws_visible_keys() {
    using member_t = typename store_type_::value_type;

    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 20; ++identifier) commit_write(store, identifier, 0);
    for (trivial_id_t identifier = 0; identifier != 20; identifier += 2) commit_erase(store, identifier);

    std::mt19937 generator(42);
    std::vector<member_t> reservoir(4);
    std::size_t seen = 0;
    st_verify_(store.sample_reservoir(trivial_key_t {0}, trivial_key_t {20}, generator, seen, reservoir.size(),
                                      reservoir.begin()));

    st_verify_eq_(seen, 10);
    for (std::size_t slot = 0; slot != reservoir.size(); ++slot) {
        st_verify_((reservoir[slot].key.unique_id % 2) == 1);
        st_verify_eq_(store.contains(reservoir[slot].key), true);
    }

    seen = 0;
    st_verify_(store.sample_reservoir(trivial_key_t {0}, trivial_key_t {4}, generator, seen, reservoir.size(),
                                      reservoir.begin()));
    st_verify_eq_(seen, 2);
}

/** @brief Tests that the bulk modifiers commit as one group and refuse the strict clash */
template <typename store_type_>
static void test_bulk_modifiers_commit_together() {
    using member_t = typename store_type_::value_type;

    store_type_ store;

    std::vector<member_t> batch;
    for (trivial_id_t identifier = 0; identifier != 5; ++identifier)
        batch.push_back(trivial_id_to_member<member_t>(identifier, 1));

    generation_t const before = store.published_stamp();
    st_verify_(store.insert(batch.begin(), batch.end()));
    st_verify_eq_(store.published_stamp(), before + 1);
    st_verify_eq_(store.size(), 5);

    // The strict variant probes what is published, so a batch overlapping a stored key is refused whole.
    std::vector<member_t> clashing;
    clashing.push_back(trivial_id_to_member<member_t>(9, 1));
    clashing.push_back(trivial_id_to_member<member_t>(3, 1));
    st_verify_eq_(store.insert(clashing.begin(), clashing.end()), status_t::key_already_exists_k);
    st_verify_eq_(store.size(), 5);
    st_verify_eq_(store.contains(trivial_key_t {9}), false);

    st_verify_(store.insert_if_missing(clashing.begin(), clashing.end()));
    st_verify_eq_(store.size(), 6);
    st_verify_eq_(mapped_or_absent(store, 3), 1);

    std::vector<member_t> revised;
    for (trivial_id_t identifier = 0; identifier != 5; ++identifier)
        revised.push_back(trivial_id_to_member<member_t>(identifier, 2));
    st_verify_(store.update(revised.begin(), revised.end()));
    for (trivial_id_t identifier = 0; identifier != 5; ++identifier)
        st_verify_eq_(mapped_or_absent(store, identifier), 2);

    std::vector<member_t> absent;
    absent.push_back(trivial_id_to_member<member_t>(77, 3));
    st_verify_eq_(store.update(absent.begin(), absent.end()), status_t::key_not_found_k);
    st_verify_eq_(store.contains(trivial_key_t {77}), false);

    std::vector<member_t> mixed;
    mixed.push_back(trivial_id_to_member<member_t>(0, 5));
    mixed.push_back(trivial_id_to_member<member_t>(42, 5));
    st_verify_(store.upsert(mixed.begin(), mixed.end()));
    st_verify_eq_(mapped_or_absent(store, 0), 5);
    st_verify_eq_(mapped_or_absent(store, 42), 5);

    st_verify_(store.insert(batch.end(), batch.end()));
}

#pragma endregion Range Operation Tests

#pragma region Order Statistic Tests

/** @brief Tests that the ordinal and the rank agree with the ordered walk, tombstones discounted */
template <typename store_type_>
static void test_order_statistics_match_the_walk() {
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 12; ++identifier) commit_write(store, identifier, 0);
    for (trivial_id_t identifier = 0; identifier != 12; identifier += 3) commit_erase(store, identifier);

    std::vector<trivial_id_t> const expected = keys_in_range(store, 0, 12);
    st_verify_eq_(expected.size(), 8);
    st_verify_eq_(store.ranked_size(), store.size());

    for (std::size_t ordinal = 0; ordinal != expected.size(); ++ordinal) {
        trivial_id_t drawn = 0;
        bool present = false;
        st_verify_(store.select(
            ordinal,
            [&](auto const &member) noexcept {
                drawn = member.key.unique_id;
                present = true;
            },
            []() noexcept {}));
        st_verify_(present);
        st_verify_eq_(drawn, expected[ordinal]);

        std::size_t ranked = 0;
        bool ranked_found = false;
        st_verify_(store.rank(
            trivial_key_t {drawn},
            [&](std::size_t position) noexcept {
                ranked = position;
                ranked_found = true;
            },
            []() noexcept {}));
        st_verify_(ranked_found);
        st_verify_eq_(ranked, ordinal);
    }

    bool overshot = false;
    st_verify_(store.select(expected.size(), [](auto const &) noexcept {}, [&]() noexcept { overshot = true; }));
    st_verify_(overshot);

    bool erased_ranked = false;
    st_verify_(store.rank(trivial_key_t {0}, [&](std::size_t) noexcept { erased_ranked = true; }, []() noexcept {}));
    st_verify_(!erased_ranked);
}

/** @brief Tests that the ordinal descends on the stored counts rather than walking the keys */
static void test_order_statistics_cost_a_logarithm() {
    using member_t = typename snapshot_ranked_map_t::value_type;
    std::size_t const size = 4096;

    snapshot_ranked_map_t store;
    std::vector<member_t> batch;
    for (trivial_id_t identifier = 0; identifier != size; ++identifier)
        batch.push_back(trivial_id_to_member<member_t>(identifier, 0));
    st_verify_(store.upsert(batch.begin(), batch.end()));
    st_verify_eq_(store.size(), size);
    st_verify_eq_(store.ranked_size(), size);

    call_tally_t::reset();
    trivial_id_t drawn = 0;
    st_verify_(
        store.select(size / 2, [&](auto const &member) noexcept { drawn = member.key.unique_id; }, []() noexcept {}));
    st_verify_eq_(drawn, size / 2);
    // A select reads only the stored subtree counts, so it never consults the comparator at all.
    st_verify_eq_(call_tally_t::comparisons_count(), 0);

    call_tally_t::reset();
    std::size_t ranked = 0;
    st_verify_(store.rank(
        trivial_key_t {size - 1}, [&](std::size_t position) noexcept { ranked = position; }, []() noexcept {}));
    st_verify_eq_(ranked, size - 1);
    // The budget covers both halves of a rank: the probe that decides the key is readable at all, and
    // the descent that sums the counts to its left.
    call_tally_t::verify_comparisons_logarithmic(size, 10);
}

/** @brief Tests that a transaction's ordinal counts its own staged writes into the committed order */
template <typename store_type_>
static void test_transaction_order_statistics_merge() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 10; identifier += 2) commit_write(store, identifier, 0);

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(5, 1)));
    st_verify_(writer->erase(trivial_key_t {4}));

    std::vector<trivial_id_t> const expected = keys_in_range(*writer, 0, 10);
    st_verify_eq_(expected.size(), 5);

    for (std::size_t ordinal = 0; ordinal != expected.size(); ++ordinal) {
        trivial_id_t drawn = 0;
        bool present = false;
        st_verify_(writer->select(
            ordinal,
            [&](auto const &member) noexcept {
                drawn = member.key.unique_id;
                present = true;
            },
            []() noexcept {}));
        st_verify_(present);
        st_verify_eq_(drawn, expected[ordinal]);

        std::size_t ranked = 0;
        bool ranked_found = false;
        st_verify_(writer->rank(
            trivial_key_t {drawn},
            [&](std::size_t position) noexcept {
                ranked = position;
                ranked_found = true;
            },
            []() noexcept {}));
        st_verify_(ranked_found);
        st_verify_eq_(ranked, ordinal);
    }

    bool overshot = false;
    st_verify_(writer->select(expected.size(), [](auto const &) noexcept {}, [&]() noexcept { overshot = true; }));
    st_verify_(overshot);

    bool erased_ranked = false;
    st_verify_(writer->rank(trivial_key_t {4}, [&](std::size_t) noexcept { erased_ranked = true; }, []() noexcept {}));
    st_verify_(!erased_ranked);
}

/** @brief Tests that the augmented counts follow every publication path, not only the point writes */
static void test_ranked_size_tracks_every_write() {
    using member_t = typename snapshot_wb_map_t::value_type;
    snapshot_wb_map_t store;

    std::vector<member_t> batch;
    for (trivial_id_t identifier = 0; identifier != 16; ++identifier)
        batch.push_back(trivial_id_to_member<member_t>(identifier, 0));
    st_verify_(store.insert(batch.begin(), batch.end()));
    st_verify_eq_(store.ranked_size(), store.size());

    st_verify_(store.erase_range(trivial_key_t {4}, trivial_key_t {12}));
    st_verify_eq_(store.ranked_size(), store.size());
    st_verify_eq_(store.ranked_size(), 8);

    st_verify_(store.update_range(trivial_key_t {0}, trivial_key_t {16},
                                  [](auto const &, int &mapped) noexcept { mapped = 7; }));
    st_verify_eq_(store.ranked_size(), store.size());

    commit_write(store, 100, 1);
    commit_erase(store, 100);
    st_verify_eq_(store.ranked_size(), store.size());

    [[maybe_unused]] std::size_t const swept = vacuumed(store);
    st_verify_eq_(store.ranked_size(), store.size());

    auto abandoned = store.transaction();
    st_verify_(abandoned.has_value());
    st_verify_(abandoned->upsert(trivial_id_to_member<member_t>(200, 1)));
    st_verify_(abandoned->stage());
    st_verify_eq_(store.ranked_size(), store.size());
    st_verify_(abandoned->rollback());
    st_verify_eq_(store.ranked_size(), store.size());
}

#pragma endregion Order Statistic Tests

#pragma region Enumeration Tests

/** @brief The identifiers an unordered enumeration reports, sorted so an ordered core cannot flatter it. */
template <typename readable_type_>
static std::vector<trivial_id_t> keys_enumerated(readable_type_ const &readable) {
    std::vector<trivial_id_t> seen;
    st_verify_(readable.for_each([&](auto const &member) noexcept { seen.push_back(member.key.unique_id); }));
    std::sort(seen.begin(), seen.end());
    return seen;
}

/** @brief Tests that the unordered walk reports every visible element once, and nothing else at all */
template <typename store_type_>
static void test_for_each_visits_every_element_once() {

    store_type_ store;
    st_verify_(keys_enumerated(store).empty());

    for (trivial_id_t identifier = 0; identifier != 12; ++identifier) commit_write(store, identifier, 0);
    for (trivial_id_t identifier = 0; identifier != 12; identifier += 3) commit_erase(store, identifier);

    std::vector<trivial_id_t> const seen = keys_enumerated(store);
    st_verify_eq_(seen.size(), store.size());
    st_verify_eq_(seen.size(), 8);
    for (std::size_t slot = 0; slot != seen.size(); ++slot) {
        st_verify_eq_(store.contains(trivial_key_t {seen[slot]}), true);
        // A key reported twice would sort next to itself, and a tombstone would sort in among the rest.
        if (slot) st_verify_ne_(seen[slot], seen[slot - 1]);
        st_verify_ne_(seen[slot] % 3, 0u);
    }
}

/** @brief Tests that a transaction enumerates its own staged writes and nobody else's */
template <typename store_type_>
static void test_for_each_merges_staged_writes() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 4; ++identifier) commit_write(store, identifier, 0);

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(9, 1)));
    st_verify_(writer->erase(trivial_key_t {1}));

    // A second transaction stages a key without publishing it, which neither side may enumerate.
    auto other = store.transaction();
    st_verify_(other.has_value());
    st_verify_(other->upsert(trivial_id_to_member<member_t>(7, 1)));
    st_verify_(other->stage());

    std::vector<trivial_id_t> const merged = keys_enumerated(*writer);
    std::vector<trivial_id_t> const expected_merged {0, 2, 3, 9};
    st_verify_(merged == expected_merged);

    std::vector<trivial_id_t> const published = keys_enumerated(store);
    std::vector<trivial_id_t> const expected_published {0, 1, 2, 3};
    st_verify_(published == expected_published);

    st_verify_(other->rollback());
}

/** @brief Tests that an enumeration answers at the reader's snapshot rather than at the newest commit */
template <typename store_type_>
static void test_for_each_follows_the_snapshot() {
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 4; ++identifier) commit_write(store, identifier, 0);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    commit_write(store, 4, 1);
    commit_erase(store, 0);

    std::vector<trivial_id_t> const observed = keys_enumerated(*reader);
    std::vector<trivial_id_t> const expected_observed {0, 1, 2, 3};
    st_verify_(observed == expected_observed);

    std::vector<trivial_id_t> const published = keys_enumerated(store);
    std::vector<trivial_id_t> const expected_published {1, 2, 3, 4};
    st_verify_(published == expected_published);
}

/** @brief Tests that an ordinal answers with one element rather than with every element after it */
template <typename store_type_>
static void test_select_answers_exactly_once() {
    trivial_id_t const size = 8;
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != size; ++identifier) commit_write(store, identifier, 0);

    for (trivial_id_t ordinal = 0; ordinal != size; ++ordinal) {
        std::size_t found_count = 0;
        std::size_t missing_count = 0;
        trivial_id_t drawn = size;
        st_verify_(store.select(
            ordinal,
            [&](auto const &member) noexcept {
                drawn = member.key.unique_id;
                ++found_count;
            },
            [&]() noexcept { ++missing_count; }));
        st_verify_eq_(found_count, 1u);
        st_verify_eq_(missing_count, 0u);
        st_verify_eq_(drawn, ordinal);
    }

    std::size_t overshot_found = 0;
    std::size_t overshot_missing = 0;
    st_verify_(
        store.select(size, [&](auto const &) noexcept { ++overshot_found; }, [&]() noexcept { ++overshot_missing; }));
    st_verify_eq_(overshot_found, 0u);
    st_verify_eq_(overshot_missing, 1u);
}

/** @brief Tests that a transaction's ordinal fires once as well, over its merged order */
template <typename store_type_>
static void test_transaction_select_answers_exactly_once() {
    trivial_id_t const size = 8;
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != size; ++identifier) commit_write(store, identifier, 0);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    for (trivial_id_t ordinal = 0; ordinal != size; ++ordinal) {
        std::size_t found_count = 0;
        std::size_t missing_count = 0;
        trivial_id_t drawn = size;
        st_verify_(reader->select(
            ordinal,
            [&](auto const &member) noexcept {
                drawn = member.key.unique_id;
                ++found_count;
            },
            [&]() noexcept { ++missing_count; }));
        st_verify_eq_(found_count, 1u);
        st_verify_eq_(missing_count, 0u);
        st_verify_eq_(drawn, ordinal);
    }
}

static void enumeration_visits_every_element_once() {
    test_for_each_visits_every_element_once<snapshot_avl_map_t>();
    test_for_each_visits_every_element_once<snapshot_wb_map_t>();
    test_for_each_visits_every_element_once<snapshot_hash_map_t>();
    test_for_each_visits_every_element_once<monotonic_avl_map_t>();
    test_for_each_visits_every_element_once<monotonic_hash_map_t>();
    test_for_each_visits_every_element_once<reference_avl_map_t>();
}

static void enumeration_merges_staged_writes() {
    test_for_each_merges_staged_writes<snapshot_avl_map_t>();
    test_for_each_merges_staged_writes<snapshot_hash_map_t>();
    test_for_each_merges_staged_writes<monotonic_avl_map_t>();
    test_for_each_merges_staged_writes<monotonic_hash_map_t>();
    test_for_each_merges_staged_writes<reference_avl_map_t>();
}

static void enumeration_follows_the_snapshot() {
    test_for_each_follows_the_snapshot<snapshot_avl_map_t>();
    test_for_each_follows_the_snapshot<snapshot_wb_map_t>();
    test_for_each_follows_the_snapshot<snapshot_hash_map_t>();
}

static void order_statistics_select_answers_exactly_once() {
    test_select_answers_exactly_once<snapshot_wb_map_t>();
    test_select_answers_exactly_once<snapshot_ranked_map_t>();
    test_select_answers_exactly_once<monotonic_wb_map_t>();
    test_transaction_select_answers_exactly_once<snapshot_wb_map_t>();
    test_transaction_select_answers_exactly_once<monotonic_wb_map_t>();
}

#pragma endregion Enumeration Tests

#pragma region Surface Convergence Tests

/** @brief The oracle, the lower isolation level, and this one, over a key that only moves. */
using snapshot_heavy_map_t =
    snapshot_avl_map<heavy_key_t, int, std::less<void>, std::allocator<mapping<heavy_key_t, int>>>;
using monotonic_heavy_map_t =
    monotonic_avl_map<heavy_key_t, int, std::less<void>, std::allocator<mapping<heavy_key_t, int>>>;
using reference_heavy_map_t = reference_map<heavy_key_t, int, std::less<void>>;

/** @brief Whether a mutator that can fail reports through a status rather than through @c void. */
template <typename store_type_>
constexpr bool update_range_reports_a_status =
    std::is_same_v<status_t, decltype(std::declval<store_type_ &>().update_range(
                                 std::declval<typename store_type_::identifier_t const &>(),
                                 std::declval<typename store_type_::identifier_t const &>(), std::declval<no_op_t>()))>;

/** @brief Whether a sweep answers with a count or with the reason it could not run, never one zero for both. */
template <typename store_type_>
constexpr bool vacuum_reports_through_expected =
    std::is_same_v<expected<std::size_t>, decltype(std::declval<store_type_ &>().vacuum())>;

/** @brief Whether an ordered walk may be asked to visit a window without being handed a callback. */
template <typename store_type_>
constexpr bool range_defaults_its_callback =
    requires(store_type_ const &store, typename store_type_::identifier_t const &key) { store.range(key, key); };

/** @brief Whether an insert that declines to overwrite may report only the branch a caller cares about. */
template <typename store_type_>
constexpr bool insert_if_missing_takes_one_callback =
    requires(store_type_ &store, typename store_type_::value_type &&element, no_op_t callback) {
        store.insert_if_missing(std::move(element), callback);
    };

/** @brief Whether a store refuses a key that is not already there, in the single and the batch form. */
template <typename store_type_>
constexpr bool refuses_an_absent_key =
    requires(store_type_ &store, typename store_type_::value_type &&element, typename store_type_::value_type *cursor) {
        store.update(std::move(element));
        store.update(cursor, cursor);
    };

/** @brief Whether a store and its transactions both answer for a first key, without being asked an ordinal. */
template <typename store_type_>
constexpr bool answers_a_smallest =
    requires(store_type_ const &store, typename store_type_::transaction_t const &transaction, no_op_t callback) {
        store.smallest(callback, callback);
        transaction.smallest(callback, callback);
    };

/** @brief Whether a store and its transactions both answer ordinals. */
template <typename store_type_>
constexpr bool answers_ordinals =
    requires(store_type_ const &store, typename store_type_::transaction_t const &transaction,
             typename store_type_::identifier_t const &key, std::size_t ordinal, no_op_t callback) {
        store.select(ordinal, callback, callback);
        store.rank(key, callback, callback);
        transaction.select(ordinal, callback, callback);
        transaction.rank(key, callback, callback);
    };

static_assert(answers_a_smallest<snapshot_avl_map_t>, "an ordered store opens a merged walk with a first key");
static_assert(answers_a_smallest<snapshot_wb_map_t>, "an ordered store opens a merged walk with a first key");
static_assert(answers_a_smallest<monotonic_avl_map_t>, "an ordered store opens a merged walk with a first key");
static_assert(!answers_a_smallest<snapshot_hash_map_t>, "an unordered core has no first key to name");
static_assert(!answers_ordinals<snapshot_avl_map_t>,
              "the AVL core keeps no subtree counts, which is why the seed must not ask for an ordinal");

static_assert(update_range_reports_a_status<snapshot_avl_map_t>, "a mutator that can fail needs a channel");
static_assert(update_range_reports_a_status<monotonic_avl_map_t>, "a mutator that can fail needs a channel");
static_assert(update_range_reports_a_status<reference_avl_map_t>, "a mutator that can fail needs a channel");

static_assert(vacuum_reports_through_expected<snapshot_avl_map_t>, "zero must not mean 'could not run'");
static_assert(vacuum_reports_through_expected<monotonic_avl_map_t>, "zero must not mean 'could not run'");
static_assert(vacuum_reports_through_expected<reference_avl_map_t>, "zero must not mean 'could not run'");

static_assert(range_defaults_its_callback<snapshot_avl_map_t>, "the trailing callback is optional everywhere");
static_assert(range_defaults_its_callback<monotonic_avl_map_t>, "the trailing callback is optional everywhere");
static_assert(range_defaults_its_callback<reference_avl_map_t>, "the trailing callback is optional everywhere");

static_assert(insert_if_missing_takes_one_callback<snapshot_avl_map_t>, "one call shape across the engines");
static_assert(insert_if_missing_takes_one_callback<monotonic_avl_map_t>, "one call shape across the engines");
static_assert(insert_if_missing_takes_one_callback<reference_avl_map_t>, "one call shape across the engines");

static_assert(refuses_an_absent_key<snapshot_avl_map_t>, "all four insert strategies, on every engine");
static_assert(refuses_an_absent_key<monotonic_avl_map_t>, "all four insert strategies, on every engine");
static_assert(refuses_an_absent_key<reference_avl_map_t>, "all four insert strategies, on every engine");

static_assert(answers_ordinals<snapshot_wb_map_t>, "an engine that can rank must rank at both levels");
static_assert(answers_ordinals<monotonic_wb_map_t>, "an engine that can rank must rank at both levels");
static_assert(answers_ordinals<reference_avl_map_t>, "the oracle ranks by walking, which is always possible");

static_assert(sample_one_surface_t::offered<snapshot_avl_map_t>, "every ordered engine draws one key");
static_assert(sample_one_surface_t::offered<monotonic_avl_map_t>, "every ordered engine draws one key");
static_assert(sample_one_surface_t::offered<reference_avl_map_t>, "every ordered engine draws one key");

static_assert(transaction_equal_range_surface_t::offered<reference_avl_map_t>,
              "the oracle answers a point range inside a transaction, like its siblings");

/** @brief Whether a store's self-alias is reachable, which is what makes it public vocabulary. */
template <typename store_type_>
constexpr bool names_itself = std::is_same_v<typename store_type_::store_t, store_type_>;

static_assert(names_itself<snapshot_avl_map_t>, "an engine names itself, as every wrapper does");
static_assert(names_itself<monotonic_avl_map_t>, "an engine names itself, as every wrapper does");
static_assert(names_itself<reference_avl_map_t>, "an engine names itself, as every wrapper does");
static_assert(names_itself<locked_store<snapshot_avl_map_t>>, "a wrapper names itself, as every engine does");
static_assert(names_itself<partitioned_store<snapshot_avl_map_t>>, "a wrapper names itself, as every engine does");

/**
 *  @brief A watch borrows the identifier it is given, and a watch on a fetched version copies through
 *    @c copy_safely - so both compile for a key that only moves.
 *
 *  Taking the addresses is the whole test: these bodies are only instantiated when something names
 *  them, so a move-only key would otherwise never reach them.
 */
template <typename store_type_>
static void test_watch_borrows_and_copies_safely() {
    using transaction_t = typename store_type_::transaction_t;
    using identifier_t = typename store_type_::identifier_t;
    using versioned_entry_t = typename store_type_::versioned_entry_t;

    [[maybe_unused]] auto const watch_identifier =
        static_cast<status_t (transaction_t::*)(identifier_t const &)>(&transaction_t::watch);
    [[maybe_unused]] auto const watch_version =
        static_cast<status_t (transaction_t::*)(versioned_entry_t const &)>(&transaction_t::watch);
}

/** @brief An insert that declines to overwrite says which branch it took, given only one callback. */
template <typename store_type_>
static void test_insert_if_missing_reports_the_fresh_insert() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    std::size_t inserted = 0;
    st_verify_(
        store.insert_if_missing(trivial_id_to_member<member_t>(1, 10), [&](auto const &) noexcept { ++inserted; }));
    st_verify_eq_(inserted, 1);

    st_verify_(
        store.insert_if_missing(trivial_id_to_member<member_t>(1, 20), [&](auto const &) noexcept { ++inserted; }));
    st_verify_eq_(inserted, 1);
    st_verify_eq_(mapped_or_absent(store, 1), 10);
}

/** @brief A strict write refuses a key that is not there, singly and in a batch, and changes nothing when it does. */
template <typename store_type_>
static void test_update_refuses_an_absent_key() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    st_verify_(store.update(trivial_id_to_member<member_t>(1, 10)) == status_t::key_not_found_k);
    st_verify_eq_(mapped_or_absent(store, 1), -1);

    st_verify_(store.upsert(trivial_id_to_member<member_t>(1, 10)));
    st_verify_(store.update(trivial_id_to_member<member_t>(1, 20)));
    st_verify_eq_(mapped_or_absent(store, 1), 20);

    member_t present_and_absent[2] = {trivial_id_to_member<member_t>(1, 30), trivial_id_to_member<member_t>(2, 40)};
    st_verify_(store.update(std::make_move_iterator(present_and_absent),
                            std::make_move_iterator(present_and_absent + 2)) == status_t::key_not_found_k);
    st_verify_eq_(mapped_or_absent(store, 1), 20);
    st_verify_eq_(mapped_or_absent(store, 2), -1);

    member_t both_present[1] = {trivial_id_to_member<member_t>(1, 50)};
    st_verify_(store.update(std::make_move_iterator(both_present), std::make_move_iterator(both_present + 1)));
    st_verify_eq_(mapped_or_absent(store, 1), 50);
}

/** @brief Every engine's ordinals must agree with the walk the oracle answers them by. */
template <typename store_type_>
static void test_ordinals_agree_with_the_oracle() {
    constexpr trivial_id_t keys_k = 16;
    store_type_ store;
    reference_avl_map_t oracle;

    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier) {
        commit_write(store, identifier, int(identifier));
        commit_write(oracle, identifier, int(identifier));
    }
    for (trivial_id_t identifier = 0; identifier < keys_k; identifier += 3) {
        commit_erase(store, identifier);
        commit_erase(oracle, identifier);
    }
    st_verify_eq_(store.size(), oracle.size());

    for (std::size_t ordinal = 0; ordinal != oracle.size() + 2; ++ordinal) {
        int expected_mapped = -1;
        int observed_mapped = -2;
        st_verify_(oracle.select(
            ordinal, [&](auto const &member) noexcept { expected_mapped = member.mapped; }, []() noexcept {}));
        st_verify_(store.select(
            ordinal, [&](auto const &member) noexcept { observed_mapped = member.mapped; },
            [&]() noexcept { observed_mapped = -1; }));
        st_verify_eq_(expected_mapped, observed_mapped);
    }

    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier) {
        std::size_t expected_rank = keys_k;
        std::size_t observed_rank = keys_k + 1;
        st_verify_(oracle.rank(
            trivial_key_t {identifier}, [&](std::size_t rank) noexcept { expected_rank = rank; }, []() noexcept {}));
        st_verify_(store.rank(
            trivial_key_t {identifier}, [&](std::size_t rank) noexcept { observed_rank = rank; },
            [&]() noexcept { observed_rank = keys_k; }));
        st_verify_eq_(expected_rank, observed_rank);
    }
}

/** @brief The oracle's transaction ranks over the merge of what it staged and what is committed. */
static void test_oracle_transaction_ordinals_include_staged_writes() {
    using member_t = typename reference_avl_map_t::value_type;
    reference_avl_map_t oracle;
    for (trivial_id_t identifier = 0; identifier != 8; identifier += 2) commit_write(oracle, identifier, 0);

    auto staging = oracle.transaction();
    st_verify_(staging.has_value());
    st_verify_(staging->upsert(trivial_id_to_member<member_t>(3, 1)));
    st_verify_(staging->erase(trivial_key_t {4}));

    // Committed 0, 2, 6 with 4 tombstoned, plus a staged 3, so the walk reads 0, 2, 3, 6.
    trivial_id_t const expected_order[4] = {0, 2, 3, 6};
    for (std::size_t ordinal = 0; ordinal != 4; ++ordinal) {
        trivial_id_t observed = 99;
        st_verify_(staging->select(
            ordinal, [&](auto const &member) noexcept { observed = member.key.unique_id; }, []() noexcept {}));
        st_verify_eq_(observed, expected_order[ordinal]);
    }

    std::size_t staged_rank = 99;
    st_verify_(
        staging->rank(trivial_key_t {3}, [&](std::size_t rank) noexcept { staged_rank = rank; }, []() noexcept {}));
    st_verify_eq_(staged_rank, 2);

    bool tombstoned_reported = false;
    st_verify_(
        staging->rank(trivial_key_t {4}, [](std::size_t) noexcept {}, [&]() noexcept { tombstoned_reported = true; }));
    st_verify_(tombstoned_reported);

    std::size_t counted = 0;
    st_verify_(staging->equal_range(trivial_key_t {3}, [&](auto const &) noexcept { ++counted; }));
    st_verify_eq_(counted, 1);
    st_verify_(staging->equal_range(trivial_key_t {4}, [&](auto const &) noexcept { ++counted; }));
    st_verify_eq_(counted, 1);
}

/** @brief A single draw lands inside its window and never names a key the newest commit hides. */
template <typename store_type_>
static void test_sample_one_draws_from_the_visible_window() {
    constexpr trivial_id_t keys_k = 16;
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier)
        commit_write(store, identifier, int(identifier));
    for (trivial_id_t identifier = 0; identifier < keys_k; identifier += 2) commit_erase(store, identifier);

    std::mt19937 generator(42);
    std::size_t drawn = 0;
    bool saw_lowest = false;
    bool saw_highest = false;
    for (int attempt = 0; attempt != 256; ++attempt)
        st_verify_(store.sample_one(trivial_key_t {4}, trivial_key_t {12}, generator, [&](auto const &member) noexcept {
            ++drawn;
            st_verify_ge_(member.key.unique_id, 4);
            st_verify_lt_(member.key.unique_id, 12);
            st_verify_(member.key.unique_id % 2 == 1);
            saw_lowest |= member.key.unique_id == 5;
            saw_highest |= member.key.unique_id == 11;
        }));
    st_verify_eq_(drawn, 256);
    st_verify_(saw_lowest && saw_highest);

    std::size_t out_of_range = 0;
    st_verify_(store.sample_one(trivial_key_t {100}, trivial_key_t {200}, generator,
                                [&](auto const &) noexcept { ++out_of_range; }));
    st_verify_eq_(out_of_range, 0);
}

/**
 *  @brief The smallest key is the one a full walk reports first, and nothing at all when none is readable.
 *
 *  Driven over a store whose entries are all tombstoned as well, since a key whose newest published
 *  version is an erase is present in the core and readable by nobody.
 */
template <typename store_type_>
static void test_smallest_opens_the_walk() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    st_verify_(store.smallest([](member_t const &) noexcept { st_verify_(false && "an empty store has no first key"); },
                              [&]() noexcept {}));

    for (trivial_id_t identifier : {40, 10, 30, 20})
        st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier, static_cast<int>(identifier))));

    trivial_id_t walked_first = 0;
    st_verify_(store.range(trivial_key_t {0}, trivial_key_t {1000}, [&](member_t const &member) noexcept {
        if (!walked_first) walked_first = mapping_key_or_itself(member).unique_id;
    }));

    trivial_id_t answered = 0;
    std::size_t missing_count = 0;
    st_verify_(
        store.smallest([&](member_t const &member) noexcept { answered = mapping_key_or_itself(member).unique_id; },
                       [&]() noexcept { ++missing_count; }));
    st_verify_eq_(missing_count, 0);
    st_verify_eq_(answered, walked_first);
    st_verify_eq_(answered, 10);

    // Erasing the incumbent has to move the answer on, not merely hide it.
    st_verify_(store.erase(trivial_id_to_key<member_t>(10)));
    st_verify_(store.smallest(
        [&](member_t const &member) noexcept { answered = mapping_key_or_itself(member).unique_id; }, no_op_t {}));
    st_verify_eq_(answered, 20);

    for (trivial_id_t identifier : {20, 30, 40}) st_verify_(store.erase(trivial_id_to_key<member_t>(identifier)));
    missing_count = 0;
    st_verify_(
        store.smallest([](member_t const &) noexcept { st_verify_(false && "a fully tombstoned store reads nothing"); },
                       [&]() noexcept { ++missing_count; }));
    st_verify_eq_(missing_count, 1);
}

/** @brief A transaction opens on its own snapshot and on its own staged writes, not on the newest commit. */
template <typename store_type_>
static void test_smallest_answers_at_the_readers_snapshot() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    for (trivial_id_t identifier : {30, 40})
        st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier, static_cast<int>(identifier))));

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    trivial_id_t answered = 0;
    st_verify_(reader->smallest(
        [&](member_t const &member) noexcept { answered = mapping_key_or_itself(member).unique_id; }, no_op_t {}));
    st_verify_eq_(answered, 30);

    // Only a reader holding a stamp is shielded from a commit landing under it; a monotonic reader
    // holds none and is meant to see the newest published state, which is the level it advertises.
    st_verify_(store.upsert(trivial_id_to_member<member_t>(10, 10)));
    st_verify_(reader->smallest(
        [&](member_t const &member) noexcept { answered = mapping_key_or_itself(member).unique_id; }, no_op_t {}));
    if constexpr (at_least(store_type_::isolation_k, isolation_t::snapshot_k)) { st_verify_eq_(answered, 30); }
    else { st_verify_eq_(answered, 10); }

    // Its own staged write does move it, at either level, so the key is put below everything committed
    // rather than between - which would only answer the same on a reader that is shielded.
    st_verify_(reader->upsert(trivial_id_to_member<member_t>(5, 5)));
    st_verify_(reader->smallest(
        [&](member_t const &member) noexcept { answered = mapping_key_or_itself(member).unique_id; }, no_op_t {}));
    st_verify_eq_(answered, 5);

    // A fresh reader opens on the committed key whether or not the earlier one was shielded from it.
    auto later = store.transaction();
    st_verify_(later.has_value());
    st_verify_(later->smallest(
        [&](member_t const &member) noexcept { answered = mapping_key_or_itself(member).unique_id; }, no_op_t {}));
    st_verify_eq_(answered, 10);
}

#pragma endregion Surface Convergence Tests

#pragma region Sharded Isolation Tests

/** @brief The smallest identifier the sharded store routes to @p partition_index. */
template <typename store_type_>
static trivial_id_t identifier_in_partition(std::size_t partition_index) {
    using hash_t = typename store_type_::hash_t;
    using identifier_t = typename store_type_::identifier_t;
    hash_t const hasher {};
    for (trivial_id_t identifier = 0; identifier != 4096; ++identifier)
        if (hasher(identifier_t {trivial_key_t {identifier}}) % store_type_::partitions_k == partition_index)
            return identifier;
    st_verify_(false && "sixteen partitions out of four thousand keys is not a hash function");
    return 0;
}

/** @brief One identifier per partition, so a transaction over them reaches every one of them. */
template <typename store_type_>
static std::vector<trivial_id_t> one_identifier_per_partition() {
    std::vector<trivial_id_t> identifiers;
    identifiers.reserve(store_type_::partitions_k);
    for (std::size_t partition_index = 0; partition_index != store_type_::partitions_k; ++partition_index)
        identifiers.push_back(identifier_in_partition<store_type_>(partition_index));
    return identifiers;
}

/** @brief Publishes @p value under every one of @p identifiers through a single committed transaction */
template <typename store_type_>
static void commit_write_many(store_type_ &store, std::vector<trivial_id_t> const &identifiers, int value) {
    using member_t = typename store_type_::value_type;
    auto writer = store.transaction();
    st_verify_(writer.has_value());
    for (trivial_id_t identifier : identifiers)
        st_verify_(writer->upsert(trivial_id_to_member<member_t>(identifier, value)));
    st_verify_(writer->stage());
    st_verify_(writer->commit());
}

/** @brief Tests that a reader answers every partition at the one snapshot it opened on */
template <typename store_type_>
static void test_sharded_reader_holds_one_snapshot() {
    store_type_ store;
    auto const identifiers = one_identifier_per_partition<store_type_>();
    commit_write_many(store, identifiers, 100);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    // Two rounds land while the reader is open, each spanning every partition. Neither may show.
    commit_write_many(store, identifiers, 200);
    commit_write_many(store, identifiers, 300);

    for (trivial_id_t identifier : identifiers) {
        int observed = -1;
        st_verify_(reader->find(
            trivial_key_t {identifier}, [&](auto const &member) noexcept { observed = member.mapped; },
            []() noexcept {}));
        st_verify_eq_(observed, 100);
    }

    // The reader's own writes are its own, in whichever partition they land.
    using member_t = typename store_type_::value_type;
    st_verify_(reader->upsert(trivial_id_to_member<member_t>(identifiers.front(), 111)));
    int written = -1;
    st_verify_(reader->find(
        trivial_key_t {identifiers.front()}, [&](auto const &member) noexcept { written = member.mapped; },
        []() noexcept {}));
    st_verify_eq_(written, 111);

    // A reader opening after both rounds sees the newer of them, whole.
    auto fresh = store.transaction();
    st_verify_(fresh.has_value());
    for (trivial_id_t identifier : identifiers) {
        int observed = -1;
        st_verify_(fresh->find(
            trivial_key_t {identifier}, [&](auto const &member) noexcept { observed = member.mapped; },
            []() noexcept {}));
        st_verify_eq_(observed, 300);
    }
}

/**
 *  @brief Tests that a transaction reused after its own commit reads that commit, in every partition
 *
 *  The snapshot a commit creates is taken on the way into the next operation rather than on the way
 *  out of the commit, so this is what proves the deferral is invisible.
 */
template <typename store_type_>
static void test_sharded_transaction_reads_its_own_commit() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    auto const identifiers = one_identifier_per_partition<store_type_>();
    commit_write_many(store, identifiers, 100);

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    for (int round = 2; round != 5; ++round) {
        for (trivial_id_t identifier : identifiers)
            st_verify_(writer->upsert(trivial_id_to_member<member_t>(identifier, round * 100)));
        st_verify_(writer->stage());
        st_verify_(writer->commit());

        // The same transaction, still open, answers every partition at the stamp it just published.
        for (trivial_id_t identifier : identifiers) {
            int observed = -1;
            st_verify_(writer->find(
                trivial_key_t {identifier}, [&](auto const &member) noexcept { observed = member.mapped; },
                []() noexcept {}));
            st_verify_eq_(observed, round * 100);
        }
    }

    // A reset moves the transaction forward, never back to a snapshot below its own commits.
    st_verify_(writer->reset());
    for (trivial_id_t identifier : identifiers) {
        int observed = -1;
        st_verify_(writer->find(
            trivial_key_t {identifier}, [&](auto const &member) noexcept { observed = member.mapped; },
            []() noexcept {}));
        st_verify_eq_(observed, 400);
    }
    for (trivial_id_t identifier : identifiers) st_verify_eq_(mapped_or_absent(store, identifier), 400);
}

/** @brief Tests that a range walked inside one transaction admits no key committed after it opened */
template <typename store_type_>
static void test_sharded_range_admits_no_phantoms() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    auto const identifiers = one_identifier_per_partition<store_type_>();
    commit_write_many(store, identifiers, 100);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    // The predicate is evaluated through the transaction, over a span wide enough that every
    // partition answers part of it.
    auto count_present = [&]() noexcept {
        std::size_t seen = 0;
        for (trivial_id_t candidate = 0; candidate != 64; ++candidate) {
            expected<bool> const is_present = reader->contains(trivial_key_t {candidate});
            st_verify_(is_present);
            if (*is_present) ++seen;
        }
        return seen;
    };
    std::size_t const before = count_present();
    st_verify_eq_(before, identifiers.size());

    // A whole second generation of keys arrives across every partition, and none of it may appear.
    std::vector<trivial_id_t> intruders;
    for (trivial_id_t candidate = 0; candidate != 64; ++candidate) {
        bool named = false;
        for (trivial_id_t identifier : identifiers) named = named || identifier == candidate;
        if (!named) intruders.push_back(candidate);
    }
    commit_write_many(store, intruders, 900);

    st_verify_eq_(count_present(), before);

    // The transaction's own insert is not a phantom - it is the transaction's own writing.
    st_verify_(reader->upsert(trivial_id_to_member<member_t>(intruders.front(), 7)));
    st_verify_eq_(count_present(), before + 1);
}

/**
 *  @brief Tests that one reader pins the version tail of every partition, not only the ones it read
 *
 *  Reclamation prunes to the oldest snapshot anybody holds, and a partition answering that question
 *  from its own readers would answer "nobody" for every partition this reader has yet to touch - and
 *  free the very version it is entitled to when it gets there.
 */
template <typename store_type_>
static void test_sharded_mark_pins_every_partition() {
    store_type_ store;
    auto const identifiers = one_identifier_per_partition<store_type_>();
    commit_write_many(store, identifiers, 100);

    {
        auto reader = store.transaction();
        st_verify_(reader.has_value());

        // Enough rounds that a version tail left unpinned would be collapsed several times over,
        // each round committed through a partition lock this reader is not holding.
        for (int round = 1; round != 8; ++round) commit_write_many(store, identifiers, 100 + round);

        // Every partition still answers with the version the reader opened on.
        for (trivial_id_t identifier : identifiers) {
            int observed = -1;
            st_verify_(reader->find(
                trivial_key_t {identifier}, [&](auto const &member) noexcept { observed = member.mapped; },
                []() noexcept {}));
            st_verify_eq_(observed, 100);
        }
    }

    // Once the reader has left, nothing is pinned and the next commit collapses the tails again.
    commit_write_many(store, identifiers, 500);
    for (trivial_id_t identifier : identifiers) st_verify_eq_(mapped_or_absent(store, identifier), 500);
}

#pragma endregion Sharded Isolation Tests

#pragma region Write Skew

/**
 *  @brief Whether two transactions can each read a shared invariant and break it between them.
 *
 *  The classic write skew: two keys hold a total nobody may drive to zero, two transactions each read
 *  both keys, and each clears the one the other did not. Their write sets are disjoint, so
 *  first-committer-wins never fires - it only ever compares what was @b written. Only a level that
 *  validates what was @b read can refuse the second one.
 *
 *  @return Whether the second transaction committed, which is the anomaly occurring.
 */
template <typename store_type_>
[[nodiscard]] static bool write_skew_commits() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    commit_write(store, 1, 1);
    commit_write(store, 2, 1);

    auto first = store.transaction();
    auto second = store.transaction();
    st_verify_(first.has_value() && second.has_value());

    // Each reads both keys and concludes the other one covers the invariant.
    std::size_t first_total = 0, second_total = 0;
    for (trivial_id_t key = 1; key != 3; ++key) {
        st_verify_(first->find(
            trivial_id_to_key<member_t>(key),
            [&](member_t const &held) noexcept { first_total += static_cast<std::size_t>(held.mapped); }, no_op_t {}));
        st_verify_(second->find(
            trivial_id_to_key<member_t>(key),
            [&](member_t const &held) noexcept { second_total += static_cast<std::size_t>(held.mapped); }, no_op_t {}));
    }
    st_verify_eq_(first_total, 2u);
    st_verify_eq_(second_total, 2u);

    st_verify_(first->upsert(trivial_id_to_member<member_t>(1, 0)));
    st_verify_(second->upsert(trivial_id_to_member<member_t>(2, 0)));

    st_verify_(first->stage());
    st_verify_(first->commit());

    if (failed(second->stage())) return false;
    return succeeded(second->commit());
}

/** @brief Snapshot isolation admits write skew, and the serializable level refuses it. */
static void test_write_skew_separates_the_levels() {
    st_verify_((write_skew_commits<snapshot_avl_map_t>()) &&
               "snapshot isolation compares only written keys, so it must admit write skew");
    st_verify_((!write_skew_commits<serializable_avl_map_t>()) &&
               "validating the read set is what refuses a transaction whose read was overwritten");
}

#pragma endregion Write Skew

/**
 *  @brief A key appearing inside a watched window refuses the transaction that walked it.
 *
 *  A phantom is not a change to any key the reader saw - it is a key that was not there to be seen, so
 *  no per-key watch can catch it. Recording the window is what makes it visible to a validator.
 */
static void test_phantom_insert_refuses_the_walker() {
    using store_t = serializable_avl_map_t;
    using member_t = typename store_t::value_type;
    store_t store;
    commit_write(store, 10, 1);
    commit_write(store, 30, 1);

    auto walker = store.transaction();
    st_verify_(walker.has_value());
    std::size_t seen = 0;
    st_verify_(walker->range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(100),
                             [&](member_t const &) noexcept { ++seen; }));
    st_verify_eq_(seen, 2u);

    commit_write(store, 20, 1); // lands inside the window the walker recorded

    st_verify_eq_(walker->stage(), status_t::phantom_conflict_k);
}

/** @brief An erase inside a watched window refuses too, since it publishes a tombstone in the window. */
static void test_phantom_erase_refuses_the_walker() {
    using store_t = serializable_avl_map_t;
    using member_t = typename store_t::value_type;
    store_t store;
    commit_write(store, 10, 1);
    commit_write(store, 30, 1);

    auto walker = store.transaction();
    st_verify_(walker.has_value());
    st_verify_(walker->range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(100), no_op_t {}));

    auto eraser = store.transaction();
    st_verify_(eraser.has_value());
    st_verify_(eraser->erase(trivial_id_to_key<member_t>(10)));
    st_verify_(eraser->stage());
    st_verify_(eraser->commit());

    st_verify_eq_(walker->stage(), status_t::phantom_conflict_k);
}

/**
 *  @brief A scan by repeated bounds is refused by a key that appears in the window it crossed.
 *
 *  Navigating by @c lower_bound is the natural way to walk an ordered store, and it used to record
 *  nothing at all - so a serializable transaction that scanned a window and acted on what it counted
 *  committed happily while a key landed inside that window behind it.
 */
static void test_phantom_refuses_a_bounded_scan() {
    using store_t = serializable_avl_map_t;
    using member_t = typename store_t::value_type;
    store_t store;
    commit_write(store, 10, 1);
    commit_write(store, 30, 1);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    // Step the window [10, 40) the way a caller would, by bounds rather than by a range call.
    std::size_t counted = 0;
    trivial_id_t at = 10;
    for (bool stepping = true; stepping;) {
        bool landed = false;
        trivial_id_t next = 0;
        st_verify_(reader->lower_bound(
            trivial_id_to_key<member_t>(at),
            [&](member_t const &member) noexcept {
                landed = true;
                next = static_cast<trivial_id_t>(mapping_key_or_itself<member_t>(member).unique_id);
            },
            no_op_t {}));
        if (!landed || next >= 40) break;
        ++counted;
        at = next + 1;
    }
    st_verify_eq_(counted, 2u);

    commit_write(store, 20, 1); // inside a window the scan crossed but never named before

    st_verify_eq_(reader->stage(), status_t::phantom_conflict_k);
}

/**
 *  @brief An ordinal read is refused by a key below it and left alone by one above it.
 *
 *  @c select and @c rank depend on how many keys precede their answer, so their window ends where the
 *  answer does. Recording them as whole-keyspace reads would be safe and would make every ordinal read
 *  conflict with every commit anywhere, which is the difference this pins.
 */
static void test_ordinal_window_ignores_a_commit_above_it() {
    using store_t = serializable_avl_map_t;

    // A commit above the ordinal's answer leaves it alone.
    {
        store_t store;
        for (trivial_id_t identifier : {10, 20, 30}) commit_write(store, identifier, 1);
        auto reader = store.transaction();
        st_verify_(reader.has_value());
        st_verify_(reader->select(0, no_op_t {}, no_op_t {}));
        commit_write(store, 99, 1); // ordered after the key `select(0)` landed on
        st_verify_(reader->stage());
        st_verify_(reader->commit());
    }

    // A commit below it moves the answer, so it must refuse.
    {
        store_t store;
        for (trivial_id_t identifier : {10, 20, 30}) commit_write(store, identifier, 1);
        auto reader = store.transaction();
        st_verify_(reader.has_value());
        st_verify_(reader->select(1, no_op_t {}, no_op_t {}));
        commit_write(store, 5, 1); // ordered before the key `select(1)` landed on
        st_verify_eq_(reader->stage(), status_t::phantom_conflict_k);
    }
}

/** @brief Each way a transaction can lose names itself, so a retry loop knows whether to bother. */
static void test_each_refusal_names_its_cause() {
    using store_t = serializable_avl_map_t;
    using member_t = typename store_t::value_type;

    // A key this transaction wrote was published over.
    {
        store_t store;
        commit_write(store, 1, 1);
        auto writer = store.transaction();
        st_verify_(writer.has_value());
        st_verify_(writer->upsert(trivial_id_to_member<member_t>(1, 7)));
        commit_write(store, 1, 2);
        st_verify_eq_(writer->stage(), status_t::write_conflict_k);
    }

    // A key this transaction read was published over.
    {
        store_t store;
        commit_write(store, 1, 1);
        auto reader = store.transaction();
        st_verify_(reader.has_value());
        st_verify_(reader->find(trivial_id_to_key<member_t>(1), no_op_t {}, no_op_t {}));
        st_verify_(reader->upsert(trivial_id_to_member<member_t>(2, 7)));
        commit_write(store, 1, 2);
        st_verify_eq_(reader->stage(), status_t::read_conflict_k);
    }

    // A window this transaction read gained a member.
    {
        store_t store;
        commit_write(store, 10, 1);
        auto walker = store.transaction();
        st_verify_(walker.has_value());
        st_verify_(walker->range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(100), no_op_t {}));
        commit_write(store, 50, 1);
        st_verify_eq_(walker->stage(), status_t::phantom_conflict_k);
    }
}

/** @brief A read of the whole keyspace names no key, so any commit at all refuses it. */
static void test_unbounded_read_refuses_any_commit() {
    using store_t = serializable_avl_map_t;
    store_t store;
    commit_write(store, 1, 1);

    auto walker = store.transaction();
    st_verify_(walker.has_value());
    st_verify_(walker->for_each(no_op_t {}));

    commit_write(store, 999, 1); // anywhere at all, since the read covered everything

    st_verify_eq_(walker->stage(), status_t::phantom_conflict_k);
}

/**
 *  @brief A read that cannot be recorded reports it where it happened, not only at the commit.
 *
 *  A validated read has to be written down before it can be validated, and writing it down allocates.
 *  Latching the failure and answering success left the caller an answer it had no reason to distrust and
 *  a transaction that could never commit, so the reason now travels out of the read that lost it. The
 *  latch stays as well, because a caller may discard the status and the commit still has to refuse.
 */
static void test_a_read_reports_what_it_could_not_record() {
    using store_t = budgeted_serializable_avl_map_t;
    using member_t = typename store_t::value_type;

    allocation_ledger_t ledger;
    {
        store_t store {typename store_t::allocator_t(ledger)};
        st_verify_(store.upsert(trivial_id_to_member<member_t>(1, 1)));
        st_verify_(store.upsert(trivial_id_to_member<member_t>(2, 1)));

        auto reader = store.transaction();
        st_verify_(reader);

        // Everything the transaction needed is already allocated, so the next request is the read set's.
        ledger.refuse_everything();
        std::size_t seen = 0;
        status_t const answered = reader->find(
            trivial_key_t {1}, [&](member_t const &held) noexcept { seen = static_cast<std::size_t>(held.mapped); },
            no_op_t {});
        st_verify_eq_(answered, status_t::out_of_memory_heap_k,
                      "a read that could not be recorded must say so at the read");
        st_verify_eq_(seen, 1u, "and it must still answer, because what it read is not in doubt");

        // A commit lands, so validation has to consult the read set rather than take its fast path.
        ledger.allow(unlimited_budget_k);
        st_verify_(store.upsert(trivial_id_to_member<member_t>(3, 1)));
        st_verify_(reader->upsert(trivial_id_to_member<member_t>(2, 5)));
        st_verify_eq_(reader->stage(), status_t::read_conflict_k,
                      "a read it cannot name is a read it cannot prove untouched");
        st_verify_(reader->reset());
    }
    ledger.verify_balanced();
}

/** @brief One rung down nothing is recorded, so the same refused allocator leaves the read untouched. */
static void test_an_unvalidated_read_records_nothing_to_lose() {
    using store_t = budgeted_snapshot_avl_map_t;
    using member_t = typename store_t::value_type;

    allocation_ledger_t ledger;
    {
        store_t store {typename store_t::allocator_t(ledger)};
        st_verify_(store.upsert(trivial_id_to_member<member_t>(1, 1)));

        auto reader = store.transaction();
        st_verify_(reader);

        ledger.refuse_everything();
        std::size_t seen = 0;
        st_verify_(reader->find(
            trivial_key_t {1}, [&](member_t const &held) noexcept { seen = static_cast<std::size_t>(held.mapped); },
            no_op_t {}));
        st_verify_eq_(seen, 1u);
        ledger.allow(unlimited_budget_k);
    }
    ledger.verify_balanced();
}

/**
 *  @brief A transaction reset after a whole-keyspace read can commit again.
 *
 *  Reading everything records that the read set names no key, which refuses any commit that saw a newer
 *  stamp - correct while that read stands. A reset clears the reads it was refusing for, so leaving the
 *  refusal behind left a transaction that could never commit again however many times it was reset.
 */
static void serializable_reset_clears_a_whole_keyspace_read() {
    using store_t = serializable_avl_map_t;
    using member_t = typename store_t::value_type;

    store_t store;
    st_verify_(store.upsert(trivial_id_to_member<member_t>(1)));

    auto reader = store.transaction();
    st_verify_(reader);
    st_verify_(reader->for_each(no_op_t {}));

    // Somebody commits, so the whole-keyspace read is now stale and the transaction must refuse.
    st_verify_(store.upsert(trivial_id_to_member<member_t>(2)));
    st_verify_eq_(reader->stage(), status_t::phantom_conflict_k);

    // Reset drops that read. What follows is a fresh transaction in all but name, and must commit.
    st_verify_(reader->reset());
    st_verify_(reader->upsert(trivial_id_to_member<member_t>(3)));

    // A commit lands on a key this transaction never read, after the reset - so validation has to look
    // past its fast path and consult the read set, which is where an uncleared refusal would be found.
    st_verify_(store.upsert(trivial_id_to_member<member_t>(4)));
    st_verify_(reader->stage());
    st_verify_((succeeded(reader->commit())) && "a reset transaction must not inherit the refusal it was reset from");
    st_verify_eq_(store.size(), 4u);
}

/**
 *  @brief The transaction's range surface answers from the transaction, not from the store.
 *
 *  A store-level walk inside a transaction would read past its own staging and write around it, which is
 *  the whole reason these exist at the transaction at all. Each is asked to see a staged write the store
 *  has not published, and each erasing walk is asked to stage rather than to publish.
 */
template <typename store_type_>
static void transaction_range_surface_sees_its_own_writes() {
    using store_t = store_type_;
    using member_t = typename store_t::value_type;

    store_t store;
    for (std::size_t identifier = 0; identifier != 8; ++identifier)
        st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier)));

    auto writer = store.transaction();
    st_verify_(writer);

    // A staged member the store has not published must be visible to every read on this transaction.
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(100)));
    st_verify_eq_(writer->count(trivial_id_to_key<member_t>(100)), std::size_t {1},
                  "a staged key must be counted by the transaction that staged it");
    st_verify_eq_(store.count(trivial_id_to_key<member_t>(100)), std::size_t {0},
                  "and must stay invisible to the store until it commits");

    auto const bounded = writer->lower_bound_copy(trivial_id_to_key<member_t>(100));
    st_verify_(bounded);
    st_verify_eq_(mapping_key_or_itself<member_t>(*bounded), trivial_id_to_key<member_t>(100),
                  "a bound copy must reach the transaction's own staged member");

    // An erasing walk stages tombstones; the store keeps its members until the commit lands.
    std::size_t erased = 0;
    st_verify_(writer->erase_range(trivial_id_to_key<member_t>(2), trivial_id_to_key<member_t>(5),
                                   [&](member_t const &) noexcept { ++erased; }));
    st_verify_eq_(erased, std::size_t {3}, "the window held three members");
    st_verify_eq_(writer->count(trivial_id_to_key<member_t>(3)), std::size_t {0},
                  "an erased key must be gone from the transaction that erased it");
    st_verify_eq_(store.count(trivial_id_to_key<member_t>(3)), std::size_t {1},
                  "and must remain in the store until the commit lands");

    st_verify_(writer->stage());
    st_verify_(writer->commit());
    st_verify_eq_(store.count(trivial_id_to_key<member_t>(3)), std::size_t {0},
                  "the commit publishes what the transaction staged");
    st_verify_eq_(store.count(trivial_id_to_key<member_t>(100)), std::size_t {1});
}

} // namespace

int main(int, char **) {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "enumeration.visits_every_element_once", enumeration_visits_every_element_once);
    failures += run_test(filter, "enumeration.merges_staged_writes", enumeration_merges_staged_writes);
    failures += run_test(filter, "enumeration.follows_the_snapshot", enumeration_follows_the_snapshot);
    failures +=
        run_test(filter, "order_statistics.select_answers_exactly_once", order_statistics_select_answers_exactly_once);

    failures += run_test(filter, "traits.avl", test_isolation_traits<snapshot_avl_map_t>);
    failures += run_test(filter, "traits.wb", test_isolation_traits<snapshot_wb_map_t>);
    failures += run_test(filter, "traits.hash", test_isolation_traits<snapshot_hash_map_t>);

    failures +=
        run_test(filter, "isolation.repeated_read_is_stable.avl", test_repeated_read_is_stable<snapshot_avl_map_t>);
    failures +=
        run_test(filter, "isolation.repeated_read_is_stable.wb", test_repeated_read_is_stable<snapshot_wb_map_t>);
    failures +=
        run_test(filter, "isolation.repeated_read_is_stable.hash", test_repeated_read_is_stable<snapshot_hash_map_t>);

    failures += run_test(filter, "isolation.reads_own_writes.avl", test_reads_own_writes<snapshot_avl_map_t>);
    failures += run_test(filter, "isolation.reads_own_writes.hash", test_reads_own_writes<snapshot_hash_map_t>);

    failures +=
        run_test(filter, "isolation.range_admits_no_phantoms.avl", test_range_admits_no_phantoms<snapshot_avl_map_t>);
    failures +=
        run_test(filter, "isolation.range_admits_no_phantoms.wb", test_range_admits_no_phantoms<snapshot_wb_map_t>);
    failures += run_test(filter, "isolation.bounds_follow_the_snapshot.avl",
                         test_bounds_follow_the_snapshot<snapshot_avl_map_t>);
    failures +=
        run_test(filter, "isolation.bounds_follow_the_snapshot.wb", test_bounds_follow_the_snapshot<snapshot_wb_map_t>);
    failures += run_test(filter, "transaction_surface.sees_its_own_writes.snapshot",
                         []() { transaction_range_surface_sees_its_own_writes<snapshot_avl_map_t>(); });
    failures += run_test(filter, "transaction_surface.sees_its_own_writes.serializable",
                         []() { transaction_range_surface_sees_its_own_writes<serializable_avl_map_t>(); });
    failures += run_test(filter, "serializable.reset_clears_a_whole_keyspace_read",
                         serializable_reset_clears_a_whole_keyspace_read);

    failures += run_test(filter, "write_skew.separates_the_levels", test_write_skew_separates_the_levels);
    failures += run_test(filter, "phantom.insert_refuses_the_walker", test_phantom_insert_refuses_the_walker);
    failures += run_test(filter, "phantom.erase_refuses_the_walker", test_phantom_erase_refuses_the_walker);
    failures += run_test(filter, "phantom.unbounded_read_refuses_any_commit", test_unbounded_read_refuses_any_commit);
    failures += run_test(filter, "phantom.refuses_a_bounded_scan", test_phantom_refuses_a_bounded_scan);
    failures += run_test(filter, "phantom.ordinal_window_ignores_a_commit_above_it",
                         test_ordinal_window_ignores_a_commit_above_it);
    failures += run_test(filter, "conflict.each_refusal_names_its_cause", test_each_refusal_names_its_cause);
    failures += run_test(filter, "conflict.first_committer_wins.avl", test_first_committer_wins<snapshot_avl_map_t>);
    failures += run_test(filter, "conflict.first_committer_wins.wb", test_first_committer_wins<snapshot_wb_map_t>);
    failures += run_test(filter, "conflict.first_committer_wins.hash", test_first_committer_wins<snapshot_hash_map_t>);
    failures +=
        run_test(filter, "conflict.caught_after_staging.avl", test_conflict_caught_after_staging<snapshot_avl_map_t>);
    failures +=
        run_test(filter, "conflict.caught_after_staging.hash", test_conflict_caught_after_staging<snapshot_hash_map_t>);
    failures +=
        run_test(filter, "conflict.watch_refuses_lost_update.avl", test_watch_refuses_lost_update<snapshot_avl_map_t>);
    failures += run_test(filter, "conflict.watch_refuses_lost_update.hash",
                         test_watch_refuses_lost_update<snapshot_hash_map_t>);
    failures += run_test(filter, "conflict.watch_records_absence.avl", test_watch_records_absence<snapshot_avl_map_t>);
    failures +=
        run_test(filter, "conflict.watch_records_absence.hash", test_watch_records_absence<snapshot_hash_map_t>);
    failures += run_test(filter, "conflict.watch_spans_insert_then_erase.avl",
                         test_watch_spans_insert_then_erase<snapshot_avl_map_t>);
    failures += run_test(filter, "conflict.watch_spans_insert_then_erase.hash",
                         test_watch_spans_insert_then_erase<snapshot_hash_map_t>);
    failures +=
        run_test(filter, "conflict.rollback_returns_writes.avl", test_rollback_returns_writes<snapshot_avl_map_t>);
    failures +=
        run_test(filter, "conflict.rollback_returns_writes.hash", test_rollback_returns_writes<snapshot_hash_map_t>);
    failures += run_test(filter, "conflict.abandoned_transaction_leaves_nothing.avl",
                         test_abandoned_transaction_leaves_nothing<snapshot_avl_map_t>);
    failures += run_test(filter, "conflict.abandoned_transaction_leaves_nothing.hash",
                         test_abandoned_transaction_leaves_nothing<snapshot_hash_map_t>);

    failures +=
        run_test(filter, "reclamation.version_tail_collapses.avl", test_version_tail_collapses<snapshot_avl_map_t>);
    failures +=
        run_test(filter, "reclamation.version_tail_collapses.wb", test_version_tail_collapses<snapshot_wb_map_t>);
    failures +=
        run_test(filter, "reclamation.version_tail_collapses.hash", test_version_tail_collapses<snapshot_hash_map_t>);
    failures += run_test(filter, "reclamation.open_transaction_pins_versions.avl",
                         test_open_transaction_pins_versions<snapshot_avl_map_t>);
    failures += run_test(filter, "reclamation.open_transaction_pins_versions.wb",
                         test_open_transaction_pins_versions<snapshot_wb_map_t>);
    failures += run_test(filter, "reclamation.open_transaction_pins_versions.hash",
                         test_open_transaction_pins_versions<snapshot_hash_map_t>);
    failures +=
        run_test(filter, "reclamation.tombstones_are_reclaimed.avl", test_tombstones_are_reclaimed<snapshot_avl_map_t>);
    failures += run_test(filter, "reclamation.tombstones_are_reclaimed.hash",
                         test_tombstones_are_reclaimed<snapshot_hash_map_t>);
    failures += run_test(filter, "reclamation.erase_with_nothing_open_costs_nothing.avl",
                         test_erase_with_nothing_open_costs_nothing<snapshot_avl_map_t>);
    failures += run_test(filter, "reclamation.erase_with_nothing_open_costs_nothing.hash",
                         test_erase_with_nothing_open_costs_nothing<snapshot_hash_map_t>);
    failures += run_test(filter, "reclamation.direct_writes_prune_themselves.avl",
                         test_direct_writes_prune_themselves<snapshot_avl_map_t>);
    failures += run_test(filter, "reclamation.direct_writes_prune_themselves.hash",
                         test_direct_writes_prune_themselves<snapshot_hash_map_t>);
    failures += run_test(filter, "reclamation.pruning_spares_staged_versions.avl",
                         test_pruning_spares_staged_versions<snapshot_avl_map_t>);
    failures += run_test(filter, "reclamation.pruning_spares_staged_versions.hash",
                         test_pruning_spares_staged_versions<snapshot_hash_map_t>);
    failures += run_test(filter, "reclamation.bulk_sweep_keeps_every_reader_whole.avl",
                         test_bulk_sweep_keeps_every_reader_whole<snapshot_avl_map_t>);
    failures += run_test(filter, "reclamation.bulk_sweep_keeps_every_reader_whole.wb",
                         test_bulk_sweep_keeps_every_reader_whole<snapshot_wb_map_t>);
    failures += run_test(filter, "reclamation.bulk_sweep_keeps_every_reader_whole.hash",
                         test_bulk_sweep_keeps_every_reader_whole<snapshot_hash_map_t>);

    failures += run_test(filter, "reclamation.low_water_mark_tracks_the_oldest_reader.avl",
                         test_low_water_mark_tracks_the_oldest_reader<snapshot_avl_map_t>);
    failures += run_test(filter, "reclamation.low_water_mark_tracks_the_oldest_reader.hash",
                         test_low_water_mark_tracks_the_oldest_reader<snapshot_hash_map_t>);
    failures += run_test(filter, "reclamation.rolling_readers_keep_reclamation_moving.avl",
                         test_rolling_readers_keep_reclamation_moving<snapshot_avl_map_t>);
    failures += run_test(filter, "reclamation.rolling_readers_keep_reclamation_moving.hash",
                         test_rolling_readers_keep_reclamation_moving<snapshot_hash_map_t>);
    failures += run_test(filter, "reclamation.long_lived_reader_holds_its_own_snapshot.avl",
                         test_long_lived_reader_holds_its_own_snapshot<snapshot_avl_map_t>);
    failures += run_test(filter, "reclamation.long_lived_reader_holds_its_own_snapshot.hash",
                         test_long_lived_reader_holds_its_own_snapshot<snapshot_hash_map_t>);

    failures += run_test(filter, "point.insert_strategies.avl", test_point_insert_strategies<snapshot_avl_map_t>);
    failures += run_test(filter, "point.insert_strategies.hash", test_point_insert_strategies<snapshot_hash_map_t>);
    failures +=
        run_test(filter, "cursor.yields_each_key_once.avl", test_cursor_yields_each_key_once<snapshot_avl_map_t>);
    failures += run_test(filter, "cursor.yields_each_key_once.wb", test_cursor_yields_each_key_once<snapshot_wb_map_t>);
    failures += run_test(filter, "cursor.transaction_range_is_sorted.avl",
                         test_transaction_range_is_sorted<snapshot_avl_map_t>);
    failures +=
        run_test(filter, "cursor.transaction_range_is_sorted.wb", test_transaction_range_is_sorted<snapshot_wb_map_t>);

    failures +=
        run_test(filter, "publication.all_or_nothing.avl", test_publication_is_all_or_nothing<snapshot_avl_map_t>);
    failures +=
        run_test(filter, "publication.all_or_nothing.hash", test_publication_is_all_or_nothing<snapshot_hash_map_t>);
    failures += run_test(filter, "publication.erases_and_discards.avl",
                         test_publication_erases_and_discards<snapshot_avl_map_t>);
    failures += run_test(filter, "publication.erases_and_discards.hash",
                         test_publication_erases_and_discards<snapshot_hash_map_t>);
    failures += run_test(filter, "publication.clear_refuses_open_readers.avl",
                         test_clear_refuses_open_readers<snapshot_avl_map_t>);
    failures += run_test(filter, "publication.clear_refuses_open_readers.hash",
                         test_clear_refuses_open_readers<snapshot_hash_map_t>);

    failures += run_test(filter, "range.equal_range_collapses_to_find.avl",
                         test_equal_range_collapses_to_find<snapshot_avl_map_t>);
    failures += run_test(filter, "range.equal_range_collapses_to_find.hash",
                         test_equal_range_collapses_to_find<snapshot_hash_map_t>);
    failures += run_test(filter, "range.erase_range_is_all_or_nothing.avl",
                         test_erase_range_is_all_or_nothing<snapshot_avl_map_t>);
    failures += run_test(filter, "range.erase_range_is_all_or_nothing.wb",
                         test_erase_range_is_all_or_nothing<snapshot_wb_map_t>);
    failures += run_test(filter, "range.erase_from_includes_its_bound.avl",
                         test_erase_from_includes_its_bound<snapshot_avl_map_t>);
    failures += run_test(filter, "range.erase_from_includes_its_bound.wb",
                         test_erase_from_includes_its_bound<snapshot_wb_map_t>);
    failures += run_test(filter, "range.erase_from_includes_its_bound.monotonic",
                         test_erase_from_includes_its_bound<monotonic_avl_map_t>);
    failures += run_test(filter, "range.erase_from_includes_its_bound.reference",
                         test_erase_from_includes_its_bound<reference_avl_map_t>);
    failures += run_test(filter, "range.erase_up_to_excludes_its_bound.avl",
                         test_erase_up_to_excludes_its_bound<snapshot_avl_map_t>);
    failures += run_test(filter, "range.erase_up_to_excludes_its_bound.wb",
                         test_erase_up_to_excludes_its_bound<snapshot_wb_map_t>);
    failures += run_test(filter, "range.erase_up_to_excludes_its_bound.monotonic",
                         test_erase_up_to_excludes_its_bound<monotonic_avl_map_t>);
    failures += run_test(filter, "range.erase_up_to_excludes_its_bound.reference",
                         test_erase_up_to_excludes_its_bound<reference_avl_map_t>);
    failures += run_test(filter, "modifiers.insert_if_missing_reports_outcome.avl",
                         test_insert_if_missing_reports_outcome<snapshot_avl_map_t>);
    failures += run_test(filter, "modifiers.insert_if_missing_reports_outcome.hash",
                         test_insert_if_missing_reports_outcome<snapshot_hash_map_t>);
    failures += run_test(filter, "modifiers.insert_if_missing_reports_outcome.monotonic",
                         test_insert_if_missing_reports_outcome<monotonic_avl_map_t>);
    failures += run_test(filter, "modifiers.insert_if_missing_reports_outcome.reference",
                         test_insert_if_missing_reports_outcome<reference_avl_map_t>);
    failures += run_test(filter, "range.update_range_leaves_readers_alone.avl",
                         test_update_range_leaves_readers_alone<snapshot_avl_map_t>);
    failures += run_test(filter, "range.update_range_leaves_readers_alone.wb",
                         test_update_range_leaves_readers_alone<snapshot_wb_map_t>);
    failures += run_test(filter, "range.bounded_vacuum_sweeps_whole_runs.avl",
                         test_bounded_vacuum_sweeps_whole_runs<snapshot_avl_map_t>);
    failures += run_test(filter, "range.bounded_vacuum_sweeps_whole_runs.wb",
                         test_bounded_vacuum_sweeps_whole_runs<snapshot_wb_map_t>);
    failures += run_test(filter, "range.sample_reservoir_draws_visible_keys.avl",
                         test_sample_reservoir_draws_visible_keys<snapshot_avl_map_t>);
    failures += run_test(filter, "range.sample_reservoir_draws_visible_keys.wb",
                         test_sample_reservoir_draws_visible_keys<snapshot_wb_map_t>);
    failures += run_test(filter, "range.bulk_modifiers_commit_together.avl",
                         test_bulk_modifiers_commit_together<snapshot_avl_map_t>);
    failures += run_test(filter, "range.bulk_modifiers_commit_together.hash",
                         test_bulk_modifiers_commit_together<snapshot_hash_map_t>);

    failures +=
        run_test(filter, "order.statistics_match_the_walk.wb", test_order_statistics_match_the_walk<snapshot_wb_map_t>);
    failures += run_test(filter, "order.statistics_cost_a_logarithm", test_order_statistics_cost_a_logarithm);
    failures += run_test(filter, "order.transaction_statistics_merge.avl",
                         test_transaction_order_statistics_merge<snapshot_avl_map_t>);
    failures += run_test(filter, "order.transaction_statistics_merge.wb",
                         test_transaction_order_statistics_merge<snapshot_wb_map_t>);
    failures += run_test(filter, "order.ranked_size_tracks_every_write", test_ranked_size_tracks_every_write);

    failures += run_test(filter, "point.heavy_keys_round_trip", test_heavy_keys_round_trip);
    failures += run_test(filter, "point.group_spans_both_stores", test_group_spans_both_stores);

    // The shared suite decides what to assert from `isolation_k`. Running it here is what compiles and
    // exercises its snapshot arms - every other container in the tree takes the weaker one.
    failures += run_test(filter, "transactional_consistency.lost_update_matches_isolation.snapshot",
                         test_lost_update_matches_isolation<snapshot_avl_map_t>);
    failures += run_test(filter, "transactional_consistency.lost_update_matches_isolation.serializable",
                         test_lost_update_matches_isolation<serializable_avl_map_t>);
    failures += run_test(filter, "serializable.a_read_reports_what_it_could_not_record",
                         test_a_read_reports_what_it_could_not_record);
    failures += run_test(filter, "snapshot.an_unvalidated_read_records_nothing_to_lose",
                         test_an_unvalidated_read_records_nothing_to_lose);
    failures += run_test(filter, "transactional_consistency.write_skew_matches_isolation.snapshot",
                         test_write_skew_matches_isolation<snapshot_avl_map_t>);
    failures += run_test(filter, "transactional_consistency.write_skew_matches_isolation.serializable",
                         test_write_skew_matches_isolation<serializable_avl_map_t>);
    failures += run_test(filter, "transactional_consistency.read_conflict_matches_isolation.snapshot",
                         test_read_conflict_matches_isolation<snapshot_avl_map_t>);
    failures += run_test(filter, "transactional_consistency.read_conflict_matches_isolation.serializable",
                         test_read_conflict_matches_isolation<serializable_avl_map_t>);
    failures += run_test(filter, "transactional_consistency.write_skew_matches_isolation.strict",
                         test_write_skew_matches_isolation<strict_serializable_avl_map_t>);
    failures += run_test(filter, "transactional_consistency.read_conflict_matches_isolation.strict",
                         test_read_conflict_matches_isolation<strict_serializable_avl_map_t>);
    failures += run_test(filter, "transactional_consistency.repeated_read_matches_isolation.avl",
                         test_repeated_read_matches_isolation<snapshot_avl_map_t>);
    failures += run_test(filter, "transactional_consistency.repeated_read_matches_isolation.wb",
                         test_repeated_read_matches_isolation<snapshot_wb_map_t>);
    failures += run_test(filter, "transactional_consistency.repeated_read_matches_isolation.hash",
                         test_repeated_read_matches_isolation<snapshot_hash_map_t>);
    failures += run_test(filter, "transactional_consistency.repeated_range_matches_isolation.avl",
                         test_repeated_range_matches_isolation<snapshot_avl_map_t>);
    failures += run_test(filter, "transactional_consistency.repeated_range_matches_isolation.wb",
                         test_repeated_range_matches_isolation<snapshot_wb_map_t>);

    // Sixteen partitions, one clock. The suites above pin what a single store promises; these pin
    // that sharding it does not quietly take that promise back.
    failures += run_test(filter, "convergence.watch_borrows_and_copies_safely.snapshot",
                         test_watch_borrows_and_copies_safely<snapshot_heavy_map_t>);
    failures += run_test(filter, "convergence.watch_borrows_and_copies_safely.monotonic",
                         test_watch_borrows_and_copies_safely<monotonic_heavy_map_t>);
    failures += run_test(filter, "convergence.watch_borrows_and_copies_safely.reference",
                         test_watch_borrows_and_copies_safely<reference_heavy_map_t>);
    failures += run_test(filter, "convergence.insert_if_missing_reports_the_fresh_insert.snapshot",
                         test_insert_if_missing_reports_the_fresh_insert<snapshot_avl_map_t>);
    failures += run_test(filter, "convergence.insert_if_missing_reports_the_fresh_insert.monotonic",
                         test_insert_if_missing_reports_the_fresh_insert<monotonic_avl_map_t>);
    failures += run_test(filter, "convergence.insert_if_missing_reports_the_fresh_insert.reference",
                         test_insert_if_missing_reports_the_fresh_insert<reference_avl_map_t>);
    failures += run_test(filter, "convergence.update_refuses_an_absent_key.snapshot",
                         test_update_refuses_an_absent_key<snapshot_avl_map_t>);
    failures += run_test(filter, "convergence.update_refuses_an_absent_key.monotonic",
                         test_update_refuses_an_absent_key<monotonic_avl_map_t>);
    failures += run_test(filter, "convergence.update_refuses_an_absent_key.reference",
                         test_update_refuses_an_absent_key<reference_avl_map_t>);
    failures += run_test(filter, "convergence.smallest_opens_the_walk.snapshot",
                         test_smallest_opens_the_walk<snapshot_avl_map_t>);
    failures += run_test(filter, "convergence.smallest_opens_the_walk.snapshot_wb",
                         test_smallest_opens_the_walk<snapshot_wb_map_t>);
    failures += run_test(filter, "convergence.smallest_opens_the_walk.monotonic",
                         test_smallest_opens_the_walk<monotonic_avl_map_t>);
    failures += run_test(filter, "convergence.smallest_answers_at_the_readers_snapshot.snapshot",
                         test_smallest_answers_at_the_readers_snapshot<snapshot_avl_map_t>);
    failures += run_test(filter, "convergence.smallest_answers_at_the_readers_snapshot.monotonic",
                         test_smallest_answers_at_the_readers_snapshot<monotonic_avl_map_t>);
    failures += run_test(filter, "convergence.ordinals_agree_with_the_oracle.snapshot",
                         test_ordinals_agree_with_the_oracle<snapshot_wb_map_t>);
    failures += run_test(filter, "convergence.ordinals_agree_with_the_oracle.monotonic",
                         test_ordinals_agree_with_the_oracle<monotonic_wb_map_t>);
    failures += run_test(filter, "convergence.oracle_transaction_ordinals_include_staged_writes",
                         test_oracle_transaction_ordinals_include_staged_writes);
    failures += run_test(filter, "convergence.sample_one_draws_from_the_visible_window.snapshot",
                         test_sample_one_draws_from_the_visible_window<snapshot_avl_map_t>);
    failures += run_test(filter, "convergence.sample_one_draws_from_the_visible_window.wb",
                         test_sample_one_draws_from_the_visible_window<snapshot_wb_map_t>);
    failures += run_test(filter, "convergence.sample_one_draws_from_the_visible_window.monotonic",
                         test_sample_one_draws_from_the_visible_window<monotonic_avl_map_t>);
    failures += run_test(filter, "convergence.sample_one_draws_from_the_visible_window.reference",
                         test_sample_one_draws_from_the_visible_window<reference_avl_map_t>);

    failures += run_test(filter, "sharded.reader_holds_one_snapshot",
                         test_sharded_reader_holds_one_snapshot<sharded_snapshot_map_t>);
    failures += run_test(filter, "sharded.transaction_reads_its_own_commit",
                         test_sharded_transaction_reads_its_own_commit<sharded_snapshot_map_t>);
    failures += run_test(filter, "sharded.range_admits_no_phantoms",
                         test_sharded_range_admits_no_phantoms<sharded_snapshot_map_t>);
    failures += run_test(filter, "sharded.mark_pins_every_partition",
                         test_sharded_mark_pins_every_partition<sharded_snapshot_map_t>);
    failures += run_test(filter, "sharded.transactional_consistency.repeated_read",
                         test_repeated_read_matches_isolation<sharded_snapshot_map_t>);
    failures += run_test(filter, "sharded.transactional_consistency.repeated_range",
                         test_repeated_range_matches_isolation<sharded_snapshot_map_t>);
    failures += run_test(filter, "sharded.refused_commit_publishes_nothing",
                         []() { test_refused_commit_publishes_nothing<sharded_snapshot_map_t>(); });
    failures += run_test(filter, "sharded.snapshot_spans_partitions",
                         []() { test_snapshot_spans_partitions<sharded_snapshot_map_t>(); });

    return report_test_failures(failures);
}
