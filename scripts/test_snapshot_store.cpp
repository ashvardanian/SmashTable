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

#include <random> // `std::mt19937`
#include <vector> // `std::vector`

#include <smashtable/snapshot_store.hpp>
#include <smashtable/monotonic_store.hpp>

#include "test.hpp"
#include "test_basic.hpp"
#include "test_consistency.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

#pragma region Type Aliases

/**
 *  Ordering: ✓ | Copy: Trivial (key & value) | Memory: Stack
 *  Tests: Snapshot reads, phantom-free ranges, run-based reclamation on a linked core
 */
using snapshot_avl_map_t =
    snapshot_avl_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

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

#pragma endregion Type Aliases

#pragma region Helpers

/** @brief The value stored under @p identifier, or @c -1 when the key is not readable. */
template <typename readable_type_>
static int mapped_or_absent(readable_type_ const &readable, trivial_id_t identifier) noexcept {
    int observed = -1;
    readable.find(
        trivial_key_t {identifier}, [&](auto const &member) noexcept { observed = member.mapped; }, []() noexcept {});
    return observed;
}

/** @brief Publishes @p value under @p identifier through a committed transaction. */
template <typename store_type_>
static void commit_write(store_type_ &store, trivial_id_t identifier, int value) {
    using member_t = typename store_type_::value_type;
    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(identifier, value))));
    st_verify_(succeeded(writer->stage()));
    st_verify_(succeeded(writer->commit()));
}

/** @brief Erases @p identifier through a committed transaction. */
template <typename store_type_>
static void commit_erase(store_type_ &store, trivial_id_t identifier) {
    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(succeeded(writer->erase(trivial_key_t {identifier})));
    st_verify_(succeeded(writer->stage()));
    st_verify_(succeeded(writer->commit()));
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
    st_verify_(reader->contains(trivial_key_t {2}));
}

/** @brief Tests that a transaction reads its own writes before and after they are published */
template <typename store_type_>
static void test_reads_own_writes() {
    using member_t = typename store_type_::value_type;
    store_type_ store;
    commit_write(store, 1, 100);

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(1, 101))));
    st_verify_eq_(mapped_or_absent(*writer, 1), 101);
    st_verify_eq_(mapped_or_absent(store, 1), 100);

    st_verify_(succeeded(writer->stage()));
    st_verify_(succeeded(writer->commit()));
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
        reader->range(trivial_key_t {0}, trivial_key_t {10},
                      [&](auto const &member) noexcept { seen.push_back(member.key.unique_id); });
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
    store.range(trivial_key_t {0}, trivial_key_t {10},
                [&](auto const &member) noexcept { outside.push_back(member.key.unique_id); });
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
    reader->upper_bound(
        trivial_key_t {2}, [&](auto const &member) noexcept { after_two = member.key.unique_id; },
        []() noexcept { st_verify_(false && "the reader's snapshot still holds key 6"); });
    st_verify_eq_(after_two, 6);

    trivial_id_t from_three = 0;
    reader->lower_bound(
        trivial_key_t {3}, [&](auto const &member) noexcept { from_three = member.key.unique_id; },
        []() noexcept { st_verify_(false && "the reader's snapshot still holds key 6"); });
    st_verify_eq_(from_three, 6);

    trivial_id_t published = 0;
    store.lower_bound(
        trivial_key_t {3}, [&](auto const &member) noexcept { published = member.key.unique_id; },
        []() noexcept { st_verify_(false && "key 4 is published"); });
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

    st_verify_(succeeded(early->upsert(trivial_id_to_member<member_t>(1, 111))));
    st_verify_(succeeded(late->upsert(trivial_id_to_member<member_t>(1, 222))));

    st_verify_(succeeded(early->stage()));
    st_verify_(succeeded(early->commit()));

    // The loser is turned away rather than overwriting a version it never read.
    st_verify_eq_(late->stage(), status_t::consistency_k);
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

    st_verify_(succeeded(early->upsert(trivial_id_to_member<member_t>(1, 111))));
    st_verify_(succeeded(late->upsert(trivial_id_to_member<member_t>(1, 222))));

    // Staged versions carry no stamp, so neither transaction sees the other yet.
    st_verify_(succeeded(early->stage()));
    st_verify_(succeeded(late->stage()));
    st_verify_(succeeded(early->commit()));

    st_verify_eq_(late->commit(), status_t::consistency_k);
    st_verify_eq_(mapped_or_absent(store, 1), 111);
    st_verify_(succeeded(late->rollback()));
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
    st_verify_(succeeded(reader->find_and_watch(
        trivial_key_t {1}, [&](auto const &member) noexcept { observed = member.mapped; }, []() noexcept {})));
    st_verify_eq_(observed, 100);
    st_verify_(succeeded(reader->upsert(trivial_id_to_member<member_t>(2, 222))));

    commit_write(store, 1, 111);
    st_verify_eq_(reader->stage(), status_t::consistency_k);
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
    st_verify_(succeeded(
        reader->find_and_watch(trivial_key_t {1}, [](auto const &) noexcept {}, [&]() noexcept { missed = true; })));
    st_verify_(missed);
    st_verify_(succeeded(reader->upsert(trivial_id_to_member<member_t>(2, 222))));

    commit_write(store, 1, 111);
    st_verify_eq_(reader->stage(), status_t::consistency_k);
}

/** @brief Tests that rollback hands staged writes back and lets a retry commit them */
template <typename store_type_>
static void test_rollback_returns_writes() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    auto writer = store.transaction();
    st_verify_(writer.has_value());
    st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(1, 100))));
    st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(2, 200))));
    st_verify_(succeeded(writer->stage()));
    st_verify_eq_(writer->changes_count(), 0);

    st_verify_(succeeded(writer->rollback()));
    st_verify_eq_(writer->changes_count(), 2);
    st_verify_eq_(store.versions_count(), 0);
    st_verify_(store.empty());

    st_verify_(succeeded(writer->stage()));
    st_verify_(succeeded(writer->commit()));
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
        st_verify_(succeeded(abandoned->upsert(trivial_id_to_member<member_t>(1, 111))));
        st_verify_(succeeded(abandoned->stage()));
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
    st_verify_eq_(store.vacuum(), 0);
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
        st_verify_eq_(store.vacuum(), 0);
        st_verify_eq_(store.versions_count(trivial_key_t {1}), 3);
        st_verify_eq_(mapped_or_absent(*reader, 1), 100);
    }

    st_verify_eq_(store.open_snapshots(), 0);
    st_verify_eq_(store.vacuum(), 2);
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
        st_verify_eq_(store.vacuum(), 0);
    }

    st_verify_eq_(store.vacuum(), 2);
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
    st_verify_eq_(store.vacuum(), 0);
}

/** @brief Tests that a direct write outside any transaction publishes and prunes on the spot */
template <typename store_type_>
static void test_direct_writes_prune_themselves() {
    using member_t = typename store_type_::value_type;
    store_type_ store;

    for (int value = 0; value != 8; ++value)
        st_verify_(succeeded(store.upsert(trivial_id_to_member<member_t>(1, value))));
    st_verify_eq_(store.versions_count(), 1);
    st_verify_eq_(mapped_or_absent(store, 1), 7);

    st_verify_(succeeded(store.erase(trivial_key_t {1})));
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
    st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(1, 111))));
    st_verify_(succeeded(writer->stage()));
    st_verify_eq_(store.versions_count(trivial_key_t {1}), 2);

    // A staged version carries no stamp, so no mark can reach past it.
    st_verify_(succeeded(store.upsert(trivial_id_to_member<member_t>(1, 999))));
    st_verify_eq_(store.vacuum(), 0);
    st_verify_eq_(store.versions_count(trivial_key_t {1}), 3);
    st_verify_eq_(mapped_or_absent(store, 1), 999);

    st_verify_(succeeded(writer->rollback()));
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
                st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(identifier, round))));
            st_verify_(succeeded(writer->stage()));
            st_verify_(succeeded(writer->commit()));
        }

        for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier) {
            st_verify_eq_(mapped_or_absent(*reader, identifier), 0);
            st_verify_eq_(mapped_or_absent(store, identifier), 3);
        }
        st_verify_eq_(store.versions_count(), keys_k * 4);
    }

    st_verify_eq_(store.vacuum(), keys_k * 3);
    st_verify_eq_(store.versions_count(), keys_k);
    st_verify_eq_(store.size(), keys_k);
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
    st_verify_(succeeded(writer->insert(trivial_id_to_member<member_t>(1))));
    st_verify_(failed(writer->insert(trivial_id_to_member<member_t>(1))));
    st_verify_(succeeded(writer->insert_if_missing(trivial_id_to_member<member_t>(1))));
    st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(1))));
    st_verify_(succeeded(writer->update(trivial_id_to_member<member_t>(1))));
    st_verify_(failed(writer->update(trivial_id_to_member<member_t>(2))));
    st_verify_(succeeded(writer->stage()));
    st_verify_(succeeded(writer->commit()));
    st_verify_eq_(store.size(), 1);

    st_verify_(failed(store.insert(trivial_id_to_member<member_t>(1))));
    st_verify_(succeeded(store.insert_if_missing(trivial_id_to_member<member_t>(1))));
    st_verify_(succeeded(store.insert(trivial_id_to_member<member_t>(2))));
    st_verify_(failed(store.update(trivial_id_to_member<member_t>(3))));
    st_verify_eq_(store.size(), 2);
    st_verify_eq_(store.count(trivial_key_t {2}), 1);
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
            st_verify_(succeeded(writer->upsert(std::move(*key))));
        }
        st_verify_(succeeded(writer->stage()));
        st_verify_(succeeded(writer->commit()));
    }
    st_verify_eq_(store.size(), 16);

    auto probe = heavy_key_t::make(trivial_id_t {3});
    st_verify_(probe.has_value());
    st_verify_(store.contains(*probe));
    st_verify_(succeeded(store.erase(*probe)));
    st_verify_(!store.contains(*probe));
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
    st_verify_(succeeded(group->template participant<0>().upsert(trivial_id_to_member<snapshot_member_t>(1, 100))));
    st_verify_(succeeded(group->template participant<1>().upsert(trivial_id_to_member<monotonic_member_t>(1, 100))));

    st_verify_(succeeded(group->stage()));
    st_verify_eq_(snapshots.size(), 0);
    st_verify_eq_(monotonics.size(), 0);

    st_verify_(succeeded(group->commit()));
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
    store.range(trivial_key_t {0}, trivial_key_t {10},
                [&](auto const &member) noexcept { pushed.push_back(member.key.unique_id); });
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
        st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(identifier, 20))));
    st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(2, 22))));
    st_verify_(succeeded(writer->erase(trivial_key_t {6})));

    std::vector<trivial_id_t> seen;
    std::vector<int> values;
    writer->range(trivial_key_t {0}, trivial_key_t {10}, [&](auto const &member) noexcept {
        seen.push_back(member.key.unique_id);
        values.push_back(member.mapped);
    });

    std::vector<trivial_id_t> const expected {0, 1, 2, 3, 4, 8, 9};
    st_verify_(seen == expected);
    // The staged overwrite of key 2 speaks for it, and the staged tombstone hides key 6 entirely.
    std::vector<int> const expected_values {10, 20, 22, 20, 10, 10, 20};
    st_verify_(values == expected_values);

    // A window that opens on a staged key and closes inside the committed ones stays sorted too.
    std::vector<trivial_id_t> window;
    writer->range(trivial_key_t {1}, trivial_key_t {5},
                  [&](auto const &member) noexcept { window.push_back(member.key.unique_id); });
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
    st_verify_(succeeded(group->reserve(keys_k)));
    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier)
        st_verify_(succeeded(group->upsert(trivial_id_to_member<member_t>(identifier, 1))));
    st_verify_eq_(group->staged_count(), keys_k);

    // Filed and unstamped, so neither the store nor any reader can see any of it.
    for (trivial_id_t identifier = 0; identifier != keys_k; ++identifier)
        st_verify_eq_(mapped_or_absent(store, identifier), 0);
    st_verify_eq_(store.published_stamp(), before);

    st_verify_(succeeded(group->publish()));
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
        st_verify_(succeeded(group->erase(trivial_key_t {1})));
        st_verify_(succeeded(group->erase(trivial_key_t {2})));
        st_verify_(succeeded(group->publish()));
    }
    st_verify_eq_(store.size(), 2);
    st_verify_eq_(mapped_or_absent(store, 1), -1);
    // The erasure is a tombstone, so the snapshot that opened before it still reads the key.
    st_verify_eq_(mapped_or_absent(*reader, 1), 10);
    st_verify_eq_(mapped_or_absent(*reader, 2), 10);

    {
        auto abandoned = store.publication();
        st_verify_(abandoned.has_value());
        st_verify_(succeeded(abandoned->upsert(trivial_id_to_member<member_t>(0, 99))));
        st_verify_(succeeded(abandoned->upsert(trivial_id_to_member<member_t>(3, 99))));
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
    st_verify_(succeeded(store.clear()));
    st_verify_(store.empty());
    st_verify_eq_(store.versions_count(), 0);
}

#pragma endregion Publication Tests

#pragma region Range Operation Tests

/** @brief Every key @p readable shows in [ @p lower, @p upper ), in order. */
template <typename readable_type_>
static std::vector<trivial_id_t> keys_in_range(readable_type_ const &readable, trivial_id_t lower, trivial_id_t upper) {
    std::vector<trivial_id_t> seen;
    readable.range(trivial_key_t {lower}, trivial_key_t {upper},
                   [&](auto const &member) noexcept { seen.push_back(member.key.unique_id); });
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
    store.equal_range(trivial_key_t {1}, [&](auto const &member) noexcept { found.push_back(member.mapped); });
    st_verify_eq_(found.size(), 1);
    st_verify_eq_(found[0], 100);

    found.clear();
    store.equal_range(trivial_key_t {2}, [&](auto const &member) noexcept { found.push_back(member.mapped); });
    st_verify_(found.empty());

    auto reader = store.transaction();
    st_verify_(reader.has_value());
    st_verify_(succeeded(reader->upsert(trivial_id_to_member<typename store_type_::value_type>(2, 222))));
    found.clear();
    reader->equal_range(trivial_key_t {2}, [&](auto const &member) noexcept { found.push_back(member.mapped); });
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
    st_verify_(succeeded(store.erase_range(trivial_key_t {2}, trivial_key_t {8}, [&](auto const &member) noexcept {
        retired.push_back(member.key.unique_id);
    })));

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

    st_verify_(succeeded(store.erase_range(trivial_key_t {0}, trivial_key_t {0})));
    st_verify_eq_(store.size(), 4);
}

/** @brief Tests that a range update publishes new versions rather than rewriting what readers hold */
template <typename store_type_>
static void test_update_range_leaves_readers_alone() {
    store_type_ store;
    for (trivial_id_t identifier = 0; identifier != 6; ++identifier) commit_write(store, identifier, 10);

    auto reader = store.transaction();
    st_verify_(reader.has_value());

    generation_t const before = store.published_stamp();
    st_verify_(succeeded(store.update_range(trivial_key_t {1}, trivial_key_t {4},
                                            [](auto const &, int &mapped) noexcept { mapped += 1; })));
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
    st_verify_(carried > 6);

    [[maybe_unused]] std::size_t const swept = store.vacuum(trivial_key_t {0}, trivial_key_t {3});
    for (trivial_id_t identifier = 0; identifier != 3; ++identifier)
        st_verify_eq_(store.versions_count(trivial_key_t {identifier}), 1);
    st_verify_eq_(store.size(), 6);

    [[maybe_unused]] std::size_t const rest = store.vacuum(trivial_key_t {3}, trivial_key_t {6});
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
    store.sample_reservoir(trivial_key_t {0}, trivial_key_t {20}, generator, seen, reservoir.size(), reservoir.begin());

    st_verify_eq_(seen, 10);
    for (std::size_t slot = 0; slot != reservoir.size(); ++slot) {
        st_verify_((reservoir[slot].key.unique_id % 2) == 1);
        st_verify_(store.contains(reservoir[slot].key));
    }

    seen = 0;
    store.sample_reservoir(trivial_key_t {0}, trivial_key_t {4}, generator, seen, reservoir.size(), reservoir.begin());
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
    st_verify_(succeeded(store.insert(batch.begin(), batch.end())));
    st_verify_eq_(store.published_stamp(), before + 1);
    st_verify_eq_(store.size(), 5);

    // The strict variant probes what is published, so a batch overlapping a stored key is refused whole.
    std::vector<member_t> clashing;
    clashing.push_back(trivial_id_to_member<member_t>(9, 1));
    clashing.push_back(trivial_id_to_member<member_t>(3, 1));
    st_verify_eq_(store.insert(clashing.begin(), clashing.end()), status_t::key_already_exists_k);
    st_verify_eq_(store.size(), 5);
    st_verify_(!store.contains(trivial_key_t {9}));

    st_verify_(succeeded(store.insert_if_missing(clashing.begin(), clashing.end())));
    st_verify_eq_(store.size(), 6);
    st_verify_eq_(mapped_or_absent(store, 3), 1);

    std::vector<member_t> revised;
    for (trivial_id_t identifier = 0; identifier != 5; ++identifier)
        revised.push_back(trivial_id_to_member<member_t>(identifier, 2));
    st_verify_(succeeded(store.update(revised.begin(), revised.end())));
    for (trivial_id_t identifier = 0; identifier != 5; ++identifier)
        st_verify_eq_(mapped_or_absent(store, identifier), 2);

    std::vector<member_t> absent;
    absent.push_back(trivial_id_to_member<member_t>(77, 3));
    st_verify_eq_(store.update(absent.begin(), absent.end()), status_t::key_not_found_k);
    st_verify_(!store.contains(trivial_key_t {77}));

    std::vector<member_t> mixed;
    mixed.push_back(trivial_id_to_member<member_t>(0, 5));
    mixed.push_back(trivial_id_to_member<member_t>(42, 5));
    st_verify_(succeeded(store.upsert(mixed.begin(), mixed.end())));
    st_verify_eq_(mapped_or_absent(store, 0), 5);
    st_verify_eq_(mapped_or_absent(store, 42), 5);

    st_verify_(succeeded(store.insert(batch.end(), batch.end())));
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
        store.select(
            ordinal,
            [&](auto const &member) noexcept {
                drawn = member.key.unique_id;
                present = true;
            },
            []() noexcept {});
        st_verify_(present);
        st_verify_eq_(drawn, expected[ordinal]);

        std::size_t ranked = 0;
        bool ranked_found = false;
        store.rank(
            trivial_key_t {drawn},
            [&](std::size_t position) noexcept {
                ranked = position;
                ranked_found = true;
            },
            []() noexcept {});
        st_verify_(ranked_found);
        st_verify_eq_(ranked, ordinal);
    }

    bool overshot = false;
    store.select(expected.size(), [](auto const &) noexcept {}, [&]() noexcept { overshot = true; });
    st_verify_(overshot);

    bool erased_ranked = false;
    store.rank(trivial_key_t {0}, [&](std::size_t) noexcept { erased_ranked = true; }, []() noexcept {});
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
    st_verify_(succeeded(store.upsert(batch.begin(), batch.end())));
    st_verify_eq_(store.size(), size);
    st_verify_eq_(store.ranked_size(), size);

    call_tally_t::reset();
    trivial_id_t drawn = 0;
    store.select(size / 2, [&](auto const &member) noexcept { drawn = member.key.unique_id; }, []() noexcept {});
    st_verify_eq_(drawn, size / 2);
    // A select reads only the stored subtree counts, so it never consults the comparator at all.
    st_verify_eq_(call_tally_t::comparisons_count(), 0);

    call_tally_t::reset();
    std::size_t ranked = 0;
    store.rank(trivial_key_t {size - 1}, [&](std::size_t position) noexcept { ranked = position; }, []() noexcept {});
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
    st_verify_(succeeded(writer->upsert(trivial_id_to_member<member_t>(5, 1))));
    st_verify_(succeeded(writer->erase(trivial_key_t {4})));

    std::vector<trivial_id_t> const expected = keys_in_range(*writer, 0, 10);
    st_verify_eq_(expected.size(), 5);

    for (std::size_t ordinal = 0; ordinal != expected.size(); ++ordinal) {
        trivial_id_t drawn = 0;
        bool present = false;
        writer->select(
            ordinal,
            [&](auto const &member) noexcept {
                drawn = member.key.unique_id;
                present = true;
            },
            []() noexcept {});
        st_verify_(present);
        st_verify_eq_(drawn, expected[ordinal]);

        std::size_t ranked = 0;
        bool ranked_found = false;
        writer->rank(
            trivial_key_t {drawn},
            [&](std::size_t position) noexcept {
                ranked = position;
                ranked_found = true;
            },
            []() noexcept {});
        st_verify_(ranked_found);
        st_verify_eq_(ranked, ordinal);
    }

    bool overshot = false;
    writer->select(expected.size(), [](auto const &) noexcept {}, [&]() noexcept { overshot = true; });
    st_verify_(overshot);

    bool erased_ranked = false;
    writer->rank(trivial_key_t {4}, [&](std::size_t) noexcept { erased_ranked = true; }, []() noexcept {});
    st_verify_(!erased_ranked);
}

/** @brief Tests that the augmented counts follow every publication path, not only the point writes */
static void test_ranked_size_tracks_every_write() {
    using member_t = typename snapshot_wb_map_t::value_type;
    snapshot_wb_map_t store;

    std::vector<member_t> batch;
    for (trivial_id_t identifier = 0; identifier != 16; ++identifier)
        batch.push_back(trivial_id_to_member<member_t>(identifier, 0));
    st_verify_(succeeded(store.insert(batch.begin(), batch.end())));
    st_verify_eq_(store.ranked_size(), store.size());

    st_verify_(succeeded(store.erase_range(trivial_key_t {4}, trivial_key_t {12})));
    st_verify_eq_(store.ranked_size(), store.size());
    st_verify_eq_(store.ranked_size(), 8);

    st_verify_(succeeded(store.update_range(trivial_key_t {0}, trivial_key_t {16},
                                            [](auto const &, int &mapped) noexcept { mapped = 7; })));
    st_verify_eq_(store.ranked_size(), store.size());

    commit_write(store, 100, 1);
    commit_erase(store, 100);
    st_verify_eq_(store.ranked_size(), store.size());

    [[maybe_unused]] std::size_t const swept = store.vacuum();
    st_verify_eq_(store.ranked_size(), store.size());

    auto abandoned = store.transaction();
    st_verify_(abandoned.has_value());
    st_verify_(succeeded(abandoned->upsert(trivial_id_to_member<member_t>(200, 1))));
    st_verify_(succeeded(abandoned->stage()));
    st_verify_eq_(store.ranked_size(), store.size());
    st_verify_(succeeded(abandoned->rollback()));
    st_verify_eq_(store.ranked_size(), store.size());
}

#pragma endregion Order Statistic Tests

int main(int, char **) {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

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

    return report_test_failures(failures);
}
