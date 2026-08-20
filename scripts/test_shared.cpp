/**
 *  @brief Tests for the primitives every container shares - the @c expected result type, the default
 *      allocator, the reader-writer lock, and the multi-store transaction group.
 *  @author Ash Vardanian
 *  @file scripts/test_shared.cpp
 *  @date August 17, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`, `SIZE_MAX`

#include <atomic>     // `std::atomic`
#include <chrono>     // `std::chrono::steady_clock`
#include <functional> // `std::equal_to`, `std::hash`, `std::less`
#include <limits>     // `std::numeric_limits`
#include <memory>     // `std::allocator`
#include <thread>     // `std::thread`
#include <utility>    // `std::pair`
#include <vector>     // `std::vector`

#include <smashtable/reference_store.hpp>
#include <smashtable/shared.hpp>
#include <smashtable/monotonic_store.hpp>

#include "test.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

namespace {

#pragma region Instrumented Value

/**
 *  @brief A value that tallies every construction and destruction, so a leak or a double free shows
 *    up as an imbalance rather than as a sanitizer report alone.
 */
struct counted_t {
    static inline std::size_t constructions = 0;
    static inline std::size_t destructions = 0;

    int payload {0};

    counted_t() noexcept { ++constructions; }
    explicit counted_t(int value) noexcept : payload(value) { ++constructions; }
    counted_t(counted_t const &other) noexcept : payload(other.payload) { ++constructions; }
    counted_t(counted_t &&other) noexcept : payload(other.payload) { ++constructions; }
    counted_t &operator=(counted_t const &other) noexcept {
        payload = other.payload;
        return *this;
    }
    counted_t &operator=(counted_t &&other) noexcept {
        payload = other.payload;
        return *this;
    }
    ~counted_t() noexcept { ++destructions; }

    static std::size_t alive() noexcept { return constructions - destructions; }
};

/** @brief A value with no copy at all, which is the shape a transaction has. */
struct move_only_t {
    int payload {0};

    move_only_t() noexcept = default;
    explicit move_only_t(int value) noexcept : payload(value) {}
    move_only_t(move_only_t &&) noexcept = default;
    move_only_t &operator=(move_only_t &&) noexcept = default;
    move_only_t(move_only_t const &) = delete;
    move_only_t &operator=(move_only_t const &) = delete;
};

#pragma endregion Instrumented Value

#pragma region Expected Tests

/** @brief Every way of constructing an @c expected, checked for a discriminant that matches storage. */
static void expected_construction_matrix() {
    std::size_t const alive_before = counted_t::alive();

    // A default-constructed result reports failure and holds nothing.
    {
        expected<counted_t> const result;
        st_verify_(!result);
        st_verify_(!result.has_value());
        st_verify_eq_(result.status(), unknown_k);
    }
    st_verify_eq_(counted_t::alive(), alive_before);

    // A reason with no value behind it constructs nothing.
    {
        std::size_t const constructions_before = counted_t::constructions;
        expected<counted_t> const result {out_of_memory_heap_k};
        st_verify_(!result);
        st_verify_eq_(result.status(), out_of_memory_heap_k);
        st_verify_eq_(counted_t::constructions, constructions_before);
    }
    st_verify_eq_(counted_t::alive(), alive_before);

    // A success handed no value cannot claim to have one, or the destructor would run over
    // storage that was never constructed.
    {
        expected<counted_t> const result {success_k};
        st_verify_(!result);
        st_verify_(!result.has_value());
        st_verify_eq_(result.status(), unknown_k);
    }
    st_verify_eq_(counted_t::alive(), alive_before);

    // A value moved into a success is there, and dies exactly once.
    {
        expected<counted_t> const result {counted_t {7}, success_k};
        st_verify_(result);
        st_verify_eq_(result->payload, 7);
        st_verify_eq_(counted_t::alive(), alive_before + 1);
    }
    st_verify_eq_(counted_t::alive(), alive_before);

    // A value handed to a failure is left to its owner, so nothing outlives the statement.
    {
        expected<counted_t> const result {counted_t {9}, out_of_memory_heap_k};
        st_verify_(!result);
        st_verify_eq_(result.status(), out_of_memory_heap_k);
        st_verify_eq_(counted_t::alive(), alive_before);
    }
    st_verify_eq_(counted_t::alive(), alive_before);

    // The same two cases through the copying constructor.
    {
        counted_t const source {11};
        expected<counted_t> const kept {source, success_k};
        st_verify_(kept);
        st_verify_eq_(kept->payload, 11);
        st_verify_eq_(counted_t::alive(), alive_before + 2);

        expected<counted_t> const dropped {source, key_not_found_k};
        st_verify_(!dropped);
        st_verify_eq_(counted_t::alive(), alive_before + 2);
    }
    st_verify_eq_(counted_t::alive(), alive_before);

    // The default status is success, so a bare value is a value.
    {
        expected<counted_t> const result {counted_t {13}};
        st_verify_(result);
        st_verify_eq_(result->payload, 13);
    }
    st_verify_eq_(counted_t::alive(), alive_before);
}

/** @brief Moves between all four combinations of full and empty, checked for balance. */
static void expected_moves_stay_balanced() {
    std::size_t const alive_before = counted_t::alive();

    // A full source, into a fresh object.
    {
        expected<counted_t> source {counted_t {1}, success_k};
        expected<counted_t> const target {std::move(source)};
        st_verify_(target);
        st_verify_eq_(target->payload, 1);
        st_verify_eq_(counted_t::alive(), alive_before + 2); // ? The moved-from value is still there
    }
    st_verify_eq_(counted_t::alive(), alive_before);

    // An empty source, into a fresh object.
    {
        expected<counted_t> source {invalid_argument_k};
        expected<counted_t> const target {std::move(source)};
        st_verify_(!target);
        st_verify_eq_(target.status(), invalid_argument_k);
        st_verify_eq_(counted_t::alive(), alive_before);
    }
    st_verify_eq_(counted_t::alive(), alive_before);

    // Full over full, full over empty, empty over full, empty over empty.
    {
        expected<counted_t> target {counted_t {2}, success_k};
        expected<counted_t> full_source {counted_t {3}, success_k};
        target = std::move(full_source);
        st_verify_(target);
        st_verify_eq_(target->payload, 3);

        expected<counted_t> empty_source {consistency_k};
        target = std::move(empty_source);
        st_verify_(!target);
        st_verify_eq_(target.status(), consistency_k);
        st_verify_eq_(counted_t::alive(), alive_before + 1); // ? Only `full_source` still holds one

        expected<counted_t> another_full {counted_t {4}, success_k};
        target = std::move(another_full);
        st_verify_(target);
        st_verify_eq_(target->payload, 4);

        expected<counted_t> another_empty {key_already_exists_k};
        target = std::move(another_empty);
        st_verify_(!target);

        // Self-assignment must not destroy anything, reached through an alias so the compiler
        // cannot mistake the deliberate case for the accidental one.
        expected<counted_t> &alias = target;
        target = std::move(alias);
        st_verify_(!target);
    }
    st_verify_eq_(counted_t::alive(), alive_before);
}

/** @brief The tuple protocol behind @c auto @c [value, status], on both paths. */
static void expected_decomposes() {
    std::size_t const alive_before = counted_t::alive();
    {
        auto [value, status] = expected<counted_t> {counted_t {5}, success_k};
        st_verify_eq_(status, success_k);
        st_verify_eq_(value.payload, 5);
    }
    {
        auto [value, status] = expected<counted_t> {out_of_memory_heap_k};
        st_verify_eq_(status, out_of_memory_heap_k);
        st_verify_eq_(value.payload, 0); // ? A failure decomposes into a default, never into garbage
    }
    st_verify_eq_(counted_t::alive(), alive_before);
}

/** @brief A move-only payload, which is the case @c std::optional could not cover. */
static void expected_carries_move_only() {
    expected<move_only_t> held {move_only_t {21}, success_k};
    st_verify_(held);
    st_verify_eq_((*held).payload, 21);

    expected<move_only_t> const taken {std::move(held)};
    st_verify_(taken);
    st_verify_eq_(taken->payload, 21);

    expected<move_only_t> const refused {operation_would_block_k};
    st_verify_(!refused);
    st_verify_eq_(refused.status(), operation_would_block_k);
}

#pragma endregion Expected Tests

#pragma region Allocator Tests

/** @brief A byte count that would wrap must be refused rather than served a tiny block. */
static void allocator_refuses_overflowing_counts() {
    default_allocator<std::uint64_t> wide_allocator;

    // Exactly the largest representable byte count, and the first count past it.
    std::size_t const largest = SIZE_MAX / sizeof(std::uint64_t);
    st_verify_eq_(wide_allocator.allocate(largest + 1), nullptr);
    st_verify_eq_(wide_allocator.allocate(SIZE_MAX), nullptr);
    st_verify_eq_(wide_allocator.allocate(SIZE_MAX / 2), nullptr);

    // A count that fits still allocates and frees.
    std::uint64_t *const block = wide_allocator.allocate(16);
    st_verify_(block != nullptr);
    block[0] = 1, block[15] = 2;
    st_verify_eq_(block[0] + block[15], 3u);
    wide_allocator.deallocate(block, 16);

    // A single-byte element has no multiplication to overflow, and must keep working.
    default_allocator<char> narrow_allocator;
    char *const bytes = narrow_allocator.allocate(32);
    st_verify_(bytes != nullptr);
    narrow_allocator.deallocate(bytes, 32);
}

#pragma endregion Allocator Tests

#pragma region Shared Mutex Tests

/** @brief Exclusion holds: a writer never overlaps a reader or another writer. */
static void shared_mutex_excludes() {
    spin_shared_mutex_t mutex;
    std::atomic<int> readers_inside {0};
    std::atomic<int> writers_inside {0};
    std::atomic<bool> violated {false};
    std::size_t guarded_left = 0, guarded_right = 0;

    constexpr std::size_t readers_count_k = 4;
    constexpr std::size_t writes_per_writer_k = 2000;
    constexpr std::size_t writers_count_k = 2;

    std::vector<std::thread> threads;
    for (std::size_t writer = 0; writer != writers_count_k; ++writer)
        threads.emplace_back([&]() noexcept {
            for (std::size_t iteration = 0; iteration != writes_per_writer_k; ++iteration) {
                unique_lock<spin_shared_mutex_t> guard {mutex};
                if (writers_inside.fetch_add(1) != 0 || readers_inside.load() != 0) violated.store(true);
                ++guarded_left;
                ++guarded_right;
                if (writers_inside.fetch_sub(1) != 1) violated.store(true);
            }
        });

    std::atomic<bool> reading {true};
    for (std::size_t reader = 0; reader != readers_count_k; ++reader)
        threads.emplace_back([&]() noexcept {
            while (reading.load(std::memory_order_relaxed)) {
                shared_lock<spin_shared_mutex_t> guard {mutex};
                readers_inside.fetch_add(1);
                if (writers_inside.load() != 0) violated.store(true);
                if (guarded_left != guarded_right) violated.store(true);
                readers_inside.fetch_sub(1);
            }
        });

    for (std::size_t writer = 0; writer != writers_count_k; ++writer) threads[writer].join();
    reading.store(false);
    for (std::size_t reader = writers_count_k; reader != threads.size(); ++reader) threads[reader].join();

    st_verify_(!violated.load());
    st_verify_eq_(guarded_left, writers_count_k * writes_per_writer_k);
    st_verify_eq_(guarded_right, guarded_left);
}

/**
 *  @brief Several writers under a steady read load all finish, which is what the waiting tally buys.
 *    One writer taking the lock must not erase the intent of the writers still queued, or a reader
 *    stream slips back in and parks them forever.
 */
static void shared_mutex_admits_every_writer() {
    spin_shared_mutex_t mutex;
    std::atomic<bool> reading {true};
    std::atomic<std::size_t> finished_writers {0};

    // The readers hold long enough, and re-enter fast enough, that their count effectively never
    // reaches zero on its own - so a writer only ever gets in by turning them away.
    constexpr std::size_t writers_count_k = 3;
    constexpr std::size_t readers_count_k = 10;
    constexpr std::size_t acquisitions_per_writer_k = 30;
    constexpr int reader_hold_spins_k = 20000;

    // The readers go first and are given a moment to saturate, so every writer arrives at a lock
    // that is already busy and has to park.
    std::vector<std::thread> readers;
    for (std::size_t reader = 0; reader != readers_count_k; ++reader)
        readers.emplace_back([&]() noexcept {
            while (reading.load(std::memory_order_relaxed)) {
                shared_lock<spin_shared_mutex_t> guard {mutex};
                for (int spin = 0; spin != reader_hold_spins_k; ++spin)
                    std::atomic_signal_fence(std::memory_order_seq_cst);
            }
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    std::vector<std::thread> writers;
    for (std::size_t writer = 0; writer != writers_count_k; ++writer)
        writers.emplace_back([&]() noexcept {
            for (std::size_t iteration = 0; iteration != acquisitions_per_writer_k; ++iteration) {
                unique_lock<spin_shared_mutex_t> guard {mutex};
                for (int spin = 0; spin != 2000; ++spin) std::atomic_signal_fence(std::memory_order_seq_cst);
            }
            finished_writers.fetch_add(1);
        });

    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (finished_writers.load() != writers_count_k && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

    st_verify_eq_(finished_writers.load(), writers_count_k);
    for (auto &writer : writers) writer.join();
    reading.store(false);
    for (auto &reader : readers) reader.join();
}

#pragma endregion Shared Mutex Tests

#pragma region Transaction Group Tests

/** @brief A store that records the order its phases are visited in, and nothing else. */
struct recording_store_t {
    using is_transactional = std::true_type;
    using identifier_t = int;
    static constexpr isolation_t isolation_k = isolation_t::monotonic_atomic_view_k;

    std::vector<int> *log {nullptr};
    int label {0};

    struct transaction_t {
        std::vector<int> *log {nullptr};
        int label {0};

        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        transaction_t(transaction_t const &) = delete;
        transaction_t &operator=(transaction_t const &) = delete;
        transaction_t(std::vector<int> *log, int label) noexcept : log(log), label(label) {}

        [[nodiscard]] status_t reserve(std::size_t) noexcept { return success_k; }
        [[nodiscard]] status_t watch(identifier_t) noexcept { return success_k; }
        [[nodiscard]] status_t stage() noexcept { return record(1); }
        [[nodiscard]] status_t commit() noexcept { return record(2); }
        [[nodiscard]] status_t rollback() noexcept { return record(3); }
        [[nodiscard]] status_t reset() noexcept { return record(4); }

      private:
        status_t record(int phase) noexcept {
            log->push_back(phase * 100 + label);
            return success_k;
        }
    };

    [[nodiscard]] expected<transaction_t> transaction() noexcept {
        return expected<transaction_t> {transaction_t {log, label}, success_k};
    }
};

/** @brief Every phase walks the participants in one address order, rollback included. */
static void transaction_group_walks_one_order() {
    std::vector<int> log;
    recording_store_t first {&log, 1}, second {&log, 2}, third {&log, 3};

    auto group_result = make_transaction_group(first, second, third);
    st_verify_(group_result);
    auto &group = *group_result;

    st_verify_eq_(group.stage(), success_k);
    std::vector<int> const staged = log;
    st_verify_eq_(staged.size(), 3u);

    st_verify_eq_(group.rollback(), success_k);
    log.clear();

    st_verify_eq_(group.stage(), success_k);
    st_verify_(log == staged);
    st_verify_eq_(group.rollback(), success_k);

    // Rollback visits the same participants in the same order as staging did.
    std::vector<int> rolled;
    for (int const phase : log)
        if (phase / 100 == 3) rolled.push_back(phase % 100);
    std::vector<int> ordered;
    for (int const phase : staged) ordered.push_back(phase % 100);
    st_verify_(rolled == ordered);

    // Commit and reset take that same order.
    log.clear();
    st_verify_eq_(group.stage(), success_k);
    st_verify_eq_(group.commit(), success_k);
    std::vector<int> committed;
    for (int const phase : log)
        if (phase / 100 == 2) committed.push_back(phase % 100);
    st_verify_(committed == ordered);

    log.clear();
    st_verify_eq_(group.reset(), success_k);
    std::vector<int> discarded;
    for (int const phase : log) discarded.push_back(phase % 100);
    st_verify_(discarded == ordered);

    // The phases refuse to run out of turn.
    st_verify_eq_(group.rollback(), operation_not_permitted_k);
    st_verify_eq_(group.commit(), operation_not_permitted_k);
}

#pragma endregion Transaction Group Tests

#pragma region Ordering Tests

using versioning_t = versioning_for<std::size_t, less_t>;
using versioned_t = versioning_t::versioned_t;
using dated_identifier_t = versioning_t::dated_identifier_t;

/** @brief A generation breaks ties only for operands that still name a key. */
static void versioned_comparator_orders_by_key_then_generation() {
    versioning_t::versioned_comparator_t comparator;

    versioned_t older {std::size_t {5}}, newer {std::size_t {5}}, other {std::size_t {6}};
    older.generation = 1, newer.generation = 2, other.generation = 0;

    st_verify_(comparator.less(older, newer));
    st_verify_(!comparator.less(newer, older));
    st_verify_(comparator.less(newer, other));
    st_verify_(!comparator.less(other, newer));
    st_verify_(comparator.same(older, older));
    st_verify_(!comparator.same(older, newer));

    // A dated identifier orders the same way, against an entry and against itself.
    dated_identifier_t const dated {std::size_t {5}, generation_t {2}};
    st_verify_(comparator.less(older, dated));
    st_verify_(!comparator.less(dated, older));
    st_verify_(comparator.same(dated, newer));

    // A plain key carries no generation, so it orders by key alone.
    st_verify_(comparator.less(std::size_t {4}, older));
    st_verify_(!comparator.less(older, std::size_t {5}));

    // A watch dates an entry without naming one, so it is never an ordering operand.
    static_assert(carries_generation<watch_t>, "a watch does carry a generation");
    static_assert(!orderable_per_version<watch_t>, "but it names no key, so it cannot be ordered");
    static_assert(orderable_per_version<versioned_t>, "an entry carries both");
    static_assert(orderable_per_version<dated_identifier_t>, "and so does a dated identifier");
}

#pragma endregion Ordering Tests

#pragma region Optimistic Concurrency Tests

/** @brief A stamp is visible to every snapshot at or past it, and an uncommitted one to none. */
static void commit_stamp_visibility_matrix() {
    constexpr generation_t stamps_k[] = {0, 1, 2, 7, 1000};
    constexpr generation_t snapshots_k[] = {0, 1, 2, 7, 1000, 1001};

    for (generation_t const stamp : stamps_k)
        for (generation_t const snapshot : snapshots_k)
            st_verify_eq_(visible_at(static_cast<commit_stamp_t>(stamp), snapshot), stamp <= snapshot);

    // The sentinel sits past every snapshot a generation counter will ever hand out, which is the
    // whole reason it is the maximum rather than zero.
    for (generation_t const snapshot : snapshots_k) st_verify_(!visible_at(commit_stamp_t::uncommitted_k, snapshot));
    st_verify_(!visible_at(commit_stamp_t::uncommitted_k, std::numeric_limits<generation_t>::max() - 1));

    // Zero as a sentinel would have read as "visible since the beginning of time".
    st_verify_(visible_at(static_cast<commit_stamp_t>(generation_t {0}), 0));
}

/** @brief Every ordered pair of isolation levels, so the enum's order is load-bearing. */
static void isolation_levels_compare_by_strength() {
    constexpr isolation_t levels_k[] = {
        isolation_t::read_committed_k,
        isolation_t::monotonic_atomic_view_k,
        isolation_t::snapshot_k,
        isolation_t::serializable_k,
    };
    constexpr std::size_t levels_count_k = sizeof(levels_k) / sizeof(levels_k[0]);

    for (std::size_t offered = 0; offered != levels_count_k; ++offered)
        for (std::size_t required = 0; required != levels_count_k; ++required)
            st_verify_eq_(at_least(levels_k[offered], levels_k[required]), offered >= required);

    static_assert(at_least(isolation_t::serializable_k, isolation_t::read_committed_k));
    static_assert(!at_least(isolation_t::read_committed_k, isolation_t::snapshot_k));
    static_assert(at_least(isolation_t::snapshot_k, isolation_t::snapshot_k));
}

/** @brief An entry standing in for what a store resolves a watched identifier to. */
struct resolved_entry_t {
    generation_t generation {0};
    presence_t presence {presence_t::present_k};

    bool operator==(watch_t const &watch) const noexcept {
        return watch.presence == presence && watch.generation == generation;
    }
    bool operator!=(watch_t const &watch) const noexcept {
        return watch.presence != presence || watch.generation != generation;
    }
};

/** @brief Watches that still match pass, and the first that drifted reports a conflict. */
static void validate_watches_catches_drift() {
    using watched_t = watched_identifier<int>;
    std::vector<watched_t> watches;
    watches.push_back(watched_t {1, watch_t {5, presence_t::present_k}});
    watches.push_back(watched_t {2, watch_t {8, presence_t::present_k}});
    watches.push_back(watched_t {3, missing_watch()});

    // The store the watches are validated against: keys 1 and 2 are there, key 3 is not.
    auto resolve = [](std::vector<std::pair<int, resolved_entry_t>> const &entries) noexcept {
        return [&entries](int id, auto &&on_found, auto &&on_missing) noexcept {
            for (auto const &entry : entries)
                if (entry.first == id) return on_found(entry.second);
            on_missing();
        };
    };

    // Nothing moved, so every watch matches itself.
    std::vector<std::pair<int, resolved_entry_t>> unchanged {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {8, presence_t::present_k}},
    };
    st_verify_eq_(validate_watches(watches, resolve(unchanged)), success_k);

    // A watched key gained a newer version.
    std::vector<std::pair<int, resolved_entry_t>> bumped {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {9, presence_t::present_k}},
    };
    st_verify_eq_(validate_watches(watches, resolve(bumped)), read_conflict_k);

    // A watched key was erased under us, which is a different shape from never having been there.
    std::vector<std::pair<int, resolved_entry_t>> erased {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {8, presence_t::erased_k}},
    };
    st_verify_eq_(validate_watches(watches, resolve(erased)), read_conflict_k);

    // A key watched as absent that has since appeared.
    std::vector<std::pair<int, resolved_entry_t>> appeared {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {8, presence_t::present_k}},
        {3, resolved_entry_t {11, presence_t::present_k}},
    };
    st_verify_eq_(validate_watches(watches, resolve(appeared)), read_conflict_k);

    // A committed tombstone resolves to missing, and the absent watch records that same shape, so
    // the two match rather than reporting a phantom conflict.
    std::vector<std::pair<int, resolved_entry_t>> tombstoned {
        {1, resolved_entry_t {5, presence_t::present_k}},
        {2, resolved_entry_t {8, presence_t::present_k}},
        {3, resolved_entry_t {absent_generation_k, presence_t::erased_k}},
    };
    st_verify_eq_(validate_watches(watches, resolve(tombstoned)), success_k);

    // A key that vanished entirely, resolved through the missing callback.
    std::vector<std::pair<int, resolved_entry_t>> vanished {{2, resolved_entry_t {8, presence_t::present_k}}};
    st_verify_eq_(validate_watches(watches, resolve(vanished)), read_conflict_k);

    // No watches at all is trivially consistent.
    std::vector<watched_t> const none;
    st_verify_eq_(validate_watches(none, resolve(unchanged)), success_k);
}

/** @brief A bare key matches every version of it, while a dated identifier matches exactly one. */
static void per_version_equals_separates_versions() {
    per_version_equals<std::equal_to<std::size_t>> const equals;
    per_key_hasher<std::hash<std::size_t>> const hasher;

    versioned_t older {std::size_t {5}}, newer {std::size_t {5}}, other {std::size_t {6}};
    older.generation = 1, newer.generation = 2, other.generation = 1;

    // Two versions of one key are distinct entries.
    st_verify_(!equals(older, newer));
    st_verify_(equals(older, older));
    st_verify_(!equals(older, other));

    // A bare key names the key alone, so it matches every version of it - the visibility walk.
    st_verify_(equals(std::size_t {5}, older));
    st_verify_(equals(std::size_t {5}, newer));
    st_verify_(!equals(std::size_t {5}, other));

    // A dated identifier names one version - the commit.
    dated_identifier_t const dated {std::size_t {5}, generation_t {2}};
    st_verify_(!equals(dated, older));
    st_verify_(equals(dated, newer));

    // The hasher keeps peeling to the bare key, so every version shares one probe run.
    st_verify_eq_(hasher(older), hasher(newer));
    st_verify_eq_(hasher(older), hasher(std::size_t {5}));
    st_verify_eq_(hasher(dated), hasher(std::size_t {5}));

    // Untouched by the widening: `per_key_equals` still collapses the versions of a key.
    per_key_equals<std::equal_to<std::size_t>> const collapsing;
    st_verify_(collapsing(older, newer));
}

// Both shipped stores are optimistically concurrent, and the seams they are read through exist.
using avl_store_t = monotonic_avl_set<std::size_t, std::less<std::size_t>, std::allocator<std::size_t>>;
using std_store_t = reference_store<std::size_t, std::less<std::size_t>, std::allocator<std::size_t>>;

static_assert(optimistically_concurrent_store<avl_store_t>, "the tree-backed store validates and stages");
static_assert(optimistically_concurrent_store<std_store_t>, "and so does the `std::set`-backed one");
static_assert(!optimistically_concurrent_store<counted_t>, "a plain value is not a store");
static_assert(at_least(avl_store_t::isolation_k, isolation_t::read_committed_k), "the floor the concept states");

#pragma endregion Optimistic Concurrency Tests

} // namespace

int main() {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "expected.construction_matrix", expected_construction_matrix);
    failures += run_test(filter, "expected.moves_stay_balanced", expected_moves_stay_balanced);
    failures += run_test(filter, "expected.decomposes", expected_decomposes);
    failures += run_test(filter, "expected.carries_move_only", expected_carries_move_only);

    failures += run_test(filter, "allocator.refuses_overflowing_counts", allocator_refuses_overflowing_counts);

    failures += run_test(filter, "shared_mutex.excludes", shared_mutex_excludes);
    failures += run_test(filter, "shared_mutex.admits_every_writer", shared_mutex_admits_every_writer);

    failures += run_test(filter, "transaction_group.walks_one_order", transaction_group_walks_one_order);

    failures += run_test(filter, "ordering.key_then_generation", versioned_comparator_orders_by_key_then_generation);

    failures += run_test(filter, "occ.commit_stamp_visibility", commit_stamp_visibility_matrix);
    failures += run_test(filter, "occ.isolation_strength", isolation_levels_compare_by_strength);
    failures += run_test(filter, "occ.validate_watches", validate_watches_catches_drift);
    failures += run_test(filter, "occ.per_version_equals", per_version_equals_separates_versions);

    return report_test_failures(failures);
}
