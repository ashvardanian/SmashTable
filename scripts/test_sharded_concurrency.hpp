/**
 *  @brief Concurrency suites for the sharded wrapper, one per defect the audit turned up.
 *  @author Ash Vardanian
 *  @file scripts/test_sharded_concurrency.hpp
 *  @date August 17, 2026
 *
 *  Each suite here exists because a specific defect was found and fixed, and would have gone on
 *  passing every other suite in the tree. They are written to be run under ThreadSanitizer, where a
 *  clean exit is the assertion - none of them can observe their own defect from a single thread.
 */
#pragma once
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`, `std::uintptr_t`

#include <atomic>      // `std::atomic`
#include <chrono>      // `std::chrono::milliseconds`
#include <iterator>    // `std::make_move_iterator`
#include <set>         // `std::set`
#include <string>      // `std::string`
#include <thread>      // `std::thread`
#include <type_traits> // `std::is_same`
#include <vector>      // `std::vector`

#include <smashtable/locked_store.hpp>

#include "test.hpp"
#include "test_consistency.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Sharded Concurrency

inline constexpr std::size_t sharded_threads_count_k = 4;

/**
 *  @brief A range walk is a read, and the wrapper must have no second overload that says otherwise.
 *
 *  Both wrappers once carried a non-const @c range differing from its const twin only in taking the
 *  lock exclusively, while handing the callback a @c value_t const & either way. A non-const handle
 *  binds the non-const overload, so every reader through one serialized - on the sharded wrapper by
 *  taking all sixteen mutexes. Naming the specialization is the assertion: a second overload makes
 *  this address ambiguous, and a non-const one makes the type below wrong.
 */
template <typename container_type_>
void test_range_walk_takes_partitions_shared() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using probe_key_t = decltype(trivial_id_to_key<member_t>(0));
    using probe_callback_t = void (*)(typename container_t::value_t const &) noexcept;

    [[maybe_unused]] auto const range_member = &container_t::template range<probe_key_t, probe_key_t, probe_callback_t>;
    static_assert(
        std::is_same<decltype(range_member), status_t (container_t::*const)(probe_key_t &&, probe_key_t &&,
                                                                            probe_callback_t &&) const noexcept>(),
        "range must be a single const overload, so a non-const handle still reads under a shared lock");
}

/**
 *  @brief Concurrent range walks crossing a writer that erases and refills whole spans.
 *
 *  @c range and @c erase_range hold every partition for the length of the walk, and each used to
 *  release them by hand - one unlock loop per call site, five of them, any of which an early return
 *  would have skipped. One guard now owns the release. ThreadSanitizer is the oracle; the returned
 *  status is checked because it was discarded before there was one.
 */
template <typename container_type_>
void test_sharded_range_walks_share_partitions(std::size_t key_span = 256, std::size_t rounds = 40) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    test_range_walk_takes_partitions_shared<container_t>();
    // The one-mutex wrapper carried the same redundant pair, so it is checked over the same store.
    if constexpr (requires { typename container_t::inner_store_t; })
        test_range_walk_takes_partitions_shared<locked_store<typename container_t::inner_store_t>>();

    container_t container;
    std::vector<member_t> batch;
    batch.reserve(key_span);
    for (std::size_t identifier = 0; identifier < key_span; ++identifier)
        batch.push_back(trivial_id_to_member<member_t>(identifier, identifier));

    // A bulk upsert opens a transaction underneath, and a refusal there is an allocation failure -
    // never the serialization conflict this once reported.
    auto const seeded = container.upsert(std::make_move_iterator(batch.begin()), std::make_move_iterator(batch.end()));
    st_verify_(seeded);
    st_verify_ne_(seeded, status_t::consistency_k, "a bulk upsert that cannot open is out of memory, not in conflict");
    st_verify_eq_(container.size(), key_span);

    std::atomic<std::size_t> elements_seen {0};
    std::vector<std::thread> threads;
    threads.reserve(sharded_threads_count_k + 1);

    // The writer empties a span and puts it back, so a walker holding the partitions shared has to
    // wait for it and it for them.
    threads.emplace_back([&]() noexcept {
        for (std::size_t round = 0; round != rounds; ++round) {
            [[maybe_unused]] status_t const cleared =
                container.erase_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(key_span / 2),
                                      [](auto const &) noexcept {});
            for (std::size_t identifier = 0; identifier < key_span / 2; ++identifier) {
                [[maybe_unused]] auto const restored =
                    container.upsert(trivial_id_to_member<member_t>(identifier, identifier));
            }
        }
    });

    for (std::size_t thread_index = 0; thread_index != sharded_threads_count_k; ++thread_index)
        threads.emplace_back([&]() noexcept {
            std::size_t counted = 0;
            for (std::size_t round = 0; round != rounds; ++round)
                st_verify_(container.range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(key_span),
                                           [&](auto const &) noexcept { ++counted; }));
            elements_seen += counted;
        });

    for (auto &thread : threads) thread.join();
    st_verify_ne_(elements_seen.load(), 0, "the walkers must have seen something, or this proves nothing");

    // The half the writer never touched is still whole, so a walk that skipped an unlock or lost an
    // element to the erasing writer would show here.
    for (std::size_t identifier = key_span / 2; identifier < key_span; ++identifier) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(identifier)), true);
    }

    // Erasing everything reports a status now, and the collection must be empty afterwards.
    st_verify_(container.erase_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(key_span * 2)));
    st_verify_eq_(container.size(), std::size_t {0});
}

/**
 *  @brief @c lower_bound is an exact probe followed by a successor probe, each locked on its own.
 *
 *  Quiescent, the two-probe shape is invisible: an exact match answers itself and a missing key
 *  answers with its successor. Under a concurrent writer it is neither atomic nor repeatable, which
 *  is what the documented warning says and what this drives under ThreadSanitizer.
 */
template <typename container_type_>
void test_sharded_lower_bound_probes_twice(std::size_t key_span = 128, std::size_t rounds = 60) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (std::size_t identifier = 0; identifier < key_span; identifier += 2)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier, identifier)));

    // Quiescent: the exact probe answers for an even key, the successor probe for an odd one.
    for (std::size_t identifier = 0; identifier + 2 < key_span; identifier += 2) {
        auto const exact = container.lower_bound_copy(trivial_id_to_key<member_t>(identifier));
        st_verify_((exact) && "an element present at the bound is its own lower bound");
        auto const successor = container.lower_bound_copy(trivial_id_to_key<member_t>(identifier + 1));
        st_verify_((successor) && "a bound between elements answers with the next one");
    }

    std::atomic<std::size_t> answers {0};
    std::vector<std::thread> threads;
    threads.reserve(sharded_threads_count_k + 1);

    threads.emplace_back([&]() noexcept {
        for (std::size_t round = 0; round != rounds; ++round)
            for (std::size_t identifier = 0; identifier < key_span; identifier += 2) {
                [[maybe_unused]] auto const erased = container.erase(trivial_id_to_key<member_t>(identifier));
                [[maybe_unused]] auto const restored =
                    container.upsert(trivial_id_to_member<member_t>(identifier, identifier));
            }
    });

    for (std::size_t thread_index = 0; thread_index != sharded_threads_count_k; ++thread_index)
        threads.emplace_back([&]() noexcept {
            std::size_t answered = 0;
            for (std::size_t round = 0; round != rounds; ++round)
                for (std::size_t identifier = 0; identifier < key_span; ++identifier)
                    if (container.lower_bound_copy(trivial_id_to_key<member_t>(identifier))) ++answered;
            answers += answered;
        });

    for (auto &thread : threads) thread.join();
    st_verify_ne_(answers.load(), 0, "the probes must have answered something, or this proves nothing");
}

/**
 *  @brief An erase-heavy writer against several walkers, on keys that own heap storage.
 *
 *  Every ordered step of a sharded collection scans all partitions for the smallest successor and
 *  then re-reads it, and that re-read once ran with no partition lock held. A trivially-copyable key
 *  hides the consequence, since the stale bytes are still readable; a key owning a buffer does not,
 *  because the walker compares a string the eraser is freeing.
 *
 *  There is nothing to assert beyond termination and a clean report - the oracle is ThreadSanitizer.
 */
template <typename container_type_>
void test_sharded_walks_never_race_erasures(std::size_t key_span = 400, std::size_t rounds = 200) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    // The range walks and the two-probe lower bound cross the same partitions under the same guard,
    // so they share this suite's registration rather than asking for one of their own.
    test_sharded_range_walks_share_partitions<container_t>();
    test_sharded_lower_bound_probes_twice<container_t>();

    container_t container;
    for (std::size_t identifier = 0; identifier < key_span; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier, identifier)));

    std::atomic<bool> stop {false};
    std::atomic<std::size_t> steps_walked {0};

    std::vector<std::thread> threads;
    threads.reserve(sharded_threads_count_k + 1);

    // The eraser churns the same span the walkers are crossing, so a key can vanish between the scan
    // that found it and the read that follows.
    threads.emplace_back([&]() noexcept {
        for (std::size_t round = 0; round != rounds && !stop.load(std::memory_order_relaxed); ++round)
            for (std::size_t identifier = 0; identifier < key_span; identifier += 2) {
                [[maybe_unused]] auto erased = container.erase(trivial_id_to_key<member_t>(identifier));
                [[maybe_unused]] auto restored =
                    container.upsert(trivial_id_to_member<member_t>(identifier, identifier));
            }
    });

    for (std::size_t thread_index = 0; thread_index != sharded_threads_count_k; ++thread_index)
        threads.emplace_back([&]() noexcept {
            std::size_t walked = 0;
            for (std::size_t round = 0; round != rounds; ++round) {
                auto cursor = trivial_id_to_key<member_t>(0);
                for (std::size_t step = 0; step != key_span; ++step) {
                    bool advanced = false;
                    st_verify_(container.upper_bound(
                        cursor,
                        [&](auto const &element) noexcept {
                            cursor = trivial_id_to_key<member_t>(0);
                            cursor = typename container_t::identifier_t(element);
                            advanced = true;
                        },
                        []() noexcept {}));
                    if (!advanced) break;
                    ++walked;
                }
            }
            steps_walked += walked;
        });

    for (auto &thread : threads) thread.join();
    stop.store(true, std::memory_order_relaxed);
    st_verify_ne_(steps_walked.load(), 0, "the walkers must have made progress, or this proves nothing");

    // The same walk driven by a comparable that is not the identifier, which is what keeps the
    // heterogeneous API reachable: the bound travels to each partition as it arrived, never narrowed
    // to `identifier_t` first.
    using comparator_t = typename container_t::comparator_t;
    if constexpr (requires { typename comparator_t::value_type; }) {
        using probe_key_t = typename comparator_t::value_type;
        std::size_t heterogeneous_steps = 0;
        probe_key_t cursor {};
        for (std::size_t step = 0; step != key_span; ++step) {
            bool advanced = false;
            st_verify_(container.upper_bound(
                cursor,
                [&](auto const &element) noexcept {
                    cursor = probe_key_t(mapping_key_or_itself(element));
                    advanced = true;
                },
                []() noexcept {}));
            if (!advanced) break;
            ++heterogeneous_steps;
        }
        st_verify_ne_(heterogeneous_steps, 0, "a heterogeneous bound must walk the collection too");
    }
}

/**
 *  @brief The scratch storage a partition array is assembled in must be aligned, and must be emptied.
 *
 *  Partitions and their transactions are placement-new'd into a local buffer and then moved into the
 *  array that gets returned. The buffer is raw storage the builder owns, so every element it built it
 *  has to destroy - a moved-from element still has a destructor - and the buffer has to be aligned for
 *  the element rather than for @c char.
 */
inline void test_partition_array_leaves_no_scratch_behind() {

    static std::size_t constructed = 0;
    static std::size_t destructed = 0;
    static std::size_t misaligned_sources = 0;

    /** @brief Counts its own lifetime events, so a skipped destructor is arithmetic. */
    struct counted_t {
        alignas(64) std::uint64_t payload = 0;

        counted_t() noexcept { ++constructed; }
        counted_t(counted_t &&other) noexcept {
            ++constructed;
            // The move source is the object sitting in the builder's scratch storage.
            if (reinterpret_cast<std::uintptr_t>(&other) % alignof(counted_t) != 0) ++misaligned_sources;
        }
        counted_t &operator=(counted_t &&) noexcept = default;
        ~counted_t() noexcept { ++destructed; }
    };

    constructed = 0;
    destructed = 0;
    misaligned_sources = 0;
    {
        auto built = generate_array_safely<counted_t, 16>(
            [](std::size_t) noexcept { return expected<counted_t> {counted_t {}}; });
        st_verify_((built) && "every element was offered, so the array must be there");
    }
    st_verify_eq_(misaligned_sources, std::size_t {0});
    st_verify_eq_(constructed, destructed);
}

/**
 *  @brief Every transaction opened concurrently must receive its own generation.
 *
 *  A generation is what tells one transaction's staged writes from another's, so two transactions
 *  sharing one can discard each other's work. The counter behind it was a plain increment, which
 *  several threads opening transactions at once could read and write together.
 */
template <typename store_type_>
void test_concurrent_transactions_get_distinct_generations(std::size_t per_thread = 500) {

    using store_t = store_type_;
    using generation_t = typename store_t::generation_t;

    auto made = store_t::make();
    st_verify_((made) && "the store must build");
    auto &store = *made;

    std::vector<std::vector<generation_t>> observed(sharded_threads_count_k);
    std::vector<std::thread> threads;
    threads.reserve(sharded_threads_count_k);

    for (std::size_t thread_index = 0; thread_index != sharded_threads_count_k; ++thread_index)
        threads.emplace_back([&, thread_index]() noexcept {
            auto &mine = observed[thread_index];
            mine.reserve(per_thread);
            for (std::size_t opened = 0; opened != per_thread; ++opened) {
                auto transaction = store.transaction();
                if (transaction) mine.push_back(transaction->generation());
            }
        });
    for (auto &thread : threads) thread.join();

    std::set<generation_t> distinct;
    std::size_t total = 0;
    for (auto const &mine : observed)
        for (generation_t generation : mine) {
            distinct.insert(generation);
            ++total;
        }

    st_verify_ne_(total, 0, "no transaction opened at all");
    st_verify_eq_(distinct.size(), total, "two transactions were handed the same generation");
}

/**
 *  @brief A stage that fails on one partition must leave every other partition untouched.
 *
 *  Staging walks the partitions a transaction wrote, one lock at a time. Returning on the first
 *  refusal leaves the partitions before it holding reservations no commit will ever publish and no
 *  rollback will ever find, so the undo has to walk back over them.
 *
 *  A watch moved from outside is what refuses, since the per-store refusal plan is spent by the
 *  store's own methods and a transaction reaches its partitions without passing through them.
 */
template <typename container_type_>
void test_sharded_stage_unwinds_on_partial_failure(std::size_t key_span = 64) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using hash_t = typename container_t::hash_t;

    // Both a collection and a transaction are built out of the same per-partition array builder, so
    // its bookkeeping is checked here rather than given a suite of its own.
    test_partition_array_leaves_no_scratch_behind();

    container_t container;

    // The stage takes the partitions ascending, so the refusal is aimed at the highest one this
    // transaction reaches and every partition below it stages before the undo walks back.
    std::size_t conflicting_identifier = 0;
    std::size_t conflicting_partition = 0;
    for (std::size_t identifier = 0; identifier < key_span; ++identifier) {
        std::size_t const partition = hash_t {}(trivial_id_to_key<member_t>(identifier)) % container_t::partitions_k;
        if (partition < conflicting_partition) continue;
        conflicting_partition = partition;
        conflicting_identifier = identifier;
    }
    st_verify_ne_(conflicting_partition, std::size_t {0},
                  "the refusal must land above the first partition, or the undo has nothing to walk");

    auto transaction = container.transaction();
    st_verify_((transaction) && "a transaction must open");
    st_verify_(transaction->watch(trivial_id_to_key<member_t>(conflicting_identifier)));

    // Spread writes across partitions, so a refusal partway through has predecessors to undo.
    for (std::size_t identifier = 0; identifier < key_span; ++identifier)
        st_verify_(transaction->upsert(trivial_id_to_member<member_t>(identifier, identifier)));

    // Publish over the watched key from outside, so the stage is refused where the watch sits.
    {
        auto interloper = container.transaction();
        st_verify_((interloper) && "the interloping transaction must open");
        st_verify_(interloper->upsert(trivial_id_to_member<member_t>(conflicting_identifier, key_span)));
        st_verify_(interloper->stage());
        st_verify_(interloper->commit());
    }

    status_t const refused = transaction->stage();
    st_verify_((failed(refused)) && "a moved watch must refuse the stage");
    st_verify_eq_(refused, status_t::read_conflict_k);

    // An undo that never ran leaves the staged prefix holding its writes, so a change set that is
    // whole again is what says every partition handed them back.
    st_verify_eq_(transaction->changes_count(), key_span,
                  "a refused stage must roll every partition it staged back into the transaction");

    // Whatever the outcome, nothing may be visible and nothing may be left reserved: a later
    // transaction writing the same keys must find every one of them free to take.
    auto follower = container.transaction();
    st_verify_((follower) && "a transaction must open after the unwind");
    for (std::size_t identifier = 0; identifier < key_span; ++identifier)
        st_verify_(follower->upsert(trivial_id_to_member<member_t>(identifier, identifier + key_span)));
    st_verify_(follower->stage());
    st_verify_(follower->commit());

    for (std::size_t identifier = 0; identifier < key_span; ++identifier) {
        auto found = container.find_copy(trivial_id_to_key<member_t>(identifier));
        st_verify_((found.has_value()) && "every key the follower committed must be visible");
    }
    st_verify_eq_(container.size(), key_span);
}

/** @brief A comparator the caller must supply, since a container cannot manufacture one. */
struct no_default_less_t {
    using is_transparent = void;
    int tag;
    no_default_less_t() = delete;
    explicit no_default_less_t(int chosen) noexcept : tag(chosen) {}
    template <typename first_type_, typename second_type_>
    bool operator()(first_type_ const &first, second_type_ const &second) const noexcept {
        return first < second;
    }
};

/**
 *  @brief The locked wrapper forwards construction, erase and conditional insert.
 *
 *  A comparator with no default constructor is the case that forced the forwarding form: a wrapper that
 *  manufactured its own would have to default-construct one, and the Python binding's comparator holds a
 *  function pointer that must come from the caller.
 */
inline void test_locked_store_forwards_construction_and_writes() {

    using entry_t = mapping<std::uint64_t, std::uint64_t>;
    using inner_t = monotonic_avl_map<std::uint64_t, std::uint64_t, no_default_less_t, std::allocator<entry_t>>;
    using guarded_t = locked_store<inner_t>;

    auto made = guarded_t::make(no_default_less_t {7}, std::allocator<entry_t> {});
    st_verify_((made) && "a comparator without a default constructor must still reach the inner store");
    auto &store = *made;
    st_verify_(store.upsert({1, 100}));

    bool inserted = false, existing = false;
    st_verify_(store.insert_if_missing(
        {1, 999}, [&](entry_t const &) noexcept { inserted = true; },
        [&](entry_t const &) noexcept { existing = true; }));
    st_verify_((!inserted && existing) && "an occupied key must report the incumbent, not a fresh insert");

    bool found = false, missing = false;
    st_verify_(store.erase(
        std::uint64_t {1}, [&](entry_t const &) noexcept { found = true; }, [&]() noexcept { missing = true; }));
    st_verify_((found && !missing) && "erase must report presence without a second probe");
    st_verify_eq_(store.size(), 0u);
}

/**
 *  @brief Whether a reader crossing a commit that spans partitions sees all of it, or may see half.
 *
 *  A commit takes and releases one partition lock at a time, so its writes land one partition after
 *  another and a reader opening in the middle of the walk can catch half of them. What denies that is
 *  the stamp: the reader fixes one snapshot for every partition, the commit draws one stamp for every
 *  partition, and the watermark only moves once the last partition has been written. A part that
 *  keeps no stamp leaves a snapshot drawn per partition instead, which caps the shard set at
 *  @c read_committed_k and shows up here within a few rounds as two rounds read at once.
 */
template <typename container_type_>
void test_commit_spans_partitions_matches_isolation(std::size_t keys_count = 16, std::size_t rounds = 300) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (std::size_t identifier = 0; identifier != keys_count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier, 0)));

    std::atomic<bool> writing {true};
    std::atomic<std::size_t> torn_across_keys {0};
    std::atomic<std::size_t> unrepeatable_reads {0};
    std::atomic<std::size_t> reads_taken {0};
    std::atomic<std::size_t> refused_reads {0};
    /** @brief Holds the writer until every reader is inside its loop, so the two provably overlap. */
    std::atomic<std::size_t> readers_ready {0};
    std::vector<std::thread> threads;
    threads.reserve(sharded_threads_count_k + 1);

    // Every round writes the same value to every key, so any two keys disagreeing is a commit read
    // half applied - the one thing a snapshot spanning the partitions has to make impossible.
    threads.emplace_back([&]() noexcept {
        while (readers_ready.load() != sharded_threads_count_k) std::this_thread::yield();
        for (std::size_t round = 1; round <= rounds; ++round) {
            auto writer = container.transaction();
            if (!writer) break;
            status_t staged = success_k;
            for (std::size_t identifier = 0; identifier != keys_count && succeeded(staged); ++identifier)
                staged = writer->upsert(trivial_id_to_member<member_t>(identifier, round));
            if (succeeded(staged)) staged = writer->stage();
            if (succeeded(staged)) { [[maybe_unused]] status_t const committed = writer->commit(); }
        }
        writing.store(false);
    });

    for (std::size_t thread_index = 0; thread_index != sharded_threads_count_k; ++thread_index)
        threads.emplace_back([&]() noexcept {
            std::size_t mine_torn = 0;
            std::size_t mine_unrepeatable = 0;
            std::size_t mine_taken = 0;
            std::size_t mine_refused = 0;

            auto read_every_key = [&](auto &reader, std::size_t &agreed_value) noexcept {
                std::size_t disagreements = 0;
                agreed_value = 0;
                for (std::size_t identifier = 0; identifier != keys_count; ++identifier) {
                    std::size_t observed = 0;
                    status_t const read = reader.find(
                        trivial_id_to_key<member_t>(identifier),
                        [&](auto const &member) noexcept { observed = static_cast<std::size_t>(member.mapped); },
                        [&]() noexcept { ++disagreements; });
                    if (failed(read)) ++mine_refused;
                    if (identifier == 0) agreed_value = observed;
                    else if (observed != agreed_value) ++disagreements;
                }
                return disagreements;
            };

            ++readers_ready;
            while (writing.load()) {
                auto reader = container.transaction();
                if (!reader) continue;
                std::size_t first_pass_value = 0;
                std::size_t second_pass_value = 0;
                if (read_every_key(*reader, first_pass_value) != 0) ++mine_torn;
                if (read_every_key(*reader, second_pass_value) != 0) ++mine_torn;
                if (first_pass_value != second_pass_value) ++mine_unrepeatable;
                ++mine_taken;
            }

            torn_across_keys += mine_torn;
            unrepeatable_reads += mine_unrepeatable;
            reads_taken += mine_taken;
            refused_reads += mine_refused;
        });

    for (auto &thread : threads) thread.join();

    // The gate above starts the writer only once every reader is looping, so this can only trip if a
    // reader gave up before taking a single pass.
    st_verify_ne_(reads_taken.load(), 0, "no reader ever crossed the writer, so this proves nothing");
    st_verify_eq_(refused_reads.load(), 0, "a point read inside an open transaction must not fail");

    if constexpr (at_least(container_t::isolation_k, whole_commits_from_k))
        st_verify_eq_(torn_across_keys.load(), 0, "a reader saw one commit half applied across partitions");
    else st_verify_ne_(torn_across_keys.load(), 0, "a level that licenses a torn walk never produced one");

    // Repeating the walk inside one transaction is a separate promise, which only a snapshot makes.
    if constexpr (at_least(container_t::isolation_k, repeatable_reads_from_k))
        st_verify_eq_(unrepeatable_reads.load(), 0, "one transaction read two different commits");
}

/**
 *  @brief A commit that refuses must have published nothing, whichever partition refused.
 *
 *  A refusal is the caller's signal to retry, so a commit that refuses after publishing part of
 *  itself makes the retry apply that part twice. Several writers over one key set conflict on nearly
 *  every round here, which is what makes the refusals frequent enough to check.
 */
template <typename container_type_>
void test_refused_commit_publishes_nothing(std::size_t keys_count = 128, std::size_t rounds = 60) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (std::size_t identifier = 0; identifier != keys_count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier, 0)));

    std::atomic<std::size_t> commit_refusals {0};
    std::atomic<std::size_t> published_refusals {0};
    std::vector<std::thread> threads;
    threads.reserve(sharded_threads_count_k);

    // Counts one refusal and looks for the marker it staged, which must be nowhere in the container.
    auto note_refusal = [&](std::size_t marker) noexcept {
        ++commit_refusals;
        bool published = false;
        for (std::size_t identifier = 0; identifier != keys_count && !published; ++identifier)
            st_verify_(container.find(
                trivial_id_to_key<member_t>(identifier),
                [&](auto const &member) noexcept {
                    published = published || static_cast<std::size_t>(member.mapped) == marker;
                },
                []() noexcept {}));
        if (published) ++published_refusals;
    };

    // Two writers staged over one key and committed in order: first-committer-wins refuses the
    // second, so the suite proves its point on every schedule rather than on a lucky interleaving.
    // The racing threads below widen the interleavings it proves it over.
    {
        auto first = container.transaction();
        auto second = container.transaction();
        st_verify_(first);
        st_verify_(second);
        st_verify_(first->upsert(trivial_id_to_member<member_t>(0, 1)));
        st_verify_(second->upsert(trivial_id_to_member<member_t>(0, 2)));
        st_verify_(first->stage());
        st_verify_(second->stage());
        st_verify_(first->commit());
        st_verify_ne_(second->commit(), success_k, "a second writer over one key must be refused");
        note_refusal(2);
        [[maybe_unused]] status_t const undone = second->reset();
    }

    // Every attempt writes a value no other attempt writes, so finding one after a refusal is proof
    // that the refused commit published something.
    for (std::size_t thread_index = 0; thread_index != sharded_threads_count_k; ++thread_index)
        threads.emplace_back([&, thread_index]() noexcept {
            std::size_t attempt = 0;
            for (std::size_t round = 0; round != rounds; ++round)
                while (true) {
                    std::size_t const marker = (thread_index + 1) * 1000000 + (++attempt);
                    auto writer = container.transaction();
                    if (!writer) return;
                    status_t staged = success_k;
                    for (std::size_t identifier = 0; identifier != keys_count && succeeded(staged); ++identifier)
                        staged = writer->upsert(trivial_id_to_member<member_t>(identifier, marker));
                    if (succeeded(staged)) staged = writer->stage();
                    if (failed(staged)) {
                        [[maybe_unused]] status_t const undone = writer->reset();
                        continue;
                    }
                    if (succeeded(writer->commit())) break;

                    note_refusal(marker);
                    [[maybe_unused]] status_t const undone = writer->reset();
                }
        });

    for (auto &thread : threads) thread.join();
    st_verify_eq_(published_refusals.load(), 0, "a commit that refused had already published its writes");
    // The staged pair above guarantees one, so this can only trip if a refusal stopped being counted.
    st_verify_ne_(commit_refusals.load(), 0, "no commit ever refused, so nothing here was exercised");
}

/**
 *  @brief An enumeration must see every element that was there for the whole of it, exactly once.
 *
 *  The sharded enumeration takes one partition at a time rather than all of them, so a writer runs
 *  alongside it and the walk is not a snapshot. What it still owes the caller is that nothing which
 *  stayed put is missed or counted twice - a key's partition is fixed by its hash, so it can neither
 *  be moved ahead of the cursor nor behind it. The churned keys carry the other half of the promise:
 *  they may or may not turn up, and either answer is allowed, but never twice in one walk.
 */
template <typename container_type_>
void test_sharded_enumeration_sees_every_stable_element(std::size_t stable_count = 128, std::size_t churn_count = 128,
                                                        std::size_t rounds = 40) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (std::size_t identifier = 0; identifier != stable_count; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    std::atomic<bool> churning {true};
    std::atomic<bool> churn_started {false};
    std::atomic<std::size_t> walks_taken {0};
    std::atomic<std::size_t> stable_misses {0};
    std::atomic<std::size_t> repeat_visits {0};

    std::thread writer([&]() noexcept {
        churn_started.store(true);
        for (std::size_t round = 0; round != rounds; ++round) {
            for (std::size_t offset = 0; offset != churn_count; ++offset) [[maybe_unused]]
                status_t const written = container.upsert(trivial_id_to_member<member_t>(stable_count + offset));
            for (std::size_t offset = 0; offset != churn_count; ++offset) [[maybe_unused]]
                status_t const removed = container.erase(trivial_id_to_key<member_t>(stable_count + offset));
        }
        churning.store(false);
    });

    std::vector<std::thread> walkers;
    walkers.reserve(sharded_threads_count_k);
    for (std::size_t thread_index = 0; thread_index != sharded_threads_count_k; ++thread_index)
        walkers.emplace_back([&]() noexcept {
            std::vector<std::size_t> visits(stable_count + churn_count, 0);
            while (!churn_started.load()) {}
            do {
                for (std::size_t &seen : visits) seen = 0;
                st_verify_(container.for_each([&](member_t const &member) noexcept {
                    std::size_t const identifier = static_cast<std::size_t>(member.unique_id);
                    if (identifier < visits.size()) ++visits[identifier];
                }));
                for (std::size_t identifier = 0; identifier != stable_count; ++identifier)
                    if (visits[identifier] != 1) ++stable_misses;
                for (std::size_t identifier = stable_count; identifier != visits.size(); ++identifier)
                    if (visits[identifier] > 1) ++repeat_visits;
                ++walks_taken;
            } while (churning.load());
        });

    writer.join();
    for (auto &walker : walkers) walker.join();

    st_verify_ge_(walks_taken.load(), sharded_threads_count_k,
                  "every walker must complete a walk begun after the writer started, or this proves nothing");
    st_verify_eq_(stable_misses.load(), 0, "an element present for the whole walk was missed or seen twice");
    st_verify_eq_(repeat_visits.load(), 0, "one walk handed the same key to the callback twice");
}

/**
 *  @brief Pauses one comparison so a commit can be caught between drawing its stamp and publishing it.
 *
 *  A commit's stamp holds the watermark down for every commit drawn after it, and the only injectable
 *  thing called between @c begin_commit and @c end_commit is the comparator the publish walk uses. So
 *  the gate lives here: whichever thread compares @c gated_key_k stops until it is let go, and its
 *  stamp stays in flight meanwhile.
 */
struct publication_gate_t {
    static constexpr std::int64_t gated_key_k = 777;

    static inline std::atomic<bool> armed {false};
    static inline std::atomic<bool> reached {false};
    static inline std::atomic<bool> holding {false};

    static void arm() noexcept {
        reached.store(false, std::memory_order_release);
        holding.store(true, std::memory_order_release);
        armed.store(true, std::memory_order_release);
    }
    static void release() noexcept {
        holding.store(false, std::memory_order_release);
        armed.store(false, std::memory_order_release);
    }
    static void wait_until_reached() noexcept {
        while (!reached.load(std::memory_order_acquire)) std::this_thread::yield();
    }
    static void pass(std::int64_t key) noexcept {
        if (key != gated_key_k || !armed.load(std::memory_order_acquire)) return;
        reached.store(true, std::memory_order_release);
        while (holding.load(std::memory_order_acquire)) std::this_thread::yield();
    }
};

/** @brief Orders keys as @c less_t does, and stops on the gated one so a commit can be held open. */
struct gated_less_t {
    using is_transparent = void;
    template <typename first_type_, typename second_type_>
    bool operator()(first_type_ const &first, second_type_ const &second) const noexcept {
        publication_gate_t::pass(static_cast<std::int64_t>(first));
        return first < second;
    }
};

/**
 *  @brief A transaction opening after a commit returned must see that commit.
 *
 *  Serializability alone does not promise it: a commit drawn earlier and still writing itself out holds
 *  the watermark below the stamp just published, so a transaction opening afterwards can read at a
 *  snapshot that predates a commit which has already answered its caller. The gate makes that window
 *  deliberate rather than hoped for, by stopping an earlier commit inside its own publication.
 */
template <typename container_type_>
void test_commit_is_visible_to_what_opens_after_it() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    constexpr bool waits_k = at_least(container_t::isolation_k, isolation_t::strict_serializable_k);

    auto made = container_t::make(gated_less_t {}, hash<std::int64_t> {});
    st_verify_(made);
    container_t &store = *made;

    st_verify_(store.upsert(member_t {publication_gate_t::gated_key_k, 0}));

    // The second key must land in another partition, or the later commit waits on the mutex the held
    // one owns and the test times the wrong wait.
    std::size_t const gated_partition =
        hash<std::int64_t> {}(publication_gate_t::gated_key_k) % container_t::partitions_k;
    std::int64_t later_key = 1;
    while (hash<std::int64_t> {}(later_key) % container_t::partitions_k == gated_partition) ++later_key;

    // Opened before the gate arms, because opening a transaction takes every partition lock in turn:
    // one of them belongs to the held commit, so a transaction opened during the hold would be waiting
    // on a mutex rather than on the watermark, which is not the wait under test.
    auto opened = store.transaction();
    st_verify_(opened);
    auto writer = std::move(*opened);

    // The earlier commit, stopped inside its own publication, so its stamp stays in flight and pins the
    // watermark one below itself for as long as the gate holds.
    auto holder = store.transaction();
    st_verify_(holder);
    st_verify_(holder->upsert(member_t {publication_gate_t::gated_key_k, 1}));
    st_verify_(holder->stage());

    publication_gate_t::arm();
    std::thread earlier {[&]() noexcept { [[maybe_unused]] status_t const answered = holder->commit(); }};
    publication_gate_t::wait_until_reached();

    std::atomic<bool> reached_commit {false};
    std::atomic<bool> answered_commit {false};
    std::atomic<bool> saw_its_own_commit {false};
    std::thread later {[&]() noexcept {
        [[maybe_unused]] status_t const wrote = writer.upsert(member_t {later_key, 42});
        [[maybe_unused]] status_t const staged = writer.stage();
        reached_commit.store(true, std::memory_order_release);
        [[maybe_unused]] status_t const committed = writer.commit();
        answered_commit.store(true, std::memory_order_release);

        // Opening this one is itself the question: a strict commit has already waited by the time it
        // returns, so nothing is held and the snapshot it draws covers its own write.
        auto reader = store.transaction();
        if (!reader) return;
        bool found = false;
        [[maybe_unused]] status_t const read =
            reader->find(later_key, [&](member_t const &member) noexcept { found = member.mapped == 42; }, no_op_t {});
        saw_its_own_commit.store(found, std::memory_order_release);
    }};

    while (!reached_commit.load(std::memory_order_acquire)) std::this_thread::yield();

    // Nothing below races. The gate holds an older stamp in flight, so the watermark cannot reach the
    // later one, and a strict commit has no state in which it is allowed to answer - the sleep only
    // keeps the check from passing vacuously. That the cheaper level answers sooner is a measurement,
    // not something a second thread can assert.
    if constexpr (waits_k) {
        std::this_thread::sleep_for(std::chrono::milliseconds {50});
        st_verify_((!answered_commit.load(std::memory_order_acquire)) &&
                   "a strict commit cannot answer while an earlier commit still pins the watermark");
    }

    publication_gate_t::release();
    later.join();
    earlier.join();

    if constexpr (waits_k)
        st_verify_(saw_its_own_commit.load(std::memory_order_acquire) &&
                   "a transaction opened after a strict commit returned must see it");

    // Both writes land at either level; only when they become visible differs.
    bool gated_is_current = false;
    bool later_is_current = false;
    st_verify_(store.find(
        publication_gate_t::gated_key_k,
        [&](member_t const &member) noexcept { gated_is_current = member.mapped == 1; }, no_op_t {}));
    st_verify_(store.find(
        later_key, [&](member_t const &member) noexcept { later_is_current = member.mapped == 42; }, no_op_t {}));
    st_verify_(gated_is_current);
    st_verify_(later_is_current);
}

#pragma endregion Sharded Concurrency

} // namespace ashvardanian::smashtable::scripts
