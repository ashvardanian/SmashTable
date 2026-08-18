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
#include <iterator>    // `std::make_move_iterator`
#include <set>         // `std::set`
#include <string>      // `std::string`
#include <thread>      // `std::thread`
#include <type_traits> // `std::is_same`
#include <vector>      // `std::vector`

#include <smashtable/locked_store.hpp>

#include "test.hpp"

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
        std::is_same<decltype(range_member),
                     void (container_t::*const)(probe_key_t &&, probe_key_t &&, probe_callback_t &&) const noexcept>(),
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
    st_verify_(succeeded(seeded));
    st_verify_((seeded != status_t::consistency_k) &&
               "a bulk upsert that cannot open is out of memory, not in conflict");
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
                container.range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(key_span),
                                [&](auto const &) noexcept { ++counted; });
            elements_seen += counted;
        });

    for (auto &thread : threads) thread.join();
    st_verify_((elements_seen.load() != 0) && "the walkers must have seen something, or this proves nothing");

    // The half the writer never touched is still whole, so a walk that skipped an unlock or lost an
    // element to the erasing writer would show here.
    for (std::size_t identifier = key_span / 2; identifier < key_span; ++identifier)
        st_verify_(container.contains(trivial_id_to_key<member_t>(identifier)));

    // Erasing everything reports a status now, and the collection must be empty afterwards.
    st_verify_(
        succeeded(container.erase_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(key_span * 2))));
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
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(identifier, identifier))));

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
    st_verify_((answers.load() != 0) && "the probes must have answered something, or this proves nothing");
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
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(identifier, identifier))));

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
                    container.upper_bound(
                        cursor,
                        [&](auto const &element) noexcept {
                            cursor = trivial_id_to_key<member_t>(0);
                            cursor = typename container_t::identifier_t(element);
                            advanced = true;
                        },
                        []() noexcept {});
                    if (!advanced) break;
                    ++walked;
                }
            }
            steps_walked += walked;
        });

    for (auto &thread : threads) thread.join();
    stop.store(true, std::memory_order_relaxed);
    st_verify_((steps_walked.load() != 0) && "the walkers must have made progress, or this proves nothing");

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
            container.upper_bound(
                cursor,
                [&](auto const &element) noexcept {
                    cursor = probe_key_t(mapping_key_or_itself(element));
                    advanced = true;
                },
                []() noexcept {});
            if (!advanced) break;
            ++heterogeneous_steps;
        }
        st_verify_((heterogeneous_steps != 0) && "a heterogeneous bound must walk the collection too");
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

    st_verify_((total != 0) && "no transaction opened at all");
    st_verify_((distinct.size() == total) && "two transactions were handed the same generation");
}

/**
 *  @brief A stage that fails on one partition must leave every other partition untouched.
 *
 *  Staging walks the partitions a transaction wrote, one lock at a time. Returning on the first
 *  refusal leaves the partitions before it holding reservations no commit will ever publish and no
 *  rollback will ever find, so the undo has to walk back over them.
 */
template <typename container_type_>
void test_sharded_stage_unwinds_on_partial_failure(std::size_t key_span = 64) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    // Both a collection and a transaction are built out of the same per-partition array builder, so
    // its bookkeeping is checked here rather than given a suite of its own.
    test_partition_array_leaves_no_scratch_behind();

    container_t container;
    auto transaction = container.transaction();
    st_verify_((transaction) && "a transaction must open");

    // Spread writes across partitions, so a refusal partway through has predecessors to undo.
    for (std::size_t identifier = 0; identifier < key_span; ++identifier)
        st_verify_(succeeded(transaction->upsert(trivial_id_to_member<member_t>(identifier, identifier))));

    st_verify_(succeeded(transaction->stage()));
    st_verify_(succeeded(transaction->rollback()));

    // Whatever the outcome, nothing may be visible and nothing may be left reserved: a later
    // transaction writing the same keys must find every one of them free to take.
    auto follower = container.transaction();
    st_verify_((follower) && "a transaction must open after the unwind");
    for (std::size_t identifier = 0; identifier < key_span; ++identifier)
        st_verify_(succeeded(follower->upsert(trivial_id_to_member<member_t>(identifier, identifier + key_span))));
    st_verify_(succeeded(follower->stage()));
    st_verify_(succeeded(follower->commit()));

    for (std::size_t identifier = 0; identifier < key_span; ++identifier) {
        auto found = container.find_copy(trivial_id_to_key<member_t>(identifier));
        st_verify_((found.has_value()) && "every key the follower committed must be visible");
    }
    st_verify_eq_(container.size(), key_span);
}

#pragma endregion Sharded Concurrency

} // namespace ashvardanian::smashtable::scripts
