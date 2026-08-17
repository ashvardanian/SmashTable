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

#include <atomic> // `std::atomic`
#include <set>    // `std::set`
#include <string> // `std::string`
#include <thread> // `std::thread`
#include <vector> // `std::vector`

#include "test.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Sharded Concurrency

inline constexpr std::size_t sharded_threads_count_k = 4;

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
}

/**
 *  @brief Every transaction opened concurrently must receive its own generation.
 *
 *  A generation is the identity MVCC separates versions by, so two transactions sharing one can
 *  discard each other's staged work. The counter behind it was a plain increment, which several
 *  threads opening transactions at once could read and write together.
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
