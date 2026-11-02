/**
 *  @file test/intrusive_stack.cpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief Tests for @c atomic_intrusive_stack: records pushed from several threads, alone and in
 *      chains, all reach the one thread taking them, once each.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint32_t`

#include <atomic> // `std::atomic`
#include <thread> // `std::thread`
#include <vector> // `std::vector`

#include "smashtable/atomic_intrusive_stack.hpp"

#include "harness.hpp"

namespace ashvardanian::smashtable::test {

/** A record whose fields are written before its push and read only after it was taken. */
struct pushed_record_t {
    pushed_record_t *next {nullptr};
    std::uint32_t pusher {0};
    std::uint32_t sequence {0};
};

static_assert(pointer_linked<pushed_record_t>);

/** Four pushers, half of their records alone and half in pairs, and one taker draining as they go:
 *  every record arrives once, with the fields its pusher wrote, and each pusher's in push order. */
void intrusive_stack_every_push_taken_once() {
    constexpr std::uint32_t pushers_count_k = 4;
    constexpr std::uint32_t records_per_pusher_k = 20000;
    std::vector<pushed_record_t> records(pushers_count_k * records_per_pusher_k);
    atomic_intrusive_stack<pushed_record_t> stack;
    std::atomic<std::uint32_t> pushers_done {0};

    std::vector<std::thread> pushers;
    for (std::uint32_t pusher = 0; pusher != pushers_count_k; ++pusher)
        pushers.emplace_back([&, pusher]() noexcept {
            pushed_record_t *const owned = records.data() + pusher * records_per_pusher_k;
            for (std::uint32_t sequence = 0; sequence < records_per_pusher_k;) {
                if (sequence % 4 == 0) {
                    pushed_record_t &oldest = owned[sequence];
                    pushed_record_t &newest = owned[sequence + 1];
                    oldest.pusher = pusher, oldest.sequence = sequence;
                    newest.pusher = pusher, newest.sequence = sequence + 1;
                    newest.next = &oldest;
                    stack.push_chain(newest, oldest);
                    sequence += 2;
                }
                else {
                    owned[sequence].pusher = pusher, owned[sequence].sequence = sequence;
                    stack.push(owned[sequence]);
                    sequence += 1;
                }
            }
            pushers_done.fetch_add(1, std::memory_order_release);
        });

    std::vector<std::uint32_t> times_taken(records.size(), 0);
    std::vector<std::uint32_t> next_sequence(pushers_count_k, 0);
    std::vector<pushed_record_t *> batch;
    std::size_t taken_count = 0;
    for (;;) {
        bool const finished = pushers_done.load(std::memory_order_acquire) == pushers_count_k;
        batch.clear();
        for (pushed_record_t *record = stack.take_all(); record; record = record->next) batch.push_back(record);
        // A batch is newest first, so its reverse is the order of the pushes.
        for (std::size_t position = batch.size(); position != 0; --position) {
            pushed_record_t const &record = *batch[position - 1];
            st_verify_lt_(record.pusher, pushers_count_k);
            st_verify_eq_(record.sequence, next_sequence[record.pusher], "each pusher's records arrive in order");
            ++next_sequence[record.pusher];
            ++times_taken[static_cast<std::size_t>(&record - records.data())];
        }
        taken_count += batch.size();
        if (finished && batch.empty()) break;
    }
    for (std::thread &pusher : pushers) pusher.join();

    st_verify_eq_(taken_count, records.size());
    for (std::uint32_t const taken : times_taken) st_verify_eq_(taken, 1u);
    st_verify_eq_(stack.take_all(), nullptr);
}

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

int main(int, char **arguments) {
    settings_t const settings = read_settings(arguments[0]);
    install_test_signal_handlers();
    print(probe_machine());
    print(settings);
    test_tally_t tally;

    tally += run_test(settings, "intrusive_stack.every_push_taken_once", intrusive_stack_every_push_taken_once);

    return report_test_failures(settings, tally);
}
