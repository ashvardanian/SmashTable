/**
 *  @file verification/atomic_intrusive_stack.cpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief GenMC client for @c atomic_intrusive_stack, two pushers and one taker, spelled over
 *      @c std::atomic with an explicit order at every site.
 *
 *  One pusher links two records itself and pushes them as a chain, the other pushes one record, and
 *  each writes its records' payloads and links as plain stores before the push. The taker drains
 *  the stack twice, once while the pushers run and once after joining them, and reads every payload
 *  and link it meets as plain loads, so a push that publishes too little is a race GenMC reports.
 *
 *  @verify pass sc,rc11,imm atomic_intrusive_stack
 *  @verify fail rc11,imm atomic_intrusive_stack_release_push_false: with a relaxed
 *      compare-exchange the taker meets a record whose payload and link it is not ordered after
 *  @verify fail rc11,imm atomic_intrusive_stack_acquire_take_false: with a relaxed exchange the
 *      taker meets a record whose payload and link it is not ordered after
 */
#include <atomic> // `std::atomic` - the top of the stack

#include <genmc.hpp>

/** The header's choices; each mutant entry changes one. */
struct knobs_t {
    bool release_push = true;
    bool acquire_take = true;
};

constexpr knobs_t header_k {};
constexpr knobs_t release_push_false_k {.release_push = false};
constexpr knobs_t acquire_take_false_k {.acquire_take = false};

/** A record with a link and a payload, both plain. */
struct record_t {
    record_t *next {nullptr};
    int payload {0};
};

record_t records[3];
std::atomic<record_t *> newest {nullptr};

/** Links the chain's oldest to the observed top and exchanges the top for the chain's newest.
 *  Models @c atomic_intrusive_stack::push_chain. */
template <knobs_t const &knobs_>
void push_chain(record_t &chain_newest, record_t &chain_oldest) noexcept {
    constexpr std::memory_order success_k = knobs_.release_push ? std::memory_order_release : std::memory_order_relaxed;
    record_t *observed = newest.load(std::memory_order_relaxed);
    do chain_oldest.next = observed;
    while (!newest.compare_exchange_weak(observed, &chain_newest, success_k, std::memory_order_relaxed));
}

/** Takes every record pushed so far. Models @c atomic_intrusive_stack::take_all. */
template <knobs_t const &knobs_>
record_t *take_all() noexcept {
    constexpr std::memory_order order_k = knobs_.acquire_take ? std::memory_order_acquire : std::memory_order_relaxed;
    return newest.exchange(nullptr, order_k);
}

/** Pushes the first two records as one chain, the second above the first. */
template <knobs_t const &knobs_>
void *push_pair(void *) noexcept {
    records[0].payload = 1;
    records[1].payload = 2;
    records[1].next = &records[0];
    push_chain<knobs_>(records[1], records[0]);
    return nullptr;
}

/** Pushes the third record alone. */
template <knobs_t const &knobs_>
void *push_single(void *) noexcept {
    records[2].payload = 3;
    push_chain<knobs_>(records[2], records[2]);
    return nullptr;
}

/** Walks a taken chain, checking each payload, and answers the payloads' sum. */
int sum_taken(record_t *record) noexcept {
    int sum = 0;
    for (; record; record = record->next) {
        verify(record->payload == (record - records) + 1);
        sum += record->payload;
    }
    return sum;
}

template <knobs_t const &knobs_>
int push_and_take() noexcept {
    thread_t const pair_pusher = spawn(push_pair<knobs_>);
    thread_t const single_pusher = spawn(push_single<knobs_>);
    int sum = sum_taken(take_all<knobs_>());
    join(single_pusher);
    join(pair_pusher);
    sum += sum_taken(take_all<knobs_>());
    verify(sum == 1 + 2 + 3);
    return 0;
}

extern "C" int atomic_intrusive_stack() { return push_and_take<header_k>(); }
extern "C" int atomic_intrusive_stack_release_push_false() { return push_and_take<release_push_false_k>(); }
extern "C" int atomic_intrusive_stack_acquire_take_false() { return push_and_take<acquire_take_false_k>(); }
