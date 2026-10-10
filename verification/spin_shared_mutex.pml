/**
 *  @file verification/spin_shared_mutex.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c spin_shared_mutex_t from `include/smashtable/shared.hpp`, as one word:
 *      readers counted in the low bits, a writer's hold as one high bit.
 *
 *  The tally of waiting writers is fairness and changes nothing a model asserts, so it is left out
 *  by default. A lock is a bounded add with acquire, @c atomic_fetch_add_if_at_most, that writes
 *  nothing when it loses, then a wait on the word; an unlock is a posted clear with release. The
 *  knobs @c lock_order and @c unlock_order weaken the two orders, for the model that includes this
 *  to show what each carries.
 *
 *  What a loser does between two attempts is `waiting_policy.pml`'s, chosen by @c waiting: the
 *  word alone decides who holds the mutex, so every property a model over this asserts has to hold
 *  under all four policies.
 *
 *  @c queued spells the slow path: a writer whose first attempt lost joins a tally of waiting
 *  writers, which turns new readers away, and leaves the tally in the very write that takes the
 *  lock. The unlock posts the held bit off without reading the word, as the code does, which far
 *  memory lands later unless the order is a release.
 *
 *  Every inline uses the caller's @c seen scratch.
 */
#include "waiting_policy.pml"

#define writer_held 8
#define writer_waiting 4
#define readers_mask 3

/** The knobs: the header's choices, which `@verify` lines override to replay counterexamples. */
#ifndef lock_order
#define lock_order order_acquire
#endif
#ifndef unlock_order
#define unlock_order order_release
#endif
#ifndef queued
#define queued false
#endif

/** What a new reader joins below: a writer's hold, or with the tally spelled, a waiting writer. */
#define readers_below (queued -> writer_waiting : writer_held)

/** The writer's exchange from an idle word, retried after a wait on it: @c spin_shared_mutex::lock
 *  over @c spin_shared_mutex::take_as_writer_ in `shared.hpp`. With the tally spelled, one bounded
 *  attempt from an idle word, then the tally joined and left in the write that takes the lock. */
inline lock(t, m) {
    if
    :: queued ->
        read_modify_write_if(t, m, lock_order, seen == 0, seen, writer_held);
        if
        :: seen == 0 -> skip
        :: else ->
            read_modify_write(t, m, order_relaxed, seen, seen + writer_waiting);
            do
            :: read_modify_write_if(t, m, lock_order, (seen & (writer_held | readers_mask)) == 0, seen,
                                    (seen - writer_waiting) | writer_held);
               if
               :: (seen & (writer_held | readers_mask)) == 0 -> break
               :: else -> wait_until(m, newest_value(m) != seen)
               fi
            od
        fi
    :: else ->
        do
        :: read_modify_write_if(t, m, lock_order, seen == 0, seen, writer_held);
           if
           :: seen == 0 -> break
           :: else -> wait_until(m, newest_value(m) == 0)
           fi
        od
    fi
}

/** One more reader while no writer holds or waits: @c spin_shared_mutex::lock_shared over
 *  @c spin_shared_mutex::take_as_reader_ in `shared.hpp`. */
inline lock_shared(t, m) {
    do
    :: read_modify_write_if(t, m, lock_order, seen < readers_below, seen, seen + 1);
       if
       :: seen < readers_below -> break
       :: else -> wait_until(m, newest_value(m) < readers_below)
       fi
    od
}

/** The held bit posted off with a release, reading nothing back:
 *  @c spin_shared_mutex::unlock in `shared.hpp`. */
inline unlock(t, m) {
    // Peeked rather than loaded, so the check adds no access and the code's protocol is what runs.
    atomic {
        assert((newest_value(m) & writer_held) != 0);
        add_no_return(t, m, unlock_order, -writer_held)
    }
}

/** One reader fewer, with a release: @c spin_shared_mutex::unlock_shared in `shared.hpp`. */
inline unlock_shared(t, m) { read_modify_write(t, m, unlock_order, seen, seen - 1) }
