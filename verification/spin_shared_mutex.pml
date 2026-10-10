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
 *  knobs @c lock_order, @c lock_shared_order and @c unlock_order weaken the orders, for the model
 *  that includes this to show what each carries.
 *
 *  A loser parks as @c standard_waiting_policy_t parks once its spin is spent: the futex-style
 *  @c wait_on of `monitor_wait.pml` on the value its failed attempt read, woken only where the code
 *  notifies, which is every @c unlock and the @c unlock_shared that lets the last reader out. That
 *  is the one policy that can lose a wake-up: a spin hint returns on its own by the
 *  @c waiting_policy contract, and a monitored or capped park wakes on more than a notify. A writer
 *  spinning without the tally never parks, so it polls the word instead.
 *
 *  @c queued spells the slow path: a writer whose first attempt lost joins a tally of waiting
 *  writers, which turns new readers away, parks until the holders are gone, and leaves the tally
 *  in the very write that takes the lock. The unlock posts the held bit off without reading the
 *  word, as the code does, which far memory lands later unless the order is a release.
 *
 *  Include after `weak_memory.pml` and `monitor_wait.pml`. Every inline here uses the caller's
 *  @c seen as its scratch.
 */

/** The word's fields, as in the code: the held bit above a tally of waiting writers, two wide here,
 *  above a reader tally of three. */
#define writer_held 16
#define writer_waiting 4
#define readers_mask 3

/** The knobs: the header's choices, which `@verify` lines override to replay counterexamples. */
#ifndef lock_order
#define lock_order order_acquire
#endif
#ifndef lock_shared_order
#define lock_shared_order order_acquire
#endif
#ifndef unlock_order
#define unlock_order order_release
#endif
#ifndef queued
#define queued false
#endif
#ifndef last_reader_notify
#define last_reader_notify true
#endif

/** What a new reader joins below: a writer's hold, or with the tally spelled, a waiting writer. */
#define readers_below (queued -> writer_waiting : writer_held)

/** The writer's exchange from an idle word, retried once the word reads idle again:
 *  @c spin_shared_mutex::lock over @c spin_shared_mutex::take_as_writer_ in `shared.hpp`. With the
 *  tally spelled, one bounded attempt from an idle word, then the tally joined, a park on the word
 *  while a holder remains, and the tally left in the write that takes the lock. */
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
               :: else -> wait_on(t, m, seen, false)
               fi
            od
        fi
    :: else ->
        do
        :: read_modify_write_if(t, m, lock_order, seen == 0, seen, writer_held);
           if
           :: seen == 0 -> break
           :: else -> (newest_value(m) == 0)
           fi
        od
    fi
}

/** One more reader while no writer holds or waits, parked on the word between two attempts:
 *  @c spin_shared_mutex::lock_shared over @c spin_shared_mutex::take_as_reader_ in `shared.hpp`. */
inline lock_shared(t, m) {
    do
    :: read_modify_write_if(t, m, lock_shared_order, seen < readers_below, seen, seen + 1);
       if
       :: seen < readers_below -> break
       :: else -> wait_on(t, m, seen, false)
       fi
    od
}

/** The held bit posted off with a release, reading nothing back, then every parked thread woken:
 *  @c spin_shared_mutex::unlock in `shared.hpp`. The wake shares the write's step: a waiter arming
 *  between the two re-checks the word, so the gap loses nothing. */
inline unlock(t, m) {
    // Peeked rather than loaded, so the check adds no access and the code's protocol is what runs.
    atomic {
        assert((newest_value(m) & writer_held) != 0);
        add_no_return(t, m, unlock_order, -writer_held);
        wake(m)
    }
}

/** One reader fewer, with a release, and the parked threads woken by the last reader out:
 *  @c spin_shared_mutex::unlock_shared in `shared.hpp`. */
inline unlock_shared(t, m) {
    atomic {
        read_modify_write(t, m, unlock_order, seen, seen - 1);
        if
        :: last_reader_notify && (seen & readers_mask) == 1 -> wake(m)
        :: else
        fi
    }
}
