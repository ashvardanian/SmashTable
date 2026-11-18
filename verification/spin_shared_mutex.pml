/**
 *  @file verification/spin_shared_mutex.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c spin_shared_mutex_t from `include/smashtable/shared.hpp`, as one word:
 *      readers counted in the low bits, a writer's hold as one high bit.
 *
 *  A writer's lock is one compare-exchange from an idle word, and a reader's is one blind add, then
 *  a bounded add with acquire, @c atomic_fetch_add_if_at_most, that writes nothing when it loses;
 *  both then wait on the word, and an unlock is a posted clear with release. A loser parks as
 *  @c standard_waiting_policy_t parks once its spin is spent: the futex-style @c wait_on of
 *  <tt>monitor_wait.pml</tt>, woken only where the code notifies, which is every @c unlock and the
 *  @c unlock_shared that lets the last reader out.
 *
 *  - That is the one policy that can lose a wake-up.
 *  - A writer spinning without the tally never parks, so it polls the word instead.
 *  - @c queued spells the slow path: a writer whose first attempt lost joins a tally.
 *  - @c blind_entry spells a reader's first attempt as one add, taken back where a writer bit or a
 *    full tally turns it away, the last unit out waking the parked threads.
 *  - Readers are admitted below @c readers_limit, and the tally's bits above it are the room the
 *    refused readers' units need before they carry into the waiting writers.
 *  - Include after <tt>weak_memory.pml</tt> and <tt>monitor_wait.pml</tt>.
 *  - Every inline here uses the caller's @c seen as its scratch.
 */

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
#ifndef blind_entry
#define blind_entry true
#endif
#ifndef back_out_notify
#define back_out_notify true
#endif
#ifndef blind_retries
#define blind_retries false
#endif
#ifndef readers_mask
#define readers_mask 7
#endif
#ifndef readers_limit
#define readers_limit (readers_mask >> 1)
#endif
#ifndef attempts_bound
#define attempts_bound 0
#endif

/** The word's fields, as in the code: the held bit above a tally of waiting writers, two bits wide
 *  here, above the reader tally. */
#define writer_waiting (readers_mask + 1)
#define writer_held (4 * writer_waiting)

byte attempts[thread_count]; // refused attempts of the lock call in flight, kept only under a bound
hidden int left_with;        // scratch: what a back-out read
hidden byte from_idle;       // scratch: whether the writer's exchange landed

/** A refused attempt counted against @c attempts_bound, which is what bounded starvation means
 *  here: a waiter retries once per wake-up owed to it, never once per arrival. */
inline count_attempt(t, admitted) {
    if
    :: attempts_bound == 0 || (admitted) -> attempts[t] = 0
    :: else -> attempts[t]++; assert(attempts[t] <= attempts_bound)
    fi
}

/** The writer's first attempt, one compare-exchange from an idle word:
 *  @c spin_shared_mutex::take_as_writer_ in <tt>shared.hpp</tt>. */
inline take_as_writer(t, m) {
    atomic {
        seen = 0;
        compare_exchange(t, m, lock_order, seen, writer_held, from_idle);
        count_attempt(t, seen == 0)
    }
}

/** One more reader while the sum stays under the limit, writing nothing otherwise: the retries of
 *  @c spin_shared_mutex::lock_shared over @c spin_shared_mutex::take_as_reader_ in <tt>shared.hpp</tt>. */
inline take_as_reader(t, m) {
    atomic {
        read_modify_write_if(t, m, lock_shared_order, seen < readers_limit, seen, seen + 1);
        count_attempt(t, seen < readers_limit)
    }
}

/** One unit added whatever the word holds, and taken back where a writer bit or a full tally turns
 *  it away, with the parked threads woken by the last unit out:
 *  @c spin_shared_mutex::enter_as_reader_ in <tt>shared.hpp</tt>. */
inline enter_as_reader(t, m) {
    atomic {
        read_modify_write(t, m, lock_shared_order, seen, seen + 1);
        // A unit carried past the tally reads as a waiting writer over no readers at all.
        assert((seen & readers_mask) != readers_mask);
        count_attempt(t, seen < readers_limit)
    };
    if
    :: seen < readers_limit -> skip
    :: else ->
        atomic {
            read_modify_write(t, m, unlock_order, left_with, left_with - 1);
            if
            :: back_out_notify && (left_with & readers_mask) == 1 -> wake(m)
            :: else
            fi
        }
    fi
}

/** The writer's exchange from an idle word, retried once the word reads idle again:
 *  @c spin_shared_mutex::lock over @c spin_shared_mutex::take_as_writer_ in `shared.hpp`. With the
 *  tally spelled, one bounded attempt from an idle word, then the tally joined, a park on the word
 *  while a holder remains, and the tally left in the write that takes the lock. */
inline lock(t, m) {
    if
    :: queued ->
        take_as_writer(t, m);
        if
        :: seen == 0 -> skip
        :: else ->
            read_modify_write(t, m, order_relaxed, seen, seen + writer_waiting);
            do
            :: atomic {
                   read_modify_write_if(t, m, lock_order, (seen & (writer_held | readers_mask)) == 0, seen,
                                        (seen - writer_waiting) | writer_held);
                   count_attempt(t, (seen & (writer_held | readers_mask)) == 0)
               };
               if
               :: (seen & (writer_held | readers_mask)) == 0 -> break
               :: else -> wait_on(t, m, seen, false)
               fi
            od
        fi
    :: else ->
        do
        :: take_as_writer(t, m);
           if
           :: seen == 0 -> break
           :: else -> (newest_value(m) == 0)
           fi
        od
    fi
}

/** A blind first attempt, then bounded retries parked on the word between two of them:
 *  @c spin_shared_mutex::lock_shared in <tt>shared.hpp</tt>. Nothing parks on what the blind attempt read,
 *  since the spin that follows it ends in a retry. */
inline lock_shared(t, m) {
    if
    :: blind_entry -> enter_as_reader(t, m)
    :: else -> seen = writer_held
    fi;
    do
    :: seen < readers_limit -> break
    :: else ->
        if
        :: blind_retries -> enter_as_reader(t, m)
        :: else -> take_as_reader(t, m)
        fi;
        if
        :: seen < readers_limit -> break
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
        assert((seen & readers_mask) != 0);
        if
        :: last_reader_notify && (seen & readers_mask) == 1 -> wake(m)
        :: else
        fi
    }
}
