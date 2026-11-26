/**
 *  @file verification/spin_shared_mutex.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c spin_shared_mutex_t from `include/smashtable/shared.hpp`, as one word:
 *      the ticket being served in the low bits, readers counted above it, queued writers on top.
 *
 *  A writer's lock is one add taking the next ticket, the serving one plus the writers queued,
 *  then a wait until the serving ticket is its own and the readers have drained; its unlock reads
 *  the serving ticket and posts the step to the next one with the writer's leave, with release. A
 *  reader's lock is one blind add, then a bounded add with acquire,
 *  @c atomic_fetch_add_if_at_most, that writes nothing when it loses. A loser parks as
 *  @c standard_waiting_policy_t parks once its spin is spent, through the futex-style @c wait_on
 *  of <tt>monitor_wait.pml</tt>, woken only where the code notifies: every @c unlock, and the
 *  @c unlock_shared that lets the last reader out.
 *
 *  - That is the one policy that can lose a wake-up.
 *  - @c blind_entry spells a reader's first attempt as one add, taken back where a queued writer or
 *    a full count turns it away, the last unit out waking the parked threads.
 *  - Readers are admitted below @c readers_limit, and the count's bits above it are the room the
 *    refused readers' units need before they carry into the queued writers.
 *  - Mutexes live at locations below eight, one bit each in the ghosts.
 *  - Include after <tt>weak_memory.pml</tt> and <tt>monitor_wait.pml</tt>.
 *  - Every inline here uses the caller's @c seen as its scratch.
 *
 *  Every scenario including this checks the writers' order beside its own assertions: a writer
 *  enters only once every writer queued ahead of it has left, an unlock finds no other writer
 *  inside, and no reader is admitted while a writer is queued. The ghosts count the threads rather
 *  than read the word, so a field carrying into another one shows.
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
#ifndef ticket_bits
#define ticket_bits 2
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

/** The word's fields, as in the code: the serving ticket at the bottom, the reader count above it,
 *  the queued writers on top, never wide enough here to wrap. */
#define serving_mask ((1 << ticket_bits) - 1)
#define one_reader (serving_mask + 1)
#define readers_field (readers_mask * one_reader)
#define one_writer ((readers_mask + 1) * one_reader)
#define mutex_bit(m) (1 << (m))

byte attempts[thread_count];    // refused attempts of the lock call in flight, kept only under a bound
byte ticket[thread_count];      // the ticket of the writer waiting, zero once it entered
byte queued_on[thread_count];   // ghost: a bit per mutex the thread queued on as a writer, until its unlock
byte waiting_for[thread_count]; // ghost: one past the mutex the thread waits on as a writer, or zero
byte ahead[thread_count];       // ghost: the writers queued ahead of it there that have not left yet
hidden int left_with;           // scratch: what a back-out read
hidden byte other;              // scratch: the ghosts' loops

/** A refused attempt counted against @c attempts_bound, which is what bounded starvation means
 *  here: a waiter retries once per wake-up owed to it, never once per arrival. */
inline count_attempt(t, admitted) {
    if
    :: attempts_bound == 0 || (admitted) -> attempts[t] = 0
    :: else -> attempts[t]++; assert(attempts[t] <= attempts_bound)
    fi
}

/** No writer queued or inside on @p m, which is what admitting a reader there claims. */
inline assert_no_writer_on(m) {
    for (other : 0 .. thread_count - 1) { assert((queued_on[other] & mutex_bit(m)) == 0) }
}

/** One more reader while the sum stays under the limit, writing nothing otherwise: the retries of
 *  @c spin_shared_mutex::lock_shared over @c spin_shared_mutex::take_as_reader_ in <tt>shared.hpp</tt>. */
inline take_as_reader(t, m) {
    atomic {
        read_modify_write_if(t, m, lock_shared_order, seen < readers_limit * one_reader, seen, seen + one_reader);
        count_attempt(t, seen < readers_limit * one_reader);
        if
        :: seen < readers_limit * one_reader -> assert_no_writer_on(m)
        :: else
        fi
    }
}

/** One unit added whatever the word holds, and taken back where a queued writer or a full count
 *  turns it away, with the parked threads woken by the last unit out:
 *  @c spin_shared_mutex::enter_as_reader_ in <tt>shared.hpp</tt>. */
inline enter_as_reader(t, m) {
    atomic {
        read_modify_write(t, m, lock_shared_order, seen, seen + one_reader);
        // A unit carried past the count reads as a queued writer over no readers at all.
        assert((seen & readers_field) != readers_field);
        count_attempt(t, seen < readers_limit * one_reader);
        if
        :: seen < readers_limit * one_reader -> assert_no_writer_on(m)
        :: else
        fi
    };
    if
    :: seen < readers_limit * one_reader -> skip
    :: else ->
        atomic {
            read_modify_write(t, m, unlock_order, left_with, left_with - one_reader);
            if
            :: back_out_notify && (left_with & readers_field) == one_reader -> wake(m)
            :: else
            fi
        }
    fi
}

/** The next ticket taken with one add, then a park on the word until the serving ticket is
 *  ours and the readers have drained: @c spin_shared_mutex::lock in <tt>shared.hpp</tt>.
 *  The ghosts count the writers already queued, as the threads rather than the word say. */
inline lock(t, m) {
    atomic {
        read_modify_write(t, m, lock_order, seen, seen + one_writer);
        ticket[t] = (seen + seen / one_writer) & serving_mask;
        seen = seen + one_writer;
        ahead[t] = 0;
        for (other : 0 .. thread_count - 1) {
            if
            :: (queued_on[other] & mutex_bit(m)) != 0 -> ahead[t]++
            :: else
            fi
        };
        queued_on[t] = queued_on[t] | mutex_bit(m);
        waiting_for[t] = (m) + 1;
        count_attempt(t, (seen & (readers_field | serving_mask)) == ticket[t])
    };
    do
    :: (seen & (readers_field | serving_mask)) == ticket[t] -> break
    :: else ->
        wait_on(t, m, seen, false);
        atomic {
            load(t, m, lock_order, seen);
            count_attempt(t, (seen & (readers_field | serving_mask)) == ticket[t])
        }
    od;
    atomic {
        assert(ahead[t] == 0);
        waiting_for[t] = 0;
        ticket[t] = 0
    }
}

/** A blind first attempt, then bounded retries parked on the word between two of them:
 *  @c spin_shared_mutex::lock_shared in <tt>shared.hpp</tt>. Nothing parks on what the blind attempt read,
 *  since the spin that follows it ends in a retry. */
inline lock_shared(t, m) {
    if
    :: blind_entry -> enter_as_reader(t, m)
    :: else -> seen = one_writer
    fi;
    do
    :: seen < readers_limit * one_reader -> break
    :: else ->
        if
        :: blind_retries -> enter_as_reader(t, m)
        :: else -> take_as_reader(t, m)
        fi;
        if
        :: seen < readers_limit * one_reader -> break
        :: else -> wait_on(t, m, seen, false)
        fi
    od
}

/** The serving ticket read relaxed, then stepped and the writer's leave posted with a release in
 *  one add, then every parked thread woken: @c spin_shared_mutex::unlock in <tt>shared.hpp</tt>.
 *  The wake shares the write's step: a waiter arming between the two re-checks the word, so the
 *  gap loses nothing. Every writer still queued here was waiting behind this one, so one fewer
 *  is ahead of each. */
inline unlock(t, m) {
    load(t, m, order_relaxed, seen);
    atomic {
        assert(waiting_for[t] == 0 && (queued_on[t] & mutex_bit(m)) != 0);
        add_no_return(t, m, unlock_order, ((seen + 1) & serving_mask) - (seen & serving_mask) - one_writer);
        queued_on[t] = queued_on[t] & ~mutex_bit(m);
        for (other : 0 .. thread_count - 1) {
            if
            :: (queued_on[other] & mutex_bit(m)) != 0 ->
                assert(waiting_for[other] == (m) + 1 && ahead[other] > 0);
                ahead[other]--
            :: else
            fi
        };
        wake(m)
    }
}

/** One reader fewer, with a release, and the parked threads woken by the last reader out:
 *  @c spin_shared_mutex::unlock_shared in `shared.hpp`. */
inline unlock_shared(t, m) {
    atomic {
        read_modify_write(t, m, unlock_order, seen, seen - one_reader);
        assert((seen & readers_field) != 0);
        if
        :: last_reader_notify && (seen & readers_field) == one_reader -> wake(m)
        :: else
        fi
    }
}
