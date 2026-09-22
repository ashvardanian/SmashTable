/**
 *  `spin_shared_mutex_t` from `include/smashtable/shared.hpp`, as one word: readers counted in
 *  the low bits, a writer's hold as one high bit; the tally of waiting writers is fairness and
 *  changes nothing a model asserts, so it is left out. A lock is a compare-exchange with acquire
 *  that writes nothing when it loses, then a wait on the word; an unlock is a read-modify-write
 *  with release. `-Dwithout_lock_acquire` and `-Dwithout_unlock_release` weaken the two orders,
 *  for the model that includes this to show what each carries.
 *
 *  What a loser does between two attempts is `waiting_policy.pml`'s, chosen by `-Dwaiting=`: the
 *  word alone decides who holds the mutex, so every property a model over this asserts has to
 *  hold under all four policies.
 *
 *  `-Dqueued` spells the slow path: a writer whose first attempt lost joins a tally of waiting
 *  writers, which turns new readers away, and leaves the tally in the very write that takes the
 *  lock. The unlock posts the held bit off without reading the word, as the code does, which
 *  `-Dmemory=far` lands later unless the order is a release.
 *
 *  Every inline uses the caller's `seen` scratch.
 */
#include "waiting_policy.pml"

#define writer_held 8
#define writer_waiting 4
#define readers_mask 3

#ifdef without_lock_acquire
#define lock_order order_relaxed
#else
#define lock_order order_acquire
#endif

#ifdef without_unlock_release
#define unlock_order order_relaxed
#else
#define unlock_order order_release
#endif

#ifdef queued
// lock: one bounded attempt from an idle word, then the tally joined and left in the write that takes the lock
inline lock(t, m) {
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
}

// lock_shared: one more reader while no writer holds or waits, the tally counting as waiting
inline lock_shared(t, m) {
    do
    :: read_modify_write_if(t, m, lock_order, seen < writer_waiting, seen, seen + 1);
       if
       :: seen < writer_waiting -> break
       :: else -> wait_until(m, newest_value(m) < writer_waiting)
       fi
    od
}
#else
// lock: the writer's exchange from an idle word, retried after a wait on it: spin_shared_mutex::lock over take_as_writer_
inline lock(t, m) {
    do
    :: read_modify_write_if(t, m, lock_order, seen == 0, seen, writer_held);
       if
       :: seen == 0 -> break
       :: else -> wait_until(m, newest_value(m) == 0)
       fi
    od
}

// lock_shared: one more reader while no writer holds or waits: spin_shared_mutex::lock_shared over take_as_reader_
inline lock_shared(t, m) {
    do
    :: read_modify_write_if(t, m, lock_order, seen < writer_held, seen, seen + 1);
       if
       :: seen < writer_held -> break
       :: else -> wait_until(m, newest_value(m) < writer_held)
       fi
    od
}
#endif

// unlock: the held bit posted off with a release, reading nothing back: spin_shared_mutex::unlock
inline unlock(t, m) { add_no_return(t, m, unlock_order, -writer_held) }

// unlock_shared: one reader fewer, with a release: spin_shared_mutex::unlock_shared
inline unlock_shared(t, m) { read_modify_write(t, m, unlock_order, seen, seen - 1) }
