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
 *  Every inline uses the caller's `seen` scratch.
 */
#include "waiting_policy.pml"

#define writer_held 8

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

// lock: the writer's exchange from an idle word, retried after a wait on it: shared.hpp:2146-2166
inline lock(t, m) {
    do
    :: read_modify_write_if(t, m, lock_order, seen == 0, seen, writer_held);
       if
       :: seen == 0 -> break
       :: else -> wait_until(m, newest_value(m) == 0)
       fi
    od
}

// unlock: the held bit dropped with a release: shared.hpp:2172
inline unlock(t, m) { read_modify_write(t, m, unlock_order, seen, seen - writer_held) }

// lock_shared: one more reader while no writer holds or waits: shared.hpp:2177-2194
inline lock_shared(t, m) {
    do
    :: read_modify_write_if(t, m, lock_order, seen < writer_held, seen, seen + 1);
       if
       :: seen < writer_held -> break
       :: else -> wait_until(m, newest_value(m) < writer_held)
       fi
    od
}

// unlock_shared: one reader fewer, with a release: shared.hpp:2197
inline unlock_shared(t, m) { read_modify_write(t, m, unlock_order, seen, seen - 1) }
