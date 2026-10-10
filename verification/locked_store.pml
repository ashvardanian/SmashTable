/**
 *  @file verification/locked_store.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief @c locked_store: one shared mutex per call, and a reader inside it seeing a commit whole.
 *
 *  Every call in `include/smashtable/locked_store.hpp` takes the store's shared mutex exactly when
 *  it reaches the wrapped store, shared where it reads and exclusive where it writes, and a
 *  transaction's staged state lives outside the mutex.
 *
 *  Two writers and two readers. The first writer stages under the unique lock, writing only the
 *  store-side reservation, then commits under a second unique lock; the second commits straight
 *  away. A commit reads the value and writes the next one with its stamp; each reader takes the
 *  shared lock and reads both.
 *
 *  Exclusion: no reader is inside while a writer is, and no writer while anyone else is.
 *
 *  A reader sees a commit whole, the value with its stamp, and never the reservation as a value:
 *  staging writes only the reservation. No commit is lost: once both writers are done, the value
 *  counts both commits.
 *
 *  This is the smallest model over the shared mutex, so the tally of waiting writers and the lost
 *  wake-ups run here; the models that layer a store protocol on the same mutex run its defaults.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc,rc11 queued=true
 *  @verify fail rc11 unlock_order=order_relaxed: the unlock's release carries the commit's value
 *      and stamp to the next holder; relaxed, the next writer reads the value from before that
 *      commit and writes the same next value over it, and one commit is lost
 *  @verify fail rc11 lock_shared_order=order_relaxed: the shared lock's acquire takes in what the
 *      last holder released; relaxed, a reader inside its shared lock reads a commit's value
 *      without its stamp
 *  @verify fail rc11 lock_order=order_relaxed: the unique lock's acquire takes in what the last
 *      holder released; relaxed, the second writer reads the value from before the first writer's
 *      commit and writes the same next value over it, and one commit is lost
 *  @verify fail rc11 queued=true unlock_order=order_relaxed: a writer that took the lock through
 *      the tally still hands over through the unlock's release; relaxed, the next writer reads the
 *      value from before that commit, and one commit is lost
 *  @verify stuck sc recheck=false: a parked reader re-reads the word after arming, as the futex
 *      does under its lock; without it, the writer's unlock and wake land between the reader's
 *      refused attempt and its arming, and the reader sleeps through the handoff for good
 *  @verify stuck sc queued=true last_reader_notify=false: @c unlock_shared wakes the parked
 *      writers when the last reader leaves; without the wake, a writer parked behind the readers
 *      sleeps once they are gone, since nothing else writes the word
 */
#define thread_count 4
#define location_count 4
#define history_depth 14
#include "weak_memory.pml"
#include "monitor_wait.pml"
#include "spin_shared_mutex.pml"

/** The words: the mutex, the store's reservation of the key, and the value with its stamp. */
#define mutex 0
#define reserved 1
#define value 2
#define value_stamp 3

byte inside_reading; // readers past the shared lock
byte inside_writing; // writers past the unique lock
byte committed;      // commits that returned

/** One commit under the unique lock: the next value and its stamp, and the reservation cleared:
 *  @c locked_store::transaction_t::commit. */
inline commit(t) {
    lock(t, mutex);
    atomic { assert(inside_reading == 0 && inside_writing == 0); inside_writing++ };
    load(t, value, order_relaxed, seen_value);
    store(t, value, order_relaxed, seen_value + 1);
    store(t, value_stamp, order_relaxed, seen_value + 1);
    store(t, reserved, order_relaxed, 0);
    inside_writing--;
    unlock(t, mutex);
    committed++
}

/** The first writer: stages the reservation under the unique lock, then commits:
 *  @c locked_store::transaction_t::stage. */
proctype stager(byte t) {
    int seen, seen_value;
    lock(t, mutex);
    atomic { assert(inside_reading == 0 && inside_writing == 0); inside_writing++ };
    store(t, reserved, order_relaxed, 1);
    inside_writing--;
    unlock(t, mutex);
    commit(t)
}

/** The second writer: a commit straight away. */
proctype writer(byte t) {
    int seen, seen_value;
    commit(t)
}

/** A reader: one shared lock around the read of the value and its stamp: @c locked_store::find. */
proctype reader(byte t) {
    int seen, seen_value, seen_stamp;
    lock_shared(t, mutex);
    atomic { assert(inside_writing == 0); inside_reading++ };
    load(t, value, order_relaxed, seen_value);
    load(t, value_stamp, order_relaxed, seen_stamp);
    assert(seen_value == seen_stamp);
    inside_reading--;
    unlock_shared(t, mutex)
}

/** The value counts both commits once both writers returned; it reads no word, plays no thread. */
proctype auditor() {
    (committed == 2);
    assert(newest_value(value) == 2)
}

init { atomic { run stager(0); run writer(1); run reader(2); run reader(3); run auditor() } }
