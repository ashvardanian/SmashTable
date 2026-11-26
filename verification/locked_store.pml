/**
 *  @file verification/locked_store.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief @c locked_store: one shared mutex per call, and a reader inside it seeing a commit whole.
 *
 *  Every call in <tt>include/smashtable/locked_store.hpp</tt> takes the store's shared mutex
 *  exactly when it reaches the wrapped store, shared where it reads and exclusive where it
 *  writes. Two writers and two readers. The first writer stages under the unique lock,
 *  writing only the store-side reservation, then commits under a second unique lock; the
 *  second commits straight away. This is the smallest model over the shared mutex, so the
 *  writers' tickets and the lost wake-ups run here.
 *
 *  - Exclusion: no reader is inside while a writer is, and no writer while anyone else is.
 *  - A reader sees a commit whole, the value with its stamp, and never the reservation as a value.
 *  - No commit is lost: once both writers are done, the value counts both commits.
 *
 *  @budget 6144
 *  @verify pass sc
 *  @verify fail rc11 unlock_order=order_relaxed: relaxed, the next writer reads the value from
 *      before that commit and writes the same next value over it, and one commit is lost
 *  @verify fail rc11 lock_shared_order=order_relaxed: relaxed, a reader inside its shared lock
 *      reads a commit's value without its stamp
 *  @verify fail rc11 lock_order=order_relaxed: relaxed, the second writer reads the value
 *      from before the first writer's commit and writes the same next value over it, and one
 *      commit is lost
 *  @verify stuck sc recheck=false: without it, the writer's unlock and wake land between the
 *      reader's refused attempt and its arming, and the reader sleeps through the handoff for good
 *  @verify stuck sc last_reader_notify=false: without the wake, a writer parked behind
 *      the readers sleeps once they are gone, since nothing else writes the word
 *  @verify pass sc,rc11 blind_entry=false: every attempt the bounded add, as the retries are, since
 *      the blind entry beside the queued writers overflows the budget under views
 *  @verify stuck sc back_out_notify=false: without the wake, a writer that parked on a
 *      refused reader's unit sleeps once the unit is gone, with the lock free
 *  @verify pass sc attempts_bound=10: no call is refused more often than the unlocks
 *      and last units out that can wake it
 *  @verify fail sc blind_retries=true attempts_bound=10: retrying blind, two readers
 *      behind a writer keep waking each other and the writer parked beside them
 *  @verify pass sc readers_limit=1: one reader fills the tally, and the other's refused
 *      unit stays inside the bits above the limit
 *  @verify fail sc readers_mask=1 readers_limit=1: admitting readers up to the whole tally, a
 *      refused reader's unit carries into the queued writers and the readers read as none
 *  @verify fail sc ticket_bits=0: one ticket for both writers, and the second one enters
 *      beside the first
 */
#define thread_count 4
#define location_count 4
#define history_depth 20
#include <weak_memory.pml>
#include <monitor_wait.pml>
#include <spin_shared_mutex.pml>

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
