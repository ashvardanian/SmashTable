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
 *  One writer and two readers. The writer stages under the unique lock, writing only the store-side
 *  reservation, then commits under a second unique lock, writing the value and its stamp; each
 *  reader takes the shared lock and reads both.
 *
 *  Exclusion: no reader is inside while the writer is, and no writer while a reader is.
 *
 *  A reader sees the commit whole, the value with its stamp or neither, and never the
 *  reservation as a value: staging writes only the reservation.
 *
 *  This is the smallest model over the shared mutex, so the waiting policies and the tally of
 *  waiting writers run here; the models that layer a store protocol on the same mutex run under
 *  the default, since a policy admitting more interleavings for the lock admits them for
 *  everything above it.
 *
 *  @verify pass sc,rc11
 *  @verify pass rc11 waiting=pausing
 *  @verify pass rc11 waiting=on_the_address
 *  @verify pass rc11 waiting=parking
 *  @verify pass rc11 queued=true
 *  @verify fail rc11 unlock_order=order_relaxed: the unlock's release carries the commit's value
 *      and stamp to the next holder; relaxed, a reader inside its shared lock reads the value
 *      without the stamp
 *  @verify fail rc11 lock_order=order_relaxed: the lock's acquire takes in what the last holder
 *      released; relaxed, a reader inside its shared lock reads the value without the stamp
 *  @verify fail rc11 queued=true unlock_order=order_relaxed: a writer that took the lock through
 *      the tally still hands over through the unlock's release; relaxed, a reader inside its
 *      shared lock reads the value without the stamp
 */
#define thread_count 3
#define location_count 4
#define history_depth 11
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

/** The words: the mutex, the store's reservation of the key, and the published value
 *  with its stamp. */
#define mutex 0
#define reserved 1
#define value 2
#define value_stamp 3
#define published 7

byte inside_reading; // readers past the shared lock
bool inside_writing; // the writer past the unique lock

/** The writer: stages the reservation, then commits the value and its stamp, each under
 *  the unique lock. */
proctype writer(byte t) {
    int seen;
    // Stage: the reservation under the unique lock: `locked_store::transaction_t::stage`
    lock(t, mutex);
    atomic { assert(inside_reading == 0); inside_writing = true };
    store(t, reserved, order_relaxed, 1);
    inside_writing = false;
    unlock(t, mutex);
    // Commit: the value and its stamp under the unique lock: `locked_store::transaction_t::commit`
    lock(t, mutex);
    atomic { assert(inside_reading == 0); inside_writing = true };
    store(t, value, order_relaxed, published);
    store(t, value_stamp, order_relaxed, published);
    store(t, reserved, order_relaxed, 0);
    inside_writing = false;
    unlock(t, mutex)
}

/** A reader: one shared lock around the read of the value and its stamp. */
proctype reader(byte t) {
    int seen, seen_value, seen_stamp;
    // Find: one shared lock around the read: `locked_store::find`
    lock_shared(t, mutex);
    atomic { assert(!inside_writing); inside_reading++ };
    load(t, value, order_relaxed, seen_value);
    load(t, value_stamp, order_relaxed, seen_stamp);
    assert((seen_value == published) == (seen_stamp == published));
    inside_reading--;
    unlock_shared(t, mutex)
}

init { atomic { run writer(0); run reader(1); run reader(2) } }
