/**
 *  `locked_store` from `include/smashtable/locked_store.hpp`: every call takes the store's
 *  shared mutex exactly when it reaches the wrapped store, shared where it reads and exclusive
 *  where it writes, and a transaction's staged state lives outside the mutex.
 *
 *  One writer and two readers. The writer stages under the unique lock, writing only the
 *  store-side reservation, then commits under a second unique lock, writing the value and its
 *  stamp; each reader takes the shared lock and reads both.
 *
 *  Invariants:
 *  - exclusion: no reader is inside while the writer is, and no writer while a reader is;
 *  - a reader sees the commit whole, the value with its stamp or neither, and never the
 *    reservation as a value: staging writes only the reservation. `-Dwithout_unlock_release`
 *    and `-Dwithout_lock_acquire` each let a reader inside the lock read the value without the
 *    stamp under views.
 */
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

// The threads, by role, apart from the processes that play them; the readers name themselves.
#define writer_thread 0
#define reader_thread(reader) (1 + (reader))

// The words: the mutex, the store's reservation of the key, and the published value with its stamp.
#define mutex 0
#define reserved 1
#define value 2
#define stamp 3
#define published 7

byte readers_started;
byte inside_reading; // readers past the shared lock
bool inside_writing; // the writer past the unique lock

active proctype writer() {
    int seen;
    // stage: the reservation under the unique lock: locked_store.hpp:364-367
    lock(writer_thread, mutex);
    atomic { assert(inside_reading == 0); inside_writing = true };
    store(writer_thread, reserved, order_relaxed, 1);
    inside_writing = false;
    unlock(writer_thread, mutex);
    // commit: the value and its stamp under the unique lock: locked_store.hpp:379-382
    lock(writer_thread, mutex);
    atomic { assert(inside_reading == 0); inside_writing = true };
    store(writer_thread, value, order_relaxed, published);
    store(writer_thread, stamp, order_relaxed, published);
    store(writer_thread, reserved, order_relaxed, 0);
    inside_writing = false;
    unlock(writer_thread, mutex)
}

active [2] proctype reader() {
    byte me;
    int seen, seen_value, seen_stamp;
    atomic { me = readers_started; readers_started++ };
    // find: one shared lock around the read: locked_store.hpp:311
    lock_shared(reader_thread(me), mutex);
    atomic { assert(!inside_writing); inside_reading++ };
    load(reader_thread(me), value, order_relaxed, seen_value);
    load(reader_thread(me), stamp, order_relaxed, seen_stamp);
    assert((seen_value == published) == (seen_stamp == published));
    inside_reading--;
    unlock_shared(reader_thread(me), mutex)
}
