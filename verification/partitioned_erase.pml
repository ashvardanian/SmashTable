/**
 *  A window write spanning the partitions of `partitioned_store` over parts sharing a clock, as
 *  `publish_every_part_` in `include/smashtable/partitioned_store.hpp` makes one: every partition held
 *  exclusively, ascending; one stamp drawn under the clock's mutex; every partition's tombstone
 *  stamped under it; and the watermark moved once, after the last.
 *
 *  One eraser over two partitions, each holding one present key, and one reader that draws its
 *  snapshot under the clock's mutex and then reads each partition under its shared lock, which is
 *  what a pinned reader and a transaction both do.
 *
 *  Invariants:
 *  - the reader sees both keys erased or neither: its snapshot names the one stamp or does not.
 *    `-Dwithout_one_stamp` draws and publishes a stamp per partition, which is what each part's own
 *    publication does when called in turn, and the reader catches the window half erased;
 *  - the watermark never covers the stamp before every partition's tombstone carries it.
 */
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

// The threads, by role, apart from the processes that play them.
#define eraser_thread 0
#define reader_thread 1
#define partitions 2

// The words: each partition's mutex and tombstone stamp, zero while unpublished, and the clock's mutex, counter and watermark.
#define mutex(partition) (partition)
#define tombstone(partition) (2 + (partition))
#define clock_mutex 4
#define commits 5
#define published_stamp 6

active proctype eraser() {
    byte partition;
    int seen, drawn;
    // every_part_lock: both partitions, ascending, exclusively
    lock(eraser_thread, mutex(0));
    lock(eraser_thread, mutex(1));
#ifdef without_one_stamp
    for (partition : 0 .. partitions - 1) {
        lock(eraser_thread, clock_mutex);
        read_modify_write(eraser_thread, commits, order_relaxed, seen, seen + 1);
        drawn = seen + 1;
        unlock(eraser_thread, clock_mutex);
        store(eraser_thread, tombstone(partition), order_relaxed, drawn);
        lock(eraser_thread, clock_mutex);
        store(eraser_thread, published_stamp, order_relaxed, drawn);
        unlock(eraser_thread, clock_mutex)
    };
#else
    // begin_commit once, publish_under on every partition, end_commit once
    lock(eraser_thread, clock_mutex);
    read_modify_write(eraser_thread, commits, order_relaxed, seen, seen + 1);
    drawn = seen + 1;
    unlock(eraser_thread, clock_mutex);
    for (partition : 0 .. partitions - 1) { store(eraser_thread, tombstone(partition), order_relaxed, drawn) };
    lock(eraser_thread, clock_mutex);
    assert(newest_value(tombstone(0)) == drawn && newest_value(tombstone(1)) == drawn);
    store(eraser_thread, published_stamp, order_relaxed, drawn);
    unlock(eraser_thread, clock_mutex);
#endif
    unlock(eraser_thread, mutex(1));
    unlock(eraser_thread, mutex(0))
}

active proctype reader() {
    byte partition;
    int seen, snapshot, tombstone_stamp;
    bool erased[partitions];
    // take_snapshot: the watermark under the clock's mutex
    lock(reader_thread, clock_mutex);
    load(reader_thread, published_stamp, order_relaxed, snapshot);
    unlock(reader_thread, clock_mutex);
    // each partition at that snapshot, under its shared lock
    for (partition : 0 .. partitions - 1) {
        lock_shared(reader_thread, mutex(partition));
        load(reader_thread, tombstone(partition), order_relaxed, tombstone_stamp);
        erased[partition] = (tombstone_stamp != 0 && tombstone_stamp <= snapshot);
        unlock_shared(reader_thread, mutex(partition))
    };
    assert(erased[0] == erased[1])
}
