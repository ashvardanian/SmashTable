/**
 *  `snapshot_clock_t` from `include/smashtable/snapshot_store.hpp`: the stamps commits draw,
 *  the list of commits in flight, the census of readers' leases, the watermark that names the
 *  newest whole drawn, and the low-water mark below which nothing is reachable; every word
 *  moved under one mutex, the watermark and the mark stored relaxed and read relaxed by the
 *  paths the docs allow.
 *
 *  Two committers, one reader and one pruner. A committer draws its drawn under the mutex,
 *  writes its version, and ends its commit under the mutex, moving the watermark to one below
 *  the oldest commit still in flight or to the newest stamp drawn. The reader takes a lease at
 *  the watermark under the mutex, reads every version, and retires the lease under the mutex.
 *  The pruner reads the low-water mark without the mutex and frees below it.
 *
 *  Invariants:
 *  - a lease's snapshot is below the drawn of every commit that began after it;
 *  - the low-water mark is at or below every live lease at every republish, and a pruner
 *    reading it without the mutex frees nothing a live lease still names, since the mark only
 *    rises and a stale read is a lower one;
 *  - a commit ending while an older one is still in flight moves the watermark not at all;
 *  - a reader sees every version of every commit its snapshot covers. The watermark is
 *    stored and read relaxed, and the mutex is what orders it: `-Dwithout_watermark_mutex`
 *    has the reader answer at the bare `published_stamp` accessor, outside the mutex and
 *    with no lease, which is logically fine and passes under `sequential`, and fails under
 *    views, which is the docblock's warning at 117-119 made visible.
 */
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

// The threads, by role, apart from the processes that play them; the committers name themselves.
#define reader_thread 2
#define pruner_thread 3

// The words: the clock's mutex, its stamp counter, the watermark, the low-water mark, and one version per committer.
#define clock_mutex 0
#define commits 1
#define published_stamp 2
#define low_water_mark 3
#define version(committer) (4 + (committer))

byte committers_started;
int in_flight[2]; // each committer's drawn stamp until its commit is whole
int stamp_of[2];  // each committer's stamp, once drawn
bool lease_live;  // the reader holds a lease
int lease_snapshot;

// republish_mark_: the census head's snapshot, or the watermark when nobody reads: snapshot_store.hpp:244-247
inline republish_mark(t) {
    if
    :: lease_live -> store(t, low_water_mark, order_relaxed, lease_snapshot)
    :: else -> store(t, low_water_mark, order_relaxed, newest_value(published_stamp))
    fi;
    assert(!lease_live || newest_value(low_water_mark) <= lease_snapshot)
}

active [2] proctype committer() {
    byte me;
    int seen, drawn, whole, mark_before;
    atomic { me = committers_started; committers_started++ };
    // begin_commit: snapshot_store.hpp:310-317
    lock(me, clock_mutex);
    read_modify_write(me, commits, order_relaxed, seen, seen + 1);
    drawn = seen + 1;
    assert(!lease_live || lease_snapshot < drawn);
    atomic { in_flight[me] = drawn; stamp_of[me] = drawn };
    unlock(me, clock_mutex);
    store(me, version(me), order_relaxed, drawn);
    // end_commit: snapshot_store.hpp:324-336
    lock(me, clock_mutex);
    in_flight[me] = 0;
    mark_before = newest_value(published_stamp);
    if
    :: in_flight[1 - me] != 0 ->
        whole = in_flight[1 - me] - 1;
        // an older commit still in flight pins the watermark where it was
        assert(in_flight[1 - me] > drawn || whole == mark_before)
    :: else -> whole = newest_value(commits)
    fi;
    assert(whole >= mark_before);
    store(me, published_stamp, order_relaxed, whole);
    republish_mark(me);
    unlock(me, clock_mutex)
}

active proctype reader() {
    byte each;
    int seen, snapshot, seen_version;
#ifdef without_watermark_mutex
    // the bare accessor: the watermark relaxed, outside the mutex, with no lease: snapshot_store.hpp:277
    load(reader_thread, published_stamp, order_relaxed, snapshot);
#else
    // take_snapshot: the watermark under the mutex, the lease on the census, the mark republished: snapshot_store.hpp:296-307
    lock(reader_thread, clock_mutex);
    load(reader_thread, published_stamp, order_relaxed, snapshot);
    atomic { lease_live = true; lease_snapshot = snapshot };
    republish_mark(reader_thread);
    unlock(reader_thread, clock_mutex);
#endif
    for (each : 0 .. 1) {
        load(reader_thread, version(each), order_relaxed, seen_version);
        if
        :: stamp_of[each] != 0 && stamp_of[each] <= snapshot -> assert(seen_version == stamp_of[each])
        :: else
        fi
    };
#ifndef without_watermark_mutex
    // retire_snapshot_: snapshot_store.hpp:262-265
    lock(reader_thread, clock_mutex);
    lease_live = false;
    republish_mark(reader_thread);
    unlock(reader_thread, clock_mutex)
#endif
}

// low_water_mark read without the mutex, then everything below it freed: snapshot_store.hpp:356-360
active proctype pruner() {
    int mark;
    load(pruner_thread, low_water_mark, order_relaxed, mark);
    assert(!lease_live || mark <= lease_snapshot)
}
