/**
 *  `partitioned_store` from `include/smashtable/partitioned_store.hpp` over a shared `basic_commit_order`:
 *  a commit holds every partition it reached, ascending, asks each whether it may proceed
 *  before any writes, draws one drawn, writes every partition under it, moves the watermark
 *  once the last is written, and steps each partition's epoch with a release as it lets go.
 *
 *  Two committers over two partitions and one reader. The first committer reaches both
 *  partitions; the second reaches the second alone or both, as it likes. A reader draws its
 *  snapshot from the watermark under the order's mutex and reads each partition under its
 *  shared lock; under `-Dscenario=cursor` it reads each partition's epoch with acquire and
 *  the partition without a lock, as a cursor deciding whether to re-read does.
 *
 *  Invariants:
 *  - a commit holds every partition it reached from its ask through its write, so no committer
 *    writes a partition another has asked about and not yet written. `-Dwithout_held_partitions`
 *    gives the partitions back between the ask and the write and takes them again, ascending as
 *    before, so no deadlock masks it, and the second committer writes what the first still holds;
 *  - the watermark never passes a drawn still in flight, and never falls;
 *  - a reader naming a drawn sees every partition of that commit: whatever it reads under its
 *    snapshot carries a version at or past every commit whose drawn the snapshot covers.
 *    `-Dwithout_watermark_mutex` draws the snapshot relaxed outside the order's mutex and
 *    reads the partitions without their locks, and under views names a drawn whose versions it
 *    cannot see;
 *  - ascending locks never deadlock: `-Dwithout_ascending_order` has the second committer take
 *    the partitions descending, and Spin finds the wait cycle as an invalid end state;
 *  - a cursor that acquired an epoch reads the partition at least as new as the epoch's step
 *    left it. `-Dwithout_epoch_release` steps the epoch relaxed and the cursor reads stale.
 *
 *  The order is nine words already with two partitions, so its steps stand here as one stamp drawn and
 *  one watermark stored under a mutex, and what `begin_commit` and `end_commit` really do to the ring
 *  of done marks is `commit_order.pml`'s.
 */
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

// The knob's values are integers, so a typo fails the range check below.
#define snapshot_reader 1
#define cursor 2
#ifndef scenario
#define scenario snapshot_reader
#endif
#if scenario < snapshot_reader || scenario > cursor
#error "scenario is snapshot_reader or cursor"
#endif

// The threads, by role, apart from the processes that play them; the committers name themselves.
#define reader_thread 2
#define partitions 2

// The words: each partition's mutex, version and epoch, and the order's mutex, watermark and stamp counter.
#define mutex(partition) (partition)
#define version(partition) (2 + (partition))
#define epoch(partition) (4 + (partition))
#define order_mutex 6
#define published_stamp 7
#define commits 8

#ifdef without_epoch_release
#define epoch_order order_relaxed
#else
#define epoch_order order_release
#endif

byte committers_started;
byte finished;
int in_flight[2];              // each committer's drawn stamp until its commit is whole
int stamp_of[2];               // each committer's stamp, once drawn
bool touches[2 * partitions];  // which partitions each committer reached
bool validated[2 * partitions]; // a partition asked about and not yet written
int version_at_epoch[2 * partitions * 4]; // the version a partition held when its epoch stepped
#define at(committer, partition) ((committer) * partitions + (partition))

active [2] proctype committer() {
    byte me, partition, first, last, step;
    int seen, drawn, whole;
    atomic { me = committers_started; committers_started++ };
    touches[at(me, 1)] = true;
    if
    :: me == 0 || skip -> touches[at(me, 0)] = true
    :: skip
    fi;
    // the reached partitions, ascending: partitioned_store::touched_parts_lock_t
#ifdef without_ascending_order
    if :: me == 1 -> first = 1; last = 0 :: else -> first = 0; last = 1 fi;
#else
    first = 0; last = 1;
#endif
    if :: touches[at(me, first)] -> lock(me, mutex(first)) :: else fi;
    if :: touches[at(me, last)] -> lock(me, mutex(last)) :: else fi;
    // validate_for_commit on every reached partition before any writes: transaction_t::commit_under_one_stamp_
    for (partition : 0 .. partitions - 1) { validated[at(me, partition)] = touches[at(me, partition)] };
#ifdef without_held_partitions
    if :: touches[at(me, last)] -> unlock(me, mutex(last)) :: else fi;
    if :: touches[at(me, first)] -> unlock(me, mutex(first)) :: else fi;
    if :: touches[at(me, first)] -> lock(me, mutex(first)) :: else fi;
    if :: touches[at(me, last)] -> lock(me, mutex(last)) :: else fi;
#endif
    // begin_commit: the stamp drawn under the order's mutex, and recorded in flight
    lock(me, order_mutex);
    read_modify_write(me, commits, order_relaxed, seen, seen + 1);
    drawn = seen + 1;
    atomic { in_flight[me] = drawn; stamp_of[me] = drawn };
    unlock(me, order_mutex);
    // publish_under on every reached partition: transaction_t::commit_under_one_stamp_
    for (partition : 0 .. partitions - 1) {
        if
        :: touches[at(me, partition)] ->
            // the other committer cannot be mid-commit on this partition: this one holds it
            assert(!validated[at(1 - me, partition)]);
            store(me, version(partition), order_relaxed, drawn);
            validated[at(me, partition)] = false
        :: else
        fi
    };
    // end_commit: the watermark moves to one below the oldest in flight, or to the newest drawn
    lock(me, order_mutex);
    in_flight[me] = 0;
    if
    :: in_flight[1 - me] != 0 -> whole = in_flight[1 - me] - 1
    :: else -> whole = newest_value(commits)
    fi;
    assert(whole >= newest_value(published_stamp));
    store(me, published_stamp, order_relaxed, whole);
    unlock(me, order_mutex);
    // release: each partition's epoch stepped with a release, then its lock dropped: touched_parts_lock_t::release over note_written_
    for (partition : 0 .. partitions - 1) {
        if
        :: touches[at(me, partition)] ->
            atomic {
                read_modify_write(me, epoch(partition), epoch_order, seen, seen + 1);
                version_at_epoch[partition * 4 + seen + 1] = newest_value(version(partition))
            };
            unlock(me, mutex(partition))
        :: else
        fi
    };
    finished++
}

active proctype reader() {
    byte partition, each;
    int seen, snapshot, seen_version, seen_epoch;
#if scenario == cursor
    // a cursor: the epoch with acquire, then the partition without a lock: ordered_cursor_t::refresh_stale_fronts_ over epoch_seen_
    for (partition : 0 .. partitions - 1) {
        load(reader_thread, epoch(partition), order_acquire, seen_epoch);
        load(reader_thread, version(partition), order_relaxed, seen_version);
        assert(seen_version >= version_at_epoch[partition * 4 + seen_epoch])
    }
#else
    // take_snapshot: the watermark under the order's mutex
#ifndef without_watermark_mutex
    lock(reader_thread, order_mutex);
#endif
    load(reader_thread, published_stamp, order_relaxed, snapshot);
#ifndef without_watermark_mutex
    unlock(reader_thread, order_mutex);
#endif
    // each partition under its shared lock: a commit the snapshot covers is seen on every partition it reached
    for (partition : 0 .. partitions - 1) {
#ifndef without_watermark_mutex
        lock_shared(reader_thread, mutex(partition));
#endif
        load(reader_thread, version(partition), order_relaxed, seen_version);
        for (each : 0 .. 1) {
            if
            :: stamp_of[each] != 0 && stamp_of[each] <= snapshot && touches[at(each, partition)] ->
                assert(seen_version >= stamp_of[each])
            :: else
            fi
        };
#ifndef without_watermark_mutex
        unlock_shared(reader_thread, mutex(partition));
#endif
    }
#endif
}
