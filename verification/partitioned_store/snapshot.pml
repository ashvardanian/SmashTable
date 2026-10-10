/**
 *  @file verification/partitioned_store/snapshot.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Two commits over two partitions of @c partitioned_store and a reader whose snapshot sees
 *      every partition of each commit it names.
 *
 *  The reader draws its snapshot from the watermark under the order's mutex and reads each
 *  partition under its shared lock: whatever it reads under its snapshot carries a version at or
 *  past every commit whose stamp the snapshot covers.
 *
 *  @verify pass sc,rc11
 *  @verify stuck sc ascending_order=false: both commits take the partitions they reached ascending;
 *      with the second taking them descending, the two hold one each and wait on the other's for
 *      good, which Spin finds as an invalid end state
 *  @verify fail sc,rc11 held_partitions=false: a commit holds every partition it reached from its
 *      ask through its write; giving them back between the two and taking them again in the same
 *      ascending order, so no deadlock masks it, the second committer writes what the first still
 *      holds as asked
 *  @verify fail rc11 watermark_mutex=false: the snapshot is drawn under the order's mutex, which
 *      stands for the acq_rel read-modify-writes @c end_commit and @c take_snapshot make on the
 *      watermark, and each partition is read under its lock; drawn relaxed outside the mutex and
 *      read without the locks, the snapshot names a stamp whose versions the reader cannot see,
 *      which is the stale read either of them forbids
 */
#define thread_count 3
#define location_count 9
#define history_depth 11
#include "protocol.pml"

proctype committer(byte t) { commit(t) }

/** @c take_snapshot: the watermark under the order's mutex, then each partition under its shared
 *  lock, where a commit the snapshot covers is seen on every partition it reached. */
proctype reader(byte t) {
    byte partition, each;
    int seen, snapshot, seen_version;
    if
    :: watermark_mutex -> lock(t, order_mutex)
    :: else
    fi;
    load(t, published_stamp, order_relaxed, snapshot);
    if
    :: watermark_mutex -> unlock(t, order_mutex)
    :: else
    fi;
    for (partition : 0 .. partitions - 1) {
        if
        :: watermark_mutex -> lock_shared(t, mutex(partition))
        :: else
        fi;
        load(t, version(partition), order_relaxed, seen_version);
        for (each : 0 .. 1) {
            if
            :: stamp_of[each] != 0 && stamp_of[each] <= snapshot && touches[at(each, partition)] ->
                assert(seen_version >= stamp_of[each])
            :: else
            fi
        };
        if
        :: watermark_mutex -> unlock_shared(t, mutex(partition))
        :: else
        fi
    }
}

init { atomic { run committer(0); run committer(1); run reader(2) } }
