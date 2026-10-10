/**
 *  @file verification/partitioned_store/cursor.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Two commits over two partitions of @c partitioned_store and a cursor that reads each
 *      partition at least as new as the epoch it acquired.
 *
 *  The cursor reads each partition's epoch with acquire, as @c refresh_stale_fronts_ does to decide
 *  whether to re-read, and then the partition without a lock. The header reads a front under the
 *  partition's shared lock, so the release on the epoch orders only the steps that @c for_all and
 *  @c publish_every_part_ post after their locks are gone.
 *
 *  @verify pass rc11
 *  @verify fail rc11 epoch_order=order_relaxed: a commit steps each partition's epoch with a
 *      release as it lets go; relaxed, a cursor that acquired the epoch reads the partition older
 *      than the step left it
 */
#define thread_count 3
#define location_count 9
#define history_depth 9
#include "protocol.pml"

proctype committer(byte t) { commit(t) }

/** A cursor: the epoch with acquire, then the partition without a lock:
 *  @c partitioned_store::ordered_cursor_t::refresh_stale_fronts_ over @c epoch_seen_. */
proctype reader(byte t) {
    byte partition;
    int seen_version, seen_epoch;
    for (partition : 0 .. partitions - 1) {
        load(t, epoch(partition), order_acquire, seen_epoch);
        load(t, version(partition), order_relaxed, seen_version);
        assert(seen_version >= version_at_epoch[partition * 4 + seen_epoch])
    }
}

init { atomic { run committer(0); run committer(1); run reader(2) } }
