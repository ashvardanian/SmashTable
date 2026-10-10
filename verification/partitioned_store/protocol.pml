/**
 *  @file verification/partitioned_store/protocol.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c partitioned_store from `include/smashtable/partitioned_store.hpp` over a
 *      shared @c basic_commit_order.
 *
 *  A commit holds every partition it reached, ascending, asks each whether it may proceed before
 *  any writes, draws one stamp, writes every partition under it, moves the watermark once the last
 *  is written, and steps each partition's epoch with a release as it lets go.
 *
 *  Two committers over two partitions. Each reaches the second partition, and the first as well or
 *  not, as it likes. A commit holds every partition it reached from its ask through its write, so
 *  no committer writes a partition another has asked about and not yet written. The watermark never
 *  passes a stamp still in flight, and never falls.
 *
 *  The order is nine words already with two partitions, so its steps stand here as one stamp drawn
 *  and one watermark stored under a mutex, and what @c begin_commit and @c end_commit really do to
 *  the ring of done marks is `commit_order_publication.pml`'s.
 */
#include "../weak_memory.pml"
#include "../spin_shared_mutex.pml"

#define partitions 2

/** The words: each partition's mutex, version and epoch; then the order's mutex, watermark and
 *  counter of stamps drawn. */
#define mutex(partition) (partition)
#define version(partition) (2 + (partition))
#define epoch(partition) (4 + (partition))
#define order_mutex 6
#define published_stamp 7
#define commits 8

/** The knobs: the header's choices, which `@verify` lines override to replay counterexamples. */
#ifndef ascending_order
#define ascending_order true
#endif
#ifndef held_partitions
#define held_partitions true
#endif
#ifndef epoch_order
#define epoch_order order_release
#endif
#ifndef watermark_mutex
#define watermark_mutex true
#endif

int in_flight[2];                         // each committer's drawn stamp until its commit is whole
int stamp_of[2];                          // each committer's stamp, once drawn
bool touches[2 * partitions];             // which partitions each committer reached
bool validated[2 * partitions];           // a partition asked about and not yet written
int version_at_epoch[2 * partitions * 4]; // the version a partition held when its epoch stepped
#define at(committer, partition) ((committer) * partitions + (partition))

/** A commit: holds the partitions it reached, asks, draws a stamp, writes, moves the watermark,
 *  and lets go. */
inline commit(t) {
    byte partition, first, last;
    int seen, drawn, whole;
    touches[at(t, 1)] = true;
    if
    :: touches[at(t, 0)] = true
    :: skip
    fi;
    // The reached partitions, ascending: `partitioned_store::touched_parts_lock_t`
    if
    :: !ascending_order && t == 1 -> first = 1; last = 0
    :: else -> first = 0; last = 1
    fi;
    if :: touches[at(t, first)] -> lock(t, mutex(first)) :: else fi;
    if :: touches[at(t, last)] -> lock(t, mutex(last)) :: else fi;
    // `validate_for_commit` on every reached partition before any writes, which is
    // `partitioned_store::transaction_t::validate_reached_parts_` under `commit_under_one_stamp_`;
    // no partition refuses here, so releasing the earlier holds at a refusal is the group model's
    for (partition : 0 .. partitions - 1) { validated[at(t, partition)] = touches[at(t, partition)] };
    if
    :: !held_partitions ->
        if :: touches[at(t, last)] -> unlock(t, mutex(last)) :: else fi;
        if :: touches[at(t, first)] -> unlock(t, mutex(first)) :: else fi;
        if :: touches[at(t, first)] -> lock(t, mutex(first)) :: else fi;
        if :: touches[at(t, last)] -> lock(t, mutex(last)) :: else fi
    :: else
    fi;
    // `begin_commit`: the stamp drawn under the order's mutex, and recorded in flight
    lock(t, order_mutex);
    read_modify_write(t, commits, order_relaxed, seen, seen + 1);
    drawn = seen + 1;
    atomic { in_flight[t] = drawn; stamp_of[t] = drawn };
    unlock(t, order_mutex);
    // `publish_under` on every reached partition:
    // `partitioned_store::transaction_t::commit_under_one_stamp_`
    for (partition : 0 .. partitions - 1) {
        if
        :: touches[at(t, partition)] ->
            // The other committer cannot be mid-commit on this partition: this one holds it
            assert(!validated[at(1 - t, partition)]);
            store(t, version(partition), order_relaxed, drawn);
            validated[at(t, partition)] = false
        :: else
        fi
    };
    // `end_commit`: the watermark moves to one below the oldest in flight, or to the newest drawn
    lock(t, order_mutex);
    in_flight[t] = 0;
    if
    :: in_flight[1 - t] != 0 -> whole = in_flight[1 - t] - 1
    :: else -> whole = newest_value(commits)
    fi;
    assert(whole >= newest_value(published_stamp));
    store(t, published_stamp, order_relaxed, whole);
    unlock(t, order_mutex);
    // Release: each partition's epoch stepped with a release, then its lock dropped:
    // `partitioned_store::touched_parts_lock_t::release` over `note_written_`
    for (partition : 0 .. partitions - 1) {
        if
        :: touches[at(t, partition)] ->
            atomic {
                read_modify_write(t, epoch(partition), epoch_order, seen, seen + 1);
                version_at_epoch[partition * 4 + seen + 1] = newest_value(version(partition))
            };
            unlock(t, mutex(partition))
        :: else
        fi
    }
}
