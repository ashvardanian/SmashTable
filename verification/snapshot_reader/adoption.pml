/**
 *  @file verification/snapshot_reader/adoption.pml
 *  @author Ash Vardanian
 *  @date September 16, 2026
 *  @brief A transaction adopts a pinned reader's stamp through @c share_snapshot and reads after
 *      the reader has closed, while a commit prunes behind it.
 *
 *  The transaction is one more member of the bucket the reader is already counted in; the reader
 *  gives its own claim back, and the transaction reads afterwards. Both run on the reader's
 *  thread, since @c transaction_t is constructed from the @c reader_t it adopts, so
 *  `commit_order/sharing.pml` is where a shared member crosses threads.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 shared_claim=false: a stamp adopted from a live claim stays pinned once
 *      that claim retires, because the adopting transaction is counted in the same bucket before
 *      the reader leaves it; copying the stamp alone, the drain that follows lets the mark past it
 */
#define committers 1
#define thread_count 2
#define location_count 9
#define history_depth 7
#include "protocol.pml"

proctype committer(byte t) { commit(t) }

/** The reader and the transaction adopting its stamp, on one thread. */
proctype reader(byte t) {
    byte reader_bucket, adopting_bucket, first, each, scanned;
    int seen, observed, opened, least, newest_stamp, mark, snapshot;
    bool exchanged;
    take_claim(t, reader_bucket, snapshot);
    // `share_snapshot`: the transaction counted in the reader's bucket while the reader is in it
    if
    :: shared_claim -> share_snapshot(t, reader_bucket, adopting_bucket)
    :: else -> adopting_bucket = reader_bucket
    fi;
    // The reader closes, leaving the transaction to pin the stamp both of them read at
    mark = 0;
    retire_snapshot(t, reader_bucket);
    lock_shared(t, partition_mutex);
    resolve(snapshot, first);
    assert(first != none && !version_freed[first]);
    unlock_shared(t, partition_mutex);
    if
    :: shared_claim -> mark = 0; retire_snapshot(t, adopting_bucket)
    :: else
    fi;
    landed(t)
}

init { atomic { run committer(0); run reader(1) } }
