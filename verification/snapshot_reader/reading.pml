/**
 *  @file verification/snapshot_reader/reading.pml
 *  @author Ash Vardanian
 *  @date September 16, 2026
 *  @brief A pinned @c snapshot_store::reader_t reads a key twice at its stamp while a commit prunes
 *      the key's version run behind it.
 *
 *  The reader takes a snapshot, reads the key twice at that stamp under the partition's shared
 *  lock, and gives its claim back after the second read. The claim it holds is what keeps the
 *  prune off the version it reads.
 *
 *  One commit by default: two of them under the views do not finish inside the memory this suite
 *  gives a model, and only the mark's ordering asks for the second one, which `committers=2`
 *  supplies under sequential consistency, since the shape it turns on is an interleaving rather
 *  than a reordering.
 *
 *  @c bucket_retag is not claimed here, because this model passes without it: its one joiner
 *  reads its stamp after joining, and a floor an opener stores before shutting the bucket was
 *  computed before that read, so it still bounds every member the bucket can take.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc committers=2
 *  @verify fail sc,rc11 held_claim=false: the reader holds its claim until after its last read;
 *      giving it back before the reads, the commit that prunes next frees the version out from
 *      under the first of them
 *  @verify fail sc,rc11 join_first=false: @c take_snapshot joins the bucket before it reads the
 *      watermark, the reader's half of the pair; reading the watermark first, a mark computed in
 *      that gap counts nobody while standing above the stamp the reader is handed
 *  @verify fail sc committers=2 watermark_first=false: @c republish_mark_ reads the watermark
 *      before it scans the buckets; scanning first, a reader joining after the scan is missed by
 *      it while the watermark read that follows, raised by the second commit inside that gap, has
 *      already passed the stamp that reader was handed
 */
#ifndef committers
#define committers 1
#endif
#define thread_count (committers + 1)
#define location_count 9
#define history_depth (4 * committers + 5)
#include "protocol.pml"

proctype committer(byte t) { commit(t) }

/** The reader: takes a snapshot, reads at its stamp twice through @c reader_t::find, each under the
 *  partition's shared lock and neither writing anything, and gives its claim back. */
proctype reader(byte t) {
    byte reader_bucket, first, second, each, scanned;
    int seen, observed, opened, least, newest_stamp, mark, snapshot;
    bool exchanged;
    take_claim(t, reader_bucket, snapshot);
    if
    :: !held_claim -> mark = 0; retire_snapshot(t, reader_bucket)
    :: else
    fi;
    lock_shared(t, partition_mutex);
    resolve(snapshot, first);
    assert(first != none && !version_freed[first]);
    unlock_shared(t, partition_mutex);
    lock_shared(t, partition_mutex);
    resolve(snapshot, second);
    assert(second == first && !version_freed[second]);
    unlock_shared(t, partition_mutex);
    if
    :: held_claim -> mark = 0; retire_snapshot(t, reader_bucket)
    :: else
    fi;
    landed(t)
}

init {
    byte each;
    atomic {
        for (each : 0 .. committers - 1) { run committer(each) };
        run reader(committers)
    }
}
