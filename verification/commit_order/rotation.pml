/**
 *  @file verification/commit_order/rotation.pml
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief Two commits in turn rotate the head twice, so a bucket a reader may have joined is
 *      recycled under a pruner.
 *
 *  The roles of `census.pml` over two landed stamps, with the reader held back until the first
 *  rotation happened, since a reader racing that first rotation is what `census.pml` already
 *  explores and two commits raced from the start do not finish. It is the only scenario that
 *  reaches a recycled bucket, so it is where the join ceiling refuses a tag above the head just
 *  read and where a second opener from the same head value finds its bucket already tagged.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc,rc11 bucket_retag=false: @c open_next_bucket_ shuts a bucket to arrivals before
 *      it stores the floor; stored first, a joiner may be counted in a bucket whose floor is being
 *      replaced, yet no mark passes its snapshot, since it reads its stamp after joining while the
 *      opener's floor was read before that, and the release sequence over the watermark keeps the
 *      floor at or below the snapshot either way. What the retag buys is a floor that never moves
 *      under a member, a shape rather than a value, which no assertion here names
 */
#define thread_count 3
#define location_count 9
#define history_depth 9
#include "protocol.pml"

/** The first commit rotates the head out of the bucket it opened at, the second recycles the bucket
 *  the reader may have joined. */
proctype committer(byte t) {
    byte scanned;
    int seen, observed, opened, least, mark, drawn;
    bool exchanged, moved;
    commit(t);
    atomic { record_mark(mark); head_rotated = true };
    commit(t);
    record_mark(mark);
    atomic { assert(newest_value(published_stamp) <= newest_value(commits)); finished++ }
}

/** The joiner the second rotation recycles a bucket under: it reads the head after the first
 *  rotation, and the head it reads may be the one before it, which names that very bucket. */
proctype reader(byte t) {
    byte joined;
    int observed, opened, snapshot;
    bool exchanged;
    (head_rotated);
    take_snapshot(t, joined, snapshot);
    atomic { reader_snapshot = snapshot; reader_settled = true; assert(freed_below <= snapshot) }
}

/** @c republish_mark_ over the recycled bucket, racing the second commit that recycles it. */
proctype pruner(byte t) {
    byte scanned;
    int observed, opened, least, mark;
    bool exchanged;
    republish_mark(t, mark);
    record_mark(mark)
}

init { atomic { run committer(0); run reader(1); run pruner(2) } }
