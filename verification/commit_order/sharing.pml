/**
 *  @file verification/commit_order/sharing.pml
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief A snapshot shared off a live claim through @c share_snapshot stays pinned once the claim
 *      it was shared from retires.
 *
 *  One reader takes a snapshot, a sharer shares it off the reader while the reader is still live,
 *  and the reader retires; one commit lands beside them, since one landed stamp is all it takes for
 *  a mark to pass the shared snapshot. The shared member is posted with release onto a bucket that
 *  still carries the claim it was shared from, so that claim's drain leaves the bucket occupied.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail far share_order=order_relaxed: @c share_snapshot posts the shared member with
 *      release; relaxed, far memory lands it only after the claim it was shared from drained the
 *      bucket, and the mark that drain recomputes passes the shared snapshot
 */
#define thread_count 3
#define location_count 9
#define history_depth 7
#include "protocol.pml"

proctype committer(byte t) {
    byte scanned;
    int seen, observed, opened, least, mark, drawn;
    bool exchanged, moved;
    commit(t);
    record_mark(mark);
    atomic { assert(newest_value(published_stamp) <= newest_value(commits)); finished++ }
}

/** The claim a snapshot is shared off, held until the share has been taken, then retired while the
 *  shared member is the only one left pinning the bucket. */
proctype reader(byte t) {
    byte joined, scanned;
    int observed, opened, least, mark, snapshot;
    bool exchanged;
    take_snapshot(t, joined, snapshot);
    atomic { reader_bucket = joined; reader_snapshot = snapshot; reader_settled = true };
    (shared_taken);
    atomic { assert(freed_below <= snapshot); reader_settled = false; mark = 0 };
    retire_snapshot(t, joined);
    atomic { record_mark(mark); reader_gone = true }
}

/** @c share_snapshot: a claim of the transaction's own, posted onto the reader's bucket while the
 *  reader still holds one, and the reader's stamp copied without a re-read of the watermark. */
proctype sharer(byte t) {
    byte joined, scanned;
    int observed, opened, least, mark, snapshot;
    bool exchanged;
    (reader_settled);
    share_snapshot(t, reader_bucket, joined);
    atomic { snapshot = reader_snapshot; sharer_snapshot = snapshot; sharer_settled = true; shared_taken = true };
    // The claim this was shared off has retired: the shared member is the only one still pinning it
    (reader_gone);
    atomic { assert(freed_below <= snapshot); sharer_settled = false; mark = 0 };
    retire_snapshot(t, joined);
    record_mark(mark);
    landed(t)
}

init { atomic { run committer(0); run reader(1); run sharer(2) } }
