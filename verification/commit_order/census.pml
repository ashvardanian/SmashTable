/**
 *  @file verification/commit_order/census.pml
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief A reader joining a bucket while one commit lands and a pruner computes the mark, with
 *      nothing sequenced between the three.
 *
 *  The reader takes its snapshot and holds the claim for the rest of the run, so every mark the
 *  committer and the pruner compute beside it is checked against it; handing a claim back is what
 *  `sharing.pml` and `retention.pml` play out. The pruner runs @c republish_mark_ on a thread of
 *  its own, which is the mark a pruning committer computes without landing a stamp.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 watermark_first=false: @c republish_mark_ reads the watermark before it
 *      scans the buckets; scanning first, it misses a reader that joined in between, while the
 *      watermark read that follows has already passed the stamp that reader was handed
 *  @verify fail rc11 take_read_modify_write=false: @c take_snapshot reads the watermark through a
 *      read-modify-write after it joins, which hands the join to every later read-modify-write of
 *      the watermark; with a plain acquire load, a commit landing after the reader took its stamp
 *      scans the buckets without seeing the join, and its mark passes the stamp the reader holds
 */
#define thread_count 3
#define location_count 9
#define history_depth 6
#include "protocol.pml"

proctype committer(byte t) {
    byte scanned;
    int seen, observed, opened, least, mark, drawn;
    bool exchanged, moved;
    commit(t);
    record_mark(mark);
    atomic { assert(newest_value(published_stamp) <= newest_value(commits)); finished++ }
}

/** @c take_snapshot: the bucket joined, then the watermark read newest, and nothing re-read. */
proctype reader(byte t) {
    byte joined;
    int observed, opened, snapshot;
    bool exchanged;
    take_snapshot(t, joined, snapshot);
    atomic { reader_snapshot = snapshot; reader_settled = true; assert(freed_below <= snapshot) }
}

proctype pruner(byte t) {
    byte scanned;
    int observed, opened, least, mark;
    bool exchanged;
    republish_mark(t, mark);
    record_mark(mark)
}

init { atomic { run committer(0); run reader(1); run pruner(2) } }
