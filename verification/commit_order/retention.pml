/**
 *  @file verification/commit_order/retention.pml
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief The handover: a successor takes its own snapshot while the first reader still holds one,
 *      and the mark recomputed once the first retires stands at the successor's.
 *
 *  The successor joins once the commit landed, the first reader retires, and the pruner recomputes
 *  the mark with only the successor left. The head rotates out of a bucket whose floor has fallen
 *  behind the watermark, so a successor that joins after a commit lands pins that commit rather
 *  than the floor the claim it took over from was admitted at.
 *
 *  Sequential memory only: what it asserts is which bucket the head names, and under weaker memory
 *  a stale head read legitimately shares an older bucket, a conservative census that retains more
 *  than it must and never less.
 *
 *  @verify pass sc
 *  @verify fail sc head_watermark=false: @c republish_mark_ seals the head bucket against the
 *      watermark; sealed against the computed mark, which equals that bucket's floor by
 *      construction wherever it holds the least one, the head never moves and the successor
 *      over-retains on the floor the first claim was admitted at
 *  @verify fail sc buckets=1: the head needs a second bucket to rotate into; with one, a reader
 *      handing over to the next pins the first one's floor for good, which is the configuration
 *      the `buckets_k >= 2` static assert refuses outright
 */
#define thread_count 4
#define location_count 9
#define history_depth 8
#include "protocol.pml"

proctype committer(byte t) {
    byte scanned;
    int seen, observed, opened, least, mark, drawn;
    bool exchanged, moved;
    commit(t);
    record_mark(mark);
    atomic { assert(newest_value(published_stamp) <= newest_value(commits)); finished++ }
}

/** The claim the successor takes over from: held until the successor has one of its own, then
 *  retired, which leaves the successor's floor the only one the mark still stands at. */
proctype reader(byte t) {
    byte joined, scanned;
    int observed, opened, least, mark, snapshot;
    bool exchanged;
    take_snapshot(t, joined, snapshot);
    atomic { reader_snapshot = snapshot; reader_settled = true };
    (successor_taken);
    atomic { reader_settled = false; mark = 0 };
    retire_snapshot(t, joined);
    atomic { record_mark(mark); reader_gone = true }
}

/** A snapshot of its own, taken once the commit landed and while the first claim is still held,
 *  which is the bucket the head has to have rotated out of. */
proctype successor(byte t) {
    byte joined;
    int observed, opened, snapshot;
    bool exchanged;
    (reader_settled && finished == 1);
    take_snapshot(t, joined, snapshot);
    atomic { successor_snapshot = snapshot; successor_settled = true; successor_taken = true }
}

/** @c republish_mark_ with only the successor's claim left, which stands at the watermark the
 *  successor joined under rather than at the floor the claim it took over from was admitted at. */
proctype pruner(byte t) {
    byte scanned;
    int observed, opened, least, mark;
    bool exchanged;
    (reader_gone);
    republish_mark(t, mark);
    atomic { record_mark(mark); assert(mark >= successor_snapshot) }
}

init { atomic { run committer(0); run reader(1); run successor(2); run pruner(3) } }
