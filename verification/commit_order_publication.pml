/**
 *  @file verification/commit_order_publication.pml
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief The stamp ring and the watermark walk of @c basic_commit_order, with one version per
 *      committer for a reader to resolve at its snapshot.
 *
 *  The stamps @c begin_commit draws from @c commits_ and the ring of marks @c end_commit lands
 *  into and walks the watermark through, from `include/smashtable/shared.hpp` under the steps of
 *  `commit_order_steps.pml`, with no census at all: the buckets, the floors and the mark are
 *  `commit_order/`'s, whose words leave no room for a version per committer.
 *
 *  Two committers, a reader and the auditor. A committer draws a stamp, writes its version, and
 *  lands the stamp. The reader takes its snapshot as the watermark's read-modify-write, which with
 *  no census is the whole of @c take_snapshot. The auditor reads no word and plays no thread: it
 *  waits for every commit to finish and reads the two counters' newest writes.
 *
 *  The watermark is monotone and never covers a stamp whose mark is absent, so every version a
 *  snapshot covers is there to be read, which the reader asserts per committer. Once every commit
 *  finished the watermark names every stamp drawn, so a walk that stopped with a mark behind it is
 *  a lost mark, which the auditor catches.
 *
 *  Two committers are the least that let one land before the other, which is what the ring asks
 *  for, and three are what overflow a ring of two; @c committers is that count.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc committers=3
 *  @verify fail rc11 done_order=order_relaxed: @c end_commit stores the landed mark with release;
 *      relaxed, the committer that walks past another's mark carries none of that committer's
 *      version to the reader, which reads a version its snapshot covers unwritten
 *  @verify pass sc done_order=order_relaxed: sequential consistency carries every write without
 *      an order, so only weak memory shows the missing release
 *  @verify fail rc11 advance_read_modify_write=false: @c end_commit reads the watermark through a
 *      read-modify-write before the walk; with a plain acquire load, two finishers pass each
 *      other's marks, and the watermark stops short of a stamp drawn
 *  @verify pass sc advance_read_modify_write=false: sequential consistency hides the store
 *      buffering the plain load lets through
 *  @verify fail sc committers=3 ring_check=false: @c end_commit waits until the watermark passed
 *      the stamp a ring slot holds before reusing it; without the wait, a third committer
 *      overwrites the mark of a stamp the watermark has not consumed, which @c land_stamp asserts
 *      against as it stores. The shape is an interleaving rather than a reordering, so sequential
 *      consistency is where it is cheapest, and three committers under the views do not finish
 *      inside the memory this suite gives a model
 */
#ifndef committers
#define committers 2
#endif
#if committers < 1 || committers > 3
#error "committers is 1 to 3"
#endif
#define ring 2
#define thread_count (committers + 1)
#define location_count (2 + ring + committers)
#define history_depth (2 * committers + 2)
#include "weak_memory.pml"

/** The words: the stamp counter, the watermark, the ring of marks, and a version per committer. */
#define commits 0
#define published_stamp 1
#define landed_at(slot) (2 + (slot))
#define version(committer) (2 + ring + (committer))

#include "commit_order_steps.pml"

byte finished;            // the committers that landed their stamp
int stamp_of[committers]; // each committer's stamp, once drawn

/** A committer: draws a stamp, writes its version, and lands the stamp. */
proctype committer(byte t) {
    int seen, observed, drawn;
    bool moved;
    // `begin_commit`: the stamp drawn, and recorded as drawn in the same step it is handed out
    atomic { draw_stamp(t, drawn); stamp_of[t] = drawn };
    store(t, version(t), order_relaxed, drawn);
    // `end_commit`: the mark, then the watermark walked in stamp order
    land_stamp(t, drawn);
    atomic { assert(newest_value(published_stamp) <= newest_value(commits)); finished++ }
}

/** The snapshot as the watermark read newest; every version it covers is written, since its stamp
 *  landed before the watermark reached it and the mark that carried it was released. */
proctype reader(byte t) {
    byte each;
    int snapshot, seen_version;
    read_modify_write(t, published_stamp, order_acq_rel, snapshot, snapshot);
    for (each : 0 .. committers - 1) {
        atomic {
            load(t, version(each), order_relaxed, seen_version);
            if
            :: stamp_of[each] != 0 && stamp_of[each] <= snapshot -> assert(seen_version == stamp_of[each])
            :: else
            fi
        }
    }
}

/** The watermark names every stamp drawn once every commit finished. */
proctype auditor() {
    byte each;
    (finished == committers);
    assert(newest_value(published_stamp) == newest_value(commits));
    for (each : 0 .. committers - 1) { assert(stamp_of[each] <= newest_value(published_stamp)) }
}

init {
    byte each;
    atomic {
        for (each : 0 .. committers - 1) { run committer(each) };
        run reader(committers);
        run auditor()
    }
}
