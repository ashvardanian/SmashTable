/**
 *  @file verification/commit_order_publication.pml
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief The word of landings and the watermark of @c basic_commit_order, with one version per
 *      committer for a reader to resolve at its snapshot.
 *
 *  The stamps @c begin_commit draws from @c commits_ and the word @c end_commit flips a bit of and
 *  raises the watermark from, with no census at all. Two committers, a reader and the auditor. A
 *  committer draws a stamp, writes its version, and lands the stamp. Two committers are the least
 *  that let one land before the other, and three are what overflow a window of two; @c committers
 *  is that count. The word is @c word_bits wide, four by default with a window of two, and
 *  @c resumed is the stamp the order restarts at. With @c parity_word=0 it is the ring of two
 *  stamps the word replaced, walked slot by slot.
 *
 *  - The watermark is monotone and never covers a stamp whose mark is absent.
 *  - Every version a snapshot covers is there to be read.
 *  - Once every commit finished the watermark names every stamp drawn.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc,rc11 committers=3
 *  @verify pass sc committers=4
 *  @verify pass sc committers=4 resumed=3: stamps four to seven, so the word laps
 *  @verify pass sc committers=4 word_bits=2: two laps of a two-bit word
 *  @verify pass sc,rc11 committers=3 read_before_flip=true: a watermark older than the word only
 *      starts the run lower, and every bit it then counts is a stamp that landed
 *  @verify fail sc committers=4 own_stamp_return=false: a lander stalled after its flip reads a
 *      watermark past its own stamp and counts bits a lap stale as landed
 *  @verify fail sc committers=4 wait_bound=ring run_bound=window: waiting on the whole word with
 *      the run capped, a capped raise stops inside a landed run, and a lander stalled before it
 *      counts the stamp a lap above an unlanded one as landed
 *  @verify pass sc committers=4 wait_bound=ring: uncapped, the last lander of a run raises through
 *      all of it, so no watermark stops inside one
 *  @verify fail sc committers=3 word_bits=2 ring_check=false: without the wait, a stamp a lap above
 *      an unlanded one flips its bit, and the watermark covers a stamp not landed
 *  @verify stuck sc committers=3 recheck=false: without the re-check, the raises that make room
 *      land and wake between its load and its arming, so it sleeps beside a window with room
 *  @verify stuck sc committers=3 watermark_notify=false: without the wake, the third committer
 *      parked for room in the window sleeps through the raises that made it
 *  @verify pass sc,rc11 parity_word=0: the ring of stamps the word replaced
 *  @verify pass sc parity_word=0 committers=3
 *  @verify fail rc11 parity_word=0 done_order=order_relaxed: relaxed, the committer that walks past
 *      another's mark carries none of that committer's version to the reader, which reads a version
 *      its snapshot covers unwritten
 *  @verify pass sc parity_word=0 done_order=order_relaxed: sequential consistency carries every
 *      write without an order, so only weak memory shows the missing release
 *  @verify fail rc11 parity_word=0 advance_read_modify_write=false: with a plain acquire load, two
 *      finishers pass each other's marks, and the watermark stops short of a stamp drawn
 *  @verify pass sc parity_word=0 advance_read_modify_write=false: sequential consistency hides the
 *      store buffering the plain load lets through
 *  @verify fail sc parity_word=0 committers=3 ring_check=false: without the wait, a third committer
 *      overwrites the mark of a stamp the watermark has not consumed. The shape is an interleaving
 *      rather than a reordering, so sequential consistency is where it is cheapest, and three
 *      committers under the views do not finish inside the memory this suite gives a model
 *  @verify fail rc11 parity_word=0 walk_order=order_relaxed: relaxed, the committer that walks past
 *      another's mark carries none of that committer's version to the reader, which reads a version
 *      its snapshot covers unwritten
 *  @verify stuck sc parity_word=0 committers=3 recheck=false
 *  @verify stuck sc parity_word=0 committers=3 watermark_notify=false
 *  @budget 6144
 */
#ifndef committers
#define committers 2
#endif
#if committers < 1 || committers > 4
#error "committers is 1 to 4"
#endif

/** One or zero, as <tt>commit_order_steps.pml</tt> asks: it lays out the words below. */
#ifndef parity_word
#define parity_word 1
#endif
#define thread_count (committers + 1)
#define history_depth (2 * committers + 2)
#ifndef word_bits
#define word_bits 4
#endif
#ifndef resumed
#define resumed 0
#endif
#if parity_word
#define ring word_bits
#define window (word_bits / 2)
#define location_count (3 + committers)
#else
#define ring 2
#define location_count (2 + ring + committers)
#endif
#include <weak_memory.pml>
#include <monitor_wait.pml>

/** The words: the stamp counter, the watermark, the ring of marks or the one word of their
 *  parities, and a version per committer. */
#define commits 0
#define published_stamp 1
#if parity_word
#define landed_parity 2
#define version(committer) (3 + (committer))
#else
#define landed_at(slot) (2 + (slot))
#define version(committer) (2 + ring + (committer))
#endif

#include <commit_order_steps.pml>

byte finished;            // the committers that landed their stamp
int stamp_of[committers]; // each committer's stamp, once drawn

/** A committer: draws a stamp, writes its version, and lands the stamp. */
proctype committer(byte t) {
    byte landed_run;
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
#if parity_word
    assert(newest_value(landed_parity) == parity_through(newest_value(commits)));
#endif
    for (each : 0 .. committers - 1) { assert(stamp_of[each] <= newest_value(published_stamp)) }
}

init {
    byte each;
    atomic {
#if parity_word
        // As `resume` seeds them: every stamp through `resumed` landed
        seed(commits, resumed);
        seed(published_stamp, resumed);
        seed(landed_parity, parity_through(resumed));
        flipped = (1 << (resumed + 1)) - 1;
#endif
        for (each : 0 .. committers - 1) { run committer(each) };
        run reader(committers);
        run auditor()
    }
}
