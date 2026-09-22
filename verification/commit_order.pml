/**
 *  `basic_commit_order` from `include/smashtable/shared.hpp` under the steps of
 *  `commit_order_steps.pml`: the stamps `begin_commit` draws from `commits_`, the ring of marks
 *  `end_commit` lands into and walks the watermark through, the buckets `take_snapshot` joins and
 *  `retire_snapshot` leaves, the member `share_snapshot` posts onto a claim that is still live, and
 *  the mark `republish_mark_` computes from the watermark and the occupied buckets. Readers are
 *  counted per bucket here, never listed one by one, and a claim is a bucket index and a stamp.
 *
 *  Five scenarios, since the nine words the memory module keeps do not hold the ring, the census
 *  and a version per committer at once, and since each invariant needs its own roles out of the
 *  module's five threads. Every scenario plays the fewest threads and the fewest commits its own
 *  assertions can fail under, because each of either multiplies what has to be explored.
 *  - `-Dscenario=publication`, the default: the ring and the watermark, with one version word per
 *    committer for a reader to resolve at its snapshot, and no census at all. Two committers, a
 *    reader and the auditor.
 *  - `-Dscenario=census`: the buckets, the floors and the mark, against a reader that joins while
 *    one commit lands and a pruner computes, with nothing sequenced between the three.
 *  - `-Dscenario=rotation`: the same three roles over two landed stamps, with the reader held back
 *    until the first rotation happened, since a reader racing that first rotation is what `census`
 *    already explores and two commits raced from the start do not finish. It is the only scenario
 *    that reaches a recycled bucket, so it is where the join ceiling refuses a tag above the head
 *    just read and where a second opener from the same head value finds its bucket already tagged.
 *  - `-Dscenario=sharing`: one reader takes a snapshot, a second shares it off the first while the
 *    first is still live, and the first retires. One commit, since one landed stamp is all it takes
 *    for a mark to pass the shared snapshot.
 *  - `-Dscenario=retention`: the handover. A successor takes its own snapshot while the first
 *    reader still holds one, the first retires, and the mark is recomputed with only the successor
 *    left. Sequential memory only: what it asserts is which bucket the head names, and under weaker
 *    memory a stale head read shares an older bucket legitimately, which the census is allowed to
 *    be conservative about.
 *
 *  A committer draws a stamp, writes its version where the scenario keeps one, and lands the stamp.
 *  The auditor reads no word and plays no thread: it waits for every commit to finish and reads the
 *  two counters' newest writes.
 *
 *  Invariants:
 *  - the watermark is monotone and never covers a stamp whose mark is absent, so every version a
 *    snapshot covers is there to be read, which the reader asserts per committer under
 *    `publication`. `-Dwithout_done_release` marks the ring relaxed, and the committer that walks
 *    past another's mark carries none of that committer's version to the reader;
 *  - once every commit finished the watermark names every stamp drawn, so a walk that stopped with
 *    a mark behind it is a lost mark, which the auditor catches. `-Dwithout_advance_rmw` re-reads
 *    the watermark with a plain acquire load, and two finishers pass each other's marks under views
 *    while sequential consistency hides it; `-Dwithout_ring_check` lands a stamp `ring` past an
 *    unpublished one, whose mark overwrites the older one's slot, and the watermark sticks below it,
 *    which takes `-Dcommitters=3` to reach past a ring of two, and which is run under
 *    `-Dmemory=sequential`, both because the shape it turns on is an interleaving rather than a
 *    reordering and because a third committer under views does not finish inside the memory this
 *    suite gives a model;
 *  - the mark never passes a snapshot a live claim names, which `record_mark` checks every mark
 *    anyone computes against. `-Dwithout_watermark_first` scans the buckets before reading the
 *    watermark and misses a reader that joined in between, which `census` catches;
 *    `-Dwithout_bucket_retag`, which the shared steps also carry, is not claimed by any scenario
 *    here, for the reason `snapshot_reader.pml` gives: a joiner reads its stamp after joining and
 *    an opener stores a floor it read before that, so the release sequence over the watermark puts
 *    the floor at or below the snapshot whichever way the two interleave;
 *  - a snapshot shared off a live claim stays pinned after that claim retires, because the shared
 *    member is posted with release onto a bucket that still carries the claim it was shared from.
 *    `-Dwithout_share_release` posts it relaxed, which `-Dmemory=far` lands only after the first
 *    claim drained the bucket, and the mark that drain recomputes passes the shared snapshot;
 *  - the head rotates out of a bucket whose floor has fallen behind the watermark, so a successor
 *    that joins after a commit lands pins that commit rather than the floor the claim it took over
 *    from was admitted at. `-Dwithout_head_watermark` seals the head against the computed mark,
 *    which equals that bucket's floor by construction wherever that bucket holds the least one, so
 *    the head never moves; `-Dbuckets=1` leaves the head nowhere to rotate to, which is the
 *    configuration `buckets_k >= 2` refuses outright.
 *
 *  The words, and the nine the module allows. `publication` uses up to eight: `commits` 0,
 *  `published_stamp` 1, `landed_at(0..1)` 2 and 3, and `version(0..2)` 4, 5 and 6. The census
 *  scenarios use all nine: `commits` 0, `published_stamp` 1, `landed_at(0..1)` 2 and 3, `head` 4,
 *  `members(0..1)` 5 and 6, and `floors(0..1)` 7 and 8. Left out of every scenario: the
 *  `low_water_mark_` word, since each role acts on the mark it just computed, which is the freshest
 *  such a read can be; and `generation_`, which dates transactions rather than their visibility.
 */
#include "weak_memory.pml"

// The knob's values are integers, so a typo fails the range check below.
#define publication 1
#define census 2
#define rotation 3
#define sharing 4
#define retention 5
#ifndef scenario
#define scenario publication
#endif
#if scenario < publication || scenario > retention
#error "scenario is publication, census, rotation, sharing or retention"
#endif

// How many commits overlap. Two is the least that lets one land before the other, which is what the
// ring asks for, and three is what overflows a ring of two; every other scenario watches one thread
// publish and needs no overlap at all.
#ifndef committers
#if scenario == publication
#define committers 2
#else
#define committers 1
#endif
#endif
#if committers < 1 || committers > 3
#error "committers is 1 to 3"
#endif
#if scenario != publication && committers != 1
#error "only publication plays more than one committer"
#endif

// The ring and the census, sized before the words are spelled over them.
#ifndef ring
#define ring 2
#endif
#if scenario == publication
#define buckets 0
#else
#ifndef buckets
#define buckets 2
#endif
#endif

// The threads, by role, apart from the processes that play them; the committers name themselves,
// and the auditor plays none.
#define reader_thread committers
#define sharer_thread (committers + 1)
#define successor_thread (committers + 1)
#if scenario == retention
#define pruner_thread (committers + 2)
#else
#define pruner_thread (committers + 1)
#endif

// How many roles a scenario plays beside its committers, against the module's threads.
#if scenario == publication
#define roles 1
#endif
#if scenario == census || scenario == rotation || scenario == sharing
#define roles 2
#endif
#if scenario == retention
#define roles 3
#endif
#if committers + roles > thread_count
#error "the committers and the scenario's other roles outgrow the module's threads"
#endif

// The words: the stamp counter and the watermark, the ring of marks, and then either one version
// per committer or the census the readers are counted in.
#define commits 0
#define published_stamp 1
#define landed_at(slot) (2 + (slot))
#if buckets > 0
#define head (2 + ring)
#define members(bucket) (3 + ring + (bucket))
#define floors(bucket) (3 + ring + buckets + (bucket))
#else
#define version(committer) (2 + ring + (committer))
#endif

// The words have to fit the module's `location_count`, which every model here shares.
#if buckets > 0
#if 3 + ring + 2 * buckets > location_count
#error "the ring and the census together outgrow the module's words"
#endif
#else
#if 2 + ring + committers > location_count
#error "the ring and one version per committer together outgrow the module's words"
#endif
#endif

#include "commit_order_steps.pml"

byte finished;

#if scenario == publication
byte committers_started;
int stamp_of[committers]; // each committer's stamp, once drawn
#endif

#if buckets > 0
bool reader_settled;  // the reader holds a claim, so every mark recorded now is checked against it
int reader_snapshot;  // the stamp that claim reads at
int freed_below;      // the newest mark anyone acted on
#endif

#if scenario == sharing || scenario == retention
bool reader_gone;     // the claim a second one took over from has been given back
#endif

#if scenario == rotation
bool head_rotated;    // the head has left the bucket it opened at, so the next rotation recycles one
#endif

#if scenario == sharing
byte reader_bucket;   // the bucket the sharer posts its own member onto
bool sharer_settled;  // the sharer holds the claim it copied
int sharer_snapshot;  // the stamp that claim reads at, copied rather than re-read
bool shared_taken;    // the share has been posted, so the claim it came off may retire
#endif

#if scenario == retention
bool successor_settled; // the successor holds a claim of its own
int successor_snapshot; // the stamp that claim reads at
bool successor_taken;   // the successor has joined, so the claim it took over from may retire
#endif

#if buckets > 0
/**
 *  Every mark anyone computes, checked against the claims that were settled when it was recorded
 *  and kept as the newest mark anyone acted on. A mark computed before a claim settled is at or
 *  below that claim's snapshot as well, since a claim reads the watermark only after it joins, so
 *  recording one step later than it was computed admits no failure of its own.
 */
inline record_mark(computed) {
    atomic {
        assert(!reader_settled || computed <= reader_snapshot);
#if scenario == sharing
        assert(!sharer_settled || computed <= sharer_snapshot);
#endif
#if scenario == retention
        assert(!successor_settled || computed <= successor_snapshot);
#endif
        if
        :: computed > freed_below -> freed_below = computed
        :: else
        fi
    }
}
#endif

#if scenario == publication
active [committers] proctype committer() {
    byte me;
    int seen, observed, drawn;
    bool moved;
    atomic { me = committers_started; committers_started++ };
    // begin_commit: the stamp drawn, and recorded as drawn in the same step it is handed out
    atomic { draw_stamp(me, drawn); stamp_of[me] = drawn };
    store(me, version(me), order_relaxed, drawn);
    // end_commit: the mark, then the watermark walked in stamp order
    land_stamp(me, drawn);
    atomic { assert(newest_value(published_stamp) <= newest_value(commits)); finished++ }
}
#endif

#if scenario == census || scenario == sharing || scenario == retention
// One commit: begin_commit draws the stamp, end_commit lands the mark, walks the watermark in stamp
// order and republishes the low-water mark.
active proctype committer() {
    byte scanned;
    int seen, observed, opened, least, mark, drawn;
    bool exchanged, moved;
    atomic { draw_stamp(0, drawn); mark = 0 };
    land_stamp(0, drawn);
    record_mark(mark);
    atomic { assert(newest_value(published_stamp) <= newest_value(commits)); finished++ }
}
#endif

#if scenario == rotation
// Two commits in turn: the first rotates the head out of the bucket it opened at, the second
// recycles the bucket the reader may have joined.
active proctype committer() {
    byte scanned;
    int seen, observed, opened, least, mark, drawn;
    bool exchanged, moved;
    atomic { draw_stamp(0, drawn); mark = 0 };
    land_stamp(0, drawn);
    atomic { record_mark(mark); head_rotated = true };
    atomic { draw_stamp(0, drawn); mark = 0 };
    land_stamp(0, drawn);
    record_mark(mark);
    atomic { assert(newest_value(published_stamp) <= newest_value(commits)); finished++ }
}
#endif

#if scenario == publication
// take_snapshot with no census is the watermark read newest; every version that snapshot covers is
// written, since its stamp landed before the watermark reached it and the mark that carried it was
// released.
active proctype reader() {
    byte joined, each;
    int snapshot, seen_version;
    take_snapshot(reader_thread, joined, snapshot);
    for (each : 0 .. committers - 1) {
        atomic {
            load(reader_thread, version(each), order_relaxed, seen_version);
            if
            :: stamp_of[each] != 0 && stamp_of[each] <= snapshot -> assert(seen_version == stamp_of[each])
            :: else
            fi
        }
    }
}
#endif

#if scenario == census
// take_snapshot: the bucket joined, then the watermark read newest, and nothing re-read. The claim
// is held for the rest of the run, so every mark the committer and the pruner compute beside it is
// checked against it; handing a claim back is what `sharing` and `retention` play out.
active proctype reader() {
    byte joined;
    int observed, opened, snapshot;
    bool exchanged;
    take_snapshot(reader_thread, joined, snapshot);
    atomic { reader_snapshot = snapshot; reader_settled = true; assert(freed_below <= snapshot) }
}

// republish_mark_ on a thread of its own, which is the mark a pruning committer computes without
// landing a stamp.
active proctype pruner() {
    byte scanned;
    int observed, opened, least, mark;
    bool exchanged;
    republish_mark(pruner_thread, mark);
    record_mark(mark)
}
#endif

#if scenario == rotation
// The joiner the second rotation recycles a bucket under: it reads the head after the first
// rotation, and the head it reads may be the one before it, which names that very bucket. Held back
// until that first rotation, because a reader racing it is what `census` plays.
active proctype reader() {
    byte joined;
    int observed, opened, snapshot;
    bool exchanged;
    (head_rotated);
    take_snapshot(reader_thread, joined, snapshot);
    atomic { reader_snapshot = snapshot; reader_settled = true; assert(freed_below <= snapshot) }
}

// republish_mark_ over the recycled bucket, racing the second commit that recycles it.
active proctype pruner() {
    byte scanned;
    int observed, opened, least, mark;
    bool exchanged;
    republish_mark(pruner_thread, mark);
    record_mark(mark)
}
#endif

#if scenario == sharing
// The claim a snapshot is shared off, held until the share has been taken, then retired while the
// shared member is the only one left pinning the bucket.
active proctype reader() {
    byte joined, scanned;
    int observed, opened, least, mark, snapshot;
    bool exchanged;
    take_snapshot(reader_thread, joined, snapshot);
    atomic { reader_bucket = joined; reader_snapshot = snapshot; reader_settled = true };
    (shared_taken);
    atomic { assert(freed_below <= snapshot); reader_settled = false; mark = 0 };
    retire_snapshot(reader_thread, joined);
    atomic { record_mark(mark); reader_gone = true }
}

// share_snapshot: a claim of the transaction's own, posted onto the reader's bucket while the
// reader still holds one, and the reader's stamp copied without a re-read of the watermark
active proctype sharer() {
    byte joined, scanned;
    int observed, opened, least, mark, snapshot;
    bool exchanged;
    (reader_settled);
    share_snapshot(sharer_thread, reader_bucket, joined);
    atomic { snapshot = reader_snapshot; sharer_snapshot = snapshot; sharer_settled = true; shared_taken = true };
    // the claim this was shared off has retired: the shared member is the only one still pinning it
    (reader_gone);
    atomic { assert(freed_below <= snapshot); sharer_settled = false; mark = 0 };
    retire_snapshot(sharer_thread, joined);
    record_mark(mark);
    landed(sharer_thread)
}
#endif

#if scenario == retention
// The claim the successor takes over from: held until the successor has one of its own, then
// retired, which leaves the successor's floor the only one the mark still stands at.
active proctype reader() {
    byte joined, scanned;
    int observed, opened, least, mark, snapshot;
    bool exchanged;
    take_snapshot(reader_thread, joined, snapshot);
    atomic { reader_snapshot = snapshot; reader_settled = true };
    (successor_taken);
    atomic { reader_settled = false; mark = 0 };
    retire_snapshot(reader_thread, joined);
    atomic { record_mark(mark); reader_gone = true }
}

// A snapshot of its own, taken once every commit landed and while the first claim is still held,
// which is the bucket the head has to have rotated out of.
active proctype successor() {
    byte joined;
    int observed, opened, snapshot;
    bool exchanged;
    (reader_settled && finished == committers);
    take_snapshot(successor_thread, joined, snapshot);
    atomic { successor_snapshot = snapshot; successor_settled = true; successor_taken = true }
}

// republish_mark_ with only the successor's claim left, which stands at the watermark the successor
// joined under rather than at the floor the claim it took over from was admitted at.
active proctype pruner() {
    byte scanned;
    int observed, opened, least, mark;
    bool exchanged;
    (reader_gone);
    republish_mark(pruner_thread, mark);
    atomic { record_mark(mark); assert(mark >= successor_snapshot) }
}
#endif

#if scenario == publication
// The watermark names every stamp drawn once every commit finished; it reads no word and takes no
// thread index, since the module's threads are the committers and the roles beside them.
active proctype auditor() {
    byte each;
    (finished == committers);
    assert(newest_value(published_stamp) == newest_value(commits));
    for (each : 0 .. committers - 1) { assert(stamp_of[each] <= newest_value(published_stamp)) }
}
#endif
