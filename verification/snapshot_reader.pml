/**
 *  `snapshot_store::reader_t` against the commits that prune behind it, over the steps of
 *  `commit_order_steps.pml`: the claim `take_snapshot` joins the census with, the reads it answers
 *  at that stamp under the partition's shared lock, the mark `republish_mark_` computes from the
 *  watermark and the occupied buckets, and the prune that frees everything that mark covers.
 *  Readers are counted per bucket here, never listed one by one, and a claim is a bucket index and
 *  a stamp rather than a pointer.
 *
 *  A commit draws its stamp outside the partition's lock, joins its version to the key's run under
 *  the lock, lands the stamp and walks the watermark, then prunes the run under the lock down to
 *  the mark its own landing computed. That mark is the freshest such a read can be, which is why
 *  the cached `low_water_mark_` word is left out; a commit whose walk moved nothing prunes at zero
 *  and frees nothing, which is what a stale read of that word would give.
 *
 *  Two scenarios.
 *  - `-Dscenario=reading`, the default: the reader takes a snapshot, reads the key twice at that
 *    stamp, and gives its claim back after the second read.
 *  - `-Dscenario=adoption`: a transaction adopts the reader's stamp through `share_snapshot`, one
 *    more member of the bucket the reader is already counted in; the reader gives its own claim
 *    back, and the transaction reads afterwards. Both run on the reader's thread, since
 *    `transaction_t` is constructed from the `reader_t` it adopts, so `commit_order.pml`'s
 *    `sharing` scenario is where a shared member crosses threads.
 *
 *  Invariants:
 *  - every read resolves the key to the version its stamp names, and that version is still there,
 *    because the bucket the claim joined carries a floor at or below that stamp and no mark passes
 *    an occupied bucket's floor. `-Dwithout_held_claim` gives the claim back before the reads
 *    rather than after them, and the commit that prunes next frees the version out from under the
 *    first of them;
 *  - the bucket is joined before the stamp is read, which is the reader's half of the pair
 *    `take_snapshot` keeps. `-Dwithout_join_first` reads the watermark first, and a mark computed
 *    in that gap counts nobody while standing above the stamp the reader is handed;
 *  - a mark reads the watermark before it scans the buckets, never after.
 *    `-Dwithout_watermark_first` scans first, so a reader joining after the scan is missed by it
 *    while the watermark read that follows has already passed the stamp that reader was handed. A
 *    second commit is what raises the watermark inside that gap, which is what `-Dcommitters=2`
 *    supplies;
 *  - a stamp adopted from a live claim stays pinned once that claim retires, because the adopting
 *    transaction is counted in the same bucket before the reader leaves it.
 *    `-Dwithout_shared_claim` copies the stamp alone, and the drain that follows lets the mark
 *    past it.
 *
 *  One commit by default: two of them under the view model do not finish inside the memory this
 *  suite gives a model, and only the third invariant asks for the second one, which
 *  `-Dcommitters=2` supplies under `-Dmemory=sequential` - the shape it turns on is an
 *  interleaving rather than a reordering, so sequential consistency is where it is cheapest.
 *
 *  `-Dwithout_bucket_retag`, which the shared steps also carry, is not claimed here, because this
 *  model passes under it: its one joiner reads its stamp after joining, and a floor an opener
 *  stores before shutting the bucket was computed before that read, so it still bounds every member
 *  the bucket can take. `commit_order.pml` is where the retag is checked.
 *
 *  The words, which are the nine the memory module allows and all of which this model spends:
 *  `partition_mutex` 0, `commits` 1, `published_stamp` 2, `landed_at(0)` 3, `head` 4,
 *  `members(0..1)` 5 and 6, and `floors(0..1)` 7 and 8. One ring slot is what two buckets leave
 *  room for, so commits publish in stamp order here and `commit_order.pml` is where a deeper ring
 *  is walked. Left out: the `low_water_mark_` word, as above; `generation_`, which dates
 *  transactions rather than their visibility; and the key's version run, which the partition's
 *  lock orders by itself and which is ghost state rather than a word.
 */
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

// The knob's values are integers, so a typo fails the range check below.
#define reading 1
#define adoption 2
#ifndef scenario
#define scenario reading
#endif
#if scenario < reading || scenario > adoption
#error "scenario is reading or adoption"
#endif

// How many commits write the key: one prunes against the reader, and two raise the watermark
// between the two reads a mark is computed from.
#ifndef committers
#define committers 1
#endif
#if committers < 1 || committers > 2
#error "committers is 1 or 2"
#endif

// The ring and the census, sized before the words are spelled over them.
#ifndef ring
#define ring 1
#endif
#ifndef buckets
#define buckets 2
#endif

// The threads, by role, apart from the processes that play them; the committers name themselves.
#define reader_thread committers

// The words: the partition's mutex, the stamp counter and the watermark, the ring of marks, and
// the census the readers are counted in.
#define partition_mutex 0
#define commits 1
#define published_stamp 2
#define landed_at(slot) (3 + (slot))
#define head (3 + ring)
#define members(bucket) (4 + ring + (bucket))
#define floors(bucket) (4 + ring + buckets + (bucket))

// The words have to fit the module's `location_count`, which every model here shares.
#if 4 + ring + 2 * buckets > location_count
#error "the mutex, the ring and the census together outgrow the module's words"
#endif

#include "commit_order_steps.pml"

// The key's version run is ordered by the partition's lock alone, so it is ghost state rather than
// words: one version per committer beside the one every snapshot starts on.
#define versions (committers + 1)
#define none 99
#define stamp_of(version) ((version) == 0 -> 0 : version_stamp[version])

byte committers_started;
int version_stamp[versions] = none; // the stamp each version was published under; version 0 is stamp 0
bool version_freed[versions];
byte versions_written = 1;

// visible_version_: the newest version whose stamp the snapshot covers, freed or not. The run is
// ghost state and the walk one step, since what a walk of it may see is the partition lock's business.
inline resolve(snapshot, found) {
    d_step {
        found = none;
        newest_stamp = -1;
        for (each : 0 .. versions - 1) {
            if
            :: stamp_of(each) != none && stamp_of(each) <= snapshot && stamp_of(each) > newest_stamp ->
                found = each; newest_stamp = stamp_of(each)
            :: else
            fi
        };
        each = 0
    }
}

// prune_key_of_: every version the mark covers but the one it resolves to, which masks the rest
inline prune(at_mark) {
    d_step {
        survivor = none;
        newest_stamp = -1;
        for (each : 0 .. versions - 1) {
            if
            :: stamp_of(each) != none && stamp_of(each) <= at_mark && stamp_of(each) > newest_stamp ->
                survivor = each; newest_stamp = stamp_of(each)
            :: else
            fi
        };
        for (each : 0 .. versions - 1) {
            if
            :: stamp_of(each) != none && stamp_of(each) <= at_mark && each != survivor ->
                version_freed[each] = true
            :: else
            fi
        };
        each = 0
    }
}

active [committers] proctype committer() {
    byte me, slot, each, scanned, survivor;
    int seen, observed, opened, least, newest_stamp, drawn, mark;
    bool exchanged, moved;
    atomic { me = committers_started; committers_started++ };
    // begin_commit: the stamp drawn outside the partition's lock, as the commit draws it
    draw_stamp(me, drawn);
    // publish_under: the version joins the key's run under the partition's lock
    lock(me, partition_mutex);
    atomic { slot = versions_written; versions_written++; version_stamp[slot] = drawn };
    unlock(me, partition_mutex);
    // end_commit: the mark landed, the watermark walked, and the low-water mark republished
    mark = 0;
    land_stamp(me, drawn);
    // prune_committed: at the mark this commit just computed
    lock(me, partition_mutex);
    prune(mark);
    unlock(me, partition_mutex)
}

active proctype reader() {
    byte reader_bucket, adopting_bucket, first, second, each, scanned, survivor;
    int seen, observed, opened, least, newest_stamp, mark, snapshot;
    bool exchanged;
#ifdef without_join_first
    // The watermark read before the bucket is joined, which is the pair in the other order
    read_modify_write(reader_thread, published_stamp, order_acq_rel, snapshot, snapshot);
    join_head(reader_thread, reader_bucket);
#else
    // take_snapshot: the head bucket joined, then the watermark read newest, and nothing re-read
    take_snapshot(reader_thread, reader_bucket, snapshot);
#endif

#if scenario == reading
#ifdef without_held_claim
    mark = 0;
    retire_snapshot(reader_thread, reader_bucket);
#endif
    // reader_t::find, twice, each under the partition's shared lock and neither writing anything
    lock_shared(reader_thread, partition_mutex);
    resolve(snapshot, first);
    assert(first != none && !version_freed[first]);
    unlock_shared(reader_thread, partition_mutex);
    lock_shared(reader_thread, partition_mutex);
    resolve(snapshot, second);
    assert(second == first && !version_freed[second]);
    unlock_shared(reader_thread, partition_mutex);
#ifndef without_held_claim
    mark = 0;
    retire_snapshot(reader_thread, reader_bucket);
#endif
#else
    // share_snapshot: the transaction counted in the reader's bucket while the reader is still in it
#ifdef without_shared_claim
    adopting_bucket = reader_bucket;
#else
    share_snapshot(reader_thread, reader_bucket, adopting_bucket);
#endif
    // the reader closes, leaving the transaction to pin the stamp both of them read at
    mark = 0;
    retire_snapshot(reader_thread, reader_bucket);
    lock_shared(reader_thread, partition_mutex);
    resolve(snapshot, first);
    assert(first != none && !version_freed[first]);
    unlock_shared(reader_thread, partition_mutex);
#ifndef without_shared_claim
    mark = 0;
    retire_snapshot(reader_thread, adopting_bucket);
#endif
#endif
    landed(reader_thread)
}
