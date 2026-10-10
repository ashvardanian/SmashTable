/**
 *  @file verification/commit_order_steps.pml
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief @c basic_commit_order from `include/smashtable/shared.hpp` as inlines, shared by every
 *      model that publishes commits or counts readers.
 *
 *  Two halves stand here. The ring is what @c begin_commit draws a stamp from and @c end_commit
 *  lands it into, walking the watermark forward through the slots that already carry their stamp.
 *  The census is what @c take_snapshot, @c share_snapshot and @c retire_snapshot_ move a reader
 *  through: readers are counted per bucket rather than listed one by one, a bucket carries the
 *  watermark it was opened at as a floor for every member of it, and the mark is the least floor
 *  over the occupied buckets.
 *
 *  Include after `weak_memory.pml`, whose accesses every step here is spelled in, after
 *  `monitor_wait.pml`, whose futex-style wait the ring-room wait in @c land_stamp parks on, and
 *  after the model named its words and sized @c ring, which is @c ring_k. The words every model
 *  defines:
 *  - @c commits, the counter @c begin_commit adds to, which is @c commits_;
 *  - @c published_stamp, the watermark, which is @c published_stamp_;
 *  - @c landed_at of a slot in `0 .. ring - 1`, which is @c landed_ indexed by stamp.
 *
 *  A model with a census also sizes @c buckets, which is @c buckets_k, and defines:
 *  - @c head, the monotone bucket cursor, which is @c head_;
 *  - @c members of a bucket in `0 .. buckets - 1`, the tagged live count, which is @c snapshots_;
 *  - @c floors of a bucket in `0 .. buckets - 1`, the watermark each bucket was opened at.
 *
 *  A model with no census never expands the census inlines, so it names none of those words: its
 *  claim is the watermark's read-modify-write alone, and giving it back gives nothing back. One
 *  bucket is the head's own, so it has nowhere to rotate to and a reader handing over to the next
 *  pins the first one's floor for good; that is the configuration the `buckets_k >= 2` static
 *  assert refuses, and a model that wants to see it fail asks for it by name.
 *
 *  Every modelled access is its own step, since the step boundary is what the memory model orders.
 *  The purely local statement that reads an access's result shares that access's @c atomic instead
 *  of standing on a step of its own, so a thread offers the schedule one point per access rather
 *  than one point per line; nothing an access can observe moves across such a boundary.
 *
 *  Every inline uses the caller's scratch, and a caller declares what its own calls reach:
 *  @c draw_stamp uses @c seen; @c land_stamp uses @c seen, @c observed and @c moved;
 *  @c take_snapshot uses @c observed, @c opened and @c exchanged; @c retire_snapshot uses
 *  @c observed, @c exchanged and @c mark; @c republish_mark uses @c observed, @c opened, @c least,
 *  @c scanned and @c exchanged; @c share_snapshot uses none. The full set is
 *  `int seen, observed, opened, least, mark`, `byte scanned` and `bool exchanged, moved`.
 *
 *  The knobs, each set on the `@verify` lines of the scenarios whose assertions it breaks:
 *  - @c ring_check, the wait in @c end_commit before a ring slot is reused;
 *  - @c watermark_notify, the wake @c end_commit sends once its walk moved the watermark;
 *  - @c done_order, the order the landed mark is stored with;
 *  - @c advance_read_modify_write, the read-modify-write that reads the watermark before the walk;
 *  - @c walk_order, the order the walk loads each next slot's mark with;
 *  - @c take_read_modify_write, the read-modify-write @c take_snapshot reads the watermark through;
 *  - @c watermark_first, the watermark read before the buckets are scanned;
 *  - @c bucket_retag, a bucket shut to arrivals before its floor is stored;
 *  - @c head_watermark, the head bucket sealed against the watermark rather than the mark;
 *  - @c share_order, the order the shared member is posted with.
 *
 *  Left out: @c low_water_mark_, since a model acts on the mark it just computed, which is the
 *  freshest such a read can be, and the word is only ever raised with a maximum and only ever read
 *  as a lower bound; @c generation_ and @c next_generation, which date transactions rather than
 *  their visibility and order nothing here; @c await_published, whose wait on the watermark is the
 *  ring-room wait's shape on the same word and the same wake; and the @c solitary_k path, which has
 *  no second thread to order against.
 */
#if ring < 1
#error "ring is one or more"
#endif

/** The knobs: the header's choices, which `@verify` lines override to replay counterexamples. */
#ifndef ring_check
#define ring_check true
#endif
#ifndef watermark_notify
#define watermark_notify true
#endif
#ifndef done_order
#define done_order order_release
#endif
#ifndef advance_read_modify_write
#define advance_read_modify_write true
#endif
#ifndef walk_order
#define walk_order order_acquire
#endif
#ifndef take_read_modify_write
#define take_read_modify_write true
#endif
#ifndef watermark_first
#define watermark_first true
#endif
#ifndef bucket_retag
#define bucket_retag true
#endif
#ifndef head_watermark
#define head_watermark true
#endif
#ifndef share_order
#define share_order order_release
#endif

/** The head value that opened a bucket sits above its live count in one word, as @c snapshots_
 *  packs a tag over a count; eight leaves room for every count and every head value a model
 *  this size reaches. */
#define census_scale 8
#define tag_of(word) ((word) / census_scale)
#define live_of(word) ((word) % census_scale)
#define tagged(opened) (((opened) % census_scale) * census_scale)

/** One below the count the next tag starts at, so the add a joiner makes cannot carry
 *  into the tag. */
#define ceiling_of(opened) (tagged(opened) + census_scale - 2)
#define bucket_of(opened) ((opened) % buckets)
#define no_floor (-1)

/** Raises a word to a floor, never lowering it, which is @c atomic_max_fetch: release only, and
 *  read relaxed, as the header spells them, since every invariant holds at those orders. */
inline raise_word(t, word, floor_value) {
    read_modify_write_if(t, word, order_release, observed < floor_value, observed, floor_value)
}

/** Models @c basic_commit_order::record_snapshot_ in `shared.hpp`: one member more under a bounded
 *  add, and the add succeeding is the whole answer. The ceiling refuses only a tag above the head
 *  just read, which a bucket recycled ahead of this thread carries; a tag below it is a bucket this
 *  thread shares, whose older floor still bounds it. */
inline join_head(t, joined) {
    do
    :: atomic { load(t, head, order_relaxed, opened); joined = bucket_of(opened) };
       atomic {
           read_modify_write_if(t, members(joined), order_acq_rel, observed <= ceiling_of(opened), observed,
                                observed + 1);
           if
           :: observed <= ceiling_of(opened) -> break
           :: else
           fi
       }
    od
}

/** Models @c basic_commit_order::open_next_bucket_ in `shared.hpp`: the bucket the next head names,
 *  shut to arrivals first, then floored, then the head raised. The retag is what makes the floor
 *  store safe: nobody can be counted in the bucket while its floor is being replaced. A bucket
 *  already carrying the tag this open would write was opened from the same head value by somebody
 *  else, whose floor store this one would only race. */
inline open_next_bucket(t, opened_head, watermark) {
    atomic {
        load(t, members(bucket_of(opened_head + 1)), order_acquire, observed);
        exchanged = live_of(observed) == 0 && tag_of(observed) != ((opened_head + 1) % census_scale)
    };
    if
    :: exchanged && bucket_retag ->
        compare_exchange(t, members(bucket_of(opened_head + 1)), order_acq_rel, observed,
                         tagged(opened_head + 1), exchanged);
        if
        :: exchanged ->
            store(t, floors(bucket_of(opened_head + 1)), order_relaxed, watermark);
            raise_word(t, head, opened_head + 1)
        :: else
        fi
    :: exchanged && !bucket_retag ->
        store(t, floors(bucket_of(opened_head + 1)), order_relaxed, watermark);
        compare_exchange(t, members(bucket_of(opened_head + 1)), order_acq_rel, observed,
                         tagged(opened_head + 1), exchanged);
        if
        :: exchanged -> raise_word(t, head, opened_head + 1)
        :: else
        fi
    :: else
    fi;
    exchanged = false
}

/** The least floor over the occupied buckets, or @c no_floor where nobody is counted anywhere. */
inline scan_floors(t, least_floor) {
    least_floor = no_floor;
    for (scanned : 0 .. buckets - 1) {
        atomic { load(t, members(scanned), order_relaxed, observed); exchanged = live_of(observed) != 0 };
        if
        :: exchanged ->
            atomic {
                load(t, floors(scanned), order_relaxed, observed);
                if
                :: least_floor == no_floor || observed < least_floor -> least_floor = observed
                :: else
                fi
            }
        :: else
        fi
    };
    scanned = 0;
    exchanged = false
}

/** Models @c basic_commit_order::republish_mark_ in `shared.hpp`: the watermark read newest first,
 *  then the buckets, so a reader joining after that read draws at or above it and one joining
 *  before it is counted by the scan. @p oldest_needed carries the watermark until the head is
 *  sealed and only then falls to the least floor, since the seal is against what is published and
 *  never against what is needed: a head bucket holding the least floor equals the mark by
 *  construction, so comparing those two would leave the head where it is and pin every later reader
 *  to a floor it has no reason to hold. */
inline republish_mark(t, oldest_needed) {
    if
    :: watermark_first ->
        read_modify_write(t, published_stamp, order_acq_rel, oldest_needed, oldest_needed);
        load(t, head, order_relaxed, opened);
        scan_floors(t, least)
    :: else ->
        scan_floors(t, least);
        load(t, head, order_relaxed, opened);
        read_modify_write(t, published_stamp, order_acq_rel, oldest_needed, oldest_needed)
    fi;
    // The seal: against the watermark, or without `head_watermark`, against the mark lowered first
    atomic {
        if
        :: !head_watermark && least != no_floor && least < oldest_needed -> oldest_needed = least
        :: else
        fi;
        load(t, floors(bucket_of(opened)), order_relaxed, observed);
        exchanged = observed < oldest_needed
    };
    if
    :: exchanged -> open_next_bucket(t, opened, oldest_needed)
    :: else
    fi;
    if
    :: least != no_floor && least < oldest_needed -> oldest_needed = least
    :: else
    fi
}

/** Models @c basic_commit_order::share_snapshot in `shared.hpp`: one more member of the bucket a
 *  live claim already holds, posted with release and never read back. The claim being live is what
 *  leaves the joiner nothing to re-read: its bucket carries a member, so it cannot have been
 *  recycled, and its floor already stands at or below the snapshot being copied. */
inline share_snapshot(t, held_bucket, joined) {
    atomic { add_no_return(t, members(held_bucket), share_order, 1); joined = held_bucket }
}

/** Models @c basic_commit_order::retire_snapshot_ in `shared.hpp`: one member fewer, released, and
 *  the mark recomputed only where the bucket drained, so the scan is spent on that departure alone.
 *  Where it did not drain, @c mark is left as the caller set it. */
inline retire_snapshot(t, joined) {
    atomic {
        read_modify_write(t, members(joined), order_release, observed, observed - 1);
        assert(live_of(observed) != 0);
        exchanged = live_of(observed) == 1
    };
    if
    :: exchanged -> republish_mark(t, mark)
    :: else
    fi
}

/** Models @c basic_commit_order::begin_commit in `shared.hpp`: one unconditional add, so a draw
 *  never waits and never refuses. */
inline draw_stamp(t, drawn) {
    atomic { read_modify_write(t, commits, order_relaxed, seen, seen + 1); drawn = seen + 1 }
}

/** Models @c basic_commit_order::end_commit in `shared.hpp`: the slot this stamp is about to reuse
 *  waited for, parked on the watermark between two loads, the mark released into it, the watermark
 *  read newest, the ring walked while the next slot carries the next stamp, and the parked
 *  committers woken where the walk moved the watermark. A model with a census follows it with
 *  @c republish_mark where @c moved says so, which is the rest of @c end_commit. */
inline land_stamp(t, drawn) {
    // Waited here rather than at the draw, so the write happened during the overlap rather than
    // after it; the slot still holds `drawn - ring`, and overwriting that mark loses it for good.
    if
    :: ring_check ->
        do
        :: atomic {
               load(t, published_stamp, order_acquire, seen);
               if
               :: drawn - seen > ring -> skip
               :: else -> break
               fi
           };
           wait_on(t, published_stamp, seen, false)
        od
    :: else
    fi;
    // The slot's previous stamp is one the watermark already passed, or its mark would be lost.
    atomic {
        assert(drawn - newest_value(published_stamp) <= ring);
        store(t, landed_at(drawn % ring), done_order, drawn)
    };
    // Reading the newest, not merely loading: a plain load could miss a neighbour's mark and leave
    // ours for the next commit to publish.
    atomic {
        if
        :: advance_read_modify_write -> read_modify_write(t, published_stamp, order_acq_rel, seen, seen)
        :: else -> load(t, published_stamp, order_acquire, seen)
        fi;
        moved = false
    };
    do
    :: atomic {
           load(t, landed_at((seen + 1) % ring), walk_order, observed);
           if
           :: observed != seen + 1 -> break
           :: else
           fi
       };
       atomic {
           read_modify_write_if(t, published_stamp, order_acq_rel, observed < seen + 1, observed, seen + 1);
           if
           :: observed < seen + 1 ->
               moved = true;
               seen = seen + 1;
               // The wake `end_commit` sends after its walk, sent with each step: a waiter that
               // arms in between re-checks the word, so the gap loses nothing
               if
               :: watermark_notify -> wake(published_stamp)
               :: else
               fi
           :: else -> seen = observed
           fi
       }
    od
}

/** Models @c basic_commit_order::take_snapshot in `shared.hpp` over a census: the bucket joined,
 *  then the watermark read newest, which is the reader's half of the pair: a mark computed after
 *  this counts the member above, and one computed before it stands at or below the stamp returned.
 *  There is no re-read: the bucket's floor is what pins retention. */
inline take_snapshot(t, joined, snapshot) {
    join_head(t, joined);
    if
    :: take_read_modify_write -> read_modify_write(t, published_stamp, order_acq_rel, snapshot, snapshot)
    :: else -> load(t, published_stamp, order_acquire, snapshot)
    fi
}
