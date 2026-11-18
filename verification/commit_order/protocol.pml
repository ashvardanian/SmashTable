/**
 *  @file verification/commit_order/protocol.pml
 *  @author Ash Vardanian
 *  @date September 22, 2026
 *  @brief The census of @c basic_commit_order: the buckets readers join, the floors they carry, and
 *      the mark, from `include/smashtable/shared.hpp` under the steps of `commit_order_steps.pml`.
 *
 *  The stamps @c begin_commit draws from @c commits_, the word @c end_commit flips a bit of and
 *  raises the watermark from, the buckets @c take_snapshot joins and @c retire_snapshot_
 *  leaves, the member @c share_snapshot posts onto a claim that is still live, and the mark
 *  @c republish_mark_ computes from the watermark and the occupied buckets. The commit order
 *  carries no mutex at all, and three orderings stand in its place.
 *
 *  - The mark never passes a snapshot a live claim names, which @c record_mark checks.
 */
#define ring 2

/** One or zero, as <tt>commit_order_steps.pml</tt> asks: it lays out the words below. */
#ifndef parity_word
#define parity_word 1
#endif
#if parity_word
#define word_bits 4
#define window ring
#define ring_words 1
#else
#define ring_words ring
#endif
#include <weak_memory.pml>
#include <monitor_wait.pml>

/** The words: the stamp counter, the watermark, the word of landings or the ring of marks, and the
 *  census readers join. */
#define commits 0
#define published_stamp 1
#if parity_word
#define landed_parity 2
#else
#define landed_at(slot) (2 + (slot))
#endif
#define head (2 + ring_words)
#define members(bucket) (3 + ring_words + (bucket))
#define floors(bucket) (3 + ring_words + buckets + (bucket))

/** The knob: how many floors the census keeps, which is @c buckets_k. */
#ifndef buckets
#define buckets 2
#endif
#if buckets < 1
#error "buckets is one or more"
#endif

#include <commit_order_steps.pml>

byte finished;          // the commits that finished
bool reader_settled;    // the reader holds a claim, so every mark recorded now is checked against it
int reader_snapshot;    // the stamp that claim reads at
int freed_below;        // the newest mark anyone acted on
bool reader_gone;       // the claim a second one took over from has been given back
bool head_rotated;      // the head has left the bucket it opened at, so the next rotation recycles one
byte reader_bucket;     // the bucket the sharer posts its own member onto
bool sharer_settled;    // the sharer holds the claim it copied
int sharer_snapshot;    // the stamp that claim reads at, copied rather than re-read
bool shared_taken;      // the share has been posted, so the claim it came off may retire
bool successor_settled; // the successor holds a claim of its own
int successor_snapshot; // the stamp that claim reads at
bool successor_taken;   // the successor has joined, so the claim it took over from may retire

/** Every mark anyone computes, checked against the claims that were settled when it was recorded
 *  and kept as the newest mark anyone acted on. A mark computed before a claim settled is at or
 *  below that claim's snapshot as well, since a claim reads the watermark only after it joins, so
 *  recording one step later than it was computed admits no failure of its own. */
inline record_mark(computed) {
    atomic {
        assert(!reader_settled || computed <= reader_snapshot);
        assert(!sharer_settled || computed <= sharer_snapshot);
        assert(!successor_settled || computed <= successor_snapshot);
        if
        :: computed > freed_below -> freed_below = computed
        :: else
        fi
    }
}

/** One commit: @c begin_commit draws the stamp, @c end_commit lands the mark, walks the watermark
 *  in stamp order and republishes the low-water mark where the walk moved it. */
inline commit(t) {
    atomic { draw_stamp(t, drawn); mark = 0 };
    land_stamp(t, drawn);
    if
    :: moved -> republish_mark(t, mark)
    :: else
    fi
}
