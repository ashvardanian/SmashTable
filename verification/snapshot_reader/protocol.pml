/**
 *  @file verification/snapshot_reader/protocol.pml
 *  @author Ash Vardanian
 *  @date September 16, 2026
 *  @brief Spin model of @c snapshot_store::reader_t against the commits that prune behind it, over
 *      the steps of `commit_order_steps.pml`.
 *
 *  The claim @c take_snapshot joins the census with, the reads it answers at that stamp under the
 *  partition's shared lock, the mark @c republish_mark_ computes from the watermark and the
 *  occupied buckets, and the prune that frees everything that mark covers. Readers are counted
 *  per bucket here, never listed one by one, and a claim is a bucket index and a stamp rather
 *  than a pointer.
 *
 *  A commit draws its stamp and lands it with no partition lock held, which admits every
 *  interleaving the header's callers do, joins its version to the key's run under the lock, lands
 *  the stamp and walks the watermark, then prunes the run under the lock down to the mark its own
 *  landing computed. That mark is the freshest such a read can be, which is why the cached
 *  @c low_water_mark_ word is left out; a commit whose walk moved nothing prunes at zero
 *  and frees nothing, which is what a stale read of that word would give.
 *
 *  Every read resolves the key to the version its stamp names, and that version is still there,
 *  because the bucket the claim joined carries a floor at or below that stamp and no mark passes
 *  an occupied bucket's floor. A claim retires from a bucket that still counts it, which
 *  @c retire_snapshot asserts on the count its own decrement reads back, so a bucket's live count
 *  never borrows from its tag.
 *
 *  The words, all nine of them: @c partition_mutex 0, @c commits 1, @c published_stamp 2,
 *  `landed_at(0)` 3, @c head 4, `members(0..1)` 5 and 6, and `floors(0..1)` 7 and 8. One ring slot
 *  is what two buckets leave room for, so commits publish in stamp order here and
 *  `commit_order_publication.pml` is where a deeper ring is walked. Left out: the
 *  @c low_water_mark_ word, as above; @c generation_, which dates transactions rather than their
 *  visibility; and the key's version run, which the partition's lock orders by itself and which
 *  is ghost state rather than a word.
 *
 *  Every scenario beside this file defines its shape and @c committers, the number of commits that
 *  write the key behind the reader.
 */
#include "../weak_memory.pml"
#include "../spin_shared_mutex.pml"

#if committers < 1 || committers > 2
#error "committers is 1 or 2"
#endif

#define ring 1
#define buckets 2

/** The words: the partition's mutex, the stamp counter and the watermark, the ring of marks, and
 *  the census the readers are counted in. */
#define partition_mutex 0
#define commits 1
#define published_stamp 2
#define landed_at(slot) (3 + (slot))
#define head (3 + ring)
#define members(bucket) (4 + ring + (bucket))
#define floors(bucket) (4 + ring + buckets + (bucket))

/** The knobs: the header's choices, which `@verify` lines override to replay counterexamples. */
#ifndef join_first
#define join_first true
#endif
#ifndef held_claim
#define held_claim true
#endif
#ifndef shared_claim
#define shared_claim true
#endif

#include "../commit_order_steps.pml"

/** The key's version run is ordered by the partition's lock alone, so it is ghost state rather
 *  than words: one version per committer beside the one every snapshot starts on. */
#define versions (committers + 1)
#define none 99
#define stamp_of(version) ((version) == 0 -> 0 : version_stamp[version])

int version_stamp[versions] = none; // the stamp each version was published under; version 0 is stamp 0
bool version_freed[versions];
byte versions_written = 1;

/** The newest version whose stamp the snapshot covers, freed or not:
 *  @c snapshot_store::visible_version_. The run is ghost state and the walk one step, since what a
 *  walk of it may see is the partition lock's business. */
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

/** Every version the mark covers but the one it resolves to, which masks the rest:
 *  @c snapshot_store::prune_key_of_. */
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

/** A commit: draws its stamp, joins its version to the run, lands the stamp, and prunes at the
 *  mark it computed. */
inline commit(t) {
    byte slot, each, scanned, survivor;
    int seen, observed, opened, least, newest_stamp, drawn, mark;
    bool exchanged, moved;
    // `begin_commit`: the stamp drawn outside the partition's lock, as the commit draws it
    draw_stamp(t, drawn);
    // `publish_under`: the version joins the key's run under the partition's lock
    lock(t, partition_mutex);
    atomic { slot = versions_written; versions_written++; version_stamp[slot] = drawn };
    unlock(t, partition_mutex);
    // `end_commit`: the mark landed, the watermark walked, and the low-water mark republished
    mark = 0;
    land_stamp(t, drawn);
    if
    :: moved -> republish_mark(t, mark)
    :: else
    fi;
    // `prune_committed`: at the mark this commit just computed
    lock(t, partition_mutex);
    prune(mark);
    unlock(t, partition_mutex)
}

/** @c take_snapshot: the head bucket joined, then the watermark read newest, and nothing re-read;
 *  with @c join_first off, the watermark read before the bucket is joined. */
inline take_claim(t, joined, snapshot) {
    if
    :: join_first -> take_snapshot(t, joined, snapshot)
    :: else -> read_modify_write(t, published_stamp, order_acq_rel, snapshot, snapshot); join_head(t, joined)
    fi
}
