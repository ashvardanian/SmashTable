/**
 *  @file verification/partitioned_erase.pml
 *  @author Ash Vardanian
 *  @date September 16, 2026
 *  @brief A store-level window write spanning every partition of @c partitioned_store under one
 *      stamp, as @c publish_every_part_ in `include/smashtable/partitioned_store.hpp` makes one.
 *
 *  Every partition is held exclusively for the whole span by @c every_part_lock, one stamp drawn by
 *  @c begin_commit once all of them staged, every partition's tombstone written under that one
 *  stamp, and the stamp landed once by @c end_commit after the last of them. @c erase_range,
 *  @c erase_from, @c erase_up_to and @c update_range all go out this way, and a transaction's own
 *  version of it is @c transaction_t::commit_under_one_stamp_. The order's steps are
 *  `commit_order_steps.pml`'s.
 *
 *  One eraser over two partitions, each holding one present key, and two observers of the pair. The
 *  reader is the pinned one: it takes a claim through @c take_snapshot and then reads each
 *  partition under that partition's shared lock, which is what a pinned reader and a transaction
 *  both do. The watcher holds nothing and claims nothing: it reads the watermark with an acquire
 *  load and then each tombstone relaxed, which is what shows that the one stamp rather than the
 *  mutual exclusion is what makes the window whole, since a reader that drew its snapshot inside
 *  the hold would see a consistent pair however the stamps were drawn.
 *
 *  No census. Nothing here prunes or reclaims, and a reader that only compares tombstone stamps
 *  against a snapshot needs the stamp alone, so its claim is the stamp the watermark's
 *  read-modify-write hands back, and giving it back gives nothing back. What the buckets, the
 *  floors and the low-water mark buy a reader is `commit_order/`'s subject, and the two words per
 *  bucket they cost are what this model spends on a second partition instead.
 *
 *  Invariants:
 *  - the reader sees both keys erased or neither: its snapshot names the one stamp or does not;
 *  - the watcher sees the watermark cover both tombstones or neither, holding no partition at all,
 *    since the watermark reaches the stamp only once both tombstones carry it and the watermark's
 *    own read-modify-write carries them to whoever acquires it.
 *
 *  Left out: @c low_water_mark_, since nothing here prunes; @c generation_, which dates
 *  transactions rather than their visibility; and a second committer, so the ring is never
 *  contended here and what @c end_commit does when two commits land at once is
 *  `commit_order_publication.pml`'s.
 *
 *  @verify pass sc,rc11,far
 *  @verify fail sc,rc11,far one_stamp=false: @c publish_every_part_ publishes every staged
 *      partition under one stamp; with a stamp per partition, as each part's own publication draws,
 *      a reader drawing its snapshot from the clock between two publications names the first
 *      partition's stamp and not the second's, and both observers see the window half erased
 */
#define partitions 2
#define ring 2
#define thread_count 3
#define location_count (2 * partitions + 2 + ring)
#define history_depth 6
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

/** The words: each partition's mutex and its tombstone stamp, zero while the key is present, then
 *  the order's stamp counter, watermark and ring of landed marks. */
#define mutex(partition) (partition)
#define tombstone(partition) (partitions + (partition))
#define commits (2 * partitions)
#define published_stamp (2 * partitions + 1)
#define landed_at(slot) (2 * partitions + 2 + (slot))

/** The knob: the header's choice, which a `@verify` line overrides to replay its counterexample. */
#ifndef one_stamp
#define one_stamp true
#endif

#include "commit_order_steps.pml"

/** Every partition held ascending, the tombstones written, the stamp landed:
 *  @c partitioned_store::publish_every_part_. */
proctype eraser(byte t) {
    byte partition;
    int seen, observed, drawn;
    bool moved;
    // `every_part_lock`, ascending and exclusive, held from the first staging to the last prune
    lock(t, mutex(0));
    lock(t, mutex(1));
    if
    :: one_stamp ->
        // `begin_commit` once, every tombstone written under that stamp, `end_commit` once
        draw_stamp(t, drawn);
        for (partition : 0 .. partitions - 1) { store(t, tombstone(partition), order_relaxed, drawn) };
        land_stamp(t, drawn)
    :: else ->
        // Each part's own publication, called in turn: a stamp drawn and landed per partition
        for (partition : 0 .. partitions - 1) {
            draw_stamp(t, drawn);
            store(t, tombstone(partition), order_relaxed, drawn);
            land_stamp(t, drawn)
        }
    fi;
    unlock(t, mutex(1));
    unlock(t, mutex(0))
}

/** A pinned reader: the claim taken outside the partitions, each partition read under
 *  its own share. */
proctype reader(byte t) {
    byte partition;
    int seen, snapshot, stamped;
    bool erased[partitions];
    // `take_snapshot`: the watermark read newest, which with no census is the whole claim
    read_modify_write(t, published_stamp, order_acq_rel, snapshot, snapshot);
    for (partition : 0 .. partitions - 1) {
        lock_shared(t, mutex(partition));
        load(t, tombstone(partition), order_relaxed, stamped);
        erased[partition] = (stamped != 0 && stamped <= snapshot);
        unlock_shared(t, mutex(partition))
    };
    // The whole window or none of it: the snapshot names the one stamp or it does not
    assert(erased[0] == erased[1])
}

/** The publication alone, with no partition held and no claim taken: what the one stamp buys a
 *  reader that the locks cannot. */
proctype watcher(byte t) {
    byte partition;
    int watermark, stamped;
    bool covered[partitions];
    load(t, published_stamp, order_acquire, watermark);
    for (partition : 0 .. partitions - 1) {
        load(t, tombstone(partition), order_relaxed, stamped);
        covered[partition] = (stamped != 0 && stamped <= watermark)
    };
    assert(covered[0] == covered[1])
}

init { atomic { run eraser(0); run reader(1); run watcher(2) } }
