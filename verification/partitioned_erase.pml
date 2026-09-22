/**
 *  A store-level window write spanning every partition of `partitioned_store`, as
 *  `publish_every_part_` in `include/smashtable/partitioned_store.hpp` makes one: every partition
 *  held exclusively for the whole span by `every_part_lock`, one stamp drawn by `begin_commit`
 *  once all of them staged, every partition's tombstone written under that one stamp, and the
 *  stamp landed once by `end_commit` after the last of them. `erase_range`, `erase_from`,
 *  `erase_up_to` and `update_range` all go out this way, and a transaction's own version of it is
 *  `transaction_t::commit_under_one_stamp_`. The order's steps are `commit_order_steps.pml`'s.
 *
 *  One eraser over two partitions, each holding one present key, and two observers of the pair.
 *  The reader is the pinned one: it takes a claim through `take_snapshot` and then reads each
 *  partition under that partition's shared lock, which is what a pinned reader and a transaction
 *  both do. The watcher holds nothing and claims nothing: it reads the watermark with an acquire
 *  load and then each tombstone relaxed, which is what shows that the one stamp rather than the
 *  mutual exclusion is what makes the window whole, since a reader that drew its snapshot inside
 *  the hold would see a consistent pair however the stamps were drawn.
 *
 *  No census. Nothing here prunes or reclaims, and a reader that only compares tombstone stamps
 *  against a snapshot needs the stamp alone, so this model defines `buckets` as zero: its claim is
 *  a bucket index nobody keeps and the stamp `take_snapshot` hands back, and `retire_snapshot`
 *  gives nothing back. What the buckets, the floors and the low-water mark buy a reader is
 *  `commit_order.pml`'s subject, and the two words per bucket they cost are what this model spends
 *  on a second partition instead.
 *
 *  Invariants:
 *  - the reader sees both keys erased or neither: its snapshot names the one stamp or does not;
 *  - the watcher sees the watermark cover both tombstones or neither, holding no partition at all,
 *    since the watermark reaches the stamp only once both tombstones carry it and the watermark's
 *    own read-modify-write carries them to whoever acquires it.
 *  `-Dwithout_one_stamp` draws and lands a stamp per partition, which is what each part's own
 *  publication does when called in turn, and both observers catch the window half erased.
 *
 *  The words, eight of the module's nine: `mutex(0..1)` 0 and 1, `tombstone(0..1)` 2 and 3,
 *  `commits` 4, `published_stamp` 5, and `landed_at(0..1)` 6 and 7. Left out: `head`, `members`
 *  and `floors`, since the model keeps no census; `low_water_mark_`, since nothing here prunes;
 *  `generation_`, which dates transactions rather than their visibility; and a second committer,
 *  so the ring is never contended here and what `end_commit` does when two commits land at once is
 *  `commit_order.pml`'s.
 */
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

// The threads, by role, apart from the processes that play them.
#define eraser_thread 0
#define reader_thread 1
#define watcher_thread 2
#define partitions 2

// The ring, and no census at all, sized before the words are spelled over them.
#ifndef ring
#define ring 2
#endif
#define buckets 0

// The words: each partition's mutex and its tombstone stamp, zero while the key is present, then
// the order's stamp counter, watermark and ring of landed marks.
#define mutex(partition) (partition)
#define tombstone(partition) (partitions + (partition))
#define commits (2 * partitions)
#define published_stamp (2 * partitions + 1)
#define landed_at(slot) (2 * partitions + 2 + (slot))

// The words have to fit the module's `location_count`, which every model here shares.
#if 2 * partitions + 2 + ring > location_count
#error "the partitions and the ring together outgrow the module's words"
#endif

#include "commit_order_steps.pml"

// publish_every_part_: every partition held ascending, the tombstones written, the stamp landed
active proctype eraser() {
    byte partition;
    int seen, observed, drawn;
    bool moved;
    // every_part_lock, ascending and exclusive, held from the first staging to the last prune
    lock(eraser_thread, mutex(0));
    lock(eraser_thread, mutex(1));
#ifdef without_one_stamp
    // each part's own publication, called in turn: a stamp drawn and landed per partition
    for (partition : 0 .. partitions - 1) {
        draw_stamp(eraser_thread, drawn);
        store(eraser_thread, tombstone(partition), order_relaxed, drawn);
        land_stamp(eraser_thread, drawn)
    };
#else
    // begin_commit once, every partition's tombstone written under that stamp, end_commit once
    draw_stamp(eraser_thread, drawn);
    for (partition : 0 .. partitions - 1) { store(eraser_thread, tombstone(partition), order_relaxed, drawn) };
    land_stamp(eraser_thread, drawn);
#endif
    unlock(eraser_thread, mutex(1));
    unlock(eraser_thread, mutex(0))
}

// A pinned reader: the claim taken outside the partitions, each partition read under its own share
active proctype reader() {
    byte partition, joined;
    int seen, snapshot, stamped;
    bool erased[partitions];
    // take_snapshot: the watermark read newest, which with no census is the whole claim
    take_snapshot(reader_thread, joined, snapshot);
    for (partition : 0 .. partitions - 1) {
        lock_shared(reader_thread, mutex(partition));
        load(reader_thread, tombstone(partition), order_relaxed, stamped);
        erased[partition] = (stamped != 0 && stamped <= snapshot);
        unlock_shared(reader_thread, mutex(partition))
    };
    // the whole window or none of it: the snapshot names the one stamp or it does not
    assert(erased[0] == erased[1]);
    // retire_snapshot: the claim given back, which with no census gives nothing back
    retire_snapshot(reader_thread, joined)
}

// The publication alone, with no partition held and no claim taken: what the one stamp buys a
// reader that the locks cannot
active proctype watcher() {
    byte partition;
    int watermark, stamped;
    bool covered[partitions];
    load(watcher_thread, published_stamp, order_acquire, watermark);
    for (partition : 0 .. partitions - 1) {
        load(watcher_thread, tombstone(partition), order_relaxed, stamped);
        covered[partition] = (stamped != 0 && stamped <= watermark)
    };
    assert(covered[0] == covered[1])
}
