/**
 *  @file verification/partitioned_store.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief A cursor of @c partitioned_store stepping past a store-wide write, and a commit over one
 *      partition beside them, from `include/smashtable/partitioned_store.hpp`.
 *
 *  The writer is @c publish_every_part_: every partition held exclusively by @c every_part_lock,
 *  one stamp drawn by @c begin_commit, a key inserted into the first partition and a version
 *  written into the second under it, the stamp landed by @c end_commit, every partition given back,
 *  and only then each partition's write count stepped with a release by @c note_written_, which
 *  @c for_all does the same way after an all-at-once hold. A flag the writer releases after the
 *  counts tells the cursor the write is over.
 *
 *  The cursor is @c ordered_cursor_t: its constructor reads each partition's count with acquire
 *  through @c epoch_seen_ and then that partition's front under its shared lock, which is
 *  @c seed_every_front_; it then acquires the flag, and @c next re-reads every front whose count
 *  moved since it was read, which is @c refresh_stale_fronts_, and hands over the least. The
 *  writer's key is the least of all, so a step taken after the cursor saw the flag owes that key.
 *  The writes insert and never erase, so the front handed over is still there and the probe that
 *  confirms it and refills behind it changes nothing this step hands over.
 *
 *  The committer is a transaction over the second partition alone: @c touched_parts_lock_t holds it
 *  from @c validate_for_commit through the write, the stamp drawn and landed in between, and its
 *  release steps the count before it unlocks. It validates the version the writer also writes, and
 *  nothing may move that version between its validation and its write. Both commits draw and land
 *  through the ring steps of `commit_order_steps.pml`, with no census, since nothing here reads at
 *  a snapshot; the second partition's lock serializes them, so the ring is walked in turn.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc,rc11 note_written=false: @c publish_every_part_ steps every partition's write
 *      count once its hold is over; without the step, the cursor that read the first partition's
 *      front before the insert never re-reads it, and hands over the second-least key after the
 *      writer is done
 *  @verify pass rc11 epoch_order=order_relaxed: the cursor reads a front under the partition's
 *      shared lock, whose read-modify-write reads the writer's unlock or a later write, so a cursor
 *      that read the stepped count reads the front after the insert, released by the step or not
 *  @verify pass rc11 front_lock=false: the count is stepped with a release after the insert and
 *      read with acquire before the front, so a cursor that read the stepped count reads the front
 *      after the insert even without the lock
 *  @verify fail rc11 front_lock=false epoch_order=order_relaxed: the shared lock and the count's
 *      release each order the front read after the insert alone; with neither, the constructor
 *      reads the stepped count and a front from before the insert, and @c next re-reads nothing
 *  @verify fail sc,rc11 held_partitions=false: a commit holds every partition it reached from its
 *      validation through its write; giving the partition back between the two and taking it
 *      again, the writer writes the version the committer validated in the gap, and the commit
 *      lands over a write it never saw
 */
#define partitions 2
#define ring 2
#define thread_count 3
#define location_count 12
#define history_depth 11
#include "weak_memory.pml"
#include "monitor_wait.pml"
#include "spin_shared_mutex.pml"

/** The words: each partition's mutex, write count and keys, the version the committer validates,
 *  the writer's flag, then the order's stamp counter, watermark and ring of landed marks. */
#define mutex(partition) (partition)
#define epoch(partition) (partitions + (partition))
#define keys(partition) (2 * partitions + (partition))
#define version (3 * partitions)
#define done (3 * partitions + 1)
#define commits (3 * partitions + 2)
#define published_stamp (3 * partitions + 3)
#define landed_at(slot) (3 * partitions + 4 + (slot))

/** The keys, one bit each: the first partition holds two, the second four, and the writer inserts
 *  one, the least of all. */
#define bit(key) (1 << (key))
#define inserted 1
#define absent 9
#define least_of(set) \
    (((set) & bit(1)) -> 1 : (((set) & bit(2)) -> 2 : (((set) & bit(3)) -> 3 : (((set) & bit(4)) -> 4 : absent))))

/** The knobs: the header's choices, which `@verify` lines override to replay counterexamples. */
#ifndef note_written
#define note_written true
#endif
#ifndef front_lock
#define front_lock true
#endif
#ifndef epoch_order
#define epoch_order order_release
#endif
#ifndef held_partitions
#define held_partitions true
#endif

#include "commit_order_steps.pml"

/** One partition's keys read under its shared lock: the probe of @c seed_one_front_. */
inline read_front(t, partition, contents) {
    if
    :: front_lock -> lock_shared(t, mutex(partition))
    :: else
    fi;
    load(t, keys(partition), order_relaxed, contents);
    if
    :: front_lock -> unlock_shared(t, mutex(partition))
    :: else
    fi
}

/** @c publish_every_part_: every partition held ascending, the key and the version written under
 *  one stamp, the partitions given back, then every count stepped and the flag released. */
proctype writer(byte t) {
    byte partition;
    int seen, observed, drawn;
    bool moved;
    lock(t, mutex(0));
    lock(t, mutex(1));
    draw_stamp(t, drawn);
    store(t, keys(0), order_relaxed, bit(2) | bit(inserted));
    store(t, version, order_relaxed, drawn);
    land_stamp(t, drawn);
    unlock(t, mutex(1));
    unlock(t, mutex(0));
    if
    :: note_written -> for (partition : 0 .. partitions - 1) { add_no_return(t, epoch(partition), epoch_order, 1) }
    :: else
    fi;
    store(t, done, order_release, 1)
}

/** @c ordered_cursor_t: every front seeded at the count read before it, the flag acquired, then
 *  one @c next over the fronts whose count moved. */
proctype cursor(byte t) {
    byte partition, handed;
    byte front[partitions];
    int seen, written, contents, saw_done;
    int seen_epoch[partitions];
    for (partition : 0 .. partitions - 1) {
        load(t, epoch(partition), order_acquire, written);
        read_front(t, partition, contents);
        front[partition] = least_of(contents);
        seen_epoch[partition] = written
    };
    load(t, done, order_acquire, saw_done);
    for (partition : 0 .. partitions - 1) {
        load(t, epoch(partition), order_acquire, written);
        if
        :: written != seen_epoch[partition] ->
            read_front(t, partition, contents);
            front[partition] = least_of(contents);
            seen_epoch[partition] = written
        :: else
        fi
    };
    handed = (front[0] < front[1] -> front[0] : front[1]);
    // An insert the cursor saw finish, ahead of where it stands, is handed over
    assert(!saw_done || handed == inserted)
}

/** A transaction over the second partition: validated, stamped, written and landed under one hold,
 *  its count stepped before the hold goes. */
proctype committer(byte t) {
    int seen, observed, drawn, validated;
    bool moved;
    lock(t, mutex(1));
    load(t, version, order_relaxed, validated);
    if
    :: !held_partitions -> unlock(t, mutex(1)); lock(t, mutex(1))
    :: else
    fi;
    draw_stamp(t, drawn);
    // `publish_under`: nothing moved since the validation, or the commit lands over a write unseen
    assert(newest_value(version) == validated);
    store(t, version, order_relaxed, drawn);
    land_stamp(t, drawn);
    add_no_return(t, epoch(1), epoch_order, 1);
    unlock(t, mutex(1))
}

init {
    atomic {
        seed(keys(0), bit(2));
        seed(keys(1), bit(4));
        run writer(0);
        run cursor(1);
        run committer(2)
    }
}
