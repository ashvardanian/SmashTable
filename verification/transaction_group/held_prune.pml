/**
 *  @file verification/transaction_group/held_prune.pml
 *  @author Ash Vardanian
 *  @date November 18, 2026
 *  @brief Two groups on one @c basic_commit_order holding their stores from the validation through
 *      the prune, beside a committer on a partition of its own and a shared reader.
 *
 *  A group validates a @c locked_store and then a @c partitioned_store, ascending, draws one stamp,
 *  publishes both, lands the stamp, awaits its publication only where a fresh snapshot is behind
 *  it, and prunes; with @c nested its one participant is a locked store over a partitioned one.
 *  The committer is @c partitioned_store::transaction_t::publish_under over a partition no group
 *  reaches: it lands under its hold and awaits its own publication after giving the hold up, which
 *  is the strict rung. The order is the ring of <tt>commit_order_steps.pml</tt> with no census.
 *
 *  - No group waits for good: every commit older than an awaiting group lands without a lock that
 *    group holds, and the oldest never waits for ring room.
 *  - A group reopens at a snapshot covering its own stamp.
 *  - The reader sees each group whole: a snapshot covering a group's stamp finds it in every store.
 *
 *  @budget 8192
 *  @verify pass sc
 *  @verify pass sc nested=true
 *  @verify pass rc11 second_group=false: one group, since two overflow the budget under views
 *  @verify pass rc11 nested=true second_group=false
 *  @verify pass sc blind_entry=false
 *  @verify pass sc,rc11 nested=true second_group=false blind_entry=false
 *  @verify pass sc ring=1: every commit past the oldest waits for ring room under its holds
 *  @verify pass sc nested=true ring=1
 *  @verify fail sc parity_word=0 ring=1 ring_check=false: a group lands over the committer's mark
 *      in the ring of stamps, so the wait for room under the holds is reached
 *  @verify pass sc locked_hold_through_prune=false partition_hold_through_prune=false: both stores
 *      giving their holds up as they publish, each prune takes its store back holding nothing
 *  @verify stuck sc locked_hold_through_prune=false: the locked store given up at its publication,
 *      the other group validates it and waits on the partition, which the first group still holds
 *      while its prune takes the locked store back
 *  @verify stuck sc nested=true second_group=false locked_hold_through_prune=false: the reader
 *      inside the locked store's shared lock waits on the partition, and the prune that would give
 *      the partition back waits on the reader
 *  @verify fail sc await_when_behind=false: landing ahead of the committer's older stamp, the
 *      group reopens at a snapshot below its own stamp
 */
#ifndef ring
#define ring 2
#endif

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
#define thread_count 4
#define location_count (7 + ring_words)
#define history_depth 16
#include <weak_memory.pml>
#include <monitor_wait.pml>
#include <spin_shared_mutex.pml>

/** The words: the locked store's mutex, the partition the groups reach and the one the committer
 *  owns, the order's stamp counter and watermark, the newest stamp each store holds, the ring. */
#define locked 0
#define partition 1
#define mutex(reached) (reached)
#define commits 3
#define published_stamp 4
#define stamp_at(reached) (5 + (reached))
#if parity_word
#define landed_parity 7
#else
#define landed_at(slot) (7 + (slot))
#endif

/** The knobs: the header's choices, which <tt>@verify</tt> lines override to replay counterexamples. */
#ifndef locked_hold_through_prune
#define locked_hold_through_prune true
#endif
#ifndef partition_hold_through_prune
#define partition_hold_through_prune true
#endif
#ifndef await_when_behind
#define await_when_behind true
#endif
#ifndef nested
#define nested false
#endif
#ifndef second_group
#define second_group true
#endif

#include <commit_order_steps.pml>

int stamp_of[2]; // the stamp each group drew, zero before it drew one

/** @c basic_commit_order::await_published in <tt>shared.hpp</tt>: parked on the watermark between two
 *  loads, woken by the walk that moves it. */
inline await_published(t, drawn) {
    do
    :: atomic {
           load(t, published_stamp, order_acquire, seen);
           if
           :: seen < drawn -> skip
           :: else -> break
           fi
       };
       wait_on(t, published_stamp, seen, false)
    od
}

/** The partition's half of a group commit: @c partitioned_store::transaction_t::publish_under of a
 *  stamp, and @c prune_committed after it, in <tt>partitioned_store.hpp</tt>. */
inline publish_partition(t, drawn) {
    store(t, stamp_at(partition), order_relaxed, drawn);
    if
    :: !partition_hold_through_prune -> unlock(t, mutex(partition))
    :: else
    fi
}
inline prune_partition(t) {
    if
    :: !partition_hold_through_prune -> lock(t, mutex(partition))
    :: else
    fi;
    unlock(t, mutex(partition))
}

/** @c transaction_group::commit in <tt>shared.hpp</tt>, where the participants share one order. */
proctype group(byte t) {
    byte landed_run;
    int seen, observed, drawn, snapshot;
    bool moved;
    // `validate_participants_`: every hold taken, ascending, before the stamp is drawn
    lock(t, mutex(locked));
    lock(t, mutex(partition));
    atomic { draw_stamp(t, drawn); stamp_of[t] = drawn };
    // `locked_store::transaction_t::publish_under` of a stamp, around the partition's where nested
    if
    :: nested -> publish_partition(t, drawn)
    :: else -> store(t, stamp_at(locked), order_relaxed, drawn)
    fi;
    if
    :: !locked_hold_through_prune -> unlock(t, mutex(locked))
    :: else
    fi;
    if
    :: !nested -> publish_partition(t, drawn)
    :: else
    fi;
    land_stamp(t, drawn);
    read_modify_write(t, published_stamp, order_acq_rel, snapshot, snapshot);
    if
    :: await_when_behind && snapshot < drawn ->
        await_published(t, drawn);
        read_modify_write(t, published_stamp, order_acq_rel, snapshot, snapshot)
    :: else
    fi;
    // `reopen_at`: a participant reading below its own stamp misses what it just wrote
    assert(snapshot >= drawn);
    // `locked_store::transaction_t::prune_committed`, around the partition's where nested
    if
    :: !locked_hold_through_prune -> lock(t, mutex(locked))
    :: else
    fi;
    if
    :: nested -> prune_partition(t)
    :: else
    fi;
    unlock(t, mutex(locked));
    if
    :: !nested -> prune_partition(t)
    :: else
    fi
}

/** A commit over a partition no group reaches, landing under its hold and awaiting its own
 *  publication once the hold is gone. */
proctype committer(byte t) {
    byte landed_run;
    int seen, observed, drawn;
    bool moved;
    lock(t, mutex(2));
    draw_stamp(t, drawn);
    land_stamp(t, drawn);
    unlock(t, mutex(2));
    await_published(t, drawn)
}

/** A group whose stamp the snapshot covers has written the store, so it holds that stamp or a
 *  newer one. */
inline see_whole(snapshot, seen_stamp) {
    for (each : 0 .. 1) {
        if
        :: stamp_of[each] != 0 && stamp_of[each] <= snapshot -> assert(seen_stamp >= stamp_of[each])
        :: else
        fi
    }
}

/** A reader: its snapshot first, then each store under its shared lock, the partition's inside the
 *  locked store's where nested. */
proctype reader(byte t) {
    byte each;
    int seen, snapshot, seen_stamp;
    read_modify_write(t, published_stamp, order_acq_rel, snapshot, snapshot);
    lock_shared(t, mutex(locked));
    if
    :: nested -> skip
    :: else ->
        load(t, stamp_at(locked), order_relaxed, seen_stamp);
        see_whole(snapshot, seen_stamp);
        unlock_shared(t, mutex(locked))
    fi;
    lock_shared(t, mutex(partition));
    load(t, stamp_at(partition), order_relaxed, seen_stamp);
    see_whole(snapshot, seen_stamp);
    unlock_shared(t, mutex(partition));
    if
    :: nested -> unlock_shared(t, mutex(locked))
    :: else
    fi
}

init {
    atomic {
        run group(0);
        if
        :: second_group -> run group(1)
        :: else
        fi;
        run committer(2);
        run reader(3)
    }
}
