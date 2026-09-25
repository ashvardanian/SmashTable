/**
 *  @file verification/atomic_hash_table.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c atomic_hash_table from `include/smashtable/atomic_hash_table.hpp` on the
 *      slot header of `hash_layout.hpp`: two bits per slot in one header word, free, deleted,
 *      populated or locked.
 *
 *  A slot is locked by driving both bits up with an acquire @c fetch_or, and released by an xor
 *  with release that lands the staged state; the counters are posted with release under it.
 *
 *  Two emplacers with distinct keys, a finder and an eraser after the first key, over two slots
 *  probed from the first. An emplacer locks each slot in turn: a populated one is compared, a
 *  deleted one skipped, a free one taken, its key written under the lock and the slot released as
 *  populated, with the populated count moved under the lock. The finder locks each slot and reads
 *  the key of a populated one; the eraser does the same and drives a match to deleted, moving both
 *  counters under the lock.
 *
 *  A populated slot's key is never the initial value: the emplacer's release and the next holder's
 *  acquire carry the key. `-Dwithout_unlock_release` and `-Dwithout_lock_acquire` each let the
 *  finder read an unwritten key under views.
 *
 *  Every slot ends unlocked, and the counters end equal to the occupied slots and never go below
 *  zero on the way. `-Dwithout_count_under_lock` moves the populated count after the unlock, the
 *  header before the fix, and an eraser slipping in between takes the count through zero. The
 *  emplacer posts its count as a no-return add with release, as the code does;
 *  `-Dwithout_count_release` posts it relaxed, which `-Dmemory=far` lands only later, after the
 *  unlock that was meant to cover it, and the eraser takes the count through zero again. The
 *  eraser's two moves are spelled performed, since the module lands one post per thread at a time.
 *
 *  Under `-Dscenario=exhausted` the table runs with one slot, so the emplacer that loses it walks
 *  off the end of its probe sequence. Nothing there is asserted of the loser itself: with one slot
 *  every claim about its probe count is a tautology, and what the refusal must not break is the
 *  counter invariant above, which the auditor already checks.
 *
 *  What a loser of the slot lock does between two attempts is `waiting_policy.pml`'s, chosen by
 *  `-Dwaiting=`: the header word alone decides who holds a slot, so every invariant above has to
 *  hold under all four policies.
 */
#include "weak_memory.pml"
#include "waiting_policy.pml"

/** The knob's values are integers, so a typo fails the range check below. */
#define roomy 1
#define exhausted 2
#ifndef scenario
#define scenario roomy
#endif
#if scenario < roomy || scenario > exhausted
#error "scenario is roomy or exhausted"
#endif

/** The threads, by role, apart from the processes that play them; the emplacers name themselves. */
#define finder_thread 2
#define eraser_thread 3

/** The words: the header, one key per slot, and the two counters. */
#define header 0
#define key(slot) (1 + (slot))
#define populated_count 3
#define deleted_count 4

/** The header's bits for a slot, the populations lane low and the deletions lane high:
 *  @c hash_bucket_head_t in `hash_layout.hpp`. */
#if scenario == exhausted
#define slots 1
#else
#define slots 2
#endif
#define populated(slot) (1 << (slot))
#define deleted(slot) (1 << (2 + (slot)))
#define mask(slot) (populated(slot) | deleted(slot))
#define unset 0
#define key_of(t) (10 + (t))

#ifdef without_lock_acquire
#define lock_order order_relaxed
#else
#define lock_order order_acquire
#endif

#ifdef without_unlock_release
#define unlock_order order_relaxed
#else
#define unlock_order order_release
#endif

#ifdef without_count_release
#define count_order order_relaxed
#else
#define count_order order_release
#endif

byte emplacers_started;
byte finished;

/** Both bits driven up with acquire, until the holder is the one that saw them down; the state seen
 *  is staged: @c hash_atomic_slot_ref::lock in `hash_layout.hpp`. */
inline lock(t, slot) {
    do
    :: read_modify_write_if(t, header, lock_order, (seen & mask(slot)) != mask(slot), seen, seen | mask(slot));
       if
       :: (seen & mask(slot)) != mask(slot) -> staged = seen & mask(slot); break
       :: else -> wait_until(header, (newest_value(header) & mask(slot)) != mask(slot))
       fi
    od
}

/** The xor of the locked state with the staged one, with release: @c hash_atomic_slot_ref::unlock
 *  in `hash_layout.hpp`. */
inline unlock(t, slot, bits) { read_modify_write(t, header, unlock_order, seen, seen ^ (mask(slot) ^ (bits))) }

/** An emplacer: probes for its key and takes the first free slot. */
active [2] proctype emplacer() {
    byte me, slot;
    int seen, staged, seen_key;
    atomic { me = emplacers_started; emplacers_started++ };
    // Emplace: each slot locked in turn, compared when populated, skipped when deleted, taken
    // when free: `atomic_hash_table::probe_to_upsert_`
    for (slot : 0 .. slots - 1) {
        lock(me, slot);
        if
        :: staged == populated(slot) ->
            load(me, key(slot), order_relaxed, seen_key);
            unlock(me, slot, populated(slot));
            if :: seen_key == key_of(me) -> goto done :: else fi
        :: staged == deleted(slot) -> unlock(me, slot, deleted(slot))
        :: else ->
            store(me, key(slot), order_relaxed, key_of(me));
#ifdef without_count_under_lock
            unlock(me, slot, populated(slot));
            add_no_return(me, populated_count, count_order, 1);
#else
            add_no_return(me, populated_count, count_order, 1);
            unlock(me, slot, populated(slot));
#endif
            goto done
        fi
    };
done:
    landed(me);
    finished++
}

/** Each slot locked in turn, a populated one's key read under the lock:
 *  @c atomic_hash_table::probe_to_find_ in `atomic_hash_table.hpp`. */
active proctype finder() {
    byte slot;
    int seen, staged, seen_key;
    for (slot : 0 .. slots - 1) {
        lock(finder_thread, slot);
        if
        :: staged == populated(slot) ->
            load(finder_thread, key(slot), order_relaxed, seen_key);
            assert(seen_key != unset);
            unlock(finder_thread, slot, populated(slot));
            if :: seen_key == key_of(0) -> goto done :: else fi
        :: staged == deleted(slot) -> unlock(finder_thread, slot, deleted(slot))
        :: else -> unlock(finder_thread, slot, 0); goto done
        fi
    };
done:
    finished++
}

/** A match driven to deleted under the lock, both counters moved under it:
 *  @c atomic_hash_table::erase in `atomic_hash_table.hpp`. */
active proctype eraser() {
    byte slot;
    int seen, staged, seen_key;
    for (slot : 0 .. slots - 1) {
        lock(eraser_thread, slot);
        if
        :: staged == populated(slot) ->
            load(eraser_thread, key(slot), order_relaxed, seen_key);
            if
            :: seen_key == key_of(0) ->
                read_modify_write(eraser_thread, deleted_count, order_release, seen, seen + 1);
                read_modify_write(eraser_thread, populated_count, order_release, seen, seen - 1);
                assert(newest_value(populated_count) >= 0);
                unlock(eraser_thread, slot, deleted(slot));
                goto done
            :: else -> unlock(eraser_thread, slot, populated(slot))
            fi
        :: staged == deleted(slot) -> unlock(eraser_thread, slot, deleted(slot))
        :: else -> unlock(eraser_thread, slot, 0); goto done
        fi
    };
done:
    finished++
}

/** Once everyone returned: no slot locked, and the counters equal to the occupied slots. */
active proctype auditor() {
    int seen, occupied;
    (finished == 4);
    seen = newest_value(header);
    assert((seen & mask(0)) != mask(0) && (seen & mask(1)) != mask(1));
    occupied = ((seen & mask(0)) != 0) + ((seen & mask(1)) != 0);
    assert(newest_value(populated_count) + newest_value(deleted_count) == occupied);
    assert(newest_value(populated_count) >= 0 && newest_value(deleted_count) >= 0)
}
