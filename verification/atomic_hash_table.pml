/**
 *  `atomic_hash_table` from `include/smashtable/atomic_hash_table.hpp` on the slot header of
 *  `hash_layout.hpp`: two bits per slot in one header word, free, deleted, populated or
 *  locked; a slot locked by driving both bits up with an acquire `fetch_or`, and released by
 *  an xor with release that lands the staged state; the counters moved relaxed beside it.
 *
 *  Two emplacers with distinct keys, a finder and an eraser after the first key, over two
 *  slots probed from the first. An emplacer locks each slot in turn: a populated one is
 *  compared, a deleted one skipped, a free one taken, its key written under the lock and the
 *  slot released as populated, with the populated count moved under the lock. The finder
 *  locks each slot and reads the key of a populated one; the eraser does the same and drives
 *  a match to deleted, moving both counters under the lock.
 *
 *  Invariants:
 *  - a populated slot's key is never the initial value: the emplacer's release and the next
 *    holder's acquire carry the key. `-Dwithout_unlock_release` and `-Dwithout_lock_acquire`
 *    each let the finder read an unwritten key under views;
 *  - every slot ends unlocked, and the counters end equal to the occupied slots and never go
 *    below zero on the way. `-Dwithout_count_under_lock` moves the populated count after the
 *    unlock, the header before the fix, and an eraser slipping in between takes the count
 *    through zero;
 *  - `-Dscenario=exhausted` runs the table with one slot: the second emplacer answers
 *    `capacity_exhausted_k` after exactly one probe.
 */
#include "weak_memory.pml"

// The knob's values are integers, so a typo fails the range check below.
#define roomy 1
#define exhausted 2
#ifndef scenario
#define scenario roomy
#endif
#if scenario < roomy || scenario > exhausted
#error "scenario is roomy or exhausted"
#endif

// The threads, by role, apart from the processes that play them; the emplacers name themselves.
#define finder_thread 2
#define eraser_thread 3

// The words: the header, one key per slot, and the two counters.
#define header 0
#define key(slot) (1 + (slot))
#define populated_count 3
#define deleted_count 4

// The header's bits for a slot, the populations lane low and the deletions lane high: hash_layout.hpp:78-103
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

byte emplacers_started;
byte finished;
byte probes[2]; // each emplacer's probes before it returned

// lock: both bits driven up with acquire, until the holder is the one that saw them down; the state seen is staged: hash_layout.hpp:453-459
inline lock(t, slot) {
    do
    :: read_modify_write_if(t, header, lock_order, (seen & mask(slot)) != mask(slot), seen, seen | mask(slot));
       if
       :: (seen & mask(slot)) != mask(slot) -> staged = seen & mask(slot); break
       :: else -> ((newest_value(header) & mask(slot)) != mask(slot))
       fi
    od
}

// unlock: the xor of the locked state with the staged one, with release: hash_layout.hpp:465-482
inline unlock(t, slot, bits) { read_modify_write(t, header, unlock_order, seen, seen ^ (mask(slot) ^ (bits))) }

active [2] proctype emplacer() {
    byte me, slot;
    int seen, staged, seen_key;
    atomic { me = emplacers_started; emplacers_started++ };
    // emplace: each slot locked in turn, compared when populated, skipped when deleted, taken when free: atomic_hash_table.hpp:375-404
    for (slot : 0 .. slots - 1) {
        probes[me]++;
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
            read_modify_write(me, populated_count, order_relaxed, seen, seen + 1);
#else
            read_modify_write(me, populated_count, order_relaxed, seen, seen + 1);
            unlock(me, slot, populated(slot));
#endif
            goto done
        fi
    };
#if scenario == exhausted
    assert(probes[me] == slots); // capacity_exhausted_k, after every slot was probed once
#endif
done:
    finished++
}

// find: each slot locked in turn, a populated one's key read under the lock: atomic_hash_table.hpp:319-344
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

// erase: a match driven to deleted under the lock, both counters moved under it: atomic_hash_table.hpp:286-293
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
                read_modify_write(eraser_thread, deleted_count, order_relaxed, seen, seen + 1);
                read_modify_write(eraser_thread, populated_count, order_relaxed, seen, seen - 1);
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

// once everyone returned: no slot locked, and the counters equal to the occupied slots
active proctype auditor() {
    int seen, occupied;
    (finished == 4);
    seen = newest_value(header);
    assert((seen & mask(0)) != mask(0) && (seen & mask(1)) != mask(1));
    occupied = ((seen & mask(0)) != 0) + ((seen & mask(1)) != 0);
    assert(newest_value(populated_count) + newest_value(deleted_count) == occupied);
    assert(newest_value(populated_count) >= 0 && newest_value(deleted_count) >= 0)
}
