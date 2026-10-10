/**
 *  @file verification/atomic_hash_table/protocol.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Spin model of @c atomic_hash_table from `include/smashtable/atomic_hash_table.hpp`, on
 *      the slot header of `hash_layout.hpp`: two bits per slot in one header word, free, deleted,
 *      populated or locked.
 *
 *  A slot is locked by driving both bits up with an acquire @c fetch_or, and released by an xor
 *  with release that lands the staged state; the counters are posted with release under it.
 *
 *  An emplacer locks each slot in turn: a populated one is compared, a deleted one skipped, a free
 *  one taken, its key written under the lock and the slot released as populated, with the populated
 *  count moved under the lock. The finder locks each slot and reads the key of a populated one; the
 *  eraser does the same and drives a match to deleted, moving both counters under the lock. The
 *  emplacer posts its count as a no-return add with release, as the code does. The eraser's two
 *  moves are spelled performed, since the module lands one post per thread at a time.
 *
 *  What a loser of the slot lock does between two attempts is `waiting_policy.pml`'s, chosen by
 *  @c waiting: the header word alone decides who holds a slot, so every invariant has to hold
 *  under all four policies.
 *
 *  Every scenario beside this file defines its shape and @c slots, the slots the probe walks.
 */
#include "../weak_memory.pml"
#include "../waiting_policy.pml"

/** The words: the header, one key per slot, and the two counters. */
#define header 0
#define key(slot) (1 + (slot))
#define populated_count 3
#define deleted_count 4

/** A slot's bits, the populations lane low and the deletions lane high: @c hash_bucket_head_t. */
#define populated(slot) (1 << (slot))
#define deleted(slot) (1 << (2 + (slot)))
#define mask(slot) (populated(slot) | deleted(slot))
#define unset 0
#define key_of(t) (10 + (t))

/** The knobs: the header's choices, which `@verify` lines override to replay counterexamples. */
#ifndef lock_order
#define lock_order order_acquire
#endif
#ifndef unlock_order
#define unlock_order order_release
#endif
#ifndef count_order
#define count_order order_release
#endif
#ifndef count_under_lock
#define count_under_lock true
#endif

byte finished; // the emplacers, the finder and the eraser that returned

/** Both bits driven up with acquire, until the holder is the one that saw them down; the state seen
 *  is staged: @c hash_atomic_slot_ref::lock in `hash_layout.hpp`. A loser reads until the two bits
 *  are not both set before it claims again, rather than re-issuing the @c fetch_or on a header
 *  thirty-two slots share. */
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

/** Probes for the thread's own key and takes the first free slot:
 *  @c atomic_hash_table::probe_to_upsert_. */
inline emplace(t) {
    byte slot;
    int seen, staged, seen_key;
    for (slot : 0 .. slots - 1) {
        lock(t, slot);
        if
        :: staged == populated(slot) ->
            load(t, key(slot), order_relaxed, seen_key);
            unlock(t, slot, populated(slot));
            if :: seen_key == key_of(t) -> goto done :: else fi
        :: staged == deleted(slot) -> unlock(t, slot, deleted(slot))
        :: else ->
            store(t, key(slot), order_relaxed, key_of(t));
            if
            :: count_under_lock -> add_no_return(t, populated_count, count_order, 1); unlock(t, slot, populated(slot))
            :: else -> unlock(t, slot, populated(slot)); add_no_return(t, populated_count, count_order, 1)
            fi;
            goto done
        fi
    };
done:
    landed(t);
    finished++
}

/** Each slot locked in turn, a populated one's key read under the lock, looking for the first
 *  emplacer's key: @c atomic_hash_table::probe_to_find_. */
inline find(t) {
    byte slot;
    int seen, staged, seen_key;
    for (slot : 0 .. slots - 1) {
        lock(t, slot);
        if
        :: staged == populated(slot) ->
            load(t, key(slot), order_relaxed, seen_key);
            assert(seen_key != unset);
            unlock(t, slot, populated(slot));
            if :: seen_key == key_of(0) -> goto done :: else fi
        :: staged == deleted(slot) -> unlock(t, slot, deleted(slot))
        :: else -> unlock(t, slot, 0); goto done
        fi
    };
done:
    finished++
}

/** The first emplacer's key driven to deleted under the lock, both counters moved under it:
 *  @c atomic_hash_table::erase. */
inline erase(t) {
    byte slot;
    int seen, staged, seen_key;
    for (slot : 0 .. slots - 1) {
        lock(t, slot);
        if
        :: staged == populated(slot) ->
            load(t, key(slot), order_relaxed, seen_key);
            if
            :: seen_key == key_of(0) ->
                read_modify_write(t, deleted_count, order_release, seen, seen + 1);
                read_modify_write(t, populated_count, order_release, seen, seen - 1);
                assert(newest_value(populated_count) >= 0);
                unlock(t, slot, deleted(slot));
                goto done
            :: else -> unlock(t, slot, populated(slot))
            fi
        :: staged == deleted(slot) -> unlock(t, slot, deleted(slot))
        :: else -> unlock(t, slot, 0); goto done
        fi
    };
done:
    finished++
}

/** Once everyone returned: no slot locked, and the counters equal to the occupied slots and never
 *  below zero. It reads no word and plays no thread. */
inline audit() {
    int seen, occupied;
    (finished == 4);
    seen = newest_value(header);
    assert((seen & mask(0)) != mask(0) && (seen & mask(1)) != mask(1));
    occupied = ((seen & mask(0)) != 0) + ((seen & mask(1)) != 0);
    assert(newest_value(populated_count) + newest_value(deleted_count) == occupied);
    assert(newest_value(populated_count) >= 0 && newest_value(deleted_count) >= 0)
}
