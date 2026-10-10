/**
 *  @file verification/atomic_hash_table/roomy.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Two emplacers, a finder and an eraser over two slots of @c atomic_hash_table: the slot
 *      lock and unlock, and the counters under it.
 *
 *  The emplacers carry distinct keys and probe from the first slot; the finder and the eraser look
 *  for the first emplacer's key. A populated slot's key is never the initial value: the emplacer's
 *  release and the next holder's acquire carry the key. Every slot ends unlocked, and the counters
 *  end equal to the occupied slots and never go below zero on the way.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 unlock_order=order_relaxed: the unlock's xor releases the key written under
 *      the slot lock; relaxed, the finder that takes the slot next reads the key of a populated
 *      slot before it was written
 *  @verify fail rc11 lock_order=order_relaxed: the lock's @c fetch_or acquires what the last holder
 *      released; relaxed, the finder reads the key of a populated slot before it was written
 *  @verify fail sc count_under_lock=false: @c emplace adds to @c populated_count before it unlocks
 *      the slot, as @c erase subtracts under the lock; adding after the unlock, an eraser slipping
 *      in between takes the unsigned count through zero, a dip in a statistic nothing decides on
 *  @verify fail far count_order=order_relaxed: the populated count's add releases; relaxed, far
 *      memory posts it and lands it only after the unlock that was meant to cover it, and the
 *      eraser takes the count through zero again
 */
#define thread_count 4
#define location_count 5
#define history_depth 15
#define slots 2
#define key_of(t) (10 + (t))
#include "protocol.pml"

proctype emplacer(byte t) { emplace(t) }

proctype finder(byte t) { find(t) }

proctype eraser(byte t) { erase(t) }

proctype auditor() { audit() }

init { atomic { run emplacer(0); run emplacer(1); run finder(2); run eraser(3); run auditor() } }
