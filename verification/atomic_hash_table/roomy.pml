/**
 *  @file verification/atomic_hash_table/roomy.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Two emplacers, a finder and an eraser over two slots of @c atomic_hash_table: the slot
 *      lock and unlock, and the counters under it.
 *
 *  The emplacers carry distinct keys and probe from the first slot; the finder and the eraser look
 *  for the first emplacer's key.
 *
 *  - A populated slot's key is never the initial value.
 *  - Every slot ends unlocked.
 *  - The counters end equal to the occupied slots and never go below zero on the way.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 unlock_order=order_relaxed: relaxed, the finder that takes the slot next reads
 *      the key of a populated slot before it was written
 *  @verify fail rc11 lock_order=order_relaxed: relaxed, the finder reads the key of a populated
 *      slot before it was written
 *  @verify fail sc count_under_lock=false: adding after the unlock, an eraser slipping in between
 *      takes the unsigned count through zero
 *  @verify fail far count_order=order_relaxed: relaxed, far memory posts it and lands it only after
 *      the unlock that was meant to cover it, and the eraser takes the count through zero again
 */
#define thread_count 4
#define location_count 5
#define history_depth 15
#define slots 2
#define key_of(t) (10 + (t))
#include <atomic_hash_table/protocol.pml>

proctype emplacer(byte t) { emplace(t) }

proctype finder(byte t) { find(t) }

proctype eraser(byte t) { erase(t) }

proctype auditor() { audit() }

init { atomic { run emplacer(0); run emplacer(1); run finder(2); run eraser(3); run auditor() } }
