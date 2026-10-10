/**
 *  @file verification/atomic_hash_table/same_key.pml
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief Two emplacers of one key, a finder and an eraser over two slots of @c atomic_hash_table:
 *      the key is never populated in two slots at once.
 *
 *  The cast of `roomy.pml` with both emplacers carrying the same key. Both probe from the first
 *  slot and lock each slot in turn, so the second meets the first one's key under the lock and
 *  stops there, or meets the tombstone the eraser left and takes the next free slot, which holds
 *  the key once.
 *
 *  @verify pass sc,rc11
 *  @verify fail rc11 lock_order=order_relaxed: the lock's @c fetch_or acquires the key the last
 *      holder wrote; relaxed, the second emplacer reads the populated slot's key before it was
 *      written, walks past its own key, and populates the next slot with it as well
 */
#define thread_count 4
#define location_count 5
#define history_depth 13
#define slots 2
#define key_of(t) 10
#include "protocol.pml"

proctype emplacer(byte t) { emplace(t) }

proctype finder(byte t) { find(t) }

proctype eraser(byte t) { erase(t) }

proctype auditor() { audit() }

init { atomic { run emplacer(0); run emplacer(1); run finder(2); run eraser(3); run auditor() } }
