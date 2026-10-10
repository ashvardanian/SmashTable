/**
 *  @file verification/atomic_hash_table/exhausted.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief A full @c atomic_hash_table: one slot, so the emplacer that loses it walks off the end of
 *      its probe sequence.
 *
 *  The cast of `roomy.pml` over a single slot. Nothing is asserted of the loser itself: with one
 *  slot every claim about its probe count is a tautology, and what the refusal must not break is
 *  the counter invariant, which the auditor already checks.
 *
 *  @verify pass sc,rc11
 */
#define thread_count 4
#define location_count 5
#define history_depth 9
#define slots 1
#include "protocol.pml"

proctype emplacer(byte t) { emplace(t) }

proctype finder(byte t) { find(t) }

proctype eraser(byte t) { erase(t) }

proctype auditor() { audit() }

init { atomic { run emplacer(0); run emplacer(1); run finder(2); run eraser(3); run auditor() } }
