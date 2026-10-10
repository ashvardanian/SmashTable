/**
 *  @file verification/transaction_group/in_turn.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Two groups over two stores that do not split their commit, so each participant validates
 *      and publishes on its own, one store at a time.
 *
 *  The in-turn branch validates and publishes under a lock it never drops, so there is no window
 *  for a lost update here. What it does promise is that a group whose commit tore ends pending, so
 *  a second commit is refused as not permitted, which the auditor asserts.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc whole_across_stores=true: a later participant's refusal tears a group that
 *      commits in turn, the tear the docs admit; asserting all or none here as well finds a group
 *      that published one store and not the other
 */
#define thread_count 3
#define location_count 4
#define history_depth 11
#define split_commit false
#define shared_clock false
#define groups_write false
#include "protocol.pml"

proctype group(byte t) { stage_and_commit(t) }

proctype writer(byte t) { overwrite(t) }

proctype auditor() { audit() }

init { atomic { run group(0); run group(1); run writer(2); run auditor() } }
