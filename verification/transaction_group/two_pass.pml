/**
 *  @file verification/transaction_group/two_pass.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Two groups over two locked stores that split their commit: every participant asked
 *      before any writes, and the holds a validation keeps until the publication or the refusal.
 *
 *  Each group publishes over the keys it watched, so either group's validation can be refused by
 *  the other's publication as well as by the writer's.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc held_validation=false: validating under a shared lock dropped before the
 *      publication's unique lock, the other group publishes over the watched key in the gap, the
 *      lost update the watch was taken against
 *  @verify stuck sc address_order=false: with one group descending, each holds a store the other
 *      waits on, which Spin finds as an invalid end state
 *  @verify stuck sc release_on_refusal=false: keeping the holds through the refusal, the first
 *      group's read of a store it validated waits on its own lock for good, an invalid end state
 *  @verify fail sc prefix_rollback=false: without the unwind, the first store keeps the
 *      group's staged mark
 */
#define thread_count 3
#define location_count 4
#define history_depth 13
#define split_commit true
#define shared_clock false
#define groups_write true
#include <transaction_group/protocol.pml>

proctype group(byte t) { stage_and_commit(t) }

proctype writer(byte t) { overwrite(t) }

proctype auditor() { audit() }

init { atomic { run group(0); run group(1); run writer(2); run auditor() } }
