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
 *  @verify fail sc held_validation=false: @c validate_for_commit takes the store's mutex
 *      exclusively and, when it answers success, keeps it until the publication, the rollback or
 *      the reset that follows; validating under a shared lock dropped before the publication's
 *      unique lock, the other group publishes over the watched key in the gap, the lost update the
 *      watch was taken against. The partition locks of a @c partitioned_store cover that gap, but a
 *      group built straight from locked stores has nothing over it
 *  @verify stuck sc address_order=false: two groups holding across their phases cannot deadlock
 *      because both take the stores ascending; with one group descending, each holds a store the
 *      other waits on, which Spin finds as an invalid end state
 *  @verify stuck sc release_on_refusal=false: a refused validation gives its hold straight back, a
 *      caller that asked several stores gives back the holds of those asked before the refusal, and
 *      the rollback takes its locks anew; keeping the holds through the refusal, the first group's
 *      read of a store it validated waits on its own lock for good, an invalid end state. A
 *      rollback straight after the refusal reuses the holds and hides this, which is why the first
 *      group reads rather than rolls back
 *  @verify fail sc prefix_rollback=false: a refused stage unwinds the participants staged before
 *      the refusal; without the unwind, the first store keeps the group's staged mark
 */
#define thread_count 3
#define location_count 4
#define history_depth 13
#define split_commit true
#define shared_clock false
#define groups_write true
#include "protocol.pml"

proctype group(byte t) { stage_and_commit(t) }

proctype writer(byte t) { overwrite(t) }

proctype auditor() { audit() }

init { atomic { run group(0); run group(1); run writer(2); run auditor() } }
