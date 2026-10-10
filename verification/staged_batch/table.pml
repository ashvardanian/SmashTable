/**
 *  @file verification/staged_batch/table.pml
 *  @author Ash Vardanian
 *  @date September 18, 2026
 *  @brief The staged batch of @c basic_hash_table: a staging vector reserved for the whole range at
 *      once, and an absorb that asks once more through @c reserve_more.
 *
 *  A table whose element duplicates without refusing stages nothing and writes the caller's range
 *  straight in; this covers that path as the same one allocation, since the two differ in what they
 *  duplicate rather than in what a refusal leaves behind. That @c apply_each_ cannot refuse once
 *  @c reserve_more has returned rests on the growth threshold leaving a quarter of the slots free
 *  and on a bounded probe walking all of them, which is `hash_layout.hpp`'s property and has no
 *  model of its own.
 *
 *  @verify pass sc
 *  @verify fail sc staging=false: the range is built in a staging vector beside the destination;
 *      built straight in the destination, the shape of an unstaged batch, a refused element leaves
 *      the prefix behind it
 *  @verify fail sc secured_room=false: @c reserve_more asks the allocator once, before a single
 *      element moves; asking once per element, a refusal part-way leaves some of the range moved
 *  @verify fail sc prior_check=false: the key check runs before the absorb; after it, a refusal
 *      lands on a destination the absorb has already written into
 */
#define relinks false
#include "protocol.pml"

proctype batch() { modify_range() }

init { atomic { run batch() } }
