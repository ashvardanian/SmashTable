/**
 *  @file verification/staged_batch/flat.pml
 *  @author Ash Vardanian
 *  @date September 18, 2026
 *  @brief The staged batch of @c basic_flat_set: a staging array reserved for the whole range at
 *      once, and an absorb that asks once more for the merged array both runs move into.
 *
 *  @verify pass sc
 *  @verify fail sc staging=false: the range is built in a staging array beside the destination;
 *      built straight in the destination, the shape of an unstaged batch, a refused element leaves
 *      the prefix behind it
 *  @verify fail sc secured_room=false: the absorb asks the allocator once, before a single element
 *      moves; asking once per element, a refusal part-way leaves some of the range moved
 *  @verify fail sc prior_check=false: the key check runs before the absorb; after it, a refusal
 *      lands on a destination the absorb has already merged into
 */
#define relinks false
#include "protocol.pml"

proctype batch() { modify_range() }

init { atomic { run batch() } }
