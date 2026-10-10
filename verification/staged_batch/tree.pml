/**
 *  @file verification/staged_batch/tree.pml
 *  @author Ash Vardanian
 *  @date September 18, 2026
 *  @brief The staged batch of @c basic_avl_tree: a staging tree that asks for a node per key, and a
 *      merge that relinks those nodes and asks for nothing.
 *
 *  Spin reports the absorb's refusal branch as unreachable here, which is the merge relinking
 *  rather than allocating, stated as a verdict rather than as a docblock.
 *
 *  @verify pass sc
 *  @verify fail sc staging=false: the range is built in a staging tree beside the destination;
 *      built straight in the destination, the shape of an unstaged batch, a refused element leaves
 *      the prefix behind it
 *  @verify fail sc secured_room=false: the merge relinks the staged nodes and asks the allocator
 *      for nothing; asking once per element, the shape a merge that copied rather than relinked
 *      would have, it asks at all, and a refusal part-way leaves some of the range moved
 *  @verify fail sc prior_check=false: the key check runs before the absorb; after it, a refusal
 *      lands on a destination already merged, and the check of @c update finds the key the absorb
 *      has just written, so the batch answers success over a destination it never had to touch
 */
#define relinks true
#include "protocol.pml"

proctype batch() { modify_range() }

init { atomic { run batch() } }
