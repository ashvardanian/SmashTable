/**
 *  @file verification/transaction_group/one_stamp.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Two groups over two stores on one @c basic_commit_order: one stamp per group, and a
 *      reader that must see each group whole.
 *
 *  The group draws one stamp under the order's mutex once every store validated, stamps each store
 *  as it publishes, and moves the watermark once, after the last; a reader draws its snapshot under
 *  the order's mutex and reads each store under its shared lock. The order's ring and census stand
 *  here as one stamp drawn and one watermark stored under a mutex, since the words run out; what
 *  @c begin_commit, @c end_commit and @c take_snapshot do in full is `commit_order/`'s to show.
 *
 *  @verify pass sc,rc11
 *  @verify fail sc one_stamp=false: the group draws one stamp for every store it publishes and
 *      moves the watermark once, after the last; drawing and publishing a stamp per store, as each
 *      store's own commit would, the reader's snapshot covers a group's stamp on one store while
 *      the other still holds an older one
 */
#define thread_count 4
#define location_count 9
#define history_depth 15
#define split_commit true
#define shared_clock true
#define groups_write false
#include "protocol.pml"

proctype group(byte t) { stage_and_commit(t) }

proctype writer(byte t) { overwrite(t) }

/** A reader: its snapshot drawn under the order's mutex, then each store under its shared lock; a
 *  group whose stamp the snapshot covers has written every store, so each store holds that stamp or
 *  a newer one. */
proctype reader(byte t) {
    byte reached, each;
    int seen, snapshot, seen_stamp;
    lock(t, order_mutex);
    load(t, published_stamp, order_relaxed, snapshot);
    unlock(t, order_mutex);
    for (reached : 0 .. stores - 1) {
        lock_shared(t, mutex(reached));
        load(t, stamp_at(reached), order_relaxed, seen_stamp);
        for (each : 0 .. 1) {
            if
            :: stamp_of[each] != 0 && stamp_of[each] <= snapshot -> assert(seen_stamp >= stamp_of[each])
            :: else
            fi
        };
        unlock_shared(t, mutex(reached))
    }
}

proctype auditor() { audit() }

init { atomic { run group(0); run group(1); run writer(2); run reader(3); run auditor() } }
