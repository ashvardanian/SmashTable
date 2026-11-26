/**
 *  @file verification/shared_mutex_order.pml
 *  @author Ash Vardanian
 *  @date November 26, 2026
 *  @brief @c spin_shared_mutex_t serving three writers in the order they took their tickets, beside
 *      a reader that yields to every one of them.
 *
 *  Three writers lock and unlock once each, and a reader takes the shared side once, all on one
 *  mutex. Three is the fewest writers where one can bypass another: one inside, two queued behind
 *  it. The order itself is asserted by <tt>spin_shared_mutex.pml</tt>.
 *
 *  - Exclusion: no reader is inside while a writer is, and no writer while anyone else is.
 *  - Writers enter in ticket order, and no reader is admitted while a writer is queued.
 *
 *  @verify pass sc,rc11
 *  @verify pass sc blind_entry=false
 *  @verify pass sc attempts_bound=10: no call is refused more often than the unlocks and last
 *      units out that can wake it
 *  @verify fail sc ticket_bits=1: two tickets for three writers, so the third takes the ticket of
 *      the one inside and enters beside it
 *  @verify stuck sc last_reader_notify=false: without the wake, the writer served behind the
 *      reader sleeps once it is gone, since nothing else writes the word
 */
#define thread_count 4
#define location_count 1
#define history_depth 12
#include <weak_memory.pml>
#include <monitor_wait.pml>
#include <spin_shared_mutex.pml>

#define mutex 0

byte inside_reading; // readers past the shared lock
byte inside_writing; // writers past the unique lock

/** A writer: one unique lock around nothing but the exclusion check. */
proctype writer(byte t) {
    int seen;
    lock(t, mutex);
    atomic { assert(inside_reading == 0 && inside_writing == 0); inside_writing++ };
    inside_writing--;
    unlock(t, mutex)
}

/** A reader: one shared lock around nothing but the exclusion check. */
proctype reader(byte t) {
    int seen;
    lock_shared(t, mutex);
    atomic { assert(inside_writing == 0); inside_reading++ };
    inside_reading--;
    unlock_shared(t, mutex)
}

init { atomic { run writer(0); run writer(1); run writer(2); run reader(3) } }
