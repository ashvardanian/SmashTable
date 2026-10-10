/**
 *  @file verification/transaction_group/protocol.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief @c transaction_group from `include/smashtable/shared.hpp` over two stores, each a
 *      @c locked_store with its own mutex.
 *
 *  Every participant staged in ascending store address and the staged prefix unwound on a refusal;
 *  a commit that asks every participant before any of them writes, where the stores split their
 *  commit, and one that commits in turn where they do not; and a participant that holds its store's
 *  mutex from its validation to its publication, so nothing moves in between.
 *
 *  Two groups and one writer. Each group watches one key per store while it stages, and publishes
 *  one key per store, which with @c groups_write is the key it watched there and the other group
 *  watches too; the writer commits over the watched key of the second store, under that store's
 *  unique lock, at any point. A group's stage may be refused at the second store, as an inner
 *  store may refuse it, and then unwinds the first. Its commit validates every watch, holding
 *  each store's lock as it goes, and publishes under those holds; a refusal gives every hold back
 *  at once, then the first group reads the store it validated first and drops the group, and the
 *  second rolls back under fresh locks. The two groups split the read and the rollback between
 *  them, since both on one group would overflow the history the one-stamp reader leaves room for.
 *
 *  Invariants:
 *  - no lost update: at every publication the watched key still holds what the group watched;
 *  - no hold outlives a refusal: a caller reading a store after a refused commit never waits on its
 *    own lock;
 *  - two groups holding across their phases never deadlock;
 *  - all or none: a group that returned success published every participant, and one that was
 *    refused published none, where the stores split their commit;
 *  - after a refused stage no store holds the group's staged mark.
 *
 *  Every scenario beside this file defines its shape, @c split_commit for stores that split their
 *  commit into a validation and a publication, @c shared_clock for stores on one
 *  @c basic_commit_order, and @c groups_write for groups whose publication writes the very key
 *  they watched, which the other group watches too.
 */
#include "../weak_memory.pml"
#include "../monitor_wait.pml"
#include "../spin_shared_mutex.pml"

#define stores 2

/** The words: each store's mutex and the version of the key each group watches there; then the
 *  order's mutex, its stamp counter, its watermark, and the newest stamp each store holds. */
#define mutex(reached) (reached)
#define watched(reached) (2 + (reached))
#define order_mutex 4
#define commits 5
#define published_stamp 6
#define stamp_at(reached) (7 + (reached))

/** What the group's ghost state spells, as @c staging_t does. */
#define pending 0
#define staged 1

/** The knobs: the header's choices, which `@verify` lines override to replay counterexamples. */
#ifndef address_order
#define address_order true
#endif
#ifndef prefix_rollback
#define prefix_rollback true
#endif
#ifndef held_validation
#define held_validation true
#endif
#ifndef release_on_refusal
#define release_on_refusal true
#endif
#ifndef one_stamp
#define one_stamp true
#endif
#ifndef whole_across_stores
#define whole_across_stores false
#endif

/** The store a group visits at a position: ascending, or with @c address_order off, the second
 *  group descending. */
#define visited(group, position) (!address_order && (group) == 1 -> 1 - (position) : (position))

byte finished;
byte staging[2];
bool staged_at[2 * stores];    // a participant holds its inner store's staged state
bool published_at[2 * stores]; // a participant published
bool refused_stage[2];
int watch_seen[2 * stores];    // the version each group watched at its stage
int stamp_of[2];               // the newest stamp each group drew, zero before it drew one
#define at(group, store) ((group) * stores + (store))

/** One group: stages every participant in order, then commits or rolls back. */
inline stage_and_commit(t) {
    byte position, reached, staged_count, held_through;
    int seen, version, drawn;
    bool refused;
    // Stage: each participant under its store's unique lock, in order, the prefix unwound on a
    // refusal: `transaction_group::stage` in `shared.hpp`.
    for (position : 0 .. stores - 1) {
        reached = visited(t, position);
        lock(t, mutex(reached));
        if
        :: reached == 1 -> refused = true // the inner store's refusal, which it may answer any time
        :: skip ->
            load(t, watched(reached), order_relaxed, version);
            watch_seen[at(t, reached)] = version;
            staged_at[at(t, reached)] = true;
            staged_count++
        fi;
        unlock(t, mutex(reached));
        if :: refused -> break :: else fi
    };
    if
    :: refused ->
        refused_stage[t] = true;
        if
        :: prefix_rollback ->
            do
            :: staged_count != 0 ->
                staged_count--;
                reached = visited(t, staged_count);
                lock(t, mutex(reached));
                staged_at[at(t, reached)] = false;
                unlock(t, mutex(reached))
            :: else -> break
            od
        :: else
        fi;
        goto done
    :: else -> staging[t] = staged
    fi;
    if
    :: split_commit ->
        // Commit, asking every participant before any writes, where a validation answering success
        // holds its store's lock until the publication: `transaction_group::validate_participants_`
        // over `locked_store::transaction_t::validate_for_commit`.
        for (position : 0 .. stores - 1) {
            reached = visited(t, position);
            if
            :: held_validation -> lock(t, mutex(reached)); load(t, watched(reached), order_relaxed, version)
            :: else ->
                lock_shared(t, mutex(reached));
                load(t, watched(reached), order_relaxed, version);
                unlock_shared(t, mutex(reached))
            fi;
            if :: version != watch_seen[at(t, reached)] -> refused = true; break :: else fi
        };
        if
        :: refused ->
            if
            :: !release_on_refusal -> held_through = position
            :: release_on_refusal && held_validation ->
                // The refusing store gives its own hold straight back, and every store asked before
                // it follows, descending: `transaction_group::validate_participants_` over
                // `locked_store::transaction_t::release_validation`.
                do
                :: unlock(t, mutex(visited(t, position)));
                   if :: position == 0 -> break :: else -> position-- fi
                od
            :: else
            fi;
            if
            :: t == 0 ->
                // The caller reads the store it validated first, which a hold left behind blocks
                // for good, and then drops the group
                lock_shared(t, mutex(visited(t, 0)));
                unlock_shared(t, mutex(visited(t, 0)))
            :: else ->
                // Rollback, ascending, each participant under a lock of its own, or reusing the
                // holds where a refusal kept them: `transaction_group::rollback` in `shared.hpp`.
                for (position : 0 .. stores - 1) {
                    reached = visited(t, position);
                    if
                    :: release_on_refusal || position > held_through -> lock(t, mutex(reached))
                    :: else
                    fi;
                    staged_at[at(t, reached)] = false;
                    unlock(t, mutex(reached))
                };
                staging[t] = pending
            fi;
            goto done
        :: else
        fi;
        // `begin_commit`: one stamp for the whole group, drawn under the order's mutex before any
        // store is written
        if
        :: shared_clock && one_stamp ->
            lock(t, order_mutex);
            read_modify_write(t, commits, order_relaxed, seen, seen + 1);
            drawn = seen + 1;
            stamp_of[t] = drawn;
            unlock(t, order_mutex)
        :: else
        fi;
        for (position : 0 .. stores - 1) {
            reached = visited(t, position);
            if
            :: !held_validation -> lock(t, mutex(reached))
            :: else
            fi;
            // `publish_under`: nothing moved since the validation, or the update is lost:
            // `locked_store::transaction_t::publish_under` in `locked_store.hpp`.
            assert(newest_value(watched(reached)) == watch_seen[at(t, reached)]);
            if
            :: groups_write -> read_modify_write(t, watched(reached), order_relaxed, seen, seen + 1)
            :: else
            fi;
            published_at[at(t, reached)] = true;
            staged_at[at(t, reached)] = false;
            if
            :: shared_clock && one_stamp ->
                store(t, stamp_at(reached), order_relaxed, drawn);
                unlock(t, mutex(reached))
            :: shared_clock && !one_stamp ->
                // A stamp per store, drawn and published as that store's own commit would
                lock(t, order_mutex);
                read_modify_write(t, commits, order_relaxed, seen, seen + 1);
                drawn = seen + 1;
                stamp_of[t] = drawn;
                unlock(t, order_mutex);
                store(t, stamp_at(reached), order_relaxed, drawn);
                unlock(t, mutex(reached));
                lock(t, order_mutex);
                store(t, published_stamp, order_relaxed, drawn);
                unlock(t, order_mutex)
            :: else -> unlock(t, mutex(reached))
            fi
        };
        // `end_commit`: the watermark moves once, after the last store, under the order's mutex
        if
        :: shared_clock && one_stamp ->
            lock(t, order_mutex);
            store(t, published_stamp, order_relaxed, drawn);
            unlock(t, order_mutex)
        :: else
        fi;
        staging[t] = pending
    :: else ->
        // Commit in turn: each participant validates and publishes on its own, and a refusal past
        // the first leaves the group pending: `transaction_group::commit` in `shared.hpp`, its
        // one-at-a-time branch.
        for (position : 0 .. stores - 1) {
            reached = visited(t, position);
            lock(t, mutex(reached));
            load(t, watched(reached), order_relaxed, version);
            if
            :: version != watch_seen[at(t, reached)] ->
                unlock(t, mutex(reached));
                if :: position != 0 -> staging[t] = pending :: else fi;
                refused = true;
                break
            :: else ->
                published_at[at(t, reached)] = true;
                staged_at[at(t, reached)] = false;
                unlock(t, mutex(reached))
            fi
        };
        // A refusal past the first left the group pending, and one at the first staged for rollback
        if
        :: refused -> skip
        :: else -> staging[t] = pending
        fi
    fi;
done:
    finished++
}

/** A commit over the second store's watched key, under its unique lock, at any time. */
inline overwrite(t) {
    int seen;
    lock(t, mutex(1));
    read_modify_write(t, watched(1), order_relaxed, seen, seen + 1);
    unlock(t, mutex(1));
    finished++
}

/** Once the groups and the writer returned: the unwind left nothing staged, a group published all
 *  or none where the stores split their commit, and a torn commit ended pending, so a second one
 *  answers @c operation_not_permitted_k. It reads no word and plays no thread. */
inline audit() {
    byte each;
    (finished == 3);
    for (each : 0 .. 1) {
        if
        :: refused_stage[each] -> assert(!staged_at[at(each, 0)] && !staged_at[at(each, 1)])
        :: else
        fi;
        assert(!(split_commit || whole_across_stores) || published_at[at(each, 0)] == published_at[at(each, 1)]);
        assert(split_commit || published_at[at(each, 0)] == published_at[at(each, 1)] || staging[each] == pending)
    }
}
