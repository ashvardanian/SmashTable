/**
 *  `transaction_group` from `include/smashtable/shared.hpp` over two `locked_store`s: every
 *  participant staged in ascending store address and the staged prefix unwound on a refusal;
 *  a commit that asks every participant before any of them writes, where the stores split
 *  their commit, and one that commits in turn where they do not; and a participant that holds
 *  its store's mutex from its validation to its publication, so nothing moves in between.
 *
 *  Two groups and one writer. Each group watches one key per store while it stages, and
 *  publishes one key per store; the writer commits over the watched key of the second store,
 *  under that store's unique lock, at any point. A group's stage may be refused at the second
 *  store, as an inner store may refuse it, and then unwinds the first. Its commit validates
 *  every watch, holding each store's lock as it goes, and publishes under those holds; a
 *  refusal is followed by the rollback that releases them. Under `-Dscenario=in_turn` each
 *  participant validates and publishes on its own, one store at a time. Under
 *  `-Dscenario=one_stamp` the stores share a `basic_commit_order`: the group draws one stamp
 *  under the order's mutex once every store validated, stamps each store as it publishes, and moves
 *  the watermark once, after the last; a reader draws its snapshot under the order's mutex and reads
 *  each store under its shared lock.
 *
 *  Invariants:
 *  - no lost update: at every publication the watched key still holds what the group watched.
 *    `-Dwithout_held_validation` validates under a shared lock it drops before the unique lock
 *    of the publication, the header before the fix, and the writer slips in between. It weakens
 *    the one-stamp branch only; the in-turn branch validates and publishes under a lock it never
 *    drops, so there is no window there to assert against;
 *  - address order: two groups holding across their phases never deadlock, since both take
 *    the stores ascending. `-Dwithout_address_order` reverses one group's order and Spin finds
 *    the wait cycle as an invalid end state;
 *  - all or none: a group that returned success published every participant, and one that was
 *    refused published none, where the stores split their commit. Under `in_turn` the docs
 *    promise less: `-Dwhole_across_stores` asserts all or none there too and fails, the tear
 *    the docs admit, while the scenario itself asserts what they do promise, that a group whose
 *    commit tore ends pending, so a second commit is refused as not permitted;
 *  - the unwind: after a refused stage no store holds the group's staged mark.
 *    `-Dwithout_prefix_rollback` leaves the first store staged;
 *  - one stamp: a reader whose snapshot covers a group's stamp finds that stamp, or a newer one, in
 *    every store. `-Dwithout_one_stamp` draws and publishes a stamp per store, as each store's own
 *    commit would, and the reader catches the group half published.
 */
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

// The knob's values are integers, so a typo fails the range check below.
#define two_pass 1
#define in_turn 2
#define one_stamp 3
#ifndef scenario
#define scenario two_pass
#endif
#if scenario < two_pass || scenario > one_stamp
#error "scenario is two_pass, in_turn or one_stamp"
#endif

// The threads, by role, apart from the processes that play them; the groups name themselves.
#define writer_thread 2
#define reader_thread 3
#define stores 2

// The words: each store's mutex and the version of the key each group watches there; then the
// order's mutex, its stamp counter and its watermark, and the newest stamp each store was written under.
#define mutex(reached) (reached)
#define watched(reached) (2 + (reached))
#define order_mutex 4
#define commits 5
#define published_stamp 6
#define stamp_at(reached) (7 + (reached))

// The order a group visits the stores in: ascending, unless one group is told otherwise.
#ifdef without_address_order
#define visited(group, position) ((group) == 1 -> 1 - (position) : (position))
#else
#define visited(group, position) (position)
#endif

// What the group's ghost state spells, as `staging_t` does.
#define pending 0
#define staged 1

byte groups_started;
byte finished;
byte staging[2];
bool staged_at[2 * stores];   // a participant holds its inner store's staged state
bool published_at[2 * stores]; // a participant published
bool refused_stage[2];
int watch_seen[2 * stores];    // the version each group watched at its stage
int stamp_of[2];               // the newest stamp each group drew, zero before it drew one
#define at(group, store) ((group) * stores + (store))

active [2] proctype group() {
    byte me, position, reached, staged_count, validated_through;
    int seen, version, drawn;
    bool refused;
    atomic { me = groups_started; groups_started++ };
    // stage: each participant under its store's unique lock, in order, the prefix unwound on a refusal: transaction_group::stage
    for (position : 0 .. stores - 1) {
        reached = visited(me, position);
        lock(me, mutex(reached));
        if
        :: reached == 1 -> refused = true // the inner store's refusal, which it may answer at any time
        :: skip ->
            load(me, watched(reached), order_relaxed, version);
            watch_seen[at(me, reached)] = version;
            staged_at[at(me, reached)] = true;
            staged_count++
        fi;
        unlock(me, mutex(reached));
        if :: refused -> break :: else fi
    };
    if
    :: refused ->
        refused_stage[me] = true;
#ifndef without_prefix_rollback
        do
        :: staged_count != 0 ->
            staged_count--;
            reached = visited(me, staged_count);
            lock(me, mutex(reached));
            staged_at[at(me, reached)] = false;
            unlock(me, mutex(reached))
        :: else -> break
        od;
#endif
        goto done
    :: else -> staging[me] = staged
    fi;
#if scenario != in_turn
    // commit, asking every participant before any writes: transaction_group::commit, its asks_before_writing_k branch
    // validate_for_commit holds the store's lock until the publication or the rollback: locked_store::transaction_t::validate_for_commit
    for (position : 0 .. stores - 1) {
        reached = visited(me, position);
#ifdef without_held_validation
        lock_shared(me, mutex(reached));
        load(me, watched(reached), order_relaxed, version);
        unlock_shared(me, mutex(reached));
#else
        lock(me, mutex(reached));
        validated_through = position;
        load(me, watched(reached), order_relaxed, version);
#endif
        if :: version != watch_seen[at(me, reached)] -> refused = true; break :: else fi
    };
    if
    :: refused ->
        // rollback, ascending: a participant holding its validation releases it, the rest take their lock: transaction_group::rollback
        for (position : 0 .. stores - 1) {
            reached = visited(me, position);
#ifdef without_held_validation
            lock(me, mutex(reached));
#else
            if :: position > validated_through -> lock(me, mutex(reached)) :: else fi;
#endif
            staged_at[at(me, reached)] = false;
            unlock(me, mutex(reached))
        };
        staging[me] = pending;
        goto done
    :: else
    fi;
#if scenario == one_stamp && !defined(without_one_stamp)
    // begin_commit: one stamp for the whole group, drawn under the order's mutex before any store is written
    lock(me, order_mutex);
    read_modify_write(me, commits, order_relaxed, seen, seen + 1);
    drawn = seen + 1;
    stamp_of[me] = drawn;
    unlock(me, order_mutex);
#endif
    for (position : 0 .. stores - 1) {
        reached = visited(me, position);
#ifdef without_held_validation
        lock(me, mutex(reached));
#endif
        // publish_under: nothing moved since the validation, or the update is lost: locked_store::transaction_t::publish_under
        assert(newest_value(watched(reached)) == watch_seen[at(me, reached)]);
        published_at[at(me, reached)] = true;
        staged_at[at(me, reached)] = false;
#if scenario == one_stamp
#ifdef without_one_stamp
        // a stamp per store, drawn and published as that store's own commit would
        lock(me, order_mutex);
        read_modify_write(me, commits, order_relaxed, seen, seen + 1);
        drawn = seen + 1;
        stamp_of[me] = drawn;
        unlock(me, order_mutex);
        store(me, stamp_at(reached), order_relaxed, drawn);
        unlock(me, mutex(reached));
        lock(me, order_mutex);
        store(me, published_stamp, order_relaxed, drawn);
        unlock(me, order_mutex)
#else
        store(me, stamp_at(reached), order_relaxed, drawn);
        unlock(me, mutex(reached))
#endif
#else
        unlock(me, mutex(reached))
#endif
    };
#if scenario == one_stamp && !defined(without_one_stamp)
    // end_commit: the watermark moves once, after the last store, under the order's mutex
    lock(me, order_mutex);
    store(me, published_stamp, order_relaxed, drawn);
    unlock(me, order_mutex);
#endif
    staging[me] = pending;
#else
    // commit in turn: each participant validates and publishes on its own, and a refusal past the first leaves the group pending: transaction_group::commit, its one-at-a-time branch
    for (position : 0 .. stores - 1) {
        reached = visited(me, position);
        lock(me, mutex(reached));
        load(me, watched(reached), order_relaxed, version);
        if
        :: version != watch_seen[at(me, reached)] ->
            unlock(me, mutex(reached));
            if :: position != 0 -> staging[me] = pending :: else fi;
            refused = true;
            break
        :: else ->
            published_at[at(me, reached)] = true;
            staged_at[at(me, reached)] = false;
            unlock(me, mutex(reached))
        fi
    };
    // the loop left a refusal past the first pending, and one at the first staged for `rollback`
    if
    :: refused -> skip
    :: else -> staging[me] = pending
    fi;
#endif
done:
    finished++
}

// a commit over the second store's watched key, under its unique lock, at any time
active proctype writer() {
    int seen;
    lock(writer_thread, mutex(1));
    read_modify_write(writer_thread, watched(1), order_relaxed, seen, seen + 1);
    unlock(writer_thread, mutex(1));
    finished++
}

#if scenario == one_stamp
// a reader: its snapshot drawn under the order's mutex, then each store under its shared lock; a group whose stamp
// the snapshot covers has written every store, so each store holds that stamp or a newer one
active proctype reader() {
    byte reached, each;
    int seen, snapshot, seen_stamp;
    lock(reader_thread, order_mutex);
    load(reader_thread, published_stamp, order_relaxed, snapshot);
    unlock(reader_thread, order_mutex);
    for (reached : 0 .. stores - 1) {
        lock_shared(reader_thread, mutex(reached));
        load(reader_thread, stamp_at(reached), order_relaxed, seen_stamp);
        for (each : 0 .. 1) {
            if
            :: stamp_of[each] != 0 && stamp_of[each] <= snapshot -> assert(seen_stamp >= stamp_of[each])
            :: else
            fi
        };
        unlock_shared(reader_thread, mutex(reached))
    }
}
#endif

// once everyone returned: the unwind left nothing staged, a group published all or none, and a torn commit ended pending
active proctype auditor() {
    byte each;
    (finished == 3);
    for (each : 0 .. 1) {
        if
        :: refused_stage[each] -> assert(!staged_at[at(each, 0)] && !staged_at[at(each, 1)])
        :: else
        fi;
#if scenario != in_turn || defined(whole_across_stores)
        assert(published_at[at(each, 0)] == published_at[at(each, 1)]);
#endif
#if scenario == in_turn
        // a commit that tore stopped calling itself staged, so a second one answers `operation_not_permitted_k`
        if
        :: published_at[at(each, 0)] != published_at[at(each, 1)] -> assert(staging[each] == pending)
        :: else
        fi;
#endif
    }
}
