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
 *  participant validates and publishes on its own, one store at a time.
 *
 *  Invariants:
 *  - no lost update: at every publication the watched key still holds what the group watched.
 *    `-Dwithout_held_validation` validates under a shared lock it drops before the unique lock
 *    of the publication, the header before the fix, and the writer slips in between;
 *  - address order: two groups holding across their phases never deadlock, since both take
 *    the stores ascending. `-Dwithout_address_order` reverses one group's order and Spin finds
 *    the wait cycle as an invalid end state;
 *  - all or none: a group that returned success published every participant, and one that was
 *    refused published none, where the stores split their commit. Under `in_turn` the docs
 *    promise less: `-Dwhole_across_stores` asserts all or none there too and fails, the tear
 *    the docs admit, while the scenario itself asserts what they do promise, that a refusal
 *    past the first participant leaves the group pending and a second commit refused;
 *  - the unwind: after a refused stage no store holds the group's staged mark.
 *    `-Dwithout_prefix_rollback` leaves the first store staged.
 */
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

// The knob's values are integers, so a typo fails the range check below.
#define two_pass 1
#define in_turn 2
#ifndef scenario
#define scenario two_pass
#endif
#if scenario < two_pass || scenario > in_turn
#error "scenario is two_pass or in_turn"
#endif

// The threads, by role, apart from the processes that play them; the groups name themselves.
#define writer_thread 2
#define stores 2

// The words: each store's mutex, and the version of the key each group watches there.
#define mutex(reached) (reached)
#define watched(reached) (2 + (reached))

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
#define at(group, store) ((group) * stores + (store))

active [2] proctype group() {
    byte me, position, reached, staged_count, validated_through;
    int seen, version;
    bool refused;
    atomic { me = groups_started; groups_started++ };
    // stage: each participant under its store's unique lock, in order, the prefix unwound on a refusal: shared.hpp:1970-1991
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
#if scenario == two_pass
    // commit, asking every participant before any writes: shared.hpp:2024-2036
    // validate_for_commit holds the store's lock until the publication or the rollback: locked_store.hpp:699-704
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
        // rollback, ascending: a participant holding its validation releases it, the rest take their lock: shared.hpp:2063-2071
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
    for (position : 0 .. stores - 1) {
        reached = visited(me, position);
#ifdef without_held_validation
        lock(me, mutex(reached));
#endif
        // publish_under: nothing moved since the validation, or the update is lost: locked_store.hpp:715-720
        assert(newest_value(watched(reached)) == watch_seen[at(me, reached)]);
        published_at[at(me, reached)] = true;
        staged_at[at(me, reached)] = false;
        unlock(me, mutex(reached))
    };
    staging[me] = pending;
#else
    // commit in turn: each participant validates and publishes on its own, and a refusal past the first leaves the group pending: shared.hpp:2039-2049
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
            assert(newest_value(watched(reached)) == watch_seen[at(me, reached)]);
            published_at[at(me, reached)] = true;
            staged_at[at(me, reached)] = false;
            unlock(me, mutex(reached))
        fi
    };
    if
    :: refused && published_at[at(me, 0)] ->
        // the documented claims: pending, and a second commit refused as not permitted
        assert(staging[me] == pending)
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

// once everyone returned: the unwind left nothing staged, and a group published all or none
active proctype auditor() {
    byte each;
    (finished == 3);
    for (each : 0 .. 1) {
        if
        :: refused_stage[each] -> assert(!staged_at[at(each, 0)] && !staged_at[at(each, 1)])
        :: else
        fi;
#if scenario == two_pass || defined(whole_across_stores)
        assert(published_at[at(each, 0)] == published_at[at(each, 1)]);
#endif
    }
}
