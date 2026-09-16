/**
 *  The pinned reader of `include/smashtable/snapshot_store.hpp` against concurrent commits and
 *  reclamation: a reader's claim joins the clock's census under its mutex at the watermark, every
 *  read answers at that stamp under the partition's shared lock with nothing of the reader's own
 *  written, and each commit prunes the key's version run under the partition lock down to the
 *  low-water mark, which it reads without the clock's mutex.
 *
 *  Two committers write one key and one reader reads it twice. Under `-Dscenario=adoption` a
 *  transaction adopts the reader's stamp, the reader closes first, and the transaction reads the
 *  key after whatever the committers pruned meanwhile.
 *
 *  Invariants:
 *  - every read resolves the key to the version the stamp names, and that version is still there:
 *    the claim holds the low-water mark at or below the stamp. `-Dwithout_lease` draws the stamp from
 *    the watermark and joins no census, and a committer prunes the version from under the reader;
 *  - an adopting transaction keeps the stamp pinned after the reader closes, because it links a
 *    claim of its own beside the reader's. `-Dwithout_shared_lease` copies the stamp alone, and the
 *    prune after the reader closes frees the version the transaction reads.
 */
#include "weak_memory.pml"
#include "spin_shared_mutex.pml"

// The knob's values are integers, so a typo fails the range check below.
#define reading 1
#define adoption 2
#ifndef scenario
#define scenario reading
#endif
#if scenario < reading || scenario > adoption
#error "scenario is reading or adoption"
#endif

// The threads, by role, apart from the processes that play them; the committers name themselves.
#define reader_thread 2

// The words: the partition's mutex, and the clock's mutex, stamp counter, watermark and low-water mark.
#define partition_mutex 0
#define clock_mutex 1
#define commits 2
#define published_stamp 3
#define low_water_mark 4

// The key's version run is ordered by the partition lock alone, so it is ghost state rather than words.
#define versions 3
#define none 99
#define stamp_of(version) ((version) == 0 -> 0 : version_stamp[version])

byte committers_started;
int version_stamp[versions] = none; // the stamp each version was published under; version 0 is stamp 0
bool version_freed[versions];
byte versions_written = 1;

// The census the clock's mutex guards: which claims are live, and the snapshot each names.
bool reader_live;
int reader_snapshot;
bool adopter_live;
int adopter_snapshot;

// republish_mark_: the oldest claim's snapshot, or the watermark when nobody reads
inline republish_mark(t) {
    if
    :: reader_live && (!adopter_live || reader_snapshot <= adopter_snapshot) ->
        store(t, low_water_mark, order_relaxed, reader_snapshot)
    :: adopter_live && (!reader_live || adopter_snapshot < reader_snapshot) ->
        store(t, low_water_mark, order_relaxed, adopter_snapshot)
    :: else -> store(t, low_water_mark, order_relaxed, newest_value(published_stamp))
    fi
}

// visible_version_: the newest version whose stamp the snapshot covers, freed or not
inline resolve(snapshot, found) {
    found = none;
    for (each : 0 .. versions - 1) {
        if
        :: stamp_of(each) != none && stamp_of(each) <= snapshot &&
           (found == none || stamp_of(each) > stamp_of(found)) -> found = each
        :: else
        fi
    }
}

active [2] proctype committer() {
    byte me, slot, each, survivor;
    int seen, drawn, mark;
    atomic { me = committers_started; committers_started++ };
    lock(me, partition_mutex);
    // begin_commit: the stamp drawn under the clock's mutex
    lock(me, clock_mutex);
    read_modify_write(me, commits, order_relaxed, seen, seen + 1);
    drawn = seen + 1;
    unlock(me, clock_mutex);
    // publish_under: the version joins the run under the partition lock
    atomic { slot = versions_written; versions_written++; version_stamp[slot] = drawn };
    // end_commit: the watermark moves and the mark is republished, both under the clock's mutex
    lock(me, clock_mutex);
    store(me, published_stamp, order_relaxed, drawn);
    republish_mark(me);
    unlock(me, clock_mutex);
    // prune_committed: the mark read bare, then every version it covers except its survivor freed
    load(me, low_water_mark, order_relaxed, mark);
    resolve(mark, survivor);
    for (each : 0 .. versions - 1) {
        if
        :: stamp_of(each) != none && stamp_of(each) <= mark && each != survivor -> version_freed[each] = true
        :: else
        fi
    };
    unlock(me, partition_mutex)
}

active proctype reader() {
    byte each, first, second;
    int seen, snapshot;
#ifdef without_lease
    load(reader_thread, published_stamp, order_relaxed, snapshot);
#else
    // reader_t: take_snapshot links the claim at the newest end of the census
    lock(reader_thread, clock_mutex);
    load(reader_thread, published_stamp, order_relaxed, snapshot);
    atomic { reader_live = true; reader_snapshot = snapshot };
    republish_mark(reader_thread);
    unlock(reader_thread, clock_mutex);
#endif

#if scenario == reading
    // reader_t::find, twice, each under the partition's shared lock
    lock_shared(reader_thread, partition_mutex);
    resolve(snapshot, first);
    assert(first != none && !version_freed[first]);
    unlock_shared(reader_thread, partition_mutex);
    lock_shared(reader_thread, partition_mutex);
    resolve(snapshot, second);
    assert(second == first && !version_freed[second]);
    unlock_shared(reader_thread, partition_mutex);
#else
    // share_snapshot: the transaction's claim linked beside the reader's, at the same snapshot
    lock(reader_thread, clock_mutex);
#ifdef without_shared_lease
    adopter_snapshot = snapshot;
#else
    atomic { adopter_live = true; adopter_snapshot = snapshot };
    republish_mark(reader_thread);
#endif
    unlock(reader_thread, clock_mutex);
#endif

    // the reader closes: retire_snapshot_
    lock(reader_thread, clock_mutex);
    reader_live = false;
    republish_mark(reader_thread);
    unlock(reader_thread, clock_mutex);

#if scenario == adoption
    // the adopting transaction reads at the adopted stamp, after the reader closed; its own claim
    // retires past the last assertion, so the model leaves it held to keep the mutex's history short
    lock_shared(reader_thread, partition_mutex);
    resolve(adopter_snapshot, first);
    assert(first != none && !version_freed[first]);
    unlock_shared(reader_thread, partition_mutex)
#endif
}
