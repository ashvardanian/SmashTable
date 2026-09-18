/**
 *  The staged batch of `basic_avl_tree`, `basic_flat_set` and `basic_hash_table`: every range
 *  modifier builds a container of its own kind beside the destination, meets both of the causes
 *  that can refuse in it, and only then absorbs it. These cores are single-writer, so there is
 *  one process here and no memory model; what the model explores is where a refusal lands.
 *
 *  The destination starts holding two keys of a three-key universe. The range is three elements
 *  of any keys of it, so a key already here, a key that is free and a key the range repeats all
 *  appear, and the verb is drawn the same way from the four the containers spell: `upsert` takes
 *  the newcomer, `insert_if_missing` keeps the incumbent, `insert` refuses the newcomer over a
 *  key already here and `update` refuses over a key that is not. Every request to the allocator
 *  may be refused and every duplication may refuse itself, which are the two causes.
 *
 *  `-Dscenario=` picks whose absorb step runs, since that is the step that differs:
 *  - `tree`, the default: the staging tree asks for a node per key it takes, and the merge that
 *    absorbs it relinks those nodes and asks for nothing at all;
 *  - `flat`: the staging array is reserved for the whole range at once, and the absorb asks once
 *    more for the merged array both runs are moved into;
 *  - `table`: the staging vector is reserved for the whole range at once, and the absorb asks
 *    once more through `reserve_more` for the room the writes then need.
 *
 *  Invariants:
 *  - a refusal leaves the destination exactly as it was, whichever cause refused and wherever it
 *    landed; a success leaves the union the verb owes, read off the range on its own;
 *  - nothing of the range reaches the destination before the staging is complete.
 *    `-Dwithout_staging` builds the range in the destination itself, the shape a batch had before
 *    it staged, and a refused element leaves the prefix behind it;
 *  - the absorb asks the allocator at most once, and for a tree not at all.
 *    `-Dwithout_secured_room` asks once per element instead, the shape a merge that copied rather
 *    than relinked would have, and a refusal part-way leaves some of the range moved;
 *  - the key check the two refusing verbs run happens before the absorb. `-Dwithout_prior_check`
 *    moves it after, where a refusal lands on a destination already merged, and where `update`'s
 *    check no longer sees the missing key at all, because the absorb has just written it in.
 */

// The knob's values are integers, so a typo fails the range check below.
#define tree 1
#define flat 2
#define table 3
#ifndef scenario
#define scenario tree
#endif
#if scenario < tree || scenario > table
#error "scenario is tree, flat or table"
#endif

// The verbs, named for the side a key held by both the range and the destination keeps.
#define takes_the_newcomer 1
#define keeps_the_incumbent 2
#define refuses_the_newcomer 3
#define updates_the_incumbent 4

// The statuses, as `success_k`, `out_of_memory_heap_k`, whatever a refused copy reported,
// `key_already_exists_k` and `key_not_found_k`.
#define success 0
#define out_of_memory 1
#define copy_refused 2
#define key_already_exists 3
#define key_not_found 4

#define keys 3
#define range_length 3
#define absent 0
#define payload_of(key, index) (10 * (key) + (index) + 1)

byte here[keys + 1];      // the destination, one payload per key
byte was[keys + 1];       // what it held before the batch
byte owed[keys + 1];      // what the verb owes when nothing refuses
byte staged[keys + 1];    // the container built beside it
byte range_key[range_length];
byte range_payload[range_length];
byte verb;
byte outcome;
byte owed_status;
byte touched;             // writes of this batch that reached the destination
byte absorb_requests;     // what the absorb asked the allocator for

// The allocator, free to refuse any request: the first of the two causes.
inline ask_allocator(granted) {
    if
    :: granted = true
    :: granted = false
    fi
}

// `copy_safely` on an element whose duplication can refuse: the second cause: shared.hpp:806-818
inline duplicate_element(granted) {
    if
    :: granted = true
    :: granted = false
    fi
}

#if scenario == tree
// The staging tree asks for one node per key it takes: basic_avl_tree.hpp:2163, 2170-2183
inline stage_room(granted) { ask_allocator(granted) }
#else
// The staging array and the staging vector fill room reserved for the whole range up front,
// so a key landing in one asks for nothing: basic_flat_set.hpp:331-333, basic_hash_table.hpp:1240-1244
inline stage_room(granted) { granted = true }
#endif

#if scenario == tree
// The merge relinks the nodes the staging tree holds, so the absorb asks for nothing and cannot
// refuse part-way: basic_avl_tree.hpp:1404-1428
inline absorb_room(granted) { granted = true }
#else
// The flat set's merged array and the table's `reserve_more`: one request for the whole absorb,
// made before a single element moves: basic_flat_set.hpp:375-377, basic_hash_table.hpp:1246-1247
inline absorb_room(granted) { absorb_requests++; ask_allocator(granted) }
#endif

#ifdef without_staging
// Without the staging every element goes straight where it will live, which is the shape a batch
// had before it staged; `touched` counts what a refusal then has nowhere to undo.
inline stage_one(where, value) { here[where] = value; touched++ }
#define built(key) here[key]
#else
inline stage_one(where, value) { staged[where] = value }
#define built(key) staged[key]
#endif

// The range checked against the destination before a single element moves, which is the only
// thing the two refusing verbs do that the other two do not: basic_avl_tree.hpp:2243-2245, 2364-2366
inline check_destination(key) {
    for (key : 1 .. keys) {
        if
        :: verb == refuses_the_newcomer && staged[key] != absent && here[key] != absent ->
            outcome = key_already_exists
        :: verb == updates_the_incumbent && staged[key] != absent && here[key] == absent ->
            outcome = key_not_found
        :: else
        fi
    }
}

// One element of the staging container landing in the destination, the side a key held on both
// sides keeps: basic_avl_tree.hpp:1414-1424, basic_flat_set.hpp:379-393
inline place(where) {
    if
    :: verb == keeps_the_incumbent && here[where] != absent -> skip
    :: else -> here[where] = staged[where]; touched++
    fi
}

active proctype batch() {
    byte index, key;
    bool granted;

    // The destination before the batch: two of the three keys, one of them free for the range.
    here[1] = 10;
    here[3] = 30;
    for (key : 1 .. keys) { was[key] = here[key] };

    // The range and the verb, both open, so every shape of overlap is explored.
    for (index : 0 .. range_length - 1) {
        if
        :: range_key[index] = 1
        :: range_key[index] = 2
        :: range_key[index] = 3
        fi;
        range_payload[index] = payload_of(range_key[index], index)
    };
    if
    :: verb = takes_the_newcomer
    :: verb = keeps_the_incumbent
    :: verb = refuses_the_newcomer
    :: verb = updates_the_incumbent
    fi;

    // What the verb owes when nothing refuses, read off the range and the destination alone.
    for (key : 1 .. keys) { owed[key] = was[key] };
    for (index : 0 .. range_length - 1) {
        key = range_key[index];
        if
        :: verb == takes_the_newcomer -> owed[key] = range_payload[index]
        :: verb == keeps_the_incumbent ->
            if
            :: owed[key] == absent -> owed[key] = range_payload[index]
            :: else
            fi
        :: verb == refuses_the_newcomer ->
            if
            :: owed[key] != absent -> owed_status = key_already_exists
            :: else -> owed[key] = range_payload[index]
            fi
        :: else ->
            if
            :: was[key] == absent -> owed_status = key_not_found
            :: else -> owed[key] = range_payload[index]
            fi
        fi
    };

#if scenario != tree
    // The room the whole range will take, asked for once before it is filled.
    ask_allocator(granted);
    if
    :: !granted -> outcome = out_of_memory; goto refused
    :: else
    fi;
#endif

    // The staging: every element duplicated outside the destination and placed in a container of
    // the destination's own kind: basic_avl_tree.hpp:2155-2184, basic_flat_set.hpp:326-346
    for (index : 0 .. range_length - 1) {
        duplicate_element(granted);
        if
        :: !granted -> outcome = copy_refused; goto refused
        :: else
        fi;
        key = range_key[index];
        if
        :: built(key) != absent ->
            // A key the range repeats collapses the way the verb's own name reads.
            if
            :: verb == refuses_the_newcomer -> outcome = key_already_exists; goto refused
            :: verb == keeps_the_incumbent -> skip
            :: else -> stage_one(key, range_payload[index])
            fi
        :: else ->
            stage_room(granted);
            if
            :: !granted -> outcome = out_of_memory; goto refused
            :: else
            fi;
            stage_one(key, range_payload[index])
        fi
    };

    // Nothing of the range has reached the destination yet: basic_avl_tree.hpp:2205-2211
    assert(touched == 0);

#ifndef without_prior_check
    check_destination(key);
    if
    :: outcome != success -> goto refused
    :: else
    fi;
#endif

#ifdef without_secured_room
    // One request per element, so a refusal part-way has already moved the elements before it.
    for (key : 1 .. keys) {
        if
        :: staged[key] != absent ->
            absorb_requests++;
            ask_allocator(granted);
            if
            :: !granted -> outcome = out_of_memory; goto refused
            :: else
            fi;
            place(key)
        :: else
        fi
    }
#else
    absorb_room(granted);
    if
    :: !granted -> outcome = out_of_memory; goto refused
    :: else
    fi;
    for (key : 1 .. keys) {
        if
        :: staged[key] != absent -> place(key)
        :: else
        fi
    }
#endif

#ifdef without_prior_check
    check_destination(key);
#endif

refused:
    // What the batch left behind: the union the verb owes when it answered success, and exactly
    // what was here when it refused, whichever cause refused and wherever it landed.
    for (key : 1 .. keys) {
        if
        :: outcome == success -> assert(here[key] == owed[key])
        :: else -> assert(here[key] == was[key])
        fi
    };

    // The refusal the range predicted, or one of the two causes.
    if
    :: outcome == success -> assert(owed_status == success)
    :: outcome == key_already_exists -> assert(owed_status == key_already_exists)
    :: outcome == key_not_found -> assert(owed_status == key_not_found)
    :: else -> assert(outcome == out_of_memory || outcome == copy_refused)
    fi;

    // The absorb asked at most once, and for a tree not at all.
    assert(absorb_requests <= 1);
#if scenario == tree
    assert(absorb_requests == 0);
#endif
}
