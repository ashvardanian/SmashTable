/**
 *  @file verification/staged_batch/protocol.pml
 *  @author Ash Vardanian
 *  @date September 18, 2026
 *  @brief The staged batch of @c basic_avl_tree, @c basic_flat_set and @c basic_hash_table: the
 *      range built beside the destination, and the refusal that leaves it as it was.
 *
 *  Every range modifier builds a container of its own kind beside the destination, meets both of
 *  the causes that can refuse in it, and only then absorbs it. These cores are single-writer, so
 *  there is one process here and no memory model; what the model explores is where a refusal lands.
 *
 *  The destination starts holding two keys of a three-key universe. The range is three elements of
 *  any keys of it, so a key already here, a key that is free and a key the range repeats all
 *  appear, and the verb is drawn the same way from the four the containers spell: @c upsert takes
 *  the newcomer, @c insert_if_missing keeps the incumbent, @c insert refuses the newcomer over a
 *  key already here and @c update refuses over a key that is not. Only @c basic_avl_tree spells
 *  @c update over a range, so the fourth verb stands for the tree alone. Every request to the
 *  allocator may be refused and every duplication may refuse itself, which are the two causes.
 *
 *  Invariants:
 *  - a refusal leaves the destination exactly as it was, whichever cause refused and wherever it
 *    landed; a success leaves the union the verb owes, read off the range on its own;
 *  - nothing of the range reaches the destination before the staging is complete;
 *  - the absorb asks the allocator at most once, and for a tree not at all;
 *  - the key check the two refusing verbs run happens before the absorb.
 *
 *  Three keys and a range of three are enough for a key already there, a key that is free and a
 *  key the range repeats, and not enough for the probe of an open table. Every scenario beside this
 *  file defines @c relinks, whether its absorb relinks the staged nodes rather than moving them.
 */

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

// The knobs: the header's choices, each overridden by a `@verify` line that replays its counterexample.
#ifndef staging
#define staging true
#endif
#ifndef secured_room
#define secured_room true
#endif
#ifndef prior_check
#define prior_check true
#endif

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

/** Where the range is built: beside the destination, or with `staging` off, straight in it. */
#define built(key) (staging -> staged[key] : here[key])

/** The allocator, free to refuse any request: the first of the two causes. */
inline ask_allocator(granted) {
    if
    :: granted = true
    :: granted = false
    fi
}

/** @c copy_safely on an element whose duplication can refuse, the second cause:
 *  @c stage_each in `shared.hpp`. */
inline duplicate_element(granted) {
    if
    :: granted = true
    :: granted = false
    fi
}

/** The room a staged key takes. The staging tree asks for one node per key it takes:
 *  @c basic_avl_tree::stage_range_ in `basic_avl_tree.hpp`. The staging array and the staging vector
 *  fill room reserved for the whole range up front, so a key landing in one asks for nothing:
 *  @c basic_flat_set::stage_range_ in `basic_flat_set.hpp` and @c basic_hash_table::absorb_range_
 *  in `basic_hash_table.hpp`. */
inline stage_room(granted) {
    if
    :: relinks -> ask_allocator(granted)
    :: else -> granted = true
    fi
}

/** The room the absorb takes. The merge relinks the nodes the staging tree holds, so it asks for
 *  nothing and cannot refuse part-way: @c basic_avl_tree::merge and
 *  @c basic_avl_tree::merge_with_upsert in `basic_avl_tree.hpp`. The flat set's merged array and the
 *  table's @c reserve_more are one request for the whole absorb, made before a single element
 *  moves: @c basic_flat_set::absorb in `basic_flat_set.hpp` and @c basic_hash_table::reserve_more in
 *  `basic_hash_table.hpp`. */
inline absorb_room(granted) {
    if
    :: relinks -> granted = true
    :: else -> absorb_requests++; ask_allocator(granted)
    fi
}

/** Every element goes into the container built beside the destination; with `staging` off, it goes
 *  straight where it will live, and @c touched counts what a refusal then has nowhere to undo. */
inline stage_one(where, value) {
    if
    :: staging -> staged[where] = value
    :: else -> here[where] = value; touched++
    fi
}

/** The range checked against the destination before a single element moves, which is the only thing
 *  the two refusing verbs do that the other two do not: @c basic_avl_tree::has_any_key and
 *  @c basic_avl_tree::has_all_keys in `basic_avl_tree.hpp`. */
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

/** One element of the staging container landing in the destination, the side a key held on both
 *  sides keeps: @c basic_avl_tree::merge_with_upsert in `basic_avl_tree.hpp` and
 *  @c basic_flat_set::absorb in `basic_flat_set.hpp`. */
inline place(where) {
    if
    :: verb == keeps_the_incumbent && here[where] != absent -> skip
    :: else -> here[where] = staged[where]; touched++
    fi
}

/** One range modifier: the range staged, checked and absorbed, then what it left behind audited. */
inline modify_range() {
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

    // The room the whole range will take, asked for once before it is filled, where it is reserved
    if
    :: !relinks ->
        ask_allocator(granted);
        if
        :: !granted -> outcome = out_of_memory; goto refused
        :: else
        fi
    :: else
    fi;

    // The staging: every element duplicated outside the destination and placed in a container of
    // the destination's own kind: `basic_avl_tree::stage_range_` in `basic_avl_tree.hpp` and
    // `basic_flat_set::stage_range_` in `basic_flat_set.hpp`
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

    // Nothing of the range has reached the destination yet: `basic_avl_tree::insert_if_missing` in
    // `basic_avl_tree.hpp`, before its merge
    assert(touched == 0);

    if
    :: prior_check ->
        check_destination(key);
        if
        :: outcome != success -> goto refused
        :: else
        fi
    :: else
    fi;

    if
    :: secured_room ->
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
    :: else ->
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
    fi;

    if
    :: !prior_check -> check_destination(key)
    :: else
    fi;

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
    assert(!relinks || absorb_requests == 0)
}
