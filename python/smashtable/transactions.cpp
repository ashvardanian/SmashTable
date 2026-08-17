/**
 *  @brief @c Transaction and @c View - the group that makes one update span several containers.
 *  @author Ash Vardanian
 *  @file python/smashtable/transactions.cpp
 *  @date October 30, 2025
 *
 *  The unit of atomicity is the group, not the container: a @c with block opens one participant per
 *  container and the block either makes every change visible or none of them.
 *
 *  @section transactions_alternatives Why the Participants Are a Variant
 *
 *  A group may mix maps and sets, whose store types differ. That set of stores is closed and known at
 *  compile time, so participants are held in a @c std::variant rather than behind a base class: every
 *  uniform operation is one @c std::visit, which lowers to a switch with no vtable, no indirect call
 *  and no allocation per participant.
 *
 *  @section transactions_ordering Why the Order Is Canonical
 *
 *  Participants are sorted by a process-wide container ordinal before anything is staged, so two groups
 *  sharing containers acquire their partition locks in the same sequence and cannot deadlock on each
 *  other. The views handed back to Python stay in the caller's argument order regardless.
 */
#include <algorithm> // `std::sort`

#include "shared.hpp"

namespace ashvardanian::smashtable::py {

#pragma region View

/** @brief The strictest mode across a group, since one object participant governs the whole pass. */
static value_mode_t group_mode(transaction_object_t const *group) noexcept {
    for (auto const &participant : group->parts)
        if (participant.mode == value_mode_t::objects_k) return value_mode_t::objects_k;
    return value_mode_t::scalars_k;
}

/**
 *  @brief This view's participant. Its layout and family are fixed for the group's life.
 *
 *  Whether the group is still open is not, so only the immutable parts - @c ops, @c mode and which
 *  alternative is engaged - may be read from it outside @c run_over_participant.
 */
static participant_t *part_of(view_object_t *view) noexcept {
    auto *group = object_as<transaction_object_t>(view->owner);
    return &group->parts[view->index];
}

/**
 *  @brief Runs one participant operation under its group's lock, refusing once the group has finished.
 *  @return 0 when @p operation ran; -1 with a @c StateError set when the group was already finished.
 *
 *  The state test and the operation are one span, so a concurrent commit either happens entirely
 *  before this or entirely after, never between the test and the write it guards.
 */
template <typename operation_type_>
static int run_over_participant(view_object_t *view, module_state_t *state, operation_type_ &&operation) noexcept {
    auto *group = object_as<transaction_object_t>(view->owner);
    participant_t &part = group->parts[view->index];
    bool finished = false;
    run_over_values(part.mode, group->lock, [&]() noexcept {
        finished = group->state == group_state_t::finished_k;
        if (!finished) operation(part);
    });
    if (!finished) return 0;
    PyErr_SetString(state->state_error, "this transaction is no longer active");
    return -1;
}

static void View_dealloc(PyObject *self) noexcept {
    auto *view = object_as<view_object_t>(self);
    PyObject_GC_UnTrack(self);
    Py_CLEAR(view->owner);
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

static int View_traverse(PyObject *self, visitproc visit, void *arg) noexcept {
    auto *view = object_as<view_object_t>(self);
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(view->owner);
    return 0;
}

static int View_clear(PyObject *self) noexcept {
    Py_CLEAR(object_as<view_object_t>(self)->owner);
    return 0;
}

static PyObject *View_subscript(PyObject *self, PyObject *key) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    participant_t *part = part_of(view);
    if (!part->is_associative()) {
        PyErr_SetString(PyExc_TypeError, "this participant is a set, which has no values to read");
        return nullptr;
    }

    key_variant_t needle;
    if (!key_from_python(key, part->ops, needle)) return nullptr;

    value_variant_t found;
    bool present = false;
    if (run_over_participant(view, state, [&](participant_t &part) noexcept { present = part.find(needle, found); }) !=
        0)
        return nullptr;

    if (!present) {
        PyErr_SetObject(PyExc_KeyError, key);
        return nullptr;
    }
    return value_to_python(found);
}

static int View_assign_subscript(PyObject *self, PyObject *key, PyObject *value) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;
    participant_t *part = part_of(view);

    key_variant_t stored_key;
    if (!key_from_python(key, part->ops, stored_key)) return -1;

    status_t status = success_k;
    if (!value) { // `del view[key]`
        if (run_over_participant(view, state, [&](participant_t &part) noexcept { status = part.erase(stored_key); }) !=
            0)
            return -1;
        return raise_for(state, status, key);
    }

    if (!part->is_associative()) {
        PyErr_SetString(PyExc_TypeError, "this participant is a set; use add() rather than assigning a value");
        return -1;
    }
    value_variant_t stored_value;
    if (!value_from_python(
            value,
            object_as<container_object_t>(PyTuple_GET_ITEM(object_as<transaction_object_t>(view->owner)->containers, 0))
                ->mode,
            stored_value))
        return -1;
    if (run_over_participant(view, state, [&](participant_t &part) noexcept {
            status = part.upsert(std::move(stored_key), std::move(stored_value));
        }) != 0)
        return -1;
    return raise_for(state, status, key);
}

static int View_contains(PyObject *self, PyObject *key) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;
    participant_t *part = part_of(view);

    key_variant_t needle;
    if (!key_from_python(key, part->ops, needle)) {
        PyErr_Clear();
        return 0;
    }
    bool found = false;
    if (run_over_participant(view, state, [&](participant_t &part) noexcept { found = part.contains(needle); }) != 0)
        return -1;
    return found ? 1 : 0;
}

static char const doc_View_get[] =                                                       //
    "get(key, default=None, /)\n"                                                        //
    "\n"                                                                                 //
    "Value for a key inside this transaction, or default when the key is absent.\n"      //
    "\n"                                                                                 //
    "Reads this transaction's own uncommitted writes, and otherwise the state as of\n"   //
    "when the group opened. It does NOT see writes another transaction has staged but\n" //
    "not committed, so a read is never dirty.\n"                                         //
    "\n"                                                                                 //
    "Repeating a read is not guaranteed to return the same value: a committed write\n"   //
    "from elsewhere becomes visible immediately. Call watch(key) to make the\n"          //
    "transaction refuse to commit if that happens.\n"                                    //
    "\n"                                                                                 //
    "Raises:\n"                                                                          //
    "  TypeError: If key is not of this container's key type, or this is a set.\n"       //
    "  StateError: If the group has already finished.\n";                                //

static PyObject *View_get(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count < 1 || count > 2) {
        PyErr_SetString(PyExc_TypeError, "get() takes one or two arguments");
        return nullptr;
    }
    PyObject *found = View_subscript(self, args[0]);
    if (found) return found;
    if (!PyErr_ExceptionMatches(PyExc_KeyError)) return nullptr;
    PyErr_Clear();
    return Py_NewRef(count == 2 ? args[1] : Py_None);
}

static char const doc_View_upsert[] =                                                  //
    "upsert(key, value, /)\n"                                                          //
    "\n"                                                                               //
    "Insert or overwrite one key, invisibly until the group commits.\n"                //
    "\n"                                                                               //
    "The write is durable within the transaction - a later read through this view\n"   //
    "returns it - but no other reader can observe it, not even after stage(), until\n" //
    "commit() publishes the whole group at once.\n"                                    //
    "\n"                                                                               //
    "Raises:\n"                                                                        //
    "  TypeError: If key is not of this container's key type, or this is a set.\n"     //
    "  StateError: If the group has already finished.\n";                              //

static PyObject *View_upsert(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 2) {
        PyErr_SetString(PyExc_TypeError, "upsert() takes exactly two arguments");
        return nullptr;
    }
    if (View_assign_subscript(self, args[0], args[1]) != 0) return nullptr;
    Py_RETURN_NONE;
}

static char const doc_View_add[] =                                                   //
    "add(member, /)\n"                                                               //
    "\n"                                                                             //
    "Insert one member into a set participant, invisibly until the group commits.\n" //
    "\n"                                                                             //
    "Idempotent, and subject to the same visibility rule as upsert.\n"               //
    "\n"                                                                             //
    "Raises:\n"                                                                      //
    "  TypeError: If this participant is a map, which needs a value.\n"              //
    "  StateError: If the group has already finished.\n";                            //

static PyObject *View_add(PyObject *self, PyObject *member) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    participant_t *part = part_of(view);
    if (part->is_associative()) {
        PyErr_SetString(PyExc_TypeError, "this participant is a map; assign a value rather than calling add()");
        return nullptr;
    }

    key_variant_t stored;
    if (!key_from_python(member, part->ops, stored)) return nullptr;
    status_t status = success_k;
    if (run_over_participant(view, state,
                             [&](participant_t &part) noexcept { status = part.add(std::move(stored)); }) != 0)
        return nullptr;
    if (raise_for(state, status, member) != 0) return nullptr;
    Py_RETURN_NONE;
}

static char const doc_View_erase[] =                                              //
    "erase(key, /)\n"                                                             //
    "\n"                                                                          //
    "Remove a key, reporting whether it was there.\n"                             //
    "\n"                                                                          //
    "Returns:\n"                                                                  //
    "  bool: True when the key was present, so a caller need not look first.\n"   //
    "\n"                                                                          //
    "The removal is invisible outside the group until commit, and is undone by\n" //
    "rollback or by the block raising.\n"                                         //
    "\n"                                                                          //
    "Raises:\n"                                                                   //
    "  TypeError: If key is not of this container's key type.\n"                  //
    "  StateError: If the group has already finished.\n";                         //

static PyObject *View_erase(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "erase() takes exactly one argument");
        return nullptr;
    }
    participant_t *part = part_of(view);

    key_variant_t stored;
    if (!key_from_python(args[0], part->ops, stored)) return nullptr;
    bool present = false;
    status_t status = success_k;
    // `erase` on a map destroys the stored value, so this is a value operation even though its
    // argument is only a key.
    if (run_over_participant(view, state, [&](participant_t &part) noexcept {
            present = part.contains(stored);
            if (present) status = part.erase(stored);
        }) != 0)
        return nullptr;
    if (present && raise_for(state, status, args[0]) != 0) return nullptr;
    return PyBool_FromLong(present ? 1 : 0);
}

static char const doc_View_watch[] =                                                    //
    "watch(key, /)\n"                                                                   //
    "\n"                                                                                //
    "Refuse to commit if another writer touches this key first.\n"                      //
    "\n"                                                                                //
    "This is what turns read-modify-write into a lost-update-free operation. Without\n" //
    "it a group is atomic but not serializable: two transactions can each read the\n"   //
    "same value, each write back, and one update is silently lost.\n"                   //
    "\n"                                                                                //
    "Covers absence as well as presence, so watching a key that does not exist still\n" //
    "conflicts if someone inserts it. The conflict surfaces at stage(), not at the\n"   //
    "write, and nothing is applied when it does.\n"                                     //
    "\n"                                                                                //
    "Raises:\n"                                                                         //
    "  ConflictError: At stage() time, never here.\n"                                   //
    "  TypeError: If key is not of this container's key type.\n"                        //
    "  StateError: If the group has already finished.\n";                               //

static PyObject *View_watch(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "watch() takes exactly one argument");
        return nullptr;
    }
    participant_t *part = part_of(view);

    key_variant_t stored;
    if (!key_from_python(args[0], part->ops, stored)) return nullptr;
    status_t status = success_k;
    if (run_over_participant(view, state, [&](participant_t &part) noexcept { status = part.watch(stored); }) != 0)
        return nullptr;
    if (raise_for(state, status, args[0]) != 0) return nullptr;
    Py_RETURN_NONE;
}

static char const doc_View_update[] =                                                   //
    "update(other, /)\n"                                                                //
    "\n"                                                                                //
    "Apply every pair of a mapping to this participant.\n"                              //
    "\n"                                                                                //
    "Atomic, unlike SortedMap.update: every pair lands with the rest of the group or\n" //
    "none of them does. A failure part-way leaves the transaction abandonable by\n"     //
    "rollback with nothing applied.\n"                                                  //
    "\n"                                                                                //
    "Raises:\n"                                                                         //
    "  TypeError: If other is not a mapping, or a key is of the wrong type.\n"          //
    "  StateError: If the group has already finished.\n";                               //

static PyObject *View_update(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "update() takes exactly one argument");
        return nullptr;
    }
    PyObject *pairs = PyMapping_Items(args[0]);
    if (!pairs) return nullptr;
    PyObject *fast = PySequence_Fast(pairs, "update() needs a mapping");
    Py_DECREF(pairs);
    if (!fast) return nullptr;

    Py_ssize_t const total = PySequence_Fast_GET_SIZE(fast);
    for (Py_ssize_t index = 0; index != total; ++index) {
        PyObject *pair = PySequence_Fast_GET_ITEM(fast, index);
        PyObject *key = nullptr;
        PyObject *value = nullptr;
        if (!PyArg_ParseTuple(pair, "OO", &key, &value) || View_assign_subscript(self, key, value) != 0) {
            Py_DECREF(fast);
            return nullptr;
        }
    }
    Py_DECREF(fast);
    Py_RETURN_NONE;
}

template <typename function_type_>
static PyCFunction as_pycfunction(function_type_ function) noexcept {
    return reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(function));
}

static PyMethodDef View_methods[] = {
    {"get", as_pycfunction(View_get), METH_FASTCALL, doc_View_get},
    {"upsert", as_pycfunction(View_upsert), METH_FASTCALL, doc_View_upsert},
    {"add", View_add, METH_O, doc_View_add},
    {"erase", as_pycfunction(View_erase), METH_FASTCALL, doc_View_erase},
    {"watch", as_pycfunction(View_watch), METH_FASTCALL, doc_View_watch},
    {"update", as_pycfunction(View_update), METH_FASTCALL, doc_View_update},
    {nullptr, nullptr, 0, nullptr},
};

static char const doc_View[] =                                                         //
    "One container's slice of a transaction.\n"                                        //
    "\n"                                                                               //
    "Reads see this transaction's own writes and the state as of when the group\n"     //
    "opened, never another transaction's staged-but-uncommitted writes. Writes stay\n" //
    "invisible to every other reader until the group commits, and vanish if it does\n" //
    "not. A view stops working once its group finishes.\n";                            //

static PyType_Slot view_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(View_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(View_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(View_clear)},
    {Py_tp_methods, reinterpret_cast<void *>(View_methods)},
    {Py_mp_subscript, reinterpret_cast<void *>(View_subscript)},
    {Py_mp_ass_subscript, reinterpret_cast<void *>(View_assign_subscript)},
    {Py_sq_contains, reinterpret_cast<void *>(View_contains)},
    {Py_tp_doc, const_cast<char *>(doc_View)},
    {0, nullptr},
};

PyType_Spec view_spec = {
    "smashtable.View", sizeof(view_object_t), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION, view_slots};

#pragma endregion View

#pragma region Transaction

static void Transaction_dealloc(PyObject *self) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    PyObject_GC_UnTrack(self);
    group->parts.~basic_vector();
    group->lock.~object_lock_t();
    Py_CLEAR(group->views);
    Py_CLEAR(group->containers);
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

static int Transaction_traverse(PyObject *self, visitproc visit, void *arg) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(group->containers);
    Py_VISIT(group->views);
    return 0;
}

static int Transaction_clear(PyObject *self) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    Py_CLEAR(group->views);
    Py_CLEAR(group->containers);
    return 0;
}

/**
 *  @brief Discards every participant's staged and pending changes, leaving the stores untouched.
 *  @param[in] ending Where the group stands afterwards - open again, or finished for good.
 */
static void transaction_reset_all(transaction_object_t *group, group_state_t ending) noexcept {
    run_over_values(group_mode(group), group->lock, [&]() noexcept {
        for (auto &participant : group->parts) [[maybe_unused]]
            auto status = participant.reset();
        group->state = ending;
    });
}

/**
 *  @brief Stages every participant, and only from an open group.
 *  @param[out] status The outcome of the pass, left untouched when the group was not open.
 *  @return Where the group stood on entry; only @c open_k means the pass ran.
 *
 *  The test and the transition are one span, so of two threads racing here exactly one finds the
 *  group open and the other is told it is already staged rather than staging it twice.
 */
static group_state_t transaction_stage_if_open(transaction_object_t *group, status_t &status) noexcept {
    group_state_t entering = group_state_t::finished_k;
    run_over_values(group_mode(group), group->lock, [&]() noexcept {
        entering = group->state;
        if (entering != group_state_t::open_k) return;
        // Participants are already in canonical order, so two groups sharing containers acquire their
        // partition locks in the same sequence and cannot deadlock.
        for (auto &participant : group->parts) {
            status = participant.stage();
            if (failed(status)) break;
        }
        // A partial stage is never observable: unwind everything before returning.
        if (failed(status))
            for (auto &participant : group->parts) [[maybe_unused]]
                auto discarded = participant.reset();
        group->state = succeeded(status) ? group_state_t::staged_k : group_state_t::open_k;
    });
    return entering;
}

static char const doc_begin[] =                                                          //
    "begin()\n"                                                                          //
    "\n"                                                                                 //
    "Tuple of per-container views, in the order the containers were given.\n"            //
    "\n"                                                                                 //
    "Participants stage in a process-wide canonical order rather than argument order,\n" //
    "which is what lets two groups naming the same containers in opposite orders run\n"  //
    "concurrently without deadlocking. The views you receive ignore that and follow\n"   //
    "your arguments.\n";                                                                 //

static PyObject *Transaction_begin(PyObject *self, PyObject *) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    // Read without the lock, deliberately: taking it would drop the GIL between opening the group and
    // the first watch, and this test guards nothing - every operation the views offer re-tests the
    // state under the lock before it touches a participant.
    if (group->state == group_state_t::finished_k) {
        PyErr_SetString(state->state_error, "this transaction has already finished");
        return nullptr;
    }
    return Py_NewRef(group->views);
}

static char const doc_stage[] =                                                          //
    "stage()\n"                                                                          //
    "\n"                                                                                 //
    "Validate every watch and reserve the writes. Nothing becomes visible.\n"            //
    "\n"                                                                                 //
    "This is the first of the two phases. After it succeeds, commit() cannot fail for\n" //
    "a reason this transaction could have avoided, which is what makes the group\n"      //
    "all-or-nothing rather than merely usually-all.\n"                                   //
    "\n"                                                                                 //
    "A partial stage is never observable: if one participant refuses, every other is\n"  //
    "unwound before this returns.\n"                                                     //
    "\n"                                                                                 //
    "Raises:\n"                                                                          //
    "  ConflictError: If a watched key changed since the group opened.\n"                //
    "  StateError: If the group is not open - already staged, or finished.\n";           //

static PyObject *Transaction_stage(PyObject *self, PyObject *) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    status_t status = success_k;
    if (transaction_stage_if_open(group, status) != group_state_t::open_k) {
        PyErr_SetString(state->state_error, "stage() requires an open transaction");
        return nullptr;
    }
    if (failed(status)) return raise_for(state, status) == 0 ? Py_NewRef(Py_None) : nullptr;
    Py_RETURN_NONE;
}

static char const doc_commit[] =                                                      //
    "commit()\n"                                                                      //
    "\n"                                                                              //
    "Publish every staged change at once, across every container in the group.\n"     //
    "\n"                                                                              //
    "A reader either sees none of the group's writes or all of them; there is no\n"   //
    "moment at which half the containers have moved. Readers are not blocked while\n" //
    "this runs.\n"                                                                    //
    "\n"                                                                              //
    "Raises:\n"                                                                       //
    "  StateError: If the group was not staged first.\n";                             //

static PyObject *Transaction_commit(PyObject *self, PyObject *) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    status_t status = success_k;
    bool unstaged = false;
    run_over_values(group_mode(group), group->lock, [&]() noexcept {
        unstaged = group->state != group_state_t::staged_k;
        if (unstaged) return;
        for (auto &participant : group->parts) {
            status = participant.commit();
            if (failed(status)) break;
        }
        group->state = group_state_t::finished_k;
    });

    if (unstaged) {
        PyErr_SetString(state->state_error, "commit() requires a staged transaction");
        return nullptr;
    }
    if (raise_for(state, status) != 0) return nullptr;
    Py_RETURN_NONE;
}

static char const doc_rollback[] =                                                  //
    "rollback()\n"                                                                  //
    "\n"                                                                            //
    "Pull staged changes back out of the stores, leaving them as they were.\n"      //
    "\n"                                                                            //
    "Valid only after stage() and before commit(). Since nothing staged was ever\n" //
    "visible, no reader can have observed what this undoes.\n"                      //
    "\n"                                                                            //
    "Raises:\n"                                                                     //
    "  StateError: If the group was not staged.\n";                                 //

static PyObject *Transaction_rollback(PyObject *self, PyObject *) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    status_t status = success_k;
    bool unstaged = false;
    run_over_values(group_mode(group), group->lock, [&]() noexcept {
        unstaged = group->state != group_state_t::staged_k;
        if (unstaged) return;
        for (auto &participant : group->parts) {
            status = participant.rollback();
            if (failed(status)) break;
        }
        group->state = group_state_t::open_k;
    });

    if (unstaged) {
        PyErr_SetString(state->state_error, "rollback() requires a staged transaction");
        return nullptr;
    }
    if (raise_for(state, status) != 0) return nullptr;
    Py_RETURN_NONE;
}

static char const doc_reset[] =                                                       //
    "reset()\n"                                                                       //
    "\n"                                                                              //
    "Discard everything pending and start the group over.\n"                          //
    "\n"                                                                              //
    "Unlike rollback, valid at any point before the group finishes, staged or not.\n" //
    "This is what a retry loop calls after catching ConflictError.\n";                //

static PyObject *Transaction_reset(PyObject *self, PyObject *) noexcept {
    transaction_reset_all(object_as<transaction_object_t>(self), group_state_t::open_k);
    Py_RETURN_NONE;
}

static PyObject *Transaction_enter(PyObject *self, PyObject *) noexcept { return Transaction_begin(self, nullptr); }

static PyObject *Transaction_exit(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    bool const body_raised = count >= 1 && args[0] != Py_None;

    if (body_raised) {
        // The caller's exception is the story; discard the changes and let it propagate.
        transaction_reset_all(group, group_state_t::finished_k);
        Py_RETURN_FALSE;
    }

    // A block that staged by hand is already past this phase, so only an open group is staged here.
    status_t status = success_k;
    group_state_t const entering = transaction_stage_if_open(group, status);
    if (entering == group_state_t::finished_k) {
        PyErr_SetString(state->state_error, "this transaction has already finished");
        return nullptr;
    }
    if (failed(status)) {
        transaction_reset_all(group, group_state_t::finished_k);
        [[maybe_unused]] int const raised = raise_for(state, status);
        return nullptr;
    }
    PyObject *committed = Transaction_commit(self, nullptr);
    if (!committed) return nullptr;
    Py_DECREF(committed);
    Py_RETURN_FALSE;
}

static PyMethodDef Transaction_methods[] = {
    {"begin", Transaction_begin, METH_NOARGS, doc_begin},
    {"stage", Transaction_stage, METH_NOARGS, doc_stage},
    {"commit", Transaction_commit, METH_NOARGS, doc_commit},
    {"rollback", Transaction_rollback, METH_NOARGS, doc_rollback},
    {"reset", Transaction_reset, METH_NOARGS, doc_reset},
    {"__enter__", Transaction_enter, METH_NOARGS, nullptr},
    {"__exit__", as_pycfunction(Transaction_exit), METH_FASTCALL, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

static char const doc_Transaction[] =                                                 //
    "A group of containers updated all-or-nothing.\n"                                 //
    "\n"                                                                              //
    "Atomicity spans the whole group: every container moves together or none does,\n" //
    "including when the body raises. Isolation is read-committed - a group never\n"   //
    "reads another's uncommitted writes, but a value it read may change underneath\n" //
    "it, so a read-modify-write needs watch() to be safe.\n"                          //
    "\n"                                                                              //
    "Durability is memory-only: this is not a database and nothing survives the\n"    //
    "process.\n"                                                                      //
    "\n"                                                                              //
    "Used as a context manager, the block is exactly begin(), the body, stage(),\n"   //
    "commit(); an exception discards everything instead.\n";                          //

static PyType_Slot transaction_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(Transaction_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(Transaction_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(Transaction_clear)},
    {Py_tp_methods, reinterpret_cast<void *>(Transaction_methods)},
    {Py_tp_doc, const_cast<char *>(doc_Transaction)},
    {0, nullptr},
};

PyType_Spec transaction_spec = {
    "smashtable.Transaction", sizeof(transaction_object_t), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION,
    transaction_slots};

#pragma endregion Transaction

#pragma region Opening a Group

PyObject *make_transaction(module_state_t *state, PyObject *containers) noexcept {
    Py_ssize_t const count = PyTuple_GET_SIZE(containers);
    if (count == 0) {
        PyErr_SetString(PyExc_TypeError, "atomic() needs at least one container");
        return nullptr;
    }

    // Every participant must be a container, and no container may appear twice - two participants over
    // one store would each believe they owned its staged state.
    for (Py_ssize_t index = 0; index != count; ++index) {
        PyObject *candidate = PyTuple_GET_ITEM(containers, index);
        bool const is_map = Py_IS_TYPE(candidate, state->sorted_map_type);
        bool const is_set = Py_IS_TYPE(candidate, state->sorted_set_type);
        if (!is_map && !is_set) {
            PyErr_Format(PyExc_TypeError, "atomic() takes SortedMap or SortedSet, not %s", Py_TYPE(candidate)->tp_name);
            return nullptr;
        }
        for (Py_ssize_t earlier = 0; earlier != index; ++earlier)
            if (PyTuple_GET_ITEM(containers, earlier) == candidate) {
                PyErr_SetString(PyExc_ValueError, "atomic() cannot take the same container twice");
                return nullptr;
            }
    }

    // Canonical order for staging, argument order for the views handed back.
    auto made_order = basic_vector<Py_ssize_t>::make(static_cast<std::size_t>(count));
    if (!made_order) return PyErr_NoMemory();
    basic_vector<Py_ssize_t> order = std::move(*made_order);
    for (Py_ssize_t index = 0; index != count; ++index) [[maybe_unused]]
        auto appended = order.push_back(assume_reserved, Py_ssize_t {index});
    std::sort(order.begin(), order.end(), [&](Py_ssize_t left, Py_ssize_t right) noexcept {
        auto const *first = object_as<container_object_t>(PyTuple_GET_ITEM(containers, left));
        auto const *second = object_as<container_object_t>(PyTuple_GET_ITEM(containers, right));
        return first->ordinal < second->ordinal;
    });

    auto *group = PyObject_GC_New(transaction_object_t, state->transaction_type);
    if (!group) return nullptr;
    Py_INCREF(state->transaction_type);
    // Constructed before anything can fail, so an early `Py_DECREF` always meets a live vector.
    new (&group->parts) basic_vector<participant_t> {};
    new (&group->lock) object_lock_t {};
    group->containers = Py_NewRef(containers);
    group->views = nullptr;
    group->state = group_state_t::open_k;

    // The only allocation `parts` ever attempts, so every append below lands in reserved storage.
    if (failed(group->parts.reserve(static_cast<std::size_t>(count)))) {
        Py_DECREF(group);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t position = 0; position != count; ++position) {
        PyObject *container = PyTuple_GET_ITEM(containers, order[static_cast<std::size_t>(position)]);
        auto const *header = object_as<container_object_t>(container);
        bool opened = false;
        if (Py_IS_TYPE(container, state->sorted_map_type)) {
            if (auto transaction = object_as<sorted_map_object_t>(container)->store.transaction())
                group->parts.push_back(assume_reserved,
                                       participant_t {std::move(*transaction), header->ops, header->mode}),
                    opened = true;
        }
        else if (auto transaction = object_as<sorted_set_object_t>(container)->store.transaction())
            group->parts.push_back(assume_reserved, participant_t {std::move(*transaction), header->ops, header->mode}),
                opened = true;
        if (!opened) {
            Py_DECREF(group);
            return PyErr_NoMemory();
        }
    }

    // Views are handed back in the caller's order, whatever order the participants stage in.
    PyObject *views = PyTuple_New(count);
    if (!views) {
        Py_DECREF(group);
        return nullptr;
    }
    // `order[position]` is the argument index staged at `position`, so writing each view straight
    // into that slot inverts the permutation without searching for it.
    for (Py_ssize_t position = 0; position != count; ++position) {
        Py_ssize_t const index = order[static_cast<std::size_t>(position)];
        auto *view = PyObject_GC_New(view_object_t, state->view_type);
        if (!view) {
            Py_DECREF(views);
            Py_DECREF(group);
            return nullptr;
        }
        Py_INCREF(state->view_type);
        view->owner = Py_NewRef(reinterpret_cast<PyObject *>(group));
        view->index = static_cast<std::size_t>(position);
        PyObject_GC_Track(view);
        PyTuple_SET_ITEM(views, index, reinterpret_cast<PyObject *>(view));
    }
    group->views = views;
    PyObject_GC_Track(group);
    return reinterpret_cast<PyObject *>(group);
}

#pragma endregion Opening a Group

} // namespace ashvardanian::smashtable::py
