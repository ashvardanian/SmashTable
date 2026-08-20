/**
 *  @brief @c Transaction and @c View - the transaction that makes one update span several containers.
 *  @author Ash Vardanian
 *  @file python/transactions.cpp
 *  @date October 30, 2025
 *
 *  The unit of atomicity is the group, not the store: a @c with block opens one participant per
 *  container, and the block either publishes every change or abandons every one. Publication walks
 *  the participants in turn, so the group is all-or-nothing against failure rather than one instant.
 *
 *  @section transactions_alternatives Why the Participants Are Type-Erased
 *
 *  A transaction may mix maps and sets, whose store types differ. A participant holds its open
 *  transaction behind a @c void* beside the @c store_ops_t table that drives it, so every uniform
 *  operation is one indirect call with no vtable dispatch of its own and no arm to add when the store
 *  matrix grows.
 *
 *  @section transactions_ordering Why the Order Is Canonical
 *
 *  Participants are sorted by a process-wide container ordinal before anything is staged, so two groups
 *  sharing containers acquire their partition locks in the same sequence and cannot deadlock on each
 *  other. The views handed back to Python stay in the caller's argument order regardless.
 */
#include <algorithm> // `std::sort`
#include <limits>    // `std::numeric_limits`

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
 *  @brief This view's participant. Its layout and family are fixed for the transaction's life.
 *
 *  Whether the transaction is still open is not, so only the immutable parts - @c ops, @c mode and which
 *  alternative is engaged - may be read from it outside @c run_over_participant.
 */
/** @brief The participant this view speaks for, or null once the collector has cleared its owner. */
static participant_t *part_of(view_object_t *view) noexcept {
    if (!view->owner) return nullptr;
    auto *group = object_as<transaction_object_t>(view->owner);
    return &group->parts[view->index];
}

/**
 *  @brief The same, raising rather than answering null.
 *
 *  Every method here reaches its participant before the state gate in @c run_over_participant, so
 *  the cleared case has to be refused here or not at all.
 */
static participant_t *part_of_or_raise(view_object_t *view, module_state_t *state) noexcept {
    participant_t *part = part_of(view);
    if (!part) PyErr_SetString(state->state_error, "this view's transaction has been collected");
    return part;
}

/**
 *  @brief Runs one participant operation under its transaction's lock, refusing once the transaction has finished.
 *  @return 0 when @p operation ran; -1 with a @c StateError set when the transaction was already finished.
 *
 *  The state test and the operation are one span, so a concurrent commit either happens entirely
 *  before this or entirely after, never between the test and the write it guards.
 */
template <typename operation_type_>
static int run_over_participant(view_object_t *view, module_state_t *state, operation_type_ &&operation) noexcept {
    // Guarded here as well as at each caller's `part_of_or_raise`, because this is the one point
    // every participant operation passes through and the cost is a null test.
    if (!view->owner) {
        PyErr_SetString(state->state_error, "this view's transaction has been collected");
        return -1;
    }
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
    participant_t *part = part_of_or_raise(view, state);
    if (!part) return nullptr;
    if (!part->is_associative()) {
        PyErr_SetString(PyExc_TypeError, "this participant is a set, which has no values to read");
        return nullptr;
    }

    key_variant_t needle;
    if (!key_from_python(key, part->ops, needle)) return nullptr;

    expected<value_variant_t> found {key_not_found_k};
    if (run_over_participant(view, state, [&](participant_t &part) noexcept { found = part.find(needle); }) != 0)
        return nullptr;

    // A serializable read records the key it answered, and a record that will not fit is a refusal
    // rather than an absence. Reporting it as a `KeyError` would have `get()` answer with its default
    // for a transaction that is already doomed to be turned away at commit.
    if (found.status() == key_not_found_k) {
        PyErr_SetObject(PyExc_KeyError, key);
        return nullptr;
    }
    if (!found) {
        [[maybe_unused]] int const raised = raise_for(state, found.status(), key);
        return nullptr;
    }
    return value_to_python(*found);
}

/**
 *  @brief Stages a tombstone for every member of a window named by a slice, with either end optional.
 *
 *  Nothing is visible until the transaction commits, which is the whole difference from the store's
 *  own slice delete: a window erased here is a window this transaction read, so a key another
 *  transaction commits into it is a phantom the commit refuses over.
 *
 *  @return 0 on success, -1 with an exception set.
 */
static int view_delete_slice(PyObject *self, PyObject *slice) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;
    participant_t *part = part_of_or_raise(view, state);
    if (!part) return -1;

    auto const *bounds = reinterpret_cast<PySliceObject *>(slice);
    if (bounds->step != Py_None) {
        PyErr_SetString(PyExc_ValueError, "a step has no meaning over a range of keys");
        return -1;
    }

    key_variant_t lower;
    key_variant_t upper;
    bool const has_lower = bounds->start != Py_None;
    bool const has_upper = bounds->stop != Py_None;
    if (has_lower && !key_from_python(bounds->start, part->ops, lower)) return -1;
    if (has_upper && !key_from_python(bounds->stop, part->ops, upper)) return -1;

    status_t status = success_k;
    if (run_over_participant(view, state, [&](participant_t &part) noexcept {
            status = part.erase_range(has_lower ? &lower : nullptr, has_upper ? &upper : nullptr);
        }) != 0)
        return -1;
    return raise_for(state, status);
}

static int View_assign_subscript(PyObject *self, PyObject *key, PyObject *value) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;
    participant_t *part = part_of_or_raise(view, state);
    if (!part) return -1;

    // A slice names a window rather than a key, and only ever arrives at a delete, exactly as it
    // does on the store. An unordered participant has no window to name and says so here, since one
    // view type speaks for every core and the refusal cannot come from the type.
    if (PySlice_Check(key)) {
        if (!part->is_ordered()) {
            PyErr_SetString(PyExc_TypeError, "this participant has no ordering, so it cannot be sliced");
            return -1;
        }
        if (value) {
            PyErr_SetString(PyExc_TypeError, "a slice of this participant cannot be assigned to");
            return -1;
        }
        return view_delete_slice(self, key);
    }

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
    // This participant's own mode, never the transaction's first. A transaction mixing a scalar container with an
    // object one would otherwise read the wrong policy for every participant but the first, which is
    // the path where a missed GIL acquisition corrupts rather than fails.
    value_variant_t stored_value;
    if (!value_from_python(value, part->mode, part->releases, stored_value)) return -1;
    if (run_over_participant(view, state, [&](participant_t &part) noexcept {
            status = part.upsert(std::move(stored_key), &stored_value);
        }) != 0)
        return -1;
    return raise_for(state, status, key);
}

static int View_contains(PyObject *self, PyObject *key) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;
    participant_t *part = part_of_or_raise(view, state);
    if (!part) return -1;

    key_variant_t needle;
    if (!key_from_python(key, part->ops, needle)) {
        PyErr_Clear();
        return 0;
    }
    bool found = false;
    status_t asked = success_k;
    if (run_over_participant(view, state, [&](participant_t &part) noexcept {
            expected<bool> const held = part.contains(needle);
            asked = held.status();
            found = held && *held;
        }) != 0)
        return -1;
    // A read that could not record itself is a refusal, and the protocol's error answer is -1.
    if (failed(asked)) {
        [[maybe_unused]] int const raised = raise_for(state, asked);
        return -1;
    }
    return found ? 1 : 0;
}

static char const doc_View_get[] =                                                             //
    "get(key, default=None, /)\n"                                                              //
    "\n"                                                                                       //
    "Value for a key inside this transaction, or default when the key is absent.\n"            //
    "\n"                                                                                       //
    "Reads this transaction's own uncommitted writes, and otherwise the state as of\n"         //
    "when the transaction opened. It does NOT see writes another transaction has staged but\n" //
    "not committed, so a read is never dirty.\n"                                               //
    "\n"                                                                                       //
    "Repeating a read is not guaranteed to return the same value: a committed write\n"         //
    "from elsewhere becomes visible immediately. Call watch(key) to make the\n"                //
    "transaction refuse to commit if that happens.\n"                                          //
    "\n"                                                                                       //
    "Raises:\n"                                                                                //
    "  TypeError: If key is not of this store's key type, or this is a set.\n"                 //
    "  StateError: If the transaction has already finished.\n";                                //

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

static char const doc_View_add[] =                                                         //
    "add(member, /)\n"                                                                     //
    "\n"                                                                                   //
    "Insert one member into a set participant, invisibly until the transaction commits.\n" //
    "\n"                                                                                   //
    "Idempotent, and subject to the same visibility rule as upsert.\n"                     //
    "\n"                                                                                   //
    "Raises:\n"                                                                            //
    "  TypeError: If this participant is a map, which needs a value.\n"                    //
    "  StateError: If the transaction has already finished.\n";                            //

static PyObject *View_add(PyObject *self, PyObject *member) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    participant_t *part = part_of_or_raise(view, state);
    if (!part) return nullptr;
    if (part->is_associative()) {
        PyErr_SetString(PyExc_TypeError, "this participant is a map; assign a value rather than calling add()");
        return nullptr;
    }

    key_variant_t stored;
    if (!key_from_python(member, part->ops, stored)) return nullptr;
    status_t status = success_k;
    if (run_over_participant(
            view, state, [&](participant_t &part) noexcept { status = part.upsert(std::move(stored), nullptr); }) != 0)
        return nullptr;
    if (raise_for(state, status, member) != 0) return nullptr;
    Py_RETURN_NONE;
}

static char const doc_View_discard[] =                                                  //
    "discard(key, /)\n"                                                                 //
    "\n"                                                                                //
    "Remove a key if it is there, reporting whether it was.\n"                          //
    "\n"                                                                                //
    "`del participant[key]` removes it too; this is the form that answers.\n"           //
    "\n"                                                                                //
    "Returns:\n"                                                                        //
    "  bool: True when the key was present, so a caller need not look first.\n"         //
    "\n"                                                                                //
    "The removal is invisible outside the transaction until commit, and is undone by\n" //
    "rollback or by the block raising.\n"                                               //
    "\n"                                                                                //
    "Raises:\n"                                                                         //
    "  TypeError: If key is not of this store's key type.\n"                            //
    "  StateError: If the transaction has already finished.\n";                         //

static PyObject *View_discard(PyObject *self, PyObject *key) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    participant_t *part = part_of_or_raise(view, state);
    if (!part) return nullptr;

    key_variant_t stored;
    if (!key_from_python(key, part->ops, stored)) return nullptr;
    status_t status = success_k;
    // One store call, which reports absence as `key_not_found_k`. Asking first and erasing after
    // would answer for a key another writer can remove in between, and under a monotonic view - where
    // a participant reads live published state - would raise for a key it did not remove.
    // `erase` on a map destroys the stored value, so this is a value operation even though its
    // argument is only a key.
    if (run_over_participant(view, state, [&](participant_t &part) noexcept { status = part.erase(stored); }) != 0)
        return nullptr;
    if (status == key_not_found_k) Py_RETURN_FALSE;
    if (failed(status) && raise_for(state, status, key) != 0) return nullptr;
    Py_RETURN_TRUE;
}

static char const doc_View_watch[] =                                                        //
    "watch(key, /)\n"                                                                       //
    "\n"                                                                                    //
    "Refuse to commit if another writer touches this key first.\n"                          //
    "\n"                                                                                    //
    "This is what turns read-modify-write into a lost-update-free operation. Without\n"     //
    "it a transaction is atomic but not serializable: two transactions can each read the\n" //
    "same value, each write back, and one update is silently lost.\n"                       //
    "\n"                                                                                    //
    "Covers absence as well as presence, so watching a key that does not exist still\n"     //
    "conflicts if someone inserts it. The conflict surfaces at stage(), not at the\n"       //
    "write, and nothing is applied when it does.\n"                                         //
    "\n"                                                                                    //
    "Raises:\n"                                                                             //
    "  ConflictError: At stage() time, never here.\n"                                       //
    "  TypeError: If key is not of this store's key type.\n"                                //
    "  StateError: If the transaction has already finished.\n";                             //

static PyObject *View_watch(PyObject *self, PyObject *key) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    participant_t *part = part_of_or_raise(view, state);
    if (!part) return nullptr;

    key_variant_t stored;
    if (!key_from_python(key, part->ops, stored)) return nullptr;
    status_t status = success_k;
    if (run_over_participant(view, state, [&](participant_t &part) noexcept { status = part.watch(stored); }) != 0)
        return nullptr;
    if (raise_for(state, status, key) != 0) return nullptr;
    Py_RETURN_NONE;
}

static char const doc_View_update[] =                                                         //
    "update(other, /)\n"                                                                      //
    "\n"                                                                                      //
    "Apply every pair of a mapping to this participant.\n"                                    //
    "\n"                                                                                      //
    "Atomic, unlike SortedMap.update: every pair lands with the rest of the transaction or\n" //
    "none of them does. A failure part-way leaves the transaction abandonable by\n"           //
    "rollback with nothing applied.\n"                                                        //
    "\n"                                                                                      //
    "Raises:\n"                                                                               //
    "  TypeError: If other is not a mapping, or a key is of the wrong type.\n"                //
    "  ValueError: If an element of other is not a pair.\n"                                   //
    "  StateError: If the transaction has already finished.\n";                               //

static PyObject *View_update(PyObject *self, PyObject *other) noexcept {
    PyObject *pairs = PyMapping_Items(other);
    if (!pairs) return nullptr;
    PyObject *fast = PySequence_Fast(pairs, "update() needs a mapping");
    Py_DECREF(pairs);
    if (!fast) return nullptr;

    Py_ssize_t const total = PySequence_Fast_GET_SIZE(fast);
    for (Py_ssize_t index = 0; index != total; ++index) {
        PyObject *pair = PySequence_Fast_GET_ITEM(fast, index);
        // Read as a sequence rather than parsed as a tuple, so a list pair is accepted as `dict`
        // accepts it instead of answering `SystemError`.
        PyObject *unpacked = PySequence_Fast(pair, "update() needs a mapping or an iterable of pairs");
        if (!unpacked) {
            Py_DECREF(fast);
            return nullptr;
        }
        if (PySequence_Fast_GET_SIZE(unpacked) != 2) {
            PyErr_Format(PyExc_ValueError, "update() needs pairs, got a sequence of length %zd",
                         PySequence_Fast_GET_SIZE(unpacked));
            Py_DECREF(unpacked);
            Py_DECREF(fast);
            return nullptr;
        }
        int const assigned =
            View_assign_subscript(self, PySequence_Fast_GET_ITEM(unpacked, 0), PySequence_Fast_GET_ITEM(unpacked, 1));
        Py_DECREF(unpacked);
        if (assigned != 0) {
            Py_DECREF(fast);
            return nullptr;
        }
    }
    Py_DECREF(fast);
    Py_RETURN_NONE;
}

static char const doc_View_scan[] =                                                                //
    "scan(start=None, stop=None, *, limit=None)\n"                                                 //
    "\n"                                                                                           //
    "List of (key, value) pairs in key order over [start, stop), or of members for a set.\n"       //
    "\n"                                                                                           //
    "Sees this transaction's own staged writes, and otherwise the state as of when it\n"           //
    "opened. Materialized in one span rather than lazily, because a walk records the\n"            //
    "window it read and nothing else may run inside that record.\n"                                //
    "\n"                                                                                           //
    "Under 'serializable' or stricter, a key another transaction commits into that\n"              //
    "window makes this transaction's commit raise PhantomConflictError, rather than\n"             //
    "let a repeat of the walk answer differently.\n"                                               //
    "\n"                                                                                           //
    "Raises:\n"                                                                                    //
    "  TypeError: If a bound is not of this store's key type, or this participant is unordered.\n" //
    "  ValueError: If limit is negative.\n"                                                        //
    "  StateError: If the transaction has already finished.\n";                                    //

static PyObject *View_scan(PyObject *self, PyObject *const *args, Py_ssize_t count, PyObject *keywords) noexcept {
    auto *view = object_as<view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    participant_t *part = part_of_or_raise(view, state);
    if (!part) return nullptr;
    if (!part->is_ordered()) {
        PyErr_SetString(PyExc_TypeError, "this participant has no ordering, so it cannot be scanned");
        return nullptr;
    }

    PyObject *start_object = nullptr;
    PyObject *stop_object = nullptr;
    Py_ssize_t limit = -1;
    if (!window_from_python("scan", args, count, keywords, start_object, stop_object, limit)) return nullptr;

    key_variant_t lower;
    key_variant_t upper;
    bool const has_stop = stop_object && stop_object != Py_None;
    bool has_start = start_object && start_object != Py_None;
    if (has_start && !key_from_python(start_object, part->ops, lower)) return nullptr;
    if (has_stop && !key_from_python(stop_object, part->ops, upper)) return nullptr;
    basic_vector<entry_t> collected;
    status_t status = success_k;
    std::size_t const wanted = limit < 0 ? std::numeric_limits<std::size_t>::max() : static_cast<std::size_t>(limit);
    if (run_over_participant(view, state, [&](participant_t &part) noexcept {
            status = part.scan(has_start ? &lower : nullptr, has_stop ? &upper : nullptr, wanted, collected);
        }) != 0)
        return nullptr;
    if (raise_for(state, status) != 0) return nullptr;

    bool const associative = part->is_associative();
    PyObject *listed = PyList_New(static_cast<Py_ssize_t>(collected.size()));
    if (!listed) return nullptr;
    for (std::size_t index = 0; index != collected.size(); ++index) {
        PyObject *element = associative ? pair_to_python(collected[index].key, collected[index].mapped)
                                        : key_to_python(collected[index].key);
        if (!element) {
            Py_DECREF(listed);
            return nullptr;
        }
        PyList_SET_ITEM(listed, static_cast<Py_ssize_t>(index), element);
    }
    return listed;
}

static PyMethodDef View_methods[] = {
    {"get", as_pycfunction(View_get), METH_FASTCALL, doc_View_get},
    {"add", View_add, METH_O, doc_View_add},
    {"discard", View_discard, METH_O, doc_View_discard},
    {"watch", View_watch, METH_O, doc_View_watch},
    {"update", View_update, METH_O, doc_View_update},
    {"scan", as_pycfunction(View_scan), ST_METHOD_FLAGS_, doc_View_scan},
    {nullptr, nullptr, 0, nullptr},
};

static char const doc_View[] =                                                               //
    "One container's slice of a transaction.\n"                                              //
    "\n"                                                                                     //
    "Reads see this transaction's own writes and the state as of when the group\n"           //
    "opened, never another transaction's staged-but-uncommitted writes. Writes stay\n"       //
    "invisible to every other reader until the transaction commits, and vanish if it does\n" //
    "not. A view stops working once its transaction finishes.\n";                            //

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
    "smashtable.Participant", sizeof(view_object_t), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_IMMUTABLETYPE | Py_TPFLAGS_DISALLOW_INSTANTIATION, view_slots};

#pragma endregion View

#pragma region Transaction

static void Transaction_dealloc(PyObject *self) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    PyObject_GC_UnTrack(self);
    group->parts.~basic_vector();
    group->lock.~spin_shared_mutex_t();
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
 *  @param[in] ending Where the transaction stands afterwards - open again, or finished for good.
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
 *  @param[out] status The outcome of the pass, left untouched when the transaction was not open.
 *  @return Where the transaction stood on entry; only @c open_k means the pass ran.
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
        std::size_t staged = 0;
        for (auto &participant : group->parts) {
            status = participant.stage();
            if (failed(status)) break;
            ++staged;
        }
        // A partial stage is never observable, so the prefix that took is unwound - and only that
        // prefix, rolled back rather than reset. Rolling back returns the staged writes to the
        // transaction, which is what leaves a refused stage retryable; resetting would discard the
        // caller's pending writes along with them, so a retry after ConflictError would have nothing
        // left to commit. This is the shape `transaction_group::stage` uses.
        if (failed(status))
            while (staged != 0) {
                --staged;
                [[maybe_unused]] status_t const unwound = group->parts[staged].rollback();
            }
        group->state = succeeded(status) ? group_state_t::staged_k : group_state_t::open_k;
    });
    return entering;
}

static char const doc_begin[] =                                                          //
    "begin()\n"                                                                          //
    "\n"                                                                                 //
    "Tuple of per-container views, in the order the stores were given.\n"                //
    "\n"                                                                                 //
    "Participants stage in a process-wide canonical order rather than argument order,\n" //
    "which is what lets two groups naming the same containers in opposite orders run\n"  //
    "concurrently without deadlocking. The views you receive ignore that and follow\n"   //
    "your arguments.\n"                                                                  //
    "\n"                                                                                 //
    "Raises:\n"                                                                          //
    "  StateError: If the transaction has already finished.\n";                          //

static PyObject *Transaction_begin(PyObject *self, PyObject *) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (!group->views) {
        PyErr_SetString(state->state_error, "this transaction has been collected");
        return nullptr;
    }
    // Read without the lock, deliberately: taking it would drop the GIL between opening the transaction and
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
    "a reason this transaction could have avoided. It can still refuse over a watched\n" //
    "key another transaction published while this one sat staged.\n"                     //
    "\n"                                                                                 //
    "A partial stage is never observable: if one participant refuses, every other is\n"  //
    "unwound before this returns.\n"                                                     //
    "\n"                                                                                 //
    "Raises:\n"                                                                          //
    "  ConflictError: If a watched key changed since the transaction opened.\n"          //
    "  StateError: If the transaction is not open - already staged, or finished.\n";     //

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

static char const doc_commit[] =                                                          //
    "commit()\n"                                                                          //
    "\n"                                                                                  //
    "Publish every staged change, one container at a time.\n"                             //
    "\n"                                                                                  //
    "Not a snapshot across containers. Each is applied in turn, so a thread reading\n"    //
    "two of them while this runs may find one a step ahead of the other; a reader that\n" //
    "needs the pair to agree should take its own transaction. Readers are never\n"        //
    "blocked while this runs.\n"                                                          //
    "\n"                                                                                  //
    "A refusal part-way leaves the containers already published and the rest staged,\n"   //
    "so the group stays staged rather than finished and can still be unwound.\n"          //
    "\n"                                                                                  //
    "Raises:\n"                                                                           //
    "  StateError: If the transaction was not staged first.\n"                            //
    "  ConflictError: If a watched key was published over while the group sat staged.\n"; //

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
        // A refusal part-way leaves the participants before it published and those after it still
        // staged. Calling that finished would seal a torn group: `rollback` and `reset` both refuse
        // a finished one, so the staged remainder could only be dropped by the destructor. It stays
        // staged instead, which is the one state from which the caller can still unwind it.
        if (succeeded(status)) group->state = group_state_t::finished_k;
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
    "  StateError: If the transaction was not staged.\n";                           //

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

static char const doc_reset[] =                                                             //
    "reset()\n"                                                                             //
    "\n"                                                                                    //
    "Discard everything pending and start the transaction over.\n"                          //
    "\n"                                                                                    //
    "Unlike rollback, valid at any point before the transaction finishes, staged or not.\n" //
    "This is what a retry loop calls after catching ConflictError.\n";                      //

static PyObject *Transaction_reset(PyObject *self, PyObject *) noexcept {
    auto *group = object_as<transaction_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    // A finished group has published or discarded everything it held, and its participants are
    // back to pending. Resetting it would hand every view a second turn at a transaction that is
    // already over, and the writes would land in the store as if they were a fresh one.
    bool finished = false;
    run_over_values(group_mode(group), group->lock, [&]() noexcept {
        finished = group->state == group_state_t::finished_k;
        if (finished) return;
        for (auto &participant : group->parts) [[maybe_unused]]
            auto status = participant.reset();
        group->state = group_state_t::open_k;
    });

    if (finished) {
        PyErr_SetString(state->state_error, "reset() requires a transaction that has not finished");
        return nullptr;
    }
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

static char const doc_Transaction[] =                                                    //
    "A transaction of containers staged together and then published in turn.\n"          //
    "\n"                                                                                 //
    "Either every store is published or none is: a body that raises resets each\n"       //
    "participant, and a stage that one refuses unwinds the rest. Publication itself\n"   //
    "walks the containers in turn rather than moving them at one instant, so a thread\n" //
    "reading two of them while a commit runs may find one a step ahead - take a\n"       //
    "transaction of your own if the pair must agree. No transaction ever reads\n"        //
    "another's uncommitted writes. What more is promised depends on the store: its\n"    //
    "isolation property reports the level, and below 'snapshot' a value already\n"       //
    "read may change underneath, so a read-modify-write needs watch() to be safe.\n"     //
    "\n"                                                                                 //
    "Durability is memory-only: this is not a database and nothing survives the\n"       //
    "process.\n"                                                                         //
    "\n"                                                                                 //
    "Used as a context manager, the block is exactly begin(), the body, stage(),\n"      //
    "commit(); an exception discards everything instead.\n";                             //

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

    // Every participant must be a store, and no container may appear twice - two participants over
    // one store would each believe they owned its staged state.
    for (Py_ssize_t index = 0; index != count; ++index) {
        PyObject *candidate = PyTuple_GET_ITEM(containers, index);
        if (!is_container(state, candidate)) {
            PyErr_Format(PyExc_TypeError, "atomic() takes a SmashTable container, not %s", Py_TYPE(candidate)->tp_name);
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
    if (!made_order) {
        [[maybe_unused]] int const raised = raise_for(state, made_order.status());
        return nullptr;
    }
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
    // No incref of the type here: `PyObject_GC_New` already took one, and the matching decref in
    // dealloc gives back exactly one. Taking a second immortalises the type in practice.
    // Constructed before anything can fail, so an early `Py_DECREF` always meets a live vector.
    new (&group->parts) basic_vector<participant_t> {};
    new (&group->lock) spin_shared_mutex_t {};
    group->containers = Py_NewRef(containers);
    group->views = nullptr;
    group->state = group_state_t::open_k;

    // The only allocation `parts` ever attempts, so every append below lands in reserved storage.
    if (status_t const reserved = group->parts.reserve(static_cast<std::size_t>(count)); failed(reserved)) {
        Py_DECREF(group);
        [[maybe_unused]] int const raised = raise_for(state, reserved);
        return nullptr;
    }

    for (Py_ssize_t position = 0; position != count; ++position) {
        PyObject *container = PyTuple_GET_ITEM(containers, order[static_cast<std::size_t>(position)]);
        auto const *header = object_as<container_object_t>(container);
        expected<void *> transaction = header->store_ops->transaction_make(header->store);
        if (!transaction) {
            Py_DECREF(group);
            [[maybe_unused]] int const raised = raise_for(state, transaction.status());
            return nullptr;
        }
        [[maybe_unused]] status_t const appended = group->parts.push_back(
            assume_reserved,
            participant_t {header->store_ops, *transaction, header->ops, &header->releases, header->mode});
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
