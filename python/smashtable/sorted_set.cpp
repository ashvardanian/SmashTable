/**
 *  @brief @c smashtable.SortedSet - an ordered, transactional set with @c set's surface.
 *  @author Ash Vardanian
 *  @file python/smashtable/sorted_set.cpp
 *  @date August 16, 2026
 *
 *  The same store as @c SortedMap with one template argument changed: a set instantiates the tree on the
 *  key directly, a map on a @c mapping of key and value. Everything below the container - the key layouts,
 *  the cursor, the transaction protocol - is shared, so this file is only the surface.
 *
 *  Set algebra probes rather than merges. Walking one side and testing membership in the other is
 *  O(n log m), against O(n + m) for a merge of two ordered walks; the probe is used because it serves a
 *  plain @c set on the right-hand side with the same code, and the merge is a later optimization once
 *  there is a benchmark asking for it.
 */
#include "shared.hpp"

namespace ashvardanian::smashtable::py {

#pragma region Construction

static char const doc_SortedSet[] =                                                     //
    "SortedSet(*, key)\n"                                                               //
    "\n"                                                                                //
    "An ordered set whose writes can be grouped into transactions.\n"                   //
    "\n"                                                                                //
    "Members are homogeneous and their type is fixed at construction. Floats and\n"     //
    "booleans are not valid members, for the same reason they are not valid keys.\n"    //
    "\n"                                                                                //
    "Args:\n"                                                                           //
    "  key (type or str): One of int, str, bytes, or 'int', 'uint', 'str', 'bytes'.\n"; //

/**
 *  @brief Builds an empty set of a given layout, without re-entering the type through Python.
 *  @param[in] state The module state, which hands out the staging ordinal.
 *  @param[in] type The heap type to allocate, borrowed.
 *  @param[in] ops The key layout the store is built around.
 *  @return A new reference, or @c nullptr with an exception set.
 */
static PyObject *sorted_set_of(module_state_t *state, PyTypeObject *type, key_ops_t const *ops) noexcept {
    auto *self = object_as<sorted_set_object_t>(type->tp_alloc(type, 0));
    if (!self) return nullptr;

    auto made = set_store_t::make(key_less_t {ops->less}, key_hash_t {ops->hash});
    if (!made) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    // `tp_alloc` hands back zeroed storage rather than a constructed object, so the store is built
    // in place. Nothing owns it but this object, and `tp_dealloc` destroys it.
    new (&self->store) set_store_t(std::move(*made));
    self->base.ops = ops;
    self->base.ordinal = state->next_ordinal.fetch_add(1, std::memory_order_relaxed);
    return reinterpret_cast<PyObject *>(self);
}

static PyObject *SortedSet_new(PyTypeObject *type, PyObject *args, PyObject *keywords) noexcept {
    if (args && PyTuple_GET_SIZE(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "SortedSet() takes no positional arguments");
        return nullptr;
    }
    PyObject *key_specification = nullptr;
    if (keywords) {
        Py_ssize_t position = 0;
        PyObject *name = nullptr;
        PyObject *value = nullptr;
        while (PyDict_Next(keywords, &position, &name, &value)) {
            if (PyUnicode_CompareWithASCIIString(name, "key") == 0) key_specification = value;
            else {
                PyErr_Format(PyExc_TypeError, "SortedSet() got an unexpected keyword argument '%U'", name);
                return nullptr;
            }
        }
    }

    key_ops_t const *ops = key_ops_from_python(key_specification);
    if (!ops) return nullptr;

    module_state_t *state = state_of_heap_type(type);
    if (!state) return nullptr;
    return sorted_set_of(state, type, ops);
}

static void SortedSet_dealloc(PyObject *self) noexcept {
    auto *set = object_as<sorted_set_object_t>(self);
    set->store.~set_store_t();
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
}

static int SortedSet_traverse(PyObject *self, visitproc visit, void *arg) noexcept {
    Py_VISIT(Py_TYPE(self));
    return 0;
}

#pragma endregion Construction

#pragma region Reading

static Py_ssize_t SortedSet_length(PyObject *self) noexcept {
    auto *set = object_as<sorted_set_object_t>(self);
    std::size_t size = 0;
    Py_BEGIN_ALLOW_THREADS;
    size = set->store.size();
    Py_END_ALLOW_THREADS;
    return static_cast<Py_ssize_t>(size);
}

static int SortedSet_contains(PyObject *self, PyObject *member) noexcept {
    auto *set = object_as<sorted_set_object_t>(self);
    key_variant_t needle;
    if (!key_from_python(member, set->base.ops, needle)) {
        PyErr_Clear(); // A member of the wrong type cannot be present, and `set` answers `False`
        return 0;
    }
    bool found = false;
    Py_BEGIN_ALLOW_THREADS;
    found = set->store.contains(needle);
    Py_END_ALLOW_THREADS;
    return found ? 1 : 0;
}

static PyObject *SortedSet_iter(PyObject *self) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    return cursor_new(state, self, cursor_yields_t::keys_k, nullptr, nullptr, -1);
}

static char const doc_scan[] =                                                      //
    "scan(start=None, stop=None, *, limit=None)\n"                                  //
    "\n"                                                                            //
    "List of members in key order, over the half-open window [start, stop).\n"      //
    "\n"                                                                            //
    "Bounds need not be present keys. Materialized rather than lazy: iterate the\n" //
    "container itself when the whole range does not need to exist at once.\n"       //
    "\n"                                                                            //
    "Raises:\n"                                                                     //
    "  TypeError: If a bound is not of this container's key type.\n";               //

/** @brief Drives the shared cursor into a list, so there is one traversal in the binding. */
static PyObject *SortedSet_scan(PyObject *self, PyObject *const *args, Py_ssize_t count, PyObject *keywords) noexcept {
    auto const *header = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (count > 2) {
        PyErr_SetString(PyExc_TypeError, "scan() takes at most two positional arguments");
        return nullptr;
    }

    PyObject *start_object = count > 0 ? args[0] : nullptr;
    PyObject *stop_object = count > 1 ? args[1] : nullptr;
    Py_ssize_t limit = -1;
    if (keywords) {
        Py_ssize_t const named = PyTuple_GET_SIZE(keywords);
        for (Py_ssize_t index = 0; index != named; ++index) {
            PyObject *name = PyTuple_GET_ITEM(keywords, index);
            PyObject *value = args[count + index];
            if (PyUnicode_CompareWithASCIIString(name, "start") == 0) {
                if (start_object) {
                    PyErr_SetString(PyExc_TypeError, "scan() got multiple values for 'start'");
                    return nullptr;
                }
                start_object = value;
            }
            else if (PyUnicode_CompareWithASCIIString(name, "stop") == 0) {
                if (stop_object) {
                    PyErr_SetString(PyExc_TypeError, "scan() got multiple values for 'stop'");
                    return nullptr;
                }
                stop_object = value;
            }
            else if (PyUnicode_CompareWithASCIIString(name, "limit") == 0) {
                if (value != Py_None) {
                    limit = PyNumber_AsSsize_t(value, PyExc_OverflowError);
                    if (limit == -1 && PyErr_Occurred()) return nullptr;
                }
            }
            else {
                PyErr_Format(PyExc_TypeError, "scan() got an unexpected keyword argument '%U'", name);
                return nullptr;
            }
        }
    }

    key_variant_t start_key;
    key_variant_t stop_key;
    bool has_start = start_object && start_object != Py_None;
    bool has_stop = stop_object && stop_object != Py_None;
    if (has_start && !key_from_python(start_object, header->ops, start_key)) return nullptr;
    if (has_stop && !key_from_python(stop_object, header->ops, stop_key)) return nullptr;

    PyObject *cursor = cursor_new(state, self, cursor_yields_t::keys_k, has_start ? &start_key : nullptr,
                                  has_stop ? &stop_key : nullptr, limit);
    if (!cursor) return nullptr;
    PyObject *collected = PySequence_List(cursor);
    Py_DECREF(cursor);
    return collected;
}

#pragma endregion Reading

#pragma region Writing

static char const doc_add[] = //
    "add(member, /)\n"        //
    "\n"                      //
    "Insert a member. Adding one that is already present does nothing.\n";

static PyObject *SortedSet_add(PyObject *self, PyObject *member) noexcept {
    auto *set = object_as<sorted_set_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    key_variant_t stored;
    if (!key_from_python(member, set->base.ops, stored)) return nullptr;

    status_t status = success_k;
    Py_BEGIN_ALLOW_THREADS;
    status = set->store.upsert(std::move(stored));
    Py_END_ALLOW_THREADS;
    if (raise_for(state, status, member) != 0) return nullptr;
    Py_RETURN_NONE;
}

/** @brief The shared body of @c discard and @c remove, which differ only in the absent case. */
static int set_erase(PyObject *self, PyObject *member, bool *was_present) noexcept {
    auto *set = object_as<sorted_set_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;

    key_variant_t stored;
    if (!key_from_python(member, set->base.ops, stored)) return -1;

    status_t status = success_k;
    bool present = false;
    Py_BEGIN_ALLOW_THREADS;
    present = set->store.contains(stored);
    if (present) status = set->store.erase(stored);
    Py_END_ALLOW_THREADS;

    *was_present = present;
    if (!present) return 0;
    return raise_for(state, status, member);
}

static char const doc_discard[] = //
    "discard(member, /)\n"        //
    "\n"                          //
    "Remove a member if present, and say nothing if not.\n";

static PyObject *SortedSet_discard(PyObject *self, PyObject *member) noexcept {
    bool present = false;
    if (set_erase(self, member, &present) != 0) {
        // A wrong-typed member simply is not there, which is what `set.discard` promises.
        if (PyErr_ExceptionMatches(PyExc_TypeError)) {
            PyErr_Clear();
            Py_RETURN_NONE;
        }
        return nullptr;
    }
    Py_RETURN_NONE;
}

static char const doc_remove[] =              //
    "remove(member, /)\n"                     //
    "\n"                                      //
    "Remove a member.\n"                      //
    "\n"                                      //
    "Raises:\n"                               //
    "  KeyError: If the member is absent.\n"; //

static PyObject *SortedSet_remove(PyObject *self, PyObject *member) noexcept {
    bool present = false;
    if (set_erase(self, member, &present) != 0) return nullptr;
    if (!present) {
        PyErr_SetObject(PyExc_KeyError, member);
        return nullptr;
    }
    Py_RETURN_NONE;
}

static char const doc_pop[] =                                     //
    "pop()\n"                                                     //
    "\n"                                                          //
    "Remove and return the smallest member.\n"                    //
    "\n"                                                          //
    "Unlike set, which pops an arbitrary member, this pops the\n" //
    "smallest, which the ordered store reaches in one lookup.\n"  //
    "\n"                                                          //
    "Raises:\n"                                                   //
    "  KeyError: If the set is empty.\n";                         //

static PyObject *SortedSet_pop(PyObject *self, PyObject *) noexcept {
    auto *set = object_as<sorted_set_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    key_variant_t smallest;
    bool present = false;
    status_t status = success_k;
    Py_BEGIN_ALLOW_THREADS;
    key_variant_t floor;
    set->base.ops->least(floor);
    set->store.lower_bound(
        floor, [&](key_variant_t const &member) noexcept { smallest = member, present = true; }, []() noexcept {});
    if (present) status = set->store.erase(smallest);
    Py_END_ALLOW_THREADS;

    if (!present) {
        PyErr_SetString(PyExc_KeyError, "pop(): set is empty");
        return nullptr;
    }
    if (raise_for(state, status) != 0) return nullptr;
    return key_to_python(smallest);
}

static char const doc_clear[] = //
    "clear()\n"                 //
    "\n"                        //
    "Remove every member. The key type stays as it was.\n";

static PyObject *SortedSet_clear(PyObject *self, PyObject *) noexcept {
    auto *set = object_as<sorted_set_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    status_t status = success_k;
    Py_BEGIN_ALLOW_THREADS;
    status = set->store.clear();
    Py_END_ALLOW_THREADS;
    if (raise_for(state, status) != 0) return nullptr;
    Py_RETURN_NONE;
}

static char const doc_update[] = //
    "update(other, /)\n"         //
    "\n"                         //
    "Add every member of an iterable. Not atomic, matching set.\n";

static PyObject *SortedSet_update(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "update() takes exactly one argument");
        return nullptr;
    }
    PyObject *iterator = PyObject_GetIter(args[0]);
    if (!iterator) return nullptr;
    PyObject *member = nullptr;
    while ((member = PyIter_Next(iterator)) != nullptr) {
        PyObject *outcome = SortedSet_add(self, member);
        Py_DECREF(member);
        if (!outcome) {
            Py_DECREF(iterator);
            return nullptr;
        }
        Py_DECREF(outcome);
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return nullptr;
    Py_RETURN_NONE;
}

#pragma endregion Writing

#pragma region Algebra

/** @brief Which members of the walked side end up in the result. */
enum class algebra_t : std::uint8_t { union_k, intersection_k, difference_k, symmetric_difference_k };

/** @brief Builds a fresh set of the same key layout, ready to receive results. */
static PyObject *set_like(PyObject *self) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    return sorted_set_of(state, state->sorted_set_type, object_as<container_object_t>(self)->ops);
}

/** @brief The other side as a set of this build sharing this layout, or null when it is neither. */
static sorted_set_object_t *same_layout_set(PyObject *self, PyObject *other, module_state_t *state) noexcept {
    if (!Py_IS_TYPE(other, state->sorted_set_type)) return nullptr;
    auto *theirs = object_as<sorted_set_object_t>(other);
    return theirs->base.ops == object_as<container_object_t>(self)->ops ? theirs : nullptr;
}

/** @brief Whether the side being walked holds a member the other side also has. */
enum class membership_t : std::uint8_t { absent_k, shared_k };

/** @brief Which of the two sets a member is being drawn from. */
enum class side_t : std::uint8_t { mine_k, theirs_k };

/**
 *  @brief Whether a member belongs in the result of @p operation.
 *  @param[in] operation Which algebra is being computed.
 *  @param[in] membership Whether the other side holds this member too.
 *  @param[in] side Which set the member came from.
 */
static bool keeps_member(algebra_t operation, membership_t membership, side_t side) noexcept {
    bool const shared = membership == membership_t::shared_k;
    bool const from_mine = side == side_t::mine_k;
    switch (operation) {
    case algebra_t::union_k: return true;
    case algebra_t::intersection_k: return from_mine && shared;
    case algebra_t::difference_k: return from_mine && !shared;
    case algebra_t::symmetric_difference_k: return !shared;
    }
    return false;
}

/**
 *  @brief Computes the algebra entirely in C++, for two sets sharing a key layout.
 *
 *  No member becomes a Python object at any point: each is compared and copied as a stored scalar.
 *  The general path below has to build one per member, and then convert it back on the way into the
 *  result, which is three conversions for a value that never needed to leave the store.
 *
 *  @return True when the result was filled; false with an exception set.
 */
static bool set_algebra_natively(sorted_set_object_t *mine, sorted_set_object_t *theirs, sorted_set_object_t *result,
                                 algebra_t operation) noexcept {
    bool failed = false;
    auto const *ops = mine->base.ops;

    auto absorb = [&](key_variant_t const &member, membership_t membership, side_t side) noexcept {
        if (failed || !keeps_member(operation, membership, side)) return;
        auto copied = member.copy();
        if (!copied || !succeeded(result->store.upsert(std::move(*copied)))) failed = true;
    };

    for_each_in_order(mine->store, ops, [&](key_variant_t const &member) noexcept {
        absorb(member, theirs->store.contains(member) ? membership_t::shared_k : membership_t::absent_k,
               side_t::mine_k);
    });

    // Union and symmetric difference also need what only the other side holds.
    if (operation == algebra_t::union_k || operation == algebra_t::symmetric_difference_k)
        for_each_in_order(theirs->store, ops, [&](key_variant_t const &member) noexcept {
            absorb(member, mine->store.contains(member) ? membership_t::shared_k : membership_t::absent_k,
                   side_t::theirs_k);
        });

    if (failed) PyErr_NoMemory();
    return !failed;
}

static PyObject *set_algebra(PyObject *self, PyObject *other, algebra_t operation) noexcept {
    if (!PyObject_HasAttrString(other, "__contains__")) Py_RETURN_NOTIMPLEMENTED;

    PyObject *result = set_like(self);
    if (!result) return nullptr;

    module_state_t *state = state_of_type(self);
    if (!state) {
        Py_DECREF(result);
        return nullptr;
    }
    if (auto *twin = same_layout_set(self, other, state)) {
        auto *mine = object_as<sorted_set_object_t>(self);
        auto *filled = object_as<sorted_set_object_t>(result);
        if (!set_algebra_natively(mine, twin, filled, operation)) {
            Py_DECREF(result);
            return nullptr;
        }
        return result;
    }

    // Walk this side, deciding each member by whether the other side holds it.
    PyObject *iterator = PyObject_GetIter(self);
    if (!iterator) {
        Py_DECREF(result);
        return nullptr;
    }
    PyObject *member = nullptr;
    while ((member = PyIter_Next(iterator)) != nullptr) {
        int const shared = PySequence_Contains(other, member);
        bool keep = false;
        if (shared < 0) PyErr_Clear(); // A member the other side cannot even hold is simply not shared
        switch (operation) {
        case algebra_t::union_k: keep = true; break;
        case algebra_t::intersection_k: keep = shared == 1; break;
        case algebra_t::difference_k:
        case algebra_t::symmetric_difference_k: keep = shared != 1; break;
        }
        if (keep) {
            PyObject *added = SortedSet_add(result, member);
            if (!added) {
                Py_DECREF(member);
                Py_DECREF(iterator);
                Py_DECREF(result);
                return nullptr;
            }
            Py_DECREF(added);
        }
        Py_DECREF(member);
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) {
        Py_DECREF(result);
        return nullptr;
    }

    // Union and symmetric difference also need whatever the other side contributes.
    if (operation == algebra_t::union_k || operation == algebra_t::symmetric_difference_k) {
        PyObject *other_iterator = PyObject_GetIter(other);
        if (!other_iterator) {
            Py_DECREF(result);
            return nullptr;
        }
        while ((member = PyIter_Next(other_iterator)) != nullptr) {
            int const mine = PySequence_Contains(self, member);
            if (mine < 0) PyErr_Clear();
            bool const keep = operation == algebra_t::union_k || mine != 1;
            if (keep) {
                PyObject *added = SortedSet_add(result, member);
                if (!added) {
                    Py_DECREF(member);
                    Py_DECREF(other_iterator);
                    Py_DECREF(result);
                    return nullptr;
                }
                Py_DECREF(added);
            }
            Py_DECREF(member);
        }
        Py_DECREF(other_iterator);
        if (PyErr_Occurred()) {
            Py_DECREF(result);
            return nullptr;
        }
    }
    return result;
}

static PyObject *SortedSet_union(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "union() takes exactly one argument");
        return nullptr;
    }
    return set_algebra(self, args[0], algebra_t::union_k);
}

static PyObject *SortedSet_intersection(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "intersection() takes exactly one argument");
        return nullptr;
    }
    return set_algebra(self, args[0], algebra_t::intersection_k);
}

static PyObject *SortedSet_difference(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "difference() takes exactly one argument");
        return nullptr;
    }
    return set_algebra(self, args[0], algebra_t::difference_k);
}

static PyObject *SortedSet_symmetric_difference(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "symmetric_difference() takes exactly one argument");
        return nullptr;
    }
    return set_algebra(self, args[0], algebra_t::symmetric_difference_k);
}

static PyObject *SortedSet_isdisjoint(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "isdisjoint() takes exactly one argument");
        return nullptr;
    }

    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (auto *twin = same_layout_set(self, args[0], state)) {
        auto *mine = object_as<sorted_set_object_t>(self);
        bool disjoint = true;
        // Probing the smaller side keeps this O(min(n, m) log max(n, m)) rather than always O(n log m).
        auto const *smaller = mine->store.size() <= twin->store.size() ? mine : twin;
        auto const *larger = smaller == mine ? twin : mine;
        for_each_in_order(const_cast<set_store_t &>(smaller->store), mine->base.ops,
                          [&](key_variant_t const &member) noexcept {
                              disjoint = !larger->store.contains(member);
                              return disjoint;
                          });
        return PyBool_FromLong(disjoint ? 1 : 0);
    }

    PyObject *iterator = PyObject_GetIter(args[0]);
    if (!iterator) return nullptr;
    PyObject *member = nullptr;
    bool disjoint = true;
    while (disjoint && (member = PyIter_Next(iterator)) != nullptr) {
        int const mine = SortedSet_contains(self, member);
        Py_DECREF(member);
        if (mine < 0) {
            Py_DECREF(iterator);
            return nullptr;
        }
        disjoint = mine == 0;
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return nullptr;
    return PyBool_FromLong(disjoint ? 1 : 0);
}

#pragma endregion Algebra

#pragma region Representation and Comparison

static constexpr Py_ssize_t repr_limit_k = 64;

static PyObject *SortedSet_repr(PyObject *self) noexcept {
    auto *set = object_as<sorted_set_object_t>(self);
    Py_ssize_t const total = SortedSet_length(self);

    PyObject *parts = PyList_New(0);
    if (!parts) return nullptr;
    PyObject *cursor = cursor_new(state_of_type(self), self, cursor_yields_t::keys_k, nullptr, nullptr, repr_limit_k);
    if (!cursor) {
        Py_DECREF(parts);
        return nullptr;
    }
    PyObject *member = nullptr;
    while ((member = PyIter_Next(cursor)) != nullptr) {
        PyObject *rendered = PyObject_Repr(member);
        Py_DECREF(member);
        if (!rendered || PyList_Append(parts, rendered) != 0) {
            Py_XDECREF(rendered);
            Py_DECREF(cursor);
            Py_DECREF(parts);
            return nullptr;
        }
        Py_DECREF(rendered);
    }
    Py_DECREF(cursor);
    if (PyErr_Occurred()) {
        Py_DECREF(parts);
        return nullptr;
    }

    PyObject *separator = PyUnicode_FromString(", ");
    PyObject *body = separator ? PyUnicode_Join(separator, parts) : nullptr;
    Py_XDECREF(separator);
    Py_DECREF(parts);
    if (!body) return nullptr;

    PyObject *result = total > repr_limit_k
                           ? PyUnicode_FromFormat("SortedSet(key='%s', {%U, ... +%zd more})", set->base.ops->name, body,
                                                  total - repr_limit_k)
                           : PyUnicode_FromFormat("SortedSet(key='%s', {%U})", set->base.ops->name, body);
    Py_DECREF(body);
    return result;
}

/** @brief Whether every member of @p self is also in @p other. */
static int set_is_subset(PyObject *self, PyObject *other, bool *answer) noexcept {
    module_state_t *state = state_of_type(self);
    if (state)
        if (auto *twin = same_layout_set(self, other, state)) {
            auto *mine = object_as<sorted_set_object_t>(self);
            bool subset = mine->store.size() <= twin->store.size();
            if (subset)
                for_each_in_order(mine->store, mine->base.ops, [&](key_variant_t const &member) noexcept {
                    subset = twin->store.contains(member);
                    return subset;
                });
            *answer = subset;
            return 0;
        }
    PyErr_Clear();

    PyObject *iterator = PyObject_GetIter(self);
    if (!iterator) return -1;
    PyObject *member = nullptr;
    bool subset = true;
    while (subset && (member = PyIter_Next(iterator)) != nullptr) {
        int const shared = PySequence_Contains(other, member);
        Py_DECREF(member);
        if (shared < 0) PyErr_Clear(), subset = false;
        else subset = shared == 1;
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return -1;
    *answer = subset;
    return 0;
}

static PyObject *SortedSet_richcompare(PyObject *self, PyObject *other, int operation) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    bool const other_is_set = Py_IS_TYPE(other, state->sorted_set_type);
    if (!other_is_set && !PyAnySet_Check(other)) Py_RETURN_NOTIMPLEMENTED;

    Py_ssize_t const mine = SortedSet_length(self);
    Py_ssize_t const theirs = PyObject_Length(other);
    if (theirs < 0) return nullptr;

    bool answer = false;
    switch (operation) {
    case Py_EQ:
    case Py_NE: {
        bool subset = false;
        if (mine == theirs && set_is_subset(self, other, &subset) != 0) return nullptr;
        bool const equal = mine == theirs && subset;
        answer = operation == Py_EQ ? equal : !equal;
        break;
    }
    case Py_LE:
    case Py_LT: {
        bool subset = false;
        if (set_is_subset(self, other, &subset) != 0) return nullptr;
        answer = operation == Py_LE ? subset : subset && mine < theirs;
        break;
    }
    case Py_GE:
    case Py_GT: {
        bool superset = false;
        if (set_is_subset(other, self, &superset) != 0) return nullptr;
        answer = operation == Py_GE ? superset : superset && mine > theirs;
        break;
    }
    default: Py_RETURN_NOTIMPLEMENTED;
    }
    return PyBool_FromLong(answer ? 1 : 0);
}

#pragma endregion Representation and Comparison

#pragma region Type Definition

static PyObject *SortedSet_key_type(PyObject *self, void *) noexcept {
    auto *set = object_as<sorted_set_object_t>(self);
    return PyUnicode_FromString(set->base.ops->name);
}

static PyGetSetDef SortedSet_getset[] = {
    {"key_type", SortedSet_key_type, nullptr,
     const_cast<char *>("The member layout this set was built around: 'int', 'uint', 'str' or 'bytes'."), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

template <typename function_type_>
static PyCFunction as_pycfunction(function_type_ function) noexcept {
    return reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(function));
}

static PyMethodDef SortedSet_methods[] = {
    {"add", SortedSet_add, METH_O, doc_add},
    {"discard", SortedSet_discard, METH_O, doc_discard},
    {"remove", SortedSet_remove, METH_O, doc_remove},
    {"pop", SortedSet_pop, METH_NOARGS, doc_pop},
    {"clear", SortedSet_clear, METH_NOARGS, doc_clear},
    {"update", as_pycfunction(SortedSet_update), METH_FASTCALL, doc_update},
    {"union", as_pycfunction(SortedSet_union), METH_FASTCALL, "Members of either side, as a new SortedSet."},
    {"intersection", as_pycfunction(SortedSet_intersection), METH_FASTCALL, "Members of both sides."},
    {"difference", as_pycfunction(SortedSet_difference), METH_FASTCALL, "Members of this side only."},
    {"symmetric_difference", as_pycfunction(SortedSet_symmetric_difference), METH_FASTCALL,
     "Members of exactly one side."},
    {"scan", as_pycfunction(SortedSet_scan), ST_METHOD_FLAGS_, doc_scan},
    {"isdisjoint", as_pycfunction(SortedSet_isdisjoint), METH_FASTCALL, "Whether the two sides share no member."},
    {nullptr, nullptr, 0, nullptr},
};

static PyType_Slot sorted_set_slots[] = {
    {Py_tp_new, reinterpret_cast<void *>(SortedSet_new)},
    {Py_tp_dealloc, reinterpret_cast<void *>(SortedSet_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(SortedSet_traverse)},
    {Py_tp_methods, reinterpret_cast<void *>(SortedSet_methods)},
    {Py_tp_getset, reinterpret_cast<void *>(SortedSet_getset)},
    {Py_tp_iter, reinterpret_cast<void *>(SortedSet_iter)},
    {Py_tp_repr, reinterpret_cast<void *>(SortedSet_repr)},
    {Py_tp_richcompare, reinterpret_cast<void *>(SortedSet_richcompare)},
    {Py_mp_length, reinterpret_cast<void *>(SortedSet_length)},
    {Py_sq_contains, reinterpret_cast<void *>(SortedSet_contains)},
    {Py_tp_doc, const_cast<char *>(doc_SortedSet)},
    {0, nullptr},
};

PyType_Spec sorted_set_spec = {"smashtable.SortedSet", sizeof(sorted_set_object_t), 0,
                               Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, sorted_set_slots};

#pragma endregion Type Definition

} // namespace ashvardanian::smashtable::py
