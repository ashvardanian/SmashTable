/**
 *  @brief @c smashtable.SortedMap - an ordered, transactional mapping with @c dict's surface.
 *  @author Ash Vardanian
 *  @file python/smashtable/sorted_map.cpp
 *  @date August 16, 2026
 *
 *  Parity with @c dict is the goal everywhere it costs nothing, and the places it is deliberately broken
 *  are three: the key type is fixed at construction and every other type is refused, iteration never
 *  raises on mutation, and @c popitem removes the smallest pair rather than the most recent.
 */
#include "shared.hpp"

namespace ashvardanian::smashtable::py {

#pragma region Construction

static char const doc_SortedMap[] =                                                      //
    "SortedMap(*, key)\n"                                                                //
    "\n"                                                                                 //
    "An ordered mapping whose writes can be grouped into transactions.\n"                //
    "\n"                                                                                 //
    "Keys are homogeneous and their type is fixed at construction, which is what lets\n" //
    "every comparison skip type dispatch. Floats and booleans may be values but never\n" //
    "keys.\n"                                                                            //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  key (type or str): One of int, str, bytes, or 'int', 'uint', 'str', 'bytes'.\n"   //
    "\n"                                                                                 //
    "Raises:\n"                                                                          //
    "  TypeError: If key is missing, or names float or bool.\n"                          //
    "  ValueError: If key names an unknown layout.\n";                                   //

static PyObject *SortedMap_new(PyTypeObject *type, PyObject *args, PyObject *keywords) noexcept {
    // Walked by hand rather than through `PyArg_ParseTupleAndKeywords`, which parses a format string
    // at runtime and cannot express the fast calling convention the rest of this file uses.
    if (args && PyTuple_GET_SIZE(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "SortedMap() takes no positional arguments");
        return nullptr;
    }
    PyObject *key_specification = nullptr;
    PyObject *value_specification = nullptr;
    if (keywords) {
        Py_ssize_t position = 0;
        PyObject *name = nullptr;
        PyObject *value = nullptr;
        while (PyDict_Next(keywords, &position, &name, &value)) {
            if (PyUnicode_CompareWithASCIIString(name, "key") == 0) key_specification = value;
            else if (PyUnicode_CompareWithASCIIString(name, "value") == 0) value_specification = value;
            else {
                PyErr_Format(PyExc_TypeError, "SortedMap() got an unexpected keyword argument '%U'", name);
                return nullptr;
            }
        }
    }

    key_ops_t const *ops = key_ops_from_python(key_specification);
    if (!ops) return nullptr;
    value_mode_t mode = value_mode_t::scalars_k;
    if (!value_mode_from_python(value_specification, mode)) return nullptr;
    module_state_t *state = state_of_heap_type(type);
    if (!state) return nullptr;

    auto *self = object_as<sorted_map_object_t>(type->tp_alloc(type, 0));
    if (!self) return nullptr;

    auto made = map_store_t::make(key_less_t {ops->less}, key_hash_t {ops->hash});
    if (!made) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    // `tp_alloc` hands back zeroed storage rather than a constructed object, so the store is built
    // in place. Nothing owns it but this object, and `tp_dealloc` destroys it.
    new (&self->store) map_store_t(std::move(*made));
    self->base.ops = ops;
    self->base.ordinal = state->next_ordinal.fetch_add(1, std::memory_order_relaxed);
    self->base.mode = mode;
    return reinterpret_cast<PyObject *>(self);
}

static void SortedMap_dealloc(PyObject *self) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    map->store.~map_store_t();
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type); // Heap types are reference-counted by their instances
}

static int SortedMap_traverse(PyObject *self, visitproc visit, void *arg) noexcept {
    Py_VISIT(Py_TYPE(self));
    return 0;
}

#pragma endregion Construction

#pragma region Reading

static Py_ssize_t SortedMap_length(PyObject *self) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    std::size_t size = 0;
    Py_BEGIN_ALLOW_THREADS;
    size = map->store.size();
    Py_END_ALLOW_THREADS;
    return static_cast<Py_ssize_t>(size);
}

static int SortedMap_contains(PyObject *self, PyObject *key) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    key_variant_t needle;
    // A key of the wrong type cannot be present, and `dict` answers `False` rather than raising, so
    // the refusal is swallowed here and only here.
    if (!key_from_python(key, map->base.ops, needle)) {
        PyErr_Clear();
        return 0;
    }

    bool found = false;
    Py_BEGIN_ALLOW_THREADS;
    found = map->store.contains(needle);
    Py_END_ALLOW_THREADS;
    return found ? 1 : 0;
}

static PyObject *SortedMap_subscript(PyObject *self, PyObject *key) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    key_variant_t needle;
    if (!key_from_python(key, map->base.ops, needle)) return nullptr;

    value_variant_t found;
    bool present = false;
    run_over_values(map->base.mode, [&]() noexcept {
        map->store.find(
            needle, [&](entry_t const &entry) noexcept { found = entry.mapped, present = true; }, []() noexcept {});
    });

    if (!present) {
        PyErr_SetObject(PyExc_KeyError, key);
        return nullptr;
    }
    return value_to_python(found);
}

static char const doc_get[] =                                     //
    "get(key, default=None, /)\n"                                 //
    "\n"                                                          //
    "Value for a key, or default when the key is absent.\n"       //
    "\n"                                                          //
    "Raises:\n"                                                   //
    "  TypeError: If key is not of this container's key type.\n"; //

static PyObject *SortedMap_get(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count < 1 || count > 2) {
        PyErr_SetString(PyExc_TypeError, "get() takes one or two arguments");
        return nullptr;
    }
    PyObject *found = SortedMap_subscript(self, args[0]);
    if (found) return found;
    if (!PyErr_ExceptionMatches(PyExc_KeyError)) return nullptr;
    PyErr_Clear();
    return Py_NewRef(count == 2 ? args[1] : Py_None);
}

#pragma endregion Reading

#pragma region Writing

static int SortedMap_assign_subscript(PyObject *self, PyObject *key, PyObject *value) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;

    key_variant_t stored_key;
    if (!key_from_python(key, map->base.ops, stored_key)) return -1;

    if (!value) { // `del map[key]`
        // One erase under one lock: the status reports `key_not_found_k` for a key that was never
        // there, so absence needs no separate probe and two threads racing on the same key cannot
        // both believe they removed it.
        status_t status = success_k;
        run_over_values(map->base.mode, [&]() noexcept { status = map->store.erase(stored_key); });
        return raise_for(state, status, key);
    }

    value_variant_t stored_value;
    if (!value_from_python(value, map->base.mode, stored_value)) return -1;

    status_t status = success_k;
    run_over_values(map->base.mode, [&]() noexcept {
        status = map->store.upsert(entry_t {std::move(stored_key), std::move(stored_value)});
    });
    return raise_for(state, status, key);
}

static char const doc_clear[] = //
    "clear()\n"                 //
    "\n"                        //
    "Remove every entry. The key type stays as it was.\n";

static PyObject *SortedMap_clear(PyObject *self, PyObject *) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    status_t status = success_k;
    run_over_values(map->base.mode, [&]() noexcept { status = map->store.clear(); });
    if (raise_for(state, status) != 0) return nullptr;
    Py_RETURN_NONE;
}

static char const doc_pop[] =                                       //
    "pop(key, default, /)\n"                                        //
    "\n"                                                            //
    "Remove a key and return its value.\n"                          //
    "\n"                                                            //
    "Raises:\n"                                                     //
    "  KeyError: If the key is absent and no default was given.\n"; //

static PyObject *SortedMap_pop(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (count < 1 || count > 2) {
        PyErr_SetString(PyExc_TypeError, "pop() takes one or two arguments");
        return nullptr;
    }

    key_variant_t needle;
    if (!key_from_python(args[0], map->base.ops, needle)) {
        if (count == 2) {
            PyErr_Clear();
            return Py_NewRef(args[1]);
        }
        return nullptr;
    }

    value_variant_t found;
    bool present = false;
    status_t status = success_k;
    run_over_values(map->base.mode, [&]() noexcept {
        map->store.find(
            needle, [&](entry_t const &entry) noexcept { found = entry.mapped, present = true; }, []() noexcept {});
        if (present) status = map->store.erase(needle);
    });

    if (!present) {
        if (count == 2) return Py_NewRef(args[1]);
        PyErr_SetObject(PyExc_KeyError, args[0]);
        return nullptr;
    }
    if (raise_for(state, status, args[0]) != 0) return nullptr;
    return value_to_python(found);
}

static char const doc_popitem[] =                                                   //
    "popitem()\n"                                                                   //
    "\n"                                                                            //
    "Remove and return the smallest (key, value) pair.\n"                           //
    "\n"                                                                            //
    "Unlike dict, which pops the most recently inserted pair, this pops the\n"      //
    "smallest key. The store steps forward only, so the largest key would cost a\n" //
    "full walk while the smallest costs a single bounded lookup.\n"                 //
    "\n"                                                                            //
    "Raises:\n"                                                                     //
    "  KeyError: If the container is empty.\n";                                     //

static PyObject *SortedMap_popitem(PyObject *self, PyObject *) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    key_variant_t smallest_key;
    value_variant_t smallest_value;
    bool present = false;
    status_t status = success_k;
    run_over_values(map->base.mode, [&]() noexcept {
        key_variant_t floor;
        map->base.ops->least(floor);
        map->store.lower_bound(
            floor,
            [&](entry_t const &entry) noexcept {
                smallest_key = entry.key, smallest_value = entry.mapped, present = true;
            },
            []() noexcept {});
        if (present) status = map->store.erase(smallest_key);
    });

    if (!present) {
        PyErr_SetString(PyExc_KeyError, "popitem(): container is empty");
        return nullptr;
    }
    if (raise_for(state, status) != 0) return nullptr;

    PyObject *key_object = key_to_python(smallest_key);
    if (!key_object) return nullptr;
    PyObject *value_object = value_to_python(smallest_value);
    if (!value_object) {
        Py_DECREF(key_object);
        return nullptr;
    }
    PyObject *pair = PyTuple_Pack(2, key_object, value_object);
    Py_DECREF(key_object);
    Py_DECREF(value_object);
    return pair;
}

static char const doc_setdefault[] =                                                //
    "setdefault(key, default=None, /)\n"                                            //
    "\n"                                                                            //
    "Value for a key, inserting default first when the key is absent.\n"            //
    "\n"                                                                            //
    "The insertion is one strict operation rather than a lookup followed by a\n"    //
    "write, so two threads racing on the same key cannot both believe they won.\n"; //

static PyObject *SortedMap_setdefault(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (count < 1 || count > 2) {
        PyErr_SetString(PyExc_TypeError, "setdefault() takes one or two arguments");
        return nullptr;
    }

    key_variant_t stored_key;
    if (!key_from_python(args[0], map->base.ops, stored_key)) return nullptr;
    PyObject *fallback = count == 2 ? args[1] : Py_None;
    value_variant_t stored_value;
    if (!value_from_python(fallback, map->base.mode, stored_value)) return nullptr;

    value_variant_t existing;
    bool present = false;
    status_t status = success_k;
    run_over_values(map->base.mode, [&]() noexcept {
        // One strict insert that says which branch it took. A key arriving concurrently keeps its own
        // value, and `callback_existing` hands back that winner rather than what we tried to store.
        status = map->store.insert_if_missing(
            entry_t {key_variant_t {stored_key}, std::move(stored_value)}, [](entry_t const &) noexcept {},
            [&](entry_t const &entry) noexcept { existing = entry.mapped, present = true; });
    });

    if (raise_for(state, status, args[0]) != 0) return nullptr;
    if (present) return value_to_python(existing);
    return Py_NewRef(fallback);
}

static char const doc_update[] =                                                  //
    "update(other, /)\n"                                                          //
    "\n"                                                                          //
    "Apply every pair of a mapping or an iterable of pairs.\n"                    //
    "\n"                                                                          //
    "Not atomic, matching dict. Open a transaction when the batch must land as\n" //
    "one unit.\n";                                                                //

static PyObject *SortedMap_update(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "update() takes exactly one argument");
        return nullptr;
    }

    bool const is_mapping = PyObject_HasAttrString(args[0], "keys");
    PyObject *pairs = is_mapping ? PyMapping_Items(args[0]) : Py_NewRef(args[0]);
    if (!pairs) return nullptr;
    PyObject *fast = PySequence_Fast(pairs, "update() needs a mapping or an iterable of pairs");
    Py_DECREF(pairs);
    if (!fast) return nullptr;

    Py_ssize_t const total = PySequence_Fast_GET_SIZE(fast);
    for (Py_ssize_t index = 0; index != total; ++index) {
        PyObject *pair = PySequence_Fast_GET_ITEM(fast, index);
        PyObject *key = nullptr;
        PyObject *value = nullptr;
        if (!PyArg_ParseTuple(pair, "OO", &key, &value)) {
            Py_DECREF(fast);
            return nullptr;
        }
        if (SortedMap_assign_subscript(self, key, value) != 0) {
            Py_DECREF(fast);
            return nullptr;
        }
    }
    Py_DECREF(fast);
    Py_RETURN_NONE;
}

#pragma endregion Writing

#pragma region Views and Iteration

static PyObject *SortedMap_iter(PyObject *self) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    return cursor_new(state, self, cursor_yields_t::keys_k, nullptr, nullptr, -1);
}

static PyObject *SortedMap_keys(PyObject *self, PyObject *) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    return mapping_view_new(state, self, cursor_yields_t::keys_k);
}

static PyObject *SortedMap_values(PyObject *self, PyObject *) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    return mapping_view_new(state, self, cursor_yields_t::values_k);
}

static PyObject *SortedMap_items(PyObject *self, PyObject *) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    return mapping_view_new(state, self, cursor_yields_t::items_k);
}

static char const doc_scan[] =                                                            //
    "scan(start=None, stop=None, *, limit=None)\n"                                        //
    "\n"                                                                                  //
    "List of (key, value) pairs in key order, over the half-open window [start, stop).\n" //
    "\n"                                                                                  //
    "Bounds need not be present keys. Materialized rather than lazy: iterate the\n"       //
    "container itself when the whole range does not need to exist at once.\n"             //
    "\n"                                                                                  //
    "Raises:\n"                                                                           //
    "  TypeError: If a bound is not of this container's key type.\n";                     //

/** @brief Drives the shared cursor into a list, so there is one traversal in the binding. */
static PyObject *SortedMap_scan(PyObject *self, PyObject *const *args, Py_ssize_t count, PyObject *keywords) noexcept {
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

    PyObject *cursor = cursor_new(state, self, cursor_yields_t::items_k, has_start ? &start_key : nullptr,
                                  has_stop ? &stop_key : nullptr, limit);
    if (!cursor) return nullptr;
    PyObject *collected = PySequence_List(cursor);
    Py_DECREF(cursor);
    return collected;
}

#pragma endregion Views and Iteration

#pragma region Representation and Equality

/** @brief How many pairs @c repr spells out before eliding, so a huge container still prints. */
static constexpr Py_ssize_t repr_limit_k = 64;

static PyObject *SortedMap_repr(PyObject *self) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    Py_ssize_t const total = SortedMap_length(self);

    PyObject *parts = PyList_New(0);
    if (!parts) return nullptr;

    PyObject *cursor = cursor_new(state_of_type(self), self, cursor_yields_t::items_k, nullptr, nullptr, repr_limit_k);
    if (!cursor) {
        Py_DECREF(parts);
        return nullptr;
    }
    PyObject *pair = nullptr;
    while ((pair = PyIter_Next(cursor)) != nullptr) {
        PyObject *rendered = PyUnicode_FromFormat("%R: %R", PyTuple_GET_ITEM(pair, 0), PyTuple_GET_ITEM(pair, 1));
        Py_DECREF(pair);
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
                           ? PyUnicode_FromFormat("SortedMap(key='%s', {%U, ... +%zd more})", map->base.ops->name, body,
                                                  total - repr_limit_k)
                           : PyUnicode_FromFormat("SortedMap(key='%s', {%U})", map->base.ops->name, body);
    Py_DECREF(body);
    return result;
}

/**
 *  @brief Compares against another @c SortedMap or a @c dict, by content and never by arrival order.
 *
 *  Walks this container's store once and probes the other side per key. Against another @c SortedMap
 *  the key never becomes a Python object at all - it is compared as a stored scalar, through the same
 *  function pointer the tree orders by. Against a @c dict the key is built once and looked up through
 *  the C hash API. Neither path materializes a copy or dispatches through the interpreter.
 *
 *  Values do go through @c PyObject_RichCompareBool, deliberately: that is what keeps @c {k: 1} equal
 *  to @c {k: 1.0} as it is for @c dict, without reimplementing cross-type numeric comparison here.
 *
 *  @c dict itself answers @c NotImplemented against a foreign mapping and defers to the other operand,
 *  so implementing this is the convention rather than an extra - without it two containers holding
 *  identical data would compare unequal by identity, silently and in both directions.
 */
static PyObject *SortedMap_richcompare(PyObject *self, PyObject *other, int operation) noexcept {
    if (operation != Py_EQ && operation != Py_NE) Py_RETURN_NOTIMPLEMENTED;

    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    bool const other_is_map = Py_IS_TYPE(other, state->sorted_map_type);
    if (!other_is_map && !PyDict_Check(other)) Py_RETURN_NOTIMPLEMENTED;

    Py_ssize_t const their_size = PyObject_Length(other);
    if (their_size < 0) return nullptr;

    auto *map = object_as<sorted_map_object_t>(self);
    bool equal = SortedMap_length(self) == their_size;
    bool raised = false;

    // Two maps of different key layouts can hold no key in common, so the sizes matching already
    // settles it unless both are empty.
    if (equal && other_is_map && object_as<container_object_t>(other)->ops != map->base.ops) equal = their_size == 0;

    if (equal) {
        auto *twin = other_is_map ? object_as<sorted_map_object_t>(other) : nullptr;
        for_each_in_order(map->store, map->base.ops, [&](entry_t const &entry) noexcept {
            if (!equal || raised) return;

            value_variant_t theirs;
            bool present = false;
            PyObject *their_value = nullptr;
            if (twin) {
                // Same layout, so the key stays a stored scalar and never becomes a Python object.
                twin->store.find(
                    entry.key, [&](entry_t const &found) noexcept { theirs = found.mapped, present = true; },
                    []() noexcept {});
            }
            else {
                PyObject *key_object = key_to_python(entry.key);
                if (!key_object) return void(raised = true);
                their_value = PyDict_GetItemWithError(other, key_object); // Borrowed, or null
                Py_DECREF(key_object);
                present = their_value != nullptr;
                if (!present && PyErr_Occurred()) return void(raised = true);
            }
            if (!present) return void(equal = false);

            PyObject *mine = value_to_python(entry.mapped);
            if (!mine) return void(raised = true);
            PyObject *theirs_object = twin ? value_to_python(theirs) : Py_NewRef(their_value);
            if (!theirs_object) {
                Py_DECREF(mine);
                return void(raised = true);
            }
            int const same = PyObject_RichCompareBool(mine, theirs_object, Py_EQ);
            Py_DECREF(mine);
            Py_DECREF(theirs_object);
            if (same < 0) return void(raised = true);
            equal = same == 1;
        });
    }

    if (raised) return nullptr;
    return PyBool_FromLong((operation == Py_EQ) == equal);
}

#pragma endregion Representation and Equality

#pragma region Attributes

static PyObject *SortedMap_key_type(PyObject *self, void *) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    return PyUnicode_FromString(map->base.ops->name);
}

static PyObject *SortedMap_value_mode(PyObject *self, void *) noexcept {
    auto *map = object_as<sorted_map_object_t>(self);
    return PyUnicode_FromString(map->base.mode == value_mode_t::objects_k ? "object" : "scalar");
}

static PyGetSetDef SortedMap_getset[] = {
    {"value_mode", SortedMap_value_mode, nullptr,
     const_cast<char *>("'scalar' when values must be scalars, 'object' when any object is stored."), nullptr},
    {"key_type", SortedMap_key_type, nullptr,
     const_cast<char *>("The key layout this container was built around: 'int', 'uint', 'str' or 'bytes'."), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

#pragma endregion Attributes

#pragma region Type Definition

/** @brief Casts a fast-convention function into the table's slot without tripping -Wcast-function-type. */
template <typename function_type_>
static PyCFunction as_pycfunction(function_type_ function) noexcept {
    return reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(function));
}

static PyMethodDef SortedMap_methods[] = {
    {"get", as_pycfunction(SortedMap_get), METH_FASTCALL, doc_get},
    {"clear", SortedMap_clear, METH_NOARGS, doc_clear},
    {"pop", as_pycfunction(SortedMap_pop), METH_FASTCALL, doc_pop},
    {"popitem", SortedMap_popitem, METH_NOARGS, doc_popitem},
    {"setdefault", as_pycfunction(SortedMap_setdefault), METH_FASTCALL, doc_setdefault},
    {"update", as_pycfunction(SortedMap_update), METH_FASTCALL, doc_update},
    {"keys", SortedMap_keys, METH_NOARGS, "A lazy view over the keys, in order."},
    {"values", SortedMap_values, METH_NOARGS, "A lazy view over the values, in key order."},
    {"scan", as_pycfunction(SortedMap_scan), ST_METHOD_FLAGS_, doc_scan},
    {"items", SortedMap_items, METH_NOARGS, "A lazy view over the (key, value) pairs, in key order."},
    {nullptr, nullptr, 0, nullptr},
};

static PyType_Slot sorted_map_slots[] = {
    {Py_tp_new, reinterpret_cast<void *>(SortedMap_new)},
    {Py_tp_dealloc, reinterpret_cast<void *>(SortedMap_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(SortedMap_traverse)},
    {Py_tp_methods, reinterpret_cast<void *>(SortedMap_methods)},
    {Py_tp_getset, reinterpret_cast<void *>(SortedMap_getset)},
    {Py_tp_iter, reinterpret_cast<void *>(SortedMap_iter)},
    {Py_tp_repr, reinterpret_cast<void *>(SortedMap_repr)},
    {Py_tp_richcompare, reinterpret_cast<void *>(SortedMap_richcompare)},
    {Py_mp_length, reinterpret_cast<void *>(SortedMap_length)},
    {Py_mp_subscript, reinterpret_cast<void *>(SortedMap_subscript)},
    {Py_mp_ass_subscript, reinterpret_cast<void *>(SortedMap_assign_subscript)},
    {Py_sq_contains, reinterpret_cast<void *>(SortedMap_contains)},
    {Py_tp_doc, const_cast<char *>(doc_SortedMap)},
    {0, nullptr},
};

PyType_Spec sorted_map_spec = {"smashtable.SortedMap", sizeof(sorted_map_object_t), 0,
                               Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, sorted_map_slots};

#pragma endregion Type Definition

} // namespace ashvardanian::smashtable::py
