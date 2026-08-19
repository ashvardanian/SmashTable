/**
 *  @brief The container classes - @c SortedMap, @c SortedSet and their unordered siblings.
 *  @author Ash Vardanian
 *  @file python/smashtable/container.cpp
 *  @date August 18, 2026
 *
 *  One file for every class, because behind @c store_ops_t they differ only in which methods their
 *  type installs. A map and a set once had a file each, near-identical but for the element shape; the
 *  core, the isolation level and the sharing strategy would each have multiplied that again.
 *
 *  Parity with @c dict and @c set is the goal everywhere it costs nothing. The places it is
 *  deliberately broken are three: the key type is fixed at construction and every other type is
 *  refused, iteration never raises on mutation, and @c popitem removes the smallest pair rather than
 *  the most recent.
 */
#include <cstring> // `std::strrchr`

#include "shared.hpp"

namespace ashvardanian::smashtable::py {

#pragma region Construction

/** @brief Casts a fast-convention function into the table's slot without tripping -Wcast-function-type. */
template <typename function_type_>
static PyCFunction as_pycfunction(function_type_ function) noexcept {
    return reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(function));
}

bool is_container(module_state_t *state, PyObject *object) noexcept {
    return Py_IS_TYPE(object, state->sorted_map_type) || Py_IS_TYPE(object, state->sorted_set_type) ||
           Py_IS_TYPE(object, state->hash_map_type) || Py_IS_TYPE(object, state->hash_set_type);
}

store_ops_t const *store_ops_for(core_t core, isolation_choice_t isolation, sharing_choice_t sharing,
                                 bool associative) noexcept {
    switch (core) {
    case core_t::sorted_k: return sorted_store_ops_for(isolation, sharing, associative);
    case core_t::hashed_k: return hashed_store_ops_for(isolation, sharing, associative);
    }
    return nullptr;
}

PyObject *container_of(module_state_t *state, PyTypeObject *type, key_ops_t const *ops, store_ops_t const *store_ops,
                       value_mode_t mode) noexcept {
    auto *self = object_as<container_object_t>(type->tp_alloc(type, 0));
    if (!self) return nullptr;

    expected<void *> made = store_ops->make(ops);
    if (!made) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    self->ops = ops;
    self->store_ops = store_ops;
    self->store = *made;
    self->ordinal = state->next_ordinal.fetch_add(1, std::memory_order_relaxed);
    self->mode = mode;
    return reinterpret_cast<PyObject *>(self);
}

/** @brief Reads the @c isolation keyword, which names a promise rather than a mechanism. */
static bool isolation_from_python(PyObject *specification, isolation_choice_t &choice) noexcept {
    if (!specification || specification == Py_None) return true;
    if (!PyUnicode_Check(specification)) {
        PyErr_SetString(PyExc_TypeError, "isolation must be 'monotonic_atomic_view', 'snapshot' or 'serializable'");
        return false;
    }
    if (PyUnicode_CompareWithASCIIString(specification, "monotonic_atomic_view") == 0) {
        choice = isolation_choice_t::monotonic_k;
        return true;
    }
    if (PyUnicode_CompareWithASCIIString(specification, "snapshot") == 0) {
        choice = isolation_choice_t::snapshot_k;
        return true;
    }
    if (PyUnicode_CompareWithASCIIString(specification, "serializable") == 0) {
        choice = isolation_choice_t::serializable_k;
        return true;
    }
    // A sharded store reports a level weaker than any that can be asked for, so a caller feeding
    // `isolation` back in deserves to be told why rather than shown its own string again.
    PyErr_Format(PyExc_ValueError,                                                               //
                 "isolation must be 'monotonic_atomic_view', 'snapshot' or 'serializable', not " //
                 "%R; a weaker level such as 'read_committed' is delivered by sharding rather "  //
                 "than requested",                                                               //
                 specification);
    return false;
}

/** @brief Reads the @c sharing keyword, which decides how far the isolation level survives. */
static bool sharing_from_python(PyObject *specification, sharing_choice_t &choice) noexcept {
    if (!specification || specification == Py_None) return true;
    if (!PyUnicode_Check(specification)) {
        PyErr_SetString(PyExc_TypeError, "sharing must be 'locked' or 'partitioned'");
        return false;
    }
    if (PyUnicode_CompareWithASCIIString(specification, "locked") == 0) {
        choice = sharing_choice_t::locked_k;
        return true;
    }
    if (PyUnicode_CompareWithASCIIString(specification, "partitioned") == 0) {
        choice = sharing_choice_t::partitioned_k;
        return true;
    }
    PyErr_Format(PyExc_ValueError, "sharing must be 'locked' or 'partitioned', not %R", specification);
    return false;
}

/**
 *  @brief The shared constructor body, which every class reaches with its own core and element shape.
 *
 *  Keywords are walked by hand rather than through @c PyArg_ParseTupleAndKeywords, which parses a
 *  format string at runtime and cannot express the fast calling convention the rest of this file uses.
 */
static PyObject *container_new(PyTypeObject *type, PyObject *args, PyObject *keywords, core_t core,
                               bool associative) noexcept {
    char const *class_name = type->tp_name;
    if (args && PyTuple_GET_SIZE(args) != 0) {
        PyErr_Format(PyExc_TypeError, "%s() takes no positional arguments", class_name);
        return nullptr;
    }

    PyObject *key_specification = nullptr;
    PyObject *value_specification = nullptr;
    PyObject *isolation_specification = nullptr;
    PyObject *sharing_specification = nullptr;
    if (keywords) {
        Py_ssize_t position = 0;
        PyObject *name = nullptr;
        PyObject *value = nullptr;
        while (PyDict_Next(keywords, &position, &name, &value)) {
            if (PyUnicode_CompareWithASCIIString(name, "key") == 0) key_specification = value;
            else if (PyUnicode_CompareWithASCIIString(name, "value") == 0) value_specification = value;
            else if (PyUnicode_CompareWithASCIIString(name, "isolation") == 0) isolation_specification = value;
            else if (PyUnicode_CompareWithASCIIString(name, "sharing") == 0) sharing_specification = value;
            else {
                PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%U'", class_name, name);
                return nullptr;
            }
        }
    }

    key_ops_t const *ops = key_ops_from_python(key_specification);
    if (!ops) return nullptr;
    value_mode_t mode = value_mode_t::scalars_k;
    if (associative && !value_mode_from_python(value_specification, mode)) return nullptr;
    if (!associative && value_specification) {
        PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument 'value'", class_name);
        return nullptr;
    }

    // Locked by default, which is the stronger of the two: a partitioned store takes and releases one
    // partition lock at a time, so a reader spanning partitions can catch a commit half-applied and
    // only Read Committed survives above a single key.
    isolation_choice_t isolation = isolation_choice_t::monotonic_k;
    sharing_choice_t sharing = sharing_choice_t::locked_k;
    if (!isolation_from_python(isolation_specification, isolation)) return nullptr;
    if (!sharing_from_python(sharing_specification, sharing)) return nullptr;

    store_ops_t const *store_ops = store_ops_for(core, isolation, sharing, associative);
    if (!store_ops) {
        PyErr_Format(PyExc_ValueError, "%s() cannot be built with this combination in this build", class_name);
        return nullptr;
    }

    module_state_t *state = state_of_heap_type(type);
    if (!state) return nullptr;
    return container_of(state, type, ops, store_ops, mode);
}

static PyObject *SortedMap_new(PyTypeObject *type, PyObject *args, PyObject *keywords) noexcept {
    return container_new(type, args, keywords, core_t::sorted_k, true);
}

static PyObject *SortedSet_new(PyTypeObject *type, PyObject *args, PyObject *keywords) noexcept {
    return container_new(type, args, keywords, core_t::sorted_k, false);
}

static PyObject *HashMap_new(PyTypeObject *type, PyObject *args, PyObject *keywords) noexcept {
    return container_new(type, args, keywords, core_t::hashed_k, true);
}

static PyObject *HashSet_new(PyTypeObject *type, PyObject *args, PyObject *keywords) noexcept {
    return container_new(type, args, keywords, core_t::hashed_k, false);
}

static void container_dealloc(PyObject *self) noexcept {
    auto *container = object_as<container_object_t>(self);
    // Untracked before anything else, and before the store is torn down: `tp_alloc` tracked this
    // object because the type is a GC type, tearing the store down drops every reference an
    // object-mode container held, and a drop can start a collection. Freeing while still tracked
    // leaves the collector holding a link into freed memory, which surfaces much later as an abort
    // inside an unrelated collection.
    PyObject_GC_UnTrack(self);
    if (container->store) container->store_ops->destroy(container->store);
    PyTypeObject *type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type); // Heap types are reference-counted by their instances
}

/**
 *  @brief Reports the type and every stored object, so a cycle through a store is collectable.
 *
 *  An object-mode container holds strong references, and a collector that is not told about them
 *  sees the objects as externally reachable and never breaks the cycle. Reporting them is what
 *  makes @c tp_clear reachable in the first place.
 */
static int container_traverse(PyObject *self, visitproc visit, void *arg) noexcept {
    Py_VISIT(Py_TYPE(self));
    auto *container = object_as<container_object_t>(self);
    if (container->mode != value_mode_t::objects_k || !container->store) return 0;
    if (!container->store_ops->visit_values) return 0;
    return container->store_ops->visit_values(container->store, visit, arg);
}

/**
 *  @brief Drops everything the store holds, which is how the collector breaks the cycle.
 *
 *  Emptying the store is the whole of it: the references it holds are the only ones a store
 *  owns. The drops go through the store call, so each is released after its lock is gone rather
 *  than inside it - a finalizer here would otherwise deadlock exactly as it did on an ordinary
 *  write.
 */
static int container_gc_clear(PyObject *self) noexcept {
    auto *container = object_as<container_object_t>(self);
    if (container->mode != value_mode_t::objects_k || !container->store) return 0;
    [[maybe_unused]] status_t const emptied = container->store_ops->clear(container->store);
    return 0;
}

#pragma endregion Construction

#pragma region Reading

static Py_ssize_t container_length(PyObject *self) noexcept {
    auto *container = object_as<container_object_t>(self);
    std::size_t size = 0;
    Py_BEGIN_ALLOW_THREADS;
    size = container->store_ops->size(container->store);
    Py_END_ALLOW_THREADS;
    return static_cast<Py_ssize_t>(size);
}

static int container_contains(PyObject *self, PyObject *key) noexcept {
    auto *container = object_as<container_object_t>(self);
    key_variant_t needle;
    // A key of the wrong type cannot be present, and `dict` answers `False` rather than raising, so
    // the refusal is swallowed here and only here.
    if (!key_from_python(key, container->ops, needle)) {
        PyErr_Clear();
        return 0;
    }

    expected<bool> held;
    Py_BEGIN_ALLOW_THREADS;
    held = container->store_ops->contains(container->store, needle);
    Py_END_ALLOW_THREADS;
    status_t const status = held.status();
    bool const found = held && *held;
    // A read that could not record itself is a refusal, and the protocol's error answer is -1.
    if (failed(status)) {
        [[maybe_unused]] int const raised = raise_for(state_of_type(self), status);
        return -1;
    }
    return found ? 1 : 0;
}

static PyObject *Map_subscript(PyObject *self, PyObject *key) noexcept {
    auto *container = object_as<container_object_t>(self);
    key_variant_t needle;
    if (!key_from_python(key, container->ops, needle)) return nullptr;

    expected<value_variant_t> found {key_not_found_k};
    run_over_values(container->mode, [&]() noexcept { found = container->store_ops->find(container->store, needle); });

    if (!found) {
        PyErr_SetObject(PyExc_KeyError, key);
        return nullptr;
    }
    return value_to_python(*found);
}

static char const doc_get[] =                                 //
    "get(key, default=None, /)\n"                             //
    "\n"                                                      //
    "Value for a key, or default when the key is absent.\n"   //
    "\n"                                                      //
    "Raises:\n"                                               //
    "  TypeError: If key is not of this store's key type.\n"; //

static PyObject *Map_get(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count < 1 || count > 2) {
        PyErr_SetString(PyExc_TypeError, "get() takes one or two arguments");
        return nullptr;
    }
    PyObject *found = Map_subscript(self, args[0]);
    if (found) return found;
    if (!PyErr_ExceptionMatches(PyExc_KeyError)) return nullptr;
    PyErr_Clear();
    return Py_NewRef(count == 2 ? args[1] : Py_None);
}

#pragma endregion Reading

#pragma region Writing

/** @brief The class name without the module prefix, which every message and repr leads with. */
static char const *class_name_of(PyObject *self) noexcept;

/**
 *  @brief Erases a half-open window named by a slice, with either end optional.
 *
 *  Installed only on the ordered classes, so an unordered container refuses this as an
 *  @c AttributeError from the type rather than as a runtime check invented here.
 *
 *  @return 0 on success, -1 with an exception set.
 */
static int container_delete_slice(PyObject *self, PyObject *slice) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;

    // A slice always carries all three members, with `None` where one was left out, so they are read
    // straight off the object rather than through an attribute lookup that could fail.
    auto const *bounds = reinterpret_cast<PySliceObject *>(slice);
    if (bounds->step != Py_None) {
        PyErr_SetString(PyExc_ValueError, "a step has no meaning over a range of keys");
        return -1;
    }

    key_variant_t lower;
    key_variant_t upper;
    bool const has_lower = bounds->start != Py_None;
    bool const has_upper = bounds->stop != Py_None;
    if (has_lower && !key_from_python(bounds->start, container->ops, lower)) return -1;
    if (has_upper && !key_from_python(bounds->stop, container->ops, upper)) return -1;

    status_t status = success_k;
    run_over_values(container->mode, [&]() noexcept {
        status = container->store_ops->erase_range(container->store, has_lower ? &lower : nullptr,
                                                   has_upper ? &upper : nullptr);
    });
    return raise_for(state, status);
}

static int Map_assign_subscript(PyObject *self, PyObject *key, PyObject *value) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;

    // A slice names a window rather than a key, and only ever arrives at a delete: assigning to one
    // would have to mean writing a value to every key in it, which no store offers. An unordered
    // container has no window to name, and says so here rather than reaching a table slot its core
    // never filled.
    if (PySlice_Check(key)) {
        if (!container->store_ops->erase_range) {
            PyErr_Format(PyExc_TypeError, "%s has no ordering, so it cannot be sliced", class_name_of(self));
            return -1;
        }
        if (value) {
            PyErr_SetString(PyExc_TypeError, "a slice of this store cannot be assigned to");
            return -1;
        }
        return container_delete_slice(self, key);
    }

    key_variant_t stored_key;
    if (!key_from_python(key, container->ops, stored_key)) return -1;

    if (!value) { // `del map[key]`
        // One erase under one lock: the status reports `key_not_found_k` for a key that was never
        // there, so absence needs no separate probe and two threads racing on the same key cannot
        // both believe they removed it.
        status_t status = success_k;
        run_over_values(container->mode, [&]() noexcept {
            status = container->store_ops->erase(container->store, stored_key).status();
        });
        return raise_for(state, status, key);
    }

    value_variant_t stored_value;
    if (!value_from_python(value, container->mode, stored_value)) return -1;

    status_t status = success_k;
    run_over_values(container->mode, [&]() noexcept {
        status = container->store_ops->upsert(container->store, std::move(stored_key), &stored_value);
    });
    return raise_for(state, status, key);
}

static char const doc_clear[] = //
    "clear()\n"                 //
    "\n"                        //
    "Remove every entry. The key type stays as it was.\n";

static PyObject *container_clear(PyObject *self, PyObject *) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    status_t status = success_k;
    run_over_values(container->mode, [&]() noexcept { status = container->store_ops->clear(container->store); });
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

static PyObject *Map_pop(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (count < 1 || count > 2) {
        PyErr_SetString(PyExc_TypeError, "pop() takes one or two arguments");
        return nullptr;
    }

    key_variant_t needle;
    if (!key_from_python(args[0], container->ops, needle)) {
        if (count == 2) {
            PyErr_Clear();
            return Py_NewRef(args[1]);
        }
        return nullptr;
    }

    expected<value_variant_t> removed {key_not_found_k};
    run_over_values(container->mode,
                    [&]() noexcept { removed = container->store_ops->erase(container->store, needle); });
    bool const present = static_cast<bool>(removed);

    if (!present) {
        // Absence is `key_not_found_k`, which is the caller's answer rather than a failure; anything
        // else genuinely went wrong and is raised.
        if (removed.status() != key_not_found_k && raise_for(state, removed.status(), args[0]) != 0) return nullptr;
        if (count == 2) return Py_NewRef(args[1]);
        PyErr_SetObject(PyExc_KeyError, args[0]);
        return nullptr;
    }
    return value_to_python(*removed);
}

static char const doc_popmin[] =                                                   //
    "popmin()\n"                                                                   //
    "\n"                                                                           //
    "Remove and return the smallest (key, value) pair.\n"                          //
    "\n"                                                                           //
    "Named for what it does rather than borrowing dict's popitem, which removes\n" //
    "the most recently inserted pair. The store steps forward only, so the\n"      //
    "largest key would cost a full walk while the smallest costs one lookup.\n"    //
    "\n"                                                                           //
    "Raises:\n"                                                                    //
    "  KeyError: If the store is empty.\n";                                        //

static PyObject *Map_popmin(PyObject *self, PyObject *) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    key_variant_t smallest_key;
    expected<value_variant_t> removed {key_not_found_k};
    run_over_values(container->mode, [&]() noexcept {
        key_variant_t floor;
        container->ops->least(floor);
        // Two locked spans, because no store offers "remove the smallest" as one. The pair reported
        // is the one the erase actually took, not the one the seek saw, so a key that vanished in
        // between reads as absent rather than as a pair nobody removed. A key inserted below it in
        // between is still missed, which is what a `pop_smallest` on the store would close.
        bool found = false;
        if (status_t const sought =
                container->store_ops->lower_bound(container->store, floor, smallest_key, nullptr, found);
            failed(sought)) {
            removed = expected<value_variant_t> {sought};
            return;
        }
        if (!found) return;
        removed = container->store_ops->erase(container->store, smallest_key);
    });

    if (!removed) {
        // The seek found nothing, or the key it found was taken before the erase reached it. Either
        // way there is no pair to hand back, and only a genuine failure is worth raising.
        if (removed.status() != key_not_found_k && raise_for(state, removed.status()) != 0) return nullptr;
        PyErr_SetString(PyExc_KeyError, "popmin(): store is empty");
        return nullptr;
    }

    PyObject *key_object = key_to_python(smallest_key);
    if (!key_object) return nullptr;
    PyObject *value_object = value_to_python(*removed);
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

static PyObject *Map_setdefault(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (count < 1 || count > 2) {
        PyErr_SetString(PyExc_TypeError, "setdefault() takes one or two arguments");
        return nullptr;
    }

    key_variant_t stored_key;
    if (!key_from_python(args[0], container->ops, stored_key)) return nullptr;
    PyObject *fallback = count == 2 ? args[1] : Py_None;
    value_variant_t stored_value;
    if (!value_from_python(fallback, container->mode, stored_value)) return nullptr;

    expected<value_variant_t> winner {key_not_found_k};
    run_over_values(container->mode, [&]() noexcept {
        // One strict insert that leaves the winner readable. A key arriving concurrently keeps its own
        // value, and what comes back is that winner rather than what we tried to store.
        winner = container->store_ops->insert_if_missing(container->store, stored_key, std::move(stored_value));
    });

    if (!winner && raise_for(state, winner.status(), args[0]) != 0) return nullptr;
    return value_to_python(*winner);
}

static char const doc_map_update[] =                                               //
    "update(other, /)\n"                                                           //
    "\n"                                                                           //
    "Apply every pair of a mapping or an iterable of pairs.\n"                     //
    "\n"                                                                           //
    "The batch lands as one unit: the store stages it and commits it once, so a\n" //
    "failure part-way leaves nothing applied. This is stronger than dict, which\n" //
    "applies pairs one at a time.\n";                                              //

static PyObject *Map_update(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
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

    // Converted in full before anything is written, so the store sees one batch and a bad pair
    // half-way through leaves the store untouched rather than half-updated. This is marshalling,
    // not a workaround: turning Python objects into stored ones is the binding's own job.
    Py_ssize_t const total = PySequence_Fast_GET_SIZE(fast);
    auto staged = basic_vector<entry_t>::make(static_cast<std::size_t>(total));
    if (!staged) {
        Py_DECREF(fast);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t index = 0; index != total; ++index) {
        PyObject *pair = PySequence_Fast_GET_ITEM(fast, index);
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

        key_variant_t key;
        value_variant_t value;
        bool const read = key_from_python(PySequence_Fast_GET_ITEM(unpacked, 0), container->ops, key) &&
                          value_from_python(PySequence_Fast_GET_ITEM(unpacked, 1), container->mode, value);
        Py_DECREF(unpacked);
        if (!read) {
            Py_DECREF(fast);
            return nullptr;
        }
        if (failed((*staged).push_back(assume_reserved, entry_t {std::move(key), std::move(value)}))) {
            Py_DECREF(fast);
            return PyErr_NoMemory();
        }
    }
    Py_DECREF(fast);

    status_t status = success_k;
    run_over_values(container->mode, [&]() noexcept {
        status = container->store_ops->upsert_entries(container->store, (*staged).data(), (*staged).size());
    });
    if (raise_for(state, status) != 0) return nullptr;
    Py_RETURN_NONE;
}

#pragma endregion Writing

#pragma region Set Surface

static char const doc_add[] = //
    "add(member, /)\n"        //
    "\n"                      //
    "Insert a member. Adding one that is already present does nothing.\n";

static PyObject *Set_add(PyObject *self, PyObject *member) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    key_variant_t stored;
    if (!key_from_python(member, container->ops, stored)) return nullptr;

    status_t status = success_k;
    Py_BEGIN_ALLOW_THREADS;
    status = container->store_ops->upsert(container->store, std::move(stored), nullptr);
    Py_END_ALLOW_THREADS;
    if (raise_for(state, status, member) != 0) return nullptr;
    Py_RETURN_NONE;
}

/** @brief The shared body of @c discard and @c remove, which differ only in the absent case. */
static int set_erase(PyObject *self, PyObject *member, bool *was_present) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return -1;

    key_variant_t stored;
    if (!key_from_python(member, container->ops, stored)) return -1;

    status_t status = success_k;
    bool present = false;
    Py_BEGIN_ALLOW_THREADS;
    expected<value_variant_t> const removed = container->store_ops->erase(container->store, stored);
    present = static_cast<bool>(removed);
    status = removed.status();
    Py_END_ALLOW_THREADS;

    *was_present = present;
    if (!present) return 0;
    return raise_for(state, status, member);
}

static char const doc_discard[] = //
    "discard(member, /)\n"        //
    "\n"                          //
    "Remove a member if present, and say nothing if not.\n";

static PyObject *Set_discard(PyObject *self, PyObject *member) noexcept {
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

static PyObject *Set_remove(PyObject *self, PyObject *member) noexcept {
    bool present = false;
    if (set_erase(self, member, &present) != 0) return nullptr;
    if (!present) {
        PyErr_SetObject(PyExc_KeyError, member);
        return nullptr;
    }
    Py_RETURN_NONE;
}

static char const doc_set_popmin[] =                              //
    "popmin()\n"                                                  //
    "\n"                                                          //
    "Remove and return the smallest member.\n"                    //
    "\n"                                                          //
    "Unlike set, which pops an arbitrary member, this pops the\n" //
    "smallest, which the ordered store reaches in one lookup.\n"  //
    "\n"                                                          //
    "Raises:\n"                                                   //
    "  KeyError: If the set is empty.\n";                         //

static PyObject *Set_popmin(PyObject *self, PyObject *) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;

    key_variant_t smallest;
    bool present = false;
    status_t status = success_k;
    Py_BEGIN_ALLOW_THREADS;
    key_variant_t floor;
    container->ops->least(floor);
    // See the map's popmin: the member reported is the one the erase took, not the one the seek saw.
    bool found = false;
    status_t const sought = container->store_ops->lower_bound(container->store, floor, smallest, nullptr, found);
    if (failed(sought)) status = sought;
    else if (found) {
        expected<value_variant_t> const removed = container->store_ops->erase(container->store, smallest);
        present = static_cast<bool>(removed);
        status = removed.status();
    }
    Py_END_ALLOW_THREADS;

    if (!present) {
        PyErr_SetString(PyExc_KeyError, "popmin(): set is empty");
        return nullptr;
    }
    if (raise_for(state, status) != 0) return nullptr;
    return key_to_python(smallest);
}

static char const doc_set_update[] =                                               //
    "update(other, /)\n"                                                           //
    "\n"                                                                           //
    "Add every member of an iterable.\n"                                           //
    "\n"                                                                           //
    "The batch lands as one unit: the store stages it and commits it once, so a\n" //
    "failure part-way leaves nothing applied. This is stronger than set, which\n"  //
    "adds members one at a time.\n";                                               //

static PyObject *Set_update(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    auto *container = object_as<container_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "update() takes exactly one argument");
        return nullptr;
    }

    PyObject *fast = PySequence_Fast(args[0], "update() needs an iterable of members");
    if (!fast) return nullptr;
    Py_ssize_t const total = PySequence_Fast_GET_SIZE(fast);

    // Converted in full first, so a member the store cannot hold leaves it untouched rather than
    // partly grown. Marshalling Python objects into stored ones is the binding's own work.
    auto staged = basic_vector<key_variant_t>::make(static_cast<std::size_t>(total));
    if (!staged) {
        Py_DECREF(fast);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t index = 0; index != total; ++index) {
        key_variant_t member;
        if (!key_from_python(PySequence_Fast_GET_ITEM(fast, index), container->ops, member)) {
            Py_DECREF(fast);
            return nullptr;
        }
        if (failed((*staged).push_back(assume_reserved, std::move(member)))) {
            Py_DECREF(fast);
            return PyErr_NoMemory();
        }
    }
    Py_DECREF(fast);

    status_t status = success_k;
    Py_BEGIN_ALLOW_THREADS;
    status = container->store_ops->upsert_members(container->store, (*staged).data(), (*staged).size());
    Py_END_ALLOW_THREADS;
    if (raise_for(state, status) != 0) return nullptr;
    Py_RETURN_NONE;
}

#pragma endregion Set Surface

#pragma region Algebra

/** @brief Which members of the walked side end up in the result. */
enum class algebra_t : std::uint8_t { union_k, intersection_k, difference_k, symmetric_difference_k };

/** @brief Builds a fresh set of the same class, layout and store configuration, ready to receive results. */
static PyObject *set_like(PyObject *self) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    auto const *container = object_as<container_object_t>(self);
    return container_of(state, Py_TYPE(self), container->ops, container->store_ops, container->mode);
}

/**
 *  @brief The other side as a set built exactly like this one, or null when it is anything else.
 *
 *  Both tables must match, not just the key layout: the native paths below drive the other side's
 *  store through this side's table, which is only sound when the two were built from the same one.
 */
static container_object_t *same_layout_set(PyObject *self, PyObject *other) noexcept {
    if (!Py_IS_TYPE(other, Py_TYPE(self))) return nullptr;
    auto *theirs = object_as<container_object_t>(other);
    auto const *mine = object_as<container_object_t>(self);
    return theirs->ops == mine->ops && theirs->store_ops == mine->store_ops ? theirs : nullptr;
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
static status_t set_algebra_natively(container_object_t *mine, container_object_t *theirs, container_object_t *result,
                                     algebra_t operation) noexcept {
    status_t outcome = success_k;
    store_ops_t const *table = mine->store_ops;

    auto absorb = [&](key_variant_t const &member, membership_t membership, side_t side) noexcept {
        if (failed(outcome) || !keeps_member(operation, membership, side)) return;
        auto copied = member.copy();
        if (!copied) {
            outcome = copied.status();
            return;
        }
        outcome = table->upsert(result->store, std::move(*copied), nullptr);
    };

    auto membership_in = [&](void *store, key_variant_t const &member) noexcept {
        expected<bool> const held = table->contains(store, member);
        if (!held) outcome = held.status();
        return held && *held ? membership_t::shared_k : membership_t::absent_k;
    };

    for_each_in_order(mine, [&](key_variant_t const &member, value_variant_t const &) noexcept {
        absorb(member, membership_in(theirs->store, member), side_t::mine_k);
    });

    // Union and symmetric difference also need what only the other side holds.
    if (operation == algebra_t::union_k || operation == algebra_t::symmetric_difference_k)
        for_each_in_order(theirs, [&](key_variant_t const &member, value_variant_t const &) noexcept {
            absorb(member, membership_in(mine->store, member), side_t::theirs_k);
        });

    // The status is threaded out rather than collapsed into a memory error: a table that ran out of
    // probe slots is not a heap that ran out of memory, and the two want different remedies.
    return outcome;
}

static PyObject *set_algebra(PyObject *self, PyObject *other, algebra_t operation) noexcept {
    if (!PyObject_HasAttrString(other, "__contains__")) Py_RETURN_NOTIMPLEMENTED;

    PyObject *result = set_like(self);
    if (!result) return nullptr;

    if (auto *twin = same_layout_set(self, other)) {
        auto *mine = object_as<container_object_t>(self);
        auto *filled = object_as<container_object_t>(result);
        status_t const outcome = set_algebra_natively(mine, twin, filled, operation);
        if (raise_for(state_of_type(self), outcome) != 0) {
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
        // A member the other side cannot even hold is simply not shared. Anything else - a failing
        // `__eq__`, an interrupt, a memory error - is a real failure and must not be swallowed.
        if (shared < 0) {
            if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
                Py_DECREF(member);
                Py_DECREF(iterator);
                Py_DECREF(result);
                return nullptr;
            }
            PyErr_Clear();
        }
        switch (operation) {
        case algebra_t::union_k: keep = true; break;
        case algebra_t::intersection_k: keep = shared == 1; break;
        case algebra_t::difference_k:
        case algebra_t::symmetric_difference_k: keep = shared != 1; break;
        }
        if (keep) {
            PyObject *added = Set_add(result, member);
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
            if (mine < 0) {
                if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
                    Py_DECREF(member);
                    Py_DECREF(other_iterator);
                    Py_DECREF(result);
                    return nullptr;
                }
                PyErr_Clear();
            }
            bool const keep = operation == algebra_t::union_k || mine != 1;
            if (keep) {
                PyObject *added = Set_add(result, member);
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

static PyObject *Set_union(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "union() takes exactly one argument");
        return nullptr;
    }
    return set_algebra(self, args[0], algebra_t::union_k);
}

static PyObject *Set_intersection(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "intersection() takes exactly one argument");
        return nullptr;
    }
    return set_algebra(self, args[0], algebra_t::intersection_k);
}

static PyObject *Set_difference(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "difference() takes exactly one argument");
        return nullptr;
    }
    return set_algebra(self, args[0], algebra_t::difference_k);
}

static PyObject *Set_symmetric_difference(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "symmetric_difference() takes exactly one argument");
        return nullptr;
    }
    return set_algebra(self, args[0], algebra_t::symmetric_difference_k);
}

static PyObject *Set_isdisjoint(PyObject *self, PyObject *const *args, Py_ssize_t count) noexcept {
    if (count != 1) {
        PyErr_SetString(PyExc_TypeError, "isdisjoint() takes exactly one argument");
        return nullptr;
    }

    if (auto *twin = same_layout_set(self, args[0])) {
        auto *mine = object_as<container_object_t>(self);
        store_ops_t const *table = mine->store_ops;
        bool disjoint = true;
        // Probing the smaller side keeps this O(min(n, m) log max(n, m)) rather than always O(n log m).
        auto const *smaller = table->size(mine->store) <= table->size(twin->store) ? mine : twin;
        auto const *larger = smaller == mine ? twin : mine;
        status_t asked = success_k;
        for_each_in_order(smaller, [&](key_variant_t const &member, value_variant_t const &) noexcept {
            expected<bool> const held = table->contains(larger->store, member);
            if (!held) asked = held.status();
            disjoint = !(held && *held);
            return disjoint;
        });
        if (failed(asked)) return raise_for(state_of_type(self), asked), nullptr;
        return PyBool_FromLong(disjoint ? 1 : 0);
    }

    auto const *container = object_as<container_object_t>(self);
    PyObject *iterator = PyObject_GetIter(args[0]);
    if (!iterator) return nullptr;
    PyObject *member = nullptr;
    bool disjoint = true;
    while (disjoint && (member = PyIter_Next(iterator)) != nullptr) {
        // Converted here rather than asked through `__contains__`, which answers False for anything
        // it cannot hold and clears the reason. A member of a foreign type is genuinely not in this
        // set, but a failing `__hash__` or an interrupt is not an answer and must not read as one.
        key_variant_t needle;
        bool const convertible = key_from_python(member, container->ops, needle);
        Py_DECREF(member);
        if (!convertible) {
            if (!PyErr_ExceptionMatches(PyExc_TypeError) && !PyErr_ExceptionMatches(PyExc_OverflowError)) {
                Py_DECREF(iterator);
                return nullptr;
            }
            PyErr_Clear();
            continue; // Nothing this set can hold, so it shares nothing with it
        }
        expected<bool> held;
        Py_BEGIN_ALLOW_THREADS;
        held = container->store_ops->contains(container->store, needle);
        Py_END_ALLOW_THREADS;
        status_t const asked = held.status();
        bool const found = held && *held;
        if (failed(asked)) {
            Py_DECREF(member);
            Py_DECREF(iterator);
            [[maybe_unused]] int const raised = raise_for(state_of_type(self), asked);
            return nullptr;
        }
        disjoint = !found;
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return nullptr;
    return PyBool_FromLong(disjoint ? 1 : 0);
}

#pragma endregion Algebra

#pragma region Views and Iteration

static PyObject *container_iter(PyObject *self) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    return cursor_new(state, self, cursor_yields_t::keys_k, nullptr, nullptr, -1);
}

static PyObject *Map_keys(PyObject *self, PyObject *) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    return mapping_view_new(state, self, cursor_yields_t::keys_k);
}

static PyObject *Map_values(PyObject *self, PyObject *) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    return mapping_view_new(state, self, cursor_yields_t::values_k);
}

static PyObject *Map_items(PyObject *self, PyObject *) noexcept {
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
    "  TypeError: If a bound is not of this store's key type.\n";                         //

/** @brief Drives the shared cursor into a list, so there is one traversal in the binding. */
static PyObject *container_scan(PyObject *self, PyObject *const *args, Py_ssize_t count, PyObject *keywords,
                                cursor_yields_t yields) noexcept {
    auto const *container = object_as<container_object_t>(self);
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
                    // Negative is the cursor's own spelling for uncounted, so a caller passing one
                    // would silently receive the whole container rather than nothing.
                    if (limit < 0) {
                        PyErr_SetString(PyExc_ValueError, "limit cannot be negative");
                        return nullptr;
                    }
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
    if (has_start && !key_from_python(start_object, container->ops, start_key)) return nullptr;
    if (has_stop && !key_from_python(stop_object, container->ops, stop_key)) return nullptr;

    PyObject *cursor =
        cursor_new(state, self, yields, has_start ? &start_key : nullptr, has_stop ? &stop_key : nullptr, limit);
    if (!cursor) return nullptr;
    PyObject *collected = PySequence_List(cursor);
    Py_DECREF(cursor);
    return collected;
}

static PyObject *Map_scan(PyObject *self, PyObject *const *args, Py_ssize_t count, PyObject *keywords) noexcept {
    return container_scan(self, args, count, keywords, cursor_yields_t::items_k);
}

static char const doc_set_scan[] =                                                 //
    "scan(start=None, stop=None, *, limit=None)\n"                                 //
    "\n"                                                                           //
    "List of members in order, over the half-open window [start, stop).\n"         //
    "\n"                                                                           //
    "Bounds need not be present members. Materialized rather than lazy: iterate\n" //
    "the set itself when the whole range does not need to exist at once.\n"        //
    "\n"                                                                           //
    "Raises:\n"                                                                    //
    "  TypeError: If a bound is not of this set's key type.\n";                    //

static PyObject *Set_scan(PyObject *self, PyObject *const *args, Py_ssize_t count, PyObject *keywords) noexcept {
    return container_scan(self, args, count, keywords, cursor_yields_t::keys_k);
}

#pragma endregion Views and Iteration

#pragma region Representation and Equality

/** @brief How many elements @c repr spells out before eliding, so a huge container still prints. */
static constexpr Py_ssize_t repr_limit_k = 64;

static char const *class_name_of(PyObject *self) noexcept {
    char const *qualified = Py_TYPE(self)->tp_name;
    char const *dot = std::strrchr(qualified, '.');
    return dot ? dot + 1 : qualified;
}

/**
 *  @brief Renders the contents in order, which only an ordered container can do.
 *  @param[in] yields Pairs for a map, bare members for a set.
 */
static PyObject *container_repr(PyObject *self, cursor_yields_t yields) noexcept {
    auto *container = object_as<container_object_t>(self);
    Py_ssize_t const total = container_length(self);

    PyObject *parts = PyList_New(0);
    if (!parts) return nullptr;

    module_state_t *state = state_of_type(self);
    if (!state) {
        Py_DECREF(parts);
        return nullptr;
    }
    PyObject *cursor = cursor_new(state, self, yields, nullptr, nullptr, repr_limit_k);
    if (!cursor) {
        Py_DECREF(parts);
        return nullptr;
    }
    PyObject *element = nullptr;
    while ((element = PyIter_Next(cursor)) != nullptr) {
        PyObject *rendered =
            yields == cursor_yields_t::items_k
                ? PyUnicode_FromFormat("%R: %R", PyTuple_GET_ITEM(element, 0), PyTuple_GET_ITEM(element, 1))
                : PyUnicode_FromFormat("%R", element);
        Py_DECREF(element);
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

    PyObject *result =
        total > repr_limit_k
            ? PyUnicode_FromFormat("%s(key='%s', {%U, ... +%zd more})", class_name_of(self), container->ops->name, body,
                                   total - repr_limit_k)
            : PyUnicode_FromFormat("%s(key='%s', {%U})", class_name_of(self), container->ops->name, body);
    Py_DECREF(body);
    return result;
}

static PyObject *Map_repr(PyObject *self) noexcept { return container_repr(self, cursor_yields_t::items_k); }
static PyObject *Set_repr(PyObject *self) noexcept { return container_repr(self, cursor_yields_t::keys_k); }

/**
 *  @brief Reports the shape rather than the contents, which an unordered core cannot walk.
 *
 *  A container that cannot enumerate itself has no honest way to print its elements, and printing
 *  some arbitrary subset would read as an order it does not have.
 */
static PyObject *Unordered_repr(PyObject *self) noexcept {
    auto const *container = object_as<container_object_t>(self);
    return PyUnicode_FromFormat("%s(key='%s', %zd entries)", class_name_of(self), container->ops->name,
                                container_length(self));
}

/**
 *  @brief Compares against another map of this build or a @c dict, by content and never by arrival order.
 *
 *  Walks this store's store once and probes the other side per key. Against another map of the same
 *  layout the key never becomes a Python object at all - it is compared as a stored scalar, through the
 *  same function pointer the tree orders by. Against a @c dict the key is built once and looked up
 *  through the C hash API. Neither path materializes a copy or dispatches through the interpreter.
 *
 *  Values do go through @c PyObject_RichCompareBool, deliberately: that is what keeps @c {k: 1} equal
 *  to @c {k: 1.0} as it is for @c dict, without reimplementing cross-type numeric comparison here.
 *
 *  @c dict itself answers @c NotImplemented against a foreign mapping and defers to the other operand,
 *  so implementing this is the convention rather than an extra - without it two containers holding
 *  identical data would compare unequal by identity, silently and in both directions.
 */
static PyObject *Map_richcompare(PyObject *self, PyObject *other, int operation) noexcept {
    if (operation != Py_EQ && operation != Py_NE) Py_RETURN_NOTIMPLEMENTED;

    bool const other_is_map = Py_IS_TYPE(other, Py_TYPE(self));
    if (!other_is_map && !PyDict_Check(other)) Py_RETURN_NOTIMPLEMENTED;

    Py_ssize_t const their_size = PyObject_Length(other);
    if (their_size < 0) return nullptr;

    auto *container = object_as<container_object_t>(self);
    bool equal = container_length(self) == their_size;
    bool raised = false;

    // Two maps of different key layouts can hold no key in common, so the sizes matching already
    // settles it unless both are empty.
    if (equal && other_is_map && object_as<container_object_t>(other)->ops != container->ops) equal = their_size == 0;

    if (equal) {
        auto *twin = other_is_map ? object_as<container_object_t>(other) : nullptr;
        for_each_in_order(container, [&](key_variant_t const &key, value_variant_t const &mapped) noexcept {
            if (!equal || raised) return;

            value_variant_t theirs;
            bool present = false;
            PyObject *their_value = nullptr;
            if (twin) {
                // Same layout, so the key stays a stored scalar and never becomes a Python object.
                auto probed = twin->store_ops->find(twin->store, key);
                present = static_cast<bool>(probed);
                if (present) theirs = std::move(*probed);
            }
            else {
                PyObject *key_object = key_to_python(key);
                if (!key_object) return void(raised = true);
                their_value = PyDict_GetItemWithError(other, key_object); // Borrowed, or null
                Py_DECREF(key_object);
                present = their_value != nullptr;
                if (!present && PyErr_Occurred()) return void(raised = true);
            }
            if (!present) return void(equal = false);

            PyObject *mine = value_to_python(mapped);
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

/** @brief Whether every member of @p self is also in @p other. */
static int set_is_subset(PyObject *self, PyObject *other, bool *answer) noexcept {
    module_state_t *state = state_of_type(self);
    if (state)
        if (auto *twin = same_layout_set(self, other)) {
            auto *mine = object_as<container_object_t>(self);
            store_ops_t const *table = mine->store_ops;
            bool subset = table->size(mine->store) <= table->size(twin->store);
            status_t asked = success_k;
            if (subset)
                for_each_in_order(mine, [&](key_variant_t const &member, value_variant_t const &) noexcept {
                    expected<bool> const held = table->contains(twin->store, member);
                    if (!held) asked = held.status();
                    subset = held && *held;
                    return subset;
                });
            if (failed(asked)) return raise_for(state, asked);
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
        if (shared < 0) {
            if (!PyErr_ExceptionMatches(PyExc_TypeError)) {
                Py_DECREF(iterator);
                return -1;
            }
            PyErr_Clear();
            subset = false;
        }
        else subset = shared == 1;
    }
    Py_DECREF(iterator);
    if (PyErr_Occurred()) return -1;
    *answer = subset;
    return 0;
}

/** @brief The whole comparison surface a @c set carries, equality and containment alike. */
static PyObject *Set_richcompare(PyObject *self, PyObject *other, int operation) noexcept {
    bool const other_is_set = Py_IS_TYPE(other, Py_TYPE(self));
    if (!other_is_set && !PyAnySet_Check(other)) Py_RETURN_NOTIMPLEMENTED;

    Py_ssize_t const mine = container_length(self);
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

#pragma endregion Representation and Equality

#pragma region Attributes

static PyObject *container_key_type(PyObject *self, void *) noexcept {
    return PyUnicode_FromString(object_as<container_object_t>(self)->ops->name);
}

static PyObject *container_value_mode(PyObject *self, void *) noexcept {
    return PyUnicode_FromString(object_as<container_object_t>(self)->mode == value_mode_t::objects_k ? "object"
                                                                                                     : "scalar");
}

static PyObject *container_isolation(PyObject *self, void *) noexcept {
    return PyUnicode_FromString(object_as<container_object_t>(self)->store_ops->isolation_name);
}

static PyObject *container_sharing(PyObject *self, void *) noexcept {
    return PyUnicode_FromString(object_as<container_object_t>(self)->store_ops->sharing_name);
}

static char const doc_key_type[] = "The key layout this store was built around: 'int', 'uint', 'str' or 'bytes'.";
static char const doc_value_mode[] = "'scalar' when values must be scalars, 'object' when any object is stored.";
static char const doc_sharing[] =                                             //
    "How this store is shared between threads: 'locked' or 'partitioned'.\n"  //
    "\n"                                                                      //
    "What was asked for, unlike isolation, which reports what is delivered."; //

static char const doc_isolation[] =                                                     //
    "What this store actually promises a reader, as Jepsen names it.\n"                 //
    "\n"                                                                                //
    "The effective level rather than the one asked for. A snapshot store keeps its\n"   //
    "level across partitions, since visibility there is a comparison against a stamp\n" //
    "rather than a lock somebody holds. A monotonic one sharded across partitions\n"    //
    "reports 'read_committed', because a reader holding no stamp can catch a commit\n"  //
    "half-applied.";                                                                    //

static PyGetSetDef map_getset[] = {
    {"value_mode", container_value_mode, nullptr, const_cast<char *>(doc_value_mode), nullptr},
    {"key_type", container_key_type, nullptr, const_cast<char *>(doc_key_type), nullptr},
    {"isolation", container_isolation, nullptr, const_cast<char *>(doc_isolation), nullptr},
    {"sharing", container_sharing, nullptr, const_cast<char *>(doc_sharing), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

static PyGetSetDef set_getset[] = {
    {"key_type", container_key_type, nullptr, const_cast<char *>(doc_key_type), nullptr},
    {"isolation", container_isolation, nullptr, const_cast<char *>(doc_isolation), nullptr},
    {"sharing", container_sharing, nullptr, const_cast<char *>(doc_sharing), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

#pragma endregion Attributes

#pragma region Type Definitions

static char const doc_SortedMap[] =                                                      //
    "SortedMap(*, key, value='scalar', isolation='monotonic', sharing='locked')\n"       //
    "\n"                                                                                 //
    "An ordered mapping whose writes can be grouped into transactions.\n"                //
    "\n"                                                                                 //
    "Keys are homogeneous and their type is fixed at construction, which is what lets\n" //
    "every comparison skip type dispatch. Floats and booleans may be values but never\n" //
    "keys.\n"                                                                            //
    "\n"                                                                                 //
    "Args:\n"                                                                            //
    "  key (type or str): One of int, str, bytes, or 'int', 'uint', 'str', 'bytes'.\n"   //
    "  value (str): 'scalar' to store copies, 'object' to hold arbitrary objects.\n"     //
    "  isolation (str): 'monotonic' or 'snapshot'. See the isolation property.\n"        //
    "  sharing (str): 'locked' for one mutex, 'partitioned' for sixteen.\n"              //
    "\n"                                                                                 //
    "Raises:\n"                                                                          //
    "  TypeError: If key is missing, or names float or bool.\n"                          //
    "  ValueError: If key, value, isolation or sharing names an unknown choice.\n";      //

static char const doc_SortedSet[] =                                                    //
    "SortedSet(*, key, isolation='monotonic', sharing='locked')\n"                     //
    "\n"                                                                               //
    "An ordered set whose writes can be grouped into transactions.\n"                  //
    "\n"                                                                               //
    "Members are homogeneous and their type is fixed at construction. Floats and\n"    //
    "booleans are not valid members, for the same reason they are not valid keys.\n"   //
    "\n"                                                                               //
    "Args:\n"                                                                          //
    "  key (type or str): One of int, str, bytes, or 'int', 'uint', 'str', 'bytes'.\n" //
    "  isolation (str): 'monotonic' or 'snapshot'. See the isolation property.\n"      //
    "  sharing (str): 'locked' for one mutex, 'partitioned' for sixteen.\n";           //

static PyMethodDef SortedMap_methods[] = {
    {"get", as_pycfunction(Map_get), METH_FASTCALL, doc_get},
    {"clear", container_clear, METH_NOARGS, doc_clear},
    {"pop", as_pycfunction(Map_pop), METH_FASTCALL, doc_pop},
    {"popmin", Map_popmin, METH_NOARGS, doc_popmin},
    {"setdefault", as_pycfunction(Map_setdefault), METH_FASTCALL, doc_setdefault},
    {"update", as_pycfunction(Map_update), METH_FASTCALL, doc_map_update},
    {"keys", Map_keys, METH_NOARGS, "A lazy view over the keys, in order."},
    {"values", Map_values, METH_NOARGS, "A lazy view over the values, in key order."},
    {"scan", as_pycfunction(Map_scan), ST_METHOD_FLAGS_, doc_scan},
    {"items", Map_items, METH_NOARGS, "A lazy view over the (key, value) pairs, in key order."},
    {nullptr, nullptr, 0, nullptr},
};

static PyMethodDef SortedSet_methods[] = {
    {"add", Set_add, METH_O, doc_add},
    {"discard", Set_discard, METH_O, doc_discard},
    {"remove", Set_remove, METH_O, doc_remove},
    {"popmin", Set_popmin, METH_NOARGS, doc_set_popmin},
    {"clear", container_clear, METH_NOARGS, doc_clear},
    {"update", as_pycfunction(Set_update), METH_FASTCALL, doc_set_update},
    {"union", as_pycfunction(Set_union), METH_FASTCALL, "Members of either side, as a new set."},
    {"intersection", as_pycfunction(Set_intersection), METH_FASTCALL, "Members of both sides."},
    {"difference", as_pycfunction(Set_difference), METH_FASTCALL, "Members of this side only."},
    {"symmetric_difference", as_pycfunction(Set_symmetric_difference), METH_FASTCALL, "Members of exactly one side."},
    {"scan", as_pycfunction(Set_scan), ST_METHOD_FLAGS_, doc_set_scan},
    {"isdisjoint", as_pycfunction(Set_isdisjoint), METH_FASTCALL, "Whether the two sides share no member."},
    {nullptr, nullptr, 0, nullptr},
};

static PyType_Slot sorted_map_slots[] = {
    {Py_tp_new, reinterpret_cast<void *>(SortedMap_new)},
    {Py_tp_dealloc, reinterpret_cast<void *>(container_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(container_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(container_gc_clear)},
    {Py_tp_methods, reinterpret_cast<void *>(SortedMap_methods)},
    {Py_tp_getset, reinterpret_cast<void *>(map_getset)},
    {Py_tp_iter, reinterpret_cast<void *>(container_iter)},
    {Py_tp_repr, reinterpret_cast<void *>(Map_repr)},
    {Py_tp_richcompare, reinterpret_cast<void *>(Map_richcompare)},
    {Py_mp_length, reinterpret_cast<void *>(container_length)},
    {Py_mp_subscript, reinterpret_cast<void *>(Map_subscript)},
    {Py_mp_ass_subscript, reinterpret_cast<void *>(Map_assign_subscript)},
    {Py_sq_contains, reinterpret_cast<void *>(container_contains)},
    {Py_tp_doc, const_cast<char *>(doc_SortedMap)},
    {0, nullptr},
};

/** @brief A set has no values, so its only subscript duty is erasing a window. */
static int Set_assign_subscript(PyObject *self, PyObject *key, PyObject *value) noexcept {
    if (!PySlice_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "a set is subscripted only by a slice, to erase a window");
        return -1;
    }
    if (value) {
        PyErr_SetString(PyExc_TypeError, "a slice of this store cannot be assigned to");
        return -1;
    }
    return container_delete_slice(self, key);
}

static PyType_Slot sorted_set_slots[] = {
    {Py_mp_ass_subscript, reinterpret_cast<void *>(Set_assign_subscript)},
    {Py_tp_new, reinterpret_cast<void *>(SortedSet_new)},
    {Py_tp_dealloc, reinterpret_cast<void *>(container_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(container_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(container_gc_clear)},
    {Py_tp_methods, reinterpret_cast<void *>(SortedSet_methods)},
    {Py_tp_getset, reinterpret_cast<void *>(set_getset)},
    {Py_tp_iter, reinterpret_cast<void *>(container_iter)},
    {Py_tp_repr, reinterpret_cast<void *>(Set_repr)},
    {Py_tp_richcompare, reinterpret_cast<void *>(Set_richcompare)},
    {Py_mp_length, reinterpret_cast<void *>(container_length)},
    {Py_sq_contains, reinterpret_cast<void *>(container_contains)},
    {Py_tp_doc, const_cast<char *>(doc_SortedSet)},
    {0, nullptr},
};

static char const doc_HashMap[] =                                                          //
    "HashMap(*, key, value='scalar', isolation='monotonic', sharing='locked')\n"           //
    "\n"                                                                                   //
    "An unordered mapping whose writes can be grouped into transactions.\n"                //
    "\n"                                                                                   //
    "Point access only. The open-addressed core supplies no ordering, so this class has\n" //
    "no iteration, no keys/values/items, no scan and no range erase - reaching for one\n"  //
    "is an AttributeError rather than an empty result. Use SortedMap when order or\n"      //
    "enumeration matters.\n"                                                               //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  key (type or str): One of int, str, bytes, or 'int', 'uint', 'str', 'bytes'.\n"     //
    "  value (str): 'scalar' to store copies, 'object' to hold arbitrary objects.\n"       //
    "  isolation (str): 'monotonic' or 'snapshot'. See the isolation property.\n"          //
    "  sharing (str): 'locked' for one mutex, 'partitioned' for sixteen.\n";               //

static char const doc_HashSet[] =                                                           //
    "HashSet(*, key, isolation='monotonic', sharing='locked')\n"                            //
    "\n"                                                                                    //
    "An unordered set whose writes can be grouped into transactions.\n"                     //
    "\n"                                                                                    //
    "Membership and mutation only. The core supplies no ordering, so this class has no\n"   //
    "iteration, no scan, no pop and none of the set algebra, each of which would have to\n" //
    "walk at least one side. Use SortedSet when any of those matter.\n";                    //

static PyMethodDef HashMap_methods[] = {
    {"get", as_pycfunction(Map_get), METH_FASTCALL, doc_get},
    {"clear", container_clear, METH_NOARGS, doc_clear},
    {"pop", as_pycfunction(Map_pop), METH_FASTCALL, doc_pop},
    {"setdefault", as_pycfunction(Map_setdefault), METH_FASTCALL, doc_setdefault},
    {"update", as_pycfunction(Map_update), METH_FASTCALL, doc_map_update},
    {nullptr, nullptr, 0, nullptr},
};

static PyMethodDef HashSet_methods[] = {
    {"add", Set_add, METH_O, doc_add},
    {"discard", Set_discard, METH_O, doc_discard},
    {"remove", Set_remove, METH_O, doc_remove},
    {"clear", container_clear, METH_NOARGS, doc_clear},
    {"update", as_pycfunction(Set_update), METH_FASTCALL, doc_set_update},
    {nullptr, nullptr, 0, nullptr},
};

static PyType_Slot hash_map_slots[] = {
    {Py_tp_new, reinterpret_cast<void *>(HashMap_new)},
    {Py_tp_dealloc, reinterpret_cast<void *>(container_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(container_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(container_gc_clear)},
    {Py_tp_methods, reinterpret_cast<void *>(HashMap_methods)},
    {Py_tp_getset, reinterpret_cast<void *>(map_getset)},
    {Py_tp_repr, reinterpret_cast<void *>(Unordered_repr)},
    {Py_mp_length, reinterpret_cast<void *>(container_length)},
    {Py_mp_subscript, reinterpret_cast<void *>(Map_subscript)},
    {Py_mp_ass_subscript, reinterpret_cast<void *>(Map_assign_subscript)},
    {Py_sq_contains, reinterpret_cast<void *>(container_contains)},
    {Py_tp_doc, const_cast<char *>(doc_HashMap)},
    {0, nullptr},
};

static PyType_Slot hash_set_slots[] = {
    {Py_tp_new, reinterpret_cast<void *>(HashSet_new)},
    {Py_tp_dealloc, reinterpret_cast<void *>(container_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(container_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(container_gc_clear)},
    {Py_tp_methods, reinterpret_cast<void *>(HashSet_methods)},
    {Py_tp_getset, reinterpret_cast<void *>(set_getset)},
    {Py_tp_repr, reinterpret_cast<void *>(Unordered_repr)},
    {Py_mp_length, reinterpret_cast<void *>(container_length)},
    {Py_sq_contains, reinterpret_cast<void *>(container_contains)},
    {Py_tp_doc, const_cast<char *>(doc_HashSet)},
    {0, nullptr},
};

PyType_Spec hash_map_spec = {"smashtable.HashMap", sizeof(container_object_t), 0,
                             Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, hash_map_slots};

PyType_Spec hash_set_spec = {"smashtable.HashSet", sizeof(container_object_t), 0,
                             Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, hash_set_slots};

PyType_Spec sorted_map_spec = {"smashtable.SortedMap", sizeof(container_object_t), 0,
                               Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, sorted_map_slots};

PyType_Spec sorted_set_spec = {"smashtable.SortedSet", sizeof(container_object_t), 0,
                               Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, sorted_set_slots};

#pragma endregion Type Definitions

} // namespace ashvardanian::smashtable::py
