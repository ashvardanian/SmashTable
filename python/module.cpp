/**
 *  @brief Module definition, per-interpreter state, exception types and the @c atomic entry point.
 *  @author Ash Vardanian
 *  @file python/module.cpp
 *  @date October 30, 2025
 *
 *  Multi-phase initialization with heap types is required rather than stylistic: @c Py_mod_gil is
 *  reachable only through a module slot, and a static type cannot be shared safely between
 *  interpreters. Everything the module owns lives in per-module state for the same reason.
 */
#include "shared.hpp"

namespace ashvardanian::smashtable::py {

#pragma region Module State

module_state_t *state_of(PyObject *module) noexcept { return static_cast<module_state_t *>(PyModule_GetState(module)); }

module_state_t *state_of_heap_type(PyTypeObject *type) noexcept {
    PyObject *module = PyType_GetModuleByDef(type, smashtable_module_def());
    return module ? state_of(module) : nullptr;
}

module_state_t *state_of_type(PyObject *self) noexcept {
    PyObject *module = PyType_GetModuleByDef(Py_TYPE(self), smashtable_module_def());
    return module ? state_of(module) : nullptr;
}

#pragma endregion Module State

#pragma region Entry Points

static char const doc_transaction[] =                                                  //
    "transaction(*stores)\n"                                                           //
    "\n"                                                                               //
    "Open one transaction spanning every given store.\n"                               //
    "\n"                                                                               //
    "Used as a context manager, the block either makes every change visible or none\n" //
    "of them. Stores may mix maps and sets, ordered and unordered, and may use\n"      //
    "different key types and isolation levels.\n"                                      //
    "\n"                                                                               //
    "Args:\n"                                                                          //
    "  *stores: One or more SortedMap, SortedSet, HashMap or HashSet, each at\n"       //
    "    most once. A store's own transaction() is this with one argument.\n"          //
    "\n"                                                                               //
    "Returns:\n"                                                                       //
    "  Transaction: Whose participants arrive in the order the stores were given.\n"   //
    "\n"                                                                               //
    "Raises:\n"                                                                        //
    "  TypeError: If no store was given, or one is not a store.\n"                     //
    "  ValueError: If the same store is given twice.\n";                               //

static PyObject *module_transaction(PyObject *module, PyObject *const *args, Py_ssize_t count) noexcept {
    module_state_t *state = state_of(module);
    if (!state) return nullptr;
    if (count < 1) {
        PyErr_SetString(PyExc_TypeError, "transaction() needs at least one store");
        return nullptr;
    }

    PyObject *containers = PyTuple_New(count);
    if (!containers) return nullptr;
    for (Py_ssize_t index = 0; index != count; ++index) PyTuple_SET_ITEM(containers, index, Py_NewRef(args[index]));
    PyObject *group = make_transaction(state, containers);
    Py_DECREF(containers);
    return group;
}

/** @brief Opens a transaction over one store alone, the single-participant case. */
static PyObject *container_transaction(PyObject *self, PyObject *) noexcept {
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    PyObject *containers = PyTuple_Pack(1, self);
    if (!containers) return nullptr;
    PyObject *group = make_transaction(state, containers);
    Py_DECREF(containers);
    return group;
}

static PyMethodDef module_methods[] = {
    {"transaction", as_pycfunction(module_transaction), METH_FASTCALL, doc_transaction},
    {nullptr, nullptr, 0, nullptr},
};

#pragma endregion Entry Points

#pragma region Initialization

/** @brief Outlives every descriptor built from it, which is why it cannot be a local. */
static PyMethodDef transaction_definition = {"transaction", container_transaction, METH_NOARGS,
                                             "Open a transaction over this store alone."};

/**
 *  @brief Attaches @c transaction() to a store type after it is built from its spec.
 *
 *  Added here rather than in each container's method table because it has to reach the module state to
 *  find the transaction type, and both container families want the identical method.
 */
static int add_transaction_method(PyTypeObject *type) noexcept {
    PyObject *descriptor = PyDescr_NewMethod(type, &transaction_definition);
    if (!descriptor) return -1;
    int const added = PyObject_SetAttrString(reinterpret_cast<PyObject *>(type), "transaction", descriptor);
    Py_DECREF(descriptor);
    return added;
}

static int module_exec(PyObject *module) noexcept {
    module_state_t *state = state_of(module);

    state->next_ordinal.store(0, std::memory_order_relaxed);

    state->error = PyErr_NewException("smashtable.SmashTableError", nullptr, nullptr);
    if (!state->error) return -1;

    PyObject *conflict_bases = PyTuple_Pack(2, state->error, PyExc_RuntimeError);
    if (!conflict_bases) return -1;
    state->conflict_error = PyErr_NewException("smashtable.ConflictError", conflict_bases, nullptr);
    Py_DECREF(conflict_bases);
    if (!state->conflict_error) return -1;

    // Each names one way an optimistic validation turned a transaction away, and each subclasses
    // `ConflictError`, so a caller that only wants to retry keeps catching the base and one that
    // wants to know whether retrying can help catches the leaf.
    PyObject *write_conflict_bases = PyTuple_Pack(1, state->conflict_error);
    if (!write_conflict_bases) return -1;
    state->write_conflict_error = PyErr_NewException("smashtable.WriteConflictError", write_conflict_bases, nullptr);
    Py_DECREF(write_conflict_bases);
    if (!state->write_conflict_error) return -1;

    PyObject *read_conflict_bases = PyTuple_Pack(1, state->conflict_error);
    if (!read_conflict_bases) return -1;
    state->read_conflict_error = PyErr_NewException("smashtable.ReadConflictError", read_conflict_bases, nullptr);
    Py_DECREF(read_conflict_bases);
    if (!state->read_conflict_error) return -1;

    PyObject *phantom_conflict_bases = PyTuple_Pack(1, state->conflict_error);
    if (!phantom_conflict_bases) return -1;
    state->phantom_conflict_error =
        PyErr_NewException("smashtable.PhantomConflictError", phantom_conflict_bases, nullptr);
    Py_DECREF(phantom_conflict_bases);
    if (!state->phantom_conflict_error) return -1;

    PyObject *duplicate_bases = PyTuple_Pack(2, state->error, PyExc_KeyError);
    if (!duplicate_bases) return -1;
    state->duplicate_key_error = PyErr_NewException("smashtable.DuplicateKeyError", duplicate_bases, nullptr);
    Py_DECREF(duplicate_bases);
    if (!state->duplicate_key_error) return -1;

    PyObject *state_bases = PyTuple_Pack(2, state->error, PyExc_RuntimeError);
    if (!state_bases) return -1;
    state->state_error = PyErr_NewException("smashtable.StateError", state_bases, nullptr);
    Py_DECREF(state_bases);
    if (!state->state_error) return -1;

    state->sorted_map_type =
        reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &sorted_map_spec, nullptr));
    if (!state->sorted_map_type) return -1;
    state->sorted_set_type =
        reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &sorted_set_spec, nullptr));
    if (!state->sorted_set_type) return -1;
    state->hash_map_type = reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &hash_map_spec, nullptr));
    if (!state->hash_map_type) return -1;
    state->hash_set_type = reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &hash_set_spec, nullptr));
    if (!state->hash_set_type) return -1;
    state->transaction_type =
        reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &transaction_spec, nullptr));
    if (!state->transaction_type) return -1;
    state->view_type = reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &view_spec, nullptr));
    if (!state->view_type) return -1;
    state->cursor_type = reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &cursor_spec, nullptr));
    if (!state->cursor_type) return -1;
    state->keys_view_type =
        reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &keys_view_spec, nullptr));
    if (!state->keys_view_type) return -1;
    state->values_view_type =
        reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &values_view_spec, nullptr));
    if (!state->values_view_type) return -1;
    state->items_view_type =
        reinterpret_cast<PyTypeObject *>(PyType_FromModuleAndSpec(module, &items_view_spec, nullptr));
    if (!state->items_view_type) return -1;

    if (add_transaction_method(state->sorted_map_type) != 0) return -1;
    if (add_transaction_method(state->sorted_set_type) != 0) return -1;
    if (add_transaction_method(state->hash_map_type) != 0) return -1;
    if (add_transaction_method(state->hash_set_type) != 0) return -1;

    if (PyModule_AddObjectRef(module, "SmashTableError", state->error) < 0) return -1;
    if (PyModule_AddObjectRef(module, "ConflictError", state->conflict_error) < 0) return -1;
    if (PyModule_AddObjectRef(module, "WriteConflictError", state->write_conflict_error) < 0) return -1;
    if (PyModule_AddObjectRef(module, "ReadConflictError", state->read_conflict_error) < 0) return -1;
    if (PyModule_AddObjectRef(module, "PhantomConflictError", state->phantom_conflict_error) < 0) return -1;
    if (PyModule_AddObjectRef(module, "DuplicateKeyError", state->duplicate_key_error) < 0) return -1;
    if (PyModule_AddObjectRef(module, "StateError", state->state_error) < 0) return -1;
    if (PyModule_AddObjectRef(module, "SortedMap", reinterpret_cast<PyObject *>(state->sorted_map_type)) < 0) return -1;
    if (PyModule_AddObjectRef(module, "SortedSet", reinterpret_cast<PyObject *>(state->sorted_set_type)) < 0) return -1;
    if (PyModule_AddObjectRef(module, "HashMap", reinterpret_cast<PyObject *>(state->hash_map_type)) < 0) return -1;
    if (PyModule_AddObjectRef(module, "HashSet", reinterpret_cast<PyObject *>(state->hash_set_type)) < 0) return -1;
    // Exported so a caller can annotate and test against what the API hands back. Both refuse
    // instantiation, so naming them grants no way to build one.
    if (PyModule_AddObjectRef(module, "Transaction", reinterpret_cast<PyObject *>(state->transaction_type)) < 0)
        return -1;
    if (PyModule_AddObjectRef(module, "Participant", reinterpret_cast<PyObject *>(state->view_type)) < 0) return -1;
    if (PyModule_AddStringConstant(module, "__version__", "0.2.0") < 0) return -1;
    return 0;
}

static int module_traverse(PyObject *module, visitproc visit, void *arg) noexcept {
    module_state_t *state = state_of(module);
    Py_VISIT(state->sorted_map_type);
    Py_VISIT(state->sorted_set_type);
    Py_VISIT(state->hash_map_type);
    Py_VISIT(state->hash_set_type);
    Py_VISIT(state->transaction_type);
    Py_VISIT(state->view_type);
    Py_VISIT(state->cursor_type);
    Py_VISIT(state->keys_view_type);
    Py_VISIT(state->values_view_type);
    Py_VISIT(state->items_view_type);
    Py_VISIT(state->error);
    Py_VISIT(state->conflict_error);
    Py_VISIT(state->write_conflict_error);
    Py_VISIT(state->read_conflict_error);
    Py_VISIT(state->phantom_conflict_error);
    Py_VISIT(state->duplicate_key_error);
    Py_VISIT(state->state_error);
    return 0;
}

static int module_clear(PyObject *module) noexcept {
    module_state_t *state = state_of(module);
    Py_CLEAR(state->sorted_map_type);
    Py_CLEAR(state->sorted_set_type);
    Py_CLEAR(state->hash_map_type);
    Py_CLEAR(state->hash_set_type);
    Py_CLEAR(state->transaction_type);
    Py_CLEAR(state->view_type);
    Py_CLEAR(state->cursor_type);
    Py_CLEAR(state->keys_view_type);
    Py_CLEAR(state->values_view_type);
    Py_CLEAR(state->items_view_type);
    Py_CLEAR(state->error);
    Py_CLEAR(state->conflict_error);
    Py_CLEAR(state->write_conflict_error);
    Py_CLEAR(state->read_conflict_error);
    Py_CLEAR(state->phantom_conflict_error);
    Py_CLEAR(state->duplicate_key_error);
    Py_CLEAR(state->state_error);
    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, reinterpret_cast<void *>(module_exec)},
    {Py_mod_multiple_interpreters, Py_MOD_PER_INTERPRETER_GIL_SUPPORTED},
#if defined(Py_mod_gil) // ? Only 3.13 and later know about the GIL slot
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, nullptr},
};

static struct PyModuleDef smashtable_module = {
    PyModuleDef_HEAD_INIT,  "smashtable",   "Safer associative stores with DBMS-like transactions in Python.",
    sizeof(module_state_t), module_methods, module_slots,
    module_traverse,        module_clear,   nullptr,
};

PyModuleDef *smashtable_module_def() noexcept { return &smashtable_module; }

#pragma endregion Initialization

} // namespace ashvardanian::smashtable::py

extern "C" PyMODINIT_FUNC PyInit_smashtable(void) {
    return PyModuleDef_Init(&ashvardanian::smashtable::py::smashtable_module);
}
