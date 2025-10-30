/**
 *  @file       smashtable.cpp
 *  @brief      Pure CPython wrapper for SmashTable transactional collections.
 *  @author     Ash Vardanian
 *  @date       October 2025
 *
 *  Thread-safe collections for Python 3.14+ with free-threading support.
 *
 *  @par Classes
 *
 *  Traditional collections designed for exception-free comparable/hashable types
 *  - @c SortedSet: ordered `set`-like atomic collection
 *  - @c SortedMap: ordered `dict`-like atomic key-value store
 *  - @c HashSet: unordered `set`-like atomic collection
 *  - @c HashMap: unordered `dict`-like atomic key-value store
 *  Sorted variants not only provide fast sorted iterators, but also rank order-statistics
 *  operations so popular in databases.
 *
 *  Transactional and thread-safe collections for free-threaded Python applications:
 *  - @c TransactionalSortedSet: ordered `set`-like collection with 2-phase commit
 *  - @c TransactionalSortedMap: ordered `dict`-like collection with 2-phase commit
 *  - @c TransactionalHashSet: unordered `set`-like collection with 2-phase commit
 *  - @c TransactionalHashMap: unordered `set`-like collection with 2-phase commit
 *
 *  Features:
 *  - GIL-free operation: all operations release the GIL for true parallelism
 *  - Subinterpreter-safe: no global state, fully isolated per-instance
 *  - Heterogeneous types: int, float, str, bytes as keys and values
 *  - MVCC transactions: Optimistic concurrency with watch/stage/commit
 *
 *  Both use std::variant<string, int64_t, uint64_t, double> internally.
 *  Type ordering: string < int64_t < uint64_t < double
 */

#include <Python.h>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <cstring>

#include <smashtable/transactional_binary_tree.hpp>
#include <smashtable/partitioned_collection.hpp>

using namespace ashvardanian::smashtable;

// Variant type supporting multiple key/value types with .copy() support
struct variant_t {
    std::variant<std::string, std::int64_t, std::uint64_t, double> value;

    variant_t() = default;
    variant_t(variant_t const &) = default;
    variant_t(variant_t &&) noexcept = default;
    variant_t &operator=(variant_t const &) = default;
    variant_t &operator=(variant_t &&) noexcept = default;

    // Constructors from underlying types
    variant_t(std::string const &v) : value(v) {}
    variant_t(std::string &&v) : value(std::move(v)) {}
    variant_t(std::int64_t v) : value(v) {}
    variant_t(std::uint64_t v) : value(v) {}
    variant_t(double v) : value(v) {}

    // Assignment from underlying types
    variant_t &operator=(std::string const &v) {
        value = v;
        return *this;
    }
    variant_t &operator=(std::string &&v) {
        value = std::move(v);
        return *this;
    }
    variant_t &operator=(std::int64_t v) {
        value = v;
        return *this;
    }
    variant_t &operator=(std::uint64_t v) {
        value = v;
        return *this;
    }
    variant_t &operator=(double v) {
        value = v;
        return *this;
    }

    // Provide copy() method for watch support
    [[nodiscard]] std::optional<variant_t> copy() const noexcept {
        try {
            return variant_t {*this};
        }
        catch (...) {
            return std::nullopt;
        }
    }

    // Access underlying variant
    auto const &get() const noexcept { return value; }
    auto &get() noexcept { return value; }
    auto index() const noexcept { return value.index(); }
};

/**
 *  @brief Transparent comparator for heterogeneous variant keys
 *
 *  Ordering: string < int64_t < uint64_t < double
 *  Within same type: natural ordering
 */
struct variant_compare_t {
    using is_transparent = void;
    using value_type = variant_t;

    static bool compare_same_type(variant_t const &a, variant_t const &b) noexcept {
        return std::visit(
            [](auto &&arg1, auto &&arg2) -> bool {
                using arg1_t = std::decay_t<decltype(arg1)>;
                using arg2_t = std::decay_t<decltype(arg2)>;
                if constexpr (std::is_same_v<arg1_t, arg2_t>) { return arg1 < arg2; }
                else { return false; } // Should not happen in same_type comparison
            },
            a.get(), b.get());
    }

    bool operator()(variant_t const &a, variant_t const &b) const noexcept {
        size_t type_index_a = a.index();
        size_t type_index_b = b.index();
        if (type_index_a != type_index_b) return type_index_a < type_index_b;
        return compare_same_type(a, b);
    }
};

// Type aliases for Map and Set
// Note: Both use map type (mapping<key, value>). For Set, we use a dummy value.
// Now using transactional_wb_map with .copy() support for watches
using tree_t = transactional_wb_map<variant_t, variant_t, variant_compare_t>;
using partitioned_t = partitioned_collection<tree_t>;
using transaction_t = typename partitioned_t::transaction_t;

#pragma region Python Variant Conversion

/**
 *  @brief Convert Python object to variant
 *  Returns false on error (sets Python exception)
 */
static bool py_to_variant(PyObject *obj, variant_t &out) {
    // String (unicode)
    if (PyUnicode_Check(obj)) {
        Py_ssize_t size;
        const char *data = PyUnicode_AsUTF8AndSize(obj, &size);
        if (!data) return false;
        out = std::string(data, size);
        return true;
    }

    // Bytes
    if (PyBytes_Check(obj)) {
        char *data;
        Py_ssize_t size;
        if (PyBytes_AsStringAndSize(obj, &data, &size) < 0) return false;
        out = std::string(data, size);
        return true;
    }

    // Float
    if (PyFloat_Check(obj)) {
        double value = PyFloat_AsDouble(obj);
        if (value == -1.0 && PyErr_Occurred()) return false;
        out = value;
        return true;
    }

    // Int (check sign to decide int64 vs uint64)
    if (PyLong_Check(obj)) {
        // Try as int64 first
        long long value = PyLong_AsLongLong(obj);
        if (value == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            // Try as uint64
            unsigned long long uvalue = PyLong_AsUnsignedLongLong(obj);
            if (uvalue == (unsigned long long)-1 && PyErr_Occurred()) { return false; }
            out = static_cast<std::uint64_t>(uvalue);
        }
        else { out = static_cast<std::int64_t>(value); }
        return true;
    }

    PyErr_Format(PyExc_TypeError, "Unsupported type: %s", Py_TYPE(obj)->tp_name);
    return false;
}

/**
 *  @brief Convert variant to Python object
 *  Returns new reference (or NULL on error)
 */
static PyObject *variant_to_py(variant_t const &v) {
    return std::visit(
        [](auto &&arg) -> PyObject * {
            using T = std::decay_t<decltype(arg)>;

            if constexpr (std::is_same_v<T, std::string>) {
                return PyUnicode_FromStringAndSize(arg.data(), arg.size());
            }
            else if constexpr (std::is_same_v<T, std::int64_t>) { return PyLong_FromLongLong(arg); }
            else if constexpr (std::is_same_v<T, std::uint64_t>) { return PyLong_FromUnsignedLongLong(arg); }
            else if constexpr (std::is_same_v<T, double>) { return PyFloat_FromDouble(arg); }
            else {
                PyErr_SetString(PyExc_RuntimeError, "Unknown variant type");
                return nullptr;
            }
        },
        v.get());
}

#pragma endregion

#pragma region Forward Declarations

static PyTypeObject MapType;
static PyTypeObject SetType;
static PyTypeObject MapTransactionType;
static PyTypeObject SetTransactionType;

#pragma endregion

#pragma region Map Type

typedef struct {
    PyObject_HEAD partitioned_t *tree;
} Map;

static void Map_dealloc(Map *self) {
    if (self->tree) {
        delete self->tree;
        self->tree = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Map_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Map *self = (Map *)type->tp_alloc(type, 0);
    if (self != NULL) {
        Py_BEGIN_ALLOW_THREADS;
        auto maybe_tree = partitioned_t::make();
        Py_END_ALLOW_THREADS;

        if (maybe_tree.has_value()) { self->tree = new partitioned_t(std::move(*maybe_tree)); }
        else {
            Py_DECREF(self);
            PyErr_SetString(PyExc_MemoryError, "Failed to create Map");
            return NULL;
        }
    }
    return (PyObject *)self;
}

static int Map_init(Map *self, PyObject *args, PyObject *kwds) { return 0; }

/**
 *  @brief size() -> int
 */
static PyObject *Map_size(Map *self, PyObject *Py_UNUSED(ignored)) {
    size_t size;
    Py_BEGIN_ALLOW_THREADS;
    size = self->tree->size();
    Py_END_ALLOW_THREADS;
    return PyLong_FromSize_t(size);
}

/**
 *  @brief empty() -> bool
 */
static PyObject *Map_empty(Map *self, PyObject *Py_UNUSED(ignored)) {
    bool is_empty;
    Py_BEGIN_ALLOW_THREADS;
    is_empty = self->tree->empty();
    Py_END_ALLOW_THREADS;
    return PyBool_FromLong(is_empty);
}

/**
 *  @brief upsert(key, value) -> bool
 *         upsert({k1: v1, k2: v2}) -> int
 */
static PyObject *Map_upsert(Map *self, PyObject *const *args, Py_ssize_t nargs) {
    // Batch: upsert({k1: v1, k2: v2})
    if (nargs == 1 && PyDict_Check(args[0])) {
        PyObject *dict = args[0];
        PyObject *key_obj, *value_obj;
        Py_ssize_t pos = 0;
        size_t success_count = 0;

        while (PyDict_Next(dict, &pos, &key_obj, &value_obj)) {
            variant_t key, value;
            if (!py_to_variant(key_obj, key)) return NULL;
            if (!py_to_variant(value_obj, value)) return NULL;

            bool success;
            Py_BEGIN_ALLOW_THREADS;
            mapping<variant_t, variant_t> kv {std::move(key), std::move(value)};
            auto status = self->tree->upsert(std::move(kv));
            success = status.errc == errc_t::success_k;
            Py_END_ALLOW_THREADS;

            if (success) success_count++;
        }

        return PyLong_FromSize_t(success_count);
    }

    // Single: upsert(key, value)
    if (nargs == 2) {
        variant_t key, value;
        if (!py_to_variant(args[0], key)) return NULL;
        if (!py_to_variant(args[1], value)) return NULL;

        bool success;
        Py_BEGIN_ALLOW_THREADS;
        mapping<variant_t, variant_t> kv {std::move(key), std::move(value)};
        auto status = self->tree->upsert(std::move(kv));
        success = status.errc == errc_t::success_k;
        Py_END_ALLOW_THREADS;

        return PyBool_FromLong(success);
    }

    PyErr_SetString(PyExc_TypeError, "upsert() takes either (key, value) or a dict");
    return NULL;
}

/**
 *  @brief find(key) -> value | None
 *         find([k1, k2]) -> {k1: v1, k2: v2}
 */
static PyObject *Map_find(Map *self, PyObject *arg) {
    // Batch: find([k1, k2]) or find((k1, k2))
    if (PyList_Check(arg) || PyTuple_Check(arg)) {
        PyObject *result = PyDict_New();
        if (!result) return NULL;

        Py_ssize_t size = PySequence_Size(arg);
        if (size == -1) {
            Py_DECREF(result);
            return NULL;
        }

        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject *key_obj = PySequence_GetItem(arg, i);
            if (!key_obj) {
                Py_DECREF(result);
                return NULL;
            }

            variant_t key;
            if (!py_to_variant(key_obj, key)) {
                Py_DECREF(key_obj);
                Py_DECREF(result);
                return NULL;
            }

            variant_t result_value;
            bool found = false;

            Py_BEGIN_ALLOW_THREADS;
            self->tree->find(key, [&](auto const &entry) noexcept {
                result_value = entry.element.value;
                found = true;
            });
            Py_END_ALLOW_THREADS;

            if (found) {
                PyObject *value_obj = variant_to_py(result_value);
                if (!value_obj) {
                    Py_DECREF(key_obj);
                    Py_DECREF(result);
                    return NULL;
                }
                if (PyDict_SetItem(result, key_obj, value_obj) < 0) {
                    Py_DECREF(value_obj);
                    Py_DECREF(key_obj);
                    Py_DECREF(result);
                    return NULL;
                }
                Py_DECREF(value_obj);
            }

            Py_DECREF(key_obj);
        }

        return result;
    }

    // Single: find(key)
    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    variant_t result_value;
    bool found = false;

    Py_BEGIN_ALLOW_THREADS;
    self->tree->find(key, [&](auto const &entry) noexcept {
        result_value = entry.element.value;
        found = true;
    });
    Py_END_ALLOW_THREADS;

    if (found) { return variant_to_py(result_value); }
    else { Py_RETURN_NONE; }
}

/**
 *  @brief erase(key) -> bool
 */
static PyObject *Map_erase(Map *self, PyObject *arg) {
    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    bool success;
    Py_BEGIN_ALLOW_THREADS;
    auto status = self->tree->erase(key);
    success = status.errc == errc_t::success_k;
    Py_END_ALLOW_THREADS;

    return PyBool_FromLong(success);
}

/**
 *  @brief clear() -> None
 */
static PyObject *Map_clear(Map *self, PyObject *Py_UNUSED(ignored)) {
    Py_BEGIN_ALLOW_THREADS;
    self->tree->clear();
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

/**
 *  @brief transaction() -> MapTransaction
 */
static PyObject *Map_transaction(Map *self, PyObject *Py_UNUSED(ignored));

static PyMethodDef Map_methods[] = {
    {"size", (PyCFunction)Map_size, METH_NOARGS, "Return number of elements"},
    {"empty", (PyCFunction)Map_empty, METH_NOARGS, "Check if empty"},
    {"upsert", (PyCFunction)Map_upsert, METH_FASTCALL, "Insert/update: (key, value) or {dict}"},
    {"find", (PyCFunction)Map_find, METH_O, "Find: key or [keys] -> value/None or {dict}"},
    {"erase", (PyCFunction)Map_erase, METH_O, "Remove key"},
    {"clear", (PyCFunction)Map_clear, METH_NOARGS, "Remove all entries"},
    {"transaction", (PyCFunction)Map_transaction, METH_NOARGS, "Begin transaction"},
    {NULL, NULL, 0, NULL}};

static PyTypeObject MapType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "smashtable.Map",
    .tp_doc = "Thread-safe partitioned AVL tree with heterogeneous keys/values",
    .tp_basicsize = sizeof(Map),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Map_new,
    .tp_init = (initproc)Map_init,
    .tp_dealloc = (destructor)Map_dealloc,
    .tp_methods = Map_methods,
};

#pragma endregion

#pragma region MapTransaction Type

typedef struct {
    PyObject_HEAD Map *parent;
    transaction_t *txn;
    bool is_active;
} MapTransaction;

static void MapTransaction_dealloc(MapTransaction *self) {
    if (self->txn) {
        delete self->txn;
        self->txn = nullptr;
    }
    Py_XDECREF(self->parent);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/**
 *  @brief upsert(key, value) -> bool
 */
static PyObject *MapTransaction_upsert(MapTransaction *self, PyObject *const *args, Py_ssize_t nargs) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "MapTransaction no longer active");
        return NULL;
    }

    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "upsert() takes exactly 2 arguments");
        return NULL;
    }

    variant_t key, value;
    if (!py_to_variant(args[0], key)) return NULL;
    if (!py_to_variant(args[1], value)) return NULL;

    bool success;
    Py_BEGIN_ALLOW_THREADS;
    mapping<variant_t, variant_t> kv {std::move(key), std::move(value)};
    auto status = self->txn->upsert(std::move(kv));
    success = status.errc == errc_t::success_k;
    Py_END_ALLOW_THREADS;

    return PyBool_FromLong(success);
}

/**
 *  @brief erase(key) -> bool
 */
static PyObject *MapTransaction_erase(MapTransaction *self, PyObject *arg) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "MapTransaction no longer active");
        return NULL;
    }

    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    bool success;
    Py_BEGIN_ALLOW_THREADS;
    ;
    auto status = self->txn->erase(key);
    success = status.errc == errc_t::success_k;
    Py_END_ALLOW_THREADS;
    ;

    return PyBool_FromLong(success);
}

/**
 *  @brief watch(key) -> bool
 */
static PyObject *MapTransaction_watch(MapTransaction *self, PyObject *arg) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "MapTransaction no longer active");
        return NULL;
    }

    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    bool success;
    Py_BEGIN_ALLOW_THREADS;
    ;
    auto status = self->txn->watch(key);
    success = status.errc == errc_t::success_k;
    Py_END_ALLOW_THREADS;
    ;

    return PyBool_FromLong(success);
}

/**
 *  @brief stage() -> None (raises on conflict)
 */
static PyObject *MapTransaction_stage(MapTransaction *self, PyObject *Py_UNUSED(ignored)) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "MapTransaction no longer active");
        return NULL;
    }

    errc_t error_code;
    Py_BEGIN_ALLOW_THREADS;
    auto status = self->txn->stage();
    error_code = status.errc;
    Py_END_ALLOW_THREADS;

    if (error_code != errc_t::success_k) {
        if (error_code == errc_t::consistency_k) {
            PyErr_SetString(PyExc_RuntimeError, "consistency violation: watched key modified");
        }
        else { PyErr_Format(PyExc_RuntimeError, "stage failed: error %d", error_code); }
        return NULL;
    }

    Py_RETURN_NONE;
}

/**
 *  @brief commit() -> None
 */
static PyObject *MapTransaction_commit(MapTransaction *self, PyObject *Py_UNUSED(ignored)) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "MapTransaction no longer active");
        return NULL;
    }

    errc_t error_code;
    Py_BEGIN_ALLOW_THREADS;
    auto status = self->txn->commit();
    error_code = status.errc;
    Py_END_ALLOW_THREADS;

    if (error_code != errc_t::success_k) {
        PyErr_Format(PyExc_RuntimeError, "commit failed: error %d", error_code);
        return NULL;
    }

    Py_RETURN_NONE;
}

/**
 *  @brief rollback() -> None
 */
static PyObject *MapTransaction_rollback(MapTransaction *self, PyObject *Py_UNUSED(ignored)) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "MapTransaction no longer active");
        return NULL;
    }

    errc_t error_code;
    Py_BEGIN_ALLOW_THREADS;
    auto status = self->txn->rollback();
    error_code = status.errc;
    Py_END_ALLOW_THREADS;

    if (error_code != errc_t::success_k) {
        PyErr_Format(PyExc_RuntimeError, "rollback failed: error %d", error_code);
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyMethodDef MapTransaction_methods[] = {
    {"upsert", (PyCFunction)MapTransaction_upsert, METH_FASTCALL, "Stage upsert"},
    {"erase", (PyCFunction)MapTransaction_erase, METH_O, "Stage erase"},
    {"watch", (PyCFunction)MapTransaction_watch, METH_O, "Watch key for conflicts"},
    {"stage", (PyCFunction)MapTransaction_stage, METH_NOARGS, "Validate and stage"},
    {"commit", (PyCFunction)MapTransaction_commit, METH_NOARGS, "Commit staged changes"},
    {"rollback", (PyCFunction)MapTransaction_rollback, METH_NOARGS, "Rollback staged changes"},
    {NULL, NULL, 0, NULL}};

static PyTypeObject MapTransactionType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "smashtable.MapTransaction",
    .tp_doc = "MapTransaction handle for atomic multi-key operations",
    .tp_basicsize = sizeof(MapTransaction),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)MapTransaction_dealloc,
    .tp_methods = MapTransaction_methods,
};

static PyObject *Map_transaction(Map *self, PyObject *Py_UNUSED(ignored)) {
    MapTransaction *txn_obj = (MapTransaction *)MapTransactionType.tp_alloc(&MapTransactionType, 0);
    if (txn_obj == NULL) return NULL;

    std::optional<transaction_t> maybe_txn;
    Py_BEGIN_ALLOW_THREADS;
    maybe_txn = self->tree->transaction();
    Py_END_ALLOW_THREADS;

    if (!maybe_txn.has_value()) {
        Py_DECREF(txn_obj);
        PyErr_SetString(PyExc_MemoryError, "Failed to create transaction");
        return NULL;
    }

    txn_obj->txn = new transaction_t(std::move(*maybe_txn));
    txn_obj->parent = self;
    txn_obj->is_active = true;
    Py_INCREF(self);

    return (PyObject *)txn_obj;
}

#pragma endregion

#pragma region Set Type

// Set uses the same backend as Map but with a dummy value
static const variant_t SET_DUMMY_VALUE = static_cast<std::int64_t>(1);

typedef struct {
    PyObject_HEAD partitioned_t *tree;
} Set;

static void Set_dealloc(Set *self) {
    if (self->tree) {
        delete self->tree;
        self->tree = nullptr;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Set_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Set *self = (Set *)type->tp_alloc(type, 0);
    if (self != NULL) {
        Py_BEGIN_ALLOW_THREADS;
        auto maybe_tree = partitioned_t::make();
        Py_END_ALLOW_THREADS;

        if (maybe_tree.has_value()) { self->tree = new partitioned_t(std::move(*maybe_tree)); }
        else {
            Py_DECREF(self);
            PyErr_SetString(PyExc_MemoryError, "Failed to create Set");
            return NULL;
        }
    }
    return (PyObject *)self;
}

static int Set_init(Set *self, PyObject *args, PyObject *kwds) { return 0; }

/**
 *  @brief size() -> int
 */
static PyObject *Set_size(Set *self, PyObject *Py_UNUSED(ignored)) {
    size_t size;
    Py_BEGIN_ALLOW_THREADS;
    size = self->tree->size();
    Py_END_ALLOW_THREADS;
    return PyLong_FromSize_t(size);
}

/**
 *  @brief empty() -> bool
 */
static PyObject *Set_empty(Set *self, PyObject *Py_UNUSED(ignored)) {
    bool is_empty;
    Py_BEGIN_ALLOW_THREADS;
    is_empty = self->tree->empty();
    Py_END_ALLOW_THREADS;
    return PyBool_FromLong(is_empty);
}

/**
 *  @brief add(key) -> bool
 *         add([keys]) -> int
 */
static PyObject *Set_add(Set *self, PyObject *arg) {
    // Batch: add([key1, key2, ...])
    if (PyList_Check(arg) || PyTuple_Check(arg)) {
        PyObject *seq = PySequence_Fast(arg, "Expected list or tuple");
        if (!seq) return NULL;

        Py_ssize_t size = PySequence_Fast_GET_SIZE(seq);
        size_t success_count = 0;

        for (Py_ssize_t i = 0; i < size; i++) {
            PyObject *key_obj = PySequence_Fast_GET_ITEM(seq, i);
            variant_t key;
            if (!py_to_variant(key_obj, key)) {
                Py_DECREF(seq);
                return NULL;
            }

            bool success;
            Py_BEGIN_ALLOW_THREADS;
            mapping<variant_t, variant_t> kv {std::move(key), SET_DUMMY_VALUE};
            auto status = self->tree->upsert(std::move(kv));
            success = status.errc == errc_t::success_k;
            Py_END_ALLOW_THREADS;

            if (success) success_count++;
        }

        Py_DECREF(seq);
        return PyLong_FromSize_t(success_count);
    }

    // Single: add(key)
    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    bool success;
    Py_BEGIN_ALLOW_THREADS;
    mapping<variant_t, variant_t> kv {std::move(key), SET_DUMMY_VALUE};
    auto status = self->tree->upsert(std::move(kv));
    success = status.errc == errc_t::success_k;
    Py_END_ALLOW_THREADS;

    return PyBool_FromLong(success);
}

/**
 *  @brief contains(key) -> bool  (also __contains__ for 'in' operator)
 */
static PyObject *Set_contains(Set *self, PyObject *arg) {
    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    bool found = false;
    Py_BEGIN_ALLOW_THREADS;
    auto status = self->tree->find(key, [&](auto const &) noexcept { found = true; });
    Py_END_ALLOW_THREADS;

    return PyBool_FromLong(found && status.errc == errc_t::success_k);
}

/**
 *  @brief remove(key) -> bool
 */
static PyObject *Set_remove(Set *self, PyObject *arg) {
    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    bool success;
    Py_BEGIN_ALLOW_THREADS;
    auto status = self->tree->erase(key);
    success = status.errc == errc_t::success_k;
    Py_END_ALLOW_THREADS;

    return PyBool_FromLong(success);
}

/**
 *  @brief discard(key) -> None  (like remove but doesn't raise if missing)
 */
static PyObject *Set_discard(Set *self, PyObject *arg) {
    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    Py_BEGIN_ALLOW_THREADS;
    self->tree->erase(key);
    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

/**
 *  @brief clear() -> None
 */
static PyObject *Set_clear(Set *self, PyObject *Py_UNUSED(ignored)) {
    Py_BEGIN_ALLOW_THREADS;
    self->tree->clear();
    Py_END_ALLOW_THREADS;
    Py_RETURN_NONE;
}

/**
 *  @brief __len__() -> int
 */
static Py_ssize_t Set_len(Set *self) {
    size_t size;
    Py_BEGIN_ALLOW_THREADS;
    size = self->tree->size();
    Py_END_ALLOW_THREADS;
    return (Py_ssize_t)size;
}

/**
 *  @brief __contains__(key) -> bool  (for 'in' operator)
 */
static int Set_sq_contains(Set *self, PyObject *key) {
    variant_t key_var;
    if (!py_to_variant(key, key_var)) return -1;

    bool found = false;
    Py_BEGIN_ALLOW_THREADS;
    auto status = self->tree->find(key_var, [&](key_value_t const &) noexcept { found = true; });
    Py_END_ALLOW_THREADS;

    return (found && status.errc == errc_t::success_k) ? 1 : 0;
}

/**
 *  @brief transaction() -> SetTransaction
 */
static PyObject *Set_transaction(Set *self, PyObject *Py_UNUSED(ignored));

static PyMethodDef Set_methods[] = {
    {"size", (PyCFunction)Set_size, METH_NOARGS, "Return number of elements"},
    {"empty", (PyCFunction)Set_empty, METH_NOARGS, "Check if empty"},
    {"add", (PyCFunction)Set_add, METH_O, "Add key (single or list)"},
    {"contains", (PyCFunction)Set_contains, METH_O, "Check if key exists"},
    {"remove", (PyCFunction)Set_remove, METH_O, "Remove key (returns False if missing)"},
    {"discard", (PyCFunction)Set_discard, METH_O, "Remove key (no error if missing)"},
    {"clear", (PyCFunction)Set_clear, METH_NOARGS, "Remove all entries"},
    {"transaction", (PyCFunction)Set_transaction, METH_NOARGS, "Begin transaction"},
    {NULL, NULL, 0, NULL}};

static PySequenceMethods Set_as_sequence = {
    .sq_length = (lenfunc)Set_len,
    .sq_contains = (objobjproc)Set_sq_contains,
};

static PyTypeObject SetType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "smashtable.Set",
    .tp_doc = "Thread-safe set with heterogeneous key types (str, int, float, bytes)",
    .tp_basicsize = sizeof(Set),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Set_new,
    .tp_init = (initproc)Set_init,
    .tp_dealloc = (destructor)Set_dealloc,
    .tp_methods = Set_methods,
    .tp_as_sequence = &Set_as_sequence,
};

#pragma endregion

#pragma region SetTransaction Type

typedef struct {
    PyObject_HEAD Set *parent;
    transaction_t *txn;
    bool is_active;
} SetTransaction;

static void SetTransaction_dealloc(SetTransaction *self) {
    if (self->txn) {
        delete self->txn;
        self->txn = nullptr;
    }
    Py_XDECREF(self->parent);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/**
 *  @brief add(key) -> bool
 */
static PyObject *SetTransaction_add(SetTransaction *self, PyObject *arg) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "Transaction no longer active");
        return NULL;
    }

    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    bool success;
    Py_BEGIN_ALLOW_THREADS;
    mapping<variant_t, variant_t> kv {std::move(key), SET_DUMMY_VALUE};
    auto status = self->txn->upsert(std::move(kv));
    success = status.errc == errc_t::success_k;
    Py_END_ALLOW_THREADS;

    return PyBool_FromLong(success);
}

/**
 *  @brief remove(key) -> bool
 */
static PyObject *SetTransaction_remove(SetTransaction *self, PyObject *arg) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "Transaction no longer active");
        return NULL;
    }

    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    bool success;
    Py_BEGIN_ALLOW_THREADS;
    auto status = self->txn->erase(key);
    success = status.errc == errc_t::success_k;
    Py_END_ALLOW_THREADS;

    return PyBool_FromLong(success);
}

/**
 *  @brief watch(key) -> None
 */
static PyObject *SetTransaction_watch(SetTransaction *self, PyObject *arg) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "Transaction no longer active");
        return NULL;
    }

    variant_t key;
    if (!py_to_variant(arg, key)) return NULL;

    Py_BEGIN_ALLOW_THREADS;
    self->txn->watch(key);
    Py_END_ALLOW_THREADS;

    Py_RETURN_NONE;
}

/**
 *  @brief stage() -> None (raises RuntimeError on conflict)
 */
static PyObject *SetTransaction_stage(SetTransaction *self, PyObject *Py_UNUSED(ignored)) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "Transaction no longer active");
        return NULL;
    }

    bool success;
    Py_BEGIN_ALLOW_THREADS;
    auto status = self->txn->stage();
    success = status.errc == errc_t::success_k;
    Py_END_ALLOW_THREADS;

    if (!success) {
        PyErr_SetString(PyExc_RuntimeError, "Transaction stage failed: consistency violation");
        return NULL;
    }

    Py_RETURN_NONE;
}

/**
 *  @brief commit() -> None
 */
static PyObject *SetTransaction_commit(SetTransaction *self, PyObject *Py_UNUSED(ignored)) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "Transaction no longer active");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    self->txn->commit();
    Py_END_ALLOW_THREADS;

    self->is_active = false;
    Py_RETURN_NONE;
}

/**
 *  @brief rollback() -> None
 */
static PyObject *SetTransaction_rollback(SetTransaction *self, PyObject *Py_UNUSED(ignored)) {
    if (!self->is_active) {
        PyErr_SetString(PyExc_RuntimeError, "Transaction no longer active");
        return NULL;
    }

    Py_BEGIN_ALLOW_THREADS;
    self->txn->rollback();
    Py_END_ALLOW_THREADS;

    self->is_active = false;
    Py_RETURN_NONE;
}

static PyMethodDef SetTransaction_methods[] = {
    {"add", (PyCFunction)SetTransaction_add, METH_O, "Add key to transaction"},
    {"remove", (PyCFunction)SetTransaction_remove, METH_O, "Remove key in transaction"},
    {"watch", (PyCFunction)SetTransaction_watch, METH_O, "Watch key for changes"},
    {"stage", (PyCFunction)SetTransaction_stage, METH_NOARGS, "Stage transaction (validate watches)"},
    {"commit", (PyCFunction)SetTransaction_commit, METH_NOARGS, "Commit transaction"},
    {"rollback", (PyCFunction)SetTransaction_rollback, METH_NOARGS, "Rollback transaction"},
    {NULL, NULL, 0, NULL}};

static PyTypeObject SetTransactionType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "smashtable.SetTransaction",
    .tp_doc = "SetTransaction handle for atomic multi-key operations on Set",
    .tp_basicsize = sizeof(SetTransaction),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_dealloc = (destructor)SetTransaction_dealloc,
    .tp_methods = SetTransaction_methods,
};

static PyObject *Set_transaction(Set *self, PyObject *Py_UNUSED(ignored)) {
    SetTransaction *txn_obj = (SetTransaction *)SetTransactionType.tp_alloc(&SetTransactionType, 0);
    if (txn_obj == NULL) return NULL;

    std::optional<transaction_t> maybe_txn;
    Py_BEGIN_ALLOW_THREADS;
    maybe_txn = self->tree->transaction();
    Py_END_ALLOW_THREADS;

    if (!maybe_txn.has_value()) {
        Py_DECREF(txn_obj);
        PyErr_SetString(PyExc_MemoryError, "Failed to create transaction");
        return NULL;
    }

    txn_obj->txn = new transaction_t(std::move(*maybe_txn));
    txn_obj->parent = self;
    txn_obj->is_active = true;
    Py_INCREF(self);

    return (PyObject *)txn_obj;
}

#pragma endregion

#pragma region Module Definition

static PyModuleDef smashtable_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "smashtable",
    .m_doc = "Thread-safe Map and Set for Python 3.14+ with free-threading and subinterpreter support.\n\n"
             "Features:\n"
             "  - GIL-free: All operations release GIL for true parallelism\n"
             "  - Subinterpreter-safe: No global state, isolated per-instance\n"
             "  - Heterogeneous types: int, float, str, bytes\n"
             "  - MVCC transactions: Optimistic concurrency control\n\n"
             "Classes:\n"
             "  - Map: Key-value store with batch operations\n"
             "  - Set: Key-only collection for membership testing\n\n"
             "Free-threading:\n"
             "  Build Python with --disable-gil or use python3.14t to enable true parallelism.\n"
             "  This module is fully thread-safe and releases the GIL during all operations.\n\n"
             "Subinterpreters:\n"
             "  Each interpreter can safely create its own Map/Set instances.\n"
             "  Use InterpreterPoolExecutor from concurrent.futures for parallel execution.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_smashtable(void) {
    PyObject *m;

    // Prepare all types
    if (PyType_Ready(&MapType) < 0) return NULL;
    if (PyType_Ready(&SetType) < 0) return NULL;
    if (PyType_Ready(&MapTransactionType) < 0) return NULL;
    if (PyType_Ready(&SetTransactionType) < 0) return NULL;

    m = PyModule_Create(&smashtable_module);
    if (m == NULL) return NULL;

    // Add Map
    Py_INCREF(&MapType);
    if (PyModule_AddObject(m, "Map", (PyObject *)&MapType) < 0) {
        Py_DECREF(&MapType);
        Py_DECREF(m);
        return NULL;
    }

    // Add Set
    Py_INCREF(&SetType);
    if (PyModule_AddObject(m, "Set", (PyObject *)&SetType) < 0) {
        Py_DECREF(&SetType);
        Py_DECREF(&MapType);
        Py_DECREF(m);
        return NULL;
    }

    // Module metadata
    PyModule_AddStringConstant(m, "__version__", "0.1.0");
    PyModule_AddIntConstant(m, "__gil_free__", 1);

    return m;
}

#pragma endregion
