/**
 *  @brief Machinery every container shares - key layouts, scalar conversion, errors, and the cursor.
 *  @author Ash Vardanian
 *  @file python/shared.cpp
 *  @date August 17, 2026
 *
 *  Mirrors @c include/smashtable/shared.hpp on the C++ side: one place for the vocabulary the
 *  container files are written in, so they contain only their Python protocol and nothing else.
 *
 *  The four regions build on each other in order. A key layout says what a key IS - how it orders,
 *  how it hashes, where its floor is. Conversion carries a Python scalar across the boundary under
 *  that layout, refusing every type the store was not built for. Errors turn a store's status
 *  into a raised exception. The cursor is the one traversal in the binding, and every walk in every
 *  container goes through it.
 */

#include <cassert>
#include <functional>
#include <string_view>

#include "shared.hpp"

namespace ashvardanian::smashtable::py {

#pragma region Key Layout Functions

/**
 *  @brief Reads the alternative this layout guarantees, checked in debug and free in release.
 *  @param[in] value The key, which the store has already validated against its layout.
 *  @return Reference to the stored alternative.
 */
template <typename alternative_type_>
static alternative_type_ const &assume_layout(key_variant_t const &key) noexcept {
    auto const *held = std::get_if<alternative_type_>(&key.value);
    assert(held && "a key reached its comparator carrying a different layout than its store");
    return *held;
}

static bool key_less_i64(key_variant_t const &first, key_variant_t const &second) noexcept {
    return assume_layout<std::int64_t>(first) < assume_layout<std::int64_t>(second);
}

static bool key_less_u64(key_variant_t const &first, key_variant_t const &second) noexcept {
    return assume_layout<std::uint64_t>(first) < assume_layout<std::uint64_t>(second);
}

static bool key_less_str(key_variant_t const &first, key_variant_t const &second) noexcept {
    return assume_layout<utf8_t>(first).text < assume_layout<utf8_t>(second).text;
}

static bool key_less_bytes(key_variant_t const &first, key_variant_t const &second) noexcept {
    return assume_layout<bytes_t>(first).data < assume_layout<bytes_t>(second).data;
}

static std::size_t key_hash_i64(key_variant_t const &key) noexcept {
    return std::hash<std::int64_t> {}(assume_layout<std::int64_t>(key));
}

static std::size_t key_hash_u64(key_variant_t const &key) noexcept {
    return std::hash<std::uint64_t> {}(assume_layout<std::uint64_t>(key));
}

static std::size_t key_hash_str(key_variant_t const &key) noexcept {
    return std::hash<std::string> {}(assume_layout<utf8_t>(key).text);
}

static std::size_t key_hash_bytes(key_variant_t const &key) noexcept {
    // Salted apart from the text hash so a `str` and a `bytes` container with the same octets do not
    // share a partition layout by accident. The two never mix inside one container, so this only
    // affects which shard a key lands in.
    return std::hash<std::string> {}(assume_layout<bytes_t>(key).data) ^ 0x9E3779B97F4A7C15ull;
}

// An unbounded walk needs a value strictly below every stored key, and a default-constructed variant
// would carry the wrong alternative for three of the four layouts - reading a `std::string` out of an
// `int64_t`. Each layout names its own floor instead.

#pragma endregion Key Layout Functions

#pragma region Key Layout Tables

key_ops_t const key_ops_i64 {key_type_t::i64_k, "int", &key_less_i64, &key_hash_i64};
key_ops_t const key_ops_u64 {key_type_t::u64_k, "uint", &key_less_u64, &key_hash_u64};
key_ops_t const key_ops_str {key_type_t::str_k, "str", &key_less_str, &key_hash_str};
key_ops_t const key_ops_bytes {key_type_t::bytes_k, "bytes", &key_less_bytes, &key_hash_bytes};

#pragma endregion Key Layout Tables

#pragma region Key Argument Parsing

key_ops_t const *key_ops_from_python(PyObject *specification) noexcept {
    if (!specification) {
        PyErr_SetString(PyExc_TypeError, "a key type is required, as key=int, 'uint', str or bytes");
        return nullptr;
    }

    // The type objects first, since `key=int` reads better than `key='int'` at a call site.
    if (specification == reinterpret_cast<PyObject *>(&PyLong_Type)) return &key_ops_i64;
    if (specification == reinterpret_cast<PyObject *>(&PyUnicode_Type)) return &key_ops_str;
    if (specification == reinterpret_cast<PyObject *>(&PyBytes_Type)) return &key_ops_bytes;

    // Named rejections, so the two excluded types explain themselves rather than falling through to
    // the generic message. `bool` is checked before `float` only because it reads better in that order.
    if (specification == reinterpret_cast<PyObject *>(&PyBool_Type)) {
        PyErr_SetString(PyExc_TypeError, "bool is not a valid key type; store it as a value instead");
        return nullptr;
    }
    if (specification == reinterpret_cast<PyObject *>(&PyFloat_Type)) {
        PyErr_SetString(PyExc_TypeError, "float is not a valid key type; store it as a value instead");
        return nullptr;
    }

    if (PyUnicode_Check(specification)) {
        Py_ssize_t length = 0;
        char const *name = PyUnicode_AsUTF8AndSize(specification, &length);
        if (!name) return nullptr;
        std::string_view const spelled {name, static_cast<std::size_t>(length)};
        if (spelled == "int") return &key_ops_i64;
        if (spelled == "uint") return &key_ops_u64;
        if (spelled == "str") return &key_ops_str;
        if (spelled == "bytes") return &key_ops_bytes;
        if (spelled == "float" || spelled == "bool") {
            PyErr_Format(PyExc_TypeError, "%s is not a valid key type; store it as a value instead", name);
            return nullptr;
        }
        PyErr_Format(PyExc_ValueError, "unknown key type '%s'; expected 'int', 'uint', 'str' or 'bytes'", name);
        return nullptr;
    }

    PyErr_SetString(PyExc_TypeError, "key must be int, str, bytes, or one of 'int', 'uint', 'str', 'bytes'");
    return nullptr;
}

bool value_mode_from_python(PyObject *specification, value_mode_t &mode) noexcept {
    if (!specification || specification == Py_None) {
        mode = value_mode_t::scalars_k;
        return true;
    }
    if (specification == reinterpret_cast<PyObject *>(&PyBaseObject_Type)) {
        mode = value_mode_t::objects_k;
        return true;
    }
    if (PyUnicode_Check(specification)) {
        if (PyUnicode_CompareWithASCIIString(specification, "scalar") == 0) {
            mode = value_mode_t::scalars_k;
            return true;
        }
        if (PyUnicode_CompareWithASCIIString(specification, "object") == 0) {
            mode = value_mode_t::objects_k;
            return true;
        }
        PyErr_SetString(PyExc_ValueError, "unknown value mode; expected 'scalar' or 'object'");
        return false;
    }
    PyErr_SetString(PyExc_TypeError, "value must be 'scalar', 'object', or the object type");
    return false;
}

#pragma endregion Key Argument Parsing

#pragma region Converting Values In

/**
 *  @brief Copies bytes into an owned string, reporting exhaustion rather than aborting on it.
 *
 *  These conversions are @c noexcept because they sit in a C-API frame with nowhere for an
 *  exception to go, and a @c std::string built from user data is exactly where allocation fails.
 *  Without this the process ends instead of raising, which is not a failure a caller can handle.
 */
static bool own_bytes(char const *source, Py_ssize_t length, std::string &result) noexcept {
    try {
        result.assign(source, static_cast<std::size_t>(length));
        return true;
    }
    catch (...) {
        PyErr_NoMemory();
        return false;
    }
}

bool value_from_python(PyObject *object, value_mode_t mode, value_variant_t &result) noexcept {
    // `bool` is a `PyLong` subclass, so it has to be tested first or it disappears into `int`.
    if (PyBool_Check(object)) {
        result = value_variant_t {object == Py_True};
        return true;
    }
    if (PyLong_Check(object)) {
        int overflow = 0;
        long long const signed_value = PyLong_AsLongLongAndOverflow(object, &overflow);
        if (!overflow) {
            result = value_variant_t {static_cast<std::int64_t>(signed_value)};
            return true;
        }
        if (overflow > 0) {
            unsigned long long const unsigned_value = PyLong_AsUnsignedLongLong(object);
            if (!PyErr_Occurred()) {
                result = value_variant_t {static_cast<std::uint64_t>(unsigned_value)};
                return true;
            }
        }
        PyErr_Clear();
        PyErr_SetString(PyExc_OverflowError, "integers outside the 64-bit range are not supported");
        return false;
    }
    if (PyFloat_Check(object)) {
        double const number = PyFloat_AsDouble(object);
        if (number == -1.0 && PyErr_Occurred()) return false;
        result = value_variant_t {number};
        return true;
    }
    if (PyUnicode_Check(object)) {
        Py_ssize_t length = 0;
        char const *utf8 = PyUnicode_AsUTF8AndSize(object, &length);
        if (!utf8) return false;
        utf8_t owned;
        if (!own_bytes(utf8, length, owned.text)) return false;
        result = value_variant_t {std::move(owned)};
        return true;
    }
    if (PyBytes_Check(object)) {
        char *buffer = nullptr;
        Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(object, &buffer, &length) != 0) return false;
        bytes_t owned;
        if (!own_bytes(buffer, length, owned.data)) return false;
        result = value_variant_t {std::move(owned)};
        return true;
    }

    // Anything else is stored by reference, and only where the store promised to hold the GIL
    // around every operation that could touch its refcount.
    if (mode == value_mode_t::objects_k) {
        result = value_variant_t {object_t {object}};
        return true;
    }
    PyErr_Format(PyExc_TypeError, "unsupported value type: %s; build the store with value='object' to store it",
                 Py_TYPE(object)->tp_name);
    return false;
}

#pragma endregion Converting Values In

#pragma region Converting Keys In

/** @brief Names the offending type in a way that points at the fix rather than just the refusal. */
static void raise_wrong_key_type(PyObject *object, key_ops_t const *ops) noexcept {
    if (PyBool_Check(object) || PyFloat_Check(object)) {
        PyErr_Format(PyExc_TypeError, "%s is not a valid key; this store takes %s keys, and %s may only be a value",
                     Py_TYPE(object)->tp_name, ops->name, Py_TYPE(object)->tp_name);
        return;
    }
    PyErr_Format(PyExc_TypeError, "%s is not a valid key; this store takes %s keys", Py_TYPE(object)->tp_name,
                 ops->name);
}

bool key_from_python(PyObject *object, key_ops_t const *ops, key_variant_t &result) noexcept {
    // `bool` before `int`, always: `PyLong_Check(True)` is true, so a later branch would accept it.
    if (PyBool_Check(object) || PyFloat_Check(object)) {
        raise_wrong_key_type(object, ops);
        return false;
    }

    switch (ops->type) {
    case key_type_t::i64_k: {
        if (!PyLong_Check(object)) break;
        int overflow = 0;
        long long const value = PyLong_AsLongLongAndOverflow(object, &overflow);
        if (overflow) {
            PyErr_SetString(PyExc_OverflowError, "key does not fit a signed 64-bit integer");
            return false;
        }
        result = key_variant_t {static_cast<std::int64_t>(value)};
        return true;
    }
    case key_type_t::u64_k: {
        if (!PyLong_Check(object)) break;
        unsigned long long const value = PyLong_AsUnsignedLongLong(object);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            PyErr_SetString(PyExc_OverflowError, "key does not fit an unsigned 64-bit integer");
            return false;
        }
        result = key_variant_t {static_cast<std::uint64_t>(value)};
        return true;
    }
    case key_type_t::str_k: {
        if (!PyUnicode_Check(object)) break;
        Py_ssize_t length = 0;
        char const *utf8 = PyUnicode_AsUTF8AndSize(object, &length);
        if (!utf8) return false;
        utf8_t owned;
        if (!own_bytes(utf8, length, owned.text)) return false;
        result = key_variant_t {std::move(owned)};
        return true;
    }
    case key_type_t::bytes_k: {
        if (!PyBytes_Check(object)) break;
        char *buffer = nullptr;
        Py_ssize_t length = 0;
        if (PyBytes_AsStringAndSize(object, &buffer, &length) != 0) return false;
        bytes_t owned;
        if (!own_bytes(buffer, length, owned.data)) return false;
        result = key_variant_t {std::move(owned)};
        return true;
    }
    }

    raise_wrong_key_type(object, ops);
    return false;
}

#pragma endregion Converting Keys In

#pragma region Converting Back Out

PyObject *key_to_python(key_variant_t const &key) noexcept {
    // Keyed by type rather than by index, so neither converter can inherit the other's numbering,
    // and through `get_if` rather than `visit`, which may throw on a valueless variant.
    if (auto const *held = std::get_if<std::int64_t>(&key.value))
        return PyLong_FromLongLong(static_cast<long long>(*held));
    if (auto const *held = std::get_if<std::uint64_t>(&key.value))
        return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(*held));
    if (auto const *held = std::get_if<utf8_t>(&key.value))
        return PyUnicode_DecodeUTF8(held->text.data(), static_cast<Py_ssize_t>(held->text.size()), "strict");
    auto const *held = std::get_if<bytes_t>(&key.value);
    assert(held && "a key carrying no alternative reached its converter");
    return PyBytes_FromStringAndSize(held->data.data(), static_cast<Py_ssize_t>(held->data.size()));
}

PyObject *value_to_python(value_variant_t const &value) noexcept {
    if (auto const *held = std::get_if<std::int64_t>(&value.value))
        return PyLong_FromLongLong(static_cast<long long>(*held));
    if (auto const *held = std::get_if<std::uint64_t>(&value.value))
        return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(*held));
    if (auto const *held = std::get_if<double>(&value.value)) return PyFloat_FromDouble(*held);
    if (auto const *held = std::get_if<bool>(&value.value)) return PyBool_FromLong(*held ? 1 : 0);
    if (auto const *held = std::get_if<utf8_t>(&value.value))
        return PyUnicode_DecodeUTF8(held->text.data(), static_cast<Py_ssize_t>(held->text.size()), "strict");
    if (auto const *held = std::get_if<bytes_t>(&value.value))
        return PyBytes_FromStringAndSize(held->data.data(), static_cast<Py_ssize_t>(held->data.size()));
    auto const *held = std::get_if<object_t>(&value.value);
    assert(held && "a value carrying no alternative reached its converter");
    return Py_XNewRef(held->held);
}

PyObject *pair_to_python(key_variant_t const &key, value_variant_t const &value) noexcept {
    PyObject *key_object = key_to_python(key);
    if (!key_object) return nullptr;
    PyObject *value_object = value_to_python(value);
    if (!value_object) {
        Py_DECREF(key_object);
        return nullptr;
    }
    // The tuple takes its own reference to each half, so both are given back whether it was built or not.
    PyObject *pair = PyTuple_Pack(2, key_object, value_object);
    Py_DECREF(key_object);
    Py_DECREF(value_object);
    return pair;
}

#pragma endregion Converting Back Out

#pragma region Errors

int raise_for(module_state_t *state, status_t status, PyObject *key) noexcept {
    if (succeeded(status)) return 0;
    switch (status) {
    case status_t::consistency_k:
        PyErr_SetString(state->conflict_error, "a watched key changed since this transaction began");
        break;
    // The three below say which of a validation's checks turned the transaction away, so a retry
    // loop can tell a race it will win next time from a read window it should narrow first.
    case status_t::write_conflict_k:
        PyErr_SetString(state->write_conflict_error, "a key this transaction wrote was published over");
        break;
    case status_t::read_conflict_k:
        PyErr_SetString(state->read_conflict_error, "a key this transaction read was published over");
        break;
    case status_t::phantom_conflict_k:
        PyErr_SetString(state->phantom_conflict_error, "a window this transaction read gained or lost a member");
        break;
    case status_t::out_of_memory_heap_k: PyErr_NoMemory(); break;
    case status_t::key_not_found_k:
        if (key) PyErr_SetObject(PyExc_KeyError, key);
        else PyErr_SetString(PyExc_KeyError, "key not found");
        break;
    case status_t::key_already_exists_k:
        if (key) PyErr_SetObject(state->duplicate_key_error, key);
        else PyErr_SetString(state->duplicate_key_error, "key already exists");
        break;
    case status_t::invalid_argument_k: PyErr_SetString(PyExc_ValueError, "invalid argument"); break;
    case status_t::operation_not_permitted_k:
        PyErr_SetString(state->state_error, "operation not permitted in this transaction state");
        break;
    // Deliberately not a `MemoryError`: the probe run is full, which is a rehash the table could
    // not perform rather than a heap that could not supply memory, and the two want different
    // remedies from anyone who catches this.
    case status_t::capacity_exhausted_k:
        PyErr_SetString(state->error, "the table's probe sequence is full and it could not grow");
        break;
    // A bounded retry that gave up, which no store produces yet but which the vocabulary names.
    case status_t::operation_would_block_k:
        PyErr_SetString(state->error, "the operation gave up rather than block");
        break;
    // Anything the vocabulary names but this mapping has not claimed, spelled rather than numbered:
    // a reader tracing an unexpected failure gets the enumerator, not a number to go look up.
    default: PyErr_Format(state->error, "operation failed with status %s", name_of(status)); break;
    }
    return -1;
}

#pragma endregion Errors

#pragma region Cursor Type

static void cursor_dealloc(PyObject *self) noexcept {
    auto *walk = object_as<cursor_object_t>(self);
    PyObject_GC_UnTrack(self);
    if (walk->walk && walk->owner) object_as<container_object_t>(walk->owner)->store_ops->cursor_destroy(walk->walk);
    Py_CLEAR(walk->owner);
    walk->lock.~object_lock_t();
    PyTypeObject *type = Py_TYPE(self);
    PyObject_GC_Del(self);
    Py_DECREF(type); // Heap types are reference-counted by their instances
}

static int cursor_traverse(PyObject *self, visitproc visit, void *arg) noexcept {
    auto *walk = object_as<cursor_object_t>(self);
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(walk->owner);
    return 0;
}

static int cursor_clear(PyObject *self) noexcept {
    auto *walk = object_as<cursor_object_t>(self);
    Py_CLEAR(walk->owner);
    return 0;
}

static PyObject *cursor_next(PyObject *self) noexcept {
    auto *walk = object_as<cursor_object_t>(self);
    if (!walk->owner) return nullptr;

    // Both tables are read back from the store rather than cached here, so neither can disagree
    // with the object this walk is actually stepping through.
    auto const *header = object_as<container_object_t>(walk->owner);
    bool const over_a_map = header->store_ops->is_associative;

    key_variant_t found_key;
    value_variant_t found_value;
    bool advanced = false;

    // The probe takes partition locks, so the GIL is dropped around it - except where a value may be
    // an object, since moving one touches a refcount. A set has no values to move, so only a map can
    // ask for the GIL back. No Python object is built until it is back either way.
    value_mode_t const step_mode = over_a_map ? header->mode : value_mode_t::scalars_k;
    run_over_values(step_mode, walk->lock, [&]() noexcept {
        if (!walk->walk || walk->remaining == 0) return;
        advanced = header->store_ops->cursor_next(walk->walk, found_key, over_a_map ? &found_value : nullptr);
        if (advanced && walk->remaining > 0) --walk->remaining;
    });

    // No exception set, which CPython reads as `StopIteration`
    if (!advanced) return nullptr;

    switch (walk->yields) {
    case cursor_yields_t::keys_k: return key_to_python(found_key);
    case cursor_yields_t::values_k: return value_to_python(found_value);
    case cursor_yields_t::items_k: return pair_to_python(found_key, found_value);
    }
    return nullptr;
}

static PyType_Slot cursor_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(cursor_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(cursor_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(cursor_clear)},
    {Py_tp_iter, reinterpret_cast<void *>(PyObject_SelfIter)},
    {Py_tp_iternext, reinterpret_cast<void *>(cursor_next)},
    {Py_tp_doc, const_cast<char *>("Lazy walk over a store in key order.")},
    {0, nullptr},
};

PyType_Spec cursor_spec = {"smashtable._Cursor", sizeof(cursor_object_t), 0,
                           Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_DISALLOW_INSTANTIATION, cursor_slots};

PyObject *cursor_new(module_state_t *state, PyObject *container, cursor_yields_t yields, key_variant_t const *start,
                     key_variant_t const *stop, Py_ssize_t limit) noexcept {
    auto const *header = object_as<container_object_t>(container);

    // A set has no values, so asking one for values or items is a programming error rather than an
    // empty result - it would yield a default-constructed zero for every member.
    if (!header->store_ops->is_associative && yields != cursor_yields_t::keys_k) {
        PyErr_SetString(PyExc_TypeError, "a set has no values to walk");
        return nullptr;
    }

    // An unordered core supplies no bounds, so there is nothing to step through. No class installs a
    // walk over one, which is what makes this unreachable rather than merely refused.
    if (!header->store_ops->is_ordered) {
        PyErr_SetString(PyExc_TypeError, "this store has no ordering to walk");
        return nullptr;
    }

    // Opened before the object is built, so a store that cannot open a walk raises rather than
    // handing back a cursor that would end on its first step.
    expected<void *> opened = header->store_ops->cursor_make(header->store, start, stop);
    if (!opened) {
        [[maybe_unused]] int const raised = raise_for(state, opened.status());
        return nullptr;
    }

    auto *walk = PyObject_GC_New(cursor_object_t, state->cursor_type);
    if (!walk) {
        header->store_ops->cursor_destroy(*opened);
        return nullptr;
    }
    // No incref of the type here: `PyObject_GC_New` already took one, and the matching decref in
    // dealloc gives back exactly one. Taking a second immortalises the type in practice.

    // Placement-new the owned members, since `PyObject_GC_New` only hands back raw storage.
    new (&walk->lock) object_lock_t {};

    walk->owner = Py_NewRef(container);
    walk->yields = yields;
    walk->walk = *opened;
    walk->remaining = limit;

    PyObject_GC_Track(walk);
    return reinterpret_cast<PyObject *>(walk);
}

#pragma endregion Cursor Type

#pragma region View Type

static void mapping_view_dealloc(PyObject *self) noexcept {
    auto *view = object_as<mapping_view_object_t>(self);
    PyObject_GC_UnTrack(self);
    Py_CLEAR(view->owner);
    PyTypeObject *type = Py_TYPE(self);
    PyObject_GC_Del(self);
    Py_DECREF(type);
}

static int mapping_view_traverse(PyObject *self, visitproc visit, void *arg) noexcept {
    auto *view = object_as<mapping_view_object_t>(self);
    Py_VISIT(Py_TYPE(self));
    Py_VISIT(view->owner);
    return 0;
}

static int mapping_view_clear(PyObject *self) noexcept {
    auto *view = object_as<mapping_view_object_t>(self);
    Py_CLEAR(view->owner);
    return 0;
}

static PyObject *mapping_view_iter(PyObject *self) noexcept {
    auto *view = object_as<mapping_view_object_t>(self);
    module_state_t *state = state_of_type(self);
    if (!state) return nullptr;
    // `tp_clear` nulls the owner when the collector breaks a cycle through this object. Every
    // consumer has to expect that, because a finalizer running in the same collection pass can
    // reach an object that has already been cleared.
    if (!view->owner) {
        PyErr_SetString(state->state_error, "this view's store has been collected");
        return nullptr;
    }
    return cursor_new(state, view->owner, view->yields, nullptr, nullptr, -1);
}

static Py_ssize_t mapping_view_length(PyObject *self) noexcept {
    auto *view = object_as<mapping_view_object_t>(self);
    if (!view->owner) return 0; // Cleared by the collector; nothing left to count
    auto const *header = object_as<container_object_t>(view->owner);
    std::size_t size = 0;
    Py_BEGIN_ALLOW_THREADS;
    size = header->store_ops->size(header->store);
    Py_END_ALLOW_THREADS;
    return static_cast<Py_ssize_t>(size);
}

static PyObject *mapping_view_mapping(PyObject *self, void *) noexcept {
    auto *view = object_as<mapping_view_object_t>(self);
    if (!view->owner) Py_RETURN_NONE; // Cleared by the collector; there is no container to name
    return Py_NewRef(view->owner);
}

static PyGetSetDef mapping_view_getset[] = {
    {"mapping", mapping_view_mapping, nullptr, const_cast<char *>("The container this view reads."), nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr},
};

static PyType_Slot mapping_view_slots[] = {
    {Py_tp_dealloc, reinterpret_cast<void *>(mapping_view_dealloc)},
    {Py_tp_traverse, reinterpret_cast<void *>(mapping_view_traverse)},
    {Py_tp_clear, reinterpret_cast<void *>(mapping_view_clear)},
    {Py_tp_iter, reinterpret_cast<void *>(mapping_view_iter)},
    {Py_mp_length, reinterpret_cast<void *>(mapping_view_length)},
    {Py_tp_getset, reinterpret_cast<void *>(mapping_view_getset)},
    {Py_tp_doc, const_cast<char *>("Lazy view over a store's keys, values or items.")},
    {0, nullptr},
};

// Three names over one layout, because a traceback saying `KeysView` says which of the three the
// caller is holding, and `_View` beside the transaction handle's `View` said nothing and collided.
PyType_Spec keys_view_spec = {"smashtable.KeysView", sizeof(mapping_view_object_t), 0,
                              Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_DISALLOW_INSTANTIATION,
                              mapping_view_slots};

PyType_Spec values_view_spec = {"smashtable.ValuesView", sizeof(mapping_view_object_t), 0,
                                Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_DISALLOW_INSTANTIATION,
                                mapping_view_slots};

PyType_Spec items_view_spec = {"smashtable.ItemsView", sizeof(mapping_view_object_t), 0,
                               Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_DISALLOW_INSTANTIATION,
                               mapping_view_slots};

PyObject *mapping_view_new(module_state_t *state, PyObject *container, cursor_yields_t yields) noexcept {
    PyTypeObject *type = yields == cursor_yields_t::keys_k     ? state->keys_view_type
                         : yields == cursor_yields_t::values_k ? state->values_view_type
                                                               : state->items_view_type;
    auto *view = PyObject_GC_New(mapping_view_object_t, type);
    if (!view) return nullptr;
    // No incref of the type here: `PyObject_GC_New` already took one, and the matching decref in
    // dealloc gives back exactly one. Taking a second immortalises the type in practice.
    view->owner = Py_NewRef(container);
    view->yields = yields;
    PyObject_GC_Track(view);
    return reinterpret_cast<PyObject *>(view);
}

#pragma endregion View Type

#pragma region Windowed Arguments

bool window_from_python(char const *called, PyObject *const *args, Py_ssize_t count, PyObject *keywords,
                        PyObject *&start, PyObject *&stop, Py_ssize_t &limit) noexcept {
    if (count > 2) {
        PyErr_Format(PyExc_TypeError, "%s() takes at most two positional arguments", called);
        return false;
    }
    start = count > 0 ? args[0] : nullptr;
    stop = count > 1 ? args[1] : nullptr;
    limit = -1;
    if (!keywords) return true;

    Py_ssize_t const named = PyTuple_GET_SIZE(keywords);
    for (Py_ssize_t index = 0; index != named; ++index) {
        PyObject *name = PyTuple_GET_ITEM(keywords, index);
        PyObject *value = args[count + index];
        if (PyUnicode_CompareWithASCIIString(name, "start") == 0) {
            if (start) {
                PyErr_Format(PyExc_TypeError, "%s() got multiple values for 'start'", called);
                return false;
            }
            start = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "stop") == 0) {
            if (stop) {
                PyErr_Format(PyExc_TypeError, "%s() got multiple values for 'stop'", called);
                return false;
            }
            stop = value;
        }
        else if (PyUnicode_CompareWithASCIIString(name, "limit") == 0) {
            if (value == Py_None) continue;
            limit = PyNumber_AsSsize_t(value, PyExc_OverflowError);
            if (limit == -1 && PyErr_Occurred()) return false;
            // Negative is the walk's own spelling for uncounted, so a caller passing one would
            // silently receive the whole window rather than nothing.
            if (limit < 0) {
                PyErr_SetString(PyExc_ValueError, "limit cannot be negative");
                return false;
            }
        }
        else {
            PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%U'", called, name);
            return false;
        }
    }
    return true;
}

#pragma endregion Windowed Arguments

} // namespace ashvardanian::smashtable::py
