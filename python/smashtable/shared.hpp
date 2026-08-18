/**
 *  @brief Shared contract for the CPython extension - key layouts, stored values, module state, object layouts.
 *  @author Ash Vardanian
 *  @file python/smashtable/shared.hpp
 *  @date October 30, 2025
 *
 *  Every translation unit includes this first and nothing else of ours. It exists so the per-domain files -
 *  @c shared.cpp, @c sorted_map.cpp and the rest - compile against one fixed contract instead
 *  of negotiating types with each other.
 *
 *  @section lib_key_typing Key Typing
 *
 *  A container's key layout is fixed at construction and every key it stores has that layout. Only four are
 *  offered - signed and unsigned 64-bit integers, UTF-8 text, and opaque bytes. Floats and booleans are
 *  values, never keys: a float key would make ordering depend on a total order over NaN, and @c bool is a
 *  subtype of @c int in Python, so accepting it would silently alias two key spaces.
 *
 *  Because the layout is fixed, the comparison and hash functions are chosen once, at construction, and held
 *  as function pointers. Nothing re-examines a variant tag per comparison. @c key_less_t deliberately has no
 *  default constructor, so any container that tried to manufacture a comparator fails to compile rather than
 *  calling through a null pointer in a branch nobody exercises.
 */
#pragma once
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#if PY_VERSION_HEX < 0x030C0000
#error "SmashTable requires CPython 3.12 or later, where per-interpreter GIL support is declarable"
#endif

#include <cassert> // `assert`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int64_t`

#include <atomic>   // `std::atomic`
#include <mutex>    // `std::mutex`
#include <optional> // `std::optional`
#include <string>   // `std::string`
#include <variant>  // `std::variant`

#include <smashtable/basic_vector.hpp>
#include <smashtable/partitioned_store.hpp>
#include <smashtable/monotonic_store.hpp>

namespace ashvardanian::smashtable::py {

/** @brief Every method taking arguments uses the fast convention; keywords are walked by hand. */
#define ST_METHOD_FLAGS_ METH_FASTCALL | METH_KEYWORDS

#pragma region Stored Values

/** @brief UTF-8 text, kept distinct from bytes so a @c str never reads back as @c bytes. */
struct utf8_t {
    std::string text;
    bool operator<(utf8_t const &other) const noexcept { return text < other.text; }
    bool operator==(utf8_t const &other) const noexcept { return text == other.text; }
};

/** @brief An opaque byte string, never decoded. */
struct bytes_t {
    std::string data;
    bool operator<(bytes_t const &other) const noexcept { return data < other.data; }
    bool operator==(bytes_t const &other) const noexcept { return data == other.data; }
};

/**
 *  @brief One owned Python reference, for a container whose values may be arbitrary objects.
 *
 *  Copying and destroying this touches a refcount, which requires the GIL. That is why an object is
 *  only ever engaged in a container built with @c value_mode_t::objects_k, and why such a container
 *  never releases the GIL around anything that reads, writes, copies or destroys a value.
 */
struct object_t {
    PyObject *held {nullptr};

    object_t() = default;
    explicit object_t(PyObject *borrowed) noexcept : held(Py_XNewRef(borrowed)) {}
    object_t(object_t const &other) noexcept : held(Py_XNewRef(other.held)) {}
    object_t(object_t &&other) noexcept : held(std::exchange(other.held, nullptr)) {}
    object_t &operator=(object_t const &other) noexcept {
        if (this != &other) {
            PyObject *previous = held;
            held = Py_XNewRef(other.held);
            Py_XDECREF(previous);
        }
        return *this;
    }
    object_t &operator=(object_t &&other) noexcept {
        std::swap(held, other.held);
        return *this;
    }
    ~object_t() noexcept { Py_XDECREF(held); }

    // Never ordered or hashed - only a value may be an object, and values are neither.
    bool operator==(object_t const &other) const noexcept { return held == other.held; }
};

/**
 *  @brief What a key may be. Four alternatives, and the alternative index IS the layout.
 *
 *  Floats and booleans are absent by construction rather than by a boundary check: a total order over
 *  NaN is not one, and @c bool subclasses @c int in Python, so admitting it would alias two key
 *  spaces. Nothing here holds a @c PyObject, so a stored key outlives the object it came from and the
 *  GIL may be released around every operation that touches only keys.
 */
struct key_variant_t {
    std::variant<std::int64_t, std::uint64_t, utf8_t, bytes_t> value {std::int64_t {0}};

    key_variant_t() = default;
    key_variant_t(key_variant_t const &) = default;
    key_variant_t(key_variant_t &&) noexcept = default;
    key_variant_t &operator=(key_variant_t const &) = default;
    key_variant_t &operator=(key_variant_t &&) noexcept = default;

    explicit key_variant_t(std::int64_t integer) noexcept : value(integer) {}
    explicit key_variant_t(std::uint64_t integer) noexcept : value(integer) {}
    explicit key_variant_t(utf8_t &&text) noexcept : value(std::move(text)) {}
    explicit key_variant_t(bytes_t &&data) noexcept : value(std::move(data)) {}

    [[nodiscard]] std::size_t index() const noexcept { return value.index(); }

    /**
     *  @brief Deep copy that reports allocation failure instead of throwing.
     *    Watches outlive the entry they refer to, so they need an owned identifier.
     */
    [[nodiscard]] expected<key_variant_t> copy() const noexcept {
        try {
            return expected<key_variant_t> {key_variant_t {*this}, success_k};
        }
        catch (...) {
            return expected<key_variant_t> {key_variant_t {}, status_t::out_of_memory_heap_k};
        }
    }
};

/**
 *  @brief What a value may be: everything a key may be, plus the three things it may not.
 *
 *  The last alternative is only ever engaged in a container whose mode admits it - see @c value_mode_t.
 *  A scalar-mode container refuses an object at the boundary, which is what lets it keep releasing the
 *  GIL around store operations.
 */
struct value_variant_t {
    std::variant<std::int64_t, std::uint64_t, double, bool, utf8_t, bytes_t, object_t> value {std::int64_t {0}};

    value_variant_t() = default;
    value_variant_t(value_variant_t const &) = default;
    value_variant_t(value_variant_t &&) noexcept = default;
    value_variant_t &operator=(value_variant_t const &) = default;
    value_variant_t &operator=(value_variant_t &&) noexcept = default;

    explicit value_variant_t(std::int64_t integer) noexcept : value(integer) {}
    explicit value_variant_t(std::uint64_t integer) noexcept : value(integer) {}
    explicit value_variant_t(double number) noexcept : value(number) {}
    explicit value_variant_t(bool flag) noexcept : value(flag) {}
    explicit value_variant_t(utf8_t &&text) noexcept : value(std::move(text)) {}
    explicit value_variant_t(bytes_t &&data) noexcept : value(std::move(data)) {}
    explicit value_variant_t(object_t &&object) noexcept : value(std::move(object)) {}

    [[nodiscard]] std::size_t index() const noexcept { return value.index(); }
    [[nodiscard]] bool holds_object() const noexcept { return std::holds_alternative<object_t>(value); }

    /** @brief Deep copy that reports allocation failure instead of throwing. */
    [[nodiscard]] expected<value_variant_t> copy() const noexcept {
        try {
            return expected<value_variant_t> {value_variant_t {*this}, success_k};
        }
        catch (...) {
            return expected<value_variant_t> {value_variant_t {}, status_t::out_of_memory_heap_k};
        }
    }
};

/**
 *  @brief What a container admits as a value, which decides whether it may release the GIL.
 *
 *  A scalar container holds nothing that touches a refcount, so every store operation runs with the
 *  GIL dropped. An object container may, so it holds the GIL across anything that reads, writes,
 *  copies or destroys a value - key-only operations still release it.
 */
enum class value_mode_t : std::uint8_t { scalars_k, objects_k };

#pragma endregion Stored Values

#pragma region GIL Policy

/**
 *  @brief A per-object lock, distinct from the partition locks a store keeps.
 *
 *  A cursor and a transaction carry mutable state of their own - a walk position, a group state -
 *  that no store lock covers, and every operation over it drops the GIL part-way. @c PyMutex exists
 *  only from CPython 3.13, so 3.12 gets a @c std::mutex instead; neither is ever acquired with the
 *  GIL attached, which is what keeps a thread waiting here from being the one holding the GIL its
 *  owner must retake before it can unlock.
 */
struct object_lock_t {
#if PY_VERSION_HEX >= 0x030D0000
    PyMutex handle {0};
    void lock() noexcept { PyMutex_Lock(&handle); }
    void unlock() noexcept { PyMutex_Unlock(&handle); }
#else
    std::mutex handle;
    void lock() noexcept { handle.lock(); }
    void unlock() noexcept { handle.unlock(); }
#endif
};

/**
 *  @brief Runs a store operation with the GIL dropped, unless a value could touch a refcount.
 *
 *  A scalar-mode container holds nothing that refers to a Python object, so the whole operation is
 *  safe with the GIL released - which is what keeps writers from serializing on each other. An
 *  object-mode container may copy or destroy an owned reference anywhere inside the store, including
 *  in an MVCC version it makes on its own, so it keeps the GIL for the duration.
 */
template <typename operation_type_>
void run_over_values(value_mode_t mode, operation_type_ &&operation) noexcept {
    if (mode == value_mode_t::objects_k) {
        operation();
        return;
    }
    Py_BEGIN_ALLOW_THREADS;
    operation();
    Py_END_ALLOW_THREADS;
}

/**
 *  @brief The same policy, with @p lock held across the whole operation.
 *
 *  The lock is taken and dropped with the GIL detached even when the operation itself needs the GIL
 *  back, so no thread ever blocks on it while holding the GIL.
 */
template <typename operation_type_>
void run_over_values(value_mode_t mode, object_lock_t &lock, operation_type_ &&operation) noexcept {
    Py_BEGIN_ALLOW_THREADS;
    lock.lock();
    if (mode == value_mode_t::objects_k) {
        Py_BLOCK_THREADS;
        operation();
        Py_UNBLOCK_THREADS;
    }
    else { operation(); }
    lock.unlock();
    Py_END_ALLOW_THREADS;
}

#pragma endregion GIL Policy

#pragma region Key Layouts

/**
 *  @brief The four key layouts a container may be built around.
 *
 *  In order: signed 64-bit integers, which a Python @c int maps to by default; unsigned 64-bit, for
 *  keys above @c 2**63-1; UTF-8 text, ordered bytewise; and opaque bytes, never decoded. Floats and
 *  booleans are absent deliberately - see @c lib_key_typing above. The underlying width is pinned
 *  because the enumerator is stored in every container object and named in every error message.
 */
enum class key_type_t : std::uint8_t { i64_k, u64_k, str_k, bytes_k };

/** @brief Binds each enumerator to the alternative it names, so the two cannot drift apart. */
template <key_type_t type_>
inline constexpr std::size_t key_alternative_k = static_cast<std::size_t>(type_);
static_assert(std::variant_size_v<decltype(key_variant_t::value)> == 4);
static_assert(
    std::is_same_v<std::variant_alternative_t<key_alternative_k<key_type_t::i64_k>, decltype(key_variant_t::value)>,
                   std::int64_t>);
static_assert(
    std::is_same_v<std::variant_alternative_t<key_alternative_k<key_type_t::u64_k>, decltype(key_variant_t::value)>,
                   std::uint64_t>);
static_assert(
    std::is_same_v<std::variant_alternative_t<key_alternative_k<key_type_t::str_k>, decltype(key_variant_t::value)>,
                   utf8_t>);
static_assert(
    std::is_same_v<std::variant_alternative_t<key_alternative_k<key_type_t::bytes_k>, decltype(key_variant_t::value)>,
                   bytes_t>);

/** @brief Strict weak ordering over two keys of one layout. Never sees a mismatched pair. */
using key_less_fn_t = bool (*)(key_variant_t const &, key_variant_t const &) noexcept;

/** @brief Hash of one key, used to pick its partition. Equal keys must hash equally. */
using key_hash_fn_t = std::size_t (*)(key_variant_t const &) noexcept;

/** @brief Writes the smallest key of a layout, which is where an unbounded walk starts. */
using key_least_fn_t = void (*)(key_variant_t &) noexcept;

/**
 *  @brief Everything a container needs to know about its key layout, resolved once at construction.
 *
 *  One table per layout, all @c constexpr, all shared by every container using that layout - so the
 *  pointer a container holds is to a cache-resident object it never writes.
 *
 *  @c type names the layout and @c name is what @c key_type reports back - @c "int", @c "uint",
 *  @c "str" or @c "bytes". The three function pointers order two keys, hash one, and write the
 *  layout's smallest key, which is where an unbounded walk begins.
 */
struct key_ops_t {
    key_type_t type;
    char const *name;
    key_less_fn_t less;
    key_hash_fn_t hash;
    key_least_fn_t least;
};

extern key_ops_t const key_ops_i64;
extern key_ops_t const key_ops_u64;
extern key_ops_t const key_ops_str;
extern key_ops_t const key_ops_bytes;

/**
 *  @brief Resolves the @c key= argument, which accepts @c int, @c str, @c bytes and the four spellings.
 *  @param[in] specification The object passed as @c key=, borrowed.
 *  @return The table to use, or @c nullptr with a @c TypeError or @c ValueError raised.
 */
key_ops_t const *key_ops_from_python(PyObject *specification) noexcept;

/**
 *  @brief Resolves the @c value argument, which is absent, @c scalar or @c object.
 *  @param[in] specification The object passed as @c value, borrowed, or @c nullptr.
 *  @param[out] mode Written only on success; defaults to scalars when nothing was given.
 *  @return True on success; false with a @c TypeError or @c ValueError set otherwise.
 */
bool value_mode_from_python(PyObject *specification, value_mode_t &mode) noexcept;

/**
 *  @brief Orders keys through the function chosen at construction, so no variant tag is examined.
 *
 *  Has no default constructor on purpose. A container that manufactured its own comparator would call
 *  through a null pointer, and the deletion turns every such site into a compile error instead.
 */
struct key_less_t {
    using is_transparent = void;
    using value_type = key_variant_t;

    key_less_fn_t less;

    key_less_t() = delete;
    explicit constexpr key_less_t(key_less_fn_t function) noexcept : less(function) {}

    bool operator()(key_variant_t const &first, key_variant_t const &second) const noexcept {
        return less(first, second);
    }
};

/** @brief Hashes keys through the function chosen at construction. Deleted default, for the same reason. */
struct key_hash_t {
    key_hash_fn_t hash;

    key_hash_t() = delete;
    explicit constexpr key_hash_t(key_hash_fn_t function) noexcept : hash(function) {}

    std::size_t operator()(key_variant_t const &key) const noexcept { return hash(key); }
};

#pragma endregion Key Layouts

#pragma region Stores

using entry_t = mapping<key_variant_t, value_variant_t>;
using map_tree_t = monotonic_avl_map<key_variant_t, value_variant_t, key_less_t, std::allocator<entry_t>>;
using map_store_t = partitioned_store<map_tree_t, key_hash_t>;
using set_tree_t = monotonic_avl_set<key_variant_t, key_less_t, std::allocator<key_variant_t>>;
using set_store_t = partitioned_store<set_tree_t, key_hash_t>;

#pragma endregion Stores

/** @brief Declared here, defined under Module State below, so the cursor factories can name it. */
struct module_state_t;

/**
 *  @brief Views a Python object as the layout it was allocated with.
 *
 *  Every slot is handed a @c PyObject and every body needs its own layout back, so the cast appears
 *  in almost every function here. Naming it once keeps the spelling out of the bodies, where it says
 *  nothing about what the code does. Only ever applied to an object this module allocated.
 */
template <typename object_type_>
object_type_ *object_as(PyObject *object) noexcept {
    return reinterpret_cast<object_type_ *>(object);
}

#pragma region Object Layouts

/**
 *  @brief The header every container shares, so one cursor and one group can serve all of them.
 *
 *  @c ops is the key layout this container was built around, never null after construction. @c mode
 *  says whether its values may be arbitrary objects, which decides whether the GIL may be released
 *  around a value.
 *
 *  @c ordinal is what stops two groups deadlocking on each other. A group stages its participants in
 *  ordinal order rather than argument order, so @c atomic(a, b) on one thread and @c atomic(b, a) on
 *  another acquire the same partition locks in the same sequence; without it each would hold what the
 *  other waits for. Any consistent total order would do - creation order is used because it is
 *  reproducible across runs, which an address is not, and a hang is the one failure worth being able
 *  to replay.
 */
struct container_object_t {
    PyObject_HEAD key_ops_t const *ops;
    std::uint64_t ordinal;
    value_mode_t mode;
};

/** @brief Holds its store inline, placement-constructed into @c tp_alloc's storage. */
struct sorted_map_object_t {
    container_object_t base;
    map_store_t store;
};

/** @brief Holds its store inline, placement-constructed into @c tp_alloc's storage. */
struct sorted_set_object_t {
    container_object_t base;
    set_store_t store;
};

/** @brief Assigns each container a process-wide rank, used to order staging deterministically. */
/** @brief The module state reached from a heap type, for the constructors that have no instance. */
module_state_t *state_of_heap_type(PyTypeObject *type) noexcept;

#pragma endregion Object Layouts

#pragma region Cursors

/** @brief Which container family a cursor is walking, so it can reach the right store. */
enum class cursor_family_t : std::uint8_t { map_k, set_k };

/** @brief What a cursor hands back per step. */
enum class cursor_yields_t : std::uint8_t { keys_k, values_k, items_k };

/**
 *  @brief Where a walk stands.
 *
 *  Three named states rather than a pair of flags, so "not begun" and "finished" cannot be confused
 *  or set at once. Fresh means nothing has been yielded and the next step seeks; walking means a key
 *  has been yielded and the next step advances strictly past it; exhausted means the walk ended, and
 *  it stays ended even if the container grows again.
 */
enum class cursor_state_t : std::uint8_t { fresh_k, walking_k, exhausted_k };

/**
 *  @brief The fixed limits of a walk: how far it may go and how many steps it may take.
 *
 *  Set once at construction and never written again, which is what separates them from the two
 *  members that move. An absent @c stop is unbounded above; a negative @c remaining is uncounted.
 */
struct walk_limits_t {
    std::optional<key_variant_t> stop;
    Py_ssize_t remaining {-1};
};

/**
 *  @brief One lazy walk in key order, holding the last key by value rather than any node pointer.
 *
 *  The step is @c lower_bound while fresh and @c upper_bound after, which is the loop @c scan already
 *  ran - lifted out and made resumable, so the binding has exactly one traversal. Holding the position
 *  by value is what makes erasing the key it sits on harmless.
 *
 *  @c owner is a strong reference keeping the container alive for the walk, and is the only place the
 *  layout and the family are recorded - both are read back from it per step rather than cached here,
 *  so a cache can never disagree with the container it describes. @c limits are fixed at construction;
 *  @c position and @c state are the only members a step writes.
 *
 *  @c lock makes one step the unit of exclusion. A step drops the GIL, so two threads pulling from one
 *  iterator would otherwise both read @c position, both step from it, and both assign it back - a data
 *  race on a @c std::string for the two text layouts.
 */
struct cursor_object_t {
    PyObject_HEAD PyObject *owner;
    object_lock_t lock;
    walk_limits_t limits;
    key_variant_t position;
    cursor_state_t state;
    cursor_yields_t yields;
};

/**
 *  @brief A lazy view over a container - what @c keys, @c values and @c items return.
 *
 *  @c owner is a strong reference to the container, so a view outlives no store; the other two say
 *  which family it belongs to and what each step produces. A fresh cursor is made per iteration.
 */
struct mapping_view_object_t {
    PyObject_HEAD PyObject *owner;
    cursor_family_t family;
    cursor_yields_t yields;
};

/**
 *  @brief Builds a cursor over a container, optionally bounded.
 *  @param[in] state The module state holding the cursor type.
 *  @param[in] container The container to walk, borrowed; a strong reference is taken.
 *  @param[in] family Which store the container holds.
 *  @param[in] yields What each step should produce.
 *  @param[in] start Inclusive lower bound, or @c nullptr to begin at the layout's floor.
 *  @param[in] stop Exclusive upper bound, or @c nullptr for unbounded.
 *  @param[in] limit Maximum number of steps, or -1.
 *  @return A new reference to a cursor, or @c nullptr with an exception set.
 */
PyObject *cursor_new(module_state_t *state, PyObject *container, cursor_yields_t yields, key_variant_t const *start,
                     key_variant_t const *stop, Py_ssize_t limit) noexcept;

/** @brief Builds a lazy view. Same ownership rules as @c cursor_new. */
PyObject *mapping_view_new(module_state_t *state, PyObject *container, cursor_yields_t yields) noexcept;

extern PyType_Spec cursor_spec;
extern PyType_Spec mapping_view_spec;
extern PyType_Spec sorted_map_spec;
extern PyType_Spec sorted_set_spec;

/**
 *  @brief Visits every element once, in key order, stepping exactly as the Python cursor does.
 *
 *  For the whole-container operations - @c __repr__, @c __eq__, @c copy - which need every element but
 *  have no reason to build a Python object per step. Same exclusive-successor stepping as the cursor, so
 *  the two cannot disagree about what "every element" means.
 */
template <typename store_type_, typename callback_type_>
void for_each_in_order(store_type_ &store, key_ops_t const *ops, callback_type_ &&callback) noexcept {
    using value_t = typename store_type_::value_t;

    key_variant_t cursor;
    ops->least(cursor);
    cursor_state_t state = cursor_state_t::fresh_k;
    bool wants_more = true;
    while (wants_more) {
        bool advanced = false;
        auto keep = [&](value_t const &element) noexcept {
            cursor = mapping_key_or_itself<value_t>(element);
            advanced = true;
            // A callback answering `bool` stops the walk when it says so; one answering `void` is
            // asking for every element, and the difference is resolved here rather than by a flag.
            if constexpr (std::is_same_v<decltype(callback(element)), bool>) wants_more = callback(element);
            else callback(element);
        };
        auto missing = []() noexcept {};
        if (state == cursor_state_t::fresh_k)
            store.lower_bound(cursor, keep, missing), state = cursor_state_t::walking_k;
        else store.upper_bound(cursor, keep, missing);
        if (!advanced) break;
    }
}

#pragma endregion Cursors

#pragma region Transactions

/**
 *  @brief One participant in a group transaction, whatever container it came from.
 *
 *  The set of stores is closed and known at compile time, so the alternatives are held in a variant
 *  rather than behind a base class. Every uniform operation - stage, commit, rollback, reset, erase,
 *  watch, contains - is one @c std::visit over a two-alternative variant, which lowers to a switch
 *  with no vtable, no indirect call and no per-participant heap allocation. The three operations that
 *  genuinely differ between a map and a set are the only ones that name the alternatives.
 */
struct participant_t {
    std::variant<map_store_t::transaction_t, set_store_t::transaction_t> inner;
    key_ops_t const *ops;
    value_mode_t mode;

    /**
     *  @brief Whether this participant stores values as well as keys.
     *
     *  Asks the store's element type rather than reading an alternative index, so reordering the
     *  variant cannot silently invert every map-versus-set decision downstream.
     */
    [[nodiscard]] bool is_associative() const noexcept {
        // Which alternative is engaged, asked as a type question rather than by index, so reordering
        // the variant cannot silently invert every map-versus-set decision downstream.
        return std::holds_alternative<map_store_t::transaction_t>(inner);
    }

    [[nodiscard]] bool contains(key_variant_t const &key) noexcept {
        return std::visit([&](auto &transaction) noexcept { return transaction.contains(key); }, inner);
    }

    /** @brief Reads a mapped value. Always false for a set, which has none. */
    [[nodiscard]] bool find(key_variant_t const &key, value_variant_t &value) noexcept {
        auto *as_map = std::get_if<map_store_t::transaction_t>(&inner);
        assert(as_map && "find on a set participant; callers check is_associative first");
        bool present = false;
        as_map->find(
            key, [&](entry_t const &entry) noexcept { value = entry.mapped, present = true; }, []() noexcept {});
        return present;
    }

    /** @brief Inserts or overwrites a key and value. Refuses on a set. */
    [[nodiscard]] status_t upsert(key_variant_t &&key, value_variant_t &&value) noexcept {
        auto *as_map = std::get_if<map_store_t::transaction_t>(&inner);
        assert(as_map && "upsert on a set participant; callers check is_associative first");
        return as_map->upsert(entry_t {std::move(key), std::move(value)});
    }

    /** @brief Inserts a bare member. Refuses on a map, which needs a value. */
    [[nodiscard]] status_t add(key_variant_t &&key) noexcept {
        auto *as_set = std::get_if<set_store_t::transaction_t>(&inner);
        assert(as_set && "add on a map participant; callers check is_associative first");
        return as_set->upsert(std::move(key));
    }

    [[nodiscard]] status_t erase(key_variant_t const &key) noexcept {
        return std::visit([&](auto &transaction) noexcept { return transaction.erase(key); }, inner);
    }
    [[nodiscard]] status_t watch(key_variant_t const &key) noexcept {
        return std::visit([&](auto &transaction) noexcept { return transaction.watch(key); }, inner);
    }
    [[nodiscard]] status_t stage() noexcept {
        return std::visit([](auto &transaction) noexcept { return transaction.stage(); }, inner);
    }
    [[nodiscard]] status_t commit() noexcept {
        return std::visit([](auto &transaction) noexcept { return transaction.commit(); }, inner);
    }
    [[nodiscard]] status_t rollback() noexcept {
        return std::visit([](auto &transaction) noexcept { return transaction.rollback(); }, inner);
    }
    [[nodiscard]] status_t reset() noexcept {
        return std::visit([](auto &transaction) noexcept { return transaction.reset(); }, inner);
    }
};

/**
 *  @brief Where a group stands.
 *
 *  One enum rather than a pair of flags, so "staged" and "finished" cannot both be true - which two
 *  booleans prevented only through the order of two assignments.
 */
enum class group_state_t : std::uint8_t { open_k, staged_k, finished_k };

/**
 *  @brief A group of containers updated all-or-nothing.
 *
 *  @c containers holds the participants in the caller's order and @c views the per-container handles
 *  parallel to it, while @c parts holds the open transactions in canonical staging order. It lives
 *  inside the object, placement-constructed into @c PyObject_GC_New's storage, and reports a failed
 *  reservation as a status rather than throwing - an exception has nowhere to go inside a C-API frame.
 *
 *  @c lock makes the state test, the pass over @c parts and the state transition one span. Each of
 *  those passes drops the GIL, so without it two threads could both see an open group and both stage
 *  it, and a write through a view could land in a participant a commit was already draining.
 */
struct transaction_object_t {
    PyObject_HEAD PyObject *containers;
    PyObject *views;
    object_lock_t lock;
    basic_vector<participant_t> parts;
    group_state_t state;
};

/**
 *  @brief One container's handle on a group, valid only while that group is open.
 *
 *  @c owner is a strong reference to the transaction, which must outlive every view onto it, and
 *  @c index names which participant this view speaks for.
 */
struct view_object_t {
    PyObject_HEAD PyObject *owner;
    std::size_t index;
};

extern PyType_Spec transaction_spec;
extern PyType_Spec view_spec;

/**
 *  @brief Opens one transaction spanning every container given, in a deadlock-free order.
 *  @param[in] state The module state.
 *  @param[in] containers A tuple of containers, borrowed; duplicates are refused.
 *  @return A new reference to a transaction, or @c nullptr with an exception set.
 */
PyObject *make_transaction(module_state_t *state, PyObject *containers) noexcept;

#pragma endregion Transactions

#pragma region Module State

/**
 *  @brief Everything this module owns, per interpreter.
 *
 *  Nothing here is a process-wide static. That is deliberate: the module declares
 *  @c Py_MOD_PER_INTERPRETER_GIL_SUPPORTED, and a hidden global would be shared by interpreters that
 *  are meant to share nothing - including @c next_ordinal, which is state rather than a constant.
 */
struct module_state_t {
    /** @brief Hands each container its staging rank. See @c container_object_t::ordinal. */
    std::atomic<std::uint64_t> next_ordinal;

    PyTypeObject *sorted_map_type;
    PyTypeObject *sorted_set_type;
    PyTypeObject *transaction_type;
    PyTypeObject *view_type;
    PyTypeObject *cursor_type;
    PyTypeObject *mapping_view_type;
    PyObject *error;
    PyObject *conflict_error;
    PyObject *duplicate_key_error;
    PyObject *state_error;
};

/** @brief Defined in @c module.cpp, once the tables it points at exist. */
PyModuleDef *smashtable_module_def() noexcept;

module_state_t *state_of(PyObject *module) noexcept;
module_state_t *state_of_type(PyObject *self) noexcept;

#pragma endregion Module State

#pragma region Errors

/**
 *  @brief Turns a @c status_t into a raised Python exception.
 *    The only place aware that @c status_t is truthy on success.
 *  @param[in] state The module state holding the custom exception types.
 *  @param[in] status The outcome to translate.
 *  @param[in] key The key to attach to a lookup failure, borrowed, or @c nullptr.
 *  @return 0 when the operation succeeded, -1 with an exception set otherwise.
 */
int raise_for(module_state_t *state, status_t status, PyObject *key = nullptr) noexcept;

#pragma endregion Errors

#pragma region Conversion

/**
 *  @brief Reads a Python object into an owned value.
 *  @param[in] object The value to read, borrowed.
 *  @param[in] mode Whether this container admits arbitrary objects or only scalars.
 *  @param[out] result Written only on success.
 *  @return True on success; false with an exception set otherwise.
 */
bool value_from_python(PyObject *object, value_mode_t mode, value_variant_t &result) noexcept;

/**
 *  @brief Reads a Python object as a key of one specific layout, rejecting every other type.
 *  @param[in] object The candidate key, borrowed.
 *  @param[in] ops The layout the container was built around.
 *  @param[out] result Written only on success.
 *  @return True on success; false with a @c TypeError or @c OverflowError set otherwise.
 */
bool key_from_python(PyObject *object, key_ops_t const *ops, key_variant_t &result) noexcept;

/**
 *  @brief Builds a new Python object from a stored key.
 *  @return A new reference, or @c nullptr with an exception set.
 */
PyObject *key_to_python(key_variant_t const &key) noexcept;

/**
 *  @brief Builds a new Python object from a stored value.
 *  @return A new reference, or @c nullptr with an exception set.
 */
PyObject *value_to_python(value_variant_t const &value) noexcept;

#pragma endregion Conversion

} // namespace ashvardanian::smashtable::py
