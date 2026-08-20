/**
 *  @brief Shared contract for the CPython extension - key layouts, stored values, module state, object layouts.
 *  @author Ash Vardanian
 *  @file python/shared.hpp
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
#include <optional> // `std::optional`
#include <string>   // `std::string`
#include <utility>  // `std::exchange`, `std::swap`
#include <variant>  // `std::variant`

#include <smashtable/basic_vector.hpp>
#include <smashtable/shared.hpp>

namespace ashvardanian::smashtable::py {

/** @brief Every method taking arguments uses the fast convention; keywords are walked by hand. */
#define ST_METHOD_FLAGS_ METH_FASTCALL | METH_KEYWORDS

/** @brief Casts a fast-convention function into a method table's slot without tripping -Wcast-function-type. */
template <typename function_type_>
PyCFunction as_pycfunction(function_type_ function) noexcept {
    return reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)()>(function));
}

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

struct releases_t;

/**
 *  @brief One owned Python reference, for a store whose values may be arbitrary objects.
 *
 *  Copying and destroying this touches a refcount, which requires the GIL. That is why an object is
 *  only ever engaged in a store built with @c value_mode_t::objects_k, and why such a store
 *  never releases the GIL around anything that reads, writes, copies or destroys a value.
 */
struct object_t {
    PyObject *held {nullptr};
    /** @brief The container's ledger this reference reports its drop to, null where none was named. */
    releases_t *releases {nullptr};

    object_t() = default;
    explicit object_t(PyObject *borrowed, releases_t *ledger = nullptr) noexcept
        : held(Py_XNewRef(borrowed)), releases(ledger) {}
    object_t(object_t const &other) noexcept : held(Py_XNewRef(other.held)), releases(other.releases) {}
    object_t(object_t &&other) noexcept
        : held(std::exchange(other.held, nullptr)), releases(std::exchange(other.releases, nullptr)) {}
    object_t &operator=(object_t const &other) noexcept {
        if (this != &other) {
            PyObject *previous = std::exchange(held, Py_XNewRef(other.held));
            releases_t *previous_ledger = std::exchange(releases, other.releases);
            give_back(previous, previous_ledger);
        }
        return *this;
    }
    object_t &operator=(object_t &&other) noexcept {
        std::swap(held, other.held);
        std::swap(releases, other.releases);
        return *this;
    }
    ~object_t() noexcept { give_back(std::exchange(held, nullptr), releases); }

    // Never ordered or hashed - only a value may be an object, and values are neither.
    bool operator==(object_t const &other) const noexcept { return held == other.held; }

  private:
    /**
     *  @brief Gives one reference back, now or once the store calls in flight have returned.
     *
     *  Defined below @c releases_t, which it needs whole. Where the ledger cannot record the drop it
     *  is released here and the deadlock it exists to avoid is back on the table - the better of two
     *  answers, since a hang cannot be recovered from.
     */
    static void give_back(PyObject *object, releases_t *ledger) noexcept;
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
 *  The last alternative is only ever engaged in a store whose mode admits it - see @c value_mode_t.
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
 *  @brief What a store admits as a value, which decides whether it may release the GIL.
 *
 *  A scalar container holds nothing that touches a refcount, so every store operation runs with the
 *  GIL dropped. An object container may, so it holds the GIL across anything that reads, writes,
 *  copies or destroys a value - key-only operations still release it.
 */
enum class value_mode_t : std::uint8_t { scalars_k, objects_k };

#pragma endregion Stored Values

#pragma region GIL Policy

#pragma region Deferred Releases

/**
 *  @brief Where references dropped inside a store call wait until that call's lock is gone.
 *
 *  A stored value's destructor gives back a reference, and the last one runs @c __del__ - arbitrary
 *  Python. The store destroys what it displaces while holding its own lock, and that lock is not
 *  recursive, so a finalizer touching the store it was stored in blocks forever against the write that
 *  freed it. While a call is in flight the drop is therefore recorded rather than performed, and what
 *  was recorded is released once the last call has returned.
 *
 *  One of these belongs to each container, and every object that container stores names it, so the
 *  ledger a drop reports to is carried rather than looked up. It locks @b shared because several calls
 *  may be in flight at once and the last one out does the releasing, which is what a reader count is;
 *  @c shared_lock is therefore the guard, and the bridge needs no guard type of its own.
 *
 *  @warning A batch of finalizers runs at the moment the last call returns, so one @c __del__ can fire
 *    inside another's user code. A @c __del__ that takes a non-reentrant lock can deadlock against
 *    itself here where it would not against @c dict, which releases at assignment instead.
 */
class releases_t {
    /** @brief How many store calls are in flight on this container, across every thread. */
    alignas(atomic_alignment<std::size_t>) std::size_t calls_in_flight_ {0};
    /** @brief Guards @c recorded_, which a partitioned store appends to from several threads at once. */
    spin_shared_mutex_t guard_;
    /** @brief What has been dropped and not yet given back. */
    basic_vector<PyObject *> recorded_;

    /**
     *  @brief Gives back one batch - what stood recorded on entry, and no more.
     *
     *  Drains under the guard and releases outside it, because one @c Py_DECREF can run a finalizer that
     *  enters the store and records more, which would deadlock against a guard still held and dangle a
     *  vector appended to while walked. Draining only one batch is what guarantees progress: looping
     *  until empty lets several writers feed one draining thread as fast as it drains, and every later
     *  call releases again, so nothing is stranded.
     */
    void drain_() noexcept {
        basic_vector<PyObject *> draining;
        {
            unique_lock locked {guard_};
            if (recorded_.empty()) return;
            draining = std::move(recorded_);
            recorded_.clear();
        }
        for (PyObject *object : draining) Py_DECREF(object);
    }

  public:
    releases_t() = default;
    releases_t(releases_t const &) = delete;
    releases_t &operator=(releases_t const &) = delete;

    /** @brief Gives back whatever a finalizer recorded during the final drain, which would else leak. */
    ~releases_t() noexcept {
        while (!recorded_.empty()) drain_();
    }

    /** @brief Whether a store call is in flight, so a drop has to wait for it. */
    [[nodiscard]] bool armed() const noexcept { return atomic_load(calls_in_flight_) != 0; }

    /**
     *  @brief Records one dropped reference, reporting why it could not.
     *
     *  @c basic_vector answers exhaustion rather than throwing, which is what a @c noexcept destructor
     *  needs. Where it cannot record, the caller releases on the spot and the deadlock is back on the
     *  table - the better of two answers, since a hang cannot be recovered from.
     */
    [[nodiscard]] status_t record(PyObject *object) noexcept {
        unique_lock locked {guard_};
        return recorded_.push_back(std::move(object));
    }

    /** @brief One store call arrives. Shared, because several may be in flight at once. */
    void lock_shared() noexcept {
        [[maybe_unused]] std::size_t const arrived = atomic_add_fetch(calls_in_flight_, std::size_t {1});
    }

    /** @brief One store call leaves, and the last one out gives back what the calls recorded. */
    void unlock_shared() noexcept {
        if (atomic_sub_fetch(calls_in_flight_, std::size_t {1}) == 0) drain_();
    }
};

#pragma endregion Deferred Releases

inline void object_t::give_back(PyObject *object, releases_t *ledger) noexcept {
    if (!object) return;
    if (ledger && ledger->armed() && succeeded(ledger->record(object))) return;
    Py_DECREF(object);
}

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
void run_over_values(value_mode_t mode, spin_shared_mutex_t &lock, operation_type_ &&operation) noexcept {
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
 *  @brief The four key layouts a store may be built around.
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

/**
 *  @brief Everything a store needs to know about its key layout, resolved once at construction.
 *
 *  One table per layout, all @c constexpr, all shared by every container using that layout - so the
 *  pointer a store holds is to a cache-resident object it never writes.
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

/**
 *  @brief Hashes a key by its layout, without consulting the table a store was built with.
 *
 *  Stateless and default-constructible on purpose: an unordered core builds its hasher itself, with
 *  no seed passed in, so the function-pointer form @c key_hash_t takes cannot be used there. The
 *  switch costs one predictable branch per operation rather than the per-comparison dispatch that
 *  @c key_ops_t exists to avoid, because a probe hashes once and then compares.
 */
struct key_variant_hash_t {
    std::size_t operator()(key_variant_t const &key) const noexcept {
        return std::visit(
            [](auto const &held) noexcept -> std::size_t {
                using held_t = std::decay_t<decltype(held)>;
                if constexpr (std::is_same_v<held_t, utf8_t>) return std::hash<std::string> {}(held.text);
                else if constexpr (std::is_same_v<held_t, bytes_t>) return std::hash<std::string> {}(held.data);
                else return std::hash<held_t> {}(held);
            },
            key.value);
    }
};

/**
 *  @brief Equality over two keys of one layout, which the variant already answers exactly.
 *    Stateless for the same reason as @c key_variant_hash_t, and seeded into an unordered core as
 *    the one policy that core does take from the caller.
 */
struct key_variant_equal_t {
    using is_transparent = void;
    bool operator()(key_variant_t const &first, key_variant_t const &second) const noexcept {
        return first.value == second.value;
    }
};

#pragma endregion Key Layouts

#pragma region Stores

using entry_t = mapping<key_variant_t, value_variant_t>;

/**
 *  @brief One store instantiation reduced to a table of function pointers, resolved once.
 *
 *  The same shape as @c key_ops_t one level up: one @c constexpr table per instantiation, all shared by
 *  every container built on it, so the pointer a store holds is to an object it never writes. It is
 *  what lets one @c container.cpp serve every core, isolation level and sharing strategy without naming
 *  a concrete store, and what keeps the participant of a transaction group free of a variant whose arms
 *  would grow with the matrix.
 *
 *  @c store is always a pointer handed back by @c make and owned by exactly one container object.
 *
 *  A slot left null says the core does not carry that operation - a hash core supplies no ordering, so
 *  its bounds and range erase are absent. Null is never tested in a method body: the class that would
 *  have called it does not install the method, so the mismatch surfaces as @c AttributeError rather than
 *  as a refusal invented here.
 *
 *  Reads answer through out-parameters rather than the callbacks the stores take, because a function
 *  pointer cannot carry a template-typed callback. Each returns whether anything was found, which
 *  collapses the found-and-missing pair the C++ side uses.
 */
struct store_ops_t {
    /** @brief What this instantiation promises, straight from the store's @c isolation_k. */
    isolation_t isolation;
    /** @brief That promise as a Jepsen name, which @c isolation reports back. */
    char const *isolation_name;
    /** @brief How the store is shared, which @c sharing reports back verbatim. */
    char const *sharing_name;
    /** @brief Whether elements carry a mapped value, deciding map-versus-set at every shared site. */
    bool is_associative;
    /** @brief Whether the core orders its keys, which is what the ordered surface rests on. */
    bool is_ordered;

    /** @brief Builds an empty store of @p ops's layout, or reports why it could not. */
    expected<void *> (*make)(key_ops_t const *ops) noexcept;
    /** @brief Destroys a store @c make handed back. Never called with null. */
    void (*destroy)(releases_t &releases, void *store) noexcept;

    std::size_t (*size)(void *store) noexcept;
    status_t (*clear)(releases_t &releases, void *store) noexcept;
    /**
     *  @brief Whether @p key is held, reported through @p present.
     *
     *  The answer is an out-parameter for the same reason the bounds use one: membership is a read,
     *  and at a serializable level a read records itself and can therefore refuse. "Not held" and
     *  "could not be answered" are different facts.
     */
    expected<bool> (*contains)(void *store, key_variant_t const &key) noexcept;
    /** @brief Reads a mapped value, or reports @c key_not_found_k. Null on a set, which has none. */
    expected<value_variant_t> (*find)(releases_t &releases, void *store, key_variant_t const &key) noexcept;
    /** @brief Inserts or overwrites. @p value is null for a set, which stores the key alone. */
    status_t (*upsert)(releases_t &releases, void *store, key_variant_t &&key, value_variant_t *value) noexcept;

    /**
     *  @brief Applies a whole batch, staged once and committed once rather than element by element.
     *
     *  The store opens one transaction for the batch, so the batch lands whole or not at all. A loop
     *  of single writes is neither: it takes the lock once per element and leaves a failure halfway
     *  through half-applied.
     */
    status_t (*upsert_entries)(releases_t &releases, void *store, entry_t *entries, std::size_t count) noexcept;

    /** @brief The same for a set, whose elements are bare keys rather than pairs. Null on a map. */
    status_t (*upsert_members)(releases_t &releases, void *store, key_variant_t *members, std::size_t count) noexcept;
    /**
     *  @brief Removes a key, answering with what it held or with why it could not.
     *
     *  One locked span rather than a probe beside the removal: a @c find followed by an @c erase is
     *  two spans, so another writer between them makes the pair report a value nobody removed, or a
     *  miss for a key this call did delete. Absence is @c key_not_found_k, which is how the store
     *  itself reports it, so nothing here repeats the answer in a second place.
     *
     *  A set has no value to give back and answers with a default one; only its status means anything.
     */
    expected<value_variant_t> (*erase)(releases_t &releases, void *store, key_variant_t const &key) noexcept;

    /**
     *  @brief Removes the least member and hands back both halves, or reports @c key_not_found_k.
     *
     *  One store call rather than a seek and an erase, so the member reported is the one that was
     *  taken. Null on an unordered core. A set fills only the key half.
     */
    expected<entry_t> (*pop_smallest)(releases_t &releases, void *store) noexcept;

    /**
     *  @brief Fills @p result with @p algebra over @p first and @p second, which share this table.
     *
     *  Only reachable when both sides were built from one table, which is what lets the second side
     *  be driven through it. Null on a map, which has no set algebra.
     */
    status_t (*set_algebra)(releases_t &releases, void *first, void *second, void *result, algebra_t algebra) noexcept;

    /**
     *  @brief Opens the store's own ordered walk, optionally bounded at either end.
     *
     *  Both ends are the store's business: it holds the comparator, so a bound tested anywhere else
     *  would be ordering keys differently from the store that holds them. Null on an unordered core.
     */
    expected<void *> (*cursor_make)(void *store, key_variant_t const *from, key_variant_t const *upper) noexcept;

    /** @brief Closes a walk opened by @c cursor_make. */
    void (*cursor_destroy)(releases_t &releases, void *cursor) noexcept;

    /**
     *  @brief Takes one step, writing the key and, for a map, the value.
     *  @return Whether a member was handed over; false once the walk is over.
     *
     *  The two halves are written into borrowed storage rather than returned, because a walk reuses one
     *  key across every step and a text layout would otherwise allocate per element.
     */
    bool (*cursor_next)(void *cursor, key_variant_t &key, value_variant_t *value) noexcept;

    /** @brief Whether every member of @p first is also in @p second, on the same terms. */
    expected<bool> (*is_subset)(void *first, void *second) noexcept;

    /** @brief Whether @p first and @p second share no member, on the same terms. */
    expected<bool> (*is_disjoint)(void *first, void *second) noexcept;
    /**
     *  @brief Inserts only when absent, answering with the value that ended up stored.
     *
     *  The winner rather than who wrote it, because that is all @c setdefault needs and it is the one
     *  answer both stores can give: a key arriving concurrently keeps its own value, and reporting the
     *  result is what makes two threads racing on one key agree on what it holds.
     */
    expected<value_variant_t> (*insert_if_missing)(releases_t &releases, void *store, key_variant_t const &key,
                                                   value_variant_t &&value) noexcept;

    // The two bounds answer through out-parameters rather than an `expected`, which is the idiom
    // everywhere else here. They are the walk, and a walk reuses one key across every step: assigning
    // into an existing `key_variant_t` reuses its string buffer, while returning a fresh one allocates
    // once per element for the text layouts. The composite also has no honest erased type - a map's
    // element is a key and a value, a set's is a key alone. Both reasons end when the cursor comes
    // from C++ and hands its key and value back separately.

    /** @brief Erases the half-open window; a null bound is unbounded on that side. Null on an unordered core. */
    status_t (*erase_range)(releases_t &releases, void *store, key_variant_t const *lower,
                            key_variant_t const *upper) noexcept;

    /**
     *  @brief Hands every stored @c PyObject to @p visit, for the collector's traversal.
     *
     *  Null where the core cannot enumerate, or where values are never objects. A container whose
     *  slot is null reports no references, so a cycle through it is seen as reachable and leaks
     *  rather than being collected wrongly.
     */
    int (*visit_values)(void *store, visitproc visit, void *arg) noexcept;

    /** @brief Opens a transaction over @p store, handing back a pointer @c transaction_destroy owns. */
    expected<void *> (*transaction_make)(void *store) noexcept;
    void (*transaction_destroy)(releases_t &releases, void *transaction) noexcept;
    expected<bool> (*transaction_contains)(void *transaction, key_variant_t const &key) noexcept;
    expected<value_variant_t> (*transaction_find)(releases_t &releases, void *transaction,
                                                  key_variant_t const &key) noexcept;
    status_t (*transaction_upsert)(releases_t &releases, void *transaction, key_variant_t &&key,
                                   value_variant_t *value) noexcept;
    status_t (*transaction_erase)(releases_t &releases, void *transaction, key_variant_t const &key) noexcept;
    status_t (*transaction_watch)(void *transaction, key_variant_t const &key) noexcept;

    /**
     *  @brief Collects the half-open window into @p collected, in key order. Null on an unordered core.
     *
     *  A whole window in one locked span rather than a resumable cursor: a cursor over a transaction
     *  would have to stay open across arbitrary Python code, and the walk is what records the read
     *  the level is validated against, so it may not be interleaved with anything else the caller does.
     *
     *  A null bound is unbounded on that side, and @p limit at its maximum is uncounted. Elements are
     *  copied rather than referenced, because a Python object is built from them after the lock drops.
     *  A set participant leaves every @c mapped default-constructed.
     */
    status_t (*transaction_scan)(releases_t &releases, void *transaction, key_variant_t const *lower,
                                 key_variant_t const *upper, std::size_t limit,
                                 basic_vector<entry_t> &collected) noexcept;
    /** @brief Erases that same window, staging a tombstone per member. Null on an unordered core. */
    status_t (*transaction_erase_range)(releases_t &releases, void *transaction, key_variant_t const *lower,
                                        key_variant_t const *upper) noexcept;

    status_t (*transaction_stage)(releases_t &releases, void *transaction) noexcept;
    status_t (*transaction_commit)(releases_t &releases, void *transaction) noexcept;
    status_t (*transaction_rollback)(releases_t &releases, void *transaction) noexcept;
    status_t (*transaction_reset)(releases_t &releases, void *transaction) noexcept;
};

/**
 *  @brief Which core a store was built around, which is the axis that decides its method set.
 *
 *  A class per enumerator, because the core is what gates the ordered surface: a hash core supplies no
 *  ordering, so its class installs no iteration, no scan and no range erase, and the mismatch is an
 *  @c AttributeError rather than a runtime refusal.
 */
enum class core_t : std::uint8_t { sorted_k, hashed_k };

/** @brief What a reader is promised, as the constructor's @c isolation argument names it. */
enum class isolation_choice_t : std::uint8_t { monotonic_k, snapshot_k, serializable_k, strict_serializable_k };

/** @brief How a store is shared between threads, as the constructor's @c sharing argument names it. */
enum class sharing_choice_t : std::uint8_t { locked_k, partitioned_k };

/**
 *  @brief The Jepsen name of a level, which @c isolation reports and nothing else spells.
 *
 *  A switch rather than an indexed table, so adding a level to @c isolation_t is a compiler warning
 *  here rather than a silent read of the neighbouring name.
 */
constexpr char const *isolation_name_of(isolation_t level) noexcept {
    switch (level) {
    case isolation_t::read_committed_k: return "read_committed";
    case isolation_t::monotonic_atomic_view_k: return "monotonic_atomic_view";
    case isolation_t::snapshot_k: return "snapshot";
    case isolation_t::serializable_k: return "serializable";
    case isolation_t::strict_serializable_k: return "strict_serializable";
    }
    return "unknown";
}

/**
 *  @brief Resolves one configuration to its table, or reports that this build does not carry it.
 *  @return The table to use, or @c nullptr with no exception set.
 */
store_ops_t const *store_ops_for(core_t core, isolation_choice_t isolation, sharing_choice_t sharing,
                                 bool associative) noexcept;

/** @brief The ordered core's half of that resolution, defined beside the tables it names. */
store_ops_t const *sorted_store_ops_for(isolation_choice_t isolation, sharing_choice_t sharing,
                                        bool associative) noexcept;

/** @brief The unordered core's half, whose tables carry no ordered slot at all. */
store_ops_t const *hashed_store_ops_for(isolation_choice_t isolation, sharing_choice_t sharing,
                                        bool associative) noexcept;

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
 *  @brief The one layout every container class shares, so one cursor and one transaction serve all of them.
 *
 *  @c ops is the key layout this store was built around and @c store_ops the store it was built on,
 *  both resolved at construction and never null after it. @c store is the type-erased store itself,
 *  owned by this object alone and destroyed through @c store_ops->destroy. @c mode says whether values
 *  may be arbitrary objects, which decides whether the GIL may be released around one.
 *
 *  There is no per-class layout. The store used to sit inline, which forced one object type per
 *  instantiation; behind the table it is a pointer, so the four classes differ only in the method
 *  tables their types install.
 *
 *  @c ordinal is what stops two groups deadlocking on each other. A transaction stages its participants in
 *  ordinal order rather than argument order, so @c atomic(a, b) on one thread and @c atomic(b, a) on
 *  another acquire the same partition locks in the same sequence; without it each would hold what the
 *  other waits for. Any consistent total order would do - creation order is used because it is
 *  reproducible across runs, which an address is not, and a hang is the one failure worth being able
 *  to replay.
 */
struct container_object_t {
    PyObject_HEAD key_ops_t const *ops;
    store_ops_t const *store_ops;
    void *store;
    /** @brief Where values dropped under this store's lock wait; see @c releases_t. */
    mutable releases_t releases;
    std::uint64_t ordinal;
    value_mode_t mode;
};

/** @brief The module state reached from a heap type, for the constructors that have no instance. */
module_state_t *state_of_heap_type(PyTypeObject *type) noexcept;

/** @brief Whether @p object is one of this module's container classes. */
bool is_container(module_state_t *state, PyObject *object) noexcept;

/**
 *  @brief Builds an empty container of one class and layout, without re-entering the type through Python.
 *  @param[in] state The module state, which hands out the staging ordinal.
 *  @param[in] type The heap type to allocate, borrowed.
 *  @param[in] ops The key layout the store is built around.
 *  @param[in] store_ops The store table to build on, normally taken from an existing container.
 *  @param[in] mode Whether values may be arbitrary objects.
 *  @return A new reference, or @c nullptr with an exception set.
 */
PyObject *container_of(module_state_t *state, PyTypeObject *type, key_ops_t const *ops, store_ops_t const *store_ops,
                       value_mode_t mode) noexcept;

#pragma endregion Object Layouts

#pragma region Cursors

/** @brief What a cursor hands back per step. */
enum class cursor_yields_t : std::uint8_t { keys_k, values_k, items_k };

/**
 *  @brief One lazy walk in key order, driven by the store's own cursor.
 *
 *  The walk itself belongs to the store: it holds the position, decides where a bound stops it, and
 *  survives a concurrent write by re-probing rather than by holding a node. This object owns only the
 *  Python side of it - the reference keeping the store alive, how many steps are left, and which half
 *  of each element to hand back.
 *
 *  @c owner is a strong reference keeping the store alive for the walk, and is the only place the
 *  layout and the family are recorded - both are read back from it per step rather than cached here,
 *  so a cache can never disagree with the store it describes.
 */
struct cursor_object_t {
    PyObject_HEAD PyObject *owner;
    spin_shared_mutex_t lock;
    /** @brief The store's walk, closed through the same table that opened it. */
    void *walk;
    /** @brief How many steps are left, or negative when the walk is uncounted. */
    Py_ssize_t remaining;
    cursor_yields_t yields;
};

/**
 *  @brief A lazy view over a store - what @c keys, @c values and @c items return.
 *
 *  @c owner is a strong reference to the store, so a view outlives no store; the other two say
 *  which family it belongs to and what each step produces. A fresh cursor is made per iteration.
 */
struct mapping_view_object_t {
    PyObject_HEAD PyObject *owner;
    cursor_yields_t yields;
};

/**
 *  @brief Builds a cursor over a store, optionally bounded.
 *  @param[in] state The module state holding the cursor type.
 *  @param[in] container The container to walk, borrowed; a strong reference is taken.
 *  @param[in] family Which store the store holds.
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
extern PyType_Spec keys_view_spec;
extern PyType_Spec values_view_spec;
extern PyType_Spec items_view_spec;
extern PyType_Spec sorted_map_spec;
extern PyType_Spec sorted_set_spec;
extern PyType_Spec hash_map_spec;
extern PyType_Spec hash_set_spec;

/**
 *  @brief Visits every element once, in key order, stepping exactly as the Python cursor does.
 *
 *  For the whole-container operations - @c __repr__, @c __eq__, the set algebra - which need every
 *  element but have no reason to build a Python object per step. Same exclusive-successor stepping as
 *  the cursor, so the two cannot disagree about what "every element" means, and the same tolerance of
 *  concurrent change: a key erased under the walk is harmless, one inserted behind it is missed.
 *
 *  Only ever called on an ordered container, whose table carries the two bounds. A set hands the
 *  callback a value nothing wrote, since it has none.
 *
 *  @return The first refusal a step reported, so a caller folding this into an answer can tell a walk
 *    that ended from one that stopped short.
 */
template <typename callback_type_>
[[nodiscard]] status_t for_each_in_order(container_object_t const *container, callback_type_ &&callback) noexcept {
    store_ops_t const *table = container->store_ops;
    assert(table->cursor_make && "ordered walk over an unordered core");

    expected<void *> opened = table->cursor_make(container->store, nullptr, nullptr);
    if (!opened) return opened.status();

    key_variant_t found_key;
    value_variant_t found_value;
    while (table->cursor_next(*opened, found_key, &found_value)) {
        // A callback answering `bool` stops the walk when it says so; one answering `void` is asking
        // for every element, and the difference is resolved here rather than by a flag.
        if constexpr (std::is_same_v<decltype(callback(found_key, found_value)), bool>) {
            if (!callback(found_key, found_value)) break;
        }
        else { callback(found_key, found_value); }
    }
    table->cursor_destroy(container->releases, *opened);
    return success_k;
}

#pragma endregion Cursors

#pragma region Transactions

/**
 *  @brief One participant in a transaction transaction, whatever container it came from.
 *
 *  Holds the open transaction type-erased, beside the table that knows how to drive it. Every uniform
 *  operation - stage, commit, rollback, reset, erase, watch, contains - is one indirect call, with no
 *  vtable dispatch of its own, no per-participant heap allocation beyond the transaction itself, and
 *  no arm to add when the store matrix grows. The three operations that genuinely differ between a
 *  map and a set ask @c is_associative rather than which type is engaged.
 *
 *  @c mode is this participant's own, not the transaction's. Reading it from any other participant picks the
 *  wrong GIL policy for a transaction that mixes a scalar container with an object one, which is the path
 *  where a missed acquisition corrupts rather than fails.
 */
struct participant_t {
    store_ops_t const *table {nullptr};
    void *transaction {nullptr};
    key_ops_t const *ops {nullptr};
    /** @brief The ledger of the container this participates in; see @c releases_t. */
    releases_t *releases {nullptr};
    value_mode_t mode {value_mode_t::scalars_k};

    participant_t() = default;
    participant_t(store_ops_t const *table, void *transaction, key_ops_t const *ops, releases_t *releases,
                  value_mode_t mode) noexcept
        : table(table), transaction(transaction), ops(ops), releases(releases), mode(mode) {}

    participant_t(participant_t const &) = delete;
    participant_t &operator=(participant_t const &) = delete;
    participant_t(participant_t &&other) noexcept
        : table(other.table), transaction(std::exchange(other.transaction, nullptr)), ops(other.ops),
          releases(other.releases), mode(other.mode) {}
    participant_t &operator=(participant_t &&other) noexcept {
        std::swap(table, other.table);
        std::swap(transaction, other.transaction);
        std::swap(ops, other.ops);
        std::swap(releases, other.releases);
        std::swap(mode, other.mode);
        return *this;
    }
    ~participant_t() noexcept {
        if (transaction) table->transaction_destroy(*releases, transaction);
    }

    /** @brief Whether this participant stores values as well as keys. */
    [[nodiscard]] bool is_associative() const noexcept { return table->is_associative; }

    [[nodiscard]] expected<bool> contains(key_variant_t const &key) noexcept {
        return table->transaction_contains(transaction, key);
    }

    /** @brief Reads a mapped value. Only ever called on a map; callers check @c is_associative first. */
    [[nodiscard]] expected<value_variant_t> find(key_variant_t const &key) noexcept {
        assert(is_associative() && "find on a set participant; callers check is_associative first");
        return table->transaction_find(*releases, transaction, key);
    }

    /**
     *  @brief Inserts or overwrites. @p value is null for a set, which stores the key alone.
     *
     *  One verb, because it is one operation: @c insert in this library is the strict form that
     *  refuses an occupied key, and this never refuses. Only the element shape differs, and that
     *  is what the null says.
     */
    [[nodiscard]] status_t upsert(key_variant_t &&key, value_variant_t *value) noexcept {
        assert(is_associative() == (value != nullptr) && "a map upsert carries a value and a set's does not");
        return table->transaction_upsert(*releases, transaction, std::move(key), value);
    }

    [[nodiscard]] status_t erase(key_variant_t const &key) noexcept {
        return table->transaction_erase(*releases, transaction, key);
    }
    [[nodiscard]] status_t watch(key_variant_t const &key) noexcept {
        return table->transaction_watch(transaction, key);
    }

    /** @brief Whether this participant orders its keys, which is what the two windowed calls rest on. */
    [[nodiscard]] bool is_ordered() const noexcept { return table->is_ordered; }

    [[nodiscard]] status_t scan(key_variant_t const *lower, key_variant_t const *upper, std::size_t limit,
                                basic_vector<entry_t> &collected) noexcept {
        assert(is_ordered() && "scan on an unordered participant; callers check is_ordered first");
        return table->transaction_scan(*releases, transaction, lower, upper, limit, collected);
    }

    [[nodiscard]] status_t erase_range(key_variant_t const *lower, key_variant_t const *upper) noexcept {
        assert(is_ordered() && "erase_range on an unordered participant; callers check is_ordered first");
        return table->transaction_erase_range(*releases, transaction, lower, upper);
    }
    [[nodiscard]] status_t stage() noexcept { return table->transaction_stage(*releases, transaction); }
    [[nodiscard]] status_t commit() noexcept { return table->transaction_commit(*releases, transaction); }
    [[nodiscard]] status_t rollback() noexcept { return table->transaction_rollback(*releases, transaction); }
    [[nodiscard]] status_t reset() noexcept { return table->transaction_reset(*releases, transaction); }
};

/**
 *  @brief Where a transaction stands.
 *
 *  One enum rather than a pair of flags, so "staged" and "finished" cannot both be true - which two
 *  booleans prevented only through the order of two assignments.
 */
enum class group_state_t : std::uint8_t { open_k, staged_k, finished_k };

/**
 *  @brief A transaction of containers updated all-or-nothing.
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
    spin_shared_mutex_t lock;
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
    PyTypeObject *hash_map_type;
    PyTypeObject *hash_set_type;
    PyTypeObject *transaction_type;
    PyTypeObject *view_type;
    PyTypeObject *cursor_type;
    PyTypeObject *keys_view_type;
    PyTypeObject *values_view_type;
    PyTypeObject *items_view_type;
    PyObject *error;
    PyObject *conflict_error;
    PyObject *write_conflict_error;
    PyObject *read_conflict_error;
    PyObject *phantom_conflict_error;
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
 *  @param[in] mode Whether this store admits arbitrary objects or only scalars.
 *  @param[out] result Written only on success.
 *  @return True on success; false with an exception set otherwise.
 */
bool value_from_python(PyObject *object, value_mode_t mode, releases_t *releases, value_variant_t &result) noexcept;

/**
 *  @brief Reads a Python object as a key of one specific layout, rejecting every other type.
 *  @param[in] object The candidate key, borrowed.
 *  @param[in] ops The layout the store was built around.
 *  @param[out] result Written only on success.
 *  @return True on success; false with a @c TypeError or @c OverflowError set otherwise.
 */
bool key_from_python(PyObject *object, key_ops_t const *ops, key_variant_t &result) noexcept;

/**
 *  @brief Builds a new Python object from a stored key.
 *  @return A new reference, or @c nullptr with an exception set.
 */
/**
 *  @brief Reads the @c (start, stop, limit) a windowed call takes, in both spellings.
 *  @param[in] called The method's name, which every message here quotes.
 *  @param[out] start Borrowed bound object, or null; @c Py_None counts as null.
 *  @param[out] stop The same for the upper end.
 *  @param[out] limit How many elements at most, or -1 for uncounted.
 *  @return True on success; false with a @c TypeError or @c ValueError set.
 */
bool window_from_python(char const *called, PyObject *const *args, Py_ssize_t count, PyObject *keywords,
                        PyObject *&start, PyObject *&stop, Py_ssize_t &limit) noexcept;

PyObject *key_to_python(key_variant_t const &key) noexcept;

/**
 *  @brief Builds a new Python object from a stored value.
 *  @return A new reference, or @c nullptr with an exception set.
 */
PyObject *value_to_python(value_variant_t const &value) noexcept;

/**
 *  @brief Builds a @c (key, value) tuple from a stored pair, giving both halves back either way.
 *  @return A new reference, or @c nullptr with an exception set.
 */
PyObject *pair_to_python(key_variant_t const &key, value_variant_t const &value) noexcept;

#pragma endregion Conversion

} // namespace ashvardanian::smashtable::py
