/**
 *  @brief Vocabulary shared by every container - status codes, @c expected, key-value @c mapping, the versioning
 *      machinery transactions are built on, and the concepts stating what a container must offer to back one.
 *  @author Ash Vardanian
 *  @file include/smashtable/shared.hpp
 *  @date October 13, 2022
 *
 *  @section shared_store_tiers Store Tiers
 *
 *  What @c transactional_store needs from its parameter is narrower than tree-ness. The concepts near
 *  the end of this file state that requirement, so a container which does not meet it says so at the
 *  call site rather than deep inside the adapter.
 *
 *  The requirements fall into two tiers, and the split is the point. Key-addressed work - staging a
 *  change, validating a watch, committing, reading one key - needs only point lookup and a
 *  transparent way to address an entry. Ordered work - bounds, ranges, order statistics - needs a
 *  total order over keys, which an open-addressed table cannot supply at any price.
 *
 *  Stating the tiers separately is what lets one adapter serve an ordered and an unordered core,
 *  gating the ordered surface behind the second concept rather than behind the container's name.
 */
#pragma once
#include <cerrno>  // `ENOMEM`, `EINVAL`, and the rest of the errno space
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::int64_t`

#include <atomic>      // `std::atomic_ref`
#include <array>       // `std::array`
#include <bit>         // `std::countl_zero`
#include <concepts>    // `std::convertible_to`, `std::same_as`
#include <optional>    // `std::optional`
#include <tuple>       // `std::tuple`
#include <type_traits> // `std::is_nothrow_invocable_v`
#include <utility>     // `std::move`, `std::index_sequence`

namespace ashvardanian::smashtable {

/** @brief Generation type for versioned elements. */
using generation_t = std::int64_t;

/**
 *  @brief The generation no stored entry can carry, standing for a key that is not there.
 *
 *  Generations are handed out pre-incremented, so the first is 1 and zero is free to mean absence.
 *  A watch on an absent key must be anchored to this rather than to the watching transaction's own
 *  generation, which @c rollback reassigns - otherwise the watch stops matching itself.
 */
inline constexpr generation_t absent_generation_k = 0;

/** @brief Error codes for the library. Zero is success, non-zero is failure. */
enum errc_t {
    success_k = 0,
    unknown_k = -1,

    consistency_k = -2, // ? Not an errno; the value 1 would collide with `EPERM`
    transaction_not_recoverable_k = ENOTRECOVERABLE,
    sequence_number_overflow_k = EOVERFLOW,

    out_of_memory_heap_k = ENOMEM,
    out_of_memory_arena_k = ENOBUFS,
    out_of_memory_disk_k = ENOSPC,

    invalid_argument_k = EINVAL,
    operation_in_progress_k = EINPROGRESS,
    operation_not_permitted_k = EPERM,
    operation_not_supported_k = EOPNOTSUPP,
    operation_would_block_k = EWOULDBLOCK,
    operation_canceled_k = ECANCELED,

    key_already_exists_k = EEXIST, // For `insert_if_missing` conflicts
    key_not_found_k = ENOENT,      // For `update` operations on missing keys

    connection_broken_k = EPIPE,
    connection_aborted_k = ECONNABORTED,
    connection_already_in_progress_k = EALREADY,
    connection_refused_k = ECONNREFUSED,
    connection_reset_k = ECONNRESET,

};

/**
 *  @brief Wraps error-codes into bool-convertible conditions.
 *  @see @c errc_t.
 */
struct status_t {
    errc_t errc = errc_t::success_k;
    constexpr operator bool() const noexcept { return errc == errc_t::success_k; }
};

/**
 *  @brief Simple key-value mapping type, cleaner & lighter than @c std::pair used by @c std::map.
 *  @see https://en.cppreference.com/w/cpp/utility/pair.html
 */
template <typename value_type_>
struct expected;

template <typename object_type_>
[[nodiscard]] expected<object_type_> copy_safely(object_type_ const &object) noexcept;

template <typename key_type_, typename value_type_>
struct mapping {
    using key_type = key_type_;
    using mapped_type = value_type_;

    key_type key {};
    mapped_type mapped {};

    constexpr mapping() = default;
    constexpr mapping(mapping const &) = default;
    constexpr mapping(mapping &&) noexcept = default;
    constexpr mapping &operator=(mapping const &) = default;
    constexpr mapping &operator=(mapping &&) noexcept = default;

    template <typename key_convertible_type_, typename mapped_convertible_type_>
    constexpr mapping(key_convertible_type_ &&key_arg, mapped_convertible_type_ &&value_arg)
        : key(std::forward<key_convertible_type_>(key_arg)), mapped(std::forward<mapped_convertible_type_>(value_arg)) {
    }

    // Excluding `mapping` itself matters: for a non-const lvalue the unconstrained template deduces
    // `mapping &` and outbids the copy constructor, which takes `mapping const &`.
    template <typename key_convertible_type_>
        requires(!std::is_same_v<std::remove_cvref_t<key_convertible_type_>, mapping>)
    constexpr explicit mapping(key_convertible_type_ &&key_arg)
        : key(std::forward<key_convertible_type_>(key_arg)), mapped() {}

    constexpr explicit operator key_type const &() const noexcept { return key; }
    constexpr explicit operator std::pair<key_type, mapped_type>() const { return {key, mapped}; }

    /** @brief Deep-copies both halves, so a mapping is copyable exactly when its members are. */
    expected<mapping> copy() const noexcept {
        auto copied_key = copy_safely(key);
        if (!copied_key) return expected<mapping>(mapping {}, copied_key.status);
        auto copied_mapped = copy_safely(mapped);
        if (!copied_mapped) return expected<mapping>(mapping {}, copied_mapped.status);
        return expected<mapping>(mapping {std::move(*copied_key), std::move(*copied_mapped)}, status_t {success_k});
    }
};

/**
 *  @brief Concept to detect if a type is an mapping (has key_type and mapped_type members).
 */
template <typename type_>
concept is_mapping = requires {
    typename std::remove_cvref_t<type_>::key_type;
    typename std::remove_cvref_t<type_>::mapped_type;
};

/**
 *  @brief Conditional callback validation traits controlled by SMASHTABLE_STRICT_CALLBACK_CHECKS.
 *
 *  When SMASHTABLE_STRICT_CALLBACK_CHECKS is defined, these traits perform full compile-time
 *  validation using std::is_nothrow_invocable_v. This catches type errors early but prevents
 *  generic lambdas like [](auto const&) noexcept {} from compiling.
 *
 *  When undefined (default), traits always return true, allowing generic lambdas while still
 *  documenting the intent that callbacks should be noexcept.
 */
#ifdef SMASHTABLE_STRICT_CALLBACK_CHECKS
template <typename callback_type_, typename... args_types_>
inline constexpr bool is_safe_callback_for = std::is_nothrow_invocable_v<callback_type_ &, args_types_...>;

template <typename callback_type_>
inline constexpr bool is_safe_callback = std::is_nothrow_invocable_v<callback_type_ &>;
#else
template <typename callback_type_, typename... args_types_>
inline constexpr bool is_safe_callback_for = true;

template <typename callback_type_>
inline constexpr bool is_safe_callback = true;
#endif

/** @brief Sentinel type for range-based iteration end conditions. */
struct end_sentinel_t {};

/** @brief Placeholder type for template conditional where no type is needed. */
struct placeholder_t {};

template <typename type_>
struct no_op_fn {
    constexpr void operator()(type_ &&) const noexcept {}
    constexpr void operator()(type_ const &) const noexcept {}
};

template <>
struct no_op_fn<void> {
    template <typename type_>
    constexpr void operator()(type_ &&) const noexcept {}
    constexpr void operator()() const noexcept {}
};

using no_op_fn_t = no_op_fn<void>;

/** @brief Callback that stores whatever it is handed into a caller-owned destination. */
template <typename element_type_>
struct copy_to_fn {
    element_type_ &target;
    template <typename type_>
    void operator()(type_ &&source) const noexcept {
        target = std::forward<type_>(source);
    }
};

/** @brief Builds a @c copy_to_fn over @p element, for a read whose result the caller wants kept. */
template <typename element_type_>
copy_to_fn<element_type_> copy_to(element_type_ &element) noexcept {
    return {element};
}

#pragma region Tag Dispatch Types

/**
 *  @brief Tag to assume capacity is pre-allocated, skip null checks.
 *    Similar to @c std::adopt_lock for assuming preconditions are met.
 *  @see https://en.cppreference.com/w/cpp/thread/lock_tag_t
 */
struct assume_reserved_t {
    explicit assume_reserved_t() = default;
};
inline constexpr assume_reserved_t assume_reserved {};

/**
 *  @brief Tag to assume elements are unique, skip equality comparisons.
 *    Similar to Boost.Container's @c ordered_unique_range_t.
 */
struct assume_unique_t {
    explicit assume_unique_t() = default;
};
inline constexpr assume_unique_t assume_unique {};

/**
 *  @brief Tag to assume input range is sorted, enable O(n) bulk construction.
 *    Allows building balanced tree from sorted range without individual insertions.
 *    Precondition: Elements must be in sorted order according to comparator.
 *  @warning If precondition violated (unsorted input), behavior is undefined.
 */
struct assume_sorted_t {
    explicit assume_sorted_t() = default;
};
inline constexpr assume_sorted_t assume_sorted {};

/**
 *  @brief Compile-time check if a type appears in parameter pack.
 *  @tparam needle_type_ Type to search for.
 *  @tparam haystack_types_ Parameter pack to search in.
 *  @return True if @p needle_type_ found in @p haystack_types_.
 */
template <typename needle_type_, typename... haystack_types_>
consteval bool contains_type() {
    return (std::is_same_v<needle_type_, haystack_types_> || ...);
}

#pragma endregion Tag Dispatch Types

#pragma region Copying and Construction

/**
 *  @brief Alternative to C++ 23 @c std::expected and C++ 17 @c std::optional with error code.
 *    Wraps a @c noexcept default-constructible @c value_type_, without all the complexity of
 *    implementing a @c union -based uninitialized storage state.
 */
template <typename value_type_>
struct expected {
    using value_t = value_type_;
    using value_type = value_t; // ? STL style

    static_assert(std::is_nothrow_default_constructible_v<value_t>,
                  "expected<T> requires T to be nothrow default-constructible");

    // Named `outcome` to avoid ambiguity with generic terms like `value` or `entry` during debugging.
    value_t outcome;
    status_t status;

    expected() = default;
    expected(value_t &&e, status_t s = status_t {}) noexcept : outcome(std::move(e)), status(s) {}
    expected(value_t const &e, status_t s = status_t {}) noexcept : outcome(e), status(s) {}
    expected(status_t s) noexcept : outcome(), status(s) {}

    expected(std::optional<value_t> const &opt) noexcept
        : outcome(opt.value_or(value_t {})), status(opt.has_value() ? status_t {success_k} : status_t {unknown_k}) {}

    expected(std::optional<value_t> &&opt) noexcept
        : outcome(opt.has_value() ? std::move(*opt) : value_t {}),
          status(opt.has_value() ? status_t {success_k} : status_t {unknown_k}) {}

    /**
     *  @brief Checks if the expected contains a successful value.
     *  @return @c true if status is success, @c false otherwise.
     */
    explicit operator bool() const noexcept { return status; }
    bool has_value() const noexcept { return status; }

    /**
     *  @brief Accesses the contained value (like @c std::optional).
     *  @return Reference to the outcome.
     */
    value_t &operator*() & noexcept { return outcome; }

    /**
     *  @brief Accesses the contained value (like @c std::optional, const version).
     *  @return Const reference to the outcome.
     */
    value_t const &operator*() const & noexcept { return outcome; }

    /**
     *  @brief Accesses the contained value (like @c std::optional, rvalue version).
     *  @return Rvalue reference to the outcome.
     */
    value_t &&operator*() && noexcept { return std::move(outcome); }

    /**
     *  @brief Member access operator (like @c std::optional).
     *  @return Pointer to the outcome.
     */
    value_t *operator->() noexcept { return &outcome; }

    /**
     *  @brief Member access operator (like @c std::optional, const version).
     *  @return Const pointer to the outcome.
     */
    value_t const *operator->() const noexcept { return &outcome; }
};

/**
 *  @brief Concept checking if a type has a @c .copy() method returning @c expected<T>.
 *    This allows non-nothrow-copyable types to participate in safe copy operations.
 */
template <typename type_>
concept has_copy_method = requires(type_ const &t) {
    { t.copy() } noexcept -> std::same_as<expected<type_>>;
};

/**
 *  @brief Concept checking if a type has a static @c .make() method returning @c expected<T>.
 *    This allows types with potentially throwing constructors to participate in safe construction.
 */
template <typename type_, typename... args_types_>
concept has_make_method = requires(args_types_ &&...args) {
    { type_::make(std::forward<args_types_>(args)...) } noexcept -> std::same_as<expected<type_>>;
};

/**
 *  @brief Helper to copy an object, using nothrow copy if available, otherwise @c .copy() method.
 *  @return @c expected<object_type_> containing the copy and status.
 *    On success, status is @c success_k. On failure, returns default-constructed object with error status.
 */
template <typename object_type_>
[[nodiscard]] expected<object_type_> copy_safely(object_type_ const &obj) noexcept {
    static_assert(std::is_nothrow_default_constructible_v<object_type_>,
                  "Type must be nothrow default-constructible to use with expected<T>");

    // Fast path: nothrow copy
    if constexpr (std::is_nothrow_copy_constructible_v<object_type_>)
        return expected<object_type_>(object_type_ {obj}, status_t {success_k});

    // Fallback: use .copy() method
    else if constexpr (has_copy_method<object_type_>) return obj.copy();

    else {
        // Type doesn't support safe copying
        static_assert(std::is_nothrow_copy_constructible_v<object_type_> || has_copy_method<object_type_>,
                      "Type must be nothrow copy constructible or provide .copy() returning an expected");
        return expected<object_type_>(object_type_ {}, status_t {errc_t::unknown_k});
    }
}

#pragma endregion Copying and Construction

#pragma region Key Extraction

/**
 *  @brief Checks if a type has a @c comparable_particle member.
 *
 *  @code{cpp}
 *  struct with_particle { using comparable_particle = int; };
 *  struct without_particle {};
 *  static_assert(has_comparable_particle<with_particle>::value, "Should have comparable_particle");
 *  static_assert(!has_comparable_particle<without_particle>::value, "Should not have comparable_particle");
 *  @endcode
 */
template <typename, typename = void>
struct has_comparable_particle : std::false_type {};

template <typename value_type_>
struct has_comparable_particle<value_type_, std::void_t<typename value_type_::comparable_particle>> : std::true_type {};

/**
 *  @brief If @c can_be_mapping_type_ is a mapping, exposes its @c key_type as @c ::type.
 *    Otherwise, exposes @c can_be_mapping_type_ itself as @c ::type.
 *
 *  @code{cpp}
 *  using pair = mapping<int, std::string>;
 *  static_assert(std::is_same_v<mapping_key_type_or_itself<pair>::type, int>, "Must be int");
 *  static_assert(std::is_same_v<mapping_key_type_or_itself<double>::type, double>, "Must be double");
 *  @endcode
 */
template <typename can_be_mapping_type_, bool = is_mapping<can_be_mapping_type_>>
struct mapping_key_type_or_itself {
    using type = can_be_mapping_type_;
};

template <typename can_be_mapping_type_>
struct mapping_key_type_or_itself<can_be_mapping_type_, true> {
    using type = typename can_be_mapping_type_::key_type;
};

/**
 *  @brief Helper method to extract mapping key or return the object itself.
 *    Useful for generic code that works with both mappings and simple keys.
 */
template <typename can_be_mapping_type_>
typename mapping_key_type_or_itself<can_be_mapping_type_>::type const & //
mapping_key_or_itself(can_be_mapping_type_ const &can_be_mapping) noexcept {
    if constexpr (is_mapping<can_be_mapping_type_>) return can_be_mapping.key;
    else return can_be_mapping;
}

/**
 *  @brief If @c can_be_mapping_type_ is a mapping, exposes its @c mapped_type as @c ::type.
 *    Otherwise, exposes @c void as @c ::type.
 *
 *  @code{cpp}
 *  using pair = mapping<int, std::string>;
 *  static_assert(std::is_same_v<mapped_value_type_or_void<pair>::type, std::string>, "Must be string");
 *  static_assert(std::is_same_v<mapped_value_type_or_void<double>::type, void>, "Must be void");
 *  @endcode
 */
template <typename can_be_mapping_type_, bool = is_mapping<can_be_mapping_type_>>
struct mapped_value_type_or_void {
    using type = void;
};

template <typename can_be_mapping_type_>
struct mapped_value_type_or_void<can_be_mapping_type_, true> {
    using type = typename can_be_mapping_type_::mapped_type;
};

/**
 *  @section shared_correctness_of_comparisons Correctness of Comparisons
 *
 *  For @c std::set, @c std::map, or similar containers we only require the keys to provide
 *  @b strict-weak-ordering comparisons. It's enough to simply define a comparator that can
 *  returns a boolean result for two comparable objects. Since C++ 20 comparisons are better
 *  formalized:
 *
 *  - @c std::strong_ordering - equivalent values are fully interchangeable, everything is comparable!
 *  - @c std::weak_ordering - equivalent values may be non-interchangeable, but everything is comparable!
 *  - @c std::partial_ordering - equivalent values may be non-interchangeable, and may be incomparable!
 *
 *  Assuming that @c float values can be @c NaN and thus incomparable, we shouldn't be able
 *  to construct a @c std::set<float> since the ordering is only partial, but GCC still compiles it.
 *
 *  @section shared_optimization_opportunities Optimization Opportunities
 *
 *  Oftentimes, when dealing with heavy objects as keys (e.g., strings, composite structures),
 *  we want to avoid storing many copies of the full object just to perform comparisons and lookups.
 *  Instead, we can distill a lightweight "particle" from the heavy object that captures
 *  its identity for ordering purposes. Examples may be:
 *
 *  - Non-owning @c std::string_view for @c std::string keys
 *  - Integer IDs extracted from composite structures
 *
 *  If the @c comparator_type_ has a @c value_type member - indicating some form of a unique
 *  identifier can be distilled from @c comparable_type_ - we expose it via @c ::type.
 *  Else, if the @c comparable_type_ is a @c mapping, we use its @c key_type as the identifier.
 *  Else, we use the @c comparable_type_ itself as the identifier.
 */
template <typename comparator_type_, typename comparable_type_, bool = has_comparable_particle<comparator_type_>::value>
struct comparable_particle_of {
    using type = typename mapping_key_type_or_itself<comparable_type_>::type;
};

template <typename comparator_type_, typename comparable_type_>
struct comparable_particle_of<comparator_type_, comparable_type_, true> {
    using type = typename comparator_type_::comparable_particle;
};

#pragma endregion Key Extraction

#pragma region Versioning Machinery

/**
 *  @brief What a reader is promised, named as Jepsen names it and ordered by strength.
 *  @see https://jepsen.io/consistency
 */
enum class isolation_t : std::uint8_t {
    read_uncommitted_k = 0,
    read_committed_k = 1,
    monotonic_atomic_view_k = 2,
    snapshot_k = 3,
    serializable_k = 4,
};

/**
 *  @brief Whether a transaction's writes are sitting in the store, invisible, or not there yet.
 *
 *  A committed transaction returns to @c pending_k and is immediately reusable, so there is no third
 *  state to name - @c commit on a @c pending_k transaction already reports @c operation_not_permitted_k.
 */
enum class staging_t : bool {
    /** @brief Accepting writes; the store holds nothing of this transaction. */
    pending_k,
    /** @brief The store has reserved every change, so @c commit cannot fail. */
    staged_k,
};

/** @brief Whether an entry says its key is there, or says it was taken away and when. */
enum class presence_t : bool {
    /** @brief The key is there and this is its value. */
    present_k,
    /** @brief The key was erased. The stamp survives so a watch can date the erasure. */
    erased_k,
};

/** @brief Whether a reader may see an entry, or a live transaction still owns it. */
enum class publication_t : bool {
    /** @brief Written or reserved by a transaction that has not committed. No reader sees it. */
    staged_k,
    /** @brief Committed. This is what a reader reads. */
    published_k,
};

/** @brief Watch metadata for versioned elements. */
struct watch_t {
    generation_t generation {0};
    presence_t presence {presence_t::present_k};

    inline bool operator==(watch_t const &watch) const noexcept {
        return watch.presence == presence && watch.generation == generation;
    }
    inline bool operator!=(watch_t const &watch) const noexcept {
        return watch.presence != presence || watch.generation != generation;
    }
};

template <typename identifier_type_>
struct dated_identifier {
    using identifier_t = identifier_type_;

    identifier_t id;
    generation_t generation {0};
};

template <typename identifier_type_>
struct watched_identifier {
    using identifier_t = identifier_type_;

    identifier_t id;
    watch_t watch;
};

template <typename type_>
struct is_dating_identifier : public std::false_type {};

template <typename identifier_type_>
struct is_dating_identifier<dated_identifier<identifier_type_>> : public std::true_type {};

template <typename identifier_type_>
struct is_dating_identifier<watched_identifier<identifier_type_>> : public std::true_type {};

template <typename type_>
inline constexpr bool is_dating_identifier_v = is_dating_identifier<type_>::value;

/**
 *  @brief Detects operands that carry a generation of their own, so ordering can separate versions.
 *    A @c watched_identifier keeps its generation inside @c watch, so it is ordered by key alone.
 */
template <typename type_>
concept carries_generation = requires(type_ const &value) {
    { value.generation } -> std::convertible_to<generation_t>;
};

/**
 *  @brief Detects a version-decorated value: a generation stamp @b and the payload it decorates.
 *    Stricter than @c carries_generation on purpose - @c watch_t carries a generation and no
 *    payload, and would otherwise be peeled down a branch that does not compile.
 */
template <typename type_>
concept carries_versioned_payload = carries_generation<type_> && requires(type_ const &value) { value.unversioned; };

/**
 *  @brief The identifier a version-decorated object is addressed by, with the metadata peeled off.
 *    A chain defers to the version it holds, a version to the value it wraps, and a dated identifier
 *    to the identifier inside it, so a plain key is reached from any of the shapes a store stores.
 */
template <typename type_>
decltype(auto) versioned_particle(type_ const &object) noexcept {
    using dereferenced_t = std::remove_reference_t<type_>;
    if constexpr (requires { typename dereferenced_t::is_version_chain; }) return versioned_particle(object.head);
    else if constexpr (is_dating_identifier_v<dereferenced_t>)
        return (typename dereferenced_t::identifier_t const &)object.id;
    else if constexpr (carries_versioned_payload<dereferenced_t>) return versioned_particle(object.unversioned);
    else return mapping_key_or_itself(object);
}

/**
 *  @brief Decorates a value type with generation and visibility metadata for transactional containers.
 *
 *  @par Template requirements
 *  - @p value_type_ must be a value type (no references), nothrow default-constructible, and nothrow move
 *    constructible/assignable so that transactional staging can remain noexcept.
 *  - @p comparator_type_ must expose @c value_type describing the identifier used for ordering, and that identifier
 *    type must be nothrow copy-constructible to support watch bookkeeping.
 *  - @p value_type_ must be convertible to the identifier type and constructible from it, enabling the containers to
 *    extract keys for lookups and manufacture key-only tombstones for erases.
 */
template <typename value_type_, typename comparator_type_>
struct versioning_for {

    using value_t = value_type_;
    using value_type = value_t; // ? STL style

    using comparator_t = comparator_type_;
    using comparator_type = comparator_t; // ? STL style

    using identifier_t = typename comparable_particle_of<comparator_t, value_t>::type;
    using identifier_type = identifier_t; // ? STL style

    using generation_t = ashvardanian::smashtable::generation_t;
    using generation_type = generation_t; // ? STL style

    using watch_t = ashvardanian::smashtable::watch_t;
    using watch_type = watch_t; // ? STL style

    using dated_identifier_t = dated_identifier<identifier_t>;
    using watched_identifier_t = watched_identifier<identifier_t>;

    static_assert(!std::is_reference<value_t>(), "Only value types are supported.");
    static_assert(std::is_nothrow_copy_constructible_v<identifier_t> || has_copy_method<identifier_t>,
                  "To WATCH, the ID must be nothrow copy constructible or provide .copy()");
    static_assert(std::is_nothrow_default_constructible<value_t>(), "We need an empty state.");
    static_assert(std::is_nothrow_move_constructible<value_t>() && std::is_nothrow_move_assignable<value_t>(),
                  "To make all the methods `noexcept`, the moves must be safe too.");

    struct versioned_t {
        value_t unversioned;
        generation_t generation {0};
        presence_t presence {presence_t::present_k};
        publication_t publication {publication_t::published_k};

        versioned_t() = default;
        versioned_t(versioned_t &&) noexcept = default;
        versioned_t &operator=(versioned_t &&) noexcept = default;
        versioned_t(versioned_t const &) noexcept = delete;
        versioned_t &operator=(versioned_t const &) noexcept = delete;
        versioned_t(value_t &&unversioned) noexcept : unversioned(std::move(unversioned)) {}

        bool operator==(watch_t const &watch) const noexcept {
            return watch.presence == presence && watch.generation == generation;
        }
        bool operator!=(watch_t const &watch) const noexcept {
            return watch.presence != presence || watch.generation != generation;
        }
    };

    using versioned_entry_t = versioned_t;

    struct versioned_comparator_t {
        using is_transparent = void;

        /**
         *  @brief The comparator this wrapper was built with, consulted for every comparison.
         *
         *  Held rather than manufactured on demand: a comparator that carries state - a direction flag, a
         *  collation table, a dispatch pointer - answers differently from a default-constructed one, so
         *  building a fresh instance per comparison silently discards whatever the container was given.
         */
        [[no_unique_address]] comparator_t comparator;

        // Constrained rather than defaulted, so a comparator that refuses default construction - one
        // carrying a dispatch pointer with no meaningful empty value - makes every site that tried to
        // manufacture one a compile error instead of a null call in a branch nobody exercises.
        versioned_comparator_t() noexcept
            requires std::is_default_constructible_v<comparator_t>
            : comparator() {}
        explicit versioned_comparator_t(comparator_t const &other) noexcept : comparator(other) {}

        /**
         *  @brief Peels a stored object down to the identifier it orders by.
         *  @note One projection, shared with the unordered path, so an ordered and an unordered
         *    store can never disagree about which part of an entry identifies it.
         */
        template <typename type_>
        decltype(auto) comparable(type_ const &object) const noexcept {
            return versioned_particle(object);
        }

        template <typename first_type_, typename second_type_>
        bool dated_compare(first_type_ const &a, second_type_ const &b) const noexcept {
            auto a_less_b = comparator(comparable(a), comparable(b));
            auto b_less_a = comparator(comparable(b), comparable(a));
            return !a_less_b && !b_less_a ? a.generation < b.generation : a_less_b;
        }

        template <typename first_type_, typename second_type_>
        bool native_compare(first_type_ const &a, second_type_ const &b) const noexcept {
            return comparator(comparable(a), comparable(b));
        }

        template <typename first_type_, typename second_type_>
        bool less(first_type_ const &a, second_type_ const &b) const noexcept {
            using first_t = std::remove_reference_t<first_type_>;
            using second_t = std::remove_reference_t<second_type_>;
            if constexpr (carries_generation<first_t> && carries_generation<second_t>) return dated_compare(a, b);
            else return native_compare(a, b);
        }

        template <typename first_type_, typename second_type_>
        bool operator()(first_type_ const &a, second_type_ const &b) const noexcept {
            return less(a, b);
        }

        template <typename first_type_, typename second_type_>
        bool same(first_type_ const &a, second_type_ const &b) const noexcept {
            return !less(a, b) && !less(b, a);
        }
    };
};

/**
 *  @brief Hashes a version-decorated object by the identifier inside it, ignoring the metadata.
 *    An unordered store addresses an entry by hash rather than by order, so this is what
 *    @c versioned_comparator_t is to an ordered one.
 */
template <typename hasher_type_>
struct versioned_hasher {
    using is_transparent = void;

    /** @brief The hasher this wrapper was built with, consulted for every element. */
    [[no_unique_address]] hasher_type_ hasher;

    // Constrained rather than defaulted, matching the ordered comparator: a hasher carrying a seed
    // with no meaningful empty value makes every site that tried to manufacture one a compile error.
    versioned_hasher() noexcept
        requires std::is_default_constructible_v<hasher_type_>
        : hasher() {}
    explicit versioned_hasher(hasher_type_ const &other) noexcept : hasher(other) {}

    template <typename type_>
    std::size_t operator()(type_ const &object) const noexcept {
        return hasher(versioned_particle(object));
    }
};

/**
 *  @brief Compares version-decorated objects by the identifier inside them, ignoring the metadata.
 *    Paired with @c versioned_hasher, since an open-addressed table needs both to place an entry.
 */
template <typename equals_type_>
struct versioned_equals {
    using is_transparent = void;

    /** @brief The equality this wrapper was built with, consulted for every comparison. */
    [[no_unique_address]] equals_type_ equals;

    versioned_equals() noexcept
        requires std::is_default_constructible_v<equals_type_>
        : equals() {}
    explicit versioned_equals(equals_type_ const &other) noexcept : equals(other) {}

    template <typename first_type_, typename second_type_>
    bool operator()(first_type_ const &first, second_type_ const &second) const noexcept {
        return equals(versioned_particle(first), versioned_particle(second));
    }
};

#pragma endregion Versioning Machinery

#pragma region Numeric Helpers

/**
 *  @brief Relaxed atomic increment of a plain counter, returning the post-increment value.
 *    Relaxed suffices for a counter read by value - a statistic, or a version stamp compared for
 *    identity and recency - because one location has a total modification order. It publishes
 *    nothing: a caller needing the data a counter describes to be visible must order that itself.
 */
template <typename integral_type_>
integral_type_ atomic_add_fetch(integral_type_ &counter, integral_type_ addend) noexcept {
    return std::atomic_ref<integral_type_>(counter).fetch_add(addend, std::memory_order_relaxed) + addend;
}

/**
 *  @brief Relaxed atomic read of a counter other threads may be incrementing.
 *    Needed wherever a plain read would race an @c atomic_add_fetch on the same field.
 */
template <typename integral_type_>
integral_type_ atomic_load(integral_type_ const &counter) noexcept {
    return std::atomic_ref<integral_type_>(const_cast<integral_type_ &>(counter)).load(std::memory_order_relaxed);
}

/** @brief Relaxed atomic decrement of a plain counter, returning the post-decrement value. */
template <typename integral_type_>
integral_type_ atomic_sub_fetch(integral_type_ &counter, integral_type_ subtrahend) noexcept {
    return std::atomic_ref<integral_type_>(counter).fetch_sub(subtrahend, std::memory_order_relaxed) - subtrahend;
}

/**
 *  @brief Rounds up an integer to the next power of two.
 *    Returns 0 for input 0, and 1 for input 1.
 *  @param[in] x Value to round up.
 *  @return Smallest power of two greater than or equal to @p x.
 */
constexpr std::size_t roundup_to_pow2(std::size_t x) noexcept {
    if (x <= 1) return x;
    return std::size_t {1} << (64 - std::countl_zero(x - 1));
}

/**
 *  @brief Rounds up a value to the next multiple of a compile-time constant.
 *  @tparam value_type_ Type of value to round (must be integral).
 *  @tparam multiple_ The multiple to round up to (compile-time constant).
 *  @param[in] x Value to round up.
 *  @return Smallest multiple of @p multiple_ greater than or equal to @p x.
 */
template <typename value_type_, value_type_ multiple_>
constexpr value_type_ roundup_to_multiple(value_type_ x) noexcept {
    return ((x + multiple_ - 1) / multiple_) * multiple_;
}

#pragma endregion Numeric Helpers

#pragma region Tree Concepts

/**
 *  @brief Concept to detect if a tree type supports order statistics operations.
 *
 *  Order statistics allow O(log n) access to the k-th smallest element (select)
 *  and finding the rank (position) of an element. Weight-balanced trees support
 *  these operations, while standard AVL trees do not.
 *
 *  @tparam tree_type_ The tree type to check.
 */
template <typename tree_type_>
concept supports_order_statistics =
    requires(tree_type_ const &tree, std::size_t k, typename tree_type_::value_t const &value) {
        { tree.select(k) } -> std::convertible_to<typename tree_type_::node_t const *>;
        { tree.rank(value) } -> std::same_as<std::size_t>;
    };

#pragma endregion Tree Concepts

#pragma region Container Traits

/**
 *  @brief A container that reports whether it maps keys to values, and how reads are delivered.
 *    Every container in this library carries these, and the test suites assert on them.
 */
template <typename collection_type_>
concept tagged_collection = requires {
    typename collection_type_::value_type;
    typename collection_type_::key_type;
    typename collection_type_::is_associative;
    { collection_type_::is_associative::value } -> std::convertible_to<bool>;
};

#pragma endregion Container Traits

#pragma region Store Tiers

/**
 *  @brief The tier every associative core must satisfy to back a transactional store.
 *    Point lookup, membership, size, and clearing - nothing here implies an ordering over keys, so
 *    an open-addressed table can meet it as readily as a search tree.
 *
 *  @note Node transfer - @c extract and @c merge - is deliberately absent. Moving nodes between a
 *    transaction's staging store and the committed one is how the tree adapter stages today, but it
 *    is an implementation strategy rather than a requirement: a chain-based store stages by pushing
 *    a version onto a key's chain and never moves a node at all.
 */
template <typename collection_type_>
concept key_addressable_collection =
    tagged_collection<collection_type_> && requires(collection_type_ &collection, collection_type_ const &constant,
                                                    typename collection_type_::key_type const &key) {
        { constant.size() } -> std::convertible_to<std::size_t>;
        { constant.empty() } -> std::convertible_to<bool>;
        { constant.contains(key) } -> std::convertible_to<bool>;
        collection.clear();
    };

/**
 *  @brief The additional tier an ordered core satisfies, and an unordered one never can.
 *    Bounds and half-open range erasure are the load-bearing pair: the transactional store expresses
 *    "every version of this key older than mine" as a range, and the Python binding walks a cursor
 *    by repeatedly asking for the exclusive successor of the last key it yielded.
 *
 *  @note A core's bounds are iterator-shaped, taking the sought key alone. The callback-shaped
 *    @c lower_bound(key, found, missing) belongs to the transactional adapter above it, which cannot
 *    hand out iterators because a version may be staged and invisible. Do not conflate the two.
 */
template <typename collection_type_>
concept ordered_collection = key_addressable_collection<collection_type_> &&
                             requires(collection_type_ &collection, typename collection_type_::key_type const &key) {
                                 collection.lower_bound(key);
                                 collection.upper_bound(key);
                                 collection.erase_range(key, key);
                             };

#pragma endregion Store Tiers

#pragma region Storage Shape

/**
 *  @brief How a container family restates itself over version-decorated elements, and how one such
 *    entry is reached, replaced and taken back out.
 *
 *  An ordered core addresses an entry by a comparator and rebinds on one; an unordered core
 *  addresses it by a hasher paired with an equality and rebinds on both. Everything a transactional
 *  store does that is not itself ordered - staging, committing, reading one key - goes through this,
 *  so the store never names either shape.
 *
 *  @tparam collection_type_ The core container family being restated.
 *  @tparam value_type_ The undecorated element the store holds.
 */
template <typename collection_type_, typename value_type_, typename = void>
struct versioned_storage_for {

    using versioning_t = versioning_for<value_type_, typename collection_type_::comparator_t>;
    using addressing_source_t = typename collection_type_::comparator_t;
    using addressing_t = typename versioning_t::versioned_comparator_t;

    template <typename element_type_>
    using rebind = typename collection_type_::template rebind<element_type_, addressing_t>;

    /** @brief Builds an empty storage over @c element_type_, seeded from a bare comparator. */
    template <typename element_type_, typename allocator_type_>
    static rebind<element_type_> build(addressing_source_t const &source, allocator_type_ const &allocator) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(addressing_t(source), typename storage_t::allocator_type(allocator));
    }

    /** @brief Builds an empty storage over @c element_type_, leaving the addressing defaulted. */
    template <typename element_type_, typename allocator_type_>
    static rebind<element_type_> build(allocator_type_ const &allocator) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(typename storage_t::allocator_type(allocator));
    }

    /** @brief Builds an empty storage over @c element_type_ addressed exactly as @p existing is. */
    template <typename element_type_, typename storage_type_>
    static rebind<element_type_> sibling(storage_type_ const &existing) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(existing.key_comp(), typename storage_t::allocator_type(existing.allocator()));
    }

    /** @brief The allocator a storage of this family was built with. */
    template <typename storage_type_>
    static decltype(auto) allocator_of(storage_type_ const &storage) noexcept {
        return storage.allocator();
    }

    /** @brief A node-based core has nothing to prepay; the placeholder pass reserves it. */
    template <typename storage_type_>
    static status_t prepare(storage_type_ &, std::size_t) noexcept {
        return status_t {success_k};
    }

    /** @brief Files @p element under its own key, overwriting whatever shared it. */
    template <typename storage_type_, typename element_type_>
    static status_t upsert(storage_type_ &storage, element_type_ &&element) noexcept {
        auto result = storage.upsert(std::move(element));
        return result.failed() ? status_t {out_of_memory_heap_k} : status_t {success_k};
    }

    /**
     *  @brief Takes the entry filed under @p identifier out whole, handing its payload to @p destination.
     *  @return Whether an entry was there to take.
     *  @note The entry leaves the index before anything is moved out of it, since the key the index
     *    addresses it by lives inside the payload and a moved-from key neither compares nor hashes.
     */
    template <typename storage_type_, typename identifier_type_, typename payload_type_>
    static bool extract_payload(storage_type_ &storage, identifier_type_ const &identifier,
                                payload_type_ &destination) noexcept {
        auto extracted = storage.extract(identifier);
        if (!extracted.node_ptr_) return false;
        destination = std::move(extracted.node_ptr_->fruit.head);
        return true;
    }
};

/**
 *  @brief The unordered shape, selected by the core naming a @c hasher the way every table does.
 */
template <typename collection_type_, typename value_type_>
struct versioned_storage_for<collection_type_, value_type_, std::void_t<typename collection_type_::hasher>> {

    using versioning_t = versioning_for<value_type_, typename collection_type_::key_equal>;
    using addressing_source_t = typename collection_type_::key_equal;
    using addressing_t = versioned_hasher<typename collection_type_::hasher>;
    using equality_t = versioned_equals<typename collection_type_::key_equal>;

    template <typename element_type_>
    using rebind = typename collection_type_::template rebind<element_type_, addressing_t, equality_t>;

    /** @brief Builds an empty storage over @c element_type_, seeded from a bare equality. */
    template <typename element_type_, typename allocator_type_>
    static rebind<element_type_> build(addressing_source_t const &source, allocator_type_ const &allocator) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(addressing_t {}, equality_t(source), typename storage_t::allocator_type(allocator));
    }

    /** @brief Builds an empty storage over @c element_type_, leaving the addressing defaulted. */
    template <typename element_type_, typename allocator_type_>
    static rebind<element_type_> build(allocator_type_ const &allocator) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(addressing_t {}, equality_t {}, typename storage_t::allocator_type(allocator));
    }

    /** @brief Builds an empty storage over @c element_type_ addressed exactly as @p existing is. */
    template <typename element_type_, typename storage_type_>
    static rebind<element_type_> sibling(storage_type_ const &existing) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(existing.hash_function(), existing.key_eq(),
                         typename storage_t::allocator_type(existing.get_allocator()));
    }

    /** @brief The allocator a storage of this family was built with. */
    template <typename storage_type_>
    static decltype(auto) allocator_of(storage_type_ const &storage) noexcept {
        return storage.get_allocator();
    }

    /** @brief A slab grows once, so every later upsert lands in a slot already paid for. */
    template <typename storage_type_>
    static status_t prepare(storage_type_ &storage, std::size_t count) noexcept {
        using slot_count_t = typename storage_type_::size_type;
        using reserve_result_t = typename storage_type_::reserve_result_t;
        auto const grown = storage.reserve_more(static_cast<slot_count_t>(count));
        return grown == reserve_result_t::failed_k ? status_t {out_of_memory_heap_k} : status_t {success_k};
    }

    /** @brief Files @p element under its own key, overwriting whatever shared it. */
    template <typename storage_type_, typename element_type_>
    static status_t upsert(storage_type_ &storage, element_type_ &&element) noexcept {
        return storage.upsert(std::move(element));
    }

    /**
     *  @brief Takes the entry filed under @p identifier out whole, handing its payload to @p destination.
     *  @return Whether an entry was there to take.
     *  @note The slot is located first and erased through that position, so the erasure never probes
     *    for a key the harvest has already emptied.
     */
    template <typename storage_type_, typename identifier_type_, typename payload_type_>
    static bool extract_payload(storage_type_ &storage, identifier_type_ const &identifier,
                                payload_type_ &destination) noexcept {
        auto found = storage.find(identifier);
        if (found == storage.end()) return false;
        using element_t = std::remove_const_t<std::remove_reference_t<decltype(*found)>>;
        destination = std::move(const_cast<element_t &>(*found).head);
        storage.erase(found);
        return true;
    }
};

/**
 *  @brief The owned element a core stores, as distinct from the view it hands out.
 *    A node-based container returns a reference to the element it holds, so the two coincide; an
 *    open-addressed table keeps keys and values in separate regions and can only view an entry as a
 *    pair of references, which nothing above it can own, copy, or version.
 */
template <typename collection_type_, typename = void>
struct owned_value_of {
    using type = typename collection_type_::value_type;
};

template <typename collection_type_>
struct owned_value_of<collection_type_, std::void_t<typename collection_type_::owned_value_type>> {
    using type = typename collection_type_::owned_value_type;
};

/**
 *  @brief The node type a linked storage hands out, or @c void where the storage has no nodes.
 *    An ordered surface walks nodes directly; an open-addressed table has none to walk, and naming
 *    one unconditionally would break the adapter at its own definition rather than at the call.
 */
template <typename collection_type_, typename = void>
struct storage_node_of {
    using type = void;
};

template <typename collection_type_>
struct storage_node_of<collection_type_, std::void_t<typename collection_type_::node_t>> {
    using type = typename collection_type_::node_t;
};

#pragma endregion Storage Shape

} // namespace ashvardanian::smashtable
