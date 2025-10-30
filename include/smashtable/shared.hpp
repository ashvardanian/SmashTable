/**
 *  @brief
 *
 *  @file   shared.hpp
 *  @author Ash Vardanian
 */
#pragma once
#include <cstdint>      //
#include <system_error> // `ENOMEM`
#include <utility>      // `std::move`

namespace ashvardanian::smashtable {

/** @brief Generation type for versioned elements. */
using generation_t = std::int64_t;

/** @brief Error codes for the library. Zero is success, non-zero is failure. */
enum errc_t {
    success_k = 0,
    unknown_k = -1,

    consistency_k = 1, // Must be non-zero to indicate error!
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

    template <typename key_convertible_type_>
    constexpr explicit mapping(key_convertible_type_ &&key_arg)
        : key(std::forward<key_convertible_type_>(key_arg)), mapped() {}

    constexpr explicit operator key_type const &() const noexcept { return key; }
    constexpr explicit operator std::pair<key_type, mapped_type>() const { return {key, mapped}; }
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

/** @brief Sink type that discards assigned values. */
struct discard_t {
    template <typename type_>
    discard_t &operator=(type_ &&) noexcept {
        return *this;
    }
};

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

template <typename type_>
struct identity_fn {
    type_ &&operator()(type_ &&value) const noexcept { return std::forward<type_ &&>(value); }
    type_ const &operator()(type_ const &value) const noexcept { return value; }
};

template <>
struct identity_fn<void> {
    template <typename type_>
    decltype(auto) operator()(type_ &&value) const noexcept {
        return std::forward<type_>(value);
    }
};

using identity_fn_t = identity_fn<void>;

template <typename element_type_>
struct copy_to_fn {
    element_type_ &target;
    template <typename type_>
    void operator()(type_ &&source) const noexcept {
        target = std::forward<type_>(source);
    }
};

template <typename element_type_>
copy_to_fn<element_type_> copy_to(element_type_ &element) noexcept {
    return {element};
}

#pragma mark - Tag Dispatch Types

/**
 *  @brief Tag to enable thread-safe atomic operations.
 *    Similar to @c std::execution::par for parallel algorithms.
 *    Typically, requires @c assume_reserved (pre-allocated capacity).
 *  @see https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t
 */
struct threadsafe_t {
    explicit threadsafe_t() = default;
};
inline constexpr threadsafe_t threadsafe {};

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
 *  @brief Tag to terminate search on first slot match without probing.
 *    Optimization for when exact probe sequence doesn't matter.
 */
struct first_match_t {
    explicit first_match_t() = default;
};
inline constexpr first_match_t first_match {};

/**
 *  @brief Tag to request iterator position in return value.
 *    Similar to @c std::allocator_arg for controlling return behavior.
 *  @see https://en.cppreference.com/w/cpp/memory/allocator_arg_t
 */
struct return_position_t {
    explicit return_position_t() = default;
};
inline constexpr return_position_t return_position {};

/**
 *  @brief Tag to atomically retrieve new size after modification.
 *    Stores updated container size in provided reference.
 */
struct return_new_size_t {
    explicit return_new_size_t() = default;
};
inline constexpr return_new_size_t return_new_size {};

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

/**
 *  @brief Extract value of specific type from parameter pack or return default.
 *  @tparam needle_type_ Type to extract.
 *  @tparam default_type_ Type to return if needle not found.
 *  @tparam haystack_types_ Parameter pack to search.
 *  @return Reference to found value or default-constructed value.
 */
template <typename needle_type_, typename default_type_, typename... haystack_types_>
decltype(auto) get_value_by_type_or(haystack_types_ &&...args) {
    if constexpr ((std::is_same_v<std::decay_t<haystack_types_>, needle_type_> || ...)) {
        default_type_ result {};
        (..., (std::is_same_v<std::decay_t<haystack_types_>, needle_type_>
                   ? (result = std::forward<haystack_types_>(args), 0)
                   : 0));
        return result;
    }
    else { return default_type_ {}; }
}

/** @brief Watch metadata for versioned elements. */
struct watch_t {
    generation_t generation {0};
    bool deleted {false};

    inline bool operator==(watch_t const &watch) const noexcept {
        return watch.deleted == deleted && watch.generation == generation;
    }
    inline bool operator!=(watch_t const &watch) const noexcept {
        return watch.deleted != deleted || watch.generation != generation;
    }
};

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
                      "Type must be either nothrow copy constructible or provide a .copy() -> "
                      "expected<T> method for copy operations");
        return expected<object_type_>(object_type_ {}, status_t {errc_t::unknown_k});
    }
}

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
 *  @section Correctness of Comparisons
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
 *  @section Optimization Opportunities
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

    static_assert(!std::is_reference<value_t>(), "Only value types are supported.");
    static_assert(std::is_nothrow_copy_constructible_v<identifier_t> || has_copy_method<identifier_t>,
                  "To WATCH, the ID must be either nothrow copy constructible or provide a .copy() method returning an "
                  "expected-like type");
    static_assert(std::is_nothrow_default_constructible<value_t>(), "We need an empty state.");
    static_assert(std::is_nothrow_move_constructible<value_t>() && std::is_nothrow_move_assignable<value_t>(),
                  "To make all the methods `noexcept`, the moves must be safe too.");

    struct versioned_t {
        value_t unversioned;
        generation_t generation {0};
        bool deleted {false};
        bool visible {true};

        versioned_t() = default;
        versioned_t(versioned_t &&) noexcept = default;
        versioned_t &operator=(versioned_t &&) noexcept = default;
        versioned_t(versioned_t const &) noexcept = delete;
        versioned_t &operator=(versioned_t const &) noexcept = delete;
        versioned_t(value_t &&unversioned) noexcept : unversioned(std::move(unversioned)) {}

        bool operator==(watch_t const &watch) const noexcept {
            return watch.deleted == deleted && watch.generation == generation;
        }
        bool operator!=(watch_t const &watch) const noexcept {
            return watch.deleted != deleted || watch.generation != generation;
        }
    };

    struct versioned_comparator_t {
        using is_transparent = void;

        template <typename type_>
        decltype(auto) comparable(type_ const &object) const noexcept {
            using dereferenced_t = std::remove_reference_t<type_>;
            if constexpr (std::is_same_v<dereferenced_t, versioned_t>) return (value_t const &)object.unversioned;
            else if constexpr (is_dating_identifier_v<dereferenced_t>) return (identifier_t const &)object.id;
            else return (dereferenced_t const &)object;
        }

        template <typename first_type_, typename second_type_>
        bool dated_compare(first_type_ const &a, second_type_ const &b) const noexcept {
            comparator_t less;
            auto a_less_b = less(comparable(a), comparable(b));
            auto b_less_a = less(comparable(b), comparable(a));
            return !a_less_b && !b_less_a ? a.generation < b.generation : a_less_b;
        }

        template <typename first_type_, typename second_type_>
        bool native_compare(first_type_ const &a, second_type_ const &b) const noexcept {
            return comparator_t {}(comparable(a), comparable(b));
        }

        template <typename first_type_, typename second_type_>
        bool less(first_type_ const &a, second_type_ const &b) const noexcept {
            using first_t = std::remove_reference_t<first_type_>;
            using second_t = std::remove_reference_t<second_type_>;
            if constexpr (is_dating_identifier_v<first_t> && is_dating_identifier_v<second_t>)
                return dated_compare(a, b);
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
 *  @brief Returns an unsigned integer with only the most significant bit set.
 *    Used for creating bitmasks in bucket metadata operations.
 *  @tparam unsigned_type_ Unsigned integer type (e.g., @c std::uint32_t).
 *  @return Value with only the top bit set (e.g., 0x80000000 for 32-bit).
 */
template <typename unsigned_type_>
constexpr unsigned_type_ enabled_top_bit() noexcept {
    return static_cast<unsigned_type_>(1) << (sizeof(unsigned_type_) * 8 - 1);
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

#pragma mark - Tree Concepts

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

/**
 *  @brief Concept to detect if a node type supports order statistics operations.
 *
 *  This is the node-level version of @c supports_order_statistics for static methods.
 *
 *  @tparam node_type_ The node type to check.
 */
template <typename node_type_>
concept node_supports_order_statistics =
    requires(node_type_ *node, std::size_t k, typename node_type_::value_t const &value,
             typename node_type_::comparator_t const &comp) {
        { node_type_::select(node, k, comp) } -> std::convertible_to<node_type_ *>;
        { node_type_::rank(node, value, comp) } -> std::same_as<std::size_t>;
    };

} // namespace ashvardanian::smashtable
