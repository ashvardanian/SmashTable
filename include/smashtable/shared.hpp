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
 *  @brief Simple key-value association type, cleaner & lighter than @c std::pair used by @c std::map.
 *  @see https://en.cppreference.com/w/cpp/utility/pair.html
 */
template <typename key_type_, typename value_type_>
struct association {
    using key_type = key_type_;
    using value_type = value_type_;

    key_type key {};
    value_type value {};

    constexpr association() = default;
    constexpr association(association const &) = default;
    constexpr association(association &&) noexcept = default;
    constexpr association &operator=(association const &) = default;
    constexpr association &operator=(association &&) noexcept = default;

    template <typename key_arg_, typename value_arg_>
    constexpr association(key_arg_ &&key_arg, value_arg_ &&value_arg)
        : key(std::forward<key_arg_>(key_arg)), value(std::forward<value_arg_>(value_arg)) {}

    template <typename key_arg_>
    constexpr explicit association(key_arg_ &&key_arg) : key(std::forward<key_arg_>(key_arg)), value() {}

    constexpr explicit operator key_type const &() const noexcept { return key; }

    constexpr explicit operator std::pair<key_type, value_type>() const { return {key, value}; }
};

/**
 *  @brief Concept to detect if a type is an association (has key_type and value_type members).
 */
template <typename type_>
concept is_association = requires {
    typename type_::key_type;
    typename type_::value_type;
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

struct no_op_t {
    constexpr void operator()() const noexcept {}
    template <typename type_>
    constexpr void operator()(type_ &&) const noexcept {}
};

struct identity_fn_t {
    template <typename type_>
    decltype(auto) operator()(type_ &&value) const noexcept {
        return std::forward<type_>(value);
    }
};

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
 *  @brief Decorates an element type with generation and visibility metadata for transactional containers.
 *
 *  @par Template requirements
 *  - @p element_type_ must be a value type (no references), nothrow default-constructible, and nothrow move
 *    constructible/assignable so that transactional staging can remain noexcept.
 *  - @p comparator_type_ must expose @c value_type describing the identifier used for ordering, and that identifier
 *    type must be nothrow copy-constructible to support watch bookkeeping.
 *  - @p element_type_ must be convertible to the identifier type and constructible from it, enabling the containers to
 *    extract keys for lookups and manufacture key-only tombstones for erases.
 */
template <typename element_type_, typename comparator_type_>
struct versioning_for {

    using element_t = element_type_;
    using comparator_t = comparator_type_;
    using identifier_t = typename comparator_t::value_type;
    using generation_t = ashvardanian::smashtable::generation_t;
    using watch_t = ashvardanian::smashtable::watch_t;

    static_assert(!std::is_reference<element_t>(), "Only value types are supported.");
    static_assert(std::is_nothrow_copy_constructible<identifier_t>(), "To WATCH, the ID must be safe to copy.");
    static_assert(std::is_nothrow_default_constructible<element_t>(), "We need an empty state.");
    static_assert(std::is_nothrow_move_constructible<element_t>() && std::is_nothrow_move_assignable<element_t>(),
                  "To make all the methods `noexcept`, the moves must be safe too.");

    struct dated_identifier_t {
        identifier_t id;
        generation_t generation {0};
    };

    struct watched_identifier_t {
        identifier_t id;
        watch_t watch;
    };

    struct versioned_entry_t {
        element_t element;
        generation_t generation {0};
        bool deleted {false};
        bool visible {true};

        versioned_entry_t() = default;
        versioned_entry_t(versioned_entry_t &&) noexcept = default;
        versioned_entry_t &operator=(versioned_entry_t &&) noexcept = default;
        versioned_entry_t(versioned_entry_t const &) noexcept = delete;
        versioned_entry_t &operator=(versioned_entry_t const &) noexcept = delete;
        versioned_entry_t(element_t &&element) noexcept : element(std::move(element)) {}

        operator element_t const &() const & noexcept { return element; }
        bool operator==(watch_t const &watch) const noexcept {
            return watch.deleted == deleted && watch.generation == generation;
        }
        bool operator!=(watch_t const &watch) const noexcept {
            return watch.deleted != deleted || watch.generation != generation;
        }
    };

    template <typename type_>
    constexpr static bool knows_generation() {
        using dereferenced_t = std::remove_reference_t<type_>;
        return std::is_same<dereferenced_t, versioned_entry_t>() || std::is_same<dereferenced_t, dated_identifier_t>();
    }

    struct versioned_comparator_t {
        using is_transparent = void;

        template <typename type_>
        decltype(auto) comparable(type_ const &object) const noexcept {
            using dereferenced_t = std::remove_reference_t<type_>;
            if constexpr (std::is_same<dereferenced_t, versioned_entry_t>()) return (element_t const &)object.element;
            else if constexpr (std::is_same<dereferenced_t, dated_identifier_t>())
                return (identifier_t const &)object.id;
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
            if constexpr (knows_generation<first_t>() && knows_generation<second_t>()) return dated_compare(a, b);
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

} // namespace ashvardanian::smashtable
