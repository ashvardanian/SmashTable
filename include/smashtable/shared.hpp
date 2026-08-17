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
 *
 *  @section shared_device_code Reaching Device Code
 *
 *  Nothing here is annotated @c __host__ @c __device__. A function reachable from a kernel is marked
 *  @c constexpr instead, which @c nvcc @c --expt-relaxed-constexpr makes callable from device code,
 *  so one unannotated definition serves both sides and the headers stay compilable by a host compiler
 *  that has never heard of CUDA. Marking a body @c constexpr that no argument could ever constant-
 *  evaluate - anything touching an atomic or a placement @c new - is only ill-formed for a
 *  non-template function, and every such body here belongs to a template.
 */
#pragma once
#include <cerrno>  // `ENOMEM`, `EINVAL`, and the rest of the errno space
#include <climits> // `CHAR_BIT`
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::int64_t`

#include <atomic>      // `std::atomic_ref`
#include <array>       // `std::array`
#include <bit>         // `std::countl_zero`
#include <concepts>    // `std::convertible_to`, `std::same_as`
#include <new>         // `::operator new`, `std::align_val_t`, `std::nothrow`
#include <tuple>       // `std::tuple`
#include <type_traits> // `std::is_nothrow_invocable_v`
#include <utility>     // `std::move`, `std::index_sequence`

#if defined(__CUDACC__)
#include <cuda/atomic>  // `cuda::atomic_ref`
#include <cuda/std/bit> // `cuda::std::popcount`, `cuda::std::countr_zero`
#endif

/**
 *  @brief Lets an empty member occupy no space, in the spelling the compiler at hand honours.
 *
 *  MSVC ignores the standard attribute for ABI reasons and offers its own, while Clang rejects an
 *  unknown scoped attribute under @c -Werror - so neither spelling can simply be written twice.
 */
#if defined(_MSC_VER)
#define ST_NO_UNIQUE_ADDRESS_ [[msvc::no_unique_address]]
#else
#define ST_NO_UNIQUE_ADDRESS_ [[no_unique_address]]
#endif

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

/**
 *  @brief Whether an operation succeeded, and why it did not. The library's only result code.
 *
 *  Values follow @c errno where one fits, so a failure can be handed to @c strerror or mapped onto a
 *  platform error without a translation table. @c success_k is zero, which is why the type is scoped
 *  and truth-testing goes through @c succeeded: an implicit conversion would read every success as
 *  false. Prefer checking an @c expected, which already answers this question.
 */
enum class status_t : int {
    success_k = 0,
    unknown_k = -1,

    consistency_k = -2, // ? Not an errno; the value 1 would collide with `EPERM`

    out_of_memory_heap_k = ENOMEM,

    invalid_argument_k = EINVAL,
    operation_not_permitted_k = EPERM,
    operation_would_block_k = EWOULDBLOCK, // For a bounded retry that gives up

    key_already_exists_k = EEXIST, // For `insert_if_missing` conflicts
    key_not_found_k = ENOENT,      // For `update` operations on missing keys
};

// Unqualified `success_k` reads better than the scope at the hundred sites that name it.
using enum status_t;

/** @brief Whether @p status reports success, which no implicit conversion can be trusted to say. */
constexpr bool succeeded(status_t status) noexcept { return status == status_t::success_k; }

/** @brief Whether @p status reports a failure, and so carries a reason. */
constexpr bool failed(status_t status) noexcept { return status != status_t::success_k; }

/**
 *  @brief The same question of a richer result - one that reports its own success, like the node
 *    handles the trees hand back - so generic code can ask it without knowing which it holds.
 */
template <typename result_type_>
    requires requires(result_type_ const &result) { result.failed(); }
constexpr bool succeeded(result_type_ const &result) noexcept {
    return !result.failed();
}

template <typename result_type_>
    requires requires(result_type_ const &result) { result.failed(); }
constexpr bool failed(result_type_ const &result) noexcept {
    return result.failed();
}

/**
 *  @brief Simple key-value mapping type, cleaner & lighter than @c std::pair used by @c std::map.
 *  @see https://en.cppreference.com/w/cpp/utility/pair.html
 */
template <typename value_type_>
class expected;

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
        if (!copied_key) return copied_key.status();
        auto copied_mapped = copy_safely(mapped);
        if (!copied_mapped) return copied_mapped.status();
        return expected<mapping>(mapping {std::move(*copied_key), std::move(*copied_mapped)}, success_k);
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

/**
 *  @brief The allocator every container defaults to, over @c ::operator @c new asked not to throw.
 *
 *  @c std::allocator would serve, and costs @c <memory> - some fifty thousand preprocessed lines - in
 *  headers that want nothing else from it. Exhaustion is reported by returning null, which is the
 *  shape every caller here already checks, rather than by an exception nothing would catch.
 */
template <typename value_type_>
struct default_allocator {
    using value_type = value_type_;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type;

    constexpr default_allocator() noexcept = default;
    template <typename other_type_>
    constexpr default_allocator(default_allocator<other_type_> const &) noexcept {}

    [[nodiscard]] value_type_ *allocate(std::size_t count) noexcept {
        std::size_t const bytes = count * sizeof(value_type_);
        if constexpr (alignof(value_type_) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
            return static_cast<value_type_ *>(
                ::operator new(bytes, std::align_val_t {alignof(value_type_)}, std::nothrow)); // allocator primitive
        else return static_cast<value_type_ *>(::operator new(bytes, std::nothrow));           // allocator primitive
    }

    void deallocate(value_type_ *pointer, std::size_t) noexcept {
        if constexpr (alignof(value_type_) > __STDCPP_DEFAULT_NEW_ALIGNMENT__)
            ::operator delete(pointer, std::align_val_t {alignof(value_type_)}, std::nothrow); // allocator primitive
        else ::operator delete(pointer, std::nothrow);                                         // allocator primitive
    }

    template <typename other_type_>
    constexpr bool operator==(default_allocator<other_type_> const &) const noexcept {
        return true;
    }
};

/**
 *  @brief Transparent ordering and equality, so a container needs no @c <functional>.
 *
 *  Both are heterogeneous - the @c is_transparent tag is what lets a lookup compare a key against
 *  something merely comparable to it, without materializing a key first - which is the only property
 *  the containers here relied on @c std::less and @c std::equal_to for.
 */
struct less_t {
    using is_transparent = void;
    template <typename left_type_, typename right_type_>
    constexpr auto operator()(left_type_ const &left, right_type_ const &right) const noexcept {
        return left < right;
    }
};

struct equal_to_t {
    using is_transparent = void;
    template <typename left_type_, typename right_type_>
    constexpr auto operator()(left_type_ const &left, right_type_ const &right) const noexcept {
        return left == right;
    }
};

/**
 *  @brief The hash a container uses unless told otherwise. Specialize it for your own key.
 *
 *  Deliberately not a forward to @c std::hash: that would drag @c <functional> into every header, and
 *  the point of a named trait is that an unhashable key says so at compile time. Hashing the bytes of
 *  an arbitrary type instead would be worse than an error - padding, or an equality that ignores some
 *  field, would give two equal keys different hashes and quietly lose lookups.
 */
template <typename key_type_, typename = void>
struct hash {
    static_assert(sizeof(key_type_) == 0,
                  "No hash for this key. Specialize ashvardanian::smashtable::hash<key_type_> for it.");
};

/** @brief Integers and enumerations, mixed so that adjacent keys do not land in adjacent buckets. */
template <typename key_type_>
struct hash<key_type_, std::enable_if_t<std::is_integral_v<key_type_> || std::is_enum_v<key_type_>>> {
    constexpr std::size_t operator()(key_type_ key) const noexcept {
        // splitmix64's finalizer: a bijection, so distinct keys stay distinct.
        std::uint64_t mixed = static_cast<std::uint64_t>(key) + 0x9E3779B97F4A7C15ull;
        mixed = (mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9ull;
        mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EBull;
        return static_cast<std::size_t>(mixed ^ (mixed >> 31));
    }
};

/** @brief A pointer hashes as the integer it is. */
template <typename key_type_>
struct hash<key_type_, std::enable_if_t<std::is_pointer_v<key_type_>>> {
    std::size_t operator()(key_type_ key) const noexcept {
        return hash<std::uintptr_t> {}(reinterpret_cast<std::uintptr_t>(key));
    }
};

/** @brief Anything exposing contiguous bytes through @c data and @c size, which covers the string types. */
template <typename key_type_>
struct hash<key_type_, std::void_t<decltype(std::declval<key_type_ const &>().data()),
                                   decltype(std::declval<key_type_ const &>().size())>> {
    std::size_t operator()(key_type_ const &key) const noexcept {
        // FNV-1a over the bytes: no table, no header, and adequate for bucket selection.
        auto const *bytes = reinterpret_cast<unsigned char const *>(key.data());
        std::size_t const count = key.size() * sizeof(*key.data());
        std::uint64_t accumulated = 0xCBF29CE484222325ull;
        for (std::size_t index = 0; index != count; ++index)
            accumulated = (accumulated ^ bytes[index]) * 0x100000001B3ull;
        return static_cast<std::size_t>(accumulated);
    }
};

/** @brief How many steps separate two iterators, without the sixteen thousand lines of @c <iterator>. */
template <typename iterator_type_>
constexpr std::size_t distance_between(iterator_type_ first, iterator_type_ last) noexcept {
    std::size_t count = 0;
    for (; first != last; ++first) ++count;
    return count;
}

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
 *  @brief A value that may not be there, and the reason when it is not - the library's only result.
 *
 *  Storage is a @c union, so the value exists only on success and @p value_type_ needs no default
 *  constructor. That is what lets one type cover both what @c std::optional carried and what a
 *  status carried, including the move-only transaction types that have no empty state to sit in.
 *  A default-constructed @c expected reports @b failure, since there is nothing in it.
 *
 *  Decomposes as @c auto @c [value, status], for value types that can be default-constructed to
 *  hand something back on the failure path; see @c get.
 */
template <typename value_type_>
class expected {
  public:
    using value_t = value_type_;
    using value_type = value_t; // ? STL style

    static_assert(std::is_nothrow_move_constructible_v<value_t>,
                  "expected<T> moves its value into place, so T must move without throwing");

  private:
    union {
        value_t outcome_; // ? Alive only while `status_` says so
    };
    status_t status_ = unknown_k;

  public:
    constexpr expected() noexcept {}
    constexpr expected(status_t status) noexcept : status_(status) {}
    constexpr expected(value_t &&value, status_t status = status_t {}) noexcept
        : outcome_(std::move(value)), status_(status) {}
    constexpr expected(value_t const &value, status_t status = status_t {}) noexcept
        requires(std::is_copy_constructible_v<value_t>)
        : outcome_(value), status_(status) {}

    expected(expected &&other) noexcept : status_(other.status_) {
        if (succeeded(status_)) new (&outcome_) value_t(std::move(other.outcome_));
    }
    expected &operator=(expected &&other) noexcept {
        if (this == &other) return *this;
        if (succeeded(status_)) outcome_.~value_t();
        status_ = other.status_;
        if (succeeded(status_)) new (&outcome_) value_t(std::move(other.outcome_));
        return *this;
    }
    ~expected() noexcept {
        if (succeeded(status_)) outcome_.~value_t();
    }

    expected(expected const &) = delete;
    expected &operator=(expected const &) = delete;

    /** @brief Whether the value is there, which is the only safe precondition for reading it. */
    constexpr explicit operator bool() const noexcept { return succeeded(status_); }
    constexpr bool has_value() const noexcept { return succeeded(status_); }

    /** @brief Why there is no value, or success when there is one. */
    constexpr status_t status() const noexcept { return status_; }

    /** @warning Reading the value of a failed @c expected is undefined - check it first. */
    constexpr value_t &operator*() & noexcept { return outcome_; }
    constexpr value_t const &operator*() const & noexcept { return outcome_; }
    constexpr value_t &&operator*() && noexcept { return std::move(outcome_); }
    constexpr value_t *operator->() noexcept { return &outcome_; }
    constexpr value_t const *operator->() const noexcept { return &outcome_; }

    /**
     *  @brief The tuple protocol behind @c auto @c [value, status], returning the value @b by value.
     *
     *  A reference would bind into storage that was never constructed when the status is a failure,
     *  which reads as garbage and trips no sanitizer, so the failure path default-constructs instead.
     *  Offered only where that is possible: a transaction cannot be decomposed, and has nothing worth
     *  binding when it fails anyway, so callers check it and dereference.
     */
    template <std::size_t index_>
        requires(std::is_nothrow_default_constructible_v<value_t>)
    constexpr auto get() && noexcept {
        if constexpr (index_ == 0) return succeeded(status_) ? value_t {std::move(outcome_)} : value_t {};
        else return status_;
    }

    template <std::size_t index_>
        requires(std::is_nothrow_default_constructible_v<value_t> && std::is_copy_constructible_v<value_t>)
    constexpr auto get() const & noexcept {
        if constexpr (index_ == 0) return succeeded(status_) ? value_t {outcome_} : value_t {};
        else return status_;
    }
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
 *  @brief Copies @p object through its @c noexcept copy constructor, or its @c copy method.
 *  @return The copy on success, or the reason it could not be made.
 */
template <typename object_type_>
[[nodiscard]] expected<object_type_> copy_safely(object_type_ const &object) noexcept {
    if constexpr (std::is_nothrow_copy_constructible_v<object_type_>)
        return expected<object_type_>(object_type_ {object}, success_k);
    else if constexpr (has_copy_method<object_type_>) return object.copy();
    else {
        static_assert(std::is_nothrow_copy_constructible_v<object_type_> || has_copy_method<object_type_>,
                      "Type must be nothrow copy constructible or provide .copy() returning an expected");
        return status_t::unknown_k;
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
        ST_NO_UNIQUE_ADDRESS_ comparator_t comparator;

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
    ST_NO_UNIQUE_ADDRESS_ hasher_type_ hasher;

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
    ST_NO_UNIQUE_ADDRESS_ equals_type_ equals;

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

#pragma region Device Portability

/**
 *  @brief The bit intrinsics every container here counts slots with.
 *
 *  @warning These are aliases rather than direct calls to @c std because libstdc++'s @c <bit> is not
 *    usable from device code. Its functions are @c constexpr, so @c --expt-relaxed-constexpr lets a
 *    kernel call them and they compile without a diagnostic, but @c nvcc folds most of them against a
 *    zeroed argument: with CUDA 12.8 and GCC 14, @c std::popcount over 64 bits, @c std::countr_zero
 *    over either width and @c std::countl_zero over 64 bits each return the answer for an input of
 *    zero. Silently wrong results, not a compile error, which is what makes the alias worth having.
 */
#if defined(__CUDACC__)

using ::cuda::std::countl_zero;
using ::cuda::std::countr_zero;
using ::cuda::std::popcount;

#else

using std::countl_zero;
using std::countr_zero;
using std::popcount;

#endif

/**
 *  @brief The atomic vocabulary every lock-free path in this library is written against.
 *
 *  Under a device compiler this resolves to @c cuda::atomic_ref at device scope, which is the widest
 *  scope a kernel spanning several blocks needs and the only one under which a header touched from
 *  two blocks is ordered at all. Everywhere else it is @c std::atomic_ref, and the two agree on the
 *  operations used here - @c fetch_or, @c fetch_xor, @c fetch_add, @c fetch_sub and @c load.
 *
 *  The memory orders travel as named constants because the CUDA enumeration is a distinct type from
 *  @c std::memory_order, so the call sites cannot spell either one directly.
 */
#if defined(__CUDACC__)

template <typename scalar_type_>
using atomic_ref = ::cuda::atomic_ref<scalar_type_, ::cuda::thread_scope_device>;

inline constexpr auto memory_order_relaxed_k = ::cuda::std::memory_order_relaxed;
inline constexpr auto memory_order_acquire_k = ::cuda::std::memory_order_acquire;
inline constexpr auto memory_order_release_k = ::cuda::std::memory_order_release;

#else

template <typename scalar_type_>
using atomic_ref = std::atomic_ref<scalar_type_>;

inline constexpr auto memory_order_relaxed_k = std::memory_order_relaxed;
inline constexpr auto memory_order_acquire_k = std::memory_order_acquire;
inline constexpr auto memory_order_release_k = std::memory_order_release;

#endif

/**
 *  @brief The alignment @c atomic_ref demands of a counter updated through it.
 *
 *  A 64-bit integer is only 4-byte aligned by default on a 32-bit target, while @c atomic_ref needs
 *  8, so a counter that is perfectly well-behaved on x86-64 becomes undefined when built for i386.
 *  Every counter the helpers below touch carries this.
 */
template <typename integral_type_>
inline constexpr std::size_t atomic_alignment = atomic_ref<integral_type_>::required_alignment;

/**
 *  @brief Relaxed atomic increment of a plain counter, returning the post-increment value.
 *    Relaxed suffices for a counter read by value - a statistic, or a version stamp compared for
 *    identity and recency - because one location has a total modification order. It publishes
 *    nothing: a caller needing the data a counter describes to be visible must order that itself.
 */
template <typename integral_type_>
constexpr integral_type_ atomic_add_fetch(integral_type_ &counter, integral_type_ addend) noexcept {
    return atomic_ref<integral_type_>(counter).fetch_add(addend, memory_order_relaxed_k) + addend;
}

/**
 *  @brief Relaxed atomic read of a counter other threads may be incrementing.
 *    Needed wherever a plain read would race an @c atomic_add_fetch on the same field.
 */
template <typename integral_type_>
constexpr integral_type_ atomic_load(integral_type_ const &counter) noexcept {
    return atomic_ref<integral_type_>(const_cast<integral_type_ &>(counter)).load(memory_order_relaxed_k);
}

/** @brief Relaxed atomic decrement of a plain counter, returning the post-decrement value. */
template <typename integral_type_>
constexpr integral_type_ atomic_sub_fetch(integral_type_ &counter, integral_type_ subtrahend) noexcept {
    return atomic_ref<integral_type_>(counter).fetch_sub(subtrahend, memory_order_relaxed_k) - subtrahend;
}

#pragma endregion Device Portability

#pragma region Numeric Helpers

/**
 *  @brief The larger of two values, and the smaller.
 *
 *  Spelled here rather than pulled from @c <algorithm>, which costs about ten thousand preprocessed
 *  lines for these two. It also sidesteps the Windows headers, which define @c min and @c max as
 *  function-like macros unless @c NOMINMAX is set, and would eat an unparenthesized @c std::max call.
 */
template <typename value_type_>
constexpr value_type_ const &larger_of(value_type_ const &first, value_type_ const &second) noexcept {
    return first < second ? second : first;
}

template <typename value_type_>
constexpr value_type_ const &smaller_of(value_type_ const &first, value_type_ const &second) noexcept {
    return second < first ? second : first;
}

/** @brief How many bits a @c std::size_t holds here, which is not 64 everywhere. */
inline constexpr std::size_t size_bits_k = sizeof(std::size_t) * CHAR_BIT;

/** @brief The largest @c std::size_t, which also serves as the "no such index" sentinel. */
inline constexpr std::size_t size_max_k = static_cast<std::size_t>(-1);

/** @brief The number of bits needed to represent @p x, so @c 0 for zero and @c 1 for one. */
constexpr std::size_t bits_to_hold(std::size_t x) noexcept {
    return size_bits_k - static_cast<std::size_t>(countl_zero(x));
}

/**
 *  @brief Rounds up an integer to the next power of two.
 *    Returns 0 for input 0, and 1 for input 1.
 *  @param[in] x Value to round up.
 *  @return Smallest power of two greater than or equal to @p x, or 0 when there is no such power.
 */
constexpr std::size_t roundup_to_pow2(std::size_t x) noexcept {
    if (x <= 1) return x;
    std::size_t const shift = bits_to_hold(x - 1);
    // Anything above the largest representable power of two has no next one to round up to, and
    // shifting by the full width is undefined rather than saturating.
    if (shift >= size_bits_k) return 0;
    return std::size_t {1} << shift;
}

/**
 *  @brief A uniform draw below @p bound, without the bias a modulo would introduce.
 *
 *  Masks to the next power of two and redraws when the value lands above the bound, which is unbiased
 *  because every value under the mask is equally likely and the surplus is simply discarded. At least
 *  half of the masked range is accepted, so fewer than two draws are expected.
 *
 *  @param[in] generator Any uniform random bit generator, which is what @c std::mt19937 and friends are.
 *  @param[in] bound One past the largest value that may be returned; zero and one both draw nothing.
 *  @note A generator narrower than the bound - @c std::mt19937 yields 32 bits - is called repeatedly
 *    and the results concatenated, since masking a short draw could never reach the high bits.
 */
template <typename generator_type_>
constexpr std::size_t draw_below(generator_type_ &&generator, std::size_t bound) noexcept {
    using generator_t = std::remove_reference_t<generator_type_>;
    if (bound <= 1) return 0;

    constexpr std::size_t generator_span_k = static_cast<std::size_t>(generator_t::max() - generator_t::min());
    constexpr std::size_t generator_bits_k = bits_to_hold(generator_span_k);
    std::size_t const mask = roundup_to_pow2(bound) - 1;
    std::size_t const wanted_bits = bits_to_hold(mask);

    while (true) {
        std::size_t draw = static_cast<std::size_t>(generator() - generator_t::min());
        if constexpr (generator_bits_k < size_bits_k)
            for (std::size_t filled = generator_bits_k; filled < wanted_bits; filled += generator_bits_k)
                draw = (draw << generator_bits_k) | static_cast<std::size_t>(generator() - generator_t::min());
        draw &= mask;
        if (draw < bound) return draw;
    }
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
        return success_k;
    }

    /** @brief Files @p element under its own key, overwriting whatever shared it. */
    template <typename storage_type_, typename element_type_>
    static status_t upsert(storage_type_ &storage, element_type_ &&element) noexcept {
        auto result = storage.upsert(std::move(element));
        return result.failed() ? out_of_memory_heap_k : success_k;
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
        return grown == reserve_result_t::failed_k ? out_of_memory_heap_k : success_k;
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

#pragma region Transaction Group

/**
 *  @brief A two-phase commit over several stores at once.
 *
 *  @tparam store_types_ The stores taking part. Each must expose a nested @c transaction_t and a
 *    @c transaction() factory, which every transactional container in this library does.
 */
template <typename... store_types_>
class transaction_group {
  public:
    static constexpr std::size_t participants_k = sizeof...(store_types_);
    static_assert(participants_k > 0, "a group needs at least one participant");

    using transactions_t = std::tuple<typename store_types_::transaction_t...>;

  private:
    transactions_t transactions_;
    /** @brief Positions into @c transactions_, ordered by the address of the store each belongs to. */
    std::array<std::size_t, participants_k> order_ {};
    staging_t staging_ {staging_t::pending_k};

    static_assert((std::is_nothrow_move_constructible_v<typename store_types_::transaction_t> && ...),
                  "a group moves its participants into place, so each must move without throwing");

    explicit transaction_group(transactions_t &&opened, std::array<void const *, participants_k> const &stores) noexcept
        : transactions_(std::move(opened)) {
        for (std::size_t position = 0; position != participants_k; ++position) order_[position] = position;
        // Insertion sort: the participant count is a handful, and this keeps the header free of
        // `<algorithm>` for a sort that will never see more than a few elements.
        for (std::size_t position = 1; position != participants_k; ++position) {
            std::size_t const carried = order_[position];
            std::size_t scan = position;
            while (scan != 0 && stores[order_[scan - 1]] > stores[carried]) {
                order_[scan] = order_[scan - 1];
                --scan;
            }
            order_[scan] = carried;
        }
    }

    /**
     *  @brief Applies @p visitor to the participant sitting at @p position of @c transactions_.
     *    A fold over the pack rather than a jump table, since the count is known and tiny.
     */
    template <typename visitor_type_, std::size_t... indices_>
    status_t visit_at_(std::size_t position, visitor_type_ &&visitor, std::index_sequence<indices_...>) noexcept {
        status_t result = success_k;
        ((indices_ == position ? (void)(result = visitor(std::get<indices_>(transactions_))) : (void)0), ...);
        return result;
    }

    template <typename visitor_type_>
    status_t visit_at_(std::size_t position, visitor_type_ &&visitor) noexcept {
        return visit_at_(position, std::forward<visitor_type_>(visitor), std::make_index_sequence<participants_k> {});
    }

  public:
    transaction_group(transaction_group &&) noexcept = default;
    transaction_group &operator=(transaction_group &&) noexcept = default;
    transaction_group(transaction_group const &) = delete;
    transaction_group &operator=(transaction_group const &) = delete;

    /**
     *  @brief Opens one transaction per store, or none at all.
     *    A store that refuses leaves the already-opened transactions to their destructors, which is
     *    why they must unwind themselves.
     */
    [[nodiscard]] static expected<transaction_group> make(store_types_ &...stores) noexcept {
        std::tuple<expected<typename store_types_::transaction_t>...> opened {stores.transaction()...};
        bool const all_opened =
            std::apply([](auto const &...maybe) noexcept { return (static_cast<bool>(maybe) && ...); }, opened);
        if (!all_opened) return status_t::out_of_memory_heap_k;

        std::array<void const *, participants_k> const addresses {static_cast<void const *>(&stores)...};
        auto moved = std::apply([](auto &...maybe) noexcept { return transactions_t {std::move(*maybe)...}; }, opened);
        return transaction_group {std::move(moved), addresses};
    }

    /** @brief This group's participant in @p store_index_, counted in the caller's argument order. */
    template <std::size_t store_index_>
    auto &participant() noexcept {
        static_assert(store_index_ < participants_k, "no such participant");
        return std::get<store_index_>(transactions_);
    }

    /** @brief Whether the group's writes are sitting in their stores, invisible. */
    staging_t staging() const noexcept { return staging_; }

    /**
     *  @brief Validates every watch and reserves every write, undoing all of it if one refuses.
     *
     *  A partial stage is never observable. The undo rolls the staged prefix back rather than
     *  resetting it, so the caller's pending writes survive and the group can be retried.
     */
    [[nodiscard]] status_t stage() noexcept {
        if (staging_ == staging_t::staged_k) return operation_not_permitted_k;

        std::size_t staged_count = 0;
        status_t result = success_k;
        for (std::size_t position = 0; position != participants_k; ++position) {
            result = visit_at_(order_[position], [](auto &transaction) noexcept { return transaction.stage(); });
            if (failed(result)) break;
            ++staged_count;
        }

        if (failed(result)) {
            while (staged_count != 0) {
                --staged_count;
                [[maybe_unused]] status_t const unwound =
                    visit_at_(order_[staged_count], [](auto &transaction) noexcept { return transaction.rollback(); });
            }
            return result;
        }

        staging_ = staging_t::staged_k;
        return success_k;
    }

    /** @brief Publishes every participant. Cannot fail once @c stage has succeeded. */
    [[nodiscard]] status_t commit() noexcept {
        if (staging_ != staging_t::staged_k) return operation_not_permitted_k;
        status_t result = success_k;
        for (std::size_t position = 0; position != participants_k; ++position) {
            status_t const one =
                visit_at_(order_[position], [](auto &transaction) noexcept { return transaction.commit(); });
            if (failed(one)) result = one;
        }
        staging_ = staging_t::pending_k;
        return result;
    }

    /** @brief Pulls every staged write back into its transaction, leaving the group retryable. */
    [[nodiscard]] status_t rollback() noexcept {
        if (staging_ != staging_t::staged_k) return operation_not_permitted_k;
        status_t result = success_k;
        for (std::size_t position = participants_k; position-- != 0;) {
            status_t const one =
                visit_at_(order_[position], [](auto &transaction) noexcept { return transaction.rollback(); });
            if (failed(one)) result = one;
        }
        staging_ = staging_t::pending_k;
        return result;
    }

    /** @brief Discards every participant's staged and pending changes. */
    [[nodiscard]] status_t reset() noexcept {
        status_t result = success_k;
        for (std::size_t position = 0; position != participants_k; ++position) {
            status_t const one =
                visit_at_(order_[position], [](auto &transaction) noexcept { return transaction.reset(); });
            if (failed(one)) result = one;
        }
        staging_ = staging_t::pending_k;
        return result;
    }
};

/** @brief Deduces the store types, so a caller names the stores and not their spellings. */
template <typename... store_types_>
[[nodiscard]] expected<transaction_group<store_types_...>> make_transaction_group(store_types_ &...stores) noexcept {
    return transaction_group<store_types_...>::make(stores...);
}

#pragma region Shared Mutex

/**
 *  @brief A reader-writer lock that spins briefly, then parks on @c std::atomic::wait.
 *
 *  Locks here are held across user code - callbacks, comparators, allocators, a whole commit loop -
 *  so a pure spinner would burn a core while a holder walks a range, and pure parking would pay a
 *  syscall for the uncontended case that dominates. A waiting writer sets a bit that turns new
 *  readers away, so a steady read load cannot starve it indefinitely.
 *
 *  Offers exactly what @c std::shared_mutex is used for here, and nothing else: no recursion, no
 *  timed acquisition, no upgrading. Both collection wrappers take the mutex as a template parameter,
 *  so this is the default rather than the only choice.
 */
class shared_mutex_t {
    static constexpr std::uint32_t writer_held_k = 1u << 31;
    static constexpr std::uint32_t writer_waiting_k = 1u << 30;
    static constexpr std::uint32_t readers_mask_k = writer_waiting_k - 1;
    static constexpr std::uint32_t writer_bits_k = writer_held_k | writer_waiting_k;

    /** @brief How long to spin before parking, which is about the cost of one uncontended handoff. */
    static constexpr int spins_before_parking_k = 64;

    std::atomic<std::uint32_t> state_ {0};

    static void pause_briefly() noexcept {
#if defined(__x86_64__) || defined(__i386__)
        __builtin_ia32_pause();
#elif defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
#endif
    }

  public:
    constexpr shared_mutex_t() noexcept = default;
    shared_mutex_t(shared_mutex_t const &) = delete;
    shared_mutex_t &operator=(shared_mutex_t const &) = delete;

    [[nodiscard]] bool try_lock() noexcept {
        std::uint32_t expected = 0;
        return state_.compare_exchange_strong(expected, writer_held_k, std::memory_order_acquire,
                                              std::memory_order_relaxed);
    }

    [[nodiscard]] bool try_lock_shared() noexcept {
        std::uint32_t observed = state_.load(std::memory_order_relaxed);
        if (observed & writer_bits_k) return false;
        return state_.compare_exchange_strong(observed, observed + 1, std::memory_order_acquire,
                                              std::memory_order_relaxed);
    }

    void lock() noexcept {
        for (int spin = 0; spin != spins_before_parking_k; ++spin) {
            std::uint32_t expected = 0;
            if (state_.compare_exchange_weak(expected, writer_held_k, std::memory_order_acquire,
                                             std::memory_order_relaxed))
                return;
            pause_briefly();
        }

        // Announcing the intent is what stops a stream of readers from renewing the lock forever.
        std::uint32_t observed = state_.fetch_or(writer_waiting_k, std::memory_order_relaxed) | writer_waiting_k;
        while (true) {
            if ((observed & (writer_held_k | readers_mask_k)) == 0) {
                if (state_.compare_exchange_weak(observed, writer_held_k, std::memory_order_acquire,
                                                 std::memory_order_relaxed))
                    return;
                continue; // ? The exchange refreshed `observed`
            }
            state_.wait(observed, std::memory_order_relaxed);
            observed = state_.load(std::memory_order_relaxed);
        }
    }

    void unlock() noexcept {
        // Any writer still waiting re-announces itself on waking, so clearing the bit loses nothing.
        state_.store(0, std::memory_order_release);
        state_.notify_all();
    }

    void lock_shared() noexcept {
        for (int spin = 0; spin != spins_before_parking_k; ++spin) {
            std::uint32_t observed = state_.load(std::memory_order_relaxed);
            if ((observed & writer_bits_k) == 0 &&
                state_.compare_exchange_weak(observed, observed + 1, std::memory_order_acquire,
                                             std::memory_order_relaxed))
                return;
            pause_briefly();
        }

        while (true) {
            std::uint32_t observed = state_.load(std::memory_order_relaxed);
            if ((observed & writer_bits_k) == 0) {
                if (state_.compare_exchange_weak(observed, observed + 1, std::memory_order_acquire,
                                                 std::memory_order_relaxed))
                    return;
            }
            else { state_.wait(observed, std::memory_order_relaxed); }
        }
    }

    void unlock_shared() noexcept {
        std::uint32_t const previous = state_.fetch_sub(1, std::memory_order_release);
        // The last reader out is the only one a waiting writer is still blocked on.
        if ((previous & readers_mask_k) == 1) state_.notify_all();
    }
};

/** @brief Holds @p mutex_type_ exclusively for the enclosing scope. */
template <typename mutex_type_>
class unique_lock {
    mutex_type_ &mutex_;

  public:
    explicit unique_lock(mutex_type_ &mutex) noexcept : mutex_(mutex) { mutex_.lock(); }
    ~unique_lock() noexcept { mutex_.unlock(); }
    unique_lock(unique_lock const &) = delete;
    unique_lock &operator=(unique_lock const &) = delete;
};

/** @brief Holds @p mutex_type_ for shared reading over the enclosing scope. */
template <typename mutex_type_>
class shared_lock {
    mutex_type_ &mutex_;

  public:
    explicit shared_lock(mutex_type_ &mutex) noexcept : mutex_(mutex) { mutex_.lock_shared(); }
    ~shared_lock() noexcept { mutex_.unlock_shared(); }
    shared_lock(shared_lock const &) = delete;
    shared_lock &operator=(shared_lock const &) = delete;
};

template <typename mutex_type_>
unique_lock(mutex_type_ &) -> unique_lock<mutex_type_>;
template <typename mutex_type_>
shared_lock(mutex_type_ &) -> shared_lock<mutex_type_>;

#pragma endregion Shared Mutex

#pragma endregion Transaction Group

} // namespace ashvardanian::smashtable

#pragma region Structured Bindings

// What `auto [value, status] = ...` looks up. The members live in a `union`, so the compiler cannot
// decompose the class itself and the tuple protocol is the only route.
namespace std {

template <typename value_type_>
struct tuple_size<::ashvardanian::smashtable::expected<value_type_>> : integral_constant<size_t, 2> {};

template <typename value_type_>
struct tuple_element<0, ::ashvardanian::smashtable::expected<value_type_>> {
    using type = value_type_;
};

template <typename value_type_>
struct tuple_element<1, ::ashvardanian::smashtable::expected<value_type_>> {
    using type = ::ashvardanian::smashtable::status_t;
};

} // namespace std

#pragma endregion Structured Bindings
