/**
 *  @file include/smashtable/shared.hpp
 *  @author Ash Vardanian
 *  @date October 13, 2022
 *  @brief Vocabulary shared by every container - status codes, @c expected, key-value @c mapping,
 *      the versioning machinery transactions are built on, and the concepts stating what a
 *      container must offer to back one.
 *
 *  @section shared_store_tiers Store Tiers
 *
 *  What @c monotonic_store needs from its parameter is narrower than tree-ness. The concepts near
 *  the end of this file state that requirement, so a container which does not meet it says so at
 *  the call site rather than deep inside the adapter.
 *
 *  The requirements fall into two tiers, and the split is the point. Key-addressed work - staging a
 *  change, validating a watch, committing, reading one key - needs only point lookup and a
 *  transparent way to address an entry. Ordered work - bounds, ranges, order statistics - needs a
 *  total order over keys, which an open-addressed table cannot supply at any price.
 *
 *  Stating the tiers separately is what lets one adapter serve an ordered and an unordered core,
 *  gating the ordered surface behind the second concept rather than behind the container's name.
 *
 *  @section shared_store_surfaces Store Surfaces
 *
 *  A wrapper forwards only what its store answers, and asks for one method per concept. Two methods
 *  behind one gate means the store offering only the second loses it through every wrapper, which
 *  is the defect @c scripts/test_surface_parity.hpp exists to catch. Composites are spelled from
 *  those atoms rather than beside them.
 *
 *  Each concept detects a method, never what the method promises: a docblock naming a contract
 *  names the method's, and a store whose @c update creates an absent key satisfies @c
 *  offers_update anyway.
 *
 *  @section shared_device_code Reaching Device Code
 *
 *  Nothing here is annotated @c __host__ @c __device__. A function reachable from a kernel is
 *  marked @c constexpr instead, which @c nvcc @c --expt-relaxed-constexpr makes callable from
 *  device code, so one unannotated definition serves both sides and the headers stay compilable by
 *  a host compiler that has never heard of CUDA. Marking a body @c constexpr that no argument could
 *  ever constant- evaluate - anything touching an atomic or a placement @c new - is only ill-formed
 *  for a non-template function, and every such body here belongs to a template.
 */
#pragma once

#include <cerrno>  // `ENOMEM`, `EINVAL`, and the rest of the errno space
#include <climits> // `CHAR_BIT`
#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::int64_t`

#include <array>       // `std::array`
#include <atomic>      // `std::atomic_ref`
#include <bit>         // `std::countl_zero`
#include <compare>     // `std::compare_three_way`, for the total order over unrelated pointers
#include <concepts>    // `std::convertible_to`, `std::same_as`
#include <limits>      // `std::numeric_limits`
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

/**
 *  @brief Whether the target is x86-64, AArch64 or 64-bit RISC-V, one always-defined flag apiece.
 *
 *  Each is 0 or 1 rather than defined or not, so a call site tests it with @c if @c constexpr and
 *  the branch it drops is still compiled for syntax.
 */
#if defined(__x86_64__) || defined(_M_X64)
#define ST_TARGET_X8664_ 1
#else
#define ST_TARGET_X8664_ 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define ST_TARGET_ARM64_ 1
#else
#define ST_TARGET_ARM64_ 0
#endif

#if defined(__riscv) && defined(__riscv_xlen) && __riscv_xlen == 64
#define ST_TARGET_RISCV64_ 1
#else
#define ST_TARGET_RISCV64_ 0
#endif

/**
 *  @brief Whether this build carries each row-search kit, which the architecture, the compiler and
 *      any @c -D a caller set decide together.
 *
 *  These say what compiles, never what the processor running it can execute - @c row_search.hpp
 *  probes that separately, because a binary built for one machine is routinely run on another.
 */
#if !defined(ST_TARGET_HASWELL)
#if ST_TARGET_X8664_ && !defined(__CUDACC__) && (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER))
#define ST_TARGET_HASWELL 1
#else
#define ST_TARGET_HASWELL 0
#endif
#elif ST_TARGET_HASWELL && !ST_TARGET_X8664_
#undef ST_TARGET_HASWELL
#define ST_TARGET_HASWELL 0
#endif

#if !defined(ST_TARGET_SKYLAKE)
#if ST_TARGET_X8664_ && !defined(__CUDACC__) && \
    (defined(__GNUC__) || defined(__clang__) || (defined(_MSC_VER) && _MSC_VER >= 1920))
#define ST_TARGET_SKYLAKE 1
#else
#define ST_TARGET_SKYLAKE 0
#endif
#elif ST_TARGET_SKYLAKE && !ST_TARGET_X8664_
#undef ST_TARGET_SKYLAKE
#define ST_TARGET_SKYLAKE 0
#endif

#if !defined(ST_TARGET_NEON)
#if ST_TARGET_ARM64_ && !defined(__CUDACC__)
#define ST_TARGET_NEON 1
#else
#define ST_TARGET_NEON 0
#endif
#elif ST_TARGET_NEON && !ST_TARGET_ARM64_
#undef ST_TARGET_NEON
#define ST_TARGET_NEON 0
#endif

#if !defined(ST_TARGET_SVE)
#if ST_TARGET_ARM64_ && !defined(__CUDACC__) && !defined(_MSC_VER) && \
    ((defined(__clang__) && __clang_major__ >= 12) || (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 11))
#define ST_TARGET_SVE 1
#else
#define ST_TARGET_SVE 0
#endif
#elif ST_TARGET_SVE && !ST_TARGET_ARM64_
#undef ST_TARGET_SVE
#define ST_TARGET_SVE 0
#endif

#if !defined(ST_TARGET_RVV)
#if ST_TARGET_RISCV64_ && !defined(__CUDACC__) && \
    ((defined(__clang__) && __clang_major__ >= 17) || (!defined(__clang__) && defined(__GNUC__) && __GNUC__ >= 14))
#define ST_TARGET_RVV 1
#else
#define ST_TARGET_RVV 0
#endif
#elif ST_TARGET_RVV && !ST_TARGET_RISCV64_
#undef ST_TARGET_RVV
#define ST_TARGET_RVV 0
#endif

// GCC on RISC-V rejects `#pragma GCC target`, so its RVV functions carry the attribute one by one.
#if ST_TARGET_RVV && !defined(__clang__) && defined(__GNUC__)
#define ST_TARGET_RVV_ATTRIBUTE_ __attribute__((target("arch=+v")))
#else
#define ST_TARGET_RVV_ATTRIBUTE_
#endif

#if ST_TARGET_HASWELL || ST_TARGET_SKYLAKE
#include <immintrin.h>
#endif
#if ST_TARGET_X8664_ && defined(_MSC_VER)
#include <intrin.h> // `__cpuidex`, `_xgetbv`
#endif
#if ST_TARGET_NEON
#include <arm_neon.h>
#endif
#if ST_TARGET_SVE
#include <arm_sve.h>
#endif
#if ST_TARGET_RVV
#include <riscv_vector.h>
#endif
#if (ST_TARGET_ARM64_ || ST_TARGET_RISCV64_) && defined(__linux__)
#include <sys/auxv.h> // `getauxval`, `AT_HWCAP`
#endif

namespace ashvardanian::smashtable {

/**
 *  @brief Whether an operation succeeded, and why it did not. The library's only result code.
 *
 *  Values follow @c errno where one fits, so a failure can be handed to @c strerror or mapped onto
 *  a platform error without a translation table. @c success_k is zero, which is why the type is
 *  scoped and truth-testing goes through @c succeeded: an implicit conversion would read every
 *  success as false. Prefer checking an @c expected, which already answers this question.
 *
 *  The three conflicts have no @c errno between them: @c EBUSY and @c EAGAIN, the closest, are
 *  about a resource being held, while these are about what another transaction published. A retry
 *  loop reads them to decide whether retrying can help, which is what @c conflicted answers.
 */
enum class [[nodiscard]] status_t : int {

    /** The operation did what it was asked, and any result it carries is live. */
    success_k = 0,

    /** A failure the library cannot name more precisely, which no caller should branch on. */
    unknown_k = -1,

    /** An invariant of the container itself did not hold. Not an errno; @c 1 would collide with @c EPERM . */
    consistency_k = -2,

    /** A key this transaction wrote was published over since its snapshot. */
    write_conflict_k = -3,

    /** A key this transaction read was published over since its snapshot. */
    read_conflict_k = -4,

    /** A window this transaction read gained or lost a member since its snapshot. */
    phantom_conflict_k = -5,

    /** The heap refused the allocation the operation needed. */
    out_of_memory_heap_k = ENOMEM,

    /** An argument was outside what the call accepts, decided before anything was written. */
    invalid_argument_k = EINVAL,

    /** The object is not in a state that admits the call, such as committing what was never staged. */
    operation_not_permitted_k = EPERM,

    /** A bounded retry gave up rather than block. */
    operation_would_block_k = EWOULDBLOCK,

    /** A probe sequence with no slot left, which a rehash mends rather than more memory. */
    capacity_exhausted_k = ENOSPC,

    /** A device or file refused a read or write, which no retry in memory mends. */
    input_output_k = EIO,

    /** A strict @c insert found the key already taken. */
    key_already_exists_k = EEXIST,

    /** An @c update or erase found no such key. */
    key_not_found_k = ENOENT,
};

using enum status_t;

/** Whether @p status reports success, which no implicit conversion can be trusted to say. */
constexpr bool succeeded(status_t status) noexcept { return status == success_k; }

/** Whether @p status reports a failure, and so carries a reason. */
constexpr bool failed(status_t status) noexcept { return status != success_k; }

/**
 *  @brief A plain condition is already its own verdict, so generic code can ask either shape.
 *
 *  Without this, a caller holding a @c bool has to spell the question differently from one holding
 *  a @c status_t, and anything asking it of both - a check, a gate, a fold - needs a branch on
 *  which it was handed.
 */
constexpr bool succeeded(bool is_satisfied) noexcept { return is_satisfied; }

/** The same of a plain condition, so @c succeeded and @c failed stay a pair for every shape. */
constexpr bool failed(bool is_satisfied) noexcept { return !is_satisfied; }

/**
 *  @brief Keeps the first refusal of a walk that reports more than once, so the reason travels.
 *
 *  The discipline a @b read walk wants: the first thing that went wrong is what the caller asked
 *  about, and nothing after it can be trusted anyway.
 */
constexpr status_t first_failure(status_t recorded, status_t next) noexcept {
    return failed(recorded) ? recorded : next;
}

/**
 *  @brief Keeps the last refusal of a walk that reports more than once.
 *
 *  The discipline a @b write walk wants, where stopping at the first refusal would leave the steps
 *  after it untouched: the walk runs to the end regardless, so the reason it carries out is the
 *  last one it met.
 */
constexpr status_t last_failure(status_t recorded, status_t next) noexcept { return failed(next) ? next : recorded; }

/** The enumerator's own spelling, for a message a person reads rather than decodes. */
constexpr char const *name_of(status_t status) noexcept {
    switch (status) {
    case status_t::success_k: return "success_k";
    case status_t::unknown_k: return "unknown_k";
    case status_t::consistency_k: return "consistency_k";
    case status_t::write_conflict_k: return "write_conflict_k";
    case status_t::read_conflict_k: return "read_conflict_k";
    case status_t::phantom_conflict_k: return "phantom_conflict_k";
    case status_t::out_of_memory_heap_k: return "out_of_memory_heap_k";
    case status_t::invalid_argument_k: return "invalid_argument_k";
    case status_t::operation_not_permitted_k: return "operation_not_permitted_k";
    case status_t::operation_would_block_k: return "operation_would_block_k";
    case status_t::capacity_exhausted_k: return "capacity_exhausted_k";
    case status_t::input_output_k: return "input_output_k";
    case status_t::key_already_exists_k: return "key_already_exists_k";
    case status_t::key_not_found_k: return "key_not_found_k";
    }
    return "unrecognized";
}

/** Whether @p status turned a transaction away over what another one published, which a retry can overcome. */
constexpr bool conflicted(status_t status) noexcept {
    return status == status_t::write_conflict_k || status == status_t::read_conflict_k ||
           status == status_t::phantom_conflict_k;
}

/** The same question of a richer result - one that reports its own success, like the node handles the trees hand
 *  back - so generic code can ask it without knowing which it holds. */
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

template <typename value_type_>
class expected;

template <typename object_type_>
expected<object_type_> copy_safely(object_type_ const &object) noexcept;

/**
 *  @brief Simple key-value mapping type, cleaner & lighter than @c std::pair used by @c std::map.
 *  @see https://en.cppreference.com/w/cpp/utility/pair.html
 */
template <typename key_type_, typename mapped_type_>
struct mapping {
    using key_type = key_type_;
    using mapped_type = mapped_type_;

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

    /** Deep-copies both halves, so a mapping is copyable exactly when its members are. */
    expected<mapping> copy() const noexcept {
        auto copied_key = copy_safely(key);
        if (!copied_key) return copied_key.status();
        auto copied_mapped = copy_safely(mapped);
        if (!copied_mapped) return copied_mapped.status();
        return expected<mapping>(mapping {std::move(*copied_key), std::move(*copied_mapped)}, success_k);
    }
};

/** Concept to detect if a type is an mapping (has key_type and mapped_type members). */
template <typename type_>
concept is_mapping = requires {
    typename std::remove_cvref_t<type_>::key_type;
    typename std::remove_cvref_t<type_>::mapped_type;
};

/**
 *  @brief Conditional callback validation traits controlled by ST_STRICT_CALLBACK_CHECKS_.
 *
 *  When ST_STRICT_CALLBACK_CHECKS_ is defined, these traits perform full compile-time validation
 *  using std::is_nothrow_invocable_v. This catches type errors early but prevents generic lambdas
 *  like no_op_t {} from compiling.
 *
 *  When undefined (default), traits always return true, allowing generic lambdas while still
 *  documenting the intent that callbacks should be noexcept.
 */
#ifdef ST_STRICT_CALLBACK_CHECKS_
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
 *  @c std::allocator would serve, and costs @c \<memory\> - some fifty thousand preprocessed lines
 *  - in headers that want nothing else from it. Exhaustion is reported by returning null, which is
 *  the shape every caller here already checks, rather than by an exception nothing would catch.
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
        // A request whose byte count wraps would otherwise hand back a small block that reads as valid.
        if (count > SIZE_MAX / sizeof(value_type_)) return nullptr;
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

/** The byte allocator every table is carved from unless a caller names another. */
using default_allocator_t = default_allocator<std::byte>;

/**
 *  @brief Transparent ordering and equality, so a container needs no @c \<functional\>.
 *
 *  Both are heterogeneous - the @c is_transparent tag is what lets a lookup compare a key against
 *  something merely comparable to it, without materializing a key first - which is the only
 *  property the containers here relied on @c std::less and @c std::equal_to for.
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
 *  Deliberately not a forward to @c std::hash: that would drag @c \<functional\> into every header,
 *  and the point of a named trait is that an unhashable key says so at compile time. Hashing the
 *  bytes of an arbitrary type instead would be worse than an error - padding, or an equality that
 *  ignores some field, would give two equal keys different hashes and quietly lose lookups.
 */
template <typename key_type_, typename = void>
struct hash {
    static_assert(sizeof(key_type_) == 0,
                  "No hash for this key. Specialize ashvardanian::smashtable::hash<key_type_> for it.");
};

/** Integers and enumerations, mixed so that adjacent keys do not land in adjacent buckets. */
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

/** A pointer hashes as the integer it is. */
template <typename key_type_>
struct hash<key_type_, std::enable_if_t<std::is_pointer_v<key_type_>>> {
    std::size_t operator()(key_type_ key) const noexcept {
        return hash<std::uintptr_t> {}(reinterpret_cast<std::uintptr_t>(key));
    }
};

/** Anything exposing contiguous bytes through @c data and @c size, which covers the string types. */
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

/** How many steps separate two iterators, without the sixteen thousand lines of @c \<iterator\>. */
template <typename iterator_type_>
constexpr std::size_t distance_between(iterator_type_ first, iterator_type_ last) noexcept {
    std::size_t count = 0;
    for (; first != last; ++first) ++count;
    return count;
}

/** Sentinel type for range-based iteration end conditions. */
struct end_sentinel_t {};

/** Placeholder type for template conditional where no type is needed. */
struct placeholder_t {};

template <typename type_>
struct no_op {
    constexpr void operator()(type_ &&) const noexcept {}
    constexpr void operator()(type_ const &) const noexcept {}
};

template <>
struct no_op<void> {
    template <typename type_>
    constexpr void operator()(type_ &&) const noexcept {}
    constexpr void operator()() const noexcept {}
};

using no_op_t = no_op<void>;

/** Callback that stores whatever it is handed into a caller-owned destination. */
template <typename element_type_>
struct copy_to_fn {
    element_type_ &target;
    template <typename type_>
    void operator()(type_ &&source) const noexcept {
        target = std::forward<type_>(source);
    }
};

/** Builds a @c copy_to_fn over @p element, for a read whose result the caller wants kept. */
template <typename element_type_>
copy_to_fn<element_type_> copy_to(element_type_ &element) noexcept {
    return {element};
}

#pragma region Tag Dispatch Types

/**
 *  @brief Tag to assume capacity is pre-allocated, skip null checks. Similar to @c std::adopt_lock
 *      for assuming preconditions are met.
 *  @see https://en.cppreference.com/w/cpp/thread/lock_tag_t
 */
struct assume_reserved_t {
    explicit assume_reserved_t() = default;
};
inline constexpr assume_reserved_t assume_reserved {};

/** Tag to assume elements are unique, skip equality comparisons. Similar to Boost.Container's
 *  @c ordered_unique_range_t. */
struct assume_unique_t {
    explicit assume_unique_t() = default;
};
inline constexpr assume_unique_t assume_unique {};

/**
 *  @brief Tag to assume the input range is ascending, which buys an O(n) bulk build.
 *  @warning An unsorted range is undefined behaviour, not a refusal.
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

/** A promise about the range itself, which a staged bulk write reads before it builds. */
template <typename type_>
concept promises_about_the_range = std::same_as<type_, assume_sorted_t> || std::same_as<type_, assume_unique_t>;

/** A promise that the room is already secured, which a write into that room reads. */
template <typename type_>
concept promises_about_the_room = std::same_as<type_, assume_reserved_t>;

/** Any promise this library reads, which is what an open tag pack admits - a foreign type is refused there. */
template <typename type_>
concept promises_something = promises_about_the_range<type_> || promises_about_the_room<type_>;

#pragma endregion Tag Dispatch Types

#pragma region Copying and Construction

/**
 *  @brief A value that may not be there, and the reason when it is not - the library's only result.
 *
 *  Storage is a @c union, so the value exists only on success and @p value_type_ needs no default
 *  constructor. That is what lets one type cover both what @c std::optional carried and what a
 *  status carried, including the move-only transaction types that have no empty state to sit in. A
 *  default-constructed @c expected reports @b failure, since there is nothing in it.
 *
 *  Decomposes as @c auto @c [value, status], for value types that can be default-constructed to
 *  hand something back on the failure path; see @c get.
 */
template <typename value_type_>
class [[nodiscard]] expected {
  public:
    using value_t = value_type_;
    using value_type = value_t; // ? STL style

    static_assert(std::is_nothrow_move_constructible_v<value_t>,
                  "expected<T> moves its value into place, so T must move without throwing");

  private:
    union {
        value_t outcome_; // ? Alive exactly while `succeeded(status_)`, which every constructor upholds
    };
    status_t status_ = unknown_k;

  public:
    constexpr expected() noexcept {}

    /** A reason with nothing behind it; a success has no value to report, so it reads as @c unknown_k. */
    constexpr expected(status_t status) noexcept : status_(status != success_k ? status : unknown_k) {}

    /** A value, kept only while @p status says there is one. A failure carries its reason and nothing else, so the
     *  argument is left to its owner. */
    expected(value_t &&value, status_t status = success_k) noexcept : status_(status) {
        if (succeeded(status_)) new (&outcome_) value_t(std::move(value));
    }
    expected(value_t const &value, status_t status = success_k) noexcept
        requires(std::is_nothrow_copy_constructible_v<value_t>)
        : status_(status) {
        if (succeeded(status_)) new (&outcome_) value_t(value);
    }

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

    /** Whether the value is there, which is the only safe precondition for reading it. */
    constexpr explicit operator bool() const noexcept { return succeeded(status_); }
    constexpr bool has_value() const noexcept { return succeeded(status_); }

    /** Whether there is no value, which is the protocol every result in the library answers. */
    constexpr bool failed() const noexcept { return !succeeded(status_); }

    /** Why there is no value, or success when there is one. */
    constexpr status_t status() const noexcept { return status_; }

    /** @warning Reading the value of a failed @c expected is undefined - check it first. */
    constexpr value_t &operator*() & noexcept { return outcome_; }
    constexpr value_t const &operator*() const & noexcept { return outcome_; }
    constexpr value_t &&operator*() && noexcept { return std::move(outcome_); }
    constexpr value_t *operator->() noexcept { return &outcome_; }
    constexpr value_t const *operator->() const noexcept { return &outcome_; }

    /**
     *  @brief The tuple protocol behind @c auto @c [value, status], returning the value @b
     *      by value.
     *
     *  A reference would bind into storage that was never constructed when the status is a failure,
     *  which reads as garbage and trips no sanitizer, so the failure path default-constructs
     *  instead. Offered only where that is possible: a transaction cannot be decomposed, and has
     *  nothing worth binding when it fails anyway, so callers check it and dereference.
     */
    template <std::size_t index_>
        requires(std::is_nothrow_default_constructible_v<value_t>)
    constexpr auto get() && noexcept {
        if constexpr (index_ == 0) return succeeded(status_) ? value_t {std::move(outcome_)} : value_t {};
        else return status_;
    }

    template <std::size_t index_>
        requires(std::is_nothrow_default_constructible_v<value_t> && std::is_nothrow_copy_constructible_v<value_t>)
    constexpr auto get() const & noexcept {
        if constexpr (index_ == 0) return succeeded(status_) ? value_t {outcome_} : value_t {};
        else return status_;
    }
};

/**
 *  @brief Whether @p result carries @p status - the reason it failed, or success when it holds
 *      a value.
 *
 *  Saves a @c .status() at the call site, and reads the way the question is asked out loud. C++20
 *  synthesizes the reversed and negated forms, so this one definition covers all four spellings.
 */
template <typename value_type_>
[[nodiscard]] constexpr bool operator==(expected<value_type_> const &result, status_t status) noexcept {
    return result.status() == status;
}

/**
 *  @brief Whether @p result holds a value equal to @p value.
 *
 *  @warning A failed @c expected equals no value, so a @c false answer covers both "held something
 *      else" and "held nothing at all". Where a test needs those apart, check the result first and
 *      compare the value second.
 *
 *  Excluded for @c status_t so the overload above stays the one that answers a status, which also
 *  keeps @c expected<status_t> unambiguous - there, this asks about the reason and not the payload.
 */
template <typename value_type_, typename comparable_type_>
    requires(!std::same_as<std::remove_cvref_t<comparable_type_>, status_t>) &&
            requires(value_type_ const &held, comparable_type_ const &other) {
                { held == other } -> std::convertible_to<bool>;
            }
[[nodiscard]] constexpr bool operator==(expected<value_type_> const &result, comparable_type_ const &value) noexcept {
    return result.has_value() && *result == value;
}

/** Concept checking if a type has a @c .copy() method returning @c expected<T>. This allows non-nothrow-copyable
 *  types to participate in safe copy operations. */
template <typename type_>
concept has_copy_method = requires(type_ const &t) {
    { t.copy() } noexcept -> std::same_as<expected<type_>>;
};

/** Concept checking if a type has a static @c .make() method returning @c expected<T>. This allows types with
 *  potentially throwing constructors to participate in safe construction. */
template <typename type_, typename... args_types_>
concept has_make_method = requires(args_types_ &&...args) {
    { type_::make(std::forward<args_types_>(args)...) } noexcept -> std::same_as<expected<type_>>;
};

/**
 *  @brief Copies @p object through its @c noexcept copy constructor, or its @c copy method.
 *  @return The copy on success, or the reason it could not be made.
 */
template <typename object_type_>
expected<object_type_> copy_safely(object_type_ const &object) noexcept {
    if constexpr (std::is_nothrow_copy_constructible_v<object_type_>)
        return expected<object_type_>(object_type_ {object}, success_k);
    else if constexpr (has_copy_method<object_type_>) return object.copy();
    else {
        static_assert(std::is_nothrow_copy_constructible_v<object_type_> || has_copy_method<object_type_>,
                      "Type must be nothrow copy constructible or provide .copy() returning an expected");
        return status_t::unknown_k;
    }
}

/**
 *  @brief Whether duplicating this type can refuse, which is the whole reason a batch stages.
 *
 *  A type that copies without throwing duplicates by copy construction and cannot report anything,
 *  so a batch over it needs no staging - securing the room is the only failure it has left. Declaring
 *  @c copy is not the question: @c mapping declares one unconditionally, and a mapping of two trivial
 *  halves still cannot refuse.
 */
template <typename type_>
concept duplication_can_refuse = !std::is_nothrow_copy_constructible_v<type_>;

/**
 *  @brief Hands @p stage_one every element of [ @p first, @p last ), duplicated outside the destination.
 *
 *  @param[in] stage_one Receives one element by rvalue and reports where it landed. Must be @c noexcept.
 *  @return The first refusal @p stage_one or a duplication reported, or @c success_k for the whole range.
 *
 *  A range handing over lvalues duplicates through @c copy_safely, so an element refusing its own copy
 *  stops the batch with that reason rather than with the allocator's; a range handing over rvalues moves,
 *  which is what a move iterator asks for. Nothing here reaches the destination - what @p stage_one does
 *  with an element is where a batch decides whether it can still be undone.
 */
template <typename value_type_, typename input_iterator_type_, typename stage_one_type_>
status_t stage_each(input_iterator_type_ first, input_iterator_type_ last, stage_one_type_ &&stage_one) noexcept {
    for (; first != last; ++first) {
        if constexpr (!std::is_lvalue_reference_v<decltype(*first)>) {
            if (status_t const staged = stage_one(value_type_(*first)); failed(staged)) return staged;
        }
        else {
            expected<std::remove_cvref_t<decltype(*first)>> duplicated = copy_safely(*first);
            if (!duplicated) return duplicated.status();
            if (status_t const staged = stage_one(value_type_(*std::move(duplicated))); failed(staged)) return staged;
        }
    }
    return success_k;
}

#pragma endregion Copying and Construction

#pragma region Key Extraction

/**
 *  @brief Checks if a type has a @c comparable_particle member.
 *
 *  @code{.cpp}
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
 *  @brief If @c can_be_mapping_type_ is a mapping, exposes its @c key_type as @c ::type. Otherwise,
 *      exposes @c can_be_mapping_type_ itself as @c ::type.
 *
 *  @code{.cpp}
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

/** Helper method to extract mapping key or return the object itself. Useful for generic code that works with both
 *  mappings and simple keys. */
template <typename can_be_mapping_type_>
typename mapping_key_type_or_itself<can_be_mapping_type_>::type const & //
mapping_key_or_itself(can_be_mapping_type_ const &can_be_mapping) noexcept {
    if constexpr (is_mapping<can_be_mapping_type_>) return can_be_mapping.key;
    else return can_be_mapping;
}

/**
 *  @brief If @c can_be_mapping_type_ is a mapping, exposes its @c mapped_type as @c ::type.
 *      Otherwise, exposes @c void as @c ::type.
 *
 *  @code{.cpp}
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
 *  - @c std::strong_ordering - equivalent values are fully interchangeable, everything
 *    is comparable!
 *  - @c std::weak_ordering - equivalent values may be non-interchangeable, but everything
 *    is comparable!
 *  - @c std::partial_ordering - equivalent values may be non-interchangeable, and may
 *    be incomparable!
 *
 *  Assuming that @c float values can be @c NaN and thus incomparable, we shouldn't be able to
 *  construct a @c std::set<float> since the ordering is only partial, but GCC still compiles it.
 *
 *  @section shared_optimization_opportunities Optimization Opportunities
 *
 *  Oftentimes, when dealing with heavy objects as keys (e.g., strings, composite structures), we
 *  want to avoid storing many copies of the full object just to perform comparisons and lookups.
 *  Instead, we can distill a lightweight "particle" from the heavy object that captures its
 *  identity for ordering purposes. Examples may be:
 *
 *  - Non-owning @c std::string_view for @c std::string keys
 *  - Integer identifiers extracted from composite structures
 *
 *  If the @c comparator_type_ has a @c value_type member - indicating some form of a unique
 *  identifier can be distilled from @c comparable_type_ - we expose it via @c ::type. Else, if the
 *  @c comparable_type_ is a @c mapping, we use its @c key_type as the identifier. Else, we use the
 *  @c comparable_type_ itself as the identifier.
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

#pragma region Optimistic Concurrency

/** Generation type for versioned elements. */
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
 *  @brief When a version became visible, or that it never has.
 *
 *  The sentinel is the @b maximum rather than zero so that visibility is a single comparison
 *  against a reader's snapshot: an uncommitted version is newer than every snapshot that will ever
 *  be taken, and so falls out of @c visible_at without a second branch asking whether it is still
 *  staged. Zero would have made "not yet visible" compare as "visible since the beginning of time".
 */
enum class commit_stamp_t : generation_t { uncommitted_k = std::numeric_limits<generation_t>::max() };

/** Whether a version stamped @p stamp is visible to a reader holding @p snapshot. */
constexpr bool visible_at(commit_stamp_t stamp, generation_t snapshot) noexcept {
    return static_cast<generation_t>(stamp) <= snapshot;
}

/** The snapshot a reader that takes none holds: every stamp a commit can ever draw is at or below it, and only
 *  @c commit_stamp_t::uncommitted_k sits above. */
inline constexpr generation_t latest_snapshot_k = std::numeric_limits<generation_t>::max() - 1;

/** Whether @p stamp was drawn at all, which is what a reader taking no snapshot asks. */
constexpr bool visible_now(commit_stamp_t stamp) noexcept { return visible_at(stamp, latest_snapshot_k); }

/**
 *  @brief What a reader is promised, named as Jepsen names it and ordered by strength.
 *
 *  The top two differ only in when a commit becomes visible, never in what they refuse: both
 *  validate every read, so neither admits write skew or a phantom. @c strict_serializable_k
 *  additionally waits for its own publication before returning, so a transaction opening after a
 *  commit returned cannot be ordered before it. A transaction commit waits at both levels, so the
 *  difference reaches only a sharded commit and a write made outside a transaction.
 *
 *  @see https://jepsen.io/consistency
 */
enum class isolation_t : std::uint8_t {
    read_committed_k = 0,
    monotonic_atomic_view_k = 1,
    snapshot_k = 2,
    serializable_k = 3,
    strict_serializable_k = 4,
};

/** Whether a walk carries on after handing an element over, for a callback that can say. */
enum class walk_control_t : bool { resume_k, halt_k };

/**
 *  @brief What a step tells the walk that handed it a member to do next, inside the library.
 *
 *  The counterpart of @c walk_control_t on the other side of the boundary: a caller's callback
 *  answers @c walk_control_t, and @c hand_over turns that into this for the walk that is running.
 */
enum class probe_control_t : std::uint8_t {

    /** The step wants whatever the walk has next - the same probe run, or the merged order. */
    resume_k,

    /** The step has what it came for, and the walk stops here. */
    halt_k,
};

/**
 *  @brief Whether @p callback_type_ can stop a walk, rather than taking every element it
 *      is offered.
 *
 *  A callback answering @c walk_control_t is asked whether to carry on; one answering @c void is
 *  asking for everything. The difference is resolved where the walk runs rather than by a flag.
 */
template <typename callback_type_, typename element_type_>
constexpr bool halts_the_walk = requires(callback_type_ &callback, element_type_ element) {
    { callback(element) } noexcept -> std::same_as<walk_control_t>;
};

/** Hands @p element to @p callback and answers whether the walk carries on. A callback that cannot say resumes it,
 *  so a walk asks here instead of branching on the shape. */
template <typename callback_type_, typename element_type_>
[[nodiscard]] constexpr walk_control_t hand_over(callback_type_ &&callback, element_type_ &&element) noexcept {
    if constexpr (halts_the_walk<callback_type_, element_type_>) return callback(element);
    else {
        callback(element);
        return walk_control_t::resume_k;
    }
}

/** Whether @p offered is at least as strong a promise as @p required. */
constexpr bool at_least(isolation_t offered, isolation_t required) noexcept {
    return static_cast<std::uint8_t>(offered) >= static_cast<std::uint8_t>(required);
}

/** Where a cursor's next step begins, which is a different question before and after the first. */
enum class cursor_seed_t : std::uint8_t {

    /** No bound was given and nothing has been handed over, so the walk starts at the least key. */
    the_smallest_k,

    /** A bound was given and nothing has been handed over, so that bound is inclusive. */
    the_given_bound_k,

    /** A key has been handed over, so the walk continues strictly above it. */
    past_the_last_k,
};

/**
 *  @brief Whether a cursor stops at a key it was given, or only when the store runs out.
 *
 *  Carried by the cursor rather than tested by the caller, because the comparator is the store's
 *  and a caller comparing a delivered key against a bound would be ordering keys the store owns.
 */
enum class cursor_limit_t : bool { the_whole_keyspace_k, up_to_the_bound_k };

/**
 *  @brief Whether an ordered cursor can still hand something over, once a step has found that
 *      it cannot.
 *
 *  A step that runs out latches this rather than re-probing, because nothing can put a key back
 *  below where the walk stopped: the store is held for the walk, and a cursor never
 *  steps backwards.
 */
enum class cursor_reach_t : bool {

    /** A step may still find a key. */
    walking_k,

    /** A step found the store spent or the bound reached, and every later step would too. */
    spent_k,
};

/**
 *  @brief Whether a transaction's writes are sitting in the store, invisible, or not there yet.
 *
 *  A committed transaction returns to @c pending_k and is immediately reusable, so there is no
 *  third state to name - @c commit on a @c pending_k transaction already reports
 *  @c operation_not_permitted_k.
 */
enum class staging_t : bool {

    /** Accepting writes; the store holds nothing of this transaction. */
    pending_k,

    /** The store has reserved every change, so @c commit allocates nothing and only stamps. */
    staged_k,
};

/** Whether an entry says its key is there, or says it was taken away and when. */
enum class presence_t : bool {

    /** The key is there and this is its value. */
    present_k,

    /** The key was erased. The stamp survives so a watch can date the erasure. */
    erased_k,
};

/**
 *  @brief What a commit found when it went looking for the version it was asked to publish.
 *
 *  @c already_published_k is what makes a repeated identifier harmless: a commit listing one key
 *  twice finds the second pass settled rather than missing, which a two-state answer could only
 *  spell as the same success a fresh publish reports.
 */
enum class unmask_outcome_t : std::uint8_t {

    /** The staged version was there and now carries this commit's stamp. */
    unmasked_k,

    /** This commit had already published that version on an earlier pass. */
    already_published_k,

    /** No entry carries that generation, so nothing became visible. */
    version_missing_k,
};

/** Watch metadata for versioned elements. */
struct watch_t {
    generation_t generation {0};
    presence_t presence {presence_t::present_k};

    inline bool operator==(watch_t const &watch) const noexcept {
        return generation == watch.generation && presence == watch.presence;
    }
};

/**
 *  @brief What a watch on a key with no entry at all records.
 *
 *  A key erased through a transaction carries a committed tombstone and dates its absence by that
 *  tombstone's generation, which is what tells an absence nothing disturbed apart from one an
 *  insert and an erase closed over again. This constant is what remains when there is no tombstone
 *  to date.
 *
 *  @warning Absence is dated only while a tombstone survives. An erase taken outside a transaction
 *      drops the entry outright, and @c vacuum, @c clear and the ranged erases reclaim one a watch
 *      may still hold - so a watch can miss drift once its tombstone is gone, and can report drift
 *      that is only the reclamation. Neither @c monotonic_store nor @c reference_store pins
 *      reclamation behind an open reader the way @c snapshot_store does.
 */
constexpr watch_t missing_watch() noexcept { return watch_t {absent_generation_k, presence_t::erased_k}; }

/**
 *  @brief The watch a read resolving to @p resolved should record, or @c missing_watch when it
 *      resolves to nothing at all. A committed tombstone keeps its own generation rather than
 *      collapsing onto the constant, so two absences separated by a commit do not compare equal.
 *
 *  Both ends of a watch must resolve through the same finder, or a tombstone one end cannot see
 *  reads as drift on the other. Both engines therefore watch through the finder validation uses.
 */
template <typename versioned_type_>
[[nodiscard]] constexpr watch_t watch_shape_of(versioned_type_ const *resolved) noexcept {
    if (!resolved) return missing_watch();
    return watch_t {resolved->generation, resolved->presence};
}

/** Whether a watched key moved between @p recorded and what it resolves to @p now. Not equality: an absence a vacuum
 *  stripped of its date is the absence that was already read. */
[[nodiscard]] inline bool watch_drifted(watch_t recorded, watch_t now) noexcept {
    if (recorded == now) return false;
    bool const stood_absent = recorded.presence == presence_t::erased_k;
    return !(stood_absent && now == missing_watch());
}

/**
 *  @brief Re-reads every watched identifier and reports whether any drifted since it was sampled.
 *  @param[in] watches The identifier-and-watch pairs a transaction accumulated.
 *  @param[in] resolve_latest Invoked as @c resolve_latest(identifier,on_found,on_missing) . Must
 *      be noexcept.
 *  @return @c success_k, or @c read_conflict_k for the first watch that drifted.
 */
template <typename watches_type_, typename resolver_type_>
status_t validate_watches(watches_type_ const &watches, resolver_type_ &&resolve_latest) noexcept {
    for (auto const &identifier_and_watch : watches) {
        // An identifier nothing resolves to keeps the shape it was given here, which is the absent one.
        watch_t latest = missing_watch();
        resolve_latest(
            identifier_and_watch.identifier, [&](auto const &entry) noexcept { latest = watch_shape_of(&entry); },
            no_op_t {});
        if (watch_drifted(identifier_and_watch.watch, latest)) return status_t::read_conflict_k;
    }
    return success_k;
}

template <typename identifier_type_>
struct dated_identifier {
    using identifier_t = identifier_type_;

    identifier_t identifier;
    generation_t generation {0};
};

template <typename identifier_type_>
struct watched_identifier {
    using identifier_t = identifier_type_;

    identifier_t identifier;
    watch_t watch;
};

/**
 *  @brief How one transaction reached one key, as independent claims rather than as a state.
 *
 *  A key can be read and be the endpoint of a range at once, so these are flags and not an
 *  enumeration. Nothing here dates anything: a transaction's snapshot is the only date its
 *  validator needs, which is what lets a read record no sampled value at all.
 *
 *  @c read_k names exactly this key, so its whole version run is validated. @c opens_k begins a
 *  range read at this key, inclusive, and the entry recorded next to it closes that range,
 *  exclusive - the adjacency is the pairing, which is what lets one sequence hold points and
 *  windows alike.
 *
 *  A window can also run off one end. An ordinal read depends on every key ordered before the one
 *  it lands on, and a bound read that finds nothing depends on everything above its bound, yet
 *  neither end has a key to name - there is no smallest @c std::string and no largest one.
 *  @c from_the_lowest_k on the opening entry and @c to_the_highest_k on the closing entry say so,
 *  and the identifier stored beside the flag is then ignored rather than meaning anything.
 */
enum class access_t : std::uint8_t {
    none_k = 0,
    read_k = 1u << 0,
    opens_k = 1u << 1,
    closes_k = 1u << 2,
    from_the_lowest_k = 1u << 3,
    to_the_highest_k = 1u << 4,
};

/** Whether @p mask carries @p flag, spelled out because a scoped enum has no @c operator&. */
[[nodiscard]] constexpr bool holds(access_t mask, access_t flag) noexcept {
    return (static_cast<std::uint8_t>(mask) & static_cast<std::uint8_t>(flag)) != 0;
}

/** Both claims at once, for a key that is read and also opens a range. */
[[nodiscard]] constexpr access_t operator|(access_t first, access_t second) noexcept {
    return static_cast<access_t>(static_cast<std::uint8_t>(first) | static_cast<std::uint8_t>(second));
}

/**
 *  @brief One key a transaction read, and in what way.
 *
 *  Carries no generation, so it orders by key alone and a validator dates it against the
 *  transaction's snapshot rather than against whatever the read happened to see.
 */
template <typename identifier_type_>
struct accessed_identifier {
    using identifier_t = identifier_type_;

    identifier_t identifier;
    access_t access {access_t::read_k};
};

template <typename type_>
struct is_dating_identifier : public std::false_type {};

template <typename identifier_type_>
struct is_dating_identifier<dated_identifier<identifier_type_>> : public std::true_type {};

template <typename identifier_type_>
struct is_dating_identifier<watched_identifier<identifier_type_>> : public std::true_type {};

template <typename type_>
inline constexpr bool is_dating_identifier_v = is_dating_identifier<type_>::value;

/** Detects operands that carry a generation of their own, so ordering can separate versions. A @c watched_identifier
 *  keeps its generation inside @c watch, so it is ordered by key alone. */
template <typename type_>
concept carries_generation = requires(type_ const &value) {
    { value.generation } -> std::convertible_to<generation_t>;
};

/** Detects a version-decorated value: a generation stamp @b and the payload it decorates. Stricter than
 *  @c carries_generation on purpose - @c watch_t carries a generation and no payload, and would otherwise be peeled
 *  down a branch that does not compile. */
template <typename type_>
concept carries_versioned_payload = carries_generation<type_> && requires(type_ const &value) { value.payload; };

/** Operands whose generation can break a tie, because a key can still be peeled out of them. A bare @c watch_t is
 *  the counter-example: it dates an entry without naming one, so it is compared against the entry it dates rather
 *  than fed to a comparator. */
template <typename type_>
concept orderable_per_version =
    carries_versioned_payload<type_> || (carries_generation<type_> && is_dating_identifier_v<type_>);

/** The identifier a version-decorated object is addressed by, with the metadata peeled off. A chain defers to the
 *  version it holds, a version to the value it wraps, and a dated identifier to the identifier inside it, so a plain
 *  key is reached from any of the shapes a store stores. */
template <typename type_>
decltype(auto) identifier_of(type_ const &object) noexcept {
    using dereferenced_t = std::remove_reference_t<type_>;
    if constexpr (requires { typename dereferenced_t::is_version_chain; }) return identifier_of(object.head);
    else if constexpr (is_dating_identifier_v<dereferenced_t>)
        return (typename dereferenced_t::identifier_t const &)object.identifier;
    else if constexpr (carries_versioned_payload<dereferenced_t>) return identifier_of(object.payload);
    else return mapping_key_or_itself(object);
}

/**
 *  @brief Decorates a value type with generation and visibility metadata for
 *      transactional containers.
 *
 *  @section shared_template_requirements Template Requirements
 *
 *  - @p value_type_ must be a value type (no references), nothrow default-constructible, and
 *    nothrow move constructible/assignable so that transactional staging can remain noexcept.
 *  - @p comparator_type_ must expose @c value_type describing the identifier used for ordering, and
 *    that identifier type must be nothrow copy-constructible to support watch bookkeeping.
 *  - @p value_type_ must be convertible to the identifier type and constructible from it, enabling
 *    the containers to extract keys for lookups and manufacture key-only tombstones for erases.
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
    using accessed_identifier_t = accessed_identifier<identifier_t>;

    static_assert(!std::is_reference<value_t>(), "Only value types are supported.");
    static_assert(std::is_nothrow_copy_constructible_v<identifier_t> || has_copy_method<identifier_t>,
                  "To WATCH, the identifier must be nothrow copy constructible or provide .copy()");
    static_assert(std::is_nothrow_default_constructible<value_t>(), "We need an empty state.");
    static_assert(std::is_nothrow_move_constructible<value_t>() && std::is_nothrow_move_assignable<value_t>(),
                  "To make all the methods `noexcept`, the moves must be safe too.");

    struct versioned_t {

        /** The stored value, with none of the metadata around it. */
        value_t payload;

        /** Which transaction wrote this version, which is how its own writer finds it again. */
        generation_t generation {0};

        /** Whether this version says the key is there, or says it was taken away. */
        presence_t presence {presence_t::present_k};

        /** When this version became visible, or @c uncommitted_k while it is only staged. */
        commit_stamp_t committed {commit_stamp_t::uncommitted_k};

        versioned_t() = default;
        versioned_t(versioned_t &&) noexcept = default;
        versioned_t &operator=(versioned_t &&) noexcept = default;
        versioned_t(versioned_t const &) noexcept = delete;
        versioned_t &operator=(versioned_t const &) noexcept = delete;
        versioned_t(value_t &&value) noexcept : payload(std::move(value)) {}

        bool operator==(watch_t const &watch) const noexcept {
            return generation == watch.generation && presence == watch.presence;
        }
    };

    using versioned_entry_t = versioned_t;

    struct versioned_comparator_t {
        using is_transparent = void;

        /**
         *  @brief The comparator this wrapper was built with, consulted for every comparison.
         *
         *  Held rather than manufactured on demand: a comparator that carries state - a direction
         *  flag, a collation table, a dispatch pointer - answers differently from a
         *  default-constructed one, so building a fresh instance per comparison silently discards
         *  whatever the container was given.
         */
        ST_NO_UNIQUE_ADDRESS_ comparator_t comparator;

        versioned_comparator_t() noexcept
            requires std::is_default_constructible_v<comparator_t>
            : comparator() {}
        explicit versioned_comparator_t(comparator_t const &other) noexcept : comparator(other) {}

        /**
         *  @brief Peels a stored object down to the identifier it orders by.
         *  @note One projection, shared with the unordered path, so an ordered and an unordered
         *      store can never disagree about which part of an entry identifies it.
         */
        template <typename type_>
        decltype(auto) comparable(type_ const &object) const noexcept {
            return identifier_of(object);
        }

        template <typename first_type_, typename second_type_>
        bool per_version_compare(first_type_ const &a, second_type_ const &b) const noexcept {
            auto a_less_b = comparator(comparable(a), comparable(b));
            auto b_less_a = comparator(comparable(b), comparable(a));
            return !a_less_b && !b_less_a ? a.generation < b.generation : a_less_b;
        }

        template <typename first_type_, typename second_type_>
        bool per_key_compare(first_type_ const &a, second_type_ const &b) const noexcept {
            return comparator(comparable(a), comparable(b));
        }

        template <typename first_type_, typename second_type_>
        bool less(first_type_ const &a, second_type_ const &b) const noexcept {
            using first_t = std::remove_reference_t<first_type_>;
            using second_t = std::remove_reference_t<second_type_>;
            static_assert(!std::is_same_v<first_t, watch_t> && !std::is_same_v<second_t, watch_t>,
                          "A watch dates an entry without naming one, so it is compared with `==`, never ordered");
            if constexpr (orderable_per_version<first_t> && orderable_per_version<second_t>)
                return per_version_compare(a, b);
            else return per_key_compare(a, b);
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

/** Hashes a version-decorated object by the identifier inside it, ignoring the metadata. An unordered store
 *  addresses an entry by hash rather than by order, so this is what @c versioned_comparator_t is to an ordered one. */
template <typename hasher_type_>
struct per_key_hasher {
    using is_transparent = void;

    /** The hasher this wrapper was built with, consulted for every element. */
    ST_NO_UNIQUE_ADDRESS_ hasher_type_ hasher;

    per_key_hasher() noexcept
        requires std::is_default_constructible_v<hasher_type_>
        : hasher() {}
    explicit per_key_hasher(hasher_type_ const &other) noexcept : hasher(other) {}

    template <typename type_>
    std::size_t operator()(type_ const &object) const noexcept {
        return hasher(identifier_of(object));
    }
};

/** Compares version-decorated objects by the identifier inside them, ignoring the metadata. Paired with
 *  @c per_key_hasher, since an open-addressed table needs both to place an entry. */
template <typename equals_type_>
struct per_key_equals {
    using is_transparent = void;

    /** The equality this wrapper was built with, consulted for every comparison. */
    ST_NO_UNIQUE_ADDRESS_ equals_type_ equals;

    per_key_equals() noexcept
        requires std::is_default_constructible_v<equals_type_>
        : equals() {}
    explicit per_key_equals(equals_type_ const &other) noexcept : equals(other) {}

    template <typename first_type_, typename second_type_>
    bool operator()(first_type_ const &first, second_type_ const &second) const noexcept {
        return equals(identifier_of(first), identifier_of(second));
    }
};

/**
 *  @brief Compares version-decorated objects by identifier @b and generation, so versions do not
 *      collapse. A bare key matches every version of that key - which is what a visibility walk
 *      asks - while a dated identifier matches exactly one, which is what a commit asks.
 *
 *  Paired with @c per_key_hasher, which keeps peeling to the bare key so every version of a key
 *  shares one probe run; only equality widens. The ordered core needs no counterpart, since
 *  @c versioned_comparator_t::less already routes to @c per_version_compare when both
 *  operands date.
 */
template <typename equals_type_>
struct per_version_equals {
    using is_transparent = void;

    /** The equality this wrapper was built with, consulted for every comparison. */
    ST_NO_UNIQUE_ADDRESS_ equals_type_ equals;

    per_version_equals() noexcept
        requires std::is_default_constructible_v<equals_type_>
        : equals() {}
    explicit per_version_equals(equals_type_ const &other) noexcept : equals(other) {}

    template <typename first_type_, typename second_type_>
    bool operator()(first_type_ const &first, second_type_ const &second) const noexcept {
        if (!equals(identifier_of(first), identifier_of(second))) return false;
        using first_t = std::remove_reference_t<first_type_>;
        using second_t = std::remove_reference_t<second_type_>;
        if constexpr (orderable_per_version<first_t> && orderable_per_version<second_t>)
            return first.generation == second.generation;
        else return true;
    }
};

/**
 *  @brief Whether a store validates what a transaction read, so a commit over it can be refused.
 *
 *  Below @c serializable_k a read files nothing and is answered and forgotten, so a wrapper marking
 *  the partition a read touched would buy a lock for a check nobody makes.
 */
template <typename store_type_>
concept records_what_it_reads = at_least(store_type_::isolation_k, isolation_t::serializable_k);

#pragma endregion Optimistic Concurrency

#pragma region Device Portability

/** Platform cache line size in bytes, typically 64 on modern CPUs. */
inline constexpr std::size_t cache_line_bytes_k = 64;

/**
 *  @brief The bit intrinsics every container here counts slots with.
 *
 *  @warning These are aliases rather than direct calls to @c std because libstdc++'s @c \<bit\> is not
 *      usable from device code. Its functions are @c constexpr, so @c --expt-relaxed-constexpr lets a
 *      kernel call them and they compile without a diagnostic, but @c nvcc folds most of them against a
 *      zeroed argument: with CUDA 12.8 and GCC 14, @c std::popcount over 64 bits, @c std::countr_zero
 *      over either width and @c std::countl_zero over 64 bits each return the answer for an input of
 *      zero. Silently wrong results, not a compile error, which is what makes the alias worth having.
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
 *  Under a device compiler this resolves to @c cuda::atomic_ref at device scope, which is the
 *  widest scope a kernel spanning several blocks needs and the only one under which a header
 *  touched from two blocks is ordered at all. Everywhere else it is @c std::atomic_ref, and the two
 *  agree on the operations used here - @c fetch_or, @c fetch_xor, @c fetch_add, @c fetch_sub and
 *  @c load.
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
 *
 *  Relaxed suffices for a counter read by value - a statistic, or a version stamp compared for
 *  identity and recency - because one location has a total modification order. It publishes
 *  nothing: a caller needing the data a counter describes to be visible must order that itself.
 */
template <typename integral_type_>
constexpr integral_type_ atomic_add_fetch(integral_type_ &counter, integral_type_ addend) noexcept {
    return atomic_ref<integral_type_>(counter).fetch_add(addend, memory_order_relaxed_k) + addend;
}

/**
 *  @brief Relaxed atomic read of a counter other threads may be incrementing. Needed wherever a
 *      plain read would race an @c atomic_add_fetch on the same field.
 *  @warning The counter is read through a @c const path but must not be a @c const @b object - an
 *      @c atomic_ref over one is undefined. Every caller reaches a mutable member of a mutable
 *      container through a @c const member function, which is what makes the cast below sound.
 */
template <typename integral_type_>
constexpr integral_type_ atomic_load(integral_type_ const &counter) noexcept {
    return atomic_ref<integral_type_>(const_cast<integral_type_ &>(counter)).load(memory_order_relaxed_k);
}

/** Relaxed atomic write of a counter other threads may be reading. The mirror of @c atomic_load, and just as
 *  unordered: a caller needing the data a counter describes to be visible alongside it must order that itself. */
template <typename integral_type_>
constexpr void atomic_store(integral_type_ &counter, integral_type_ value) noexcept {
    atomic_ref<integral_type_>(counter).store(value, memory_order_relaxed_k);
}

/** Relaxed atomic decrement of a plain counter, returning the post-decrement value. */
template <typename integral_type_>
constexpr integral_type_ atomic_sub_fetch(integral_type_ &counter, integral_type_ subtrahend) noexcept {
    return atomic_ref<integral_type_>(counter).fetch_sub(subtrahend, memory_order_relaxed_k) - subtrahend;
}

/**
 *  @brief Parks until @p counter stops reading @p observed, so a waiter costs no core while
 *      it waits.
 *  @warning Wakes only where the writer calls @c atomic_notify_all; a bare @c atomic_store
 *      wakes nobody.
 *  @sa The same @c const-path caveat as @c atomic_load applies to the cast.
 */
template <typename integral_type_>
constexpr void atomic_wait(integral_type_ const &counter, integral_type_ observed) noexcept {
    atomic_ref<integral_type_>(const_cast<integral_type_ &>(counter)).wait(observed, memory_order_relaxed_k);
}

/** Wakes every waiter parked on @p counter. The mirror of @c atomic_wait. */
template <typename integral_type_>
constexpr void atomic_notify_all(integral_type_ &counter) noexcept {
    atomic_ref<integral_type_>(counter).notify_all();
}

#pragma endregion Device Portability

#pragma region Extended Atomics

/**
 *  @brief An atomic reference offering the read-modify-writes the standard has no spelling for.
 *
 *  @tparam reference_type_ The reference being checked.
 *  @tparam value_type_ The word it owns.
 *
 *  Two shapes matter here. A @b posted write is one whose result nobody reads, which a target with
 *  a no-return form can fire and never wait on. A @b bounded read-modify-write is one that writes
 *  only while the result stays inside a bound, which a target with a conditional form decides in
 *  one instruction - and, where it refuses, without taking the line exclusively at all.
 *
 *  Nothing here is a dependency: @c atomic_ref does not satisfy this, and every operation below
 *  falls back to the loop or the discarded result it replaces. ForkUnion ships references that do
 *  satisfy it, one per instruction set, so a caller already linking it passes one straight in.
 *
 *  @see https://github.com/ashvardanian/ForkUnion
 */
template <typename reference_type_, typename value_type_>
concept extended_atomic_ref = requires(reference_type_ reference, value_type_ value) {
    reference.add(value, memory_order_relaxed_k);
    reference.sub(value, memory_order_relaxed_k);
    reference.set_bits(value, memory_order_relaxed_k);
    reference.clear_bits(value, memory_order_relaxed_k);
    reference.flip_bits(value, memory_order_relaxed_k);
    reference.fetch_add_if_at_most(value, value, memory_order_relaxed_k);
    reference.fetch_sub_if_at_least(value, value, memory_order_relaxed_k);
};

/** Posts @p addend onto @p word without reading what was there. */
template <template <typename> class atomic_reference_ = atomic_ref, typename integral_type_, typename order_type_>
constexpr void atomic_post_add(integral_type_ &word, integral_type_ addend, order_type_ order) noexcept {
    atomic_reference_<integral_type_> reference(word);
    if constexpr (extended_atomic_ref<atomic_reference_<integral_type_>, integral_type_>) reference.add(addend, order);
    else reference.fetch_add(addend, order);
}

/** Posts @p subtrahend off @p word without reading what was there. */
template <template <typename> class atomic_reference_ = atomic_ref, typename integral_type_, typename order_type_>
constexpr void atomic_post_sub(integral_type_ &word, integral_type_ subtrahend, order_type_ order) noexcept {
    atomic_reference_<integral_type_> reference(word);
    if constexpr (extended_atomic_ref<atomic_reference_<integral_type_>, integral_type_>)
        reference.sub(subtrahend, order);
    else reference.fetch_sub(subtrahend, order);
}

/** Posts @p bits set in @p word without reading what was there. */
template <template <typename> class atomic_reference_ = atomic_ref, typename integral_type_, typename order_type_>
constexpr void atomic_post_set_bits(integral_type_ &word, integral_type_ bits, order_type_ order) noexcept {
    atomic_reference_<integral_type_> reference(word);
    if constexpr (extended_atomic_ref<atomic_reference_<integral_type_>, integral_type_>)
        reference.set_bits(bits, order);
    else reference.fetch_or(bits, order);
}

/** Posts @p bits cleared in @p word without reading what was there. */
template <template <typename> class atomic_reference_ = atomic_ref, typename integral_type_, typename order_type_>
constexpr void atomic_post_clear_bits(integral_type_ &word, integral_type_ bits, order_type_ order) noexcept {
    atomic_reference_<integral_type_> reference(word);
    if constexpr (extended_atomic_ref<atomic_reference_<integral_type_>, integral_type_>)
        reference.clear_bits(bits, order);
    else reference.fetch_and(static_cast<integral_type_>(~bits), order);
}

/** Posts @p bits flipped in @p word without reading what was there. */
template <template <typename> class atomic_reference_ = atomic_ref, typename integral_type_, typename order_type_>
constexpr void atomic_post_flip_bits(integral_type_ &word, integral_type_ bits, order_type_ order) noexcept {
    atomic_reference_<integral_type_> reference(word);
    if constexpr (extended_atomic_ref<atomic_reference_<integral_type_>, integral_type_>)
        reference.flip_bits(bits, order);
    else reference.fetch_xor(bits, order);
}

/**
 *  @brief Adds @p addend to @p word only while the sum stays at or below @p limit.
 *  @return What @p word held before, which is above @p limit exactly when nothing was written.
 *
 *  The bound is the whole point: a predicate a loop would re-check after every failed exchange
 *  becomes an argument the instruction decides on, so a refused caller writes nothing and leaves
 *  the line where it was. Without the extended form this is the compare-exchange loop it replaces.
 */
template <template <typename> class atomic_reference_ = atomic_ref, typename integral_type_, typename order_type_>
[[nodiscard]] constexpr integral_type_ atomic_fetch_add_if_at_most(integral_type_ &word, integral_type_ addend,
                                                                   integral_type_ limit, order_type_ order) noexcept {
    atomic_reference_<integral_type_> reference(word);
    if constexpr (extended_atomic_ref<atomic_reference_<integral_type_>, integral_type_>)
        return reference.fetch_add_if_at_most(addend, limit, order);
    else {
        integral_type_ observed = reference.load(memory_order_acquire_k);
        while (observed <= limit && static_cast<integral_type_>(limit - observed) >= addend &&
               !reference.compare_exchange_weak(observed, static_cast<integral_type_>(observed + addend), order,
                                                memory_order_acquire_k)) {}
        return observed;
    }
}

/**
 *  @brief Subtracts @p subtrahend from @p word only while the difference stays at or above @p floor.
 *  @return What @p word held before, which is below @p floor exactly when nothing was written.
 */
template <template <typename> class atomic_reference_ = atomic_ref, typename integral_type_, typename order_type_>
[[nodiscard]] constexpr integral_type_ atomic_fetch_sub_if_at_least(integral_type_ &word, integral_type_ subtrahend,
                                                                    integral_type_ floor, order_type_ order) noexcept {
    atomic_reference_<integral_type_> reference(word);
    if constexpr (extended_atomic_ref<atomic_reference_<integral_type_>, integral_type_>)
        return reference.fetch_sub_if_at_least(subtrahend, floor, order);
    else {
        integral_type_ observed = reference.load(memory_order_acquire_k);
        while (observed >= floor && static_cast<integral_type_>(observed - floor) >= subtrahend &&
               !reference.compare_exchange_weak(observed, static_cast<integral_type_>(observed - subtrahend), order,
                                                memory_order_acquire_k)) {}
        return observed;
    }
}

#pragma endregion Extended Atomics

#pragma region Numeric Helpers

/**
 *  @brief The larger of two values, and the smaller.
 *
 *  Spelled here rather than pulled from @c \<algorithm\>, which costs about ten thousand
 *  preprocessed lines for these two. It also sidesteps the Windows headers, which define @c min and
 *  @c max as function-like macros unless @c NOMINMAX is set, and would eat an unparenthesized
 *  @c std::max call.
 */
template <typename value_type_>
constexpr value_type_ const &larger_of(value_type_ const &first, value_type_ const &second) noexcept {
    return first < second ? second : first;
}

template <typename value_type_>
constexpr value_type_ const &smaller_of(value_type_ const &first, value_type_ const &second) noexcept {
    return second < first ? second : first;
}

/** How many bits a @c std::size_t holds here, which is not 64 everywhere. */
inline constexpr std::size_t size_bits_k = sizeof(std::size_t) * CHAR_BIT;

/** The largest @c std::size_t, which also serves as the "no such index" sentinel. */
inline constexpr std::size_t size_max_k = static_cast<std::size_t>(-1);

/** The number of bits needed to represent @p x, so @c 0 for zero and @c 1 for one. */
constexpr std::size_t bits_to_hold(std::size_t x) noexcept {
    return size_bits_k - static_cast<std::size_t>(countl_zero(x));
}

/**
 *  @brief Rounds up an integer to the next power of two. Returns 0 for input 0, and 1 for input 1.
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
 *  Masks to the next power of two and redraws when the value lands above the bound, which is
 *  unbiased because every value under the mask is equally likely and the surplus is simply
 *  discarded. At least half of the masked range is accepted, so fewer than two draws are expected.
 *
 *  @param[in] generator Any uniform random bit generator, which is what @c std::mt19937 and
 *      friends are.
 *  @param[in] bound One past the largest value that may be returned; zero and one both
 *      draw nothing.
 *  @note A generator narrower than the bound - @c std::mt19937 yields 32 bits - is called
 *      repeatedly and the results concatenated, since masking a short draw could never reach the
 *      high bits.
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
 *  Order statistics allow O(log n) access to the k-th smallest element (select) and finding the
 *  rank (position) of an element. Weight-balanced trees support these operations, while standard
 *  AVL trees do not.
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

/** A store whose members are bare keys, which is what @c *_set names. */
template <typename store_type_>
concept set_shaped_store = !is_mapping<typename store_type_::value_t>;

/** A store whose members pair a key with a value, which is what @c *_map names. */
template <typename store_type_>
concept map_shaped_store = is_mapping<typename store_type_::value_t>;

/** A container that reports whether it maps keys to values, and how reads are delivered. Every container in this
 *  library carries these, and the test suites assert on them. */
template <typename collection_type_>
concept tagged_collection = requires {
    typename collection_type_::value_type;
    typename collection_type_::key_type;
    typename collection_type_::is_associative;
    { collection_type_::is_associative::value } -> std::convertible_to<bool>;
};

/**
 *  @brief What an absorbed batch does with a key the destination already holds.
 *
 *  Implementation vocabulary rather than a tag: a caller picks between these by naming a verb, and
 *  the three verbs share one staging helper that this tells apart.
 */
enum class incumbent_policy_t : std::uint8_t {

    /** The element already stored stays, and the newcomer is dropped. */
    keeps_the_incumbent_k,

    /** The newcomer takes the key, overwriting what was stored under it. */
    takes_the_newcomer_k,

    /** Neither is stored: a key arriving over one already there refuses the whole batch. */
    refuses_the_newcomer_k,
};

/** Whether the collection writes a whole range, overwriting the keys already there. */
template <typename collection_type_>
concept offers_batch_upsert = requires(collection_type_ &collection, typename collection_type_::value_type *cursor) {
    { collection.upsert(cursor, cursor) } -> std::same_as<status_t>;
};

/** Whether the collection writes a whole range, refusing it over a key already there. */
template <typename collection_type_>
concept offers_batch_insert = requires(collection_type_ &collection, typename collection_type_::value_type *cursor) {
    { collection.insert(cursor, cursor) } -> std::same_as<status_t>;
};

/** Whether the collection writes a whole range, leaving the keys already there alone. */
template <typename collection_type_>
concept offers_batch_insert_if_missing =
    requires(collection_type_ &collection, typename collection_type_::value_type *cursor) {
        { collection.insert_if_missing(cursor, cursor) } -> std::same_as<status_t>;
    };

/**
 *  @brief Whether a container takes a whole batch or none of it, under every verb that writes one.
 *
 *  Checks only the shape the promise needs - three range modifiers, each reporting through a status
 *  so a refusal has somewhere to go. The rollback itself no concept can see; that is what a suite
 *  refusing the allocator at every point of a batch is for.
 */
template <typename collection_type_>
concept batches_atomically = offers_batch_upsert<collection_type_> && offers_batch_insert<collection_type_> &&
                             offers_batch_insert_if_missing<collection_type_>;

#pragma endregion Container Traits

#pragma region Store Tiers

/**
 *  @brief The tier every associative core must satisfy to back a transactional store. Point lookup,
 *      membership, size, and clearing - nothing here implies an ordering over keys, so an
 *      open-addressed table can meet it as readily as a search tree.
 *
 *  @note Node transfer - @c extract and @c merge - is deliberately absent. Moving nodes between a
 *      transaction's staging store and the committed one is how the tree adapter stages today, but
 *      it is an implementation strategy rather than a requirement: a chain-based store stages by
 *      pushing a version onto a key's chain and never moves a node at all.
 */
template <typename collection_type_>
concept key_addressable_collection =
    tagged_collection<collection_type_> && requires(collection_type_ &collection, collection_type_ const &constant,
                                                    typename collection_type_::key_type const &key) {
        { constant.size() } -> std::convertible_to<std::size_t>;
        { constant.empty() } -> std::convertible_to<bool>;
        { constant.contains(key) } -> std::same_as<expected<bool>>;
        collection.clear();
    };

/**
 *  @brief The additional tier an ordered core satisfies, and an unordered one never can.
 *
 *  Bounds and half-open range erasure are the load-bearing pair: the transactional store expresses
 *  "every version of this key older than mine" as a range, and the Python binding walks a cursor by
 *  repeatedly asking for the exclusive successor of the last key it yielded.
 *
 *  @note A core's bounds are iterator-shaped, taking the sought key alone. The callback-shaped
 *      @c lower_bound(key, found, missing) belongs to the transactional adapter above it, which
 *      cannot hand out iterators because a version may be staged and invisible. Do not conflate
 *      the two.
 */
template <typename collection_type_>
concept ordered_collection = key_addressable_collection<collection_type_> &&
                             requires(collection_type_ &collection, typename collection_type_::key_type const &key) {
                                 collection.lower_bound(key);
                                 collection.upper_bound(key);
                                 collection.erase_range(key, key);
                             };

#pragma endregion Store Tiers

#pragma region Store Surfaces

/** Whether the store carries @c insert_or_assign, whose contract is to write over an occupied key. */
template <typename store_type_>
concept offers_insert_or_assign = requires(store_type_ &store, typename store_type_::value_t &&element) {
    store.insert_or_assign(std::move(element));
};

/** Whether the store counts the versions it keeps, in total and for one key. */
template <typename store_type_>
concept offers_versions_count = requires(store_type_ const &store, typename store_type_::identifier_t const &key) {
    store.versions_count();
    store.versions_count(key);
};

/** Whether the store visits every member, in whatever order it holds them. */
template <typename store_type_>
concept offers_for_each = requires(store_type_ const &store, no_op_t callback) { store.for_each(callback); };

/** Whether the store hands an element back by value rather than to a callback. */
template <typename store_type_>
concept offers_find_copy =
    requires(store_type_ const &store, typename store_type_::identifier_t const &key) { store.find_copy(key); };

/** Whether the store erases every key at or after a bound. */
template <typename store_type_>
concept offers_erase_from = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                     no_op_t callback) { store.erase_from(key, callback); };

/** Whether the store erases every key before a bound. */
template <typename store_type_>
concept offers_erase_up_to = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                      no_op_t callback) { store.erase_up_to(key, callback); };

/** Whether the store answers the first member at or after a key. */
template <typename store_type_>
concept offers_lower_bound = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                      no_op_t callback) { store.lower_bound(key, callback, callback); };

/** Whether the store answers the first member strictly after a key. */
template <typename store_type_>
concept offers_upper_bound = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                      no_op_t callback) { store.upper_bound(key, callback, callback); };

/** Whether the store walks a half-open window of the keyspace. */
template <typename store_type_>
concept offers_range = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                no_op_t callback) { store.range(key, key, callback); };

/** Whether the store erases a half-open window of the keyspace. */
template <typename store_type_>
concept offers_erase_range = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                      no_op_t callback) { store.erase_range(key, key, callback); };

/** Whether the store answers the member at an ordinal. */
template <typename store_type_>
concept offers_select = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                 std::size_t ordinal, no_op_t callback) { store.select(ordinal, callback, callback); };

/** Whether the store answers how many members precede a key. */
template <typename store_type_>
concept offers_rank = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                               std::size_t ordinal, no_op_t callback) { store.rank(key, callback, callback); };

/** Whether the store names the snapshot no open reader sits below. */
template <typename store_type_>
concept offers_low_water_mark = requires(store_type_ const &store) { store.low_water_mark(); };

/** Whether the store hands over every member equal to a key, not only the first. */
template <typename store_type_>
concept offers_equal_range = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                      no_op_t callback) { store.equal_range(key, callback); };

/** Whether the store names its smallest member without being given a bound to beat. */
template <typename store_type_>
concept offers_smallest = requires(store_type_ const &store, no_op_t callback) { store.smallest(callback, callback); };

/** Whether the store removes its smallest member as one operation. */
template <typename store_type_>
concept offers_pop_smallest =
    requires(store_type_ &store, no_op_t callback) { store.pop_smallest(callback, callback); };

/** Whether the store reclaims superseded versions on demand. */
template <typename store_type_>
concept offers_vacuum = requires(store_type_ &store) { store.vacuum(); };

/** Whether the store reclaims a window of the keyspace rather than all of it. */
template <typename store_type_>
concept offers_vacuum_window =
    requires(store_type_ &store, typename store_type_::identifier_t const &key) { store.vacuum(key, key); };

/** Whether the store carries @c update, whose contract is to refuse an absent key. */
template <typename store_type_>
concept offers_update =
    requires(store_type_ &store, typename store_type_::value_t &&element) { store.update(std::move(element)); };

/** Whether the store carries @c insert, whose contract is to refuse an occupied key. */
template <typename store_type_>
concept offers_insert =
    requires(store_type_ &store, typename store_type_::value_t &&element) { store.insert(std::move(element)); };

/** Whether the store answers both ends of a window, which is less than the whole ordered surface. */
template <typename store_type_>
concept offers_both_bounds = offers_lower_bound<store_type_> && offers_upper_bound<store_type_>;

/** Whether the store can open a merged walk at all, from either end it knows how to name. */
template <typename store_type_>
concept offers_a_merged_walk = offers_smallest<store_type_> || offers_select<store_type_>;

/** Whether the store carries @c insert_if_missing, which leaves a key already there alone. */
template <typename store_type_>
concept offers_insert_if_missing = requires(store_type_ &store, typename store_type_::value_t &&element) {
    store.insert_if_missing(std::move(element));
};

/** Whether the three-argument @c insert also hands over the element already holding the key. */
template <typename store_type_>
concept offers_insert_naming_occupant =
    requires(store_type_ &store, typename store_type_::value_t &&element, no_op_t callback) {
        store.insert(std::move(element), callback, callback);
    };

/** Whether the store rewrites the mapped side of a window in place. */
template <typename store_type_>
concept offers_update_range = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                       no_op_t callback) { store.update_range(key, key, callback); };

/** Whether the store draws one member of a window at random. */
template <typename store_type_>
concept offers_sample_one = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                     no_op_t callback) { store.sample_one(key, key, callback, callback); };

/** Whether the store fills a reservoir from a window. */
template <typename store_type_>
concept offers_sample_reservoir =
    requires(store_type_ const &store, typename store_type_::identifier_t const &key, no_op_t callback,
             std::size_t seen) { store.sample_reservoir(key, key, callback, seen, seen, callback); };

/** Whether an open transaction stages a tombstone for every member it reads. */
template <typename store_type_>
concept transaction_offers_clear = requires(typename store_type_::transaction_t &transaction) { transaction.clear(); };

/** Whether an open transaction counts the members matching a key. */
template <typename store_type_>
concept transaction_offers_count = requires(typename store_type_::transaction_t const &transaction,
                                            typename store_type_::identifier_t const &key) { transaction.count(key); };

/** Whether an open transaction visits what its snapshot and its own writes show. */
template <typename store_type_>
concept transaction_offers_for_each = requires(typename store_type_::transaction_t const &transaction,
                                               no_op_t callback) { transaction.for_each(callback); };

/** Whether an open transaction erases a half-open window of its own keys. */
template <typename store_type_>
concept transaction_offers_erase_range =
    requires(typename store_type_::transaction_t &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.erase_range(key, key, callback); };

/** Whether an open transaction erases every key at or after a bound. */
template <typename store_type_>
concept transaction_offers_erase_from =
    requires(typename store_type_::transaction_t &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.erase_from(key, callback); };

/** Whether an open transaction erases every key before a bound. */
template <typename store_type_>
concept transaction_offers_erase_up_to =
    requires(typename store_type_::transaction_t &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.erase_up_to(key, callback); };

/** Whether an open transaction answers the first member strictly after a key. */
template <typename store_type_>
concept transaction_offers_upper_bound =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.upper_bound(key, callback, callback); };

/** Whether an open transaction answers an inclusive bound, which a merged scan seeds from. */
template <typename store_type_>
concept transaction_offers_lower_bound =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.lower_bound(key, callback, callback); };

/** Whether an open transaction hands over every member equal to a key. */
template <typename store_type_>
concept transaction_offers_equal_range =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.equal_range(key, callback); };

/** Whether an open transaction carries @c update, whose contract is to refuse an absent key. */
template <typename store_type_>
concept transaction_offers_update =
    requires(typename store_type_::transaction_t &transaction, typename store_type_::value_t &&element) {
        transaction.update(std::move(element));
    };

/** Whether an open transaction carries @c insert, whose contract is to refuse an occupied key. */
template <typename store_type_>
concept transaction_offers_insert =
    requires(typename store_type_::transaction_t &transaction, typename store_type_::value_t &&element) {
        transaction.insert(std::move(element));
    };

/** Whether an open transaction says if anything is staged. */
template <typename store_type_>
concept transaction_offers_has_changes =
    requires(typename store_type_::transaction_t const &transaction) { transaction.has_changes(); };

/** Whether an open transaction says how much is staged. */
template <typename store_type_>
concept transaction_offers_changes_count =
    requires(typename store_type_::transaction_t const &transaction) { transaction.changes_count(); };

/** Whether a transaction's three-argument @c insert also hands over the element holding the key. */
template <typename store_type_>
concept transaction_offers_insert_naming_occupant =
    requires(typename store_type_::transaction_t &transaction, typename store_type_::value_t &&element,
             no_op_t callback) { transaction.insert(std::move(element), callback, callback); };

/** Whether an open transaction rewrites the mapped side of a window in place. */
template <typename store_type_>
concept transaction_offers_update_range =
    requires(typename store_type_::transaction_t &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.update_range(key, key, callback); };

/** Whether a transaction carries @c insert_if_missing, whose contract is to leave an occupied key. */
template <typename store_type_>
concept transaction_offers_insert_if_missing =
    requires(typename store_type_::transaction_t &transaction, typename store_type_::value_t &&element) {
        transaction.insert_if_missing(std::move(element));
    };

/** Whether the transaction hands an element back by value rather than to a callback. */
template <typename store_type_>
concept transaction_offers_find_copy =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key) {
        transaction.find_copy(key);
    };

/** Whether the transaction names the generation it writes under. */
template <typename store_type_>
concept transaction_offers_generation =
    requires(typename store_type_::transaction_t const &transaction) { transaction.generation(); };

/** Whether an open transaction answers both ends of a window. */
template <typename store_type_>
concept transaction_offers_both_bounds =
    transaction_offers_lower_bound<store_type_> && transaction_offers_upper_bound<store_type_>;

/** Whether an open transaction walks a half-open window of the keyspace. */
template <typename store_type_>
concept transaction_offers_range =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.range(key, key, callback); };

/** Whether an open transaction answers the member at an ordinal. */
template <typename store_type_>
concept transaction_offers_select =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.select(std::size_t {0}, callback, callback); };

/** Whether an open transaction answers how many members precede a key. */
template <typename store_type_>
concept transaction_offers_rank =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.rank(key, callback, callback); };

/** Whether an open transaction names its smallest member without a bound to beat. */
template <typename store_type_>
concept transaction_offers_smallest = requires(typename store_type_::transaction_t const &transaction,
                                               no_op_t callback) { transaction.smallest(callback, callback); };

/** The ordered surface, spelled from its atoms so a second definition of "ordered" cannot drift from this one. */
template <typename store_type_>
concept offers_ordered_surface = offers_lower_bound<store_type_> && offers_upper_bound<store_type_> &&
                                 offers_range<store_type_> && offers_erase_range<store_type_>;

/** The ordinal surface, which only a core summing subtree counts can answer. */
template <typename store_type_>
concept offers_order_statistics = offers_select<store_type_> && offers_rank<store_type_>;

/** Erasure of a window with one end left open, at both ends. */
template <typename store_type_>
concept offers_open_range_erasure = offers_erase_from<store_type_> && offers_erase_up_to<store_type_>;

/** Every window erase an open transaction stages, bounded and open-ended alike. */
template <typename store_type_>
concept transaction_offers_range_erasure =
    transaction_offers_erase_range<store_type_> && transaction_offers_erase_from<store_type_> &&
    transaction_offers_erase_up_to<store_type_>;

/** The ordinal surface of an open transaction. */
template <typename store_type_>
concept transaction_offers_order_statistics =
    transaction_offers_select<store_type_> && transaction_offers_rank<store_type_>;

/** What an open transaction says about its own staging, whether and how much. */
template <typename store_type_>
concept transaction_offers_change_report =
    transaction_offers_has_changes<store_type_> && transaction_offers_changes_count<store_type_>;

/** Whether an open transaction walks every key at or after a bound. */
template <typename store_type_>
concept transaction_offers_range_from =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.range_from(key, callback); };

/** Whether an open transaction walks every key before a bound. */
template <typename store_type_>
concept transaction_offers_range_up_to =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.range_up_to(key, callback); };

/** Whether an open transaction copies out the first member at or after a key. */
template <typename store_type_>
concept transaction_offers_lower_bound_copy =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key) {
        transaction.lower_bound_copy(key);
    };

/** Whether an open transaction copies out the first member after a key. */
template <typename store_type_>
concept transaction_offers_upper_bound_copy =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key) {
        transaction.upper_bound_copy(key);
    };

/** Whether an open transaction draws one member of a window at random. */
template <typename store_type_>
concept transaction_offers_sample_one =
    requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
             no_op_t callback) { transaction.sample_one(key, key, callback, callback); };

/** Whether an open transaction fills a reservoir from a window. */
template <typename store_type_>
concept transaction_offers_sample_reservoir = requires(
    typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
    no_op_t callback, std::size_t seen) { transaction.sample_reservoir(key, key, callback, seen, seen, callback); };

/** Whether the store says how many keys its ordinal surface indexes. */
template <typename store_type_>
concept offers_ranked_size = requires(store_type_ const &store) { store.ranked_size(); };

/** Whether an open transaction sizes its watch list ahead of the writes that fill it. */
template <typename store_type_>
concept transaction_offers_reserve =
    requires(typename store_type_::transaction_t &transaction, std::size_t size) { transaction.reserve(size); };

#pragma endregion Store Surfaces

#pragma region Shared Clock

/**
 *  @brief What a store keeping no stamps contributes to a shard set, which is nothing at all.
 *
 *  Named so a wrapper declares its clock and its reader's claim unconditionally, and pays nothing
 *  for either where there is no clock to share.
 */
struct no_clock_t {

    /** The claim a reader of such a store never takes. */
    using snapshot_lease_t = no_clock_t;
};

/** The clock a store shares with its siblings, or @c no_clock_t when it keeps no stamps. */
template <typename store_type_, typename = void>
struct shared_clock_of {
    using type = no_clock_t;
};
template <typename store_type_>
struct shared_clock_of<store_type_, std::void_t<typename store_type_::clock_t>> {
    using type = typename store_type_::clock_t;
};

/** Whether the store takes a clock from outside, so a shard set can stamp every part alike. */
template <typename store_type_, typename clock_type_>
concept offers_attach_clock = requires(store_type_ &store, clock_type_ &clock) { store.attach_clock(clock); };

/** Whether the store opens a transaction at a stamp somebody else drew. */
template <typename store_type_>
concept offers_transaction_at =
    requires(store_type_ &store, typename store_type_::generation_t stamp) { store.transaction_at(stamp, stamp); };

/** Both halves of sharing a clock: taking one, and opening at a stamp drawn from it. */
template <typename store_type_, typename clock_type_>
concept offers_shared_clock = offers_attach_clock<store_type_, clock_type_> && offers_transaction_at<store_type_>;

#pragma endregion Shared Clock

#pragma region Storage Shape

/**
 *  @brief How a container family restates itself over version-decorated elements, and how one such
 *      entry is reached, replaced and taken back out.
 *
 *  An ordered core addresses an entry by a comparator and rebinds on one; an unordered core
 *  addresses it by a hasher paired with an equality and rebinds on both. Everything a transactional
 *  store does that is not itself ordered - staging, committing, reading one key - goes through
 *  this, so the store never names either shape.
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

    /** The same storage keyed by identifier @b and generation, holding one entry per version. Unchanged here, since
     *  the ordered comparator already breaks a tie on the generation. */
    template <typename element_type_>
    using rebind_dated = rebind<element_type_>;

    /** Builds an empty storage over @c element_type_, seeded from a bare comparator. */
    template <typename element_type_, typename allocator_type_>
    static rebind<element_type_> build(addressing_source_t const &source, allocator_type_ const &allocator) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(addressing_t(source), typename storage_t::allocator_type(allocator));
    }

    /** Builds an empty storage over @c element_type_, leaving the addressing defaulted. */
    template <typename element_type_, typename allocator_type_>
    static rebind<element_type_> build(allocator_type_ const &allocator) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(typename storage_t::allocator_type(allocator));
    }

    /** Builds an empty storage over @c element_type_ addressed exactly as @p existing is. */
    template <typename element_type_, typename storage_type_>
    static rebind<element_type_> sibling(storage_type_ const &existing) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(existing.key_comp(), typename storage_t::allocator_type(existing.allocator()));
    }

    /** The allocator a storage of this family was built with. */
    template <typename storage_type_>
    static decltype(auto) allocator_of(storage_type_ const &storage) noexcept {
        return storage.allocator();
    }

    /** A node-based core has nothing to prepay; the placeholder pass reserves it. */
    template <typename storage_type_>
    static status_t prepare(storage_type_ &, std::size_t) noexcept {
        return success_k;
    }

    /** Files @p element under its own key, overwriting whatever shared it. */
    template <typename storage_type_, typename element_type_>
    static status_t upsert(storage_type_ &storage, element_type_ &&element) noexcept {
        auto result = storage.upsert(std::move(element));
        return result.failed() ? out_of_memory_heap_k : success_k;
    }

    /**
     *  @brief Takes the entry filed under @p identifier out whole, handing its payload to
     *      @p destination.
     *  @return Whether an entry was there to take.
     *  @note The entry leaves the index before anything is moved out of it, since the key the index
     *      addresses it by lives inside the payload and a moved-from key neither compares
     *      nor hashes.
     */
    template <typename storage_type_, typename identifier_type_, typename payload_type_>
    static bool extract_payload(storage_type_ &storage, identifier_type_ const &identifier,
                                payload_type_ &destination) noexcept {
        auto extracted = storage.extract(identifier);
        if (!extracted.node_ptr_) return false;
        destination = std::move(extracted.node_ptr_->payload.head);
        return true;
    }
};

/** The unordered shape, selected by the core naming a @c hasher the way every table does. */
template <typename collection_type_, typename value_type_>
struct versioned_storage_for<collection_type_, value_type_, std::void_t<typename collection_type_::hasher>> {

    using versioning_t = versioning_for<value_type_, typename collection_type_::key_equal>;
    using addressing_source_t = typename collection_type_::key_equal;
    using addressing_t = per_key_hasher<typename collection_type_::hasher>;
    using equality_t = per_key_equals<typename collection_type_::key_equal>;

    template <typename element_type_>
    using rebind = typename collection_type_::template rebind<element_type_, addressing_t, equality_t>;

    /** The equality that keeps one slot per version, rather than one per key. */
    using dated_equality_t = per_version_equals<typename collection_type_::key_equal>;

    /** The same storage keyed by identifier @b and generation, holding one entry per version. The hasher is
     *  untouched, so every version of a key shares one probe run. */
    template <typename element_type_>
    using rebind_dated = typename collection_type_::template rebind<element_type_, addressing_t, dated_equality_t>;

    /** Builds an empty storage over @c element_type_, seeded from a bare equality. */
    template <typename element_type_, typename allocator_type_>
    static rebind<element_type_> build(addressing_source_t const &source, allocator_type_ const &allocator) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(addressing_t {}, equality_t(source), typename storage_t::allocator_type(allocator));
    }

    /** Builds an empty storage over @c element_type_, leaving the addressing defaulted. */
    template <typename element_type_, typename allocator_type_>
    static rebind<element_type_> build(allocator_type_ const &allocator) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(addressing_t {}, equality_t {}, typename storage_t::allocator_type(allocator));
    }

    /** Builds an empty storage over @c element_type_ addressed exactly as @p existing is. */
    template <typename element_type_, typename storage_type_>
    static rebind<element_type_> sibling(storage_type_ const &existing) noexcept {
        using storage_t = rebind<element_type_>;
        return storage_t(existing.hash_function(), existing.key_eq(),
                         typename storage_t::allocator_type(existing.get_allocator()));
    }

    /** The allocator a storage of this family was built with. */
    template <typename storage_type_>
    static decltype(auto) allocator_of(storage_type_ const &storage) noexcept {
        return storage.get_allocator();
    }

    /** A slab grows once, so every later upsert lands in a slot already paid for. */
    template <typename storage_type_>
    static status_t prepare(storage_type_ &storage, std::size_t count) noexcept {
        using slot_count_t = typename storage_type_::size_type;
        using reserve_result_t = typename storage_type_::reserve_result_t;
        auto const grown = storage.reserve_more(static_cast<slot_count_t>(count));
        return grown == reserve_result_t::failed_k ? out_of_memory_heap_k : success_k;
    }

    /** Files @p element under its own key, overwriting whatever shared it. */
    template <typename storage_type_, typename element_type_>
    static status_t upsert(storage_type_ &storage, element_type_ &&element) noexcept {
        return storage.upsert(std::move(element));
    }

    /**
     *  @brief Takes the entry filed under @p identifier out whole, handing its payload to
     *      @p destination.
     *  @return Whether an entry was there to take.
     *  @note The slot is located first and erased through that position, so the erasure never
     *      probes for a key the harvest has already emptied.
     */
    template <typename storage_type_, typename identifier_type_, typename payload_type_>
    static bool extract_payload(storage_type_ &storage, identifier_type_ const &identifier,
                                payload_type_ &destination) noexcept {
        auto found = storage.find(identifier);
        if (found == storage.end()) return false;
        using value_t = std::remove_const_t<std::remove_reference_t<decltype(*found)>>;
        destination = std::move(const_cast<value_t &>(*found).head);
        storage.erase(found);
        return true;
    }
};

/**
 *  @brief The owned element a core stores, as distinct from the view it hands out.
 *
 *  A node-based container returns a reference to the element it holds, so the two coincide; an
 *  open-addressed table keeps keys and values in separate regions and can only view an entry as a
 *  pair of references, which nothing above it can own, copy, or version.
 */
template <typename collection_type_, typename = void>
struct owned_value_of {
    using type = typename collection_type_::value_type;
};

template <typename collection_type_>
struct owned_value_of<collection_type_, std::void_t<typename collection_type_::owned_value_type>> {
    using type = typename collection_type_::owned_value_type;
};

/** The node type a linked storage hands out, or @c void where the storage has no nodes. An ordered surface walks
 *  nodes directly; an open-addressed table has none to walk, and naming one unconditionally would break the adapter
 *  at its own definition rather than at the call. */
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
 *  @brief A store whose transactions validate what they watched and stage before they commit.
 *
 *  The isolation floor is @c read_committed_k, which is also the weakest level named: a validated
 *  optimistic transaction stages behind a flag no reader honours, so a store satisfying the rest
 *  cannot expose an uncommitted write, and a store that could is describing something else.
 *
 *  @c reserve is deliberately absent: it is a capacity hint over the watch list, not part of the
 *  contract, and a sharded store has no single list to size.
 *
 *  A transaction must move, must move-assign, and must not copy. The middle requirement is what
 *  catches a defaulted move assignment the compiler quietly deleted - a reference or const member
 *  is enough to do it - which leaves a transaction that can be built into a container but never
 *  rearranged inside one. Movable-and-not-move-assignable is almost never intended, and it fails at
 *  the call site rather than at the declaration, so it is asserted here instead.
 */
template <typename store_type_>
concept optimistically_concurrent_store =
    requires(store_type_ &store, typename store_type_::transaction_t &transaction) {
        typename store_type_::transaction_t;
        requires store_type_::is_transactional::value;
        requires std::is_nothrow_move_constructible_v<typename store_type_::transaction_t>;
        requires std::is_move_assignable_v<typename store_type_::transaction_t>;
        requires !std::is_copy_constructible_v<typename store_type_::transaction_t>;
        requires at_least(store_type_::isolation_k, isolation_t::read_committed_k);
        { store.transaction() } noexcept -> std::same_as<expected<typename store_type_::transaction_t>>;
        { transaction.watch(std::declval<typename store_type_::identifier_t>()) } noexcept -> std::same_as<status_t>;
        { transaction.stage() } noexcept -> std::same_as<status_t>;
        { transaction.commit() } noexcept -> std::same_as<status_t>;
        { transaction.rollback() } noexcept -> std::same_as<status_t>;
        { transaction.reset() } noexcept -> std::same_as<status_t>;
    };

/**
 *  @brief Stages @p stage_changes into @p transaction and commits it, trying at most @p
 *      attempts times.
 *  @param[in] stage_changes Invoked as @c stage_changes(transaction) on every attempt. Must be
 *      @c noexcept.
 *  @return Success; the first failure that is not a conflict; or @c operation_would_block_k once
 *      every attempt met a conflict.
 *
 *  Every failed attempt is reset before the next is made or the failure is returned, so nothing it
 *  staged stays behind and a retry recomputes from the newest snapshot. What a conflict means and
 *  when one is raised is the store's isolation level, which this neither weakens nor strengthens.
 *
 *  @warning A transaction opened at an older snapshot, such as one adopting a reader's, reads at
 *      the newest one from its second attempt on.
 */
template <typename transaction_type_, typename stage_changes_type_>
status_t commit_with_retries(transaction_type_ &transaction, std::size_t attempts,
                             stage_changes_type_ &&stage_changes) noexcept {
    for (std::size_t attempt = 0; attempt != attempts; ++attempt) {
        status_t status = stage_changes(transaction);
        if (succeeded(status)) status = transaction.stage();
        if (succeeded(status)) status = transaction.commit();
        if (succeeded(status)) return status;

        status_t const discarded = transaction.reset();
        if (!conflicted(status)) return status;
        if (failed(discarded)) return discarded;
    }
    return status_t::operation_would_block_k;
}

/**
 *  @brief Whether an open transaction can be asked to commit in two steps rather than one.
 *
 *  A commit spanning several stores has to learn that every one of them may proceed before any of
 *  them writes. A transaction offering only @c commit decides and writes in the same call, so a
 *  caller cannot ask first, and a refusal from a later participant arrives over writes an earlier
 *  one has already published.
 *
 *  @c publish_under is the no-stamp spelling: an engine keeping a clock draws its own stamp inside
 *  it, and one keeping none orders its own versions. Neither can refuse.
 */
template <typename transaction_type_>
concept splits_its_commit = requires(transaction_type_ &transaction) {
    { transaction.validate_for_commit() } noexcept -> std::same_as<status_t>;
    transaction.publish_under();
};

/**
 *  @brief A two-phase commit over several stores at once.
 *
 *  @tparam store_types_ The stores taking part, each staging and committing on its own.
 */
template <optimistically_concurrent_store... store_types_>
class transaction_group {
  public:
    static constexpr std::size_t participants_k = sizeof...(store_types_);
    static_assert(participants_k > 0, "a group needs at least one participant");

    using transactions_t = std::tuple<typename store_types_::transaction_t...>;

  private:
    transactions_t transactions_;

    /** Positions into @c transactions_, ordered by the address of the store each belongs to. */
    std::array<std::size_t, participants_k> order_ {};
    staging_t staging_ {staging_t::pending_k};

    static_assert((std::is_nothrow_move_constructible_v<typename store_types_::transaction_t> && ...),
                  "a group moves its participants into place, so each must move without throwing");

    explicit transaction_group(transactions_t &&opened, std::array<void const *, participants_k> const &stores) noexcept
        : transactions_(std::move(opened)) {
        for (std::size_t position = 0; position != participants_k; ++position) order_[position] = position;
        // Insertion sort: the participant count is a handful, and this keeps the header free of
        // `<algorithm>` for a sort that will never see more than a few elements. Builtin `>` over
        // pointers into unrelated objects is unspecified, so the total order is asked for by name.
        auto const after = [](void const *first, void const *second) noexcept {
            return std::compare_three_way {}(first, second) > 0;
        };
        for (std::size_t position = 1; position != participants_k; ++position) {
            std::size_t const carried = order_[position];
            std::size_t scan = position;
            while (scan != 0 && after(stores[order_[scan - 1]], stores[carried])) {
                order_[scan] = order_[scan - 1];
                --scan;
            }
            order_[scan] = carried;
        }
    }

    /** Applies @p visitor to the participant sitting at @p position of @c transactions_. A fold over the pack rather
     *  than a jump table, since the count is known and tiny. */
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
     *
     *  @return The group; @c invalid_argument_k when one store is named twice; or the first store's
     *      own refusal, which leaves the transactions opened before it to their destructors.
     *
     *  A store named twice would take two participants over one snapshot, neither seeing the
     *  other's writes, so the later commit overwrites the earlier one and no conflict is reported.
     */
    static expected<transaction_group> make(store_types_ &...stores) noexcept {
        std::array<void const *, participants_k> const addresses {static_cast<void const *>(&stores)...};
        // Quadratic over a handful of participants known at compile time, which beats a set here.
        for (std::size_t position = 1; position != participants_k; ++position)
            for (std::size_t earlier = 0; earlier != position; ++earlier)
                if (addresses[position] == addresses[earlier]) return status_t::invalid_argument_k;

        std::tuple<expected<typename store_types_::transaction_t>...> opened {stores.transaction()...};
        status_t const refused = std::apply(
            [](auto const &...maybe) noexcept {
                status_t first = success_k;
                ((first = first_failure(first, maybe.status())), ...);
                return first;
            },
            opened);
        if (failed(refused)) return refused;

        auto moved = std::apply([](auto &...maybe) noexcept { return transactions_t {std::move(*maybe)...}; }, opened);
        return transaction_group {std::move(moved), addresses};
    }

    /** This group's participant in @p store_index_, counted in the caller's argument order. */
    template <std::size_t store_index_>
    auto &participant() noexcept {
        static_assert(store_index_ < participants_k, "no such participant");
        return std::get<store_index_>(transactions_);
    }

    /** Whether the group's writes are sitting in their stores, invisible. */
    staging_t staging() const noexcept { return staging_; }

    /**
     *  @brief Validates every watch and reserves every write, undoing all of it if one refuses.
     *
     *  A partial stage is never observable. The undo rolls the staged prefix back rather than
     *  resetting it, so the caller's pending writes survive and the group can be retried.
     */
    status_t stage() noexcept {
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

    /**
     *  @brief Whether this group asks every participant before letting any of them write.
     *
     *  Needs all of them to offer the split, since one participant deciding and writing in the same
     *  call puts a refusal from a later one over writes an earlier one already published. A lone
     *  participant is excluded because it has nothing to tear against, and asking it separately
     *  would only widen the window a wrapper's own lock closes when it commits in one call.
     */
    static constexpr bool asks_before_writing_k =
        participants_k > 1 && (splits_its_commit<typename store_types_::transaction_t> && ...);

    /**
     *  @brief Publishes the participants, drawing each one's commit stamp.
     *
     *  @return Success; whichever check turned a participant away; or @c operation_not_permitted_k
     *      when the group is not staged.
     *
     *  Where @c asks_before_writing_k, every participant is asked before any writes, so a refusal
     *  publishes nothing and the group stays staged, which is the one state @c rollback accepts. A
     *  participant may still be committed over between the two passes - what the split buys is that
     *  the second pass cannot refuse, not that nothing can change beneath it.
     *
     *  Where it does not hold, each participant is committed in turn. A refusal by the first has
     *  published nothing and leaves the group staged too. A refusal by any later one drops the
     *  group to pending, which refuses both @c commit and @c rollback - the position decides that,
     *  not whether anything was published, since a torn commit cannot be told apart from an untorn
     *  one without asking every participant what it did. Only @c reset then clears the rest.
     */
    status_t commit() noexcept {
        if (staging_ != staging_t::staged_k) return operation_not_permitted_k;
        if constexpr (asks_before_writing_k) {
            for (std::size_t position = 0; position != participants_k; ++position)
                if (status_t const refused = visit_at_(
                        order_[position], [](auto &transaction) noexcept { return transaction.validate_for_commit(); });
                    failed(refused))
                    return refused;

            for (std::size_t position = 0; position != participants_k; ++position) {
                [[maybe_unused]] status_t const published = visit_at_(order_[position], [](auto &transaction) noexcept {
                    transaction.publish_under();
                    return success_k;
                });
            }
        }
        else {
            for (std::size_t position = 0; position != participants_k; ++position)
                if (status_t const refused =
                        visit_at_(order_[position], [](auto &transaction) noexcept { return transaction.commit(); });
                    failed(refused)) {
                    // Whatever the earlier participants published is out, so the group stops calling
                    // itself staged and leaves `reset` to discard the rest.
                    if (position != 0) staging_ = staging_t::pending_k;
                    return refused;
                }
        }
        staging_ = staging_t::pending_k;
        return success_k;
    }

    /**
     *  @brief Pulls every staged write back into its transaction, leaving the group retryable.
     *
     *  @return Success; the refusal it stopped at; or @c operation_not_permitted_k when the group
     *      is not staged.
     *
     *  Only a rollback that reached every participant clears the staged flag. Clearing it after a
     *  refusal would advertise a group whose later participants are still staged, and the next
     *  @c stage would stage a second time over the first.
     */
    status_t rollback() noexcept {
        if (staging_ != staging_t::staged_k) return operation_not_permitted_k;
        // Ascending store address, as every forward pass takes - only an unwind descends - so a
        // participant holding a lock across the phases cannot deadlock against another group.
        for (std::size_t position = 0; position != participants_k; ++position)
            if (status_t const refused =
                    visit_at_(order_[position], [](auto &transaction) noexcept { return transaction.rollback(); });
                failed(refused))
                return refused;
        staging_ = staging_t::pending_k;
        return success_k;
    }

    /** Discards every participant's staged and pending changes. */
    status_t reset() noexcept {
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

/** Deduces the store types, so a caller names the stores and not their spellings. */
template <optimistically_concurrent_store... store_types_>
expected<transaction_group<store_types_...>> make_transaction_group(store_types_ &...stores) noexcept {
    return transaction_group<store_types_...>::make(stores...);
}

#pragma region Shared Mutex

#pragma region Waiting Policies

/**
 *  @brief What a spin loop asks of the policy deciding what its core does between attempts.
 *
 *  @tparam policy_type_ The policy being checked.
 *  @tparam watched_type_ The word the loop retries over, as the policy sees it - an atomic object,
 *      or a pointer to a word owned through @c atomic_ref.
 *  @tparam value_type_ The bit pattern the loop last read out of that word.
 *
 *  One call takes the watched word, the value that was there, and how many attempts the loop has
 *  already spent, and it must come back on its own: a wait ending only on someone else's action
 *  deadlocks wherever the spinner is the thread that action waits for. Every architectural hint
 *  fits - a stall of a few tens of cycles, a hint handing the pipeline to a sibling thread, or a
 *  monitored wait ended by a store to the watched line or by a deadline. Ignoring all three
 *  arguments is legitimate, and is what a bare stall does.
 */
template <typename policy_type_, typename watched_type_, typename value_type_>
concept waiting_policy = requires(policy_type_ const &policy, watched_type_ &watched, value_type_ const observed,
                                  std::size_t const attempts_spent) {
    { policy(watched, observed, attempts_spent) } noexcept;
};

/**
 *  @brief A waiting policy that can also take a thread off its core until the watched word moves.
 *
 *  @tparam policy_type_ The policy being checked.
 *  @tparam watched_type_ The word a parked thread blocks on.
 *  @tparam value_type_ The bit pattern that has to change before a parked thread is owed a wake.
 *
 *  Asked only where a loop gives up on spinning, which is where a lock is held across user code and
 *  a burnt core costs more than a syscall. @c wait_until_changed may return spuriously, since every
 *  caller re-reads the word and decides again, but it has to return once @c notify_waiters lands,
 *  or a waiter sleeps through the handoff it parked for.
 */
template <typename policy_type_, typename watched_type_, typename value_type_>
concept parking_waiting_policy =
    waiting_policy<policy_type_, watched_type_, value_type_> &&
    requires(policy_type_ const &policy, watched_type_ &watched, value_type_ const observed) {
        { policy.wait_until_changed(watched, observed) } noexcept;
        { policy.notify_waiters(watched) } noexcept;
    };

/**
 *  @brief The policy a spin takes when nobody names one: no instruction at all between attempts.
 *
 *  Leaves a spin the bare retry loop it reads as, which is the one choice needing neither an
 *  architecture nor a runtime probe, and the only one that compiles for every target this header
 *  reaches, a device included. A deployment that knows what its cores are names a policy stalling
 *  them properly - @c PAUSE, @c YIELD, @c UMWAIT, @c WFET - and one that does not pays nothing for
 *  the question.
 */
struct bare_waiting_policy_t {

    /** Spends no instruction between attempts, whatever the loop watches. */
    template <typename watched_type_, typename value_type_, typename attempts_type_>
    constexpr void operator()(watched_type_ const &, value_type_, attempts_type_) const noexcept {}
};

/**
 *  @brief Parks a spent spin on @c std::atomic::wait, leaving the spin itself to
 *      @p pausing_policy_type_.
 *
 *  @tparam pausing_policy_type_ What a core does between attempts, defaulting to nothing at all.
 *
 *  This is how a pause hint reaches a path that also parks: the hint answers for the spin and the
 *  standard library answers for the sleep, so a policy that only knows how to stall a core needs to
 *  say nothing about parking to be usable on a lock that parks.
 */
template <typename pausing_policy_type_ = bare_waiting_policy_t>
struct standard_waiting_policy {

    /** What a core does between attempts, asked once per attempt the spin spends. */
    ST_NO_UNIQUE_ADDRESS_ pausing_policy_type_ pausing {};

    /** Hands one spent attempt to the pausing policy, which is the whole of the spin here. */
    template <typename watched_type_, typename value_type_, typename attempts_type_>
    constexpr void operator()(watched_type_ const &watched, value_type_ const observed,
                              attempts_type_ const attempts_spent) const noexcept {
        pausing(watched, observed, attempts_spent);
    }

    /** Parks until @p watched stops reading @p observed, or until the wait returns spuriously. */
    template <typename value_type_>
    void wait_until_changed(value_type_ &watched, value_type_ const observed) const noexcept {
        atomic_wait(watched, observed);
    }

    /** Wakes every thread parked on @p watched. */
    template <typename value_type_>
    void notify_waiters(value_type_ &watched) const noexcept {
        atomic_notify_all(watched);
    }
};

/** What every path that parks here takes unless a caller names another: a bare retry loop, parking
 *  on @c std::atomic::wait once the spin is spent. */
using standard_waiting_policy_t = standard_waiting_policy<>;

#pragma endregion Waiting Policies

/**
 *  @brief A reader-writer lock that spins briefly, then parks.
 *
 *  @tparam waiting_policy_type_ Decides what a core does between attempts, and how a spent spin
 *      parks. The default spends nothing and parks on @c std::atomic::wait.
 *  @tparam atomic_reference_ The reference the word is owned through, for one operation at a time.
 *
 *  Both acquires are bounded adds rather than compare-exchange loops, which is what the word's
 *  layout buys: every writer bit sits above the reader tally, so @em no @em writer @em here and
 *  @em room @em for @em one @em more @em reader is the single question does the sum stay inside
 *  the tally. A reference satisfying @c extended_atomic_ref answers it in one instruction and, when
 *  the answer is no, without taking the line at all; @c atomic_ref answers it with the loop that
 *  question replaces. Dropping the held bit reads nothing back, so it is posted rather than waited
 *  on. Releasing a reader is the one path that cannot be either, since the count it returns is what
 *  tells the last reader out to wake a writer.
 *
 *  Locks here are held across user code - callbacks, comparators, allocators, a whole commit loop -
 *  so a pure spinner would burn a core while a holder walks a range, and pure parking would pay a
 *  syscall for the uncontended case that dominates. Waiting writers are counted, and a non-zero
 *  count turns new readers away, so a steady read load cannot starve any of them indefinitely.
 *
 *  Offers exactly what @c std::shared_mutex is used for here, and nothing else: no recursion, no
 *  timed acquisition, no upgrading.
 */
template <parking_waiting_policy<std::uint32_t, std::uint32_t> waiting_policy_type_ = standard_waiting_policy_t,
          template <typename> class atomic_reference_ = atomic_ref>
class spin_shared_mutex {

    /** Set while one writer owns the lock; the reader count is zero for as long as it is. */
    static constexpr std::uint32_t writer_held_k = 1u << 31;

    /** One unit of the waiting-writer tally, which occupies the fifteen bits below the held bit. */
    static constexpr std::uint32_t one_writer_waiting_k = 1u << 16;

    /** Where the waiting-writer tally lives, capped at 32'767 writers parked at once. */
    static constexpr std::uint32_t writers_waiting_mask_k = 0x7FFF0000u;

    /** Where the reader tally lives, capped at 65'535 readers holding at once. */
    static constexpr std::uint32_t readers_mask_k = 0x0000FFFFu;

    /** What turns an arriving reader away: an owning writer, or any writer queued ahead of it. */
    static constexpr std::uint32_t writer_bits_k = writer_held_k | writers_waiting_mask_k;

    /** How long to spin before parking, which is about the cost of one uncontended handoff. */
    static constexpr std::size_t spins_before_parking_k = 64;

    /** The owning bit, the waiting-writer tally and the reader tally, in one word every path drives.
     *  Plain rather than an @c std::atomic, so the reference it is owned through is the caller's. */
    alignas(atomic_alignment<std::uint32_t>) std::uint32_t state_ {0};

    /** The reference every operation below wraps around @c state_, for one operation and no longer. */
    using word_ref_t = atomic_reference_<std::uint32_t>;

    /** What a waiting thread does with its core, and what it parks on once its spin is spent. */
    ST_NO_UNIQUE_ADDRESS_ waiting_policy_type_ waiting_ {};

  public:
    constexpr spin_shared_mutex() noexcept = default;
    spin_shared_mutex(spin_shared_mutex const &) = delete;
    spin_shared_mutex &operator=(spin_shared_mutex const &) = delete;

  private:
    /** Takes the word from idle to held, which is the only sum that stays within the held bit. */
    [[nodiscard]] std::uint32_t claim_as_writer_() noexcept {
        return atomic_fetch_add_if_at_most<atomic_reference_>(state_, writer_held_k, writer_held_k,
                                                              memory_order_acquire_k);
    }

    /** Adds one reader while the sum stays inside the reader tally, which every writer bit overflows. */
    [[nodiscard]] std::uint32_t claim_as_reader_() noexcept {
        return atomic_fetch_add_if_at_most<atomic_reference_>(state_, 1u, readers_mask_k, memory_order_acquire_k);
    }

  public:
    [[nodiscard]] bool try_lock() noexcept { return claim_as_writer_() == 0; }

    [[nodiscard]] bool try_lock_shared() noexcept { return claim_as_reader_() < readers_mask_k; }

    void lock() noexcept {
        for (std::size_t attempts_spent = 0; attempts_spent != spins_before_parking_k; ++attempts_spent) {
            std::uint32_t const observed = claim_as_writer_();
            if (observed == 0) return;
            waiting_(state_, observed, attempts_spent);
        }

        // Joining the tally is what stops a stream of readers from renewing the lock forever. A tally
        // rather than a flag, so one writer taking the lock cannot erase the intent of the others.
        word_ref_t state_ref(state_);
        std::uint32_t observed =
            state_ref.fetch_add(one_writer_waiting_k, memory_order_relaxed_k) + one_writer_waiting_k;
        while (true) {
            // Leaving the tally and taking the lock are one write, so no arrival slips between them,
            // and the pair is not a sum any bound can decide.
            if ((observed & (writer_held_k | readers_mask_k)) == 0) {
                std::uint32_t const taken = (observed - one_writer_waiting_k) | writer_held_k;
                if (state_ref.compare_exchange_weak(observed, taken, memory_order_acquire_k, memory_order_relaxed_k))
                    return;
                continue; // ? The exchange refreshed `observed`
            }
            waiting_.wait_until_changed(state_, observed);
            observed = state_ref.load(memory_order_relaxed_k);
        }
    }

    void unlock() noexcept {
        // Only the held bit is ours to drop: the writers queued behind us keep their places, and
        // keep new readers out while they wait. Nothing reads what the word held, so nothing waits.
        atomic_post_clear_bits<atomic_reference_>(state_, writer_held_k, memory_order_release_k);
        waiting_.notify_waiters(state_);
    }

    void lock_shared() noexcept {
        for (std::size_t attempts_spent = 0; attempts_spent != spins_before_parking_k; ++attempts_spent) {
            std::uint32_t const observed = claim_as_reader_();
            if (observed < readers_mask_k) return;
            waiting_(state_, observed, attempts_spent);
        }

        while (true) {
            std::uint32_t const observed = claim_as_reader_();
            if (observed < readers_mask_k) return;
            waiting_.wait_until_changed(state_, observed);
        }
    }

    void unlock_shared() noexcept {
        // The count is read back, so this one cannot be posted: the last reader out is the only one
        // a waiting writer is still blocked on.
        std::uint32_t const previous = word_ref_t(state_).fetch_sub(1, memory_order_release_k);
        if ((previous & readers_mask_k) == 1) waiting_.notify_waiters(state_);
    }
};

/** The reader-writer lock every collection wrapper here defaults to, spinning without a hint and
 *  parking on @c std::atomic::wait. */
using spin_shared_mutex_t = spin_shared_mutex<>;

/** Holds @p mutex_type_ exclusively for the enclosing scope, or for as long as whoever it is moved into keeps it; an
 *  empty one holds nothing, so a hold can outlive one call. */
template <typename mutex_type_>
class unique_lock {
    mutex_type_ *mutex_ = nullptr;

  public:
    unique_lock() noexcept = default;
    explicit unique_lock(mutex_type_ &mutex) noexcept : mutex_(&mutex) { mutex_->lock(); }
    unique_lock(unique_lock &&other) noexcept : mutex_(std::exchange(other.mutex_, nullptr)) {}
    unique_lock &operator=(unique_lock &&other) noexcept {
        release();
        mutex_ = std::exchange(other.mutex_, nullptr);
        return *this;
    }
    ~unique_lock() noexcept { release(); }
    unique_lock(unique_lock const &) = delete;
    unique_lock &operator=(unique_lock const &) = delete;

    /** Whether this holds a mutex. */
    explicit operator bool() const noexcept { return mutex_ != nullptr; }

    /** Gives the mutex back early; harmless on an empty one. */
    void release() noexcept {
        if (mutex_) std::exchange(mutex_, nullptr)->unlock();
    }
};

/** Holds @p mutex_type_ for shared reading over the enclosing scope. */
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

#pragma region Set Algebra

/** Which members of the two sides a walk keeps. */
enum class algebra_t : std::uint8_t {
    union_k,
    intersection_k,
    difference_k,
    symmetric_difference_k,
};

/** Whether the side being walked is the only one holding a member, or both are. */
enum class membership_t : bool { on_one_side_k, on_both_sides_k };

/** Whether a member of the first side, at @p membership, is kept by @c algebra_. */
template <algebra_t algebra_>
[[nodiscard]] constexpr bool algebra_keeps_first(membership_t membership) noexcept {
    if constexpr (algebra_ == algebra_t::union_k) return true;
    else if constexpr (algebra_ == algebra_t::intersection_k) return membership == membership_t::on_both_sides_k;
    else return membership == membership_t::on_one_side_k;
}

/** Whether @c algebra_ needs the second side walked as well as the first. Only the two that can keep a member the
 *  first side never saw. */
template <algebra_t algebra_>
[[nodiscard]] constexpr bool algebra_walks_both() noexcept {
    return algebra_ == algebra_t::union_k || algebra_ == algebra_t::symmetric_difference_k;
}

/**
 *  @brief Whether a member of the second side is kept, which needs no algebra to answer.
 *
 *  The second side is walked only to pick up what the first never had, so both algebras that reach
 *  it keep the same members - anything already shared was handed over by the first sweep.
 */
[[nodiscard]] constexpr bool algebra_keeps_second(membership_t membership) noexcept {
    return membership == membership_t::on_one_side_k;
}

/** Whether a crossed walk carries on past a member, or the answer no longer needs it. */
enum class comparison_t : bool { carry_on_k, settled_k };

/** How many members a crossed walk carries out of one side before it puts that side down. The only memory such a
 *  walk holds, on its own stack; a larger chunk re-seeds the walk fewer times and costs that much more of it. */
inline constexpr std::size_t algebra_chunk_k = 512;

/** Whether a side hands out a read transaction whose ordered walk resumes from a key it saw. A side that does need
 *  not be held while the other is taken, which is what two crossing comparisons need in order not to wedge on a
 *  writer queued between them. */
template <typename side_type_>
concept resumes_from_a_key = requires(side_type_ &side, typename side_type_::transaction_t const &reading,
                                      typename side_type_::identifier_t const &key, no_op_t callback) {
    typename side_type_::value_t;
    side.transaction();
    reading.smallest(callback, callback);
    reading.range_from(key, callback);
    reading.contains(key);
    requires std::is_nothrow_default_constructible_v<typename side_type_::value_t>;
};

/**
 *  @brief Hands @p visitor every member of @p walked with whether @p probing holds it too, stopping
 *      where the visitor calls the answer settled.
 *
 *  A pair that resumes from a key is read through one read transaction each, @p walked a chunk at a
 *  time, and neither side is held while the other is taken. A pair that does not - an unordered
 *  store, or a transaction a caller opened - probes @p probing from inside the walk of @p walked,
 *  so two crossing comparisons can wedge there.
 *
 *  @note A chunked side shows one instant for the whole comparison from @c snapshot_k up, and a
 *      chunk boundary is a point it may move at below that. From @c serializable_k up every probe
 *      is recorded, so the probing side's read set grows with the walked side.
 */
template <typename walked_type_, typename probing_type_, typename visitor_type_>
status_t compare_crossed_(walked_type_ &walked, probing_type_ &probing, visitor_type_ &&visitor) noexcept {

    if constexpr (resumes_from_a_key<walked_type_> && resumes_from_a_key<probing_type_>) {
        using member_t = typename walked_type_::value_t;
        auto reading_walked = walked.transaction();
        if (!reading_walked) return reading_walked.status();
        auto reading_probing = probing.transaction();
        if (!reading_probing) return reading_probing.status();

        member_t chunk[algebra_chunk_k] {};
        member_t resume {};
        member_t next_resume {};
        status_t copied = success_k;
        auto take = [&](member_t &target, member_t const &member) noexcept {
            auto copy = copy_safely(member);
            if (!copy) return void(copied = copy.status());
            target = std::move(*copy);
        };

        // The walk needs a key to start from and no level names a floor, so the smallest member is one.
        status_t const seeded =
            reading_walked->smallest([&](member_t const &member) noexcept { take(resume, member); }, no_op_t {});
        if (failed(seeded)) return seeded;
        if (failed(copied)) return copied;

        for (;;) {
            std::size_t seen = 0;
            auto collect = [&](member_t const &member) noexcept -> walk_control_t {
                if (seen > algebra_chunk_k || failed(copied)) return walk_control_t::halt_k;
                take(seen == algebra_chunk_k ? next_resume : chunk[seen], member);
                ++seen;
                return seen > algebra_chunk_k ? walk_control_t::halt_k : walk_control_t::resume_k;
            };
            status_t const swept = reading_walked->range_from(mapping_key_or_itself<member_t>(resume), collect);
            if (failed(swept)) return swept;
            if (failed(copied)) return copied;

            std::size_t const filled = seen < algebra_chunk_k ? seen : algebra_chunk_k;
            for (std::size_t index = 0; index != filled; ++index) {
                expected<bool> const shared = reading_probing->contains(mapping_key_or_itself<member_t>(chunk[index]));
                if (!shared) return shared.status();
                membership_t const membership = *shared ? membership_t::on_both_sides_k : membership_t::on_one_side_k;
                if (visitor(chunk[index], membership) == comparison_t::settled_k) return success_k;
            }
            if (seen <= algebra_chunk_k) return success_k;
            resume = std::move(next_resume);
        }
    }
    else {
        status_t probed = success_k;
        status_t const swept = walked.for_each([&](auto const &member) noexcept -> walk_control_t {
            expected<bool> const shared = probing.contains(member);
            if (!shared) {
                probed = shared.status();
                return walk_control_t::halt_k;
            }
            membership_t const membership = *shared ? membership_t::on_both_sides_k : membership_t::on_one_side_k;
            return visitor(member, membership) == comparison_t::settled_k ? walk_control_t::halt_k
                                                                          : walk_control_t::resume_k;
        });
        if (failed(swept)) return swept;
        return probed;
    }
}

/**
 *  @brief Hands @p callback every member of @c algebra_ over @p first and @p second.
 *
 *  @warning The two sides keep separate clocks, so the pair is never one instant however it is
 *      read. Each side is read through a read transaction opened here, which fixes what that side
 *      shows from @c snapshot_k up; pass transactions to choose the instants yourself, at the cost
 *      of the nested walk @c compare_crossed_ falls back to.
 */
template <algebra_t algebra_, typename first_type_, typename second_type_, typename callback_type_ = no_op_t>
status_t walk_algebra(first_type_ &first, second_type_ &second, callback_type_ &&callback) noexcept {

    auto sweep = [&](auto &side, auto &other, auto keeps) noexcept {
        return compare_crossed_(side, other, [&](auto const &member, membership_t membership) noexcept {
            if (keeps(membership)) callback(member);
            return comparison_t::carry_on_k;
        });
    };

    status_t swept = sweep(first, second,
                           [](membership_t membership) noexcept { return algebra_keeps_first<algebra_>(membership); });
    if (failed(swept)) return swept;

    if constexpr (algebra_walks_both<algebra_>()) {
        swept = sweep(second, first, [](membership_t membership) noexcept { return algebra_keeps_second(membership); });
        if (failed(swept)) return swept;
    }
    return success_k;
}

/**
 *  @brief Whether every member of @p first is also in @p second. Settled at the first member
 *      @p second lacks, and the walk of @p first stops there.
 *
 *  A side with more members than the other cannot be contained in it, which is the whole answer
 *  without a walk. Both counts are read before either side is touched, so the pair is a decision
 *  about the sizes at one moment rather than a size from one moment weighed against a walk
 *  from another.
 */
template <typename first_type_, typename second_type_>
expected<bool> is_subset(first_type_ &first, second_type_ &second) noexcept {
    // One side contains itself, and the unordered walk below would probe the lock it already holds.
    if constexpr (std::is_same_v<first_type_, second_type_>)
        if (&first == &second) return true;
    if (first.size() > second.size()) return false;
    bool subset = true;
    status_t const compared = compare_crossed_(first, second, [&](auto const &, membership_t membership) noexcept {
        subset = membership == membership_t::on_both_sides_k;
        return subset ? comparison_t::carry_on_k : comparison_t::settled_k;
    });
    if (failed(compared)) return compared;
    return subset;
}

/** Walks @p first probing @p second, settling at the first member they share. */
template <typename first_type_, typename second_type_>
expected<bool> is_disjoint_walking_(first_type_ &first, second_type_ &second) noexcept {
    bool disjoint = true;
    status_t const compared = compare_crossed_(first, second, [&](auto const &, membership_t membership) noexcept {
        disjoint = membership == membership_t::on_one_side_k;
        return disjoint ? comparison_t::carry_on_k : comparison_t::settled_k;
    });
    if (failed(compared)) return compared;
    return disjoint;
}

/**
 *  @brief Whether the two sides share no member, on the same terms as @c is_subset.
 *
 *  Sharing is symmetric, so the walk takes the smaller side and probes the larger.
 */
template <typename first_type_, typename second_type_>
expected<bool> is_disjoint(first_type_ &first, second_type_ &second) noexcept {
    if constexpr (std::is_same_v<first_type_, second_type_>) {
        // Self against self shares every member it has, and the unordered walk below would probe the
        // lock it already holds.
        if (&first == &second) return first.size() == 0;
        // Decided once rather than by calling back the other way round, which two racing sizes could
        // keep bouncing, and which would demand the reverse instantiation of an asymmetric pair.
        if (first.size() > second.size()) return is_disjoint_walking_(second, first);
    }
    return is_disjoint_walking_(first, second);
}

#pragma endregion Set Algebra

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
