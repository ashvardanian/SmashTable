/**
 *  @brief Template test functions for basic container operations. Includes insert/erase/find/range operations and
 *      heterogeneous lookup tests. Templates can be instantiated for any container supporting the common interface.
 *  @author Ash Vardanian
 *  @file scripts/test_basic.hpp
 *  @date January 12, 2023
 */
#pragma once
#include <cstdint> // `std::uintptr_t`
#include <cstring> // `std::memcmp`

#include <algorithm>   // `std::min`
#include <atomic>      // `std::atomic`
#include <compare>     // `std::strong_ordering`
#include <exception>   // `std::exception`
#include <functional>  // `std::less`, `std::equal_to`
#include <limits>      // `std::numeric_limits`
#include <new>         // `std::nothrow`
#include <string_view> // `std::string_view`
#include <type_traits> // `std::remove_cvref_t`
#include <vector>      // `std::vector`

#include <smashtable/basic_vector.hpp>

#include "test.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Keys and Associations

/**
 *  @brief Even when testing some non-trivial keys, to make tests more uniform, they
 *    should be constructible from a single unsigned integer identifier and comparable to it,
 *    unless we are explicitly testing a non-heterogeneous comparator.
 */
using trivial_id_t = std::size_t;

/**
 *  @brief Strongly-typed trivial key @b without any payload or heterogenous comparisons.
 */
struct trivial_key_t {
    trivial_id_t unique_id = 0;

    explicit trivial_key_t(trivial_id_t i = 0) noexcept : unique_id(i) {}

    trivial_key_t(trivial_key_t &&) noexcept = default;
    trivial_key_t(trivial_key_t const &) noexcept = default;
    trivial_key_t &operator=(trivial_key_t &&) noexcept = default;
    trivial_key_t &operator=(trivial_key_t const &) noexcept = default;

    bool operator<(trivial_key_t const &other) const noexcept { return unique_id < other.unique_id; }
    bool operator==(trivial_key_t const &other) const noexcept { return unique_id == other.unique_id; }
    bool operator!=(trivial_key_t const &other) const noexcept { return unique_id != other.unique_id; }
    bool operator>(trivial_key_t const &other) const noexcept { return unique_id > other.unique_id; }
    bool operator<=(trivial_key_t const &other) const noexcept { return unique_id <= other.unique_id; }
    bool operator>=(trivial_key_t const &other) const noexcept { return unique_id >= other.unique_id; }
};

/**
 *  @brief Strongly-typed lightweight key that explicitly converts to the inner identifier.
 *    The identifier is inferred from comparator - @c composite_key_compare_t::value_type.
 *    Both the key and the underlying identifier are @c noexcept copyable.
 */
struct composite_key_t {
    std::uint64_t some_metadata = 0;
    trivial_id_t unique_id = 0;
    double some_float = 0.0;

    explicit composite_key_t(trivial_id_t i = 0) noexcept : unique_id(i) {}

    composite_key_t(composite_key_t &&) noexcept = default;
    composite_key_t(composite_key_t const &) noexcept = default;
    composite_key_t &operator=(composite_key_t &&) noexcept = default;
    composite_key_t &operator=(composite_key_t const &) noexcept = default;

    explicit operator trivial_id_t() const noexcept { return unique_id; }
    bool operator<(composite_key_t const &other) const noexcept { return unique_id < other.unique_id; }
    bool operator==(composite_key_t const &other) const noexcept { return unique_id == other.unique_id; }
    bool operator!=(composite_key_t const &other) const noexcept { return unique_id != other.unique_id; }
    bool operator>(composite_key_t const &other) const noexcept { return unique_id > other.unique_id; }
    bool operator<=(composite_key_t const &other) const noexcept { return unique_id <= other.unique_id; }
    bool operator>=(composite_key_t const &other) const noexcept { return unique_id >= other.unique_id; }
};

struct composite_key_compare_t {
    using is_transparent = void;     // ? Enable heterogeneous lookup by prefix
    using value_type = trivial_id_t; // ? The only part we need to resolve transactions without carrying full object

    inline bool operator()(composite_key_t const &a, composite_key_t const &b) const noexcept {
        return a.unique_id < b.unique_id;
    }
    inline bool operator()(value_type a, composite_key_t const &b) const noexcept { return a < b.unique_id; }
    inline bool operator()(composite_key_t const &a, value_type b) const noexcept { return a.unique_id < b; }
};

/**
 *  @brief Strongly-typed key with heavy payload, that doesn't have a @c noexcept constructors,
 *    but provides @c ::make(...) and @c .copy() interfaces for safe construction and copying.
 *
 *  For compatibility with integer-based tests, provides @c ::make(trivial_id_t) that encodes
 *  the integer as a hexadecimal string and @c <=> and @c == operators for comparisons.
 */
struct heavy_key_t {

    basic_vector<char> text;

    heavy_key_t() noexcept = default;
    heavy_key_t(heavy_key_t &&) noexcept = default;
    heavy_key_t &operator=(heavy_key_t &&) noexcept = default;

    /**
     *  @brief Creates a @c heavy_key_t from an integer by converting to hex string.
     *  @param[in] integer_to_encode Integer to encode as hexadecimal string.
     *  @return expected<heavy_key_t> containing the key or error on OOM.
     */
    static expected<heavy_key_t> make(trivial_id_t integer_to_encode) noexcept {
        heavy_key_t result;
        char buffer[32];
        int length =
            std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(integer_to_encode));
        if (length <= 0 || length >= static_cast<int>(sizeof(buffer)))
            return expected<heavy_key_t>(heavy_key_t {}, status_t::unknown_k);

        for (int i = 0; i < length; ++i) {
            auto status = result.text.push_back(char(buffer[i]));
            if (failed(status)) return expected<heavy_key_t>(heavy_key_t {}, status);
        }
        return result;
    }

    /**
     *  @brief Creates a @c heavy_key_t from a string by copying into internal buffer.
     *  @param[in] std_string String to copy.
     *  @return expected<heavy_key_t> containing the key or error on OOM.
     */
    static expected<heavy_key_t> make(std::string_view std_string) noexcept {
        heavy_key_t result;
        for (std::size_t i = 0; i < std_string.size(); ++i) {
            auto status = result.text.push_back(char(std_string[i]));
            if (failed(status)) return expected<heavy_key_t>(heavy_key_t {}, status);
        }
        return result;
    }

    /**
     *  @brief Deep copies this @c heavy_key_t.
     *  @return expected<heavy_key_t> containing the copy or error on OOM.
     */
    expected<heavy_key_t> copy() const noexcept {
        heavy_key_t result;
        for (std::size_t i = 0; i < text.size(); ++i) {
            auto status = result.text.push_back(char(text.data()[i]));
            if (failed(status)) return expected<heavy_key_t>(heavy_key_t {}, status);
        }
        return result;
    }

    std::string_view view() const noexcept { return {text.data(), text.size()}; }

    /**
     *  @brief Orders two byte ranges, tolerating an empty one.
     *
     *  A zero-length @c std::memcmp still requires both pointers to be valid, and an empty
     *  container's @c data() is null - so the shared length is checked before the compare, never
     *  folded into it.
     */
    static std::strong_ordering compare_bytes(std::string_view first, std::string_view second) noexcept {
        std::size_t const shared = std::min(first.size(), second.size());
        int const ordered = shared == 0 ? 0 : std::memcmp(first.data(), second.data(), shared);
        if (ordered != 0) return ordered <=> 0;
        return first.size() <=> second.size();
    }

    std::strong_ordering operator<=>(heavy_key_t const &other) const noexcept {
        return compare_bytes(view(), other.view());
    }
    bool operator==(heavy_key_t const &other) const noexcept { return compare_bytes(view(), other.view()) == 0; }

    // The rewritten candidates C++20 synthesizes from these cover `string_view` on the left too,
    // so a transparent comparator probing both orders needs no free-standing reversal.
    std::strong_ordering operator<=>(std::string_view other) const noexcept { return compare_bytes(view(), other); }
    bool operator==(std::string_view other) const noexcept { return compare_bytes(view(), other) == 0; }
};

static_assert(std::is_same_v<versioning_for<heavy_key_t, std::less<void>>::value_type, heavy_key_t>);
static_assert(std::is_same_v<versioning_for<heavy_key_t, std::less<void>>::identifier_type, heavy_key_t>);

/**
 *  @brief Lifecycle-guarded payload that detects common anti-patterns.
 *    All state is self-contained - no globals, thread-safe.
 *
 *  Detects:
 *  - Double construction (constructor called twice)
 *  - Double destruction (destructor called twice)
 *  - Use-after-free (operations after destructor)
 *  - Use-after-move (reading moved-from object)
 *  - Operations on uninitialized memory
 *  - Memory corruption (via canaries)
 */
class guarded_payload_t {
    enum class state_t : std::uint8_t {
        uninitialized_k = 0x00, // Fresh memory, never constructed
        constructed_k = 0xC0,   // Valid, living object
        moved_from_k = 0x3F,    // Source of move (still destructible)
        destroyed_k = 0xDE      // Destructor was called
    };

    // Canaries to detect buffer overflows and memcpy misuse
    static constexpr std::uint32_t canary_front_k = 0xCAFEBABE;
    static constexpr std::uint32_t canary_back_k = 0xDEADC0DE;

    std::uint32_t canary_front_ = 0;
    state_t state_ = state_t::uninitialized_k;
    std::uint8_t construction_count_ = 0; // Should be 0 or 1
    std::uint8_t destruction_count_ = 0;  // Should be 0 or 1
    std::uint32_t generation_ = 0;        // Mutation counter
    trivial_id_t value_ = 0;
    std::uint32_t canary_back_ = 0;

  public:
    guarded_payload_t() noexcept {
        verify_not_double_constructed();
        canary_front_ = canary_front_k;
        canary_back_ = canary_back_k;
        state_ = state_t::constructed_k;
        construction_count_ = 1;
    }

    explicit guarded_payload_t(trivial_id_t v) noexcept {
        verify_not_double_constructed();
        canary_front_ = canary_front_k;
        canary_back_ = canary_back_k;
        state_ = state_t::constructed_k;
        construction_count_ = 1;
        value_ = v;
        generation_ = 1;
    }

    ~guarded_payload_t() noexcept {
        verify_canaries();
        assert(destruction_count_ == 0 && "Double destruction detected!");
        assert(state_ != state_t::destroyed_k && "Destructor called twice!");
        assert((state_ == state_t::constructed_k || state_ == state_t::moved_from_k) &&
               "Destructing unconstructed object!");

        destruction_count_ = 1;
        state_ = state_t::destroyed_k;
        value_ = static_cast<trivial_id_t>(0xDEADBEEFDEADBEEFULL);
    }

    // Copy constructor - creates new independent object
    guarded_payload_t(guarded_payload_t const &other) noexcept {
        other.verify_readable();
        verify_not_double_constructed();

        canary_front_ = canary_front_k;
        canary_back_ = canary_back_k;
        value_ = other.value_;
        generation_ = other.generation_;
        state_ = state_t::constructed_k;
        construction_count_ = 1;
        destruction_count_ = 0;
    }

    // Move constructor - transfers value, marks source
    guarded_payload_t(guarded_payload_t &&other) noexcept {
        other.verify_readable();
        verify_not_double_constructed();

        canary_front_ = canary_front_k;
        canary_back_ = canary_back_k;
        value_ = other.value_;
        generation_ = other.generation_;
        state_ = state_t::constructed_k;
        construction_count_ = 1;
        destruction_count_ = 0;

        // Mark source as moved-from (still destructible, but not readable)
        other.state_ = state_t::moved_from_k;
    }

    guarded_payload_t &operator=(guarded_payload_t const &other) noexcept {
        other.verify_readable();
        verify_writable();
        value_ = other.value_;
        generation_++;
        return *this;
    }

    guarded_payload_t &operator=(guarded_payload_t &&other) noexcept {
        other.verify_readable();
        verify_writable();
        value_ = other.value_;
        generation_++;
        other.state_ = state_t::moved_from_k;
        return *this;
    }

    guarded_payload_t &operator=(trivial_id_t v) noexcept {
        verify_writable();
        value_ = v;
        generation_++;
        return *this;
    }

    trivial_id_t get() const noexcept {
        verify_readable();
        return value_;
    }

    std::uint32_t generation() const noexcept {
        verify_readable();
        return generation_;
    }

    bool operator==(trivial_id_t other) const noexcept {
        verify_readable();
        return value_ == other;
    }

    bool operator!=(trivial_id_t other) const noexcept { return !(*this == other); }

    bool operator==(guarded_payload_t const &other) const noexcept {
        verify_readable();
        other.verify_readable();
        return value_ == other.value_;
    }

  private:
    void verify_canaries() const noexcept {
        assert(canary_front_ == canary_front_k && "Front canary corrupted - buffer overflow or memcpy?");
        assert(canary_back_ == canary_back_k && "Back canary corrupted - buffer overflow or memcpy?");
    }

    void verify_not_double_constructed() const noexcept {
        assert(construction_count_ == 0 && "Double construction detected!");
        assert(state_ == state_t::uninitialized_k && "Constructing over existing object!");
    }

    void verify_readable() const noexcept {
        verify_canaries();
        assert(state_ == state_t::constructed_k && "Reading invalid object (use-after-free or use-after-move)!");
        assert(construction_count_ == 1 && "Reading unconstructed object!");
        assert(destruction_count_ == 0 && "Use after free!");
    }

    void verify_writable() noexcept {
        verify_canaries();
        assert(state_ == state_t::constructed_k && "Writing to invalid object!");
        assert(construction_count_ == 1 && "Writing to unconstructed object!");
        assert(destruction_count_ == 0 && "Writing to destroyed object!");
    }
};

#pragma endregion Keys and Associations

#pragma region Budgets

/** @brief How a budgeted resource answered one request, named so a log entry never reads as a bare flag. */
enum class budget_outcome_t : bool { granted_k, refused_k };

/** @brief The countdown that never runs out, as opposed to @c 0, which refuses the very next request. */
inline constexpr std::size_t unlimited_budget_k = std::numeric_limits<std::size_t>::max();

/**
 *  @brief Countdown arming @c budgeted_key_t::copy, so a rollback or a watch copy can be failed on demand.
 *
 *  The budget is process-wide because the key it governs is copied deep inside a container, where a test
 *  has no reference to hand it.
 */
struct copy_budget_t {
    static inline std::size_t copies_till_fail {unlimited_budget_k};
    static inline std::size_t refusals_count {0};

    static void reset() noexcept {
        copies_till_fail = unlimited_budget_k;
        refusals_count = 0;
    }

    /** @brief Permits @p count more copies, refusing every one after them. */
    static void allow(std::size_t count) noexcept { copies_till_fail = count; }

    /** @brief Spends one copy, reporting whether it was granted. */
    static budget_outcome_t spend() noexcept {
        if (copies_till_fail == unlimited_budget_k) return budget_outcome_t::granted_k;
        if (copies_till_fail == 0) {
            ++refusals_count;
            return budget_outcome_t::refused_k;
        }
        --copies_till_fail;
        return budget_outcome_t::granted_k;
    }
};

#pragma endregion Budgets

#pragma region Counting Element

/**
 *  @brief Key and value tallying its own lifetime, so a leak or a double destruction is arithmetic.
 *
 *  A sanitizer sees a freed block, not the element left unconstructed-over or never destroyed inside one,
 *  which is exactly the defect a node allocator's own bookkeeping hides. The tallies are process-wide,
 *  because the objects live inside containers a test cannot reach into.
 */
struct counted_key_t {
    static constexpr std::uint32_t alive_magic_k = 0xA11FE000;
    static constexpr std::uint32_t dead_magic_k = 0xDEAD0000;

    static inline std::atomic<std::size_t> constructions {0};
    static inline std::atomic<std::size_t> destructions {0};
    static inline std::atomic<std::size_t> copies {0};
    static inline std::atomic<std::size_t> moves {0};
    static inline std::atomic<std::size_t> defects {0};

    trivial_id_t unique_id {0};
    std::uint32_t magic {alive_magic_k};

    counted_key_t() noexcept { note_construction(); }
    explicit counted_key_t(trivial_id_t identifier) noexcept : unique_id(identifier) { note_construction(); }

    counted_key_t(counted_key_t const &other) noexcept : unique_id(other.unique_id) {
        other.verify_alive();
        copies.fetch_add(1, std::memory_order_relaxed);
        note_construction();
    }

    /** @brief Leaves the source alive, so its own destructor still balances the pair. */
    counted_key_t(counted_key_t &&other) noexcept : unique_id(other.unique_id) {
        other.verify_alive();
        moves.fetch_add(1, std::memory_order_relaxed);
        note_construction();
    }

    counted_key_t &operator=(counted_key_t const &other) noexcept {
        other.verify_alive();
        verify_alive();
        unique_id = other.unique_id;
        copies.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    counted_key_t &operator=(counted_key_t &&other) noexcept {
        other.verify_alive();
        verify_alive();
        unique_id = other.unique_id;
        moves.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }

    ~counted_key_t() noexcept {
        if (magic != alive_magic_k) {
            defects.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        magic = dead_magic_k;
        destructions.fetch_add(1, std::memory_order_relaxed);
    }

    std::strong_ordering operator<=>(counted_key_t const &other) const noexcept {
        verify_alive();
        other.verify_alive();
        return unique_id <=> other.unique_id;
    }
    bool operator==(counted_key_t const &other) const noexcept {
        verify_alive();
        other.verify_alive();
        return unique_id == other.unique_id;
    }
    std::strong_ordering operator<=>(trivial_id_t other) const noexcept {
        verify_alive();
        return unique_id <=> other;
    }
    bool operator==(trivial_id_t other) const noexcept {
        verify_alive();
        return unique_id == other;
    }

    static void reset() noexcept {
        constructions.store(0, std::memory_order_relaxed);
        destructions.store(0, std::memory_order_relaxed);
        copies.store(0, std::memory_order_relaxed);
        moves.store(0, std::memory_order_relaxed);
        defects.store(0, std::memory_order_relaxed);
    }

    static std::size_t constructions_count() noexcept { return constructions.load(std::memory_order_relaxed); }
    static std::size_t destructions_count() noexcept { return destructions.load(std::memory_order_relaxed); }
    static std::size_t copies_count() noexcept { return copies.load(std::memory_order_relaxed); }
    static std::size_t moves_count() noexcept { return moves.load(std::memory_order_relaxed); }
    static std::size_t defects_count() noexcept { return defects.load(std::memory_order_relaxed); }

    /** @brief Objects built and not yet destroyed - negative when something was destroyed twice. */
    static std::ptrdiff_t alive() noexcept {
        return static_cast<std::ptrdiff_t>(constructions_count()) - static_cast<std::ptrdiff_t>(destructions_count());
    }

    /** @brief Aborts unless every object built has been destroyed exactly once, and none used after that. */
    static void verify_balanced() noexcept {
        st_verify_eq_(alive(), std::ptrdiff_t {0});
        st_verify_eq_(defects_count(), std::size_t {0});
    }

  private:
    void note_construction() noexcept {
        magic = alive_magic_k;
        constructions.fetch_add(1, std::memory_order_relaxed);
    }

    /** @brief Records, rather than asserts, a read of an object that was already destroyed. */
    void verify_alive() const noexcept {
        if (magic != alive_magic_k) defects.fetch_add(1, std::memory_order_relaxed);
    }
};

#pragma endregion Counting Element

#pragma region Collisions and Alignment

/**
 *  @brief Key hashing to its identifier's group index, so the identifiers sharing a group share one home
 *    slot and a probe run walks all of them.
 *
 *  Only the group survives hashing, so a table of at least @c collision_run_length_k slots is what makes
 *  the run long rather than merely crowded.
 */
struct colliding_key_t {
    static constexpr trivial_id_t collision_run_length_k = 8;

    trivial_id_t unique_id {0};

    explicit colliding_key_t(trivial_id_t identifier = 0) noexcept : unique_id(identifier) {}

    /** @brief The home slot's identity - two keys collide exactly when this matches. */
    static constexpr trivial_id_t group_of(trivial_id_t identifier) noexcept {
        return identifier / collision_run_length_k;
    }

    std::strong_ordering operator<=>(colliding_key_t const &other) const noexcept {
        return unique_id <=> other.unique_id;
    }
    bool operator==(colliding_key_t const &other) const noexcept { return unique_id == other.unique_id; }
};

/** @brief The alignment @c overaligned_key_t demands, wider than @c ::operator @c new promises. */
inline constexpr std::size_t overaligned_alignment_k = 64;

/**
 *  @brief Element demanding cache-line alignment, which a node allocator over plain @c ::operator @c new
 *    cannot meet, so the misalignment shows up where the element sits rather than where it was asked for.
 */
struct alignas(overaligned_alignment_k) overaligned_key_t {
    trivial_id_t unique_id {0};

    explicit overaligned_key_t(trivial_id_t identifier = 0) noexcept : unique_id(identifier) {}

    /** @brief Whether this object actually sits where its alignment demands. */
    bool is_correctly_aligned() const noexcept {
        return reinterpret_cast<std::uintptr_t>(this) % overaligned_alignment_k == 0;
    }

    std::strong_ordering operator<=>(overaligned_key_t const &other) const noexcept {
        return unique_id <=> other.unique_id;
    }
    bool operator==(overaligned_key_t const &other) const noexcept { return unique_id == other.unique_id; }
};

#pragma endregion Collisions and Alignment

#pragma region Fallible Copy Key

/**
 *  @brief Heap-allocating key whose @c copy refuses on @c copy_budget_t, driving the rollback and
 *    watch-copy failure paths a key that always copies leaves untested.
 *
 *  Move-only, like the heavy key it wraps, so every duplication in a container goes through @c copy and
 *  is therefore schedulable.
 */
struct budgeted_key_t {
    heavy_key_t text;

    budgeted_key_t() noexcept = default;
    budgeted_key_t(budgeted_key_t &&) noexcept = default;
    budgeted_key_t &operator=(budgeted_key_t &&) noexcept = default;

    static expected<budgeted_key_t> make(trivial_id_t identifier) noexcept {
        auto made = heavy_key_t::make(identifier);
        if (!made) return made.status();
        budgeted_key_t result;
        result.text = *std::move(made);
        return result;
    }

    static expected<budgeted_key_t> make(std::string_view std_string) noexcept {
        auto made = heavy_key_t::make(std_string);
        if (!made) return made.status();
        budgeted_key_t result;
        result.text = *std::move(made);
        return result;
    }

    expected<budgeted_key_t> copy() const noexcept {
        if (copy_budget_t::spend() == budget_outcome_t::refused_k)
            return expected<budgeted_key_t>(status_t::out_of_memory_heap_k);
        auto copied = text.copy();
        if (!copied) return copied.status();
        budgeted_key_t result;
        result.text = *std::move(copied);
        return result;
    }

    std::string_view view() const noexcept { return text.view(); }

    std::strong_ordering operator<=>(budgeted_key_t const &other) const noexcept { return text <=> other.text; }
    bool operator==(budgeted_key_t const &other) const noexcept { return text == other.text; }
    std::strong_ordering operator<=>(std::string_view other) const noexcept { return text <=> other; }
    bool operator==(std::string_view other) const noexcept { return text == other; }
};

#pragma endregion Fallible Copy Key

#pragma region Throwing Element

#pragma endregion Throwing Element

#pragma region Stateful Comparator

/** @brief Which way a @c stateful_comparator orders, named so a call site never reads as a bare flag. */
enum class ordering_t : bool { ascending_k, descending_k };

/**
 *  @brief Stateful comparator with runtime configuration.
 *    Tests that comparators with member state work correctly.
 */
template <typename baseline_comparator_ = std::less<void>>
struct stateful_comparator {
    using baseline_comparator_t = baseline_comparator_;
    using is_transparent = void;

    baseline_comparator_t baseline_comparator {};
    ordering_t ordering {ordering_t::ascending_k};

    stateful_comparator() noexcept = default;
    explicit stateful_comparator(ordering_t requested) noexcept : ordering(requested) {}

    template <typename lhs_type_, typename rhs_type_>
    bool operator()(lhs_type_ const &lhs, rhs_type_ const &rhs) const noexcept {
        return ordering == ordering_t::descending_k ? baseline_comparator(rhs, lhs) : baseline_comparator(lhs, rhs);
    }
};

using stateful_comparator_t = stateful_comparator<>;

#pragma endregion Stateful Comparator

#pragma region Counting Comparator and Hasher

/**
 *  @brief Process-wide tallies of ordering, equality and hashing invocations.
 *
 *  Reset immediately before the operation under test and read immediately after: a comparator is copied
 *  by value into every container that holds one, so there is nowhere else the count could live.
 */
struct call_tally_t {
    static inline std::atomic<std::size_t> comparisons {0};
    static inline std::atomic<std::size_t> equalities {0};
    static inline std::atomic<std::size_t> hashes {0};

    static void reset() noexcept {
        comparisons.store(0, std::memory_order_relaxed);
        equalities.store(0, std::memory_order_relaxed);
        hashes.store(0, std::memory_order_relaxed);
    }

    static void note_comparison() noexcept { comparisons.fetch_add(1, std::memory_order_relaxed); }
    static void note_equality() noexcept { equalities.fetch_add(1, std::memory_order_relaxed); }
    static void note_hash() noexcept { hashes.fetch_add(1, std::memory_order_relaxed); }

    static std::size_t comparisons_count() noexcept { return comparisons.load(std::memory_order_relaxed); }
    static std::size_t equalities_count() noexcept { return equalities.load(std::memory_order_relaxed); }
    static std::size_t hashes_count() noexcept { return hashes.load(std::memory_order_relaxed); }

    /** @brief Ceiling of the base-two logarithm of @p size, which is the depth a balanced tree promises. */
    static std::size_t logarithm_of(std::size_t size) noexcept {
        std::size_t bits = 0;
        while ((std::size_t {1} << bits) < size) ++bits;
        return bits;
    }

    /**
     *  @brief Aborts unless the comparisons since the last reset stay within @p multiple logarithms of
     *    @p size, which is what separates an order statistic from a linear walk.
     *
     *  The bound is @p multiple × (log2(@p size) + 1), the trailing term covering the empty and
     *  single-element cases where the logarithm is zero but one comparison still happens.
     */
    static void verify_comparisons_logarithmic(std::size_t size, std::size_t multiple) noexcept {
        std::size_t const allowed = multiple * (logarithm_of(size) + 1);
        st_verify_((comparisons_count() <= allowed) && "the operation compared more than a logarithm of times");
    }
};

/**
 *  @brief Transparent comparator counting every invocation, so a walk's cost becomes an assertion.
 */
template <typename baseline_comparator_ = std::less<void>>
struct counting_comparator {
    using baseline_comparator_t = baseline_comparator_;
    using is_transparent = void;

    baseline_comparator_t baseline_comparator {};

    template <typename lhs_type_, typename rhs_type_>
    bool operator()(lhs_type_ const &lhs, rhs_type_ const &rhs) const noexcept {
        call_tally_t::note_comparison();
        return baseline_comparator(lhs, rhs);
    }
};

using counting_comparator_t = counting_comparator<>;

/**
 *  @brief Transparent equality counting every invocation, for the cores that probe rather than descend.
 */
template <typename baseline_equals_ = std::equal_to<void>>
struct counting_equals {
    using baseline_equals_t = baseline_equals_;
    using is_transparent = void;

    baseline_equals_t baseline_equals {};

    template <typename lhs_type_, typename rhs_type_>
    bool operator()(lhs_type_ const &lhs, rhs_type_ const &rhs) const noexcept {
        call_tally_t::note_equality();
        return baseline_equals(lhs, rhs);
    }
};

using counting_equals_t = counting_equals<>;

/**
 *  @brief Transparent hasher counting every invocation, forwarding to the library's @c hash.
 *    Heterogeneous by construction, so a probe by a bare identifier is counted the same way.
 */
struct counting_hash_t {
    using is_transparent = void;

    template <typename key_type_>
    std::size_t operator()(key_type_ const &key) const noexcept {
        call_tally_t::note_hash();
        return hash<std::remove_cvref_t<key_type_>> {}(key);
    }
};

#pragma endregion Counting Comparator and Hasher

#pragma region new_member Construction Helpers

/**
 * @brief Helper to construct a new key matching the @c member_type_ from an integer value
 */
template <typename member_type_>
auto trivial_id_to_key(trivial_id_t value) {
    if constexpr (has_make_method<member_type_, trivial_id_t>) {
        auto made = member_type_::make(value);
        st_verify_((made) && "the fixture key must be constructible from an identifier");
        return *std::move(made);
    }
    else if constexpr (is_mapping<member_type_>) return trivial_id_to_key<typename member_type_::key_type>(value);
    else return member_type_ {value};
}

/**
 * @brief Helper to construct a new @c member_type_ from an integer value
 */
template <typename member_type_>
auto trivial_id_to_member(trivial_id_t value) {
    if constexpr (has_make_method<member_type_, trivial_id_t>) {
        auto made = member_type_::make(value);
        st_verify_((made) && "the fixture member must be constructible from an identifier");
        return *std::move(made);
    }
    else if constexpr (is_mapping<member_type_>)
        return member_type_ {trivial_id_to_key<member_type_>(value), typename member_type_::mapped_type(value)};
    else return member_type_ {value};
}

/**
 * @brief Helper to construct a new @c member_type_ from an integer value
 */
template <typename member_type_, typename value_type_>
member_type_ trivial_id_to_member(trivial_id_t identifier, value_type_ value) {
    if constexpr (is_mapping<member_type_>)
        return {trivial_id_to_key<member_type_>(identifier), typename member_type_::mapped_type(value)};
    else return trivial_id_to_key<member_type_>(identifier);
}

#pragma endregion new_member Construction Helpers

#pragma region Stateful Allocator

/** @brief One request an allocator was handed, kept in the order it arrived. */
struct allocation_request_t {
    std::size_t elements_count {0};
    std::size_t element_size {0};
    budget_outcome_t outcome {budget_outcome_t::granted_k};
};

/**
 *  @brief Shared state behind @c stateful_allocator - a budget, the tallies, and the request log.
 *
 *  Lives outside the allocator because a container rebinds and copies its allocator freely, and counts
 *  kept in the allocator itself would be scattered across those copies instead of accumulating.
 */
struct allocation_ledger_t {
    std::size_t allocations_till_fail {unlimited_budget_k};
    std::size_t granted_count {0};
    std::size_t refused_count {0};
    std::size_t deallocation_count {0};
    std::size_t unlogged_count {0}; // ? Requests the log itself had no memory to record
    basic_vector<allocation_request_t> requests;

    void reset() noexcept {
        allocations_till_fail = unlimited_budget_k;
        granted_count = 0;
        refused_count = 0;
        deallocation_count = 0;
        unlogged_count = 0;
        requests.clear();
    }

    /** @brief Permits @p count more allocations, refusing every one after them. */
    void allow(std::size_t count) noexcept { allocations_till_fail = count; }

    /** @brief Answers @c nullptr from the very next request onwards. */
    void refuse_everything() noexcept { allocations_till_fail = 0; }

    /** @brief Blocks granted and not yet returned - negative when something was freed twice. */
    std::ptrdiff_t live_allocations() const noexcept {
        return static_cast<std::ptrdiff_t>(granted_count) - static_cast<std::ptrdiff_t>(deallocation_count);
    }

    /** @brief The element count of the largest request seen, granted or refused. */
    std::size_t largest_request_elements() const noexcept {
        std::size_t largest = 0;
        for (std::size_t index = 0; index != requests.size(); ++index)
            largest = larger_of(largest, requests.data()[index].elements_count);
        return largest;
    }

    /** @brief Aborts unless every granted block has been returned, and the log is complete. */
    void verify_balanced() const noexcept {
        st_verify_eq_(live_allocations(), std::ptrdiff_t {0});
        st_verify_eq_(unlogged_count, std::size_t {0});
    }

    /** @brief Answers one request, spending the budget and appending to the log. */
    budget_outcome_t note_request(std::size_t elements_count, std::size_t element_size) noexcept {
        budget_outcome_t const outcome =
            allocations_till_fail == 0 ? budget_outcome_t::refused_k : budget_outcome_t::granted_k;
        if (outcome == budget_outcome_t::granted_k) {
            if (allocations_till_fail != unlimited_budget_k) --allocations_till_fail;
            ++granted_count;
        }
        else { ++refused_count; }
        if (failed(requests.push_back(allocation_request_t {elements_count, element_size, outcome}))) ++unlogged_count;
        return outcome;
    }

    void note_deallocation() noexcept { ++deallocation_count; }
};

/**
 *  @brief Allocator carrying an identity and, optionally, a shared @c allocation_ledger_t that budgets
 *    and records what it was asked for.
 *
 *  Refuses with @c nullptr rather than an exception, since the containers it feeds promise to throw none;
 *  an allocator with no ledger allocates freely and records nothing.
 */
template <typename element_type_>
struct stateful_allocator {
    using value_type = element_type_;
    using propagate_on_container_move_assignment = std::true_type; // Required for AVL trees

    /** @brief The byte width one element occupies, standing in for a @c void element with one byte. */
    static constexpr std::size_t element_size_k =
        sizeof(std::conditional_t<std::is_void_v<element_type_>, std::byte, element_type_>);

    int allocator_id {0};
    allocation_ledger_t *ledger {nullptr};

    stateful_allocator() noexcept = default;
    explicit stateful_allocator(int identifier) noexcept : allocator_id(identifier) {}
    explicit stateful_allocator(allocation_ledger_t &shared) noexcept : ledger(&shared) {}
    stateful_allocator(int identifier, allocation_ledger_t &shared) noexcept
        : allocator_id(identifier), ledger(&shared) {}

    template <typename other_type_>
    stateful_allocator(stateful_allocator<other_type_> const &other) noexcept
        : allocator_id(other.allocator_id), ledger(other.ledger) {}

    [[nodiscard]] element_type_ *allocate(std::size_t count) noexcept {
        if (ledger && ledger->note_request(count, element_size_k) == budget_outcome_t::refused_k) return nullptr;
        return static_cast<element_type_ *>(
            ::operator new(count * element_size_k, std::nothrow)); // allocator primitive
    }

    void deallocate(element_type_ *pointer, std::size_t) noexcept {
        if (ledger) ledger->note_deallocation();
        ::operator delete(pointer, std::nothrow); // allocator primitive
    }

    template <typename other_type_>
    bool operator==(stateful_allocator<other_type_> const &other) const noexcept {
        return allocator_id == other.allocator_id && ledger == other.ledger;
    }

    template <typename other_type_>
    bool operator!=(stateful_allocator<other_type_> const &other) const noexcept {
        return !(*this == other);
    }
};

using stateful_allocator_t = stateful_allocator<std::byte>;

#pragma endregion Stateful Allocator

#pragma region Basic Operation Test Templates

/** @brief Erases a range whether or not the container reports a status for it. */
template <typename container_type_, typename lower_type_, typename upper_type_>
void erase_range_of(container_type_ &container, lower_type_ const &lower, upper_type_ const &upper) {
    if constexpr (std::is_void_v<decltype(container.erase_range(lower, upper))>) container.erase_range(lower, upper);
    else { [[maybe_unused]] auto const status = container.erase_range(lower, upper); }
}

/**
 *  @brief Tests operations on empty container don't crash
 */
template <typename container_type_>
void test_empty_container_operations() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    // Operations on empty container should not crash
    st_verify_(!(container.contains(trivial_id_to_key<member_t>(1))));
    erase_range_of(container, trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(10));
    st_verify_eq_(container.size(), 0);
}

/**
 *  @brief Tests operations on single-element container
 */
template <typename container_type_>
void test_single_element_operations() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    auto new_member = trivial_id_to_member<member_t>(42);
    st_verify_(container.upsert(std::move(new_member)));
    st_verify_eq_(container.size(), 1);
    st_verify_(container.contains(trivial_id_to_key<member_t>(42)));
    erase_range_of(container, trivial_id_to_key<member_t>(42), trivial_id_to_key<member_t>(43));
    st_verify_eq_(container.size(), 0);
}

/**
 * @brief Tests insertion in ascending, descending, and random order
 */
template <typename container_type_>
void test_basic_insertion_patterns(std::size_t size = 100, unsigned int seed = 42) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    // Test 1: Ascending insertion
    for (std::size_t index = 0; index < size; ++index) {
        auto new_member = trivial_id_to_member<member_t>(index);
        st_verify_(container.upsert(std::move(new_member)));
        st_verify_(container.contains(trivial_id_to_key<member_t>(index)));
        st_verify_eq_(container.size(), index + 1);
    }
    st_verify_eq_(container.size(), size);
    clear_container(container);
    st_verify_eq_(container.size(), 0);

    // Test 2: Descending insertion (tests AVL rebalancing)
    for (std::size_t index = size; index > 0; --index) {
        auto new_member = trivial_id_to_member<member_t>(index);
        st_verify_(container.upsert(std::move(new_member)));
        st_verify_(container.contains(trivial_id_to_key<member_t>(index)));
    }
    st_verify_eq_(container.size(), size);
    clear_container(container);
    st_verify_eq_(container.size(), 0);

    // Test 3: Random insertion (tests worst-case AVL patterns)
    std::srand(seed); // Fixed seed for reproducibility
    for (std::size_t index = 0; index < size; ++index) {
        trivial_id_t random_id = static_cast<trivial_id_t>(std::rand());
        auto new_member = trivial_id_to_member<member_t>(random_id);
        st_verify_(container.upsert(std::move(new_member)));
        st_verify_(container.contains(trivial_id_to_key<member_t>(random_id)));
    }
}

/**
 *  @brief Tests bulk insertion via @b move iterators
 */
template <typename container_type_>
void test_bulk_insertion_from_iterators(std::size_t size = 100) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    std::vector<member_t> members;
    members.reserve(size);
    for (std::size_t index = 0; index < size; ++index) members.push_back(trivial_id_to_member<member_t>(index));

    st_verify_(
        container.insert_if_missing(std::make_move_iterator(members.begin()), std::make_move_iterator(members.end())));
    st_verify_eq_(container.size(), size);
    for (std::size_t index = 0; index < size; ++index)
        st_verify_(container.contains(trivial_id_to_key<member_t>(index)));
    clear_container(container);
    st_verify_eq_(container.size(), 0);

    // Lets do the same with update-or-insert semantics, on entries the first pass has not emptied
    members.clear();
    for (std::size_t index = 0; index < size; ++index) members.push_back(trivial_id_to_member<member_t>(index));
    st_verify_(
        succeeded(container.upsert(std::make_move_iterator(members.begin()), std::make_move_iterator(members.end()))));
    st_verify_eq_(container.size(), size);
    for (std::size_t index = 0; index < size; ++index)
        st_verify_(container.contains(trivial_id_to_key<member_t>(index)));
    clear_container(container);
    st_verify_eq_(container.size(), 0);
}

/**
 *  @brief Tests bulk upsert correctly overwrites duplicate keys
 *
 *  This test verifies that when bulk inserting elements with duplicate keys,
 *  the implementation uses upsert semantics (overwrite) rather than
 *  insert_if_missing semantics (skip).
 */
template <typename container_type_>
void test_bulk_upsert_with_duplicates() {

    static_assert(container_type_::is_associative::value, //
                  "Container must be associative for upsert test");
    static_assert(std::is_same_v<typename container_type_::value_type::mapped_type, int>,
                  "Container value_type must be int for value verification");

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    // First, insert some initial values
    for (std::size_t index = 0; index < 10; ++index) {
        auto new_member = trivial_id_to_member<member_t>(index);
        st_verify_(container.upsert(std::move(new_member)));
    }
    st_verify_eq_(container.size(), 10);

    // Verify initial values exist
    st_verify_(container.contains(trivial_id_to_key<member_t>(5)));

    // Now bulk insert with overlapping keys but different values
    std::vector<member_t> members;
    for (std::size_t index = 5; index < 15; ++index) {
        auto new_member = trivial_id_to_member<member_t>(index);
        new_member.mapped = static_cast<int>(index * 100);
        members.push_back(std::move(new_member));
    }

    st_verify_(
        succeeded(container.upsert(std::make_move_iterator(members.begin()), std::make_move_iterator(members.end()))));

    // Size should be 15 (0-14), not 20
    st_verify_eq_(container.size(), 15);

    // Verify that overlapping keys (5-9) have UPDATED values (for int-valued maps only)
    for (std::size_t index = 5; index < 10; ++index) {
        auto found_entry = container.find_copy(trivial_id_to_key<member_t>(index));
        st_verify_((found_entry) && "key missing after bulk upsert");
        st_verify_eq_(found_entry->mapped, static_cast<int>(index * 100));
    }

    // Verify new keys (10-14) were inserted
    for (std::size_t index = 10; index < 15; ++index)
        st_verify_(container.contains(trivial_id_to_key<member_t>(index)));

    // Verify old keys (0-4) still exist
    for (std::size_t index = 0; index < 5; ++index) st_verify_(container.contains(trivial_id_to_key<member_t>(index)));
}

/**
 *  @brief Tests range queries on committed HEAD state
 */
template <typename container_type_>
void test_range_query_head_state(std::size_t size = 100) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using key_t = typename mapping_key_type_or_itself<member_t>::type;
    static_assert(std::is_nothrow_copy_constructible_v<key_t>,
                  "Key type must be noexcept copy constructible for this test");

    container_t container;
    for (std::size_t index = 0; index < size; ++index)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(index)));

    // Query in windows and verify we get elements within the range
    for (std::size_t index = 0; index < size; index += 10) {
        std::size_t count = 0;
        auto min_key = trivial_id_to_key<member_t>(size);
        auto max_key = trivial_id_to_key<member_t>(0);
        container.range(trivial_id_to_key<member_t>(index), trivial_id_to_key<member_t>(index + 9),
                        [&](member_t const &member) noexcept {
                            min_key = std::min(min_key, mapping_key_or_itself<member_t>(member));
                            max_key = std::max(max_key, mapping_key_or_itself<member_t>(member));
                            count++;
                        });
        st_verify_((count) > (0));
        if (count > 0) {
            st_verify_((min_key) >= (trivial_id_to_key<member_t>(index)));
            st_verify_((max_key) <= (trivial_id_to_key<member_t>(index + 10)));
        }
    }

    for (std::size_t index = 0; index < size - 1; ++index) {
        auto idx_key = trivial_id_to_key<member_t>(index);
        auto maybe_member = container.upper_bound_copy(idx_key);
        st_verify_(maybe_member);
        st_verify_((mapping_key_or_itself(*maybe_member)) >= (idx_key));
    }

    for (std::size_t index = 0; index < size - 1; ++index) {
        auto idx_key = trivial_id_to_key<member_t>(index);
        auto maybe_member = container.lower_bound_copy(idx_key);
        st_verify_(maybe_member);
        st_verify_((mapping_key_or_itself(*maybe_member)) >= (idx_key));
    }
}

/**
 *  @brief Tests @c erase_range on committed HEAD state
 */
template <typename container_type_>
void test_erase_range_head_state(std::size_t size = 100) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    for (std::size_t index = 0; index < size; ++index)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(index)));

    for (std::size_t index = 0; index < size; index += 10) {
        auto start_key = trivial_id_to_key<member_t>(index);
        auto end_key = trivial_id_to_key<member_t>(index + 10);
        erase_range_of(container, start_key, end_key);
        for (std::size_t i = index; i < index + 10; ++i)
            st_verify_(!(container.contains(trivial_id_to_key<member_t>(i))));
    }
}

#pragma endregion Basic Operation Test Templates

#pragma region Heterogeneous Lookup Test Templates

/**
 *  @brief Tests heterogeneous lookup for @c composite_key_t by @c trivial_id_t identifier.
 *    @c composite_key_t stores metadata + unique_id, but supports lookup by just the identifier.
 *    Verifies transparent comparator allows searching without materializing full key.
 */
template <typename container_type_>
void test_heterogeneous_composite_find() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    container_t container;

    // Insert composite keys
    for (std::size_t i = 0; i < 10; ++i) {
        auto new_member = trivial_id_to_member<member_t>(i);
        if constexpr (container_t::is_associative::value) {
            new_member.key.some_metadata = i * 100;
            new_member.key.some_float = static_cast<double>(i) / 3.0;
        }
        st_verify_(container.upsert(std::move(new_member)));
    }

    // Test heterogeneous lookups
    for (std::size_t i = 0; i < 10; ++i) {
        auto lookup_by_id = container.find_copy(static_cast<trivial_id_t>(i));
        auto lookup_by_key = container.find_copy(trivial_id_to_key<member_t>(i));

        st_verify_((lookup_by_id) && "heterogeneous lookup must find the key");
        st_verify_((lookup_by_key) && "key lookup must find the key");

        st_verify_eq_(mapping_key_or_itself<member_t>(*lookup_by_id).unique_id, i);
        st_verify_eq_(mapping_key_or_itself<member_t>(*lookup_by_key).unique_id, i);

        if constexpr (container_t::is_associative::value) {
            st_verify_eq_(lookup_by_id->key.some_metadata, i * 100);
            st_verify_eq_(lookup_by_key->key.some_metadata, i * 100);
        }
    }

    // Test that non-existent keys are not found
    {
        auto lookup_by_id = container.find_copy(static_cast<trivial_id_t>(999));
        auto lookup_by_key = container.find_copy(trivial_id_to_key<member_t>(999));

        st_verify_(!(lookup_by_id));
        st_verify_(!(lookup_by_key));
    }
}

/**
 *  @brief Tests heterogeneous lookup for @c heavy_key_t by @c std::string_view.
 *    @c heavy_key_t stores heap-allocated text, but supports lookup by @c string_view.
 *    Verifies transparent comparator allows searching without materializing full key.
 */
template <typename container_type_>
void test_heterogeneous_heavy_string_view_find() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    using mapped_t = typename mapped_value_type_or_void<member_t>::type;
    container_t container;

    using namespace std::literals::string_view_literals;
    std::vector<std::string_view> test_strings = {"hello"sv, "world"sv, "foo"sv, "bar"sv, "baz"sv};

    auto make_new_key = [&](std::string_view text) { return *heavy_key_t::make(text); };
    auto make_new_member = [&](std::string_view text) {
        if constexpr (container_t::is_associative::value) return member_t {make_new_key(text), mapped_t {}};
        else return member_t {make_new_key(text)};
    };

    // Insert heavy keys from strings
    for (auto const &text : test_strings) {
        auto new_member = make_new_member(text);
        st_verify_(container.upsert(std::move(new_member)));
    }

    // Test heterogeneous lookup
    for (auto const &text : test_strings) {

        auto lookup_by_id = container.find_copy(text);
        auto lookup_by_key = container.find_copy(make_new_key(text));

        st_verify_((lookup_by_id) && "heterogeneous lookup must find the string key");
        st_verify_((lookup_by_key) && "key lookup must find the string key");

        st_verify_eq_(mapping_key_or_itself<member_t>(*lookup_by_id), text);
        st_verify_eq_(mapping_key_or_itself<member_t>(*lookup_by_key), text);
    }

    // Test that non-existent keys are not found
    {
        auto lookup_by_id = container.find_copy("missing"sv);
        auto lookup_by_key = container.find_copy(make_new_key("missing"sv));

        st_verify_(!(lookup_by_id));
        st_verify_(!(lookup_by_key));
    }
}

#pragma endregion Heterogeneous Lookup Test Templates

} // namespace ashvardanian::smashtable::scripts

/** @brief Hashes for the fixture keys, so the partitioned collection can shard them. */
namespace ashvardanian::smashtable {

template <>
struct hash<scripts::trivial_key_t> {
    std::size_t operator()(scripts::trivial_key_t const &key) const noexcept {
        return hash<scripts::trivial_id_t> {}(key.unique_id);
    }
};

template <>
struct hash<scripts::composite_key_t> {
    std::size_t operator()(scripts::composite_key_t const &key) const noexcept {
        return hash<scripts::trivial_id_t> {}(key.unique_id);
    }
};

template <>
struct hash<scripts::heavy_key_t> {
    std::size_t operator()(scripts::heavy_key_t const &key) const noexcept {
        return hash<std::string_view> {}(key.view());
    }
};

template <>
struct hash<scripts::budgeted_key_t> {
    std::size_t operator()(scripts::budgeted_key_t const &key) const noexcept {
        return hash<std::string_view> {}(key.view());
    }
};

template <>
struct hash<scripts::counted_key_t> {
    std::size_t operator()(scripts::counted_key_t const &key) const noexcept {
        return hash<scripts::trivial_id_t> {}(key.unique_id);
    }
};

template <>
struct hash<scripts::overaligned_key_t> {
    std::size_t operator()(scripts::overaligned_key_t const &key) const noexcept {
        return hash<scripts::trivial_id_t> {}(key.unique_id);
    }
};

/** @brief Only the group survives, so every identifier inside one shares a home slot. */
template <>
struct hash<scripts::colliding_key_t> {
    std::size_t operator()(scripts::colliding_key_t const &key) const noexcept {
        return scripts::colliding_key_t::group_of(key.unique_id);
    }
};

} // namespace ashvardanian::smashtable
