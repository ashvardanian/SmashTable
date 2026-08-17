/**
 *  @brief Template test functions for basic container operations. Includes insert/erase/find/range operations and
 *      heterogeneous lookup tests. Templates can be instantiated for any container supporting the common interface.
 *  @author Ash Vardanian
 *  @file scripts/test_basic.hpp
 *  @date January 12, 2023
 */
#pragma once
#include <cstring> // `std::memcmp`

#include <algorithm>   // `std::min`
#include <compare>     // `std::strong_ordering`
#include <string_view> // `std::string_view`
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
        int length = std::snprintf(buffer, sizeof(buffer), "%016lx", integer_to_encode);
        if (length <= 0 || length >= static_cast<int>(sizeof(buffer)))
            return expected<heavy_key_t>(heavy_key_t {}, status_t {errc_t::unknown_k});

        for (int i = 0; i < length; ++i) {
            auto status = result.text.push_back(char(buffer[i]));
            if (!status) return expected<heavy_key_t>(heavy_key_t {}, status);
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
            if (!status) return expected<heavy_key_t>(heavy_key_t {}, status);
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
            if (!status) return expected<heavy_key_t>(heavy_key_t {}, status);
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

/** @brief Hashes for the fixture keys, so the partitioned collection can shard them. */
} // namespace ashvardanian::smashtable::scripts

template <>
struct std::hash<ashvardanian::smashtable::scripts::trivial_key_t> {
    std::size_t operator()(ashvardanian::smashtable::scripts::trivial_key_t const &key) const noexcept {
        return std::hash<ashvardanian::smashtable::scripts::trivial_id_t> {}(key.unique_id);
    }
};

template <>
struct std::hash<ashvardanian::smashtable::scripts::composite_key_t> {
    std::size_t operator()(ashvardanian::smashtable::scripts::composite_key_t const &key) const noexcept {
        return std::hash<ashvardanian::smashtable::scripts::trivial_id_t> {}(key.unique_id);
    }
};

template <>
struct std::hash<ashvardanian::smashtable::scripts::heavy_key_t> {
    std::size_t operator()(ashvardanian::smashtable::scripts::heavy_key_t const &key) const noexcept {
        return std::hash<std::string_view> {}(key.view());
    }
};

namespace ashvardanian::smashtable::scripts {

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

#pragma region new_member Construction Helpers

/**
 * @brief Helper to construct a new key matching the @c member_type_ from an integer value
 */
template <typename member_type_>
auto trivial_id_to_key(trivial_id_t value) {
    if constexpr (has_make_method<member_type_, trivial_id_t>) return *member_type_::make(value);
    else if constexpr (is_mapping<member_type_>) return trivial_id_to_key<typename member_type_::key_type>(value);
    else return member_type_ {value};
}

/**
 * @brief Helper to construct a new @c member_type_ from an integer value
 */
template <typename member_type_>
auto trivial_id_to_member(trivial_id_t value) {
    if constexpr (has_make_method<member_type_, trivial_id_t>) return *member_type_::make(value);
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

/**
 *  @brief Stateful allocator that tracks allocations.
 *    Tests that allocators with member state work correctly.
 */
template <typename element_type_>
struct stateful_allocator {
    using value_type = element_type_;
    using propagate_on_container_move_assignment = std::true_type; // Required for AVL trees

    int allocator_id {0};               // Allocator instance identifier
    std::size_t allocation_count {0};   // Track number of allocations
    std::size_t deallocation_count {0}; // Detect memory leaks
    std::size_t allocations_till_fail {std::numeric_limits<std::size_t>::max()};

    stateful_allocator() noexcept = default;
    explicit stateful_allocator(int identifier) noexcept : allocator_id(identifier) {}

    template <typename other_type_>
    stateful_allocator(stateful_allocator<other_type_> const &other) noexcept
        : allocator_id(other.allocator_id), allocation_count(other.allocation_count) {}

    element_type_ *allocate(std::size_t n) {
        ++allocation_count;
        return static_cast<element_type_ *>(::operator new(n * sizeof(element_type_)));
    }

    void deallocate(element_type_ *p, std::size_t) noexcept { ::operator delete(p); }

    template <typename other_type_>
    bool operator==(stateful_allocator<other_type_> const &other) const noexcept {
        return allocator_id == other.allocator_id;
    }

    template <typename other_type_>
    bool operator!=(stateful_allocator<other_type_> const &other) const noexcept {
        return !(*this == other);
    }
};

using stateful_allocator_t = stateful_allocator<std::byte>;

#pragma endregion Stateful Allocator

#pragma region Basic Operation Test Templates

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
    container.erase_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(10));
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
    container.erase_range(trivial_id_to_key<member_t>(42), trivial_id_to_key<member_t>(43));
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
    st_verify_(container.upsert(std::make_move_iterator(members.begin()), std::make_move_iterator(members.end())));
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

    st_verify_(container.upsert(std::make_move_iterator(members.begin()), std::make_move_iterator(members.end())));

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
        container.erase_range(start_key, end_key);
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
