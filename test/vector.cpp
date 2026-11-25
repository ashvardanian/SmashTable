/**
 *  @file test/vector.cpp
 *  @author Ash Vardanian
 *  @date August 17, 2026
 *  @brief Tests for @c basic_vector - the fallible-construction path, growth arithmetic, rollback.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdint> // `SIZE_MAX`

#include <memory>      // `std::allocator` - the rounding allocator's storage
#include <span>        // `std::span`
#include <type_traits> // `std::true_type`, `std::is_convertible_v`, `std::is_trivially_copyable_v`
#include <utility>     // `std::move`

#include <smashtable/basic_vector.hpp>

#include "harness.hpp"
#include "sequence.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Element Types

/** An element only constructible through @c make, which is what drives @c emplace_back's slow path.
 *  Its value constructor is deliberately throwing, so the nothrow branch cannot claim it. */
struct fallible_element_t {
    int value {0};

    fallible_element_t() noexcept = default;
    fallible_element_t(fallible_element_t &&) noexcept = default;
    fallible_element_t &operator=(fallible_element_t &&) noexcept = default;
    fallible_element_t(fallible_element_t const &) = default;
    fallible_element_t &operator=(fallible_element_t const &) = default;

    /** Throwing on purpose, so @c std::is_nothrow_constructible_v rejects this path. */
    explicit fallible_element_t(int requested, int) : value(requested) {}

    /** Refuses negative values, which is the failure the caller must see as a status. */
    [[nodiscard]] static expected<fallible_element_t> make(int requested) noexcept {
        if (requested < 0) return status_t::invalid_argument_k;
        fallible_element_t made;
        made.value = requested;
        return expected<fallible_element_t>(std::move(made), success_k);
    }

    /** Deep copy that never fails, so @c copy_safely has a route for this type. */
    [[nodiscard]] expected<fallible_element_t> copy() const noexcept {
        fallible_element_t made;
        made.value = value;
        return expected<fallible_element_t>(std::move(made), success_k);
    }
};

/** An element whose copy fails past a budget, exercising the rollback paths. */
struct budgeted_element_t {
    static inline std::size_t copies_left = 0;

    int value {0};

    budgeted_element_t() noexcept = default;
    budgeted_element_t(budgeted_element_t &&) noexcept = default;
    budgeted_element_t &operator=(budgeted_element_t &&) noexcept = default;
    budgeted_element_t(budgeted_element_t const &) = delete;
    budgeted_element_t &operator=(budgeted_element_t const &) = delete;

    [[nodiscard]] expected<budgeted_element_t> copy() const noexcept {
        if (copies_left == 0) return status_t::out_of_memory_heap_k;
        --copies_left;
        budgeted_element_t made;
        made.value = value;
        return expected<budgeted_element_t>(std::move(made), success_k);
    }
};

/** A trivially copyable element whose default constructor counts its calls, so a resize meant to
 *  construct nothing can be seen constructing nothing. */
struct construction_counted_element_t {
    static inline std::size_t constructions = 0;

    int value {0};

    construction_counted_element_t() noexcept { ++constructions; }
};

/** An allocator that refuses every request and remembers the largest one it was asked for. A
 *  hand-written allocator that multiplies without checking would answer a wrapping byte count with
 *  a small block, so the vector must never put such a count in front of one. */
template <typename value_type_>
struct recording_allocator {
    // Read only for the instantiation a test names; the rebound ones only ever record.
    [[maybe_unused]] static inline std::size_t largest_request = 0;
    [[maybe_unused]] static inline std::size_t requests_count = 0;

    using value_type = value_type_;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::true_type;

    constexpr recording_allocator() noexcept = default;
    template <typename other_type_>
    constexpr recording_allocator(recording_allocator<other_type_> const &) noexcept {}

    [[nodiscard]] value_type_ *allocate(std::size_t count) noexcept {
        largest_request = max_of(largest_request, count);
        ++requests_count;
        return nullptr;
    }
    void deallocate(value_type_ *, std::size_t) noexcept {}

    template <typename other_type_>
    constexpr bool operator==(recording_allocator<other_type_> const &) const noexcept {
        return true;
    }
};

#pragma endregion Element Types

#pragma region Tests

/** The @c make branch of @c emplace_back must compile at all, and must report its failures. */
static void test_emplace_back_through_make() {
    static_assert(!std::is_nothrow_constructible_v<fallible_element_t, int>,
                  "the test element must not qualify for the nothrow branch");
    static_assert(a_type_with_make_method<fallible_element_t, int>,
                  "the test element must qualify for the make branch");

    basic_vector<fallible_element_t> vector;
    expect_equal(vector.emplace_back(7), success_k);
    expect_equal(vector.emplace_back(8), success_k);
    if (!expect_equal(vector.size(), 2u)) return;
    expect_equal(vector[0].value, 7);
    expect_equal(vector[1].value, 8);

    // A refused `make` propagates its reason and leaves the vector as it was.
    expect_equal(vector.emplace_back(-1), status_t::invalid_argument_k);
    if (!expect_equal(vector.size(), 2u)) return;

    expect_equal(vector.reserve(8), success_k);
    expect_equal(vector.emplace_back(assume_reserved, 9), success_k);
    if (!expect_equal(vector.size(), 3u)) return;
    expect_equal(vector[2].value, 9);
    expect_equal(vector.emplace_back(assume_reserved, -5), status_t::invalid_argument_k);
    expect_equal(vector.size(), 3u);

    // The nothrow branch still wins where it applies.
    basic_vector<int> integers;
    expect_equal(integers.emplace_back(42), success_k);
    expect_equal(integers[0], 42);
}

/** A capacity whose byte count cannot be addressed is refused rather than allocated small. */
static void test_reserve_refuses_wrapping_capacity() {
    constexpr std::size_t max_capacity = SIZE_MAX / sizeof(long long);
    using recorded_vector_t = basic_vector<long long, recording_allocator<long long>>;

    recording_allocator<long long>::largest_request = 0;
    recording_allocator<long long>::requests_count = 0;

    recorded_vector_t vector;
    expect_equal(vector.reserve(max_capacity + 2), status_t::out_of_memory_heap_k);
    expect_equal(vector.capacity(), 0u);
    expect_equal(vector.size(), 0u);
    // The request never reached the allocator, which would have answered it with a small block.
    expect_equal(recording_allocator<long long>::requests_count, 0u);

    // A capacity that is merely unsatisfiable still goes through, and goes through honestly.
    expect_equal(vector.reserve(max_capacity), status_t::out_of_memory_heap_k);
    expect_equal(vector.capacity(), 0u);
    expect_equal(recording_allocator<long long>::requests_count, 1u);
    expect_true(recording_allocator<long long>::largest_request <= max_capacity);

    // Ordinary growth is untouched by the guard: an empty vector seeds four slots, and the next
    // reserve doubles those rather than granting the five that were asked for. Both counts encode
    // the growth policy, so a change to it is meant to fail here.
    basic_vector<long long> heap_vector;
    expect_equal(heap_vector.reserve(1), success_k);
    expect_equal(heap_vector.capacity(), 4u);
    expect_equal(heap_vector.reserve(5), success_k);
    expect_equal(heap_vector.capacity(), 8u);
    expect_equal(heap_vector.push_back(11), success_k);
    expect_equal(heap_vector[0], 11);
}

/** An allocator whose @c allocate_at_least grants element counts rounded up to a multiple of 16. */
template <typename value_type_>
struct rounding_allocator {
    using value_type = value_type_;
    using propagate_on_container_move_assignment = std::true_type;

    struct granted_t {
        value_type_ *ptr;
        std::size_t count;
    };

    rounding_allocator() noexcept = default;
    template <typename other_type_>
    rounding_allocator(rounding_allocator<other_type_> const &) noexcept {}

    [[nodiscard]] value_type_ *allocate(std::size_t count) noexcept {
        return std::allocator<value_type_> {}.allocate(count);
    }
    [[nodiscard]] granted_t allocate_at_least(std::size_t count) noexcept {
        std::size_t const granted = (count + 15) / 16 * 16;
        return {allocate(granted), granted};
    }
    void deallocate(value_type_ *pointer, std::size_t count) noexcept {
        std::allocator<value_type_> {}.deallocate(pointer, count);
    }
    friend bool operator==(rounding_allocator const &, rounding_allocator const &) noexcept { return true; }
};

/** An allocator granting more than asked is kept in full. */
static void test_keeps_the_granted_capacity() {
    basic_vector<int, rounding_allocator<int>> vector;
    expect_equal(vector.push_back(int {1}), success_k);
    expect_true(vector.capacity() == 16u);
    std::size_t const granted = vector.capacity();
    for (std::size_t index = 1; index < granted; ++index) expect_equal(vector.push_back(int {2}), success_k);
    expect_equal(vector.capacity(), granted);
    expect_equal(vector.push_back(int {3}), success_k);
    expect_true(vector.capacity() > granted);
}

/** Growth amortizes: appending one at a time must double rather than grow by one. */
static void test_growth_is_amortized() {
    basic_vector<int> vector;
    for (int value = 0; value < 100; ++value) expect_equal(vector.push_back(int {value}), success_k);
    if (!expect_equal(vector.size(), 100u)) return;
    // Seeded at 4 and doubled to 8, 16, 32, 64, 128 - the growth policy spelled out, so a change to
    // it fails here instead of hiding inside a range.
    expect_equal(vector.capacity(), 128u);
    for (int value = 0; value < 100; ++value) expect_equal(vector[static_cast<std::size_t>(value)], value);

    vector.pop_back();
    expect_equal(vector.size(), 99u);
    vector.clear();
    expect_true(vector.empty());
    expect_equal(vector.capacity(), 128u);
}

/** A failed @c resize leaves the elements alone, whatever it did to the capacity. */
static void test_resize_rolls_back_elements() {
    basic_vector<budgeted_element_t> vector;
    expect_equal(vector.push_back(budgeted_element_t {}), success_k);
    vector[0].value = 3;

    budgeted_element_t pattern;
    pattern.value = 9;

    // Two copies granted, four demanded: the three constructed before the refusal must be destroyed.
    budgeted_element_t::copies_left = 2;
    expect_equal(vector.resize(5, pattern), status_t::out_of_memory_heap_k);
    if (!expect_equal(vector.size(), 1u)) return;
    expect_equal(vector[0].value, 3);

    budgeted_element_t::copies_left = 16;
    expect_equal(vector.resize(5, pattern), success_k);
    if (!expect_equal(vector.size(), 5u)) return;
    expect_equal(vector[4].value, 9);

    expect_equal(vector.resize(2), success_k);
    expect_equal(vector.size(), 2u);
}

/** The uninitialized @c resize moves the size around the elements it keeps and constructs none. */
static void test_resizes_without_constructing() {
    static_assert(std::is_trivially_copyable_v<construction_counted_element_t>,
                  "the test element must qualify for the uninitialized resize");

    basic_vector<construction_counted_element_t> vector;
    expect_equal(vector.resize(3), success_k);
    for (std::size_t index = 0; index < 3; ++index) vector[index].value = static_cast<int>(index) + 7;
    construction_counted_element_t::constructions = 0;

    expect_equal(vector.resize(assume_uninitialized, 40), success_k);
    if (!expect_equal(vector.size(), 40u)) return;
    expect_true(vector.capacity() >= 40u);
    expect_equal(construction_counted_element_t::constructions, 0u);
    for (std::size_t index = 0; index < 3; ++index) expect_equal(vector[index].value, static_cast<int>(index) + 7);

    std::size_t const capacity_before = vector.capacity();
    expect_equal(vector.resize(assume_uninitialized, 2), success_k);
    if (!expect_equal(vector.size(), 2u)) return;
    expect_equal(vector.capacity(), capacity_before);
    expect_equal(vector[1].value, 8);
}

/** @c assign replaces shorter and longer contents alike, and takes a vector through its span. */
static void test_assign_replaces_contents() {
    static_assert(std::is_convertible_v<basic_vector<int> &, std::span<int>>, "a vector lends itself as a span");

    basic_vector<int> source;
    for (int value = 0; value < 6; ++value) expect_equal(source.push_back(int {value}), success_k);

    basic_vector<int> vector;
    expect_equal(vector.push_back(42), success_k);
    expect_equal(vector.assign(source), success_k);
    if (!expect_equal(vector.size(), 6u)) return;
    for (std::size_t index = 0; index < 6; ++index) expect_equal(vector[index], source[index]);

    int const shorter[2] = {7, 8};
    expect_equal(vector.assign(shorter), success_k);
    if (!expect_equal(vector.size(), 2u)) return;
    expect_equal(vector[0], 7);
    expect_equal(vector[1], 8);
}

/** A refused allocation or element copy leaves the elements and the storage as they were. */
static void test_refused_growth_changes_nothing() {
    allocation_ledger_t ledger;
    {
        basic_vector<int, stateful_allocator<int>> vector {stateful_allocator<int>(ledger)};
        for (int value = 0; value < 3; ++value) expect_equal(vector.push_back(int {value}), success_k);
        std::size_t const capacity_before = vector.capacity();
        int const longer[8] = {10, 11, 12, 13, 14, 15, 16, 17};

        ledger.refuse_everything();
        expect_equal(vector.resize(assume_uninitialized, 64), status_t::out_of_memory_heap_k);
        expect_equal(vector.assign(longer), status_t::out_of_memory_heap_k);
        ledger.allow(unlimited_budget_k);
        if (!expect_equal(vector.size(), 3u)) return;
        expect_equal(vector.capacity(), capacity_before);
        for (std::size_t index = 0; index < 3; ++index) expect_equal(vector[index], static_cast<int>(index));
    }
    ledger.check_balanced();

    // Trivially copyable with a deleted copy constructor, so `assign` copies through `.copy()`.
    basic_vector<budgeted_element_t> source;
    for (int value = 0; value < 5; ++value) {
        expect_equal(source.push_back(budgeted_element_t {}), success_k);
        source.back().value = value;
    }
    basic_vector<budgeted_element_t> vector;
    expect_equal(vector.push_back(budgeted_element_t {}), success_k);
    vector[0].value = 3;

    budgeted_element_t::copies_left = 2;
    expect_equal(vector.assign(source), status_t::out_of_memory_heap_k);
    if (!expect_equal(vector.size(), 1u)) return;
    expect_equal(vector[0].value, 3);

    budgeted_element_t::copies_left = 16;
    expect_equal(vector.assign(source), success_k);
    if (!expect_equal(vector.size(), 5u)) return;
    expect_equal(vector[4].value, 4);
}

/** @c copy is all-or-nothing, and @c swap moves the storage across. */
static void test_copy_and_swap() {
    basic_vector<fallible_element_t> vector;
    for (int value = 0; value < 5; ++value) expect_equal(vector.emplace_back(value), success_k);

    auto [copied, status] = vector.copy();
    if (!expect_equal(status, success_k)) return;
    expect_equal(copied.size(), 5u);
    for (std::size_t index = 0; index < 5; ++index) expect_equal(copied[index].value, vector[index].value);

    basic_vector<fallible_element_t> empty;
    expect_equal(empty.swap(vector), success_k);
    expect_equal(empty.size(), 5u);
    expect_equal(vector.size(), 0u);

    // A failing copy hands back nothing at all rather than a partial vector.
    basic_vector<budgeted_element_t> budgeted;
    for (int value = 0; value < 4; ++value) expect_equal(budgeted.push_back(budgeted_element_t {}), success_k);
    budgeted_element_t::copies_left = 2;
    expect_equal(budgeted.copy().status(), status_t::out_of_memory_heap_k);
    expect_equal(budgeted.size(), 4u);
}

/** Positional inserts land where asked and shift the rest up, at both ends and in the middle. */
static void test_inserts_at_a_position() {
    basic_vector<int> vector;
    for (int value = 0; value < 5; ++value) expect_equal(vector.push_back(int {value}), success_k);

    expect_equal(vector.insert(0, 99), success_k);
    if (!expect_equal(vector.size(), 6u)) return;
    expect_equal(vector[0], 99);
    expect_equal(vector[1], 0);
    expect_equal(vector[5], 4);

    expect_equal(vector.insert(vector.size(), 77), success_k);
    if (!expect_equal(vector.size(), 7u)) return;
    expect_equal(vector[6], 77);

    expect_equal(vector.insert(3, 55), success_k);
    if (!expect_equal(vector.size(), 8u)) return;
    expect_equal(vector[2], 1);
    expect_equal(vector[3], 55);
    expect_equal(vector[4], 2);

    // An insert into an empty vector is the same call, not a special case for the caller.
    basic_vector<int> empty;
    expect_equal(empty.insert(0, 7), success_k);
    if (!expect_equal(empty.size(), 1u)) return;
    expect_equal(empty[0], 7);

    // Every element survives an insert that had to grow the storage.
    basic_vector<int> growing;
    for (int value = 0; value < 64; ++value) expect_equal(growing.insert(0, int {value}), success_k);
    if (!expect_equal(growing.size(), 64u)) return;
    for (std::size_t index = 0; index < growing.size(); ++index)
        expect_equal(growing[index], static_cast<int>(63 - index));
}

/** Positional erases remove exactly what was named and shift the rest down. */
static void test_erases_at_a_position() {
    basic_vector<int> vector;
    for (int value = 0; value < 6; ++value) expect_equal(vector.push_back(int {value}), success_k);

    vector.erase(0);
    if (!expect_equal(vector.size(), 5u)) return;
    expect_equal(vector[0], 1);

    vector.erase(vector.size() - 1);
    if (!expect_equal(vector.size(), 4u)) return;
    expect_equal(vector[3], 4);

    vector.erase(1);
    if (!expect_equal(vector.size(), 3u)) return;
    expect_equal(vector[0], 1);
    expect_equal(vector[1], 3);
    expect_equal(vector[2], 4);

    // A ranged erase takes a whole run out at once, including the run that reaches the end.
    basic_vector<int> ranged;
    for (int value = 0; value < 8; ++value) expect_equal(ranged.push_back(int {value}), success_k);
    ranged.erase(2, 3);
    if (!expect_equal(ranged.size(), 5u)) return;
    expect_equal(ranged[1], 1);
    expect_equal(ranged[2], 5);
    ranged.erase(3, 2);
    if (!expect_equal(ranged.size(), 3u)) return;
    expect_equal(ranged[2], 5);
    ranged.erase(0, 3);
    expect_equal(ranged.size(), 0u);

    // Erasing nothing is legal and moves nothing.
    basic_vector<int> untouched;
    for (int value = 0; value < 3; ++value) expect_equal(untouched.push_back(int {value}), success_k);
    untouched.erase(1, 0);
    if (!expect_equal(untouched.size(), 3u)) return;
    expect_equal(untouched[1], 1);
}

/** A shift moves every element exactly once, so no value is duplicated or lost across the gap. */
static void test_shifts_every_element_once() {
    basic_vector<fallible_element_t> vector;
    for (int value = 0; value < 32; ++value) expect_equal(vector.emplace_back(value), success_k);

    expect_equal(vector.insert(7, fallible_element_t {}), success_k);
    vector[7].value = 999;
    if (!expect_equal(vector.size(), 33u)) return;
    for (std::size_t index = 0; index < 7; ++index) expect_equal(vector[index].value, static_cast<int>(index));
    expect_equal(vector[7].value, 999);
    for (std::size_t index = 8; index < vector.size(); ++index)
        expect_equal(vector[index].value, static_cast<int>(index - 1));

    vector.erase(7);
    if (!expect_equal(vector.size(), 32u)) return;
    for (std::size_t index = 0; index < vector.size(); ++index)
        expect_equal(vector[index].value, static_cast<int>(index));
}

/** The reserved form shifts into room already secured and asks the allocator for nothing. */
static void test_inserts_into_reserved_room() {
    basic_vector<int> vector;
    expect_equal(vector.reserve(8), success_k);
    for (int value = 0; value < 4; ++value) expect_equal(vector.push_back(int {value}), success_k);
    std::size_t const capacity_before = vector.capacity();

    expect_equal(vector.insert(assume_reserved, 2, 42), success_k);
    if (!expect_equal(vector.size(), 5u)) return;
    expect_equal(vector.capacity(), capacity_before);
    expect_equal(vector[1], 1);
    expect_equal(vector[2], 42);
    expect_equal(vector[3], 2);
    expect_equal(vector[4], 3);

    expect_equal(vector.insert(assume_reserved, vector.size(), 77), success_k);
    if (!expect_equal(vector.size(), 6u)) return;
    expect_equal(vector[5], 77);
}

/** Every erased element reaches the callback exactly once, before the tail shifts over it. */
static void test_erases_hand_each_element_out() {
    basic_vector<int> vector;
    for (int value = 0; value < 8; ++value) expect_equal(vector.push_back(int {value}), success_k);

    int handed[8] = {};
    std::size_t handed_count = 0;
    auto collect = [&](int &element) noexcept { handed[handed_count++] = element; };

    vector.erase(2, 3, collect);
    expect_equal(handed_count, 3u);
    expect_equal(handed[0], 2);
    expect_equal(handed[1], 3);
    expect_equal(handed[2], 4);
    if (!expect_equal(vector.size(), 5u)) return;
    expect_equal(vector[1], 1);
    expect_equal(vector[2], 5);
    expect_equal(vector[4], 7);

    vector.erase(0, collect);
    expect_equal(handed_count, 4u);
    expect_equal(handed[3], 0);
    expect_equal(vector[0], 1);

    // A count beside the position is a count, never a callback.
    vector.erase(1, 2);
    expect_equal(handed_count, 4u);
    if (!expect_equal(vector.size(), 2u)) return;
    expect_equal(vector[0], 1);
    expect_equal(vector[1], 7);
}

/** A predicate sweeps in one pass, keeps the survivors in order, and answers how many went. */
static void test_erase_if_compacts_in_one_pass() {
    basic_vector<int> vector;
    for (int value = 0; value < 10; ++value) expect_equal(vector.push_back(int {value}), success_k);

    std::size_t handed_evens = 0;
    auto count_even = [&](int &element) noexcept { handed_evens += element % 2 == 0; };
    auto is_even = [](int const &element) noexcept { return element % 2 == 0; };

    expect_equal(vector.erase_if(is_even, count_even), 5u);
    expect_equal(handed_evens, 5u);
    if (!expect_equal(vector.size(), 5u)) return;
    for (std::size_t index = 0; index < vector.size(); ++index)
        expect_equal(vector[index], static_cast<int>(2 * index + 1));

    expect_equal(vector.erase_if(is_even), 0u);
    expect_equal(vector.size(), 5u);

    expect_equal(vector.erase_if([](int const &) noexcept { return true; }), 5u);
    expect_equal(vector.size(), 0u);
}

/** Moves transfer ownership and leave the source empty rather than aliasing its storage. */
static void test_move_semantics() {
    auto [built, status] = basic_vector<int>::make(16);
    if (!expect_equal(status, success_k)) return;
    basic_vector<int> vector = std::move(built);
    // A first reserve grants exactly what it was asked for once that clears the four-slot seed.
    expect_equal(vector.capacity(), 16u);
    for (int value = 0; value < 16; ++value) expect_equal(vector.push_back(assume_reserved, int {value}), success_k);

    basic_vector<int> moved = std::move(vector);
    expect_equal(moved.size(), 16u);
    expect_equal(vector.size(), 0u);
    expect_equal(vector.capacity(), 0u);
    expect_equal(vector.data(), nullptr);

    basic_vector<int> assigned;
    expect_equal(assigned.push_back(1), success_k);
    assigned = std::move(moved);
    if (!expect_equal(assigned.size(), 16u)) return;
    expect_equal(assigned.back(), 15);
    expect_equal(assigned.front(), 0);
    expect_true(assigned.at(15) != nullptr);
    expect_equal(assigned.at(16), nullptr);
}

/** The suite every sequence answers, over a vector of counted elements and one of tracked
 *  allocations. */
void test_sequence_suite() {
    check_sequence_tags<basic_vector<counted_key_t>>();
    check_sequence_order<basic_vector<counted_key_t>>();
    check_sequence_element_lifetimes<basic_vector<counted_key_t>>();
    check_sequence_move_semantics<basic_vector<counted_key_t>>();
    check_sequence_allocator_ledger<basic_vector<counted_key_t, stateful_allocator<counted_key_t>>>();
}

#pragma endregion Tests

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

int main(int argc, char **argv) {
    start_test_run(argc, argv);

    run_test_case("test_emplace_back_through_make", test_emplace_back_through_make);
    run_test_case("test_reserve_refuses_wrapping_capacity", test_reserve_refuses_wrapping_capacity);
    run_test_case("test_keeps_the_granted_capacity", test_keeps_the_granted_capacity);
    run_test_case("test_growth_is_amortized", test_growth_is_amortized);
    run_test_case("test_resize_rolls_back_elements", test_resize_rolls_back_elements);
    run_test_case("test_resizes_without_constructing", test_resizes_without_constructing);
    run_test_case("test_assign_replaces_contents", test_assign_replaces_contents);
    run_test_case("test_refused_growth_changes_nothing", test_refused_growth_changes_nothing);
    run_test_case("test_copy_and_swap", test_copy_and_swap);
    run_test_case("test_move_semantics", test_move_semantics);
    run_test_case("test_inserts_at_a_position", test_inserts_at_a_position);
    run_test_case("test_erases_at_a_position", test_erases_at_a_position);
    run_test_case("test_shifts_every_element_once", test_shifts_every_element_once);
    run_test_case("test_inserts_into_reserved_room", test_inserts_into_reserved_room);
    run_test_case("test_erases_hand_each_element_out", test_erases_hand_each_element_out);
    run_test_case("test_erase_if_compacts_in_one_pass", test_erase_if_compacts_in_one_pass);
    run_test_case("test_sequence_suite", test_sequence_suite);

    return finish_test_run();
}
