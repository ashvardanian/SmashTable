/**
 *  @file test/vector.cpp
 *  @author Ash Vardanian
 *  @date August 17, 2026
 *  @brief Tests for @c basic_vector - the fallible-construction path, growth arithmetic, rollback.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1

#include <cstddef> // `std::size_t`
#include <cstdint> // `SIZE_MAX`

#include <new>         // `::operator new`, `std::nothrow`
#include <type_traits> // `std::true_type`
#include <utility>     // `std::move`

#include <smashtable/basic_vector.hpp>

#include "harness.hpp"
#include "sequence.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::test;

namespace {

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
        largest_request = larger_of(largest_request, count);
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
static void vector_emplace_back_through_make() {
    static_assert(!std::is_nothrow_constructible_v<fallible_element_t, int>,
                  "the test element must not qualify for the nothrow branch");
    static_assert(has_make_method<fallible_element_t, int>, "the test element must qualify for the make branch");

    basic_vector<fallible_element_t> vector;
    st_verify_(vector.emplace_back(7));
    st_verify_(vector.emplace_back(8));
    st_verify_eq_(vector.size(), 2u);
    st_verify_eq_(vector[0].value, 7);
    st_verify_eq_(vector[1].value, 8);

    // A refused `make` propagates its reason and leaves the vector as it was.
    st_verify_eq_(vector.emplace_back(-1), status_t::invalid_argument_k);
    st_verify_eq_(vector.size(), 2u);

    st_verify_(vector.reserve(8));
    st_verify_(vector.emplace_back(assume_reserved, 9));
    st_verify_eq_(vector.size(), 3u);
    st_verify_eq_(vector[2].value, 9);
    st_verify_eq_(vector.emplace_back(assume_reserved, -5), status_t::invalid_argument_k);
    st_verify_eq_(vector.size(), 3u);

    // The nothrow branch still wins where it applies.
    basic_vector<int> integers;
    st_verify_(integers.emplace_back(42));
    st_verify_eq_(integers[0], 42);
}

/** A capacity whose byte count cannot be addressed is refused rather than allocated small. */
static void vector_reserve_refuses_wrapping_capacity() {
    constexpr std::size_t max_capacity = SIZE_MAX / sizeof(long long);
    using recorded_vector_t = basic_vector<long long, recording_allocator<long long>>;

    recording_allocator<long long>::largest_request = 0;
    recording_allocator<long long>::requests_count = 0;

    recorded_vector_t vector;
    st_verify_eq_(vector.reserve(max_capacity + 2), status_t::out_of_memory_heap_k);
    st_verify_eq_(vector.capacity(), 0u);
    st_verify_eq_(vector.size(), 0u);
    // The request never reached the allocator, which would have answered it with a small block.
    st_verify_eq_(recording_allocator<long long>::requests_count, 0u);

    // A capacity that is merely unsatisfiable still goes through, and goes through honestly.
    st_verify_eq_(vector.reserve(max_capacity), status_t::out_of_memory_heap_k);
    st_verify_eq_(vector.capacity(), 0u);
    st_verify_eq_(recording_allocator<long long>::requests_count, 1u);
    st_verify_le_(recording_allocator<long long>::largest_request, max_capacity);

    // Ordinary growth is untouched by the guard: an empty vector seeds four slots, and the next
    // reserve doubles those rather than granting the five that were asked for. Both counts encode
    // the growth policy, so a change to it is meant to fail here.
    basic_vector<long long> heap_vector;
    st_verify_(heap_vector.reserve(1));
    st_verify_eq_(heap_vector.capacity(), 4u);
    st_verify_(heap_vector.reserve(5));
    st_verify_eq_(heap_vector.capacity(), 8u);
    st_verify_(heap_vector.push_back(11));
    st_verify_eq_(heap_vector[0], 11);
}

/** Growth amortizes: appending one at a time must double rather than grow by one. */
static void vector_growth_is_amortized() {
    basic_vector<int> vector;
    for (int value = 0; value < 100; ++value) st_verify_(vector.push_back(int {value}));
    st_verify_eq_(vector.size(), 100u);
    // Seeded at 4 and doubled to 8, 16, 32, 64, 128 - the growth policy spelled out, so a change to
    // it fails here instead of hiding inside a range.
    st_verify_eq_(vector.capacity(), 128u);
    for (int value = 0; value < 100; ++value) st_verify_eq_(vector[static_cast<std::size_t>(value)], value);

    vector.pop_back();
    st_verify_eq_(vector.size(), 99u);
    vector.clear();
    st_verify_(vector.empty());
    st_verify_eq_(vector.capacity(), 128u);
}

/** A failed @c resize leaves the elements alone, whatever it did to the capacity. */
static void vector_resize_rolls_back_elements() {
    basic_vector<budgeted_element_t> vector;
    st_verify_(vector.push_back(budgeted_element_t {}));
    vector[0].value = 3;

    budgeted_element_t pattern;
    pattern.value = 9;

    // Two copies granted, four demanded: the three constructed before the refusal must be destroyed.
    budgeted_element_t::copies_left = 2;
    st_verify_eq_(vector.resize(5, pattern), status_t::out_of_memory_heap_k);
    st_verify_eq_(vector.size(), 1u);
    st_verify_eq_(vector[0].value, 3);

    budgeted_element_t::copies_left = 16;
    st_verify_(vector.resize(5, pattern));
    st_verify_eq_(vector.size(), 5u);
    st_verify_eq_(vector[4].value, 9);

    st_verify_(vector.resize(2));
    st_verify_eq_(vector.size(), 2u);
}

/** @c copy is all-or-nothing, and @c swap moves the storage across. */
static void vector_copy_and_swap() {
    basic_vector<fallible_element_t> vector;
    for (int value = 0; value < 5; ++value) st_verify_(vector.emplace_back(value));

    auto copied = vector.copy();
    st_verify_(static_cast<bool>(copied));
    st_verify_eq_(copied->size(), 5u);
    for (std::size_t index = 0; index < 5; ++index) st_verify_eq_((*copied)[index].value, vector[index].value);

    basic_vector<fallible_element_t> empty;
    st_verify_(empty.swap(vector));
    st_verify_eq_(empty.size(), 5u);
    st_verify_eq_(vector.size(), 0u);

    // A failing copy hands back nothing at all rather than a partial vector.
    basic_vector<budgeted_element_t> budgeted;
    for (int value = 0; value < 4; ++value) st_verify_(budgeted.push_back(budgeted_element_t {}));
    budgeted_element_t::copies_left = 2;
    auto refused = budgeted.copy();
    st_verify_(!refused);
    st_verify_eq_(refused.status(), status_t::out_of_memory_heap_k);
    st_verify_eq_(budgeted.size(), 4u);
}

/** Positional inserts land where asked and shift the rest up, at both ends and in the middle. */
static void vector_inserts_at_a_position() {
    basic_vector<int> vector;
    for (int value = 0; value < 5; ++value) st_verify_(vector.push_back(int {value}));

    st_verify_(vector.insert(0, 99));
    st_verify_eq_(vector.size(), 6u);
    st_verify_eq_(vector[0], 99);
    st_verify_eq_(vector[1], 0);
    st_verify_eq_(vector[5], 4);

    st_verify_(vector.insert(vector.size(), 77));
    st_verify_eq_(vector.size(), 7u);
    st_verify_eq_(vector[6], 77);

    st_verify_(vector.insert(3, 55));
    st_verify_eq_(vector.size(), 8u);
    st_verify_eq_(vector[2], 1);
    st_verify_eq_(vector[3], 55);
    st_verify_eq_(vector[4], 2);

    // An insert into an empty vector is the same call, not a special case for the caller.
    basic_vector<int> empty;
    st_verify_(empty.insert(0, 7));
    st_verify_eq_(empty.size(), 1u);
    st_verify_eq_(empty[0], 7);

    // Every element survives an insert that had to grow the storage.
    basic_vector<int> growing;
    for (int value = 0; value < 64; ++value) st_verify_(growing.insert(0, int {value}));
    st_verify_eq_(growing.size(), 64u);
    for (std::size_t index = 0; index < growing.size(); ++index)
        st_verify_eq_(growing[index], static_cast<int>(63 - index));
}

/** Positional erases remove exactly what was named and shift the rest down. */
static void vector_erases_at_a_position() {
    basic_vector<int> vector;
    for (int value = 0; value < 6; ++value) st_verify_(vector.push_back(int {value}));

    vector.erase(0);
    st_verify_eq_(vector.size(), 5u);
    st_verify_eq_(vector[0], 1);

    vector.erase(vector.size() - 1);
    st_verify_eq_(vector.size(), 4u);
    st_verify_eq_(vector[3], 4);

    vector.erase(1);
    st_verify_eq_(vector.size(), 3u);
    st_verify_eq_(vector[0], 1);
    st_verify_eq_(vector[1], 3);
    st_verify_eq_(vector[2], 4);

    // A ranged erase takes a whole run out at once, including the run that reaches the end.
    basic_vector<int> ranged;
    for (int value = 0; value < 8; ++value) st_verify_(ranged.push_back(int {value}));
    ranged.erase(2, 3);
    st_verify_eq_(ranged.size(), 5u);
    st_verify_eq_(ranged[1], 1);
    st_verify_eq_(ranged[2], 5);
    ranged.erase(3, 2);
    st_verify_eq_(ranged.size(), 3u);
    st_verify_eq_(ranged[2], 5);
    ranged.erase(0, 3);
    st_verify_eq_(ranged.size(), 0u);

    // Erasing nothing is legal and moves nothing.
    basic_vector<int> untouched;
    for (int value = 0; value < 3; ++value) st_verify_(untouched.push_back(int {value}));
    untouched.erase(1, 0);
    st_verify_eq_(untouched.size(), 3u);
    st_verify_eq_(untouched[1], 1);
}

/** A shift moves every element exactly once, so no value is duplicated or lost across the gap. */
static void vector_shifts_every_element_once() {
    basic_vector<fallible_element_t> vector;
    for (int value = 0; value < 32; ++value) st_verify_(vector.emplace_back(value));

    st_verify_(vector.insert(7, fallible_element_t {}));
    vector[7].value = 999;
    st_verify_eq_(vector.size(), 33u);
    for (std::size_t index = 0; index < 7; ++index) st_verify_eq_(vector[index].value, static_cast<int>(index));
    st_verify_eq_(vector[7].value, 999);
    for (std::size_t index = 8; index < vector.size(); ++index)
        st_verify_eq_(vector[index].value, static_cast<int>(index - 1));

    vector.erase(7);
    st_verify_eq_(vector.size(), 32u);
    for (std::size_t index = 0; index < vector.size(); ++index)
        st_verify_eq_(vector[index].value, static_cast<int>(index));
}

/** The reserved form shifts into room already secured and asks the allocator for nothing. */
static void vector_inserts_into_reserved_room() {
    basic_vector<int> vector;
    st_verify_(vector.reserve(8));
    for (int value = 0; value < 4; ++value) st_verify_(vector.push_back(int {value}));
    std::size_t const capacity_before = vector.capacity();

    st_verify_(vector.insert(assume_reserved, 2, 42));
    st_verify_eq_(vector.size(), 5u);
    st_verify_eq_(vector.capacity(), capacity_before);
    st_verify_eq_(vector[1], 1);
    st_verify_eq_(vector[2], 42);
    st_verify_eq_(vector[3], 2);
    st_verify_eq_(vector[4], 3);

    st_verify_(vector.insert(assume_reserved, vector.size(), 77));
    st_verify_eq_(vector.size(), 6u);
    st_verify_eq_(vector[5], 77);
}

/** Every erased element reaches the callback exactly once, before the tail shifts over it. */
static void vector_erases_hand_each_element_out() {
    basic_vector<int> vector;
    for (int value = 0; value < 8; ++value) st_verify_(vector.push_back(int {value}));

    int handed[8] = {};
    std::size_t handed_count = 0;
    auto collect = [&](int &element) noexcept { handed[handed_count++] = element; };

    vector.erase(2, 3, collect);
    st_verify_eq_(handed_count, 3u);
    st_verify_eq_(handed[0], 2);
    st_verify_eq_(handed[1], 3);
    st_verify_eq_(handed[2], 4);
    st_verify_eq_(vector.size(), 5u);
    st_verify_eq_(vector[1], 1);
    st_verify_eq_(vector[2], 5);
    st_verify_eq_(vector[4], 7);

    vector.erase(0, collect);
    st_verify_eq_(handed_count, 4u);
    st_verify_eq_(handed[3], 0);
    st_verify_eq_(vector[0], 1);

    // A count beside the position is a count, never a callback.
    vector.erase(1, 2);
    st_verify_eq_(handed_count, 4u);
    st_verify_eq_(vector.size(), 2u);
    st_verify_eq_(vector[0], 1);
    st_verify_eq_(vector[1], 7);
}

/** A predicate sweeps in one pass, keeps the survivors in order, and answers how many went. */
static void vector_erase_if_compacts_in_one_pass() {
    basic_vector<int> vector;
    for (int value = 0; value < 10; ++value) st_verify_(vector.push_back(int {value}));

    std::size_t handed_evens = 0;
    auto count_even = [&](int &element) noexcept { handed_evens += element % 2 == 0; };
    auto is_even = [](int const &element) noexcept { return element % 2 == 0; };

    st_verify_eq_(vector.erase_if(is_even, count_even), 5u);
    st_verify_eq_(handed_evens, 5u);
    st_verify_eq_(vector.size(), 5u);
    for (std::size_t index = 0; index < vector.size(); ++index)
        st_verify_eq_(vector[index], static_cast<int>(2 * index + 1));

    st_verify_eq_(vector.erase_if(is_even), 0u);
    st_verify_eq_(vector.size(), 5u);

    st_verify_eq_(vector.erase_if([](int const &) noexcept { return true; }), 5u);
    st_verify_eq_(vector.size(), 0u);
}

/** Moves transfer ownership and leave the source empty rather than aliasing its storage. */
static void vector_move_semantics() {
    auto made = basic_vector<int>::make(16);
    st_verify_(static_cast<bool>(made));
    basic_vector<int> vector = std::move(*made);
    // A first reserve grants exactly what it was asked for once that clears the four-slot seed.
    st_verify_eq_(vector.capacity(), 16u);
    for (int value = 0; value < 16; ++value) st_verify_(vector.push_back(assume_reserved, int {value}));

    basic_vector<int> moved = std::move(vector);
    st_verify_eq_(moved.size(), 16u);
    st_verify_eq_(vector.size(), 0u);
    st_verify_eq_(vector.capacity(), 0u);
    st_verify_eq_(vector.data(), nullptr);

    basic_vector<int> assigned;
    st_verify_(assigned.push_back(1));
    assigned = std::move(moved);
    st_verify_eq_(assigned.size(), 16u);
    st_verify_eq_(assigned.back(), 15);
    st_verify_eq_(assigned.front(), 0);
    st_verify_ne_(assigned.at(15), nullptr);
    st_verify_eq_(assigned.at(16), nullptr);
}

/** The suite every sequence answers, over a vector of counted elements and one of tracked
 *  allocations. */
void vector_sequence_suite() {
    test_sequence_tags<basic_vector<counted_key_t>>();
    test_sequence_order<basic_vector<counted_key_t>>();
    test_sequence_element_lifetimes<basic_vector<counted_key_t>>();
    test_sequence_move_semantics<basic_vector<counted_key_t>>();
    test_sequence_allocator_ledger<basic_vector<counted_key_t, stateful_allocator<counted_key_t>>>();
}

#pragma endregion Tests

} // namespace

int main(int, char **arguments) {
    test_environment_t const environment = read_test_environment(arguments[0]);
    install_test_signal_handlers();
    log_environment(environment);
    test_tally_t tally;

    tally += run_test(environment, "vector.emplace_back_through_make", vector_emplace_back_through_make);
    tally +=
        run_test(environment, "vector.reserve_refuses_wrapping_capacity", vector_reserve_refuses_wrapping_capacity);
    tally += run_test(environment, "vector.growth_is_amortized", vector_growth_is_amortized);
    tally += run_test(environment, "vector.resize_rolls_back_elements", vector_resize_rolls_back_elements);
    tally += run_test(environment, "vector.copy_and_swap", vector_copy_and_swap);
    tally += run_test(environment, "vector.move_semantics", vector_move_semantics);
    tally += run_test(environment, "vector.inserts_at_a_position", vector_inserts_at_a_position);
    tally += run_test(environment, "vector.erases_at_a_position", vector_erases_at_a_position);
    tally += run_test(environment, "vector.shifts_every_element_once", vector_shifts_every_element_once);
    tally += run_test(environment, "vector.inserts_into_reserved_room", vector_inserts_into_reserved_room);
    tally += run_test(environment, "vector.erases_hand_each_element_out", vector_erases_hand_each_element_out);
    tally += run_test(environment, "vector.erase_if_compacts_in_one_pass", vector_erase_if_compacts_in_one_pass);
    tally += run_test(environment, "vector.sequence_suite", vector_sequence_suite);

    return report_test_failures(environment, tally);
}
