/**
 *  @brief Tests for @c basic_vector - the fallible-construction path, growth arithmetic, and rollback.
 *  @author Ash Vardanian
 *  @file scripts/test_vector.cpp
 *  @date August 17, 2026
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define ST_STRICT_CALLBACK_CHECKS_ 1

#include <cstddef> // `std::size_t`
#include <cstdint> // `SIZE_MAX`

#include <new>         // `::operator new`, `std::nothrow`
#include <type_traits> // `std::true_type`
#include <utility>     // `std::move`

#include <smashtable/basic_vector.hpp>

#include "test.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::scripts;

#pragma region Element Types

/**
 *  @brief An element only constructible through @c make, which is what drives @c emplace_back's slow path.
 *    Its value constructor is deliberately throwing, so the nothrow branch cannot claim it.
 */
struct fallible_element_t {
    int value {0};

    fallible_element_t() noexcept = default;
    fallible_element_t(fallible_element_t &&) noexcept = default;
    fallible_element_t &operator=(fallible_element_t &&) noexcept = default;
    fallible_element_t(fallible_element_t const &) = default;
    fallible_element_t &operator=(fallible_element_t const &) = default;

    /** @brief Throwing on purpose, so @c std::is_nothrow_constructible_v rejects this path. */
    explicit fallible_element_t(int requested, int) : value(requested) {}

    /** @brief Refuses negative values, which is the failure the caller must see as a status. */
    [[nodiscard]] static expected<fallible_element_t> make(int requested) noexcept {
        if (requested < 0) return status_t::invalid_argument_k;
        fallible_element_t made;
        made.value = requested;
        return expected<fallible_element_t>(std::move(made), success_k);
    }

    /** @brief Deep copy that never fails, so @c copy_safely has a route for this type. */
    [[nodiscard]] expected<fallible_element_t> copy() const noexcept {
        fallible_element_t made;
        made.value = value;
        return expected<fallible_element_t>(std::move(made), success_k);
    }
};

/** @brief An element whose copy fails past a budget, exercising the rollback paths. */
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

/**
 *  @brief An allocator that refuses every request and remembers the largest one it was asked for.
 *    A hand-written allocator that multiplies without checking would answer a wrapping byte count
 *    with a small block, so the vector must never put such a count in front of one.
 */
template <typename value_type_>
struct recording_allocator {
    static inline std::size_t largest_request = 0;
    static inline std::size_t requests_count = 0;

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

/** @brief The @c make branch of @c emplace_back must compile at all, and must report its failures. */
static void vector_emplace_back_through_make() {
    static_assert(!std::is_nothrow_constructible_v<fallible_element_t, int>,
                  "the test element must not qualify for the nothrow branch");
    static_assert(has_make_method<fallible_element_t, int>, "the test element must qualify for the make branch");

    basic_vector<fallible_element_t> vector;
    st_verify_(succeeded(vector.emplace_back(7)));
    st_verify_(succeeded(vector.emplace_back(8)));
    st_verify_eq_(vector.size(), 2u);
    st_verify_eq_(vector[0].value, 7);
    st_verify_eq_(vector[1].value, 8);

    // A refused `make` propagates its reason and leaves the vector as it was.
    st_verify_eq_(vector.emplace_back(-1), status_t::invalid_argument_k);
    st_verify_eq_(vector.size(), 2u);

    st_verify_(succeeded(vector.reserve(8)));
    st_verify_(succeeded(vector.emplace_back(assume_reserved, 9)));
    st_verify_eq_(vector.size(), 3u);
    st_verify_eq_(vector[2].value, 9);
    st_verify_eq_(vector.emplace_back(assume_reserved, -5), status_t::invalid_argument_k);
    st_verify_eq_(vector.size(), 3u);

    // The nothrow branch still wins where it applies.
    basic_vector<int> integers;
    st_verify_(succeeded(integers.emplace_back(42)));
    st_verify_eq_(integers[0], 42);
}

/** @brief A capacity whose byte count cannot be addressed is refused rather than allocated small. */
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
    st_verify_(recording_allocator<long long>::largest_request <= max_capacity);

    // Ordinary growth is untouched by the guard, and still at least doubles.
    basic_vector<long long> heap_vector;
    st_verify_(succeeded(heap_vector.reserve(1)));
    st_verify_eq_(heap_vector.capacity(), 4u);
    st_verify_(succeeded(heap_vector.reserve(5)));
    st_verify_(heap_vector.capacity() >= 8u);
    st_verify_(succeeded(heap_vector.push_back(11)));
    st_verify_eq_(heap_vector[0], 11);
}

/** @brief Growth amortizes: appending one at a time must double rather than grow by one. */
static void vector_growth_is_amortized() {
    basic_vector<int> vector;
    for (int value = 0; value < 100; ++value) st_verify_(succeeded(vector.push_back(int {value})));
    st_verify_eq_(vector.size(), 100u);
    st_verify_(vector.capacity() >= 100u);
    st_verify_(vector.capacity() < 200u);
    for (int value = 0; value < 100; ++value) st_verify_eq_(vector[static_cast<std::size_t>(value)], value);

    vector.pop_back();
    st_verify_eq_(vector.size(), 99u);
    vector.clear();
    st_verify_(vector.empty());
    st_verify_(vector.capacity() >= 100u);
}

/** @brief A failed @c resize leaves the elements alone, whatever it did to the capacity. */
static void vector_resize_rolls_back_elements() {
    basic_vector<budgeted_element_t> vector;
    st_verify_(succeeded(vector.push_back(budgeted_element_t {})));
    vector[0].value = 3;

    budgeted_element_t pattern;
    pattern.value = 9;

    // Two copies granted, four demanded: the three constructed before the refusal must be destroyed.
    budgeted_element_t::copies_left = 2;
    st_verify_eq_(vector.resize(5, pattern), status_t::out_of_memory_heap_k);
    st_verify_eq_(vector.size(), 1u);
    st_verify_eq_(vector[0].value, 3);

    budgeted_element_t::copies_left = 16;
    st_verify_(succeeded(vector.resize(5, pattern)));
    st_verify_eq_(vector.size(), 5u);
    st_verify_eq_(vector[4].value, 9);

    st_verify_(succeeded(vector.resize(2)));
    st_verify_eq_(vector.size(), 2u);
}

/** @brief @c copy is all-or-nothing, and @c swap moves the storage across. */
static void vector_copy_and_swap() {
    basic_vector<fallible_element_t> vector;
    for (int value = 0; value < 5; ++value) st_verify_(succeeded(vector.emplace_back(value)));

    auto copied = vector.copy();
    st_verify_(static_cast<bool>(copied));
    st_verify_eq_(copied->size(), 5u);
    for (std::size_t index = 0; index < 5; ++index) st_verify_eq_((*copied)[index].value, vector[index].value);

    basic_vector<fallible_element_t> empty;
    st_verify_(succeeded(empty.swap(vector)));
    st_verify_eq_(empty.size(), 5u);
    st_verify_eq_(vector.size(), 0u);

    // A failing copy hands back nothing at all rather than a partial vector.
    basic_vector<budgeted_element_t> budgeted;
    for (int value = 0; value < 4; ++value) st_verify_(succeeded(budgeted.push_back(budgeted_element_t {})));
    budgeted_element_t::copies_left = 2;
    auto refused = budgeted.copy();
    st_verify_(!refused);
    st_verify_eq_(refused.status(), status_t::out_of_memory_heap_k);
    st_verify_eq_(budgeted.size(), 4u);
}

/** @brief Moves transfer ownership and leave the source empty rather than aliasing its storage. */
static void vector_move_semantics() {
    auto made = basic_vector<int>::make(16);
    st_verify_(static_cast<bool>(made));
    basic_vector<int> vector = std::move(*made);
    st_verify_(vector.capacity() >= 16u);
    for (int value = 0; value < 16; ++value) st_verify_(succeeded(vector.push_back(assume_reserved, int {value})));

    basic_vector<int> moved = std::move(vector);
    st_verify_eq_(moved.size(), 16u);
    st_verify_eq_(vector.size(), 0u);
    st_verify_eq_(vector.capacity(), 0u);
    st_verify_eq_(vector.data(), nullptr);

    basic_vector<int> assigned;
    st_verify_(succeeded(assigned.push_back(1)));
    assigned = std::move(moved);
    st_verify_eq_(assigned.size(), 16u);
    st_verify_eq_(assigned.back(), 15);
    st_verify_eq_(assigned.front(), 0);
    st_verify_ne_(assigned.at(15), nullptr);
    st_verify_eq_(assigned.at(16), nullptr);
}

#pragma endregion Tests

int main() {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "vector.emplace_back_through_make", vector_emplace_back_through_make);
    failures += run_test(filter, "vector.reserve_refuses_wrapping_capacity", vector_reserve_refuses_wrapping_capacity);
    failures += run_test(filter, "vector.growth_is_amortized", vector_growth_is_amortized);
    failures += run_test(filter, "vector.resize_rolls_back_elements", vector_resize_rolls_back_elements);
    failures += run_test(filter, "vector.copy_and_swap", vector_copy_and_swap);
    failures += run_test(filter, "vector.move_semantics", vector_move_semantics);

    return report_test_failures(failures);
}
