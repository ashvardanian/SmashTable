/**
 *  @file test/flat_set.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Tests for @c basic_flat_set: random operations against a presence oracle, through every
 *      kit that runs here.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <span> // `std::span`

#include <smashtable/basic_flat_set.hpp>

#include "harness.hpp"
#include "fixtures.hpp"
#include "surfaces.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Element Types

/** A read request ordered by descriptor then offset, a comparator shape the kits never see. */
struct request_t {
    int descriptor {0};
    std::uint64_t offset {0};
};

struct request_order_t {
    constexpr bool operator()(request_t const &left, request_t const &right) const noexcept {
        return left.descriptor != right.descriptor ? left.descriptor < right.descriptor : left.offset < right.offset;
    }
};

/** How many distinct elements a suite draws from, small enough for a presence-table oracle. */
constexpr std::size_t domain_size_k = 512;

/** Maps a domain index to an element, preserving order and crossing each type's sign boundary. */
template <typename value_type_>
[[nodiscard]] constexpr value_type_ element_at(std::size_t index) noexcept {
    std::uint64_t const position = static_cast<std::uint64_t>(index);
    if constexpr (std::same_as<value_type_, std::uint32_t>) return static_cast<std::uint32_t>(position * 8388608u);
    else if constexpr (std::same_as<value_type_, std::uint64_t>) return position << 54;
    else if constexpr (std::same_as<value_type_, std::int64_t>)
        return (static_cast<std::int64_t>(position) - 256) * (std::int64_t {1} << 53);
    else if constexpr (std::same_as<value_type_, key128_t>) return key128_t {position >> 4, (position & 15) << 60};
    else return request_t {static_cast<int>(index / 32), position * 4096};
}

template <typename value_type_>
[[nodiscard]] constexpr bool same_element(value_type_ const &left, value_type_ const &right) noexcept {
    if constexpr (std::same_as<value_type_, request_t>)
        return left.descriptor == right.descriptor && left.offset == right.offset;
    else return left == right;
}

#pragma endregion Element Types

#pragma region Suites

template <typename set_type_>
void check_against_presence(set_type_ const &set, std::span<std::uint8_t const, domain_size_k> present) {
    using value_t = typename set_type_::value_t;
    std::size_t below = 0;
    std::size_t listed = 0;
    for (std::size_t index = 0; index < domain_size_k; ++index) {
        value_t const element = element_at<value_t>(index);
        expect_equal(set.rank(element), below);
        expect_equal(set.contains(element), present[index] != 0);
        if (present[index]) {
            expect_true(same_element(set[listed], element));
            ++listed;
        }
        below += present[index];
    }
    expect_equal(set.size(), below);
}

template <typename kit_type_, typename value_type_, typename comparator_type_>
void check_random_operations(split_mix_generator_t &generator) {
    using set_t = basic_flat_set<value_type_, comparator_type_, kit_type_>;
    std::uint8_t present[domain_size_k] = {};
    set_t set;
    for (std::size_t step = 0; step < 6000; ++step) {
        std::size_t const index = generator() % domain_size_k;
        value_type_ const element = element_at<value_type_>(index);
        switch (generator() % 4) {
        case 0:
            expect_equal(set.insert(element), present[index] ? status_t::key_already_exists_k : status_t::success_k);
            present[index] = true;
            break;
        case 1:
            expect_equal(set.upsert(element), success_k);
            present[index] = true;
            break;
        case 2:
            expect_equal(set.erase(element), present[index] ? status_t::success_k : status_t::key_not_found_k);
            present[index] = false;
            break;
        default:
            value_type_ const *const found = set.find(element);
            expect_equal(found != nullptr, present[index] != 0);
            if (found) expect_true(same_element(*found, element));
            break;
        }
        if (step % 1000 == 999) check_against_presence(set, std::span<std::uint8_t const, domain_size_k>(present));
    }

    // A batch adds only what is missing and reserves once for all of it.
    value_type_ incoming[64];
    for (value_type_ &element : incoming) {
        std::size_t const index = generator() % domain_size_k;
        element = element_at<value_type_>(index);
        present[index] = true;
    }
    expect_equal(set.insert_if_missing(incoming, incoming + 64), success_k);
    check_against_presence(set, std::span<std::uint8_t const, domain_size_k>(present));

    auto [copied, status] = set.copy();
    if (!expect_equal(status, success_k)) return;
    check_against_presence(copied, std::span<std::uint8_t const, domain_size_k>(present));
    set.clear();
    expect_true(set.empty());
    expect_equal(copied.size() > 0, true);
}

template <typename kit_type_>
void check_kit(kit_type_) {
    split_mix_generator_t generator {case_stream_key()};
    check_random_operations<kit_type_, std::uint32_t, less_t>(generator);
    check_random_operations<kit_type_, std::uint64_t, less_t>(generator);
    check_random_operations<kit_type_, std::int64_t, less_t>(generator);
    check_random_operations<kit_type_, key128_t, less_t>(generator);
    check_random_operations<kit_type_, request_t, request_order_t>(generator);
}

#pragma endregion Suites

#pragma region Tests

/** @c check_kit through @p kit , skipped where this machine cannot run it. */
case_result_t test_kit(row_kit_t kit) {
    if (!row_kit_supported(kit)) return case_result_t::skipped(needs_kit_reason(kit));
    visit_row_kit(kit, [](row_kit auto kit_instance) noexcept { check_kit(kit_instance); });
    return {};
}

/** A refused allocation reports itself and leaves the set as it was. */
void test_refused_allocation() {
    using refused_set_t = basic_flat_set<std::uint64_t, less_t, serial_row_kit_t, refusing_allocator<std::uint64_t>>;
    expect_equal(refused_set_t::make(16).status(), status_t::out_of_memory_heap_k);
    refused_set_t set;
    expect_equal(set.insert(7), status_t::out_of_memory_heap_k);
    expect_equal(set.size(), 0u);
    std::uint64_t const incoming[] = {1, 2, 3};
    expect_equal(set.insert_if_missing(incoming, incoming + 3), status_t::out_of_memory_heap_k);
    expect_true(set.empty());

    auto [built, status] = basic_flat_set<std::uint64_t>::make(32);
    if (!expect_equal(status, success_k)) return;
    expect_true(built.capacity() >= 32u);
    basic_flat_set<std::uint64_t> moved = std::move(built);
    expect_equal(moved.insert(3), success_k);
    expect_equal(moved.insert(1), success_k);
    expect_equal(moved[0], 1u);
    expect_equal(moved.rank(std::uint64_t {2}), 1u);
}

/** The suites every associative container answers, over the element shapes a flat set can hold. */
void test_shared_suites() {
    using trivial_set_t =
        basic_flat_set<trivial_key_t, std::less<trivial_key_t>, serial_row_kit_t, std::allocator<trivial_key_t>>;
    using tracking_set_t = basic_flat_set<trivial_key_t, stateful_comparator_t, serial_row_kit_t, stateful_allocator_t>;
    using composite_set_t =
        basic_flat_set<composite_key_t, composite_key_compare_t, serial_row_kit_t, std::allocator<composite_key_t>>;
    using heavy_set_t = basic_flat_set<heavy_key_t, std::less<void>, serial_row_kit_t, std::allocator<heavy_key_t>>;
    using native_set_t = basic_flat_set<std::uint64_t>;

    check_empty_container_operations<trivial_set_t>();
    check_empty_container_operations<tracking_set_t>();
    check_empty_container_operations<native_set_t>();
    check_single_element_operations<trivial_set_t>();
    check_single_element_operations<heavy_set_t>();
    check_basic_insertion_patterns<trivial_set_t>();
    check_basic_insertion_patterns<native_set_t>();
    check_bulk_insertion_from_iterators<trivial_set_t>();
    check_bulk_insertion_from_iterators<heavy_set_t>();
    check_range_query_head_state<trivial_set_t>();
    check_range_query_head_state<native_set_t>();
    check_erase_range_head_state<trivial_set_t>();
    check_erase_range_head_state<native_set_t>();
    check_heterogeneous_composite_find<composite_set_t>();
    check_heterogeneous_heavy_string_view_find<heavy_set_t>();
}

/** Every range modifier refused at every point it asks for memory or copies an element leaves the
 *  set as it was. */
void test_batch_is_all_or_nothing() {
    using ledger_set_t = basic_flat_set<trivial_key_t, less_t, serial_row_kit_t, stateful_allocator_t>;
    check_every_offered_surface<ledger_set_t>(
        [](allocation_ledger_t &ledger) noexcept { return ledger_set_t(less_t {}, stateful_allocator_t(1, ledger)); });
    check_batch_refuses_copies_whole<
        basic_flat_set<budgeted_key_t, std::less<void>, serial_row_kit_t, std::allocator<budgeted_key_t>>>();
}

/** One predicate sweep keeps the survivors ordered and yields each removed element in turn. */
void test_erase_if_keeps_order() {
    basic_flat_set<std::uint64_t> set;
    for (std::uint64_t value = 0; value < 16; ++value) expect_equal(set.insert(std::uint64_t {value}), success_k);

    std::size_t handed = 0;
    std::size_t const removed = set.erase_if([](std::uint64_t const &element) noexcept { return element % 4 == 0; },
                                             [&](std::uint64_t &) noexcept { ++handed; });
    expect_equal(removed, 4u);
    expect_equal(handed, 4u);
    expect_equal(set.size(), 12u);
    std::uint64_t previous = 0;
    for (std::uint64_t const &element : set) {
        expect_true(element % 4 != 0);
        expect_true(previous < element || previous == 0);
        previous = element;
    }
    expect_equal(set.erase_if([](std::uint64_t const &) noexcept { return false; }), 0u);
    expect_equal(set.size(), 12u);
}

#pragma endregion Tests

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::test;

int main(int argc, char **argv) {
    start_test_run(argc, argv);

    run_test_case("test_kit(row_kit_t::serial_k)", test_kit, row_kit_t::serial_k);
    run_test_case("test_kit(row_kit_t::haswell_k)", test_kit, row_kit_t::haswell_k);
    run_test_case("test_kit(row_kit_t::skylake_k)", test_kit, row_kit_t::skylake_k);
    run_test_case("test_kit(row_kit_t::icelake_k)", test_kit, row_kit_t::icelake_k);
    run_test_case("test_kit(row_kit_t::neon_k)", test_kit, row_kit_t::neon_k);
    run_test_case("test_kit(row_kit_t::sve_k)", test_kit, row_kit_t::sve_k);
    run_test_case("test_kit(row_kit_t::rvv_k)", test_kit, row_kit_t::rvv_k);
    run_test_case("test_kit(row_kit_t::powervsx_k)", test_kit, row_kit_t::powervsx_k);
    run_test_case("test_kit(row_kit_t::v128_k)", test_kit, row_kit_t::v128_k);
    run_test_case("test_refused_allocation", test_refused_allocation);
    run_test_case("test_shared_suites", test_shared_suites);

    run_test_case("test_batch_is_all_or_nothing", test_batch_is_all_or_nothing);
    run_test_case("test_erase_if_keeps_order", test_erase_if_keeps_order);

    return finish_test_run();
}
