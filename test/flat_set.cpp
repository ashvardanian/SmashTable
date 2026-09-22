/**
 *  @file test/flat_set.cpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief Tests for @c basic_flat_set: random operations against a presence oracle, through every
 *      kit that runs here.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define ST_STRICT_CALLBACK_CHECKS_ 1

#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <random>      // `std::mt19937_64`
#include <span>        // `std::span`
#include <type_traits> // `std::true_type`

#include <smashtable/basic_flat_set.hpp>

#include "harness.hpp"
#include "fixtures.hpp"
#include "surfaces.hpp"

using namespace ashvardanian::smashtable;
using namespace ashvardanian::smashtable::test;

namespace {

#pragma region Element Types

/** A read request ordered by descriptor and then offset, the shape of a comparator the kits never see. */
struct request_t {
    int descriptor {0};
    std::uint64_t offset {0};
};

struct request_order_t {
    constexpr bool operator()(request_t const &left, request_t const &right) const noexcept {
        return left.descriptor != right.descriptor ? left.descriptor < right.descriptor : left.offset < right.offset;
    }
};

/** How many distinct elements a suite draws from, small enough that a presence table is the oracle. */
constexpr std::size_t domain_size_k = 512;

/** Maps a domain index to an element so that the map preserves order and crosses each type's sign boundary. */
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
void verify_against_presence(set_type_ const &set, std::span<std::uint8_t const, domain_size_k> present) {
    using value_t = typename set_type_::value_t;
    std::size_t below = 0;
    std::size_t listed = 0;
    for (std::size_t index = 0; index < domain_size_k; ++index) {
        value_t const element = element_at<value_t>(index);
        st_verify_eq_(set.rank(element), below);
        st_verify_eq_(set.contains(element), present[index]);
        if (present[index]) {
            st_verify_(same_element(set[listed], element));
            ++listed;
        }
        below += present[index];
    }
    st_verify_eq_(set.size(), below);
}

template <typename kit_type_, typename value_type_, typename comparator_type_>
void verify_random_operations(std::mt19937_64 &generator) {
    using set_t = basic_flat_set<value_type_, comparator_type_, kit_type_>;
    std::uint8_t present[domain_size_k] = {};
    set_t set;
    for (std::size_t step = 0; step < 6000; ++step) {
        std::size_t const index = generator() % domain_size_k;
        value_type_ const element = element_at<value_type_>(index);
        switch (generator() % 4) {
        case 0:
            st_verify_eq_(set.insert(element), present[index] ? status_t::key_already_exists_k : status_t::success_k);
            present[index] = true;
            break;
        case 1:
            st_verify_(set.upsert(element));
            present[index] = true;
            break;
        case 2:
            st_verify_eq_(set.erase(element), present[index] ? status_t::success_k : status_t::key_not_found_k);
            present[index] = false;
            break;
        default:
            value_type_ const *const found = set.find(element);
            st_verify_eq_(found != nullptr, present[index]);
            if (found) st_verify_(same_element(*found, element));
            break;
        }
        if (step % 1000 == 999) verify_against_presence(set, std::span<std::uint8_t const, domain_size_k>(present));
    }

    // A batch adds only what is missing and reserves once for all of it.
    value_type_ incoming[64];
    for (value_type_ &element : incoming) {
        std::size_t const index = generator() % domain_size_k;
        element = element_at<value_type_>(index);
        present[index] = true;
    }
    st_verify_(set.insert_if_missing(incoming, incoming + 64));
    verify_against_presence(set, std::span<std::uint8_t const, domain_size_k>(present));

    auto copied = set.copy();
    st_verify_(copied);
    verify_against_presence(*copied, std::span<std::uint8_t const, domain_size_k>(present));
    set.clear();
    st_verify_(set.empty());
    st_verify_eq_(copied->size() > 0, true);
}

template <typename kit_type_>
void verify_kit(kit_type_) {
    std::mt19937_64 generator(test_seed_for(name_of(kit_type_::kit_k)));
    verify_random_operations<kit_type_, std::uint32_t, less_t>(generator);
    verify_random_operations<kit_type_, std::uint64_t, less_t>(generator);
    verify_random_operations<kit_type_, std::int64_t, less_t>(generator);
    verify_random_operations<kit_type_, key128_t, less_t>(generator);
    verify_random_operations<kit_type_, request_t, request_order_t>(generator);
}

void verify_through(row_kit_t kit) {
    if (!row_kit_supported(kit)) {
        print_line(stdout, "  {} kit: {}", name_of(kit),
                   row_kit_compiled(kit) ? "compiled, but not runnable here" : "not compiled");
        return;
    }
    print_line(stdout, "  {} kit: running", name_of(kit));
    visit_row_kit(kit, [](row_kit auto kit_instance) noexcept { verify_kit(kit_instance); });
}

#pragma endregion Suites

#pragma region Tests

void flat_set_serial_kit() { verify_through(row_kit_t::serial_k); }
void flat_set_haswell_kit() { verify_through(row_kit_t::haswell_k); }
void flat_set_skylake_kit() { verify_through(row_kit_t::skylake_k); }
void flat_set_neon_kit() { verify_through(row_kit_t::neon_k); }
void flat_set_sve_kit() { verify_through(row_kit_t::sve_k); }
void flat_set_rvv_kit() { verify_through(row_kit_t::rvv_k); }

/** A refused allocation reports itself and leaves the set as it was. */
void flat_set_refused_allocation() {
    using refused_set_t = basic_flat_set<std::uint64_t, less_t, serial_row_kit_t, refusing_allocator<std::uint64_t>>;
    st_verify_eq_(refused_set_t::make(16).status(), status_t::out_of_memory_heap_k);
    refused_set_t set;
    st_verify_eq_(set.insert(7), status_t::out_of_memory_heap_k);
    st_verify_eq_(set.size(), 0u);
    std::uint64_t const incoming[] = {1, 2, 3};
    st_verify_eq_(set.insert_if_missing(incoming, incoming + 3), status_t::out_of_memory_heap_k);
    st_verify_(set.empty());

    auto made = basic_flat_set<std::uint64_t>::make(32);
    st_verify_(made);
    st_verify_ge_(made->capacity(), 32u);
    basic_flat_set<std::uint64_t> moved = std::move(*made);
    st_verify_(moved.insert(3));
    st_verify_(moved.insert(1));
    st_verify_eq_(moved[0], 1u);
    st_verify_eq_(moved.rank(std::uint64_t {2}), 1u);
}

/** The suites every associative container answers, over the element shapes a flat set can hold. */
void flat_set_shared_suites() {
    using trivial_set_t =
        basic_flat_set<trivial_key_t, std::less<trivial_key_t>, serial_row_kit_t, std::allocator<trivial_key_t>>;
    using tracking_set_t = basic_flat_set<trivial_key_t, stateful_comparator_t, serial_row_kit_t, stateful_allocator_t>;
    using composite_set_t =
        basic_flat_set<composite_key_t, composite_key_compare_t, serial_row_kit_t, std::allocator<composite_key_t>>;
    using heavy_set_t = basic_flat_set<heavy_key_t, std::less<void>, serial_row_kit_t, std::allocator<heavy_key_t>>;
    using native_set_t = basic_flat_set<std::uint64_t>;

    test_empty_container_operations<trivial_set_t>();
    test_empty_container_operations<tracking_set_t>();
    test_empty_container_operations<native_set_t>();
    test_single_element_operations<trivial_set_t>();
    test_single_element_operations<heavy_set_t>();
    test_basic_insertion_patterns<trivial_set_t>();
    test_basic_insertion_patterns<native_set_t>();
    test_bulk_insertion_from_iterators<trivial_set_t>();
    test_bulk_insertion_from_iterators<heavy_set_t>();
    test_range_query_head_state<trivial_set_t>();
    test_range_query_head_state<native_set_t>();
    test_erase_range_head_state<trivial_set_t>();
    test_erase_range_head_state<native_set_t>();
    test_heterogeneous_composite_find<composite_set_t>();
    test_heterogeneous_heavy_string_view_find<heavy_set_t>();
}

/** Every range modifier refused at every point it asks for memory leaves the set as it was. */
void flat_set_batch_is_all_or_nothing() {
    using ledger_set_t = basic_flat_set<trivial_key_t, less_t, serial_row_kit_t, stateful_allocator_t>;
    test_every_offered_surface<ledger_set_t>(
        [](allocation_ledger_t &ledger) noexcept { return ledger_set_t(less_t {}, stateful_allocator_t(1, ledger)); });
}

/** A predicate sweeps the array in one pass, keeps the survivors ordered, and hands each removed element out. */
void flat_set_erase_if_keeps_order() {
    basic_flat_set<std::uint64_t> set;
    for (std::uint64_t value = 0; value < 16; ++value) st_verify_(set.insert(std::uint64_t {value}));

    std::size_t handed = 0;
    std::size_t const removed = set.erase_if([](std::uint64_t const &element) noexcept { return element % 4 == 0; },
                                             [&](std::uint64_t &) noexcept { ++handed; });
    st_verify_eq_(removed, 4u);
    st_verify_eq_(handed, 4u);
    st_verify_eq_(set.size(), 12u);
    std::uint64_t previous = 0;
    for (std::uint64_t const &element : set) {
        st_verify_(element % 4 != 0);
        st_verify_(previous < element || previous == 0);
        previous = element;
    }
    st_verify_eq_(set.erase_if([](std::uint64_t const &) noexcept { return false; }), 0u);
    st_verify_eq_(set.size(), 12u);
}

#pragma endregion Tests

} // namespace

int main() {
    install_test_signal_handlers();
    char const *const filter = test_filter();
    std::size_t failures = 0;

    failures += run_test(filter, "flat_set.serial_kit", flat_set_serial_kit);
    failures += run_test(filter, "flat_set.haswell_kit", flat_set_haswell_kit);
    failures += run_test(filter, "flat_set.skylake_kit", flat_set_skylake_kit);
    failures += run_test(filter, "flat_set.neon_kit", flat_set_neon_kit);
    failures += run_test(filter, "flat_set.sve_kit", flat_set_sve_kit);
    failures += run_test(filter, "flat_set.rvv_kit", flat_set_rvv_kit);
    failures += run_test(filter, "flat_set.refused_allocation", flat_set_refused_allocation);
    failures += run_test(filter, "flat_set.shared_suites", flat_set_shared_suites);

    failures += run_test(filter, "flat_set.batch_is_all_or_nothing", flat_set_batch_is_all_or_nothing);
    failures += run_test(filter, "flat_set.erase_if_keeps_order", flat_set_erase_if_keeps_order);

    return report_test_failures(failures);
}
