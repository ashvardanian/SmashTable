/**
 *  @file test/hash_index.cpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief Tests for @c basic_hash_index: capacity rules, a random mix against a scalar oracle
 *      through every row kit, colliding homes, confirmed lookups and erasure past the table's end.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build
#define SMASHTABLE_STRICT_CALLBACK_CHECKS 1

#include <cstddef> // `std::byte`, `std::size_t`
#include <cstdint> // `std::int32_t`, `std::uint32_t`, `std::uint64_t`

#include <concepts> // `std::same_as`
#include <limits>   // `std::numeric_limits`
#include <random>   // `std::mt19937_64`
#include <utility>  // `std::move`
#include <vector>   // `std::vector`

#include "smashtable/basic_hash_index.hpp"

#include "fixtures.hpp"
#include "harness.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Fixtures

/** Sends every key home to slot zero, so a whole index is one run. */
struct one_home_hash_t {
    template <typename key_type_>
    std::size_t operator()(key_type_ const &) const noexcept {
        return 0;
    }
};

/** Sends a key home to its own value, so a test places runs exactly. */
struct identity_hash_t {
    std::size_t operator()(std::uint64_t key) const noexcept { return static_cast<std::size_t>(key); }
};

/** Keeps only the high bits of a key, so sixteen neighbours share each home. */
struct coarse_hash_t {
    std::size_t operator()(std::uint64_t key) const noexcept { return static_cast<std::size_t>(key >> 4); }
};

/** Draws one of @p range keys, few enough that several positions share one. */
template <typename key_type_>
key_type_ draw_key(std::mt19937_64 &generator, std::uint64_t range) noexcept {
    std::uint64_t const drawn = generator() % range;
    if constexpr (std::same_as<key_type_, key128_t>) return key128_t {drawn * 0x9E3779B97F4A7C15ull, drawn};
    else return static_cast<key_type_>(drawn);
}

#pragma endregion Fixtures

#pragma region Suites

/** Inserts, erases and finds at random against an array of the key each position holds. */
template <typename index_type_>
void verify_random_mix(std::mt19937_64 &generator) {
    using key_t = typename index_type_::key_t;
    constexpr std::size_t capacity_k = 200;
    constexpr std::uint64_t keys_range_k = 64;
    auto index = index_type_::make(capacity_k);
    st_verify_(index);

    std::vector<key_t> keys(capacity_k);
    std::vector<unsigned char> present(capacity_k, 0);
    std::size_t present_count = 0;
    for (std::size_t step = 0; step != 20000; ++step) {
        std::uint32_t const position = static_cast<std::uint32_t>(generator() % capacity_k);
        key_t const key = draw_key<key_t>(generator, keys_range_k);
        switch (generator() % 3) {
        case 0:
            if (present[position]) break;
            st_verify_(index->insert(key, position, assume_unique));
            keys[position] = key, present[position] = 1, ++present_count;
            break;
        case 1:
            if (present[position] && !(keys[position] == key))
                st_verify_eq_(index->erase(key, position), status_t::key_not_found_k, "the pair must match");
            if (!present[position]) {
                st_verify_eq_(index->erase(key, position), status_t::key_not_found_k);
                break;
            }
            st_verify_(index->erase(keys[position], position));
            present[position] = 0, --present_count;
            break;
        default: {
            std::uint32_t const found = index->find(key);
            if (found == index_type_::absent_k)
                for (std::size_t other = 0; other != capacity_k; ++other)
                    st_verify_((!present[other] || !(keys[other] == key)) && "a stored key must be found");
            else st_verify_((present[found] && keys[found] == key) && "a found position must hold the key");
            std::uint32_t const confirmed =
                index->find(key, [&](std::uint32_t candidate) noexcept { return candidate == position; });
            st_verify_eq_(confirmed == position, present[position] && keys[position] == key);
            break;
        }
        }
        st_verify_eq_(index->size(), present_count);
    }
}

/** Keys at the edges of their width are stored and found like any other. */
template <typename key_type_, typename kit_type_>
void verify_extreme_keys() {
    using index_t = basic_hash_index<key_type_, std::uint32_t, default_hash_t, default_allocator_t, kit_type_>;
    for (std::uint64_t const word : {0ull, 0x7FFFFFFFull, 0x80000000ull, 0xFFFFFFFFull, 1ull << 63, ~0ull}) {
        key_type_ key;
        if constexpr (std::same_as<key_type_, key128_t>) key = key128_t {word, ~word};
        else key = static_cast<key_type_>(word);
        auto index = index_t::make(4);
        st_verify_(index);
        st_verify_(index->insert(key, 1u, assume_unique));
        st_verify_eq_(index->find(key), 1u);
    }
}

/** The random mix over every key width and a hash crowding homes, through one kit. */
template <typename kit_type_>
void verify_kit(test_context_t const &context, kit_type_) {
    std::mt19937_64 generator(mix_seed(context.seed, __func__));
    verify_random_mix<basic_hash_index<std::uint32_t, std::uint32_t, default_hash_t, default_allocator_t, kit_type_>>(
        generator);
    verify_random_mix<basic_hash_index<std::int32_t, std::uint32_t, default_hash_t, default_allocator_t, kit_type_>>(
        generator);
    verify_random_mix<basic_hash_index<std::uint64_t, std::uint32_t, default_hash_t, default_allocator_t, kit_type_>>(
        generator);
    verify_random_mix<basic_hash_index<key128_t, std::uint32_t, default_hash_t, default_allocator_t, kit_type_>>(
        generator);
    verify_random_mix<basic_hash_index<std::uint64_t, std::uint32_t, coarse_hash_t, default_allocator_t, kit_type_>>(
        generator);
    verify_extreme_keys<std::int32_t, kit_type_>();
    verify_extreme_keys<std::uint64_t, kit_type_>();
    verify_extreme_keys<key128_t, kit_type_>();
}

static void verify_through(test_context_t const &context, row_kit_t kit) {
    if (row_kit_supported(kit))
        visit_row_kit(kit, [&](row_kit auto kit_instance) noexcept { verify_kit(context, kit_instance); });
}

#pragma endregion Suites

#pragma region Tests

/** Capacity is fixed at @c make, zero accepts nothing, and refusals are reported. */
static void hash_index_capacity_rules() {
    using index_t = basic_hash_index<std::uint64_t>;
    st_verify_eq_(index_t::make(std::numeric_limits<std::size_t>::max()).status(), status_t::invalid_argument_k);
    using refused_index_t =
        basic_hash_index<std::uint64_t, std::uint32_t, default_hash_t, refusing_allocator<std::byte>>;
    st_verify_eq_(refused_index_t::make(16).status(), status_t::out_of_memory_heap_k);

    auto empty = index_t::make(0);
    st_verify_(empty);
    st_verify_eq_(empty->insert(1u, 0u, assume_unique), status_t::capacity_exhausted_k);
    st_verify_eq_(empty->find(1u), index_t::absent_k);
    st_verify_eq_(empty->erase(1u, 0u), status_t::key_not_found_k);

    auto index = index_t::make(4);
    st_verify_(index);
    for (std::uint32_t position = 0; position != 4; ++position)
        st_verify_(index->insert(position * 7u, position, assume_unique));
    st_verify_eq_(index->insert(99u, 4u, assume_unique), status_t::capacity_exhausted_k);
    st_verify_eq_(index->erase(7u, 2u), status_t::key_not_found_k, "a key under another position stays");
    st_verify_(index->erase(7u, 1u));
    st_verify_(index->insert(99u, 4u, assume_unique));
    st_verify_eq_(index->find(99u), 4u);
    st_verify_eq_(index->size(), 4u);

    index->clear();
    st_verify_(index->empty());
    for (std::uint32_t position = 0; position != 4; ++position)
        st_verify_eq_(index->find(position * 7u), index_t::absent_k, "a cleared slot must not match its old key");
    st_verify_(index->insert(14u, 2u, assume_unique));
    st_verify_eq_(index->find(14u), 2u);

    auto moved = std::move(*index);
    st_verify_eq_(moved.find(14u), 2u);
    st_verify_(index->empty());
}

/** Every key sharing one home stays reachable while the run is erased from its middle. */
static void hash_index_colliding_homes() {
    auto index = basic_hash_index<std::uint64_t, std::uint32_t, one_home_hash_t>::make(40);
    st_verify_(index);
    for (std::uint32_t position = 0; position != 40; ++position)
        st_verify_(index->insert(std::uint64_t {position} * 3, position, assume_unique));
    for (std::uint32_t position = 0; position < 40; position += 3)
        st_verify_(index->erase(std::uint64_t {position} * 3, position));
    for (std::uint32_t position = 0; position != 40; ++position)
        st_verify_eq_(index->find(std::uint64_t {position} * 3),
                      position % 3 == 0 ? basic_hash_index<std::uint64_t>::absent_k : position);
}

/** A confirm tells apart positions under one key, and an erase removes only the one it names. */
static void hash_index_confirm_separates_equal_keys() {
    auto index = basic_hash_index<std::uint64_t>::make(8);
    st_verify_(index);
    for (std::uint32_t position = 1; position <= 3; ++position) st_verify_(index->insert(7u, position, assume_unique));
    st_verify_eq_(index->find(7u), 1u, "the first position in probe order");
    st_verify_eq_(index->find(7u, [](std::uint32_t candidate) noexcept { return candidate == 3; }), 3u);
    st_verify_eq_(index->find(7u, [](std::uint32_t candidate) noexcept { return candidate == 9; }),
                  basic_hash_index<std::uint64_t>::absent_k);
    st_verify_(index->erase(7u, 1u));
    st_verify_eq_(index->find(7u), 2u);
    st_verify_eq_(index->find(7u, [](std::uint32_t candidate) noexcept { return candidate == 3; }), 3u);
}

/** A run wrapping past the last slot shifts back across the end, and every key stays reachable. */
static void hash_index_erase_wraps_table_end() {
    using index_t = basic_hash_index<std::uint64_t, std::uint32_t, identity_hash_t>;
    // Sixteen keys take 32 slots, so homes 30 and 31 run into slots 0 to 2.
    auto index = index_t::make(16);
    st_verify_(index);
    std::uint64_t const keys[] = {30, 62, 94, 31, 0};
    for (std::uint32_t position = 0; position != 5; ++position)
        st_verify_(index->insert(keys[position], position, assume_unique));
    st_verify_(index->erase(30u, 0u));
    for (std::uint32_t position = 1; position != 5; ++position) st_verify_eq_(index->find(keys[position]), position);
    st_verify_(index->erase(94u, 2u));
    st_verify_eq_(index->find(62u), 1u);
    st_verify_eq_(index->find(31u), 3u);
    st_verify_eq_(index->find(0u), 4u);
    st_verify_(index->erase(62u, 1u));
    st_verify_(index->erase(31u, 3u));
    st_verify_eq_(index->find(0u), 4u);

    // Homes 14, 46 and 78 share slot 14 modulo 32, one run crosses the 15 to 16 boundary.
    st_verify_(index->insert(14u, 5u, assume_unique));
    st_verify_(index->insert(46u, 6u, assume_unique));
    st_verify_(index->insert(78u, 7u, assume_unique));
    st_verify_(index->erase(14u, 5u));
    st_verify_eq_(index->find(78u), 7u);
    st_verify_eq_(index->size(), 3u);
}

static void hash_index_serial_kit(test_context_t const &context) { verify_through(context, row_kit_t::serial_k); }
static void hash_index_haswell_kit(test_context_t const &context) { verify_through(context, row_kit_t::haswell_k); }
static void hash_index_skylake_kit(test_context_t const &context) { verify_through(context, row_kit_t::skylake_k); }
static void hash_index_icelake_kit(test_context_t const &context) { verify_through(context, row_kit_t::icelake_k); }
static void hash_index_neon_kit(test_context_t const &context) { verify_through(context, row_kit_t::neon_k); }
static void hash_index_sve_kit(test_context_t const &context) { verify_through(context, row_kit_t::sve_k); }
static void hash_index_rvv_kit(test_context_t const &context) { verify_through(context, row_kit_t::rvv_k); }
static void hash_index_v128_kit(test_context_t const &context) { verify_through(context, row_kit_t::v128_k); }

#pragma endregion Tests

} // namespace ashvardanian::smashtable::test

using namespace ashvardanian::smashtable::test;

int main(int, char **arguments) {
    settings_t const settings = read_settings(arguments[0]);
    install_test_signal_handlers();
    print(probe_machine());
    print(settings);
    test_tally_t tally;

    tally += run_test(settings, "hash_index.capacity_rules", hash_index_capacity_rules);
    tally += run_test(settings, "hash_index.colliding_homes", hash_index_colliding_homes);
    tally += run_test(settings, "hash_index.confirm_separates_equal_keys", hash_index_confirm_separates_equal_keys);
    tally += run_test(settings, "hash_index.erase_wraps_table_end", hash_index_erase_wraps_table_end);
    tally += run_test(settings, "hash_index.serial_kit", hash_index_serial_kit);
    tally += run_test(settings, "hash_index.haswell_kit", hash_index_haswell_kit);
    tally += run_test(settings, "hash_index.skylake_kit", hash_index_skylake_kit);
    tally += run_test(settings, "hash_index.icelake_kit", hash_index_icelake_kit);
    tally += run_test(settings, "hash_index.neon_kit", hash_index_neon_kit);
    tally += run_test(settings, "hash_index.sve_kit", hash_index_sve_kit);
    tally += run_test(settings, "hash_index.rvv_kit", hash_index_rvv_kit);
    tally += run_test(settings, "hash_index.v128_kit", hash_index_v128_kit);

    return report_test_failures(settings, tally);
}
