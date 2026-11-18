/**
 *  @file test/sequence.hpp
 *  @author Ash Vardanian
 *  @date September 17, 2026
 *  @brief The suite every sequence container answers: its type tags, the order it keeps, the
 *      lifetimes it balances, and what a move leaves behind.
 *
 *  A sequence keeps what it was handed in the order it was handed and indexes it from the oldest,
 *  which is all these suites assume. Which end a removal takes, and whether a full one grows or
 *  refuses, belongs to the container's own file.
 */
#pragma once
#include <cstddef> // `std::size_t`

#include <utility> // `std::move`

#include "harness.hpp"
#include "fixtures.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Sequence Test Templates

/** Appends one element, whichever name the sequence gives that. */
template <typename container_type_>
[[nodiscard]] status_t append_to(container_type_ &container, typename container_type_::value_type &&element) noexcept {
    if constexpr (requires { container.push_back(std::move(element)); }) return container.push_back(std::move(element));
    else return container.push(std::move(element));
}

/** Tests that a sequence names its element and its allocator the way the library does. */
template <typename container_type_>
void check_sequence_tags() {

    using container_t = container_type_;
    static_assert(std::is_same_v<typename container_t::value_type, typename container_t::value_t>,
                  "the STL tag and the library alias name one type");
    static_assert(std::is_same_v<typename container_t::allocator_type, typename container_t::allocator_t>,
                  "the STL tag and the library alias name one allocator");
    static_assert(
        std::allocator_traits<typename container_t::allocator_type>::propagate_on_container_move_assignment::value,
        "a move assignment that cannot carry the allocator would have to allocate, and may not");
}

/** Tests that elements read back in the order they arrived, and that clearing keeps the slots. */
template <typename container_type_>
void check_sequence_order(std::size_t size = 64) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    auto [container, status] = container_t::make(size);
    if (!expect_equal(status, success_k)) return;
    expect_true(container.empty());
    expect_true(container.capacity() >= size);

    for (std::size_t index = 0; index < size; ++index) {
        expect_equal(append_to(container, member_t(static_cast<trivial_id_t>(index))), success_k);
        expect_equal(container.size(), index + 1);
    }
    for (std::size_t index = 0; index < size; ++index)
        expect_equal(container[index].unique_id, static_cast<trivial_id_t>(index));

    container.clear();
    expect_true(container.empty());
    expect_equal(container.size(), 0u);
    expect_true(container.capacity() >= size);
}

/** Tests every element a sequence holds is destroyed exactly once, by a clear or by teardown. */
template <typename container_type_>
void check_sequence_element_lifetimes(std::size_t size = 32) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    static_assert(std::is_same_v<member_t, counted_key_t>, "this suite counts through counted_key_t");

    counted_key_t::reset();
    {
        auto [made, status] = container_t::make(size);
        if (!expect_equal(status, success_k)) return;
        for (std::size_t index = 0; index < size; ++index)
            expect_equal(append_to(made, member_t(static_cast<trivial_id_t>(index))), success_k);
        made.clear();
        expect_equal(counted_key_t::alive(), std::ptrdiff_t {0});

        for (std::size_t index = 0; index < size; ++index)
            expect_equal(append_to(made, member_t(static_cast<trivial_id_t>(index))), success_k);
    }
    counted_key_t::check_balanced();
}

/** Tests that a move hands the whole sequence over and leaves the source empty. */
template <typename container_type_>
void check_sequence_move_semantics(std::size_t size = 16) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    auto [made, status] = container_t::make(size);
    if (!expect_equal(status, success_k)) return;
    for (std::size_t index = 0; index < size; ++index)
        expect_equal(append_to(made, member_t(static_cast<trivial_id_t>(index))), success_k);

    container_t moved_into(std::move(made));
    if (!expect_equal(moved_into.size(), size)) return;
    expect_true(made.empty());
    for (std::size_t index = 0; index < size; ++index)
        expect_equal(moved_into[index].unique_id, static_cast<trivial_id_t>(index));

    container_t assigned_into;
    assigned_into = std::move(moved_into);
    if (!expect_equal(assigned_into.size(), size)) return;
    expect_true(moved_into.empty());
    for (std::size_t index = 0; index < size; ++index)
        expect_equal(assigned_into[index].unique_id, static_cast<trivial_id_t>(index));
}

/** Tests that a sequence over a budgeted allocator returns every block it was granted. */
template <typename container_type_>
void check_sequence_allocator_ledger(std::size_t size = 16) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    allocation_ledger_t ledger;
    {
        auto [made, status] = container_t::make(size, typename container_t::allocator_t(1, ledger));
        if (!expect_equal(status, success_k)) return;
        expect_equal(ledger.live_allocations(), std::ptrdiff_t {1});
        for (std::size_t index = 0; index < size; ++index)
            expect_equal(append_to(made, member_t(static_cast<trivial_id_t>(index))), success_k);

        // A move assignment carries the allocator, so the destination frees what the source was granted.
        container_t assigned_into;
        assigned_into = std::move(made);
        expect_equal(assigned_into.size(), size);
    }
    ledger.check_balanced();

    ledger.reset();
    ledger.refuse_everything();
    expect_equal(container_t::make(size, typename container_t::allocator_t(2, ledger)).status(),
                 status_t::out_of_memory_heap_k);
    ledger.check_balanced();
}

#pragma endregion Sequence Test Templates

} // namespace ashvardanian::smashtable::test
