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
void test_sequence_tags() {

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
void test_sequence_order(std::size_t size = 64) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    expected<container_t> made = container_t::make(size);
    st_verify_(made);
    container_t &container = *made;
    st_verify_(container.empty());
    st_verify_ge_(container.capacity(), size);

    for (std::size_t index = 0; index < size; ++index) {
        st_verify_(append_to(container, member_t(static_cast<trivial_id_t>(index))));
        st_verify_eq_(container.size(), index + 1);
    }
    for (std::size_t index = 0; index < size; ++index)
        st_verify_eq_(container[index].unique_id, static_cast<trivial_id_t>(index));

    container.clear();
    st_verify_(container.empty());
    st_verify_eq_(container.size(), 0u);
    st_verify_ge_(container.capacity(), size);
}

/** Tests that every element a sequence holds is destroyed exactly once, by a clear or by teardown. */
template <typename container_type_>
void test_sequence_element_lifetimes(std::size_t size = 32) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    static_assert(std::is_same_v<member_t, counted_key_t>, "this suite counts through counted_key_t");

    counted_key_t::reset();
    {
        expected<container_t> made = container_t::make(size);
        st_verify_(made);
        for (std::size_t index = 0; index < size; ++index)
            st_verify_(append_to(*made, member_t(static_cast<trivial_id_t>(index))));
        made->clear();
        st_verify_eq_(counted_key_t::alive(), std::ptrdiff_t {0});

        for (std::size_t index = 0; index < size; ++index)
            st_verify_(append_to(*made, member_t(static_cast<trivial_id_t>(index))));
    }
    counted_key_t::verify_balanced();
}

/** Tests that a move hands the whole sequence over and leaves the source empty. */
template <typename container_type_>
void test_sequence_move_semantics(std::size_t size = 16) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    expected<container_t> made = container_t::make(size);
    st_verify_(made);
    for (std::size_t index = 0; index < size; ++index)
        st_verify_(append_to(*made, member_t(static_cast<trivial_id_t>(index))));

    container_t moved_into(std::move(*made));
    st_verify_eq_(moved_into.size(), size);
    st_verify_(made->empty());
    for (std::size_t index = 0; index < size; ++index)
        st_verify_eq_(moved_into[index].unique_id, static_cast<trivial_id_t>(index));

    container_t assigned_into;
    assigned_into = std::move(moved_into);
    st_verify_eq_(assigned_into.size(), size);
    st_verify_(moved_into.empty());
    for (std::size_t index = 0; index < size; ++index)
        st_verify_eq_(assigned_into[index].unique_id, static_cast<trivial_id_t>(index));
}

/** Tests that a sequence over a budgeted allocator returns every block it was granted. */
template <typename container_type_>
void test_sequence_allocator_ledger(std::size_t size = 16) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    allocation_ledger_t ledger;
    {
        expected<container_t> made = container_t::make(size, typename container_t::allocator_t(1, ledger));
        st_verify_(made);
        st_verify_eq_(ledger.live_allocations(), std::ptrdiff_t {1});
        for (std::size_t index = 0; index < size; ++index)
            st_verify_(append_to(*made, member_t(static_cast<trivial_id_t>(index))));

        // A move assignment carries the allocator, so the destination frees what the source was granted.
        container_t assigned_into;
        assigned_into = std::move(*made);
        st_verify_eq_(assigned_into.size(), size);
    }
    ledger.verify_balanced();

    ledger.reset();
    ledger.refuse_everything();
    st_verify_eq_(container_t::make(size, typename container_t::allocator_t(2, ledger)).status(),
                  status_t::out_of_memory_heap_k);
    ledger.verify_balanced();
}

#pragma endregion Sequence Test Templates

} // namespace ashvardanian::smashtable::test
