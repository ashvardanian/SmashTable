/**
 *  @file include/smashtable/intrusive_list.hpp
 *  @author Ash Vardanian
 *  @date November 2, 2026
 *  @brief A first-in first-out list threaded through the @c next indices of records kept elsewhere.
 *
 *  The list owns two indices and nothing else; the records are any array indexed by position, and
 *  each operation takes them as an argument. Records that also carry a @c previous index are
 *  unlinked from the middle in constant time, the others by a walk from the oldest.
 */
#pragma once
#include <cstdint> // `std::uint32_t`

#include <concepts> // `std::convertible_to`, `std::unsigned_integral`
#include <limits>   // `std::numeric_limits`

#include "smashtable/shared.hpp"

namespace ashvardanian::smashtable {

/** Records indexed by position, each naming the one after it through an index @c next. */
template <typename records_type_, typename index_type_>
concept singly_linked_records = requires(records_type_ &records, index_type_ index) {
    records[index].next = index;
    { records[index].next } -> std::convertible_to<index_type_>;
};

/** Linked records also naming the one before them through @c previous, absent when unlinked. */
template <typename records_type_, typename index_type_>
concept doubly_linked_records =
    singly_linked_records<records_type_, index_type_> && requires(records_type_ &records, index_type_ index) {
        records[index].previous = index;
        { records[index].previous } -> std::convertible_to<index_type_>;
    };

/**
 *  @brief The oldest and newest of a first-in first-out list of record indices.
 *  @tparam index_type_ Names a record, whose largest value is @c absent_k.
 */
template <std::unsigned_integral index_type_ = std::uint32_t>
struct intrusive_list {
    using index_t = index_type_;

    /** Ends a list and names no record. */
    static constexpr index_t absent_k = (std::numeric_limits<index_t>::max)();

    /** The record @c pop answers next. */
    index_t oldest {absent_k};

    /** The record @c push linked last. */
    index_t newest {absent_k};

    [[nodiscard]] bool empty() const noexcept { return oldest == absent_k; }

    /** Links the record at @p index after the newest. */
    template <typename records_type_>
        requires singly_linked_records<records_type_, index_t>
    void push(records_type_ &records, index_t index) noexcept {
        records[index].next = absent_k;
        if constexpr (doubly_linked_records<records_type_, index_t>) records[index].previous = newest;
        if (newest != absent_k) records[newest].next = index;
        else oldest = index;
        newest = index;
    }

    /** Unlinks the oldest record and answers its index, or @c absent_k when the list is empty. */
    template <typename records_type_>
        requires singly_linked_records<records_type_, index_t>
    index_t pop(records_type_ &records) noexcept {
        index_t const index = oldest;
        if (index == absent_k) return index;
        oldest = records[index].next;
        if (oldest == absent_k) newest = absent_k;
        else if constexpr (doubly_linked_records<records_type_, index_t>) records[oldest].previous = absent_k;
        records[index].next = absent_k;
        return index;
    }

    /**
     *  @brief Unlinks the record at @p index from wherever it sits.
     *
     *  @par Statuses:
     *  - @c key_not_found_k The record was never pushed, or was popped or erased since.
     *    A record of another list is not detected.
     */
    template <typename records_type_>
        requires singly_linked_records<records_type_, index_t>
    status_t erase(records_type_ &records, index_t index) noexcept {
        index_t previous = absent_k;
        if constexpr (doubly_linked_records<records_type_, index_t>) {
            previous = records[index].previous;
            if (previous == absent_k && oldest != index) return key_not_found_k;
        }
        else {
            index_t current = oldest;
            for (; current != absent_k && current != index; current = records[current].next) previous = current;
            if (current == absent_k) return key_not_found_k;
        }
        index_t const next = records[index].next;
        if (previous != absent_k) records[previous].next = next;
        else oldest = next;
        if (next != absent_k) {
            if constexpr (doubly_linked_records<records_type_, index_t>) records[next].previous = previous;
        }
        else newest = previous;
        records[index].next = absent_k;
        if constexpr (doubly_linked_records<records_type_, index_t>) records[index].previous = absent_k;
        return success_k;
    }
};

static_assert(sizeof(intrusive_list<>) == 8, "two 32-bit indices");

} // namespace ashvardanian::smashtable
