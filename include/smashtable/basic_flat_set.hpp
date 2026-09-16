/**
 *  @brief A small sorted set in one contiguous array, searched through a row kit where the element is a plain key.
 *  @author Ash Vardanian
 *  @file include/smashtable/basic_flat_set.hpp
 *  @date September 15, 2026
 *
 *  Inserting and erasing shift the tail of the array, so the set suits a few hundred elements rather than millions,
 *  in exchange for a search that reads adjacent memory and allocates nothing.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::size_t`

#include <concepts>    // `std::same_as`
#include <span>        // `std::span`
#include <type_traits> // `std::is_nothrow_move_assignable_v`
#include <utility>     // `std::move`

#include "basic_vector.hpp"
#include "row_search.hpp"

namespace ashvardanian::smashtable {

/**
 *  A sorted set of unique @p element_type_ ordered by @p comparator_type_.
 *  The rank search goes through @p row_kit_type_ when the element is a @c row_searchable_key ordered by @c less_t,
 *  and through a binary search with the comparator otherwise.
 */
template <typename element_type_, typename comparator_type_ = less_t, row_kit row_kit_type_ = native_row_kit_t,
          typename allocator_type_ = default_allocator<element_type_>>
class basic_flat_set {
  public:
    using element_t = element_type_;
    using comparator_t = comparator_type_;
    using kit_t = row_kit_type_;
    using allocator_t = allocator_type_;

    static_assert(std::is_nothrow_move_assignable_v<element_t>,
                  "shifting the tail moves elements inside noexcept code, so they must move-assign without throwing");

  private:
    static constexpr bool searches_through_kit_k = row_searchable_key<element_t> && std::same_as<comparator_t, less_t>;

    basic_vector<element_t, allocator_t> elements_;
    ST_NO_UNIQUE_ADDRESS_ comparator_t comparator_ {};

    basic_flat_set(basic_vector<element_t, allocator_t> &&elements, comparator_t comparator) noexcept
        : elements_(std::move(elements)), comparator_(std::move(comparator)) {}

  public:
    basic_flat_set() noexcept = default;

    explicit basic_flat_set(comparator_t comparator, allocator_t allocator = {}) noexcept
        : elements_(std::move(allocator)), comparator_(std::move(comparator)) {}

    basic_flat_set(basic_flat_set &&) noexcept = default;
    basic_flat_set &operator=(basic_flat_set &&) noexcept = default;
    basic_flat_set(basic_flat_set const &) = delete;
    basic_flat_set &operator=(basic_flat_set const &) = delete;

    /** An empty set with room for @p capacity elements, or @c out_of_memory_heap_k. */
    [[nodiscard]] static expected<basic_flat_set> make(std::size_t capacity, comparator_t comparator = {},
                                                       allocator_t allocator = {}) noexcept {
        auto made = basic_vector<element_t, allocator_t>::make(capacity, std::move(allocator));
        if (!made) return made.status();
        return expected<basic_flat_set>(basic_flat_set(std::move(*made), std::move(comparator)), success_k);
    }

    /** A deep copy, or the reason one could not be made. */
    [[nodiscard]] expected<basic_flat_set> copy() const noexcept {
        auto copied = elements_.copy();
        if (!copied) return copied.status();
        return expected<basic_flat_set>(basic_flat_set(std::move(*copied), comparator_), success_k);
    }

    [[nodiscard]] status_t reserve(std::size_t capacity) noexcept { return elements_.reserve(capacity); }
    void clear() noexcept { elements_.clear(); }

    [[nodiscard]] std::size_t size() const noexcept { return elements_.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return elements_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return elements_.empty(); }

    [[nodiscard]] element_t const *data() const noexcept { return elements_.data(); }
    [[nodiscard]] element_t const *begin() const noexcept { return elements_.begin(); }
    [[nodiscard]] element_t const *end() const noexcept { return elements_.end(); }

    /** The element at position @p ordinal in sorted order. */
    [[nodiscard]] element_t const &operator[](std::size_t ordinal) const noexcept { return elements_[ordinal]; }

    /** How many elements order below @p wanted, which is also where it is or would be inserted. */
    template <typename comparable_type_>
    [[nodiscard]] std::size_t rank(comparable_type_ const &wanted) const noexcept {
        if constexpr (searches_through_kit_k && std::same_as<comparable_type_, element_t>)
            return count_below_sorted<kit_t>(std::span<element_t const>(elements_.data(), elements_.size()), wanted);
        else {
            element_t const *const elements = elements_.data();
            std::size_t low = 0;
            std::size_t length = elements_.size();
            while (length > 0) {
                std::size_t const half = length / 2;
                if (comparator_(elements[low + half], wanted)) {
                    low += half + 1;
                    length -= half + 1;
                }
                else length = half;
            }
            return low;
        }
    }

    /** The element equivalent to @p wanted, or null. */
    template <typename comparable_type_>
    [[nodiscard]] element_t const *find(comparable_type_ const &wanted) const noexcept {
        std::size_t const offset = rank(wanted);
        if (offset == elements_.size() || comparator_(wanted, elements_.data()[offset])) return nullptr;
        return elements_.data() + offset;
    }

    template <typename comparable_type_>
    [[nodiscard]] bool contains(comparable_type_ const &wanted) const noexcept {
        return find(wanted) != nullptr;
    }

    /** Adds @p element, or answers @c key_already_exists_k and leaves the set alone. */
    [[nodiscard]] status_t insert(element_t element) noexcept {
        std::size_t const offset = rank(element);
        if (offset < elements_.size() && !comparator_(element, elements_.data()[offset])) return key_already_exists_k;
        return insert_at_(offset, std::move(element));
    }

    /** Adds @p element, replacing an equivalent one if the set holds it. */
    [[nodiscard]] status_t upsert(element_t element) noexcept {
        std::size_t const offset = rank(element);
        if (offset < elements_.size() && !comparator_(element, elements_.data()[offset])) {
            elements_.data()[offset] = std::move(element);
            return success_k;
        }
        return insert_at_(offset, std::move(element));
    }

    /** Adds every element of @p incoming the set lacks, reserving once, so only the reservation can fail. */
    [[nodiscard]] status_t insert_missing(std::span<element_t const> incoming) noexcept
        requires std::is_nothrow_copy_constructible_v<element_t>
    {
        status_t const status = elements_.reserve(elements_.size() + incoming.size());
        if (failed(status)) return status;
        for (element_t const &element : incoming) {
            std::size_t const offset = rank(element);
            if (offset < elements_.size() && !comparator_(element, elements_.data()[offset])) continue;
            [[maybe_unused]] status_t const inserted = insert_at_(offset, element_t {element});
            assert(succeeded(inserted) && "the reservation covers every insertion");
        }
        return success_k;
    }

    /** Removes the element equivalent to @p wanted, or answers @c key_not_found_k. */
    template <typename comparable_type_>
    [[nodiscard]] status_t erase(comparable_type_ const &wanted) noexcept {
        std::size_t const offset = rank(wanted);
        element_t *const elements = elements_.data();
        if (offset == elements_.size() || comparator_(wanted, elements[offset])) return key_not_found_k;
        for (std::size_t index = offset; index + 1 < elements_.size(); ++index)
            elements[index] = std::move(elements[index + 1]);
        elements_.pop_back();
        return success_k;
    }

  private:
    [[nodiscard]] status_t insert_at_(std::size_t offset, element_t &&element) noexcept {
        status_t const status = elements_.push_back(std::move(element));
        if (failed(status)) return status;
        element_t *const elements = elements_.data();
        std::size_t index = elements_.size() - 1;
        if (index == offset) return success_k;
        element_t appended = std::move(elements[index]);
        for (; index > offset; --index) elements[index] = std::move(elements[index - 1]);
        elements[offset] = std::move(appended);
        return success_k;
    }
};

} // namespace ashvardanian::smashtable
