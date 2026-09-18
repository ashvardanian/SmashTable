/**
 *  @file include/smashtable/basic_flat_set.hpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief A small sorted set in one contiguous array, searched through a row kit where the element
 *      is a plain key.
 *
 *  Inserting and erasing shift the tail of the array, so the set suits a few hundred elements
 *  rather than millions, in exchange for a search that reads adjacent memory and allocates nothing.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint64_t`

#include <concepts>    // `std::same_as`
#include <iterator>    // `std::distance`, `std::forward_iterator`
#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span`
#include <type_traits> // `std::false_type`, `std::is_nothrow_move_assignable_v`
#include <utility>     // `std::move`

#include "basic_vector.hpp"
#include "row_search.hpp"

namespace ashvardanian::smashtable {

/** A sorted set of unique @p value_type_ ordered by @p comparator_type_. The rank search goes
 *  through @p row_kit_type_ when the element is a @c row_searchable_key ordered by @c less_t, and
 *  through a binary search with the comparator otherwise. */
template <typename value_type_, typename comparator_type_ = less_t, row_kit row_kit_type_ = native_row_kit_t,
          typename allocator_type_ = default_allocator<value_type_>>
class basic_flat_set {
  public:
#pragma region Type Vocabulary

    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility

    /** The key half, which for a set is the whole element. */
    using key_t = value_t;
    using key_type = key_t; // ? STL compatibility

    /** The mapped half, which a set never has. */
    using mapped_t = void;
    using mapped_type = mapped_t; // ? STL compatibility

    using comparator_t = comparator_type_;
    using comparator_type = comparator_t; // ? STL compatibility

    using kit_t = row_kit_type_;

    using allocator_t = allocator_type_;
    using allocator_type = allocator_t; // ? STL compatibility

    using is_associative = std::false_type;

    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "basic_flat_set requires allocators that propagate on move assignment");
    static_assert(std::is_nothrow_move_assignable_v<value_t>,
                  "shifting the tail moves elements inside noexcept code, so they must move-assign without throwing");

  private:
    static constexpr bool searches_through_kit_k = row_searchable_key<value_t> && std::same_as<comparator_t, less_t>;

    using elements_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<value_t>;
    using elements_t = basic_vector<value_t, elements_allocator_t>;

    elements_t elements_;
    ST_NO_UNIQUE_ADDRESS_ comparator_t comparator_ {};

    basic_flat_set(elements_t &&elements, comparator_t comparator) noexcept
        : elements_(std::move(elements)), comparator_(std::move(comparator)) {}

  public:
#pragma endregion Type Vocabulary

#pragma region Constructors and Assignment

    basic_flat_set() noexcept = default;

    explicit basic_flat_set(comparator_t comparator, allocator_t allocator = {}) noexcept
        : elements_(elements_allocator_t(allocator)), comparator_(std::move(comparator)) {}

    basic_flat_set(basic_flat_set &&) noexcept = default;
    basic_flat_set &operator=(basic_flat_set &&) noexcept = default;
    basic_flat_set(basic_flat_set const &) = delete;
    basic_flat_set &operator=(basic_flat_set const &) = delete;

    /** An empty set with room for @p capacity elements, or @c out_of_memory_heap_k. */
    static expected<basic_flat_set> make(std::size_t capacity, comparator_t comparator = {},
                                         allocator_t allocator = {}) noexcept {
        expected<elements_t> made = elements_t::make(capacity, elements_allocator_t(allocator));
        if (!made) return made.status();
        return expected<basic_flat_set>(basic_flat_set(std::move(*made), std::move(comparator)), success_k);
    }

    /** A deep copy, or the reason one could not be made. */
    expected<basic_flat_set> copy() const noexcept {
        auto copied = elements_.copy();
        if (!copied) return copied.status();
        return expected<basic_flat_set>(basic_flat_set(std::move(*copied), comparator_), success_k);
    }

#pragma endregion Constructors and Assignment

#pragma region Capacity

    status_t reserve(std::size_t capacity) noexcept { return elements_.reserve(capacity); }
    void clear() noexcept { elements_.clear(); }

    [[nodiscard]] std::size_t size() const noexcept { return elements_.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return elements_.capacity(); }
    [[nodiscard]] bool empty() const noexcept { return elements_.empty(); }

#pragma endregion Capacity

#pragma region Element Access

    [[nodiscard]] value_t const *data() const noexcept { return elements_.data(); }
    [[nodiscard]] value_t const *begin() const noexcept { return elements_.begin(); }
    [[nodiscard]] value_t const *end() const noexcept { return elements_.end(); }

    /** The element at position @p ordinal in sorted order. */
    [[nodiscard]] value_t const &operator[](std::size_t ordinal) const noexcept { return elements_[ordinal]; }

#pragma endregion Element Access

#pragma region Lookup

    /** How many elements order below @p wanted, which is also where it is or would be inserted. */
    template <typename comparable_type_>
    [[nodiscard]] std::size_t rank(comparable_type_ const &wanted) const noexcept {
        if constexpr (searches_through_kit_k && std::same_as<comparable_type_, value_t>)
            return count_below_sorted<kit_t>(std::span<value_t const>(elements_.data(), elements_.size()), wanted);
        else {
            value_t const *const elements = elements_.data();
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
    [[nodiscard]] value_t const *find(comparable_type_ const &wanted) const noexcept {
        std::size_t const offset = rank(wanted);
        if (offset == elements_.size() || comparator_(wanted, elements_.data()[offset])) return nullptr;
        return elements_.data() + offset;
    }

    /** Whether an element equivalent to @p wanted is here. Always success; a sorted array cannot refuse a read. */
    template <typename comparable_type_>
    expected<bool> contains(comparable_type_ const &wanted) const noexcept {
        return find(wanted) != nullptr;
    }

    /** The first element not below @p wanted, or @c end. */
    template <typename comparable_type_>
    [[nodiscard]] value_t const *lower_bound(comparable_type_ const &wanted) const noexcept {
        return elements_.data() + rank(wanted);
    }

    /** The first element above @p wanted, or @c end. */
    template <typename comparable_type_>
    [[nodiscard]] value_t const *upper_bound(comparable_type_ const &wanted) const noexcept {
        std::size_t offset = rank(wanted);
        if (offset < elements_.size() && !comparator_(wanted, elements_.data()[offset])) ++offset;
        return elements_.data() + offset;
    }

    /** Copies out the element equivalent to @p wanted, or reports @c key_not_found_k. */
    template <typename comparable_type_>
    expected<value_t> find_copy(comparable_type_ const &wanted) const noexcept {
        value_t const *const found = find(wanted);
        if (!found) return key_not_found_k;
        return copy_safely(*found);
    }

    /** Copies out the first element not below @p wanted, or reports @c key_not_found_k. */
    template <typename comparable_type_>
    expected<value_t> lower_bound_copy(comparable_type_ const &wanted) const noexcept {
        value_t const *const found = lower_bound(wanted);
        if (found == elements_.end()) return key_not_found_k;
        return copy_safely(*found);
    }

    /** Copies out the first element above @p wanted, or reports @c key_not_found_k. */
    template <typename comparable_type_>
    expected<value_t> upper_bound_copy(comparable_type_ const &wanted) const noexcept {
        value_t const *const found = upper_bound(wanted);
        if (found == elements_.end()) return key_not_found_k;
        return copy_safely(*found);
    }

    /** Visits every element in [lower, upper) in sorted order, or fewer if the callback halts. */
    template <typename lower_type_ = value_t, typename upper_type_ = value_t, typename callback_type_ = no_op_t>
    status_t range(lower_type_ const &lower, upper_type_ const &upper, callback_type_ &&callback) const noexcept {
        std::size_t const last = rank(upper);
        value_t const *const elements = elements_.data();
        for (std::size_t index = rank(lower); index < last; ++index)
            if (hand_over(callback, elements[index]) == walk_control_t::halt_k) break;
        return success_k;
    }

#pragma endregion Lookup

#pragma region Modifiers

    /** Adds @p element, or answers @c key_already_exists_k and leaves the set alone. */
    status_t insert(value_t element) noexcept {
        std::size_t const offset = rank(element);
        if (offset < elements_.size() && !comparator_(element, elements_.data()[offset])) return key_already_exists_k;
        return insert_at_(offset, std::move(element));
    }

    /** Adds @p element, replacing an equivalent one if the set holds it. */
    status_t upsert(value_t element) noexcept {
        std::size_t const offset = rank(element);
        if (offset < elements_.size() && !comparator_(element, elements_.data()[offset])) {
            elements_.data()[offset] = std::move(element);
            return success_k;
        }
        return insert_at_(offset, std::move(element));
    }

    /**
     *  @brief Adds every element of [ @p first, @p last ) this set lacks, leaving incumbents alone.
     *  @return @c success_k however many keys were already here, or the first refusal.
     *
     *  All-or-nothing over this set from the first element on: the staging set holds every element
     *  before the merge begins, and the merge makes its one allocation before it moves anything.
     */
    template <typename input_iterator_type_>
    status_t insert_if_missing(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        if (first == last) return success_k;
        basic_flat_set staged = stage_alike();
        if (status_t const built = stage_range_<incumbent_policy_t::keeps_the_incumbent_k>(staged, first, last);
            failed(built))
            return built;
        return absorb<incumbent_policy_t::keeps_the_incumbent_k>(staged);
    }

    /**
     *  @brief Adds every element of [ @p first, @p last ), refusing the batch over a key already here.
     *  @return @c key_already_exists_k when any key is taken, or the first refusal from the build.
     *
     *  All-or-nothing over this set from the first element on, a taken key included: the whole range
     *  is staged and checked before the merge absorbing it allocates anything.
     */
    template <typename input_iterator_type_>
    status_t insert(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        if (first == last) return success_k;
        basic_flat_set staged = stage_alike();
        if (status_t const built = stage_range_<incumbent_policy_t::refuses_the_newcomer_k>(staged, first, last);
            failed(built))
            return built;
        for (value_t const &candidate : staged)
            if (find(candidate)) return key_already_exists_k;
        return absorb<incumbent_policy_t::keeps_the_incumbent_k>(staged);
    }

    /**
     *  @brief Writes every element of [ @p first, @p last ), replacing the equivalent ones held here.
     *  @return The first refusal from the build or the merge, or @c success_k for the whole range.
     *
     *  All-or-nothing over this set from the first element on; an element already here is overwritten
     *  by the merge, which moves rather than copies once its buffer is secured.
     */
    template <typename input_iterator_type_>
    status_t upsert(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        if (first == last) return success_k;
        basic_flat_set staged = stage_alike();
        if (status_t const built = stage_range_<incumbent_policy_t::takes_the_newcomer_k>(staged, first, last);
            failed(built))
            return built;
        return absorb<incumbent_policy_t::takes_the_newcomer_k>(staged);
    }

    /** Removes the element equivalent to @p wanted, or answers @c key_not_found_k. */
    template <typename comparable_type_>
    status_t erase(comparable_type_ const &wanted) noexcept {
        std::size_t const offset = rank(wanted);
        value_t *const elements = elements_.data();
        if (offset == elements_.size() || comparator_(wanted, elements[offset])) return key_not_found_k;
        for (std::size_t index = offset; index + 1 < elements_.size(); ++index)
            elements[index] = std::move(elements[index + 1]);
        elements_.pop_back();
        return success_k;
    }

    /** Erases the half-open range [lower, upper), handing each removed element to @p callback first. */
    template <typename lower_type_, typename upper_type_, typename callback_type_ = no_op_t>
    void erase_range(lower_type_ const &lower, upper_type_ const &upper, callback_type_ &&callback = {}) noexcept {
        std::size_t const first = rank(lower);
        std::size_t const last = rank(upper);
        if (first >= last) return;
        value_t *const elements = elements_.data();
        for (std::size_t index = first; index < last; ++index) callback(elements[index]);
        for (std::size_t index = last; index < elements_.size(); ++index)
            elements[index - (last - first)] = std::move(elements[index]);
        for (std::size_t removed = last - first; removed > 0; --removed) elements_.pop_back();
    }

#pragma endregion Modifiers

  private:
#pragma region Staging

    /** A set ordered and allocated exactly as this one is, for a batch to be built in. */
    basic_flat_set stage_alike() const noexcept {
        return basic_flat_set(elements_t(elements_.get_allocator()), comparator_);
    }

    /**
     *  @brief Fills @p staged with [ @p first, @p last ), duplicating every element outside this set.
     *  @return The first refusal, naming its own cause, or @c success_k for the whole range.
     */
    template <incumbent_policy_t policy_, typename input_iterator_type_>
    static status_t stage_range_(basic_flat_set &staged, input_iterator_type_ first,
                                 input_iterator_type_ last) noexcept {
        if constexpr (std::forward_iterator<input_iterator_type_>)
            if (status_t const room = staged.elements_.reserve(static_cast<std::size_t>(std::distance(first, last)));
                failed(room))
                return room;
        return stage_each<value_t>(first, last, [&](value_t &&candidate) noexcept -> status_t {
            // Each verb collapses a key repeated inside the range the way its own name reads.
            if constexpr (policy_ == incumbent_policy_t::takes_the_newcomer_k)
                return staged.upsert(std::move(candidate));
            else {
                std::size_t const offset = staged.rank(candidate);
                bool const repeated =
                    offset < staged.size() && !staged.comparator_(candidate, staged.elements_[offset]);
                if constexpr (policy_ == incumbent_policy_t::refuses_the_newcomer_k)
                    if (repeated) return key_already_exists_k;
                return repeated ? success_k : staged.insert_at_(offset, std::move(candidate));
            }
        });
    }

    /**
     *  @brief Merges @p staged into this set, which afterwards holds the union of the two.
     *  @tparam policy_ Which side keeps a key both sets hold.
     *  @return @c out_of_memory_heap_k when the replacement array is refused, which happens before
     *      any element moves, so this set is untouched either way.
     */
    template <incumbent_policy_t policy_>
    status_t absorb(basic_flat_set &staged) noexcept {
        if (staged.empty()) return success_k;
        if (empty()) {
            elements_ = std::move(staged.elements_);
            return success_k;
        }

        // Both runs are sorted, so one walk counts the overlap and the union follows by arithmetic.
        std::size_t shared = 0;
        for (std::size_t here = 0, there = 0; here != elements_.size() && there != staged.elements_.size();) {
            if (comparator_(elements_[here], staged.elements_[there])) ++here;
            else if (comparator_(staged.elements_[there], elements_[here])) ++there;
            else {
                ++here;
                ++there;
                ++shared;
            }
        }

        expected<elements_t> merged =
            elements_t::make(elements_.size() + staged.elements_.size() - shared, elements_.get_allocator());
        if (!merged) return merged.status();

        // Every write below moves into room already secured, so none of them can refuse.
        auto take = [&](value_t &&element) noexcept {
            [[maybe_unused]] status_t const placed = merged->push_back(assume_reserved, std::move(element));
        };
        std::size_t here = 0, there = 0;
        while (here != elements_.size() || there != staged.elements_.size()) {
            if (there == staged.elements_.size()) take(std::move(elements_[here++]));
            else if (here == elements_.size()) take(std::move(staged.elements_[there++]));
            else if (comparator_(elements_[here], staged.elements_[there])) take(std::move(elements_[here++]));
            else if (comparator_(staged.elements_[there], elements_[here])) take(std::move(staged.elements_[there++]));
            else {
                if constexpr (policy_ == incumbent_policy_t::keeps_the_incumbent_k) take(std::move(elements_[here]));
                else take(std::move(staged.elements_[there]));
                ++here;
                ++there;
            }
        }
        elements_ = *std::move(merged);
        return success_k;
    }

#pragma endregion Staging

    status_t insert_at_(std::size_t offset, value_t &&element) noexcept {
        status_t const status = elements_.push_back(std::move(element));
        if (failed(status)) return status;
        value_t *const elements = elements_.data();
        std::size_t index = elements_.size() - 1;
        if (index == offset) return success_k;
        value_t appended = std::move(elements[index]);
        for (; index > offset; --index) elements[index] = std::move(elements[index - 1]);
        elements[offset] = std::move(appended);
        return success_k;
    }
};

static_assert(ordered_collection<basic_flat_set<std::uint64_t>>,
              "a flat set answers a key, a bound and a range the way every ordered collection does");
static_assert(batches_atomically<basic_flat_set<std::uint64_t>>, "a flat set takes a whole batch or none of it");
static_assert(!basic_flat_set<std::uint64_t>::is_associative::value, "a flat set stores bare keys");

} // namespace ashvardanian::smashtable
