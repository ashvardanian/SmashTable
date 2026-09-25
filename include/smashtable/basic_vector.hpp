/**
 *  @file include/smashtable/basic_vector.hpp
 *  @author Ash Vardanian
 *  @date October 19, 2025
 *  @brief Exception-free dynamic array with explicit error handling.
 *
 *  @section basic_vector_features Features
 *
 *  Unlike @c std::vector, this implementation:
 *  - Never throws exceptions - returns @c status_t or @c expected<T> for fallible operations
 *  - For both @c noexcept-constructible elements and ones with fallible @c .make() methods
 *  - Offers performance variants with @c assume_reserved tag for pre-checked hot paths
 *  - Batch operations like @c resize() are all-or-nothing on failure, leaving elements as they were
 *
 *  @section basic_vector_requirements Requirements
 *
 *  @section basic_vector_element_type Element Type
 *
 *  - Nothrow move-constructible (required); nothrow default-constructible for @c resize()
 *  - For @c emplace_back(): Nothrow constructible OR provides `::make(...) → expected<T>`
 *  - For @c copy()/resize(): Nothrow copy-constructible OR provides `.copy() const → expected<T>`
 *
 *  @section basic_vector_allocator_type Allocator Type
 *
 *  - Must propagate on move assignment @c propagate_on_container_move_assignment==true
 *  - Must report exhaustion by returning null from @c allocate, never by throwing, which is why the
 *    default is @c default_allocator rather than @c std::allocator
 *  - For @c swap(): If non-propagating, both vectors must use equal allocators, otherwise
 *    @c invalid_argument_k is returned
 *
 *  @see https://en.wikipedia.org/wiki/Vector_(C%2B%2B)
 *  @see https://en.cppreference.com/w/cpp/container/vector
 */
#pragma once
#include <cassert> // `assert`

#include <concepts> // `std::invocable`
#include <memory>   // `std::allocator_traits`
#include <utility>  // `std::exchange`, `std::forward`

#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief Exception-free dynamic array that uses error codes instead of exceptions. Provides RAII
 *      memory management with explicit failure handling, unlike @c std::vector.
 *
 *  @section basic_vector_template_requirements Template Requirements
 *
 *  - @p value_type_ must be nothrow move-constructible, and nothrow default-constructible to be
 *    resized.
 *  - For types with potentially throwing constructors, provide a static @c .make() method returning
 *    @c expected<value_type_> to enable exception-free construction via @c emplace_back().
 *
 *  @see https://en.cppreference.com/w/cpp/container/vector
 */
template <typename value_type_, typename allocator_type_ = default_allocator<value_type_>>
class basic_vector {
  public:
    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility

    using allocator_t = allocator_type_;
    using allocator_type = allocator_t; // ? STL compatibility

    // Allocator propagation requirement for exception-free move assignment
    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "basic_vector requires allocators that propagate on move assignment");

    // Reallocation moves every element inside a `noexcept` function, so a throwing move terminates
    static_assert(std::is_nothrow_move_constructible_v<value_t>,
                  "basic_vector relocates elements in noexcept code, so the element must move without throwing");

  private:
    value_t *data_ {nullptr};
    std::size_t size_ {0};
    std::size_t capacity_ {0};
    SMASHTABLE_NO_UNIQUE_ADDRESS_ allocator_t allocator_ {};

  public:
    /** Default constructor creates an empty vector. */
    basic_vector() noexcept = default;

    /**
     *  @brief Constructor with custom allocator.
     *  @param[in] allocator The allocator instance to use.
     */
    explicit basic_vector(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}

    /** Destructor deallocates memory and destroys all elements. */
    ~basic_vector() noexcept {
        // Destroy all constructed elements
        for (std::size_t index = 0; index < size_; ++index) data_[index].~value_t();
        // Deallocate memory
        if (data_) allocator_.deallocate(data_, capacity_);
    }

    /** Move constructor transfers ownership. */
    basic_vector(basic_vector &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          capacity_(std::exchange(other.capacity_, 0)), allocator_(std::move(other.allocator_)) {}

    /** Move assignment transfers ownership after destroying current elements. */
    basic_vector &operator=(basic_vector &&other) noexcept {
        if (this != &other) {
            // Destroy current elements
            for (std::size_t index = 0; index < size_; ++index) data_[index].~value_t();
            // Deallocate current memory
            if (data_) allocator_.deallocate(data_, capacity_);
            // Transfer ownership
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            capacity_ = std::exchange(other.capacity_, 0);
            allocator_ = std::move(other.allocator_);
        }
        return *this;
    }

    /** Copy construction is deleted (use @c .copy() method for explicit copies). */
    basic_vector(basic_vector const &) = delete;

    /** Copy assignment is deleted (use @c .copy() method for explicit copies). */
    basic_vector &operator=(basic_vector const &) = delete;

    /**
     *  @brief Creates a new vector with specified capacity.
     *  @param[in] initial_capacity Initial capacity to allocate.
     *  @return @c expected<basic_vector> containing the new vector, or error status.
     */
    static expected<basic_vector> make(std::size_t initial_capacity) noexcept {
        return make(initial_capacity, allocator_t {});
    }

    /**
     *  @brief Creates a new vector with specified capacity and allocator.
     *  @param[in] initial_capacity Initial capacity to allocate.
     *  @param[in] allocator The allocator instance to use.
     *  @return @c expected<basic_vector> containing the new vector, or error status.
     */
    static expected<basic_vector> make(std::size_t initial_capacity, allocator_t allocator) noexcept {
        basic_vector vec(std::move(allocator));
        if (initial_capacity > 0) {
            auto status = vec.reserve(initial_capacity);
            if (failed(status)) return expected<basic_vector>(basic_vector(vec.allocator_), status);
        }
        return expected<basic_vector>(std::move(vec), success_k);
    }

    /**
     *  @brief Creates a deep copy of this vector.
     *  @return @c expected<basic_vector> containing the copy, or error status.
     */
    expected<basic_vector> copy() const noexcept {
        basic_vector result(allocator_);
        if (capacity_ > 0) {
            auto status = result.reserve(capacity_);
            if (failed(status)) return expected<basic_vector>(basic_vector(allocator_), status);
        }
        // Copy all elements using copy_safely
        for (std::size_t index = 0; index < size_; ++index) {
            auto copy_result = copy_safely(data_[index]);
            if (!copy_result) {
                // Clean up partially constructed elements
                for (std::size_t built = 0; built < result.size_; ++built) result.data_[built].~value_t();
                result.size_ = 0;
                return expected<basic_vector>(basic_vector(allocator_), copy_result.status());
            }
            new (&result.data_[result.size_++]) value_t(std::move(*copy_result));
        }
        return expected<basic_vector>(std::move(result), success_k);
    }

    /**
     *  @brief Reserves capacity for at least @p new_capacity elements. Grows by at least a
     *      doubling, so asking for one more slot at a time still amortizes to constant cost.
     *  @param[in] new_capacity The new capacity.
     *  @return Success, or @c out_of_memory_heap_k if allocation fails or the request is invalid.
     */
    status_t reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return success_k;

        // A capacity whose byte count does not fit an address is refused here rather than passed on:
        // an allocator that multiplies without checking would answer it with a small block, and the
        // recorded capacity would then invite every later write past the end of it.
        constexpr std::size_t max_capacity = static_cast<std::size_t>(-1) / sizeof(value_t);
        if (new_capacity > max_capacity) return out_of_memory_heap_k;

        std::size_t const doubled_capacity = capacity_ > max_capacity / 2 ? max_capacity : capacity_ * 2;
        std::size_t const seeded_capacity = larger_of<std::size_t>(doubled_capacity, 4);
        std::size_t const grown_capacity =
            smaller_of<std::size_t>(larger_of<std::size_t>(new_capacity, seeded_capacity), max_capacity);
        auto new_data = allocator_.allocate(grown_capacity);
        if (!new_data) return out_of_memory_heap_k;

        // Move existing elements to new storage
        for (std::size_t index = 0; index < size_; ++index) new (&new_data[index]) value_t(std::move(data_[index]));

        // Destroy moved-from elements in old storage
        for (std::size_t index = 0; index < size_; ++index) data_[index].~value_t();

        if (data_) allocator_.deallocate(data_, capacity_);
        data_ = new_data;
        capacity_ = grown_capacity;
        return success_k;
    }

    /**
     *  @brief Appends an element without capacity check (performance variant). Requires
     *      pre-reserved capacity. Use for hot paths after @c reserve().
     *
     *  Tag indicating capacity was pre-reserved.
     *
     *  @param[in] value Element to append (moved into the vector).
     *  @return Always returns success for noexcept move construction.
     */
    status_t push_back(assume_reserved_t, value_t &&value) noexcept {
        assert(size_ < capacity_ && "push_back with assume_reserved requires pre-reserved capacity");
        new (&data_[size_++]) value_t(std::move(value));
        return success_k;
    }

    /**
     *  @brief Appends an element to the end of the vector. Automatically grows capacity using 2x
     *      strategy if needed.
     *
     *  @param[in] value Element to append (moved into the vector).
     *  @return Success, or @c out_of_memory_heap_k if reallocation fails.
     */
    status_t push_back(value_t &&value) noexcept {
        if (size_ >= capacity_) {
            auto status = reserve(size_ + 1);
            if (failed(status)) return status;
        }
        return push_back(assume_reserved, std::move(value));
    }

    /** Removes the last element from the vector. Precondition: Vector must not be empty. */
    void pop_back() noexcept {
        assert(size_ > 0 && "pop_back requires non-empty vector");
        data_[--size_].~value_t();
    }

    /**
     *  @brief Inserts @p value at @p position into reserved room, shifting the tail up by one.
     *  @param[in] position Index the new element takes, at most @c size().
     *  @param[in] value Element to insert (moved into the vector).
     *  @return Always success, since nothing here can refuse.
     */
    status_t insert(assume_reserved_t, std::size_t position, value_t &&value) noexcept {
        assert(position <= size_ && size_ < capacity_ &&
               "insert with assume_reserved requires room and a position inside");
        if (position == size_) return push_back(assume_reserved, std::move(value));

        new (&data_[size_]) value_t(std::move(data_[size_ - 1]));
        ++size_;
        for (std::size_t index = size_ - 2; index != position; --index) data_[index] = std::move(data_[index - 1]);
        data_[position] = std::move(value);
        return success_k;
    }

    /**
     *  @brief Inserts @p value at @p position, shifting the elements at and after it up by one.
     *
     *  Linear in the elements after @p position, which is what a sorted run of a few hundred
     *  entries costs and why an ordered container built on this stays small.
     *
     *  @param[in] position Index the new element takes, at most @c size().
     *  @param[in] value Element to insert (moved into the vector), never one of this vector's own.
     *  @return Success, or @c out_of_memory_heap_k if reallocation fails.
     */
    status_t insert(std::size_t position, value_t &&value) noexcept {
        assert((&value < data_ || &value >= data_ + size_) && "insert cannot take an element of this very vector");
        if (size_ >= capacity_)
            if (status_t const grown = reserve(size_ + 1); failed(grown)) return grown;
        return insert(assume_reserved, position, std::move(value));
    }

    /**
     *  @brief Removes @p position's element, handing it to @p callback, then shifts the rest down.
     *  @param[in] position Index of the element to remove, below @c size().
     *  @param[in] callback Receives the element before it is moved away. Must be @c noexcept.
     */
    template <typename callback_type_ = no_op_t>
        requires std::invocable<callback_type_ &, value_t &>
    void erase(std::size_t position, callback_type_ &&callback = {}) noexcept {
        erase(position, 1, std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Removes @p count elements from @p position, handing each to @p callback first and
     *      shifting the rest down.
     *  @param[in] position Index of the first element to remove.
     *  @param[in] count Elements to remove, which must not run past the end.
     *  @param[in] callback Receives each element before it is moved away. Must be @c noexcept.
     */
    template <typename callback_type_ = no_op_t>
        requires std::invocable<callback_type_ &, value_t &>
    void erase(std::size_t position, std::size_t count, callback_type_ &&callback = {}) noexcept {
        assert(position <= size_ && count <= size_ - position && "erase requires a range inside the vector");
        for (std::size_t index = position; index != position + count; ++index) callback(data_[index]);
        for (std::size_t index = position; index + count != size_; ++index)
            data_[index] = std::move(data_[index + count]);
        for (std::size_t removed = 0; removed != count; ++removed) pop_back();
    }

    /**
     *  @brief Removes every element @p predicate admits in one pass, keeping the order of the rest.
     *  @param[in] predicate Decides which elements go. Must be @c noexcept.
     *  @param[in] callback Receives each removed element before it moves away. Must be @c noexcept.
     *  @return How many elements were removed.
     */
    template <typename predicate_type_, typename callback_type_ = no_op_t>
        requires std::invocable<callback_type_ &, value_t &>
    std::size_t erase_if(predicate_type_ &&predicate, callback_type_ &&callback = {}) noexcept {
        std::size_t kept = 0;
        for (std::size_t index = 0; index != size_; ++index) {
            if (predicate(data_[index])) {
                callback(data_[index]);
                continue;
            }
            if (kept != index) data_[kept] = std::move(data_[index]);
            ++kept;
        }
        std::size_t const removed = size_ - kept;
        for (std::size_t dropped = 0; dropped != removed; ++dropped) pop_back();
        return removed;
    }

    /**
     *  @brief Constructs element in-place without capacity check (performance variant). Requires
     *      pre-reserved capacity. Use for hot paths after @c reserve().
     *
     *  @tparam args_types_ Types of arguments to forward to element constructor.
     *
     *  Tag indicating capacity was pre-reserved.
     *
     *  @param[in] args Arguments to forward to element constructor.
     *  @return Success, or error from @c .make() method if construction can throw.
     */
    template <typename... args_types_>
    status_t emplace_back(assume_reserved_t, args_types_ &&...args) noexcept {
        assert(size_ < capacity_ && "emplace_back with assume_reserved requires pre-reserved capacity");

        // Fast path: noexcept constructor - construct directly in place
        if constexpr (std::is_nothrow_constructible_v<value_t, args_types_...>) {
            new (&data_[size_++]) value_t(std::forward<args_types_>(args)...);
            return success_k;
        }
        // Slow path: potentially throwing constructor - use .make() method
        else if constexpr (has_make_method<value_t, args_types_...>) {
            auto result = value_t::make(std::forward<args_types_>(args)...);
            if (!result) return result.status();
            new (&data_[size_++]) value_t(std::move(*result));
            return success_k;
        }
        else {
            static_assert(
                std::is_nothrow_constructible_v<value_t, args_types_...> || has_make_method<value_t, args_types_...>,
                "Type must be nothrow constructible or provide a static .make(...)");
            return status_t::unknown_k;
        }
    }

    /**
     *  @brief Constructs element in-place at the end of the vector. Automatically grows capacity
     *      using 2x strategy if needed.
     *
     *  @tparam args_types_ Types of arguments to forward to element constructor.
     *
     *  @param[in] args Arguments to forward to element constructor.
     *  @return Success, or @c out_of_memory_heap_k if reallocation fails.
     *
     *  @note For types with potentially throwing constructors, provide a static @c .make() method
     *      returning @c expected<value_t> to enable exception-free construction.
     */
    template <typename... args_types_>
    status_t emplace_back(args_types_ &&...args) noexcept {
        if (size_ >= capacity_) {
            auto status = reserve(size_ + 1);
            if (failed(status)) return status;
        }

        // Forward to assume_reserved variant - status propagates through
        auto status = emplace_back(assume_reserved, std::forward<args_types_>(args)...);
        return status;
    }

    /**
     *  @brief Resizes the vector to contain @p new_size elements. If @p new_size < @p size(),
     *      elements are destroyed. If @p new_size > @p size(), new elements are
     *      default-constructed.
     *
     *  @param[in] new_size The new size.
     *  @return Success, or error code on failure. On failure the elements are unchanged, though the
     *      capacity may already have grown.
     */
    status_t resize(std::size_t new_size) noexcept {
        static_assert(std::is_nothrow_default_constructible_v<value_t>,
                      "resize default-constructs in noexcept code, so the element must construct without throwing");

        // Shrink: destroy excess elements
        if (new_size < size_) {
            for (std::size_t index = new_size; index < size_; ++index) data_[index].~value_t();
            size_ = new_size;
            return success_k;
        }
        // Grow: ensure capacity and default-construct new elements
        else if (new_size > size_) {
            if (new_size > capacity_) {
                auto status = reserve(new_size);
                if (failed(status)) return status;
            }
            // Default-construct new elements
            for (std::size_t index = size_; index < new_size; ++index) new (&data_[index]) value_t();
            size_ = new_size;
        }
        return success_k;
    }

    /**
     *  @brief Resizes the vector and initializes new elements with @p value. If @p new_size >
     *      @p size(), new elements are copy-constructed from @p value.
     *
     *  @param[in] new_size The new size.
     *  @param[in] value Value to copy into new elements.
     *  @return Success, or error code on failure. On failure the elements are unchanged, though the
     *      capacity may already have grown.
     */
    status_t resize(std::size_t new_size, value_t const &value) noexcept {
        // Shrink: destroy excess elements
        if (new_size < size_) {
            for (std::size_t index = new_size; index < size_; ++index) data_[index].~value_t();
            size_ = new_size;
            return success_k;
        }
        // Grow: ensure capacity
        else if (new_size > size_) {
            if (new_size > capacity_) {
                auto status = reserve(new_size);
                if (failed(status)) return status;
            }
            // Copy-construct new elements (with rollback on failure)
            std::size_t old_size = size_;
            for (std::size_t index = old_size; index < new_size; ++index) {
                auto copy_result = copy_safely(value);
                // Rollback: destroy partially constructed elements
                if (!copy_result) {
                    for (std::size_t built = old_size; built < index; ++built) data_[built].~value_t();
                    return copy_result.status();
                }
                new (&data_[index]) value_t(std::move(*copy_result));
            }
            size_ = new_size;
        }
        return success_k;
    }

    /** Destroys all elements but keeps allocated capacity. */
    void clear() noexcept {
        // Destroy all elements
        for (std::size_t index = 0; index < size_; ++index) data_[index].~value_t();
        size_ = 0;
    }

    /**
     *  @brief Swaps contents with another vector.
     *  @param[inout] other The vector to swap with.
     *  @return Success, or @c invalid_argument_k if allocators are incompatible.
     *
     *  @note If @c propagate_on_container_swap is false (e.g., @c std::allocator), allocators must
     *      compare equal. Attempting to swap vectors with unequal non-propagating allocators
     *      returns @c invalid_argument_k and leaves both vectors unchanged.
     */
    status_t swap(basic_vector &other) noexcept {
        // For non-propagating allocators, they must be equal (C++ standard requirement)
        if constexpr (!std::allocator_traits<allocator_t>::propagate_on_container_swap::value)
            if (!(allocator_ == other.allocator_)) return invalid_argument_k;

        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);

        // Only swap allocators if they propagate on swap
        if constexpr (std::allocator_traits<allocator_t>::propagate_on_container_swap::value)
            std::swap(allocator_, other.allocator_);

        return success_k;
    }

#pragma region Element Access

    /**
     *  @brief Accesses element at @p index without bounds checking.
     *  @param[in] index The element index.
     *  @return Reference to the element.
     */
    value_t &operator[](std::size_t index) noexcept {
        assert(index < size_ && "Index out of bounds");
        return data_[index];
    }

    /**
     *  @brief Accesses element at @p index without bounds checking (const version).
     *  @param[in] index The element index.
     *  @return Const reference to the element.
     */
    value_t const &operator[](std::size_t index) const noexcept {
        assert(index < size_ && "Index out of bounds");
        return data_[index];
    }

    /**
     *  @brief Accesses element at @p index with bounds checking.
     *  @param[in] index The element index.
     *  @return Pointer to the element, or @c nullptr if out of bounds.
     */
    value_t *at(std::size_t index) noexcept { return index < size_ ? &data_[index] : nullptr; }

    /**
     *  @brief Accesses element at @p index with bounds checking (const version).
     *  @param[in] index The element index.
     *  @return Const pointer to the element, or @c nullptr if out of bounds.
     */
    value_t const *at(std::size_t index) const noexcept { return index < size_ ? &data_[index] : nullptr; }

    /**
     *  @brief Accesses the first element.
     *  @return Reference to the first element.
     */
    value_t &front() noexcept {
        assert(size_ > 0 && "front() requires non-empty vector");
        return data_[0];
    }

    /**
     *  @brief Accesses the first element (const version).
     *  @return Const reference to the first element.
     */
    value_t const &front() const noexcept {
        assert(size_ > 0 && "front() requires non-empty vector");
        return data_[0];
    }

    /**
     *  @brief Accesses the last element.
     *  @return Reference to the last element.
     */
    value_t &back() noexcept {
        assert(size_ > 0 && "back() requires non-empty vector");
        return data_[size_ - 1];
    }

    /**
     *  @brief Accesses the last element (const version).
     *  @return Const reference to the last element.
     */
    value_t const &back() const noexcept {
        assert(size_ > 0 && "back() requires non-empty vector");
        return data_[size_ - 1];
    }

    /**
     *  @brief Returns pointer to underlying array.
     *  @return Pointer to the underlying element storage.
     */
    value_t *data() noexcept { return data_; }

    /**
     *  @brief Returns pointer to underlying array (const version).
     *  @return Const pointer to the underlying element storage.
     */
    value_t const *data() const noexcept { return data_; }

#pragma endregion Element Access

#pragma region Capacity

    /**
     *  @brief Checks whether the vector is empty.
     *  @return @c true if the vector is empty, @c false otherwise.
     */
    bool empty() const noexcept { return size_ == 0; }

    /**
     *  @brief Returns the number of elements.
     *  @return The number of elements in the vector.
     */
    std::size_t size() const noexcept { return size_; }

    /**
     *  @brief Returns the capacity.
     *  @return The number of elements that can be held in currently allocated storage.
     */
    std::size_t capacity() const noexcept { return capacity_; }

    /** A copy of this vector's allocator, for building a sibling that allocates the same way. */
    [[nodiscard]] allocator_t get_allocator() const noexcept { return allocator_; }

#pragma endregion Capacity

#pragma region Iterators

    /** Returns an iterator to the beginning. */
    value_t *begin() noexcept { return data_; }

    /** Returns an iterator to the end. */
    value_t *end() noexcept { return data_ + size_; }

    /** Returns a const iterator to the beginning. */
    value_t const *begin() const noexcept { return data_; }

    /** Returns a const iterator to the end. */
    value_t const *end() const noexcept { return data_ + size_; }

#pragma endregion Iterators
};

} // namespace ashvardanian::smashtable
