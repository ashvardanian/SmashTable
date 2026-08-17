/**
 *  @brief Exception-free dynamic array with explicit error handling.
 *  @author Ash Vardanian
 *  @file include/smashtable/basic_vector.hpp
 *  @date October 20, 2025
 *
 *  @section basic_vector_features Features
 *
 *  Unlike @c std::vector, this implementation:
 *  - Never throws exceptions - returns @c status_t or @c expected<T> for fallible operations
 *  - For both @c noexcept-constructible elements and ones with fallible @c .make() methods
 *  - Offers performance variants with @c assume_reserved tag for pre-checked hot paths
 *  - Guarantees atomicity for batch operations like @c resize()
 *
 *  @section basic_vector_requirements Requirements
 *
 *  @par Element Type
 *  - Nothrow default-constructible and nothrow move constructible/assignable (required)
 *  - For @c emplace_back(): Nothrow constructible OR provides
 *      @code ::make(...) -> expected<T> @endcode
 *  - For @c copy() & @c resize(): Nothrow copy-constructible OR provides
 *      @code .copy() const -> @c expected<T> @endcode
 *
 *  @par Allocator Type
 *  - Must propagate on move assignment @c propagate_on_container_move_assignment==true
 *  - For @c swap(): If non-propagating, both vectors must use equal allocators,
 *    otherwise @c invalid_argument_k is returned
 *
 *
 *  @see https://en.wikipedia.org/wiki/Vector_(C%2B%2B)
 *  @see https://en.cppreference.com/w/cpp/container/vector
 */
#pragma once
#include <cassert> // `assert`

#include <memory>  // `std::allocator`
#include <utility> // `std::exchange`, `std::forward`

#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief Exception-free dynamic array that uses error codes instead of exceptions.
 *    Provides RAII memory management with explicit failure handling, unlike @c std::vector.
 *
 *  @par Template requirements
 *  - @p element_type_ must be nothrow default-constructible and nothrow move constructible/assignable.
 *  - For types with potentially throwing constructors, provide a static @c .make() method
 *    returning @c expected<element_type_> to enable exception-free construction via @c emplace_back().
 *
 *  @see https://en.cppreference.com/w/cpp/container/vector
 */
template <typename element_type_, typename allocator_type_ = std::allocator<element_type_>>
class basic_vector {
  public:
    using element_t = element_type_;
    using allocator_t = allocator_type_;

    // Allocator propagation requirement for exception-free move assignment
    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "basic_vector requires allocators that propagate on move assignment");

  private:
    element_t *data_ {nullptr};
    std::size_t size_ {0};
    std::size_t capacity_ {0};
    ST_NO_UNIQUE_ADDRESS_ allocator_t allocator_ {};

  public:
    /**
     *  @brief Default constructor creates an empty vector.
     */
    basic_vector() noexcept = default;

    /**
     *  @brief Constructor with custom allocator.
     *  @param[in] allocator The allocator instance to use.
     */
    explicit basic_vector(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}

    /**
     *  @brief Destructor deallocates memory and destroys all elements.
     */
    ~basic_vector() noexcept {
        // Destroy all constructed elements
        for (std::size_t i = 0; i < size_; ++i) data_[i].~element_t();
        // Deallocate memory
        if (data_) allocator_.deallocate(data_, capacity_);
    }

    /**
     *  @brief Move constructor transfers ownership.
     */
    basic_vector(basic_vector &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          capacity_(std::exchange(other.capacity_, 0)), allocator_(std::move(other.allocator_)) {}

    /**
     *  @brief Move assignment transfers ownership after destroying current elements.
     */
    basic_vector &operator=(basic_vector &&other) noexcept {
        if (this != &other) {
            // Destroy current elements
            for (std::size_t i = 0; i < size_; ++i) data_[i].~element_t();
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

    /**
     *  @brief Copy construction is deleted (use @c .copy() method for explicit copies).
     */
    basic_vector(basic_vector const &) = delete;

    /**
     *  @brief Copy assignment is deleted (use @c .copy() method for explicit copies).
     */
    basic_vector &operator=(basic_vector const &) = delete;

    /**
     *  @brief Creates a new vector with specified capacity.
     *  @param[in] initial_capacity Initial capacity to allocate.
     *  @return @c expected<basic_vector> containing the new vector, or error status.
     */
    [[nodiscard]] static expected<basic_vector> make(std::size_t initial_capacity) noexcept {
        return make(initial_capacity, allocator_t {});
    }

    /**
     *  @brief Creates a new vector with specified capacity and allocator.
     *  @param[in] initial_capacity Initial capacity to allocate.
     *  @param[in] allocator The allocator instance to use.
     *  @return @c expected<basic_vector> containing the new vector, or error status.
     */
    [[nodiscard]] static expected<basic_vector> make(std::size_t initial_capacity, allocator_t allocator) noexcept {
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
    [[nodiscard]] expected<basic_vector> copy() const noexcept {
        basic_vector result(allocator_);
        if (capacity_ > 0) {
            auto status = result.reserve(capacity_);
            if (failed(status)) return expected<basic_vector>(basic_vector(allocator_), status);
        }
        // Copy all elements using copy_safely
        for (std::size_t i = 0; i < size_; ++i) {
            auto copy_result = copy_safely(data_[i]);
            if (!copy_result) {
                // Clean up partially constructed elements
                for (std::size_t j = 0; j < result.size_; ++j) result.data_[j].~element_t();
                result.size_ = 0;
                return expected<basic_vector>(basic_vector(allocator_), copy_result.status());
            }
            new (&result.data_[result.size_++]) element_t(std::move(*copy_result));
        }
        return expected<basic_vector>(std::move(result), success_k);
    }

    /**
     *  @brief Reserves capacity for at least @p new_capacity elements.
     *    Grows by at least a doubling, so a caller asking for one more slot at a time
     *    still amortizes to constant reallocation cost.
     *  @param[in] new_capacity The new capacity.
     *  @return Success, or @c out_of_memory_heap_k if allocation fails.
     */
    [[nodiscard]] status_t reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return success_k;

        std::size_t const grown_capacity = larger_of<std::size_t>(new_capacity, capacity_ == 0 ? 4 : capacity_ * 2);
        auto new_data = allocator_.allocate(grown_capacity);
        if (!new_data) return out_of_memory_heap_k;

        // Move existing elements to new storage
        for (std::size_t i = 0; i < size_; ++i) new (&new_data[i]) element_t(std::move(data_[i]));

        // Destroy moved-from elements in old storage
        for (std::size_t i = 0; i < size_; ++i) data_[i].~element_t();

        if (data_) allocator_.deallocate(data_, capacity_);
        data_ = new_data;
        capacity_ = grown_capacity;
        return success_k;
    }

    /**
     *  @brief Appends an element without capacity check (performance variant).
     *    Requires pre-reserved capacity. Use for hot paths after @c reserve().
     *
     *  @param[in] assume_reserved Tag indicating capacity was pre-reserved.
     *  @param[in] value Element to append (moved into the vector).
     *  @return Always returns success for noexcept move construction.
     */
    [[nodiscard]] status_t push_back(assume_reserved_t, element_t &&value) noexcept {
        assert(size_ < capacity_ && "push_back with assume_reserved requires pre-reserved capacity");
        new (&data_[size_++]) element_t(std::move(value));
        return success_k;
    }

    /**
     *  @brief Appends an element to the end of the vector.
     *    Automatically grows capacity using 2x strategy if needed.
     *
     *  @param[in] value Element to append (moved into the vector).
     *  @return Success, or @c out_of_memory_heap_k if reallocation fails.
     */
    [[nodiscard]] status_t push_back(element_t &&value) noexcept {
        if (size_ >= capacity_) {
            auto status = reserve(size_ + 1);
            if (failed(status)) return status;
        }
        return push_back(assume_reserved, std::move(value));
    }

    /**
     *  @brief Removes the last element from the vector.
     *    Precondition: Vector must not be empty.
     */
    void pop_back() noexcept {
        assert(size_ > 0 && "pop_back requires non-empty vector");
        data_[--size_].~element_t();
    }

    /**
     *  @brief Constructs element in-place without capacity check (performance variant).
     *    Requires pre-reserved capacity. Use for hot paths after @c reserve().
     *
     *  @tparam args_types_ Types of arguments to forward to element constructor.
     *  @param[in] assume_reserved Tag indicating capacity was pre-reserved.
     *  @param[in] args Arguments to forward to element constructor.
     *  @return Success, or error from @c .make() method if construction can throw.
     */
    template <typename... args_types_>
    [[nodiscard]] status_t emplace_back(assume_reserved_t, args_types_ &&...args) noexcept {
        assert(size_ < capacity_ && "emplace_back with assume_reserved requires pre-reserved capacity");

        // Fast path: noexcept constructor - construct directly in place
        if constexpr (std::is_nothrow_constructible_v<element_t, args_types_...>) {
            new (&data_[size_++]) element_t(std::forward<args_types_>(args)...);
            return success_k;
        }
        // Slow path: potentially throwing constructor - use .make() method
        else if constexpr (has_make_method<element_t, args_types_...>) {
            auto result = element_t::make(std::forward<args_types_>(args)...);
            if (failed(result)) return result.status();
            new (&data_[size_++]) element_t(std::move(*result));
            return success_k;
        }
        else {
            static_assert(std::is_nothrow_constructible_v<element_t, args_types_...> ||
                              has_make_method<element_t, args_types_...>,
                          "Type must be nothrow constructible or provide a static .make(...)");
            return status_t::unknown_k;
        }
    }

    /**
     *  @brief Constructs element in-place at the end of the vector.
     *    Automatically grows capacity using 2x strategy if needed.
     *
     *  @tparam args_types_ Types of arguments to forward to element constructor.
     *  @param[in] args Arguments to forward to element constructor.
     *  @return Success, or @c out_of_memory_heap_k if reallocation fails.
     *
     *  @note For types with potentially throwing constructors, provide a static @c .make() method
     *    returning @c expected<element_t> to enable exception-free construction.
     */
    template <typename... args_types_>
    [[nodiscard]] status_t emplace_back(args_types_ &&...args) noexcept {
        if (size_ >= capacity_) {
            auto status = reserve(size_ + 1);
            if (failed(status)) return status;
        }

        // Forward to assume_reserved variant - status propagates through
        auto status = emplace_back(assume_reserved, std::forward<args_types_>(args)...);
        return status;
    }

    /**
     *  @brief Resizes the vector to contain @p new_size elements.
     *    If @p new_size < @p size(), elements are destroyed. If @p new_size > @p size(),
     *    new elements are default-constructed.
     *
     *  @param[in] new_size The new size.
     *  @return Success, or error code on failure.
     *    On failure, the vector is unchanged (strong exception guarantee).
     */
    [[nodiscard]] status_t resize(std::size_t new_size) noexcept {
        // Shrink: destroy excess elements
        if (new_size < size_) {
            for (std::size_t i = new_size; i < size_; ++i) data_[i].~element_t();
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
            for (std::size_t i = size_; i < new_size; ++i) new (&data_[i]) element_t();
            size_ = new_size;
        }
        return success_k;
    }

    /**
     *  @brief Resizes the vector and initializes new elements with @p value.
     *    If @p new_size > @p size(), new elements are copy-constructed from @p value.
     *
     *  @param[in] new_size The new size.
     *  @param[in] value Value to copy into new elements.
     *  @return Success, or error code on failure.
     *    On failure, the vector is unchanged (strong exception guarantee).
     */
    [[nodiscard]] status_t resize(std::size_t new_size, element_t const &value) noexcept {
        // Shrink: destroy excess elements
        if (new_size < size_) {
            for (std::size_t i = new_size; i < size_; ++i) data_[i].~element_t();
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
            for (std::size_t i = old_size; i < new_size; ++i) {
                auto copy_result = copy_safely(value);
                // Rollback: destroy partially constructed elements
                if (!copy_result) {
                    for (std::size_t j = old_size; j < i; ++j) data_[j].~element_t();
                    return copy_result.status();
                }
                new (&data_[i]) element_t(std::move(*copy_result));
            }
            size_ = new_size;
        }
        return success_k;
    }

    /**
     *  @brief Destroys all elements but keeps allocated capacity.
     */
    void clear() noexcept {
        // Destroy all elements
        for (std::size_t i = 0; i < size_; ++i) data_[i].~element_t();
        size_ = 0;
    }

    /**
     *  @brief Swaps contents with another vector.
     *  @param[in,out] other The vector to swap with.
     *  @return Success, or @c invalid_argument_k if allocators are incompatible.
     *
     *  @note If @c propagate_on_container_swap is false (e.g., @c std::allocator), allocators
     *    must compare equal. Attempting to swap vectors with unequal non-propagating allocators
     *    returns @c invalid_argument_k and leaves both vectors unchanged.
     */
    [[nodiscard]] status_t swap(basic_vector &other) noexcept {
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
    element_t &operator[](std::size_t index) noexcept {
        assert(index < size_ && "Index out of bounds");
        return data_[index];
    }

    /**
     *  @brief Accesses element at @p index without bounds checking (const version).
     *  @param[in] index The element index.
     *  @return Const reference to the element.
     */
    element_t const &operator[](std::size_t index) const noexcept {
        assert(index < size_ && "Index out of bounds");
        return data_[index];
    }

    /**
     *  @brief Accesses element at @p index with bounds checking.
     *  @param[in] index The element index.
     *  @return Pointer to the element, or @c nullptr if out of bounds.
     */
    element_t *at(std::size_t index) noexcept { return index < size_ ? &data_[index] : nullptr; }

    /**
     *  @brief Accesses element at @p index with bounds checking (const version).
     *  @param[in] index The element index.
     *  @return Const pointer to the element, or @c nullptr if out of bounds.
     */
    element_t const *at(std::size_t index) const noexcept { return index < size_ ? &data_[index] : nullptr; }

    /**
     *  @brief Accesses the first element.
     *  @return Reference to the first element.
     */
    element_t &front() noexcept {
        assert(size_ > 0 && "front() requires non-empty vector");
        return data_[0];
    }

    /**
     *  @brief Accesses the first element (const version).
     *  @return Const reference to the first element.
     */
    element_t const &front() const noexcept {
        assert(size_ > 0 && "front() requires non-empty vector");
        return data_[0];
    }

    /**
     *  @brief Accesses the last element.
     *  @return Reference to the last element.
     */
    element_t &back() noexcept {
        assert(size_ > 0 && "back() requires non-empty vector");
        return data_[size_ - 1];
    }

    /**
     *  @brief Accesses the last element (const version).
     *  @return Const reference to the last element.
     */
    element_t const &back() const noexcept {
        assert(size_ > 0 && "back() requires non-empty vector");
        return data_[size_ - 1];
    }

    /**
     *  @brief Returns pointer to underlying array.
     *  @return Pointer to the underlying element storage.
     */
    element_t *data() noexcept { return data_; }

    /**
     *  @brief Returns pointer to underlying array (const version).
     *  @return Const pointer to the underlying element storage.
     */
    element_t const *data() const noexcept { return data_; }

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

#pragma endregion Capacity

#pragma region Iterators

    /**
     *  @brief Returns an iterator to the beginning.
     */
    element_t *begin() noexcept { return data_; }

    /**
     *  @brief Returns an iterator to the end.
     */
    element_t *end() noexcept { return data_ + size_; }

    /**
     *  @brief Returns a const iterator to the beginning.
     */
    element_t const *begin() const noexcept { return data_; }

    /**
     *  @brief Returns a const iterator to the end.
     */
    element_t const *end() const noexcept { return data_ + size_; }
};

#pragma endregion Iterators

} // namespace ashvardanian::smashtable
