/**
 *  @brief  Minimalistic growable array implementation.
 *    Not thread-safe by itself. Doesn't raise any exceptions unlike STL-based alternatives.
 *
 *  @file   basic_vector.hpp
 *  @author Ash Vardanian
 *  @see    https://en.wikipedia.org/wiki/Vector_(C%2B%2B)
 */
#pragma once
#include <cassert> // `assert`
#include <memory>  // `std::allocator`
#include <utility> // `std::exchange`

#include "status.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief Exception-free dynamic array that uses error codes instead of exceptions.
 *    Provides RAII memory management with explicit failure handling, unlike @c std::vector.
 *
 *  @see https://en.cppreference.com/w/cpp/container/vector
 */
template <typename element_type_, typename allocator_type_ = std::allocator<element_type_>>
class basic_vector {
  public:
    using element_t = element_type_;
    using allocator_t = allocator_type_;

  private:
    element_t *data_ {nullptr};
    std::size_t size_ {0};
    std::size_t capacity_ {0};
    [[no_unique_address]] allocator_t allocator_ {};

  public:
    basic_vector() noexcept = default;

    ~basic_vector() noexcept {
        if (data_) allocator_.deallocate(data_, capacity_);
    }

    basic_vector(basic_vector &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          capacity_(std::exchange(other.capacity_, 0)), allocator_(std::move(other.allocator_)) {}

    basic_vector &operator=(basic_vector &&other) noexcept {
        if (this != &other) {
            if (data_) allocator_.deallocate(data_, capacity_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            capacity_ = std::exchange(other.capacity_, 0);
            allocator_ = std::move(other.allocator_);
        }
        return *this;
    }

    basic_vector(basic_vector const &) = delete;
    basic_vector &operator=(basic_vector const &) = delete;

    [[nodiscard]] status_t try_reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return {success_k};

        auto new_data = allocator_.allocate(new_capacity);
        if (!new_data) return {out_of_memory_heap_k};

        // Move existing elements
        for (std::size_t i = 0; i < size_; ++i) new (&new_data[i]) element_t(std::move(data_[i]));

        if (data_) allocator_.deallocate(data_, capacity_);
        data_ = new_data;
        capacity_ = new_capacity;
        return {success_k};
    }

    [[nodiscard]] status_t try_push_back(element_t &&value) noexcept {
        // Auto-grow if needed (2x growth strategy)
        if (size_ >= capacity_) {
            std::size_t new_capacity = capacity_ == 0 ? 4 : capacity_ * 2;
            auto status = try_reserve(new_capacity);
            if (!status) return status;
        }
        new (&data_[size_++]) element_t(std::move(value));
        return {success_k};
    }

    void clear() noexcept { size_ = 0; }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }

    element_t *begin() noexcept { return data_; }
    element_t *end() noexcept { return data_ + size_; }
    element_t const *begin() const noexcept { return data_; }
    element_t const *end() const noexcept { return data_ + size_; }
};

} // namespace ashvardanian::smashtable