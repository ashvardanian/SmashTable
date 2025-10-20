/**
 *  @brief
 *
 *  @file   locked_collection.hpp
 *  @author Ash Vardanian
 */
#pragma once
#include <mutex>        // `std::unique_lock`
#include <shared_mutex> // `std::shared_mutex`

namespace ashvardanian::smashtable {

/**
 *  @brief Wraps and protects any "Consistent Set" under a shared mutex.
 *
 *  The collection itself becomes @b thread-safe, but the transaction don't!
 *  Detects dead-locks and reports @c operation_would_block_k.
 */
template <typename collection_type_, typename shared_mutex_type_ = std::shared_mutex>
class locked_collection {

  public:
    using locked_t = locked_collection;
    using unlocked_t = collection_type_;
    using unlocked_transaction_t = typename unlocked_t::transaction_t;
    using shared_mutex_t = shared_mutex_type_;

    using element_t = typename unlocked_t::element_t;
    using comparator_t = typename unlocked_t::comparator_t;
    using identifier_t = typename unlocked_t::identifier_t;
    using generation_t = typename unlocked_t::generation_t;

    class transaction_t {
        friend class locked_collection;
        locked_collection &store_;
        unlocked_transaction_t unlocked_;
        static_assert(std::is_nothrow_move_constructible<unlocked_transaction_t>());

      public:
        transaction_t(locked_collection &db, unlocked_transaction_t &&unlocked) noexcept
            : store_(db), unlocked_(std::move(unlocked)) {}
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        generation_t generation() const noexcept { return unlocked_.generation(); }

        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            std::shared_lock _ {store_.mutex_};
            return unlocked_.watch(id);
        }

        [[nodiscard]] status_t reserve(std::size_t size) noexcept { return unlocked_.reserve(size); }
        [[nodiscard]] status_t upsert(element_t &&element) noexcept { return unlocked_.upsert(std::move(element)); }
        [[nodiscard]] status_t erase(identifier_t const &id) noexcept { return unlocked_.erase(id); }

        [[nodiscard]] status_t stage() noexcept {
            std::unique_lock _ {store_.mutex_};
            return unlocked_.stage();
        }

        [[nodiscard]] status_t reset() noexcept {
            std::unique_lock _ {store_.mutex_};
            return unlocked_.reset();
        }

        [[nodiscard]] status_t rollback() noexcept {
            std::unique_lock _ {store_.mutex_};
            return unlocked_.rollback();
        }

        [[nodiscard]] status_t commit() noexcept {
            std::unique_lock _ {store_.mutex_};
            return unlocked_.commit();
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept {
            std::shared_lock _ {store_.mutex_};
            return unlocked_.find(std::forward<comparable_type_>(comparable),
                                  std::forward<callback_found_type_>(callback_found),
                                  std::forward<callback_missing_type_>(callback_missing));
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept {
            std::shared_lock _ {store_.mutex_};
            return unlocked_.upper_bound(std::forward<comparable_type_>(comparable),
                                         std::forward<callback_found_type_>(callback_found),
                                         std::forward<callback_missing_type_>(callback_missing));
        }
    };

  private:
    mutable shared_mutex_t mutex_;
    unlocked_t unlocked_;

    locked_collection(unlocked_t &&unlocked) noexcept : unlocked_(std::move(unlocked)) {}
    locked_collection &operator=(locked_collection &&other) noexcept {
        std::unique_lock _ {mutex_};
        unlocked_ = std::move(other.unlocked_);
        return *this;
    }

  public:
    locked_collection(locked_collection &&other) noexcept : unlocked_(std::move(other.unlocked_)) {}

    [[nodiscard]] std::size_t size() const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.empty();
    }

    [[nodiscard]] static std::optional<locked_collection> make() noexcept {
        std::optional<locked_collection> result;
        if (std::optional<unlocked_t> unlocked = unlocked_t::make(); unlocked)
            result.emplace(locked_collection {std::move(unlocked).value()});
        return result;
    }

    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        std::optional<transaction_t> result;
        std::unique_lock _ {mutex_};
        if (auto unlocked = unlocked_.transaction(); unlocked)
            result.emplace(transaction_t {*this, std::move(unlocked).value()});
        return result;
    }

    [[nodiscard]] status_t upsert(element_t &&element) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.upsert(std::forward<element_t>(element));
    }

    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.upsert(begin, end);
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.find(std::forward<comparable_type_>(comparable),
                              std::forward<callback_found_type_>(callback_found),
                              std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.upper_bound(std::forward<comparable_type_>(comparable),
                                     std::forward<callback_found_type_>(callback_found),
                                     std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                               std::forward<callback_type_>(callback));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                               std::forward<callback_type_>(callback));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.erase_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                     std::forward<callback_type_>(callback));
    }

    [[nodiscard]] status_t clear() noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.clear();
    }

    [[nodiscard]] status_t reserve(std::size_t size) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.reserve(size);
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                        callback_type_ &&callback) const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.sample_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                      std::forward<generator_type_>(generator), std::forward<callback_type_>(callback));
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    [[nodiscard]] status_t sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                        std::size_t &seen, std::size_t reservoir_capacity,
                                        output_iterator_type_ &&reservoir) const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.sample_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                      std::forward<generator_type_>(generator), seen, reservoir_capacity,
                                      std::forward<output_iterator_type_>(reservoir));
    }
};

} // namespace ashvardanian::smashtable