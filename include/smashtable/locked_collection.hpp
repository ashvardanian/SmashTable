/**
 *  @brief Wraps any transactional collection behind one shared mutex, making the collection thread-safe while its
 *      transactions stay single-threaded.
 *  @author Ash Vardanian
 *  @file include/smashtable/locked_collection.hpp
 *  @date October 13, 2022
 */
#pragma once
#include <mutex>        // `std::unique_lock`
#include <optional>     // `std::optional`
#include <shared_mutex> // `std::shared_mutex`

#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief Wraps and protects any "Consistent Set" under a shared mutex.
 *
 *  The collection becomes @b thread-safe; a single transaction does @b not - it belongs to the thread
 *  that opened it, and its staged changes live outside the mutex entirely. Only the points where a
 *  transaction reaches the store - @c watch, @c stage, @c commit, @c rollback, @c find - take the lock.
 *
 *  One mutex spans the whole commit loop, which is what keeps the isolation level its parts promise.
 */
template <typename collection_type_, typename shared_mutex_type_ = std::shared_mutex>
class locked_collection {

  public:
    using locked_t = locked_collection;
    using unlocked_t = collection_type_;
    using unlocked_transaction_t = typename unlocked_t::transaction_t;
    using shared_mutex_t = shared_mutex_type_;

    using value_t = typename unlocked_t::value_t;
    using value_type = value_t; // ? STL style
    using key_type = typename mapping_key_type_or_itself<value_t>::type;
    using mapped_type = typename mapped_value_type_or_void<value_t>::type;
    using is_associative = std::bool_constant<is_mapping<value_t>>;
    using is_transactional = std::true_type;
    using callback_reads = std::true_type;

    /** @brief One mutex serializes whole transactions, so the inner store's promise carries over intact. */
    static constexpr isolation_t isolation_k = unlocked_t::isolation_k;

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
        [[nodiscard]] status_t upsert(value_t &&element) noexcept { return unlocked_.upsert(std::move(element)); }
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

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
            std::shared_lock _ {store_.mutex_};
            unlocked_.find(std::forward<comparable_type_>(comparable),
                           std::forward<callback_found_type_>(callback_found),
                           std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Copies out the member equal to @p comparable, including this transaction's writes. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            find(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
            return result;
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept {
            std::shared_lock _ {store_.mutex_};
            unlocked_.upper_bound(std::forward<comparable_type_>(comparable),
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
    locked_collection() noexcept = default;
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

    [[nodiscard]] status_t upsert(value_t &&element) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.upsert(std::forward<value_t>(element));
    }

    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.upsert(begin, end);
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept {
        std::shared_lock _ {mutex_};
        unlocked_.find(std::forward<comparable_type_>(comparable), std::forward<callback_found_type_>(callback_found),
                       std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Existence check, expressed through @c find so the lock discipline stays in one place. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
        bool found = false;
        find(
            std::forward<comparable_type_>(comparable), [&](value_t const &) noexcept { found = true; },
            []() noexcept {});
        return found;
    }

    /** @brief Number of elements matching @p comparable, which is 0 or 1 for unique keys. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t count(comparable_type_ &&comparable) const noexcept {
        return contains(std::forward<comparable_type_>(comparable)) ? 1u : 0u;
    }

    /** @brief Inserts a batch, leaving already-present keys untouched. */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t insert_if_missing(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.insert_if_missing(begin, end);
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept {
        std::shared_lock _ {mutex_};
        unlocked_.lower_bound(std::forward<comparable_type_>(comparable),
                              std::forward<callback_found_type_>(callback_found),
                              std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Copies out the member equal to @p comparable, or reports @c key_not_found_k. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        find(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    /** @brief Copies out the first element ordered at or after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    /** @brief Copies out the first element ordered strictly after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        upper_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept {
        std::shared_lock _ {mutex_};
        unlocked_.upper_bound(std::forward<comparable_type_>(comparable),
                              std::forward<callback_found_type_>(callback_found),
                              std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        std::shared_lock _ {mutex_};
        unlocked_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                        std::forward<callback_type_>(callback));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        std::unique_lock _ {mutex_};
        unlocked_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                        std::forward<callback_type_>(callback));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) noexcept {
        std::unique_lock _ {mutex_};
        unlocked_.erase_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
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

    template <typename lower_type_, typename upper_type_, typename generator_type_,
              typename callback_type_ = no_op_fn_t>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                      callback_type_ &&callback) const noexcept {
        std::shared_lock _ {mutex_};
        unlocked_.sample_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                               std::forward<generator_type_>(generator), std::forward<callback_type_>(callback));
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator, std::size_t &seen,
                      std::size_t reservoir_capacity, output_iterator_type_ &&reservoir) const noexcept {
        std::shared_lock _ {mutex_};
        unlocked_.sample_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                               std::forward<generator_type_>(generator), seen, reservoir_capacity,
                               std::forward<output_iterator_type_>(reservoir));
    }
};

} // namespace ashvardanian::smashtable
