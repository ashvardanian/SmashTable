#pragma once
#include <shared_mutex> // `std::shared_mutex`

namespace av {

/**
 * @brief
 * The collection itself becomes @b thread-safe, but the transaction don't!
 * Detects dead-locks and reports `operation_would_block`.
 */
template <typename collection_at>
class locked_gt {

  public:
    using locked_t = locked_gt;
    using unlocked_t = collection_at;
    using unlocked_transaction_t = typename unlocked_t::transaction_t;

    using element_t = typename unlocked_t::element_t;
    using comparator_t = typename unlocked_t::comparator_t;
    using identifier_t = typename unlocked_t::identifier_t;
    using generation_t = typename unlocked_t::generation_t;
    using status_t = typename unlocked_t::status_t;

    class transaction_t {
        friend locked_gt;
        locked_gt& store_;
        unlocked_transaction_t unlocked_;
        static_assert(std::is_nothrow_move_constructible<unlocked_transaction_t>());

      public:
        transaction_t(locked_gt& db, unlocked_transaction_t&& unlocked) noexcept
            : store_(db), unlocked_(std::move(unlocked)) {}
        transaction_t(transaction_t&&) noexcept = default;
        transaction_t& operator=(transaction_t&&) noexcept = default;

        [[nodiscard]] status_t watch(identifier_t const& id) noexcept { return unlocked_.watch(id); }
        [[nodiscard]] status_t upsert(element_t&& element) noexcept { return unlocked_.upsert(std::move(element)); }
        [[nodiscard]] status_t erase(identifier_t const& id) noexcept { return unlocked_.erase(id); }

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

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t find(comparable_at&& comparable,
                                    callback_found_at&& callback_found,
                                    callback_missing_at&& callback_missing = {}) const noexcept {
            std::shared_lock _ {store_.mutex_};
            return unlocked_.find(std::forward<comparable_at>(comparable),
                                  std::forward<callback_found_at>(callback_found),
                                  std::forward<callback_missing_at>(callback_missing));
        }

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t find_next(comparable_at&& comparable,
                                         callback_found_at&& callback_found,
                                         callback_missing_at&& callback_missing = {}) const noexcept {
            std::shared_lock _ {store_.mutex_};
            return unlocked_.find_next(std::forward<comparable_at>(comparable),
                                       std::forward<callback_found_at>(callback_found),
                                       std::forward<callback_missing_at>(callback_missing));
        }
    };

  private:
    mutable std::shared_mutex mutex_;
    unlocked_t unlocked_;

    locked_gt(unlocked_t&& unlocked) noexcept : unlocked_(std::move(unlocked)) {}
    locked_gt& operator=(locked_gt&& other) noexcept {
        std::unique_lock _ {mutex_};
        unlocked_ = std::move(other.unlocked_);
        return *this;
    }

  public:
    locked_gt(locked_gt&& other) noexcept : unlocked_(std::move(other.unlocked_)) {}

    [[nodiscard]] std::size_t size() const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.size();
    }

    [[nodiscard]] static std::optional<locked_gt> make() noexcept {
        std::optional<locked_gt> result;
        if (std::optional<unlocked_t> unlocked = unlocked_t::make(); unlocked)
            result.emplace(locked_gt {std::move(unlocked).value()});
        return result;
    }

    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        std::optional<transaction_t> result;
        std::unique_lock _ {mutex_};
        if (auto unlocked = unlocked_.transaction(); unlocked)
            result.emplace(transaction_t {*this, std::move(unlocked).value()});
        return result;
    }

    [[nodiscard]] status_t upsert(element_t&& element) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.upsert(std::forward<element_t>(element));
    }

    template <typename elements_begin_at, typename elements_end_at = elements_begin_at>
    [[nodiscard]] status_t upsert(elements_begin_at begin, elements_end_at end) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.upsert(begin, end);
    }

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find(comparable_at&& comparable,
                                callback_found_at&& callback_found,
                                callback_missing_at&& callback_missing = {}) const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.find(std::forward<comparable_at>(comparable),
                              std::forward<callback_found_at>(callback_found),
                              std::forward<callback_missing_at>(callback_missing));
    }

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find_next(comparable_at&& comparable,
                                     callback_found_at&& callback_found,
                                     callback_missing_at&& callback_missing = {}) const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.find_next(std::forward<comparable_at>(comparable),
                                   std::forward<callback_found_at>(callback_found),
                                   std::forward<callback_missing_at>(callback_missing));
    }

    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t find_equals(comparable_at&& comparable, callback_at&& callback) const noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.find_equals(std::forward<comparable_at>(comparable), std::forward<callback_at>(callback));
    }

    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t find_equals(comparable_at&& comparable, callback_at&& callback) noexcept {
        std::shared_lock _ {mutex_};
        return unlocked_.find_equals(std::forward<comparable_at>(comparable), std::forward<callback_at>(callback));
    }

    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t erase_equals(comparable_at&& comparable, callback_at&& callback = {}) noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.erase_equals(std::forward<comparable_at>(comparable), std::forward<callback_at>(callback));
    }

    [[nodiscard]] status_t clear() noexcept {
        std::unique_lock _ {mutex_};
        return unlocked_.clear();
    }
};

} // namespace av