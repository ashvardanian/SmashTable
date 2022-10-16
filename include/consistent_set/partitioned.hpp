#pragma once
#include <array>      // `std::array`
#include <functional> // `std::hash`

namespace av {

/**
 * @brief Hashes inputs to route them into separate sets, which can
 * be concurrent, or have a separate state-full allocator attached.
 */
template <typename collection_at,
          typename hash_at = std::hash<typename collection_at::identifier_t>,
          std::size_t parts_ak = 16>
class partitioned_gt {

  public:
    using partitioned_t = partitioned_gt;
    using hash_t = hash_at;
    using part_t = collection_at;
    using part_transaction_t = typename part_t::transaction_t;
    static constexpr std::size_t parts_k = parts_ak;

    using element_t = typename part_t::element_t;
    using comparator_t = typename part_t::comparator_t;
    using identifier_t = typename part_t::identifier_t;
    using generation_t = typename part_t::generation_t;
    using status_t = typename part_t::status_t;

    static std::size_t bucket(identifier_t const& id) noexcept { return hash_t {}(id) % parts_k; }

    class transaction_t {
        friend partitioned_gt;
        partitioned_gt& store_;
        std::array<part_transaction_t, parts_k> parts_;
        static_assert(std::is_nothrow_move_constructible<part_transaction_t>());

        /**
         * @brief Walks around all the parts, trying to perform operations on them,
         * until all the tasks are exhausted.
         */
        template <typename callable_at>
        status_t for_all(callable_at&& callable) noexcept {
            status_t status;
            std::array<bool, parts_k> finished {false};
            std::size_t remaining = parts_k;
            do {
                for (std::size_t part_idx = 0; part_idx != parts_k; ++part_idx) {
                    if (finished[part_idx])
                        continue;
                    std::unique_lock lock {store_.mutexes_[part_idx], std::try_to_lock_t {}};
                    if (!lock)
                        continue;

                    status = parts_.stage();
                    if (!status)
                        break;

                    finished[part_idx] = true;
                    --remaining;
                }
            } while (status && remaining);
            return status;
        }

      public:
        transaction_t(partitioned_gt& db, part_transaction_t&& unlocked) noexcept
            : store_(db), parts_(std::move(unlocked)) {}
        transaction_t(transaction_t&&) noexcept = default;
        transaction_t& operator=(transaction_t&&) noexcept = default;

        [[nodiscard]] status_t watch(identifier_t const& id) noexcept { return parts_[bucket(id)].watch(id); }
        [[nodiscard]] status_t upsert(element_t&& element) noexcept { return parts_[bucket(id)].upsert(element); }
        [[nodiscard]] status_t erase(identifier_t const& id) noexcept { return parts_[bucket(id)].erase(id); }
        [[nodiscard]] status_t stage() noexcept { return for_all(std::mem_fn(&part_transaction_t::stage)); }
        [[nodiscard]] status_t reset() noexcept { return for_all(std::mem_fn(&part_transaction_t::stage)); }
        [[nodiscard]] status_t rollback() noexcept { return for_all(std::mem_fn(&part_transaction_t::stage)); }
        [[nodiscard]] status_t commit() noexcept { return for_all(std::mem_fn(&part_transaction_t::stage)); }

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t find(comparable_at&& comparable,
                                    callback_found_at&& callback_found,
                                    callback_missing_at&& callback_missing = {}) const noexcept {
            std::shared_lock _ {store_.mutexes_[bucket(id)]};
            return parts_.find(std::forward<comparable_at>(comparable),
                               std::forward<callback_found_at>(callback_found),
                               std::forward<callback_missing_at>(callback_missing));
        }

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t find_next(comparable_at&& comparable,
                                         callback_found_at&& callback_found,
                                         callback_missing_at&& callback_missing = {}) const noexcept {
            std::shared_lock _ {store_.mutexes_[bucket(id)]};
            return parts_.find_next(std::forward<comparable_at>(comparable),
                                    std::forward<callback_found_at>(callback_found),
                                    std::forward<callback_missing_at>(callback_missing));
        }
    };

  private:
    mutable std::array<std::shared_mutex, parts_k> mutexes_;
    std::array<part_t, parts_k> parts_;

    partitioned_gt(part_t&& unlocked) noexcept : parts_(std::move(unlocked)) {}
    partitioned_gt& operator=(partitioned_gt&& other) noexcept {
        std::unique_lock _ {mutex_};
        parts_ = std::move(other.parts_);
        return *this;
    }

    std::array<part_t, parts_k> new_parts() const noexcept {
        std::array<part_t, parts_k> new_parts;
        for (auto& part : new_parts)
            if (auto new_part = part_t::make(); new_part)
                part = std::move(new_part).value();
        return parts_;
    }

  public:
    partitioned_gt(partitioned_gt&& other) noexcept : parts_(std::move(other.parts_)) {}

    [[nodiscard]] std::size_t size() const noexcept {
        std::shared_lock _ {mutex_};
        return parts_.size();
    }

    [[nodiscard]] static std::optional<partitioned_gt> make() noexcept {
        std::optional<partitioned_gt> result;
        if (std::optional<part_t> unlocked = part_t::make(); unlocked)
            result.emplace(partitioned_gt {std::move(unlocked).value()});
        return result;
    }

    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        std::optional<transaction_t> result;
        std::unique_lock _ {mutex_};
        if (auto unlocked = parts_.transaction(); unlocked)
            result.emplace(transaction_t {*this, std::move(unlocked).value()});
        return result;
    }

    [[nodiscard]] status_t upsert(element_t&& element) noexcept {
        std::unique_lock _ {mutexes_[bucket(element)]};
        return parts_.upsert(std::forward<element_t>(element));
    }

    template <typename elements_begin_at, typename elements_end_at = elements_begin_at>
    [[nodiscard]] status_t upsert(elements_begin_at begin, elements_end_at end) noexcept {
        std::unique_lock _ {mutex_};
        return parts_.upsert(begin, end);
    }

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find(comparable_at&& comparable,
                                callback_found_at&& callback_found,
                                callback_missing_at&& callback_missing = {}) const noexcept {
        std::shared_lock _ {mutexes_[bucket(comparable)]};
        return parts_.find(std::forward<comparable_at>(comparable),
                           std::forward<callback_found_at>(callback_found),
                           std::forward<callback_missing_at>(callback_missing));
    }

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find_next(comparable_at&& comparable,
                                     callback_found_at&& callback_found,
                                     callback_missing_at&& callback_missing = {}) const noexcept {
        std::shared_lock _ {mutexes_[bucket(comparable)]};
        return parts_.find_next(std::forward<comparable_at>(comparable),
                                std::forward<callback_found_at>(callback_found),
                                std::forward<callback_missing_at>(callback_missing));
    }

    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t find_equals(comparable_at&& comparable, callback_at&& callback) const noexcept {
        std::shared_lock _ {mutexes_[bucket(comparable)]};
        return parts_.find_equals(std::forward<comparable_at>(comparable), std::forward<callback_at>(callback));
    }

    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t find_equals(comparable_at&& comparable, callback_at&& callback) noexcept {
        std::shared_lock _ {mutexes_[bucket(comparable)]};
        return parts_.find_equals(std::forward<comparable_at>(comparable), std::forward<callback_at>(callback));
    }

    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t erase_equals(comparable_at&& comparable, callback_at&& callback = {}) noexcept {
        std::unique_lock _ {mutex_};
        return parts_.erase_equals(std::forward<comparable_at>(comparable), std::forward<callback_at>(callback));
    }

    [[nodiscard]] status_t clear() noexcept {

        std::array<part_t, parts_k> new_parts;
        for (auto& part : new_parts)
            if (auto new_part = part_t::make(); new_part)
                part = std::move(new_part).value();

        for (auto& mutex : mutexes_)
            mutex.lock();
        auto parts_.clear();
    }
};

} // namespace av