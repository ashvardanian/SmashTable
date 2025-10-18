#pragma once
#include <array>        // `std::array`
#include <optional>     // `std::optional`
#include <functional>   // `std::hash`
#include <shared_mutex> // `std::shared_mutex`
#include <atomic>

namespace ashvardanian::smashtable {

template <typename type_, std::size_t count_, std::size_t... sequence_>
constexpr std::array<type_, count_> move_to_array_impl(type_ (&a)[count_], std::index_sequence<sequence_...>) noexcept {
    return {{std::move(a[sequence_])...}};
}

/**
 *  @brief This is a slightly tweaked implementation of `std::to_array` coming in C++20.
 *  @see https://en.cppreference.com/w/cpp/container/array/to_array
 */
template <typename type_, std::size_t count_>
constexpr std::array<type_, count_> move_to_array(type_ (&a)[count_]) noexcept {
    return move_to_array_impl(a, std::make_index_sequence<count_> {});
}

/**
 *  @brief Takes a generator that produces `bool`-convertible and dereference-able objects like `std::optional`,
 *  and builds up fixed-size of array of such object, but only if all were successfully built. If type_ least one
 *  generator call fails, the entire resulting `std::optional` is returned to NULL state.
 */
template <typename type_, std::size_t count_, typename generator_type_>
static std::optional<std::array<type_, count_>> generate_array_safely(generator_type_ &&generator) noexcept {
    constexpr std::size_t count_k = count_;
    using element_t = type_;
    using raw_array_t = element_t[count_];
    char raw_parts_mem[count_k * sizeof(element_t)];
    element_t *raw_parts = reinterpret_cast<element_t *>(raw_parts_mem);
    for (std::size_t part_idx = 0; part_idx != count_k; ++part_idx) {

        if (auto new_part = generator(part_idx); new_part)
            new (raw_parts + part_idx) element_t(std::move(new_part).value());
        else {
            // Destruct all the previous parts.
            for (std::size_t destructed_idx = 0; destructed_idx != part_idx; ++destructed_idx)
                raw_parts[destructed_idx].~element_t();
            return {};
        }
    }

    return move_to_array<element_t, count_k>((raw_array_t &)raw_parts_mem);
}

/**
 * @brief Hashes inputs to route them into separate sets, which can
 * be concurrent, or have a separate state-full allocator attached.
 *
 * @tparam hash_type_ Keys that compare equal must have the same hashes.
 */
template <typename collection_type_, typename hash_type_ = std::hash<typename collection_type_::identifier_t>,
          typename shared_mutex_type_ = std::shared_mutex, std::size_t parts_ = 16>
class partitioned_gt {

  public:
    static constexpr std::size_t parts_k = parts_;
    using partitioned_t = partitioned_gt;
    using hash_t = hash_type_;
    using part_t = collection_type_;
    using part_transaction_t = typename part_t::transaction_t;
    using shared_mutex_t = shared_mutex_type_;
    using shared_lock_t = std::shared_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

    using mutexes_t = std::array<shared_mutex_t, parts_k>;
    using parts_t = std::array<part_t, parts_k>;
    using part_transactions_t = std::array<part_transaction_t, parts_k>;

    using element_t = typename part_t::element_t;
    using comparator_t = typename part_t::comparator_t;
    using identifier_t = typename part_t::identifier_t;
    using generation_t = typename part_t::generation_t;

  private:
    static std::size_t bucket(identifier_t const &id) noexcept { return hash_t {}(id) % parts_k; }

    template <typename lock_type_, typename mutexes_type_>
    static void lock_out_of_order(mutexes_type_ &mutexes) noexcept {
        status_t status;
        std::array<bool, parts_k> finished {false};
        std::size_t remaining_count = parts_k;

        constexpr bool make_shared = std::is_same<lock_type_, shared_lock_t>();
        constexpr bool make_unique = std::is_same<lock_type_, unique_lock_t>();
        static_assert(make_shared || make_unique);

    cycle:
        // We may need to cycle multiple times, attempting to acquire locks,
        // until the `remaining_count == 0`.
        // This can also be implemented via `do {} while`, but becomes to
        // cumbersome with more complex conditions in cases like `upper_bound`
        // implemented via `for_all_next_lookups`.
        for (std::size_t part_idx = 0; part_idx != parts_k; ++part_idx) {
            if (finished[part_idx]) continue;
            auto &mutex = mutexes[part_idx];
            if constexpr (make_unique) remaining_count -= finished[part_idx] = mutex.try_lock();
            else
                remaining_count -= finished[part_idx] = mutex.try_lock_shared();
        }

        if (remaining_count) goto cycle;
    }

    /**
     * @brief Walks around all the parts, trying to perform operations on them,
     * until all the tasks are exhausted.
     */
    template <typename lock_type_, typename parts_type_, typename mutexes_type_, typename callable_type_>
    static status_t for_all(parts_type_ &parts, mutexes_type_ &mutexes, callable_type_ &&callable) noexcept {
        status_t status;
        std::array<bool, parts_k> finished {false};
        std::size_t remaining_count = parts_k;

    cycle:
        for (std::size_t part_idx = 0; part_idx != parts_k; ++part_idx) {
            if (finished[part_idx]) continue;
            lock_type_ lock {mutexes[part_idx], std::try_to_lock_t {}};
            if (!lock) continue;

            auto &part = parts[part_idx];
            status = callable(part);
            if (!status) return status;

            finished[part_idx] = true;
            --remaining_count;
        }

        if (remaining_count) goto cycle;

        return status;
    }

    template <typename parts_type_, typename mutexes_type_, typename comparable_type_, typename callback_found_type_,
              typename callback_missing_type_>
    static status_t for_all_next_lookups(parts_type_ &parts, mutexes_type_ &mutexes, comparable_type_ &&comparable,
                                         callback_found_type_ &&callback_found,
                                         callback_missing_type_ &&callback_missing) noexcept {

        status_t status;
        std::array<bool, parts_k> finished;
        std::size_t remaining_count;
        identifier_t smallest_id;
        std::size_t smallest_idx;
        constexpr std::size_t not_found_idx = std::numeric_limits<std::size_t>::max();

    restart:
        smallest_idx = not_found_idx;
        remaining_count = parts_k;
        std::fill_n(finished.begin(), parts_k, false);

    cycle:
        for (std::size_t part_idx = 0; part_idx != parts_k; ++part_idx) {
            if (finished[part_idx]) continue;
            shared_lock_t lock {mutexes[part_idx], std::try_to_lock_t {}};
            if (!lock) continue;

            auto &part = parts[part_idx];
            status = part.upper_bound(comparable, [&](element_t const &element) {
                if (smallest_idx != not_found_idx && !comparator_t {}(element, smallest_id)) return;
                smallest_id = identifier_t(element);
                smallest_idx = part_idx;
            });
            if (!status) return status;

            finished[part_idx] = true;
            --remaining_count;
        }
        if (remaining_count) goto cycle;

        if (smallest_idx == not_found_idx) return invoke_safely(std::forward<callback_missing_type_>(callback_missing));

        // Unless the underlying the engine implements Snapshot Isolation,
        // the repeated lookup of the entry in the underlying store can fail
        // and we will have to `restart` all over.
        bool should_restart = false;
        status = parts[smallest_idx].find(smallest_id, std::forward<callback_found_type_>(callback_found),
                                          [&] { should_restart = true; });
        if (should_restart) goto restart;
        return status;
    }

  public:
    class transaction_t {
        friend class partitioned_gt;
        partitioned_gt &store_;
        part_transactions_t parts_;
        generation_t generation_;
        static_assert(std::is_nothrow_move_constructible<part_transaction_t>());

        template <typename callable_type_>
        status_t for_parts(callable_type_ &&callable) noexcept {
            return partitioned_t::for_all<unique_lock_t>(parts_, store_.mutexes_,
                                                         std::forward<callable_type_>(callable));
        }

      public:
        transaction_t(partitioned_gt &db, part_transactions_t &&unlocked) noexcept
            : store_(db), parts_(std::move(unlocked)), generation_(db.new_generation()) {}
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        generation_t generation() const noexcept { return generation_; }

        [[nodiscard]] status_t reset() noexcept {
            auto status = for_parts(std::mem_fn(&part_transaction_t::reset));
            if (status) generation_ = store_.new_generation();
            return status;
        }
        [[nodiscard]] status_t rollback() noexcept {
            auto status = for_parts(std::mem_fn(&part_transaction_t::rollback));
            if (status) generation_ = store_.new_generation();
            return status;
        }

        [[nodiscard]] status_t stage() noexcept { return for_parts(std::mem_fn(&part_transaction_t::stage)); }
        [[nodiscard]] status_t commit() noexcept { return for_parts(std::mem_fn(&part_transaction_t::commit)); }

        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            std::size_t part_idx = bucket(id);
            shared_lock_t _ {store_.mutexes_[part_idx]};
            return parts_[part_idx].watch(id);
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept {
            std::size_t part_idx = bucket(identifier_t(comparable));
            shared_lock_t _ {store_.mutexes_[part_idx]};
            return parts_[part_idx].find(std::forward<comparable_type_>(comparable),
                                         std::forward<callback_found_type_>(callback_found),
                                         std::forward<callback_missing_type_>(callback_missing));
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept {
            return partitioned_t::for_all_next_lookups(parts_, store_.mutexes_,
                                                       std::forward<comparable_type_>(comparable),
                                                       std::forward<callback_found_type_>(callback_found),
                                                       std::forward<callback_missing_type_>(callback_missing));
        }

        [[nodiscard]] status_t upsert(element_t &&element) noexcept {
            return parts_[bucket(identifier_t(element))].upsert(std::move(element));
        }

        [[nodiscard]] status_t erase(identifier_t const &id) noexcept { //
            return parts_[bucket(id)].erase(id);
        }
    };

  private:
    mutable mutexes_t mutexes_;
    parts_t parts_;
    std::atomic<generation_t> generation_;

    friend class transaction_t;

    partitioned_gt(parts_t &&unlocked) noexcept : parts_(std::move(unlocked)) {}
    partitioned_gt &operator=(partitioned_gt &&other) noexcept {
        lock_out_of_order<unique_lock_t>(mutexes_);
        parts_ = std::move(other.parts_);
        for (auto &mutex : mutexes_) mutex.unlock();
        return *this;
    }

    static std::optional<parts_t> new_parts() noexcept {
        return generate_array_safely<part_t, parts_k>([](std::size_t) { return part_t::make(); });
    }

    generation_t new_generation() noexcept { return ++generation_; }

  public:
    partitioned_gt(partitioned_gt &&other) noexcept : parts_(std::move(other.parts_)) {}

    [[nodiscard]] std::size_t size() const noexcept {
        std::size_t total = 0;
        lock_out_of_order<shared_lock_t>(mutexes_);
        for (auto const &part : parts_) total += parts_.size();
        for (auto &mutex : mutexes_) mutex.unlock_shared();
        return total;
    }

    [[nodiscard]] static std::optional<partitioned_gt> make() noexcept {
        std::optional<partitioned_gt> result;
        if (std::optional<parts_t> unlocked = new_parts(); unlocked)
            result.emplace(partitioned_gt {std::move(unlocked).value()});
        return result;
    }

    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        auto maybe = generate_array_safely<part_transaction_t, parts_k>(
            [&](std::size_t part_idx) { return parts_[part_idx].transaction(); });
        if (!maybe) return {};

        return transaction_t(*this, std::move(maybe).value());
    }

    [[nodiscard]] status_t upsert(element_t &&element) noexcept {
        std::size_t part_idx = bucket(identifier_t(element));
        unique_lock_t _ {mutexes_[part_idx]};
        return parts_[part_idx].upsert(std::move(element));
    }

    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        // This might be implemented more efficiently, but using
        // a transaction beneath looks like the most straightforward approach.
        auto maybe = transaction();
        if (!maybe) return {consistency_k};
        for (; begin != end; ++begin)
            if (auto status = maybe->upsert(*begin); !status) return status;
        if (auto status = maybe->stage(); !status) return status;
        return maybe->commit();
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept {
        std::size_t part_idx = bucket(identifier_t(comparable));
        shared_lock_t _ {mutexes_[part_idx]};
        return parts_[part_idx].find(std::forward<comparable_type_>(comparable),
                                     std::forward<callback_found_type_>(callback_found),
                                     std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept {
        return for_all_next_lookups(parts_, mutexes_, std::forward<comparable_type_>(comparable),
                                    std::forward<callback_found_type_>(callback_found),
                                    std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        lock_out_of_order<shared_lock_t>(mutexes_);
        status_t status;
        for (auto &part : parts_)
            if (status = part.range(lower, upper, callback); !status) break;
        for (auto &mutex : mutexes_) mutex.unlock_shared();
        return status;
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        lock_out_of_order<unique_lock_t>(mutexes_);
        status_t status;
        for (auto &part : parts_)
            if (status = part.range(lower, upper, callback); !status) break;
        for (auto &mutex : mutexes_) mutex.unlock_shared();
        return status;
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        lock_out_of_order<unique_lock_t>(mutexes_);
        status_t status;
        for (auto &part : parts_)
            if (status = part.erase_range(lower, upper, callback); !status) break;
        for (auto &mutex : mutexes_) mutex.unlock_shared();
        return status;
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                        callback_type_ &&callback) const noexcept {
        // ! Here the assumption is that every part will have a somewhat equal
        // ! number of entries that compare equal to the provided range.
        std::size_t part_idx = generator() % parts_k;
        shared_lock_t _ {mutexes_[part_idx]};
        return parts_[part_idx].sample_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                             std::forward<generator_type_>(generator),
                                             std::forward<callback_type_>(callback));
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    [[nodiscard]] status_t sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                        std::size_t &seen, std::size_t reservoir_capacity,
                                        output_iterator_type_ &&reservoir) const noexcept {
        // ! This function trades consistency for performance!
        return for_all<shared_lock_t>(parts_, mutexes_, [&](part_t const &part) noexcept {
            return part.sample_range(lower, upper, generator, seen, reservoir_capacity, reservoir);
        });
    }

    [[nodiscard]] status_t clear() noexcept {

        auto maybe = new_parts();
        if (!maybe) return {unknown_k};

        lock_out_of_order<unique_lock_t>(mutexes_);
        parts_ = std::move(maybe).value();
        for (auto &mutex : mutexes_) mutex.unlock();
        return {success_k};
    }
};

} // namespace ashvardanian::smashtable