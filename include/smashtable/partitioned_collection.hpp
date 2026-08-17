/**
 *  @brief Shards a transactional collection across independently locked partitions, so writers touching different keys
 *      rarely contend.
 *  @author Ash Vardanian
 *  @file include/smashtable/partitioned_collection.hpp
 *  @date October 16, 2022
 */
#pragma once
#include <array> // `std::array`
#include <atomic>
#include <functional>   // `std::hash`
#include <mutex>        // `std::unique_lock`
#include <optional>     // `std::optional`
#include <shared_mutex> // `std::shared_mutex`, `std::shared_lock`

#include "shared.hpp"

namespace ashvardanian::smashtable {

template <typename type_, std::size_t count_, std::size_t... sequence_>
constexpr std::array<type_, count_> move_to_array_impl(type_ (&a)[count_], std::index_sequence<sequence_...>) noexcept {
    return {{std::move(a[sequence_])...}};
}

/**
 *  @brief This is a slightly tweaked implementation of @c std::to_array coming in C++20.
 *  @see https://en.cppreference.com/w/cpp/container/array/to_array
 */
template <typename type_, std::size_t count_>
constexpr std::array<type_, count_> move_to_array(type_ (&a)[count_]) noexcept {
    return move_to_array_impl(a, std::make_index_sequence<count_> {});
}

/**
 *  @brief Takes a generator that produces @c bool -convertible and dereference-able objects like @c std::optional,
 *    and builds up fixed-size of array of such object, but only if all were successfully built. If type_ least one
 *    generator call fails, the entire resulting @c std::optional is returned to NULL state.
 */
template <typename type_, std::size_t count_, typename generator_type_>
static std::optional<std::array<type_, count_>> generate_array_safely(generator_type_ &&generator) noexcept {
    constexpr std::size_t count_k = count_;
    using value_t = type_;
    using raw_array_t = value_t[count_];
    char raw_parts_mem[count_k * sizeof(value_t)];
    value_t *raw_parts = reinterpret_cast<value_t *>(raw_parts_mem);
    for (std::size_t part_idx = 0; part_idx != count_k; ++part_idx) {

        if (auto new_part = generator(part_idx); new_part)
            new (raw_parts + part_idx) value_t(std::move(new_part).value());
        else {
            // Destruct all the previous parts.
            for (std::size_t destructed_idx = 0; destructed_idx != part_idx; ++destructed_idx)
                raw_parts[destructed_idx].~value_t();
            return {};
        }
    }

    return move_to_array<value_t, count_k>((raw_array_t &)raw_parts_mem);
}

/**
 *  @brief Hashes inputs to route them into separate sets, which can
 *    be concurrent, or have a separate state-full allocator attached.
 *
 *  @tparam collection_type_ Type of the underlying collection, like @c transactional_std_store.
 *  @tparam hash_type_ Keys that compare equal must have the same hashes.
 *  @tparam shared_mutex_type_ Mutex type to use for partition locking, like @c std::shared_mutex.
 *  @tparam parts_count_ Number of partitions to split the collection into, default 16.
 */
template <typename collection_type_, typename hash_type_ = std::hash<typename collection_type_::identifier_t>,
          typename shared_mutex_type_ = std::shared_mutex, std::size_t parts_count_ = 16>
class partitioned_collection {

  public:
    static constexpr std::size_t parts_k = parts_count_;
    using partitioned_t = partitioned_collection;
    using hash_t = hash_type_;
    using part_t = collection_type_;
    using part_transaction_t = typename part_t::transaction_t;
    using shared_mutex_t = shared_mutex_type_;
    using shared_lock_t = std::shared_lock<shared_mutex_t>;
    using unique_lock_t = std::unique_lock<shared_mutex_t>;

    using mutexes_t = std::array<shared_mutex_t, parts_k>;
    using parts_t = std::array<part_t, parts_k>;
    using part_transactions_t = std::array<part_transaction_t, parts_k>;

    using value_t = typename part_t::value_t;
    using value_type = value_t; // ? STL style
    using key_type = typename mapping_key_type_or_itself<value_t>::type;
    using mapped_type = typename mapped_value_type_or_void<value_t>::type;
    using is_associative = std::bool_constant<is_mapping<value_t>>;
    using is_transactional = std::true_type;
    using callback_reads = std::true_type;

    using comparator_t = typename part_t::comparator_t;
    using identifier_t = typename part_t::identifier_t;
    using generation_t = typename part_t::generation_t;

  private:
    std::size_t bucket_(identifier_t const &id) const noexcept { return hasher_(id) % parts_k; }

    template <typename lock_type_, typename mutexes_type_>
    static void lock_out_of_order_(mutexes_type_ &mutexes) noexcept {
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
            if constexpr (make_unique) { remaining_count -= finished[part_idx] = mutex.try_lock(); }
            else { remaining_count -= finished[part_idx] = mutex.try_lock_shared(); }
        }

        if (remaining_count) goto cycle;
    }

    /**
     * @brief Walks around all the parts, trying to perform operations on them, until all the tasks are exhausted.
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
    static void for_all_next_lookups(comparator_t const &comparator, parts_type_ &parts, mutexes_type_ &mutexes,
                                     comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                     callback_missing_type_ &&callback_missing) noexcept {

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
            part.upper_bound(comparable, [&](value_t const &element) noexcept {
                if (smallest_idx != not_found_idx && !comparator(mapping_key_or_itself(element), smallest_id)) return;
                smallest_id = identifier_t(element);
                smallest_idx = part_idx;
            });

            finished[part_idx] = true;
            --remaining_count;
        }
        if (remaining_count) goto cycle;

        if (smallest_idx == not_found_idx) {
            callback_missing();
            return;
        }

        // Under "Read Committed" isolation, the element we found during `upper_bound` scanning
        // may have been deleted or modified by another transaction before we lock it for reading.
        // If the lookup fails (element deleted/changed), restart the entire scan to find the new minimum.
        // This is expected behavior for Read Committed - non-repeatable reads are allowed.
        bool should_restart = false;
        parts[smallest_idx].find(smallest_id, std::forward<callback_found_type_>(callback_found),
                                 [&]() noexcept { should_restart = true; });
        if (should_restart) goto restart;
    }

  public:
    class transaction_t {
        friend class partitioned_collection;
        partitioned_collection &store_;
        part_transactions_t parts_;
        generation_t generation_;
        std::array<bool, parts_k> dirty_ {}; // Track which partitions have been modified
        static_assert(std::is_nothrow_move_constructible<part_transaction_t>());

        template <typename callable_type_>
        status_t for_parts_(callable_type_ &&callable) noexcept {
            return partitioned_t::for_all<unique_lock_t>(parts_, store_.mutexes_,
                                                         std::forward<callable_type_>(callable));
        }

        template <typename callable_type_>
        status_t for_dirty_parts_(callable_type_ &&callable) noexcept {
            status_t status;
            for (std::size_t part_idx = 0; part_idx != parts_k; ++part_idx) {
                if (!dirty_[part_idx]) continue;
                unique_lock_t lock {store_.mutexes_[part_idx]};
                status = callable(parts_[part_idx]);
                if (!status) return status;
            }
            return status;
        }

      public:
        transaction_t(partitioned_collection &db, part_transactions_t &&unlocked) noexcept
            : store_(db), parts_(std::move(unlocked)), generation_(db.new_generation()) {}
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        generation_t generation() const noexcept { return generation_; }

        [[nodiscard]] status_t reset() noexcept {
            auto status = for_parts_(std::mem_fn(&part_transaction_t::reset));
            if (status) {
                generation_ = store_.new_generation();
                std::fill_n(dirty_.begin(), parts_k, false);
            }
            return status;
        }
        [[nodiscard]] status_t rollback() noexcept {
            auto status = for_dirty_parts_(std::mem_fn(&part_transaction_t::rollback));
            if (status) {
                generation_ = store_.new_generation();
                std::fill_n(dirty_.begin(), parts_k, false);
            }
            return status;
        }

        [[nodiscard]] status_t stage() noexcept {
            return for_dirty_parts_([&](part_transaction_t &part) noexcept { return part.stage(); });
        }
        [[nodiscard]] status_t commit() noexcept {
            auto status = for_dirty_parts_([&](part_transaction_t &part) noexcept { return part.commit(); });
            if (status) std::fill_n(dirty_.begin(), parts_k, false);
            return status;
        }

        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            std::size_t part_idx = store_.bucket_(id);
            dirty_[part_idx] = true;
            shared_lock_t _ {store_.mutexes_[part_idx]};
            return parts_[part_idx].watch(id);
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
            std::size_t part_idx = store_.bucket_(identifier_t(comparable));
            shared_lock_t _ {store_.mutexes_[part_idx]};
            parts_[part_idx].find(std::forward<comparable_type_>(comparable),
                                  std::forward<callback_found_type_>(callback_found),
                                  std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Copies out the member equal to @p comparable, including this transaction's writes. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result;
            result.status.errc = errc_t::key_not_found_k;
            find(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
            return result;
        }

        template <typename comparable_type_ = identifier_t>
        bool contains(comparable_type_ &&comparable) const noexcept {
            std::size_t part_idx = store_.bucket_(identifier_t(comparable));
            shared_lock_t _ {store_.mutexes_[part_idx]};
            return parts_[part_idx].contains(std::forward<comparable_type_>(comparable));
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept {
            partitioned_t::for_all_next_lookups(store_.comparator_, parts_, store_.mutexes_,
                                                std::forward<comparable_type_>(comparable),
                                                std::forward<callback_found_type_>(callback_found),
                                                std::forward<callback_missing_type_>(callback_missing));
        }

        [[nodiscard]] status_t upsert(value_t &&element) noexcept {
            std::size_t part_idx = store_.bucket_(identifier_t(element));
            dirty_[part_idx] = true;
            return parts_[part_idx].upsert(std::move(element));
        }

        [[nodiscard]] status_t erase(identifier_t const &id) noexcept {
            std::size_t part_idx = store_.bucket_(id);
            dirty_[part_idx] = true;
            return parts_[part_idx].erase(id);
        }
    };

  private:
    mutable mutexes_t mutexes_;
    parts_t parts_;
    std::atomic<generation_t> generation_;

    // Held rather than default-constructed per call: a hasher or comparator carrying state answers
    // differently from a fresh one, so rebuilding either would discard what the collection was given.
    [[no_unique_address]] hash_t hasher_ {};
    [[no_unique_address]] comparator_t comparator_ {};

    friend class transaction_t;

    partitioned_collection(parts_t &&unlocked, hash_t const &hasher = {}, comparator_t const &comparator = {}) noexcept
        : parts_(std::move(unlocked)), hasher_(hasher), comparator_(comparator) {}
    partitioned_collection &operator=(partitioned_collection &&other) noexcept {
        lock_out_of_order_<unique_lock_t>(mutexes_);
        parts_ = std::move(other.parts_);
        hasher_ = other.hasher_;
        comparator_ = other.comparator_;
        for (auto &mutex : mutexes_) mutex.unlock();
        return *this;
    }

    static std::optional<parts_t> new_parts() noexcept {
        return generate_array_safely<part_t, parts_k>([](std::size_t) { return part_t::make(); });
    }

    static std::optional<parts_t> new_parts(comparator_t const &comparator) noexcept {
        // Backends differ in whether they also take an allocator here, so the shape is detected rather
        // than assumed - a tree seeds both policies, a `std::set`-backed store only the comparator.
        return generate_array_safely<part_t, parts_k>([&](std::size_t) {
            if constexpr (requires { part_t::make(comparator, typename part_t::allocator_t {}); })
                return part_t::make(comparator, typename part_t::allocator_t {});
            else return part_t::make(comparator);
        });
    }

    generation_t new_generation() noexcept { return ++generation_; }

  public:
    partitioned_collection() noexcept = default;
    partitioned_collection(partitioned_collection &&other) noexcept
        : parts_(std::move(other.parts_)), hasher_(other.hasher_), comparator_(other.comparator_) {}

    [[nodiscard]] std::size_t size() const noexcept {
        std::size_t total = 0;
        lock_out_of_order_<shared_lock_t>(mutexes_);
        for (auto const &part : parts_) total += part.size();
        for (auto &mutex : mutexes_) mutex.unlock_shared();
        return total;
    }

    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    [[nodiscard]] static std::optional<partitioned_collection> make() noexcept {
        std::optional<partitioned_collection> result;
        if (std::optional<parts_t> unlocked = new_parts(); unlocked)
            result.emplace(partitioned_collection {std::move(unlocked).value()});
        return result;
    }

    /**
     *  @brief Builds a collection whose every partition shares one comparator and one hasher.
     *  @param[in] comparator The instance every comparison consults, in each partition and across them.
     *  @param[in] hasher The instance that maps an identifier to its partition.
     *  @return Collection instance or empty optional on failure.
     */
    [[nodiscard]] static std::optional<partitioned_collection> make(comparator_t const &comparator,
                                                                    hash_t const &hasher) noexcept {
        std::optional<partitioned_collection> result;
        if (std::optional<parts_t> unlocked = new_parts(comparator); unlocked)
            result.emplace(partitioned_collection {std::move(unlocked).value(), hasher, comparator});
        return result;
    }

    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        auto maybe = generate_array_safely<part_transaction_t, parts_k>(
            [&](std::size_t part_idx) { return parts_[part_idx].transaction(); });
        if (!maybe) return {};

        return transaction_t(*this, std::move(maybe).value());
    }

    [[nodiscard]] status_t upsert(value_t &&element) noexcept {
        std::size_t part_idx = bucket_(identifier_t(element));
        unique_lock_t _ {mutexes_[part_idx]};
        return parts_[part_idx].upsert(std::move(element));
    }

    /**
     *  @brief Removes one element, reporting whether it was there.
     *    Touches only the partition that owns the key, so the rest stay unlocked.
     *  @param[in] id The identifier to remove.
     *  @param[in] callback_found Receives the element that was removed.
     *  @param[in] callback_missing Fires when no such element existed.
     *  @return @c key_not_found_k when absent, so the answer is available without a second probe.
     */
    template <typename callback_found_type_ = no_op_fn_t, typename callback_missing_type_ = no_op_fn_t>
    [[nodiscard]] status_t erase(identifier_t const &id, callback_found_type_ &&callback_found = {},
                                 callback_missing_type_ &&callback_missing = {}) noexcept {
        std::size_t part_idx = bucket_(id);
        unique_lock_t _ {mutexes_[part_idx]};
        return parts_[part_idx].erase(id, std::forward<callback_found_type_>(callback_found),
                                      std::forward<callback_missing_type_>(callback_missing));
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

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept {
        std::size_t part_idx = bucket_(identifier_t(comparable));
        shared_lock_t _ {mutexes_[part_idx]};
        parts_[part_idx].find(std::forward<comparable_type_>(comparable),
                              std::forward<callback_found_type_>(callback_found),
                              std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename comparable_type_ = identifier_t>
    bool contains(comparable_type_ &&comparable) const noexcept {
        std::size_t part_idx = bucket_(identifier_t(comparable));
        shared_lock_t _ {mutexes_[part_idx]};
        return parts_[part_idx].contains(std::forward<comparable_type_>(comparable));
    }

    /** @brief Number of elements matching @p comparable, which is 0 or 1 for unique keys. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t count(comparable_type_ &&comparable) const noexcept {
        return contains(std::forward<comparable_type_>(comparable)) ? 1u : 0u;
    }

    /** @brief Inserts one element, refusing with @c key_already_exists_k when the key is already present. */
    [[nodiscard]] status_t insert_if_missing(value_t &&element) noexcept {
        std::size_t part_idx = bucket_(identifier_t(element));
        unique_lock_t lock {mutexes_[part_idx]};
        return parts_[part_idx].insert_if_missing(std::move(element));
    }

    /**
     *  @brief Inserts one element, saying which of the two happened rather than leaving it inferred.
     *  @param[in] element The element to insert.
     *  @param[in] callback_inserted Receives the element once stored.
     *  @param[in] callback_existing Receives the element already present, which is what declined the insert.
     */
    template <typename callback_inserted_type_, typename callback_existing_type_>
    [[nodiscard]] status_t insert_if_missing(value_t &&element, callback_inserted_type_ &&callback_inserted,
                                             callback_existing_type_ &&callback_existing) noexcept {
        std::size_t part_idx = bucket_(identifier_t(element));
        unique_lock_t lock {mutexes_[part_idx]};
        return parts_[part_idx].insert_if_missing(std::move(element),
                                                  std::forward<callback_inserted_type_>(callback_inserted),
                                                  std::forward<callback_existing_type_>(callback_existing));
    }

    /** @brief Inserts a batch, leaving already-present keys untouched. */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t insert_if_missing(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        for (; begin != end; ++begin) {
            value_t element(*begin);
            std::size_t part_idx = bucket_(identifier_t(element));
            unique_lock_t _ {mutexes_[part_idx]};
            if (auto status = parts_[part_idx].insert_if_missing(std::move(element)); !status) return status;
        }
        return {success_k};
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept {
        for_all_next_lookups(comparator_, parts_, mutexes_, std::forward<comparable_type_>(comparable),
                             std::forward<callback_found_type_>(callback_found),
                             std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief The first element at or after @p comparable, which may live in any partition. */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept {
        bool found_exact = false;
        find(
            comparable,
            [&](value_t const &value) noexcept {
                found_exact = true;
                callback_found(value);
            },
            []() noexcept {});
        if (found_exact) return;
        upper_bound(std::forward<comparable_type_>(comparable), std::forward<callback_found_type_>(callback_found),
                    std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Copies out the member equal to @p comparable, or reports @c key_not_found_k. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result;
        result.status.errc = errc_t::key_not_found_k;
        find(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    /** @brief Copies out the first element ordered at or after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result;
        result.status.errc = errc_t::key_not_found_k;
        lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    /** @brief Copies out the first element ordered strictly after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result;
        result.status.errc = errc_t::key_not_found_k;
        upper_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        lock_out_of_order_<shared_lock_t>(mutexes_);
        for (auto &part : parts_) part.range(lower, upper, callback);
        for (auto &mutex : mutexes_) mutex.unlock_shared();
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        lock_out_of_order_<unique_lock_t>(mutexes_);
        for (auto &part : parts_) part.range(lower, upper, callback);
        for (auto &mutex : mutexes_) mutex.unlock();
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) noexcept {
        lock_out_of_order_<unique_lock_t>(mutexes_);
        for (auto &part : parts_) part.erase_range(lower, upper, callback);
        for (auto &mutex : mutexes_) mutex.unlock();
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_,
              typename callback_type_ = no_op_fn_t>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                      callback_type_ &&callback) const noexcept {
        // ! Here the assumption is that every part will have a somewhat equal
        // ! number of entries that compare equal to the provided range.
        std::size_t part_idx = generator() % parts_k;
        shared_lock_t _ {mutexes_[part_idx]};
        parts_[part_idx].sample_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                      std::forward<generator_type_>(generator), std::forward<callback_type_>(callback));
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator, std::size_t &seen,
                      std::size_t reservoir_capacity, output_iterator_type_ &&reservoir) const noexcept {
        // ! This function trades consistency for performance!
        std::array<bool, parts_k> finished {false};
        std::size_t remaining_count = parts_k;

    cycle:
        for (std::size_t part_idx = 0; part_idx != parts_k; ++part_idx) {
            if (finished[part_idx]) continue;
            shared_lock_t lock {mutexes_[part_idx], std::try_to_lock_t {}};
            if (!lock) continue;

            parts_[part_idx].sample_range(lower, upper, generator, seen, reservoir_capacity, reservoir);

            finished[part_idx] = true;
            --remaining_count;
        }

        if (remaining_count) goto cycle;
    }

    [[nodiscard]] status_t clear() noexcept {

        // Rebuilt around the comparator this collection holds, not a default-constructed one: a
        // `clear()` must empty a container, never silently change how it orders what comes next.
        auto maybe = new_parts(comparator_);
        if (!maybe) return {unknown_k};

        lock_out_of_order_<unique_lock_t>(mutexes_);
        parts_ = std::move(maybe).value();
        for (auto &mutex : mutexes_) mutex.unlock();
        return {success_k};
    }

    [[nodiscard]] status_t reserve(std::size_t size) noexcept {
        return for_all<unique_lock_t>(parts_, mutexes_, [size](part_t &part) noexcept { //
            return part.reserve(size / parts_k);
        });
    }
};

} // namespace ashvardanian::smashtable
