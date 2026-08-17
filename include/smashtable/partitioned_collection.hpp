/**
 *  @brief Shards a transactional collection across independently locked partitions, so writers touching different keys
 *      rarely contend.
 *  @author Ash Vardanian
 *  @file include/smashtable/partitioned_collection.hpp
 *  @date October 16, 2022
 */
#pragma once
#include <array>        // `std::array`
#include <bit>          // `std::countr_zero`
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
    for (std::size_t part_index = 0; part_index != count_k; ++part_index) {

        if (auto new_part = generator(part_index); new_part)
            new (raw_parts + part_index) value_t(std::move(new_part).value());
        else {
            // Destruct all the previous parts.
            for (std::size_t destructed_index = 0; destructed_index != part_index; ++destructed_index)
                raw_parts[destructed_index].~value_t();
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

    /**
     *  @brief A commit takes and releases one partition's lock at a time, so a reader crossing
     *    partitions can catch a transaction half-applied. Only a single partition is atomic to it.
     */
    static constexpr isolation_t isolation_k = parts_k == 1 ? part_t::isolation_k : isolation_t::read_committed_k;

    using comparator_t = typename part_t::comparator_t;
    using identifier_t = typename part_t::identifier_t;
    using generation_t = typename part_t::generation_t;

  private:
    /**
     *  @brief A set of partition indices, one bit each, sized to @c parts_k rather than capped by it.
     *
     *  Iterating set bits rather than scanning every partition is the point: a transaction usually
     *  writes one or two of them, and only those may be locked.
     */
    struct dirty_partitions_t {
        static constexpr std::size_t bits_per_word_k = 64;
        static constexpr std::size_t words_k = (parts_k + bits_per_word_k - 1) / bits_per_word_k;

        std::array<std::uint64_t, words_k> words {};

        void mark(std::size_t part_index) noexcept {
            words[part_index / bits_per_word_k] |= std::uint64_t {1} << (part_index % bits_per_word_k);
        }
        void clear() noexcept { words.fill(0); }

        /** @brief The lowest marked partition, or @c parts_k when none is. */
        std::size_t first_set() const noexcept { return scan_from_(0); }

        /** @brief The lowest marked partition above @p part_index, or @c parts_k when none is. */
        std::size_t next_set(std::size_t part_index) const noexcept { return scan_from_(part_index + 1); }

      private:
        std::size_t scan_from_(std::size_t part_index) const noexcept {
            for (std::size_t word_index = part_index / bits_per_word_k; word_index < words_k; ++word_index) {
                std::uint64_t left = words[word_index];
                // Mask off the bits below where this scan starts, only in the word it starts in.
                if (word_index == part_index / bits_per_word_k) {
                    std::size_t const bit = part_index % bits_per_word_k;
                    left &= bit == 0 ? ~std::uint64_t {0} : ~std::uint64_t {0} << bit;
                }
                if (left) return word_index * bits_per_word_k + static_cast<std::size_t>(std::countr_zero(left));
            }
            return parts_k;
        }
    };

    std::size_t bucket_(identifier_t const &id) const noexcept { return hasher_(id) % parts_k; }

    /**
     *  @brief Takes every partition, in ascending index order.
     *
     *  Order is the whole of the deadlock argument: every caller wants all of them, and every caller
     *  asks in this same sequence, so two threads cannot each hold what the other is waiting for.
     *  Acquiring what is available and spinning for the rest has no such argument - two threads that
     *  win disjoint subsets wait on each other for good, since neither gives back what it holds.
     */
    template <typename lock_type_, typename mutexes_type_>
    static void lock_every_part_(mutexes_type_ &mutexes) noexcept {
        constexpr bool make_unique = std::is_same<lock_type_, unique_lock_t>();
        static_assert(make_unique || std::is_same<lock_type_, shared_lock_t>());

        for (std::size_t part_index = 0; part_index != parts_k; ++part_index) {
            if constexpr (make_unique) mutexes[part_index].lock();
            else mutexes[part_index].lock_shared();
        }
    }

    /**
     * @brief Walks around all the parts, trying to perform operations on them, until all the tasks are exhausted.
     */
    template <typename lock_type_, typename parts_type_, typename mutexes_type_, typename callable_type_>
    static status_t for_all(parts_type_ &parts, mutexes_type_ &mutexes, callable_type_ &&callable) noexcept {
        status_t status;
        // Ascending order, one partition at a time, blocking. Taking whichever partitions happen to be
        // free and retrying the rest reads as politer, but it shares no order with `lock_every_part_`,
        // and two all-partition operations without a common order are two operations that can wait on
        // each other for good.
        for (std::size_t part_index = 0; part_index != parts_k; ++part_index) {
            lock_type_ lock {mutexes[part_index]};
            status = callable(parts[part_index]);
            if (!status) return status;
        }
        return status;
    }

    template <typename parts_type_, typename mutexes_type_, typename comparable_type_, typename callback_found_type_,
              typename callback_missing_type_>
    static void for_all_next_lookups(comparator_t const &comparator, parts_type_ &parts, mutexes_type_ &mutexes,
                                     comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                     callback_missing_type_ &&callback_missing) noexcept {

        identifier_t smallest_id;
        std::size_t smallest_index;
        constexpr std::size_t not_found_index = std::numeric_limits<std::size_t>::max();

        // A key that was observed and then found gone is a bound the scan may step over, which is what
        // stops a restart from asking the same question again. Null until the first miss.
        identifier_t vanished_key;
        identifier_t *scan_from = nullptr;

    restart:
        smallest_index = not_found_index;

        // Ascending order, blocking, one partition at a time - the same order every other
        // all-partition walk here uses, which is what makes them safe to run against each other.
        for (std::size_t part_index = 0; part_index != parts_k; ++part_index) {
            shared_lock_t lock {mutexes[part_index]};
            auto &part = parts[part_index];
            auto const &lower = scan_from ? static_cast<identifier_t const &>(*scan_from)
                                          : static_cast<identifier_t const &>(comparable);
            part.upper_bound(lower, [&](value_t const &element) noexcept {
                if (smallest_index != not_found_index && !comparator(mapping_key_or_itself(element), smallest_id))
                    return;
                smallest_id = identifier_t(element);
                smallest_index = part_index;
            });
        }

        if (smallest_index == not_found_index) {
            callback_missing();
            return;
        }

        // Under "Read Committed" isolation the winner may be erased between the scan that found it and
        // the read below, which is the non-repeatable read that level permits. The re-read takes that
        // partition's lock, since losing the entry is allowed but reading it mid-update is not - that
        // would hand the callback a half-destroyed value.
        //
        // A miss then rescans from the missing key rather than from the original bound. Restarting from
        // the same place has no progress guarantee: a writer churning one key just above the cursor
        // makes the scan find it and lose it forever. Stepping over a key already observed absent moves
        // the bound strictly forward, so the walk terminates, and skipping a key reinserted behind the
        // cursor is the same thing this level already permits.
        bool should_restart = false;
        {
            shared_lock_t lock {mutexes[smallest_index]};
            parts[smallest_index].find(smallest_id, std::forward<callback_found_type_>(callback_found),
                                       [&]() noexcept { should_restart = true; });
        }
        if (should_restart) {
            vanished_key = std::move(smallest_id);
            scan_from = &vanished_key;
            goto restart;
        }
    }

  public:
    class transaction_t {
        friend class partitioned_collection;
        partitioned_collection &store_;
        part_transactions_t parts_;
        /**
         *  @brief Which partitions this transaction wrote, so a clean one is never locked.
         *
         *  One bit each, in as many words as @c parts_k needs, and the walks below visit only the
         *  bits that are set - a transaction touching two partitions of a hundred pays for two.
         */
        dirty_partitions_t dirty_ {};
        static_assert(std::is_nothrow_move_constructible<part_transaction_t>());

        template <typename callable_type_>
        status_t for_parts_(callable_type_ &&callable) noexcept {
            return partitioned_t::for_all<unique_lock_t>(parts_, store_.mutexes_,
                                                         std::forward<callable_type_>(callable));
        }

        template <typename callable_type_>
        status_t for_dirty_parts_(callable_type_ &&callable) noexcept {
            status_t status;
            for (std::size_t part_index = dirty_.first_set(); part_index != parts_k;
                 part_index = dirty_.next_set(part_index)) {
                unique_lock_t lock {store_.mutexes_[part_index]};
                status = callable(parts_[part_index]);
                if (!status) return status;
            }
            return status;
        }

      public:
        transaction_t(partitioned_collection &db, part_transactions_t &&unlocked) noexcept
            : store_(db), parts_(std::move(unlocked)) {}
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;

        [[nodiscard]] status_t reset() noexcept {
            auto status = for_parts_(std::mem_fn(&part_transaction_t::reset));
            if (status) dirty_.clear();
            return status;
        }
        [[nodiscard]] status_t rollback() noexcept {
            auto status = for_dirty_parts_(std::mem_fn(&part_transaction_t::rollback));
            if (status) dirty_.clear();
            return status;
        }

        /**
         *  @brief Stages every written partition, undoing the ones that already landed if a later
         *    partition refuses.
         *
         *  A partial stage is never observable. Partitions are taken ascending, so a group racing for
         *  the same set meets this one in a consistent order; the undo walks the staged prefix and
         *  rolls it back rather than resetting it, which leaves the caller's writes intact for a retry.
         */
        [[nodiscard]] status_t stage() noexcept {
            dirty_partitions_t staged;
            status_t status;
            for (std::size_t part_index = dirty_.first_set(); part_index != parts_k;
                 part_index = dirty_.next_set(part_index)) {
                unique_lock_t lock {store_.mutexes_[part_index]};
                status = parts_[part_index].stage();
                if (!status) break;
                staged.mark(part_index);
            }
            if (status) return status;

            for (std::size_t part_index = staged.first_set(); part_index != parts_k;
                 part_index = staged.next_set(part_index)) {
                unique_lock_t lock {store_.mutexes_[part_index]};
                [[maybe_unused]] status_t const unwound = parts_[part_index].rollback();
            }
            return status;
        }
        [[nodiscard]] status_t commit() noexcept {
            auto status = for_dirty_parts_([&](part_transaction_t &part) noexcept { return part.commit(); });
            if (status) dirty_.clear();
            return status;
        }

        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            std::size_t part_index = store_.bucket_(id);
            dirty_.mark(part_index);
            shared_lock_t _ {store_.mutexes_[part_index]};
            return parts_[part_index].watch(id);
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
            std::size_t part_index = store_.bucket_(identifier_t(comparable));
            shared_lock_t _ {store_.mutexes_[part_index]};
            parts_[part_index].find(std::forward<comparable_type_>(comparable),
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
            std::size_t part_index = store_.bucket_(identifier_t(comparable));
            shared_lock_t _ {store_.mutexes_[part_index]};
            return parts_[part_index].contains(std::forward<comparable_type_>(comparable));
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
            std::size_t part_index = store_.bucket_(identifier_t(element));
            dirty_.mark(part_index);
            return parts_[part_index].upsert(std::move(element));
        }

        [[nodiscard]] status_t erase(identifier_t const &id) noexcept {
            std::size_t part_index = store_.bucket_(id);
            dirty_.mark(part_index);
            return parts_[part_index].erase(id);
        }
    };

  private:
    mutable mutexes_t mutexes_;
    parts_t parts_;

    // Held rather than default-constructed per call: a hasher or comparator carrying state answers
    // differently from a fresh one, so rebuilding either would discard what the collection was given.
    [[no_unique_address]] hash_t hasher_ {};
    [[no_unique_address]] comparator_t comparator_ {};

    friend class transaction_t;

    partitioned_collection(parts_t &&unlocked, hash_t const &hasher = {}, comparator_t const &comparator = {}) noexcept
        : parts_(std::move(unlocked)), hasher_(hasher), comparator_(comparator) {}
    partitioned_collection &operator=(partitioned_collection &&other) noexcept {
        lock_every_part_<unique_lock_t>(mutexes_);
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

  public:
    partitioned_collection() noexcept = default;
    partitioned_collection(partitioned_collection &&other) noexcept
        : parts_(std::move(other.parts_)), hasher_(other.hasher_), comparator_(other.comparator_) {}

    [[nodiscard]] std::size_t size() const noexcept {
        std::size_t total = 0;
        lock_every_part_<shared_lock_t>(mutexes_);
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
            [&](std::size_t part_index) { return parts_[part_index].transaction(); });
        if (!maybe) return {};

        return transaction_t(*this, std::move(maybe).value());
    }

    [[nodiscard]] status_t upsert(value_t &&element) noexcept {
        std::size_t part_index = bucket_(identifier_t(element));
        unique_lock_t _ {mutexes_[part_index]};
        return parts_[part_index].upsert(std::move(element));
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
        std::size_t part_index = bucket_(id);
        unique_lock_t _ {mutexes_[part_index]};
        return parts_[part_index].erase(id, std::forward<callback_found_type_>(callback_found),
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
        std::size_t part_index = bucket_(identifier_t(comparable));
        shared_lock_t _ {mutexes_[part_index]};
        parts_[part_index].find(std::forward<comparable_type_>(comparable),
                                std::forward<callback_found_type_>(callback_found),
                                std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename comparable_type_ = identifier_t>
    bool contains(comparable_type_ &&comparable) const noexcept {
        std::size_t part_index = bucket_(identifier_t(comparable));
        shared_lock_t _ {mutexes_[part_index]};
        return parts_[part_index].contains(std::forward<comparable_type_>(comparable));
    }

    /** @brief Number of elements matching @p comparable, which is 0 or 1 for unique keys. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t count(comparable_type_ &&comparable) const noexcept {
        return contains(std::forward<comparable_type_>(comparable)) ? 1u : 0u;
    }

    /** @brief Inserts one element, refusing with @c key_already_exists_k when the key is already present. */
    [[nodiscard]] status_t insert_if_missing(value_t &&element) noexcept {
        std::size_t part_index = bucket_(identifier_t(element));
        unique_lock_t lock {mutexes_[part_index]};
        return parts_[part_index].insert_if_missing(std::move(element));
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
        std::size_t part_index = bucket_(identifier_t(element));
        unique_lock_t lock {mutexes_[part_index]};
        return parts_[part_index].insert_if_missing(std::move(element),
                                                    std::forward<callback_inserted_type_>(callback_inserted),
                                                    std::forward<callback_existing_type_>(callback_existing));
    }

    /** @brief Inserts a batch, leaving already-present keys untouched. */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t insert_if_missing(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        for (; begin != end; ++begin) {
            value_t element(*begin);
            std::size_t part_index = bucket_(identifier_t(element));
            unique_lock_t _ {mutexes_[part_index]};
            if (auto status = parts_[part_index].insert_if_missing(std::move(element)); !status) return status;
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
        lock_every_part_<shared_lock_t>(mutexes_);
        for (auto &part : parts_) part.range(lower, upper, callback);
        for (auto &mutex : mutexes_) mutex.unlock_shared();
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        lock_every_part_<unique_lock_t>(mutexes_);
        for (auto &part : parts_) part.range(lower, upper, callback);
        for (auto &mutex : mutexes_) mutex.unlock();
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) noexcept {
        lock_every_part_<unique_lock_t>(mutexes_);
        for (auto &part : parts_) part.erase_range(lower, upper, callback);
        for (auto &mutex : mutexes_) mutex.unlock();
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_,
              typename callback_type_ = no_op_fn_t>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                      callback_type_ &&callback) const noexcept {
        // ! Here the assumption is that every part will have a somewhat equal
        // ! number of entries that compare equal to the provided range.
        std::size_t part_index = generator() % parts_k;
        shared_lock_t _ {mutexes_[part_index]};
        parts_[part_index].sample_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                        std::forward<generator_type_>(generator),
                                        std::forward<callback_type_>(callback));
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator, std::size_t &seen,
                      std::size_t reservoir_capacity, output_iterator_type_ &&reservoir) const noexcept {
        // ! This function trades consistency for performance!
        // Ascending order, blocking, like every other all-partition walk here.
        for (std::size_t part_index = 0; part_index != parts_k; ++part_index) {
            shared_lock_t lock {mutexes_[part_index]};
            parts_[part_index].sample_range(lower, upper, generator, seen, reservoir_capacity, reservoir);
        }
    }

    [[nodiscard]] status_t clear() noexcept {

        // Rebuilt around the comparator this collection holds, not a default-constructed one: a
        // `clear()` must empty a container, never silently change how it orders what comes next.
        auto maybe = new_parts(comparator_);
        if (!maybe) return {unknown_k};

        lock_every_part_<unique_lock_t>(mutexes_);
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
