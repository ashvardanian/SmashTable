/**
 *  @brief Wraps any transactional store behind one shared mutex, making the store thread-safe while its
 *      transactions stay single-threaded.
 *  @author Ash Vardanian
 *  @file include/smashtable/locked_store.hpp
 *  @date October 13, 2022
 */
#pragma once

#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief Wraps and protects any transactional store under a shared mutex.
 *
 *  The store becomes @b thread-safe; a single transaction does @b not - it belongs to the thread
 *  that opened it, and its staged changes live outside the mutex entirely. The rule is one line and
 *  holds for every entry point, present and future: a call takes the lock exactly when it reaches the
 *  wrapped store, shared where it only reads that store and exclusive where it writes to it, and a
 *  call answered from the transaction's own staged state takes no lock at all.
 *
 *  @c stage and @c commit take the mutex separately and drop it in between, so the lock is not what
 *  spans them. The store-side reservation is: staging claims every key the transaction wrote, and no
 *  other transaction can take those keys until this one publishes or unwinds, which is what carries
 *  the inner store's isolation level across the gap.
 *
 *  @warning Every callback runs with the mutex held, and it is not recursive, so a callback that calls
 *    back into this store - or into anything that eventually does - deadlocks against itself.
 *    Copy out what a callback needs and do the rest after it returns.
 *  @warning Moving a store leaves every open transaction pointing at the husk, whose contents are
 *    empty and whose mutex guards nothing, so commits land nowhere and report success. Move only a
 *    store no transaction is open on and no other thread is touching.
 */
template <typename store_type_, typename shared_mutex_type_ = spin_shared_mutex>
class locked_store {

  public:
    using store_t = locked_store;
    using inner_store_t = store_type_;
    using inner_transaction_t = typename inner_store_t::transaction_t;
    using mutex_t = shared_mutex_type_;

    using value_t = typename inner_store_t::value_t;
    using value_type = value_t; // ? STL style
    using key_type = typename mapping_key_type_or_itself<value_t>::type;
    using mapped_type = typename mapped_value_type_or_void<value_t>::type;
    using is_associative = std::bool_constant<is_mapping<value_t>>;
    using is_transactional = std::true_type;

    /** @brief The staging reservation carries the inner store's promise over, not the mutex; see the class note. */
    static constexpr isolation_t isolation_k = inner_store_t::isolation_k;

    using comparator_t = typename inner_store_t::comparator_t;
    using identifier_t = typename inner_store_t::identifier_t;
    using generation_t = typename inner_store_t::generation_t;

    /**
     *  @brief The clock the wrapped store draws its stamps from, or @c no_clock_t where it keeps none.
     *    An outer wrapper decides whether a set of stores can share one snapshot by looking for exactly
     *    this alias, so dropping it would make a store weaker for being wrapped.
     */
    using clock_t = typename shared_clock_of<inner_store_t>::type;

    /**
     *  @brief Whether the wrapped store carries the ordered surface, which an unordered core denies it.
     *    The forwards below are gated on this, so an unordered store loses them at overload resolution
     *    rather than deep inside an instantiation.
     */
    static constexpr bool inner_is_ordered_k =
        requires(inner_store_t &store, identifier_t const &key, no_op_t callback) {
            store.lower_bound(key, callback, callback);
            store.upper_bound(key, callback, callback);
            store.range(key, key, callback);
            store.erase_range(key, key, callback);
        };

    /** @brief Whether the wrapped store names its smallest member without being given a bound to beat. */
    static constexpr bool inner_names_its_smallest_k =
        requires(inner_store_t const &store, no_op_t callback) { store.smallest(callback, callback); };

    /** @brief Whether the wrapped store removes its smallest member as one operation. */
    static constexpr bool inner_pops_its_smallest_k =
        requires(inner_store_t &store, no_op_t callback) { store.pop_smallest(callback, callback); };

    /** @brief Whether an open transaction names the smallest member it reads, staged writes included. */
    static constexpr bool inner_transaction_names_its_smallest_k = requires(
        inner_transaction_t const &transaction, no_op_t callback) { transaction.smallest(callback, callback); };

    /** @brief Whether the wrapped store draws one member of a range at random. */
    static constexpr bool inner_samples_one_k =
        requires(inner_store_t const &store, identifier_t const &key, no_op_t callback) {
            store.sample_one(key, key, callback, callback);
        };

    /** @brief Whether the wrapped store fills a reservoir from a range. */
    static constexpr bool inner_samples_reservoir_k =
        requires(inner_store_t const &store, identifier_t const &key, no_op_t callback, std::size_t seen) {
            store.sample_reservoir(key, key, callback, seen, seen, callback);
        };

    /** @brief Whether an open transaction carries the ordered surface its store does. */
    static constexpr bool inner_transaction_is_ordered_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.upper_bound(key, callback, callback);
        };

    /** @brief Whether an open transaction walks a window with one end left open. */
    static constexpr bool inner_transaction_walks_open_range_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.range_from(key, callback);
            transaction.range_up_to(key, callback);
        };

    /** @brief Whether the wrapped store enumerates its members with no ordering to walk them in. */
    static constexpr bool inner_enumerates_k =
        requires(inner_store_t const &store, no_op_t callback) { store.for_each(callback); };

    /** @brief Whether the wrapped store reclaims superseded versions on demand. */
    static constexpr bool inner_reclaims_k = requires(inner_store_t &store) { store.vacuum(); };

    /** @brief Whether the wrapped store reclaims a window of the keyspace rather than all of it. */
    static constexpr bool inner_reclaims_range_k =
        requires(inner_store_t &store, identifier_t const &key) { store.vacuum(key, key); };

    /**
     *  @brief Whether the wrapped store answers by ordinal, which only an order-statistics core does.
     *    The forwards below are gated on this, so a core keeping no subtree counts loses them at
     *    overload resolution rather than deep inside an instantiation.
     */
    static constexpr bool inner_is_ranked_k =
        requires(inner_store_t const &store, identifier_t const &key, std::size_t ordinal, no_op_t callback) {
            store.select(ordinal, callback, callback);
            store.rank(key, callback, callback);
        };

    /** @brief Whether the wrapped store rewrites the mapped side of a range in place. */
    static constexpr bool inner_revises_range_k = requires(
        inner_store_t &store, identifier_t const &key, no_op_t callback) { store.update_range(key, key, callback); };

    /** @brief Whether the wrapped store refuses an occupied key rather than writing over it. */
    static constexpr bool inner_refuses_occupied_key_k =
        requires(inner_store_t &store, value_t &&element) { store.insert(std::move(element)); };

    /** @brief Whether that refusal also says which element declined the insert. */
    static constexpr bool inner_reports_occupied_key_k =
        requires(inner_store_t &store, value_t &&element, no_op_t callback) {
            store.insert(std::move(element), callback, callback);
        };

    /** @brief Whether the wrapped store refuses an absent key rather than creating it. */
    static constexpr bool inner_refuses_absent_key_k =
        requires(inner_store_t &store, value_t &&element) { store.update(std::move(element)); };

    /** @brief Whether an open transaction refuses an occupied key rather than writing over it. */
    static constexpr bool inner_transaction_refuses_occupied_key_k =
        requires(inner_transaction_t &transaction, value_t &&element) { transaction.insert(std::move(element)); };

    /** @brief Whether that refusal, inside a transaction, also says which element declined the insert. */
    static constexpr bool inner_transaction_reports_occupied_key_k =
        requires(inner_transaction_t &transaction, value_t &&element, no_op_t callback) {
            transaction.insert(std::move(element), callback, callback);
        };

    /** @brief Whether an open transaction refuses an absent key rather than creating it. */
    static constexpr bool inner_transaction_refuses_absent_key_k =
        requires(inner_transaction_t &transaction, value_t &&element) { transaction.update(std::move(element)); };

    /** @brief Whether the wrapped store erases an open-ended window of the keyspace. */
    static constexpr bool inner_erases_open_range_k =
        requires(inner_store_t &store, identifier_t const &key, no_op_t callback) {
            store.erase_from(key, callback);
            store.erase_up_to(key, callback);
        };

    /** @brief Whether the wrapped store answers every member equal to a key rather than only the first. */
    static constexpr bool inner_matches_equals_k = requires(inner_store_t const &store, identifier_t const &key,
                                                            no_op_t callback) { store.equal_range(key, callback); };

    /** @brief Whether the wrapped store says how many keys its ordinal surface indexes. */
    static constexpr bool inner_counts_ranked_k = requires(inner_store_t const &store) { store.ranked_size(); };

    /** @brief Whether the wrapped store keeps superseded versions and can count them. */
    static constexpr bool inner_counts_versions_k = requires(inner_store_t const &store, identifier_t const &key) {
        store.versions_count();
        store.versions_count(key);
    };

    /** @brief Whether the wrapped store names the snapshot no open reader sits below. */
    static constexpr bool inner_marks_low_water_k = requires(inner_store_t const &store) { store.low_water_mark(); };

    /** @brief Whether the wrapped store spells an overwriting write as @c insert_or_assign. */
    static constexpr bool inner_assigns_over_key_k =
        requires(inner_store_t &store, value_t &&element) { store.insert_or_assign(std::move(element)); };

    /**
     *  @brief Whether the wrapped store lets a shard set hand it a clock and open one part of a
     *    transaction on a snapshot drawn elsewhere, which is what a snapshot spanning shards is made of.
     */
    static constexpr bool inner_shares_clock_k = requires(inner_store_t &store, clock_t &clock, generation_t stamp) {
        store.attach_clock(clock);
        store.transaction_at(stamp, stamp);
    };

    /** @brief Whether an open transaction reports what it has staged so far. */
    static constexpr bool inner_transaction_reports_changes_k = requires(inner_transaction_t const &transaction) {
        transaction.has_changes();
        transaction.changes_count();
    };

    /** @brief Whether an open transaction refuses an occupied key silently rather than with a status. */
    static constexpr bool inner_transaction_skips_occupied_key_k = requires(
        inner_transaction_t &transaction, value_t &&element) { transaction.insert_if_missing(std::move(element)); };

    /** @brief Whether an open transaction enumerates what its snapshot and its own writes show. */
    static constexpr bool inner_transaction_enumerates_k =
        requires(inner_transaction_t const &transaction, no_op_t callback) { transaction.for_each(callback); };

    /** @brief Whether an open transaction walks a half-open window of the keyspace. */
    static constexpr bool inner_transaction_walks_range_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.range(key, key, callback);
        };

    /** @brief Whether the wrapped transaction counts the members matching a key. */
    static constexpr bool inner_transaction_counts_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key) { transaction.count(key); };
    /** @brief Whether the wrapped transaction copies out the member at a bound. */
    static constexpr bool inner_transaction_copies_bounds_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key) {
            transaction.lower_bound_copy(key);
            transaction.upper_bound_copy(key);
        };
    /** @brief Whether the wrapped transaction erases a window of its own keys. */
    static constexpr bool inner_transaction_erases_range_k =
        requires(inner_transaction_t &transaction, identifier_t const &key, no_op_t callback) {
            transaction.erase_range(key, key, callback);
            transaction.erase_from(key, callback);
            transaction.erase_up_to(key, callback);
        };
    /** @brief Whether the wrapped transaction revises a window of its own members. */
    static constexpr bool inner_transaction_revises_range_k =
        requires(inner_transaction_t &transaction, identifier_t const &key, no_op_t callback) {
            transaction.update_range(key, key, callback);
        };
    /** @brief Whether the wrapped transaction draws members from a window. */
    static constexpr bool inner_transaction_samples_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback, std::size_t seen) {
            transaction.sample_one(key, key, callback, callback);
            transaction.sample_reservoir(key, key, callback, seen, seen, callback);
        };

    /** @brief Whether an open transaction answers every member equal to a key. */
    static constexpr bool inner_transaction_matches_equals_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.equal_range(key, callback);
        };

    /** @brief Whether an open transaction answers the first key at or after a bound. */
    static constexpr bool inner_transaction_lower_bounds_k =
        requires(inner_transaction_t const &transaction, identifier_t const &key, no_op_t callback) {
            transaction.lower_bound(key, callback, callback);
        };

    /**
     *  @brief Whether an open transaction can be driven as one part of a sharded commit, which asks
     *    every part whether it may proceed before any of them publishes.
     */
    static constexpr bool inner_transaction_shards_k = requires(inner_transaction_t &transaction, generation_t stamp) {
        transaction.validate_for_commit();
        transaction.publish_under(static_cast<commit_stamp_t>(stamp));
        transaction.adopt_snapshot(stamp);
        transaction.prune_committed();
        transaction.reset_at(stamp);
    };

    /**
     *  @brief Whether the wrapped transaction decides and writes in two steps, stamping its own versions.
     *
     *  The engines keeping no shared clock split their commit the same way, but publish without being
     *  handed a stamp - each orders its own versions. A shard set spanning them still has to learn that
     *  every partition may commit before any of them writes, so this surface has to travel too.
     */
    static constexpr bool inner_transaction_splits_commit_k = requires(inner_transaction_t &transaction) {
        { transaction.validate_for_commit() } noexcept -> std::same_as<status_t>;
        transaction.publish_under();
    };

    class transaction_t {
        friend class locked_store;
        /**
         *  @brief The store this transaction reaches through, held by pointer rather than reference.
         *    A reference member deletes the defaulted move assignment, and a transaction that moves but
         *    cannot be move-assigned is one no container can hold.
         */
        locked_store *store_;
        /** @brief The inner store's own transaction, which stages entirely outside the mutex. */
        inner_transaction_t inner_transaction_;
        static_assert(std::is_nothrow_move_constructible<inner_transaction_t>());

      public:
        transaction_t(locked_store &db, inner_transaction_t &&inner_transaction) noexcept
            : store_(&db), inner_transaction_(std::move(inner_transaction)) {}
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        generation_t generation() const noexcept { return inner_transaction_.generation(); }

        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.watch(id);
        }

        [[nodiscard]] status_t reserve(std::size_t size) noexcept { return inner_transaction_.reserve(size); }
        [[nodiscard]] status_t upsert(value_t &&element) noexcept {
            return inner_transaction_.upsert(std::move(element));
        }
        /** @brief Stages an erase, reporting @c key_not_found_k when this transaction reads no such key. */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t erase(identifier_t const &id, callback_found_type_ &&callback_found = {},
                                     callback_missing_type_ &&callback_missing = {}) noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.erase(id, std::forward<callback_found_type_>(callback_found),
                                            std::forward<callback_missing_type_>(callback_missing));
        }

        /**
         *  @brief Stages @p element only if its key is free, refusing rather than writing over it.
         *    Takes the lock, unlike @c upsert: a strict insert reads the store to learn whether the
         *    key is already there, and only staging itself lives outside the mutex.
         */
        [[nodiscard]] status_t insert(value_t &&element) noexcept
            requires inner_transaction_refuses_occupied_key_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.insert(std::move(element));
        }

        /**
         *  @brief Stages @p element only if its key is free, and says which branch was taken.
         *  @param[in] callback_inserted Receives the element once staged.
         *  @param[in] callback_existing Receives the element already under the key, which refused the insert.
         */
        template <typename callback_inserted_type_, typename callback_existing_type_>
        [[nodiscard]] status_t insert(value_t &&element, callback_inserted_type_ &&callback_inserted,
                                      callback_existing_type_ &&callback_existing) noexcept
            requires inner_transaction_reports_occupied_key_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.insert(std::move(element),
                                             std::forward<callback_inserted_type_>(callback_inserted),
                                             std::forward<callback_existing_type_>(callback_existing));
        }

        /** @brief Stages @p element only if its key is already taken, refusing to create one. */
        [[nodiscard]] status_t update(value_t &&element) noexcept
            requires inner_transaction_refuses_absent_key_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.update(std::move(element));
        }

        [[nodiscard]] status_t stage() noexcept {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.stage();
        }

        [[nodiscard]] status_t reset() noexcept {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.reset();
        }

        [[nodiscard]] status_t rollback() noexcept {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.rollback();
        }

        [[nodiscard]] status_t commit() noexcept {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.commit();
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.find(std::forward<comparable_type_>(comparable),
                                           std::forward<callback_found_type_>(callback_found),
                                           std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Copies out the member equal to @p comparable, including this transaction's writes. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            status_t const looked_up = find(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            if (failed(looked_up)) return looked_up;
            return result;
        }

        /** @brief Whether @p comparable is there, including this transaction's own writes. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.contains(std::forward<comparable_type_>(comparable));
        }

        /**
         *  @brief Finds the member equal to @p comparable and records what it saw into the read set.
         *    The watching counterpart to @c find, and the one that can fail, because a read set is memory.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find_and_watch(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                              callback_missing_type_ &&callback_missing = {}) noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.find_and_watch(std::forward<comparable_type_>(comparable),
                                                     std::forward<callback_found_type_>(callback_found),
                                                     std::forward<callback_missing_type_>(callback_missing));
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept
            requires inner_transaction_is_ordered_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.upper_bound(std::forward<comparable_type_>(comparable),
                                                  std::forward<callback_found_type_>(callback_found),
                                                  std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Hands @p callback_found the smallest member this transaction reads, or reports none. */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t smallest(callback_found_type_ &&callback_found,
                                        callback_missing_type_ &&callback_missing = {}) const noexcept
            requires inner_transaction_names_its_smallest_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.smallest(std::forward<callback_found_type_>(callback_found),
                                               std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Hands @p callback_found the first member at or after @p comparable, or reports none. */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept
            requires inner_transaction_lower_bounds_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.lower_bound(std::forward<comparable_type_>(comparable),
                                                  std::forward<callback_found_type_>(callback_found),
                                                  std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Hands @p callback every member equal to @p comparable, this transaction's writes included. */
        template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept
            requires inner_transaction_matches_equals_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.equal_range(std::forward<comparable_type_>(comparable),
                                                  std::forward<callback_type_>(callback));
        }

        /** @brief Hands @p callback every member of [ @p lower, @p upper ), this transaction's writes included. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires inner_transaction_walks_range_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                            std::forward<callback_type_>(callback));
        }

        /** @brief Hands @p callback every member at or after @p lower, with no upper end. */
        template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t range_from(lower_type_ &&lower, callback_type_ &&callback) const noexcept
            requires inner_transaction_walks_open_range_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.range_from(std::forward<lower_type_>(lower),
                                                 std::forward<callback_type_>(callback));
        }

        /** @brief Hands @p callback every member before @p upper, @p upper excluded, with no lower end. */
        template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t range_up_to(upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires inner_transaction_walks_open_range_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.range_up_to(std::forward<upper_type_>(upper),
                                                  std::forward<callback_type_>(callback));
        }

        /** @brief Hands @p callback every member this transaction reads, in whatever order the store keeps. */
        template <typename callback_type_ = no_op_t>
        [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept
            requires inner_transaction_enumerates_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.for_each(std::forward<callback_type_>(callback));
        }

        /** @brief Stages @p element only if its key is free, leaving an incumbent untouched. */
        [[nodiscard]] status_t insert_if_missing(value_t &&element) noexcept
            requires inner_transaction_skips_occupied_key_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.insert_if_missing(std::move(element));
        }

#pragma region Transaction Range Operations

        /** @brief How many members equal @p comparable, this transaction's own writes included. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept
            requires inner_transaction_counts_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.count(std::forward<comparable_type_>(comparable));
        }

        /** @brief Copies out the first member at or after @p comparable. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
            requires inner_transaction_copies_bounds_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.lower_bound_copy(std::forward<comparable_type_>(comparable));
        }

        /** @brief Copies out the first member after @p comparable. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
            requires inner_transaction_copies_bounds_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.upper_bound_copy(std::forward<comparable_type_>(comparable));
        }

        /** @brief Stages a tombstone for every member in [ @p lower, @p upper ). */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires inner_transaction_erases_range_k
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.erase_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                                  std::forward<callback_type_>(callback));
        }

        /** @brief Stages a tombstone for every member at or after @p lower. */
        template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback) noexcept
            requires inner_transaction_erases_range_k
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.erase_from(std::forward<lower_type_>(lower),
                                                 std::forward<callback_type_>(callback));
        }

        /** @brief Stages a tombstone for every member before @p upper. */
        template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires inner_transaction_erases_range_k
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.erase_up_to(std::forward<upper_type_>(upper),
                                                  std::forward<callback_type_>(callback));
        }

        /** @brief Hands @p callback each member in [ @p lower, @p upper ) to revise, and stages the result. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper,
                                            callback_type_ &&callback) noexcept
            requires inner_transaction_revises_range_k
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.update_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                                   std::forward<callback_type_>(callback));
        }

        /** @brief Draws one member uniformly from [ @p lower, @p upper ). */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                          callback_type_ &&callback) const noexcept
            requires inner_transaction_samples_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.sample_one(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                                 std::forward<generator_type_>(generator),
                                                 std::forward<callback_type_>(callback));
        }

        /** @brief Fills @p reservoir with up to @p capacity members drawn from [ @p lower, @p upper ). */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename output_iterator_type_ = no_op_t>
        [[nodiscard]] status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                                std::size_t &seen, std::size_t capacity,
                                                output_iterator_type_ &&reservoir) const noexcept
            requires inner_transaction_samples_k
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.sample_reservoir(std::forward<lower_type_>(lower),
                                                       std::forward<upper_type_>(upper),
                                                       std::forward<generator_type_>(generator), seen, capacity,
                                                       std::forward<output_iterator_type_>(reservoir));
        }

#pragma endregion Transaction Range Operations

        /** @brief Whether anything is staged, which is transaction-local and needs no lock. */
        [[nodiscard]] bool has_changes() const noexcept
            requires inner_transaction_reports_changes_k
        {
            return inner_transaction_.has_changes();
        }

        /** @brief How many writes are staged, which is transaction-local and needs no lock. */
        [[nodiscard]] std::size_t changes_count() const noexcept
            requires inner_transaction_reports_changes_k
        {
            return inner_transaction_.changes_count();
        }

#pragma region Sharded Commit

        /**
         *  @brief The part of a commit an outer shard set drives itself, one call per phase.
         *
         *  A shard set asks every part whether it may proceed, publishes all of them under one stamp,
         *  then reclaims - so the phases are separate calls rather than one @c commit. Each takes this
         *  store's own mutex; the outer wrapper holds its partition locks around all of them, and the
         *  two orders never cross, since a partition lock is always taken first.
         */
        [[nodiscard]] status_t validate_for_commit() const noexcept
            requires(inner_transaction_shards_k || inner_transaction_splits_commit_k)
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.validate_for_commit();
        }

        /** @brief Publishes everything staged under @p stamp, which the shard set drew for the whole commit. */
        void publish_under(commit_stamp_t stamp) noexcept
            requires inner_transaction_shards_k
        {
            unique_lock _ {store_->mutex_};
            inner_transaction_.publish_under(stamp);
        }

        /** @brief Publishes everything staged, the wrapped store stamping its own versions. */
        void publish_under() noexcept
            requires inner_transaction_splits_commit_k
        {
            unique_lock _ {store_->mutex_};
            inner_transaction_.publish_under();
        }

        /** @brief Moves where this part reads, which is transaction-local and touches no store. */
        void adopt_snapshot(generation_t snapshot) noexcept
            requires inner_transaction_shards_k
        {
            inner_transaction_.adopt_snapshot(snapshot);
        }

        /** @brief Frees whatever the commit just published left unreachable. */
        void prune_committed() noexcept
            requires inner_transaction_shards_k
        {
            unique_lock _ {store_->mutex_};
            inner_transaction_.prune_committed();
        }

        /** @brief Discards everything staged and pending, at a snapshot the shard set drew once. */
        [[nodiscard]] status_t reset_at(generation_t snapshot) noexcept
            requires inner_transaction_shards_k
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.reset_at(snapshot);
        }

#pragma endregion Sharded Commit
    };

  private:
    mutable mutex_t mutex_;
    inner_store_t inner_store_;

    locked_store(inner_store_t &&inner_store) noexcept : inner_store_(std::move(inner_store)) {}
    /** @warning @p other must have no open transaction and no other thread touching it; see the class note. */
    locked_store &operator=(locked_store &&other) noexcept {
        unique_lock _ {mutex_};
        inner_store_ = std::move(other.inner_store_);
        return *this;
    }

  public:
    locked_store() noexcept = default;
    /** @warning @p other must have no open transaction and no other thread touching it; see the class note. */
    locked_store(locked_store &&other) noexcept : inner_store_(std::move(other.inner_store_)) {}

    [[nodiscard]] std::size_t size() const noexcept {
        shared_lock _ {mutex_};
        return inner_store_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        shared_lock _ {mutex_};
        return inner_store_.empty();
    }

    /**
     *  @brief Builds the inner store from @p arguments and takes ownership of it.
     *    Forwards whatever the core needs - a comparator for a tree, a hasher and equality for a table -
     *    so a store whose comparator has no default constructor is still constructible.
     */
    template <typename... arguments_type_>
    [[nodiscard]] static expected<locked_store> make(arguments_type_ &&...arguments) noexcept {
        expected<inner_store_t> inner_store = inner_store_t::make(std::forward<arguments_type_>(arguments)...);
        if (!inner_store) return inner_store.status();
        return locked_store {std::move(*inner_store)};
    }

    [[nodiscard]] expected<transaction_t> transaction() noexcept {
        unique_lock _ {mutex_};
        auto opened = inner_store_.transaction();
        if (!opened) return opened.status();
        return transaction_t {*this, std::move(*opened)};
    }

    [[nodiscard]] status_t upsert(value_t &&element) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.upsert(std::forward<value_t>(element));
    }

    /**
     *  @brief Erases whatever equals @p comparable, reporting through the callbacks so presence needs
     *    no second probe. Takes any type the inner store compares against, not only an identifier.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found = {},
                                 callback_missing_type_ &&callback_missing = {}) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.erase(std::forward<comparable_type_>(comparable),
                                  std::forward<callback_found_type_>(callback_found),
                                  std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Inserts @p element only if its key is absent, leaving an incumbent untouched. */
    [[nodiscard]] status_t insert_if_missing(value_t &&element) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.insert_if_missing(std::move(element));
    }

    /**
     *  @brief Inserts @p element only if its key is absent, and says which branch was taken.
     *  @param[in] callback_inserted Receives the element once stored.
     *  @param[in] callback_existing Receives the element already present, which is what declined the insert.
     */
    template <typename callback_inserted_type_, typename callback_existing_type_>
    [[nodiscard]] status_t insert_if_missing(value_t &&element, callback_inserted_type_ &&callback_inserted,
                                             callback_existing_type_ &&callback_existing) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.insert_if_missing(std::move(element),
                                              std::forward<callback_inserted_type_>(callback_inserted),
                                              std::forward<callback_existing_type_>(callback_existing));
    }

    /** @brief Inserts @p element only if its key is free, refusing with the inner store's own status. */
    [[nodiscard]] status_t insert(value_t &&element) noexcept
        requires inner_refuses_occupied_key_k
    {
        unique_lock _ {mutex_};
        return inner_store_.insert(std::move(element));
    }

    /**
     *  @brief Inserts @p element only if its key is free, and says which branch was taken.
     *  @param[in] callback_inserted Receives the element once stored.
     *  @param[in] callback_existing Receives the element already under the key, which refused the insert.
     */
    template <typename callback_inserted_type_, typename callback_existing_type_>
    [[nodiscard]] status_t insert(value_t &&element, callback_inserted_type_ &&callback_inserted,
                                  callback_existing_type_ &&callback_existing) noexcept
        requires inner_reports_occupied_key_k
    {
        unique_lock _ {mutex_};
        return inner_store_.insert(std::move(element), std::forward<callback_inserted_type_>(callback_inserted),
                                   std::forward<callback_existing_type_>(callback_existing));
    }

    /** @brief Writes @p element only if its key is already taken, refusing to create one. */
    [[nodiscard]] status_t update(value_t &&element) noexcept
        requires inner_refuses_absent_key_k
    {
        unique_lock _ {mutex_};
        return inner_store_.update(std::move(element));
    }

    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.upsert(begin, end);
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept {
        shared_lock _ {mutex_};
        return inner_store_.find(std::forward<comparable_type_>(comparable),
                                 std::forward<callback_found_type_>(callback_found),
                                 std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Existence check, expressed through @c find so the lock discipline stays in one place. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
        bool present = false;
        status_t const answered = find(
            std::forward<comparable_type_>(comparable), [&](value_t const &) noexcept { present = true; }, no_op_t {});
        if (failed(answered)) return answered;
        return present;
    }

    /** @brief Number of elements matching @p comparable, which is 0 or 1 for unique keys. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept {
        expected<bool> const present = contains(std::forward<comparable_type_>(comparable));
        if (!present) return present.status();
        return *present ? std::size_t {1} : std::size_t {0};
    }

    /** @brief Inserts a batch, leaving already-present keys untouched. */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t insert_if_missing(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.insert_if_missing(begin, end);
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ordered_k
    {
        shared_lock _ {mutex_};
        return inner_store_.lower_bound(std::forward<comparable_type_>(comparable),
                                        std::forward<callback_found_type_>(callback_found),
                                        std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief A resumable walk in ascending key order, holding no lock between its steps.
     *
     *  The position is a key rather than an iterator, so a write between two steps cannot invalidate
     *  it: the next step re-probes from the last key handed over. That is what lets the walk survive
     *  concurrent change, and it is why a step costs a lookup rather than an increment.
     */
    class ordered_cursor_t {
        friend class locked_store;

        locked_store const *store_ {nullptr};
        /** @brief The bound the next step is taken from - the one given, then the last key handed over. */
        identifier_t position_;
        /** @brief The key the walk stops before, meaningful only under @c up_to_the_bound_k. */
        identifier_t bound_;
        cursor_seed_t seed_ {cursor_seed_t::the_smallest_k};
        cursor_limit_t limit_ {cursor_limit_t::the_whole_keyspace_k};
        bool drained_ {false};

        explicit ordered_cursor_t(locked_store const &store, cursor_seed_t seed, cursor_limit_t limit) noexcept
            : store_(&store), seed_(seed), limit_(limit) {}

      public:
        ordered_cursor_t() noexcept = default;

        /** @brief Whether the walk is over, which includes having passed the bound it was given. */
        [[nodiscard]] bool exhausted() const noexcept { return !store_ || drained_; }

        /**
         *  @brief Hands @p callback_found the next member, or reports the walk is over.
         *  @warning @p callback_found runs under the store's lock and must not write to it.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        void next(callback_found_type_ &&callback_found, callback_missing_type_ &&callback_missing = {}) noexcept {
            if (exhausted()) return void(callback_missing());

            shared_lock _ {store_->mutex_};
            bool handed = false;
            auto take = [&](value_t const &element) noexcept {
                identifier_t const &key = mapping_key_or_itself<value_t>(element);
                if (limit_ == cursor_limit_t::up_to_the_bound_k && !store_->inner_store_.key_comp()(key, bound_))
                    return;
                position_ = identifier_t(key);
                handed = true;
                callback_found(element);
            };

            [[maybe_unused]] status_t const stepped =
                seed_ == cursor_seed_t::past_the_last_k
                    ? store_->inner_store_.upper_bound(position_, take, no_op_t {})
                    : (seed_ == cursor_seed_t::the_given_bound_k
                           ? store_->inner_store_.lower_bound(position_, take, no_op_t {})
                           : store_->inner_store_.smallest(take, no_op_t {}));

            if (!handed) {
                drained_ = true;
                callback_missing();
                return;
            }
            seed_ = cursor_seed_t::past_the_last_k;
        }
    };

    /** @brief A walk of every member in ascending order, resumable and holding no lock between steps. */
    [[nodiscard]] ordered_cursor_t cursor() const noexcept
        requires inner_is_ordered_k && inner_names_its_smallest_k
    {
        return ordered_cursor_t {*this, cursor_seed_t::the_smallest_k, cursor_limit_t::the_whole_keyspace_k};
    }

    /** @brief The same walk, begun at the first member ordered at or after @p from. */
    [[nodiscard]] ordered_cursor_t cursor_from(identifier_t from) const noexcept
        requires inner_is_ordered_k
    {
        ordered_cursor_t walking {*this, cursor_seed_t::the_given_bound_k, cursor_limit_t::the_whole_keyspace_k};
        walking.position_ = std::move(from);
        return walking;
    }

    /** @brief The same walk, stopping before @p upper. */
    [[nodiscard]] ordered_cursor_t cursor_up_to(identifier_t upper) const noexcept
        requires inner_is_ordered_k && inner_names_its_smallest_k
    {
        ordered_cursor_t walking {*this, cursor_seed_t::the_smallest_k, cursor_limit_t::up_to_the_bound_k};
        walking.bound_ = std::move(upper);
        return walking;
    }

    /** @brief The same walk over [ @p from, @p upper ). */
    [[nodiscard]] ordered_cursor_t cursor_range(identifier_t from, identifier_t upper) const noexcept
        requires inner_is_ordered_k
    {
        ordered_cursor_t walking {*this, cursor_seed_t::the_given_bound_k, cursor_limit_t::up_to_the_bound_k};
        walking.position_ = std::move(from);
        walking.bound_ = std::move(upper);
        return walking;
    }

    /** @brief Hands @p callback_found the smallest member, or reports the store is empty. */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t smallest(callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_names_its_smallest_k
    {
        shared_lock _ {mutex_};
        return inner_store_.smallest(std::forward<callback_found_type_>(callback_found),
                                     std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief Removes the smallest member and hands it over, or reports the store is empty.
     *  The choice and the removal happen under one exclusive hold, so no writer can take the member
     *  between the two.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t pop_smallest(callback_found_type_ &&callback_found = {},
                                        callback_missing_type_ &&callback_missing = {}) noexcept
        requires inner_pops_its_smallest_k
    {
        unique_lock _ {mutex_};
        return inner_store_.pop_smallest(std::forward<callback_found_type_>(callback_found),
                                         std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Copies out the smallest member and removes it, or reports @c key_not_found_k. */
    [[nodiscard]] expected<value_t> pop_smallest_copy() noexcept
        requires inner_pops_its_smallest_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const popped =
            pop_smallest([&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(popped)) return popped;
        return result;
    }

    /** @brief Copies out the smallest member, or reports @c key_not_found_k. */
    [[nodiscard]] expected<value_t> smallest_copy() const noexcept
        requires inner_names_its_smallest_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const looked_up =
            smallest([&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(looked_up)) return looked_up;
        return result;
    }

    /** @brief Copies out the member equal to @p comparable, or reports @c key_not_found_k. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const looked_up = find(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(looked_up)) return looked_up;
        return result;
    }

    /** @brief Copies out the first element ordered at or after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
        requires inner_is_ordered_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const bounded = lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(bounded)) return bounded;
        return result;
    }

    /** @brief Copies out the first element ordered strictly after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
        requires inner_is_ordered_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const bounded = upper_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(bounded)) return bounded;
        return result;
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ordered_k
    {
        shared_lock _ {mutex_};
        return inner_store_.upper_bound(std::forward<comparable_type_>(comparable),
                                        std::forward<callback_found_type_>(callback_found),
                                        std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
        requires inner_is_ordered_k
    {
        shared_lock _ {mutex_};
        return inner_store_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                  std::forward<callback_type_>(callback));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                       callback_type_ &&callback = {}) noexcept
        requires inner_is_ordered_k
    {
        unique_lock _ {mutex_};
        return inner_store_.erase_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                        std::forward<callback_type_>(callback));
    }

    /** @brief Erases every element at or after @p lower, reporting each to @p callback. */
    template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback = {}) noexcept
        requires inner_erases_open_range_k
    {
        unique_lock _ {mutex_};
        return inner_store_.erase_from(std::forward<lower_type_>(lower), std::forward<callback_type_>(callback));
    }

    /** @brief Erases every element before @p upper, reporting each to @p callback. */
    template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires inner_erases_open_range_k
    {
        unique_lock _ {mutex_};
        return inner_store_.erase_up_to(std::forward<upper_type_>(upper), std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Rewrites the mapped side of every element in [ @p lower, @p upper ).
     *  @param[in] callback Invoked with (key const &, mapped &) per element. Must be @c noexcept.
     *  @return Success, or an allocation failure. A store whose own walk cannot fail always succeeds.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires inner_revises_range_k
    {
        unique_lock _ {mutex_};
        return inner_store_.update_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                         std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Hands @p callback every member the store holds, in whatever order it keeps them.
     *
     *  The one walk an unordered core can offer, and the whole of it runs under one shared lock, so a
     *  writer is held off for its length and the enumeration is a consistent snapshot: every element
     *  present when the call began is visited exactly once, and no element inserted during it is seen.
     */
    template <typename callback_type_ = no_op_t>
    [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept
        requires inner_enumerates_k
    {
        shared_lock _ {mutex_};
        return inner_store_.for_each(std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Frees every version no reader can still reach.
     *  @return How many versions were reclaimed, or why none could be. A store whose sweep cannot
     *    refuse always answers with a count.
     */
    [[nodiscard]] expected<std::size_t> vacuum() noexcept
        requires inner_reclaims_k
    {
        unique_lock _ {mutex_};
        return inner_store_.vacuum();
    }

    /**
     *  @brief Frees the unreachable versions of every key in [ @p lower, @p upper ), so a caller can
     *    step through the keyspace instead of paying for one pass over all of it.
     *  @return How many versions were reclaimed, or why none could be.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t>
    [[nodiscard]] expected<std::size_t> vacuum(lower_type_ &&lower, upper_type_ &&upper) noexcept
        requires inner_reclaims_range_k
    {
        unique_lock _ {mutex_};
        return inner_store_.vacuum(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper));
    }

    /**
     *  @brief Hands @p callback_found the element at zero-based position @p ordinal.
     *  @param[in] callback_missing Fires when fewer elements are there. Must be @c noexcept.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                                  callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ranked_k
    {
        shared_lock _ {mutex_};
        return inner_store_.select(ordinal, std::forward<callback_found_type_>(callback_found),
                                   std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief Hands @p callback_found how many elements the store orders before @p comparable.
     *  @param[in] callback_missing Fires when @p comparable is not there at all. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ranked_k
    {
        shared_lock _ {mutex_};
        return inner_store_.rank(std::forward<comparable_type_>(comparable),
                                 std::forward<callback_found_type_>(callback_found),
                                 std::forward<callback_missing_type_>(callback_missing));
    }

    [[nodiscard]] status_t clear() noexcept {
        unique_lock _ {mutex_};
        return inner_store_.clear();
    }

    [[nodiscard]] status_t reserve(std::size_t size) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.reserve(size);
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                      callback_type_ &&callback) const noexcept
        requires inner_samples_one_k
    {
        shared_lock _ {mutex_};
        return inner_store_.sample_one(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                       std::forward<generator_type_>(generator),
                                       std::forward<callback_type_>(callback));
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    [[nodiscard]] status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                            std::size_t &seen, std::size_t reservoir_capacity,
                                            output_iterator_type_ &&reservoir) const noexcept
        requires inner_samples_reservoir_k
    {
        shared_lock _ {mutex_};
        return inner_store_.sample_reservoir(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                             std::forward<generator_type_>(generator), seen, reservoir_capacity,
                                             std::forward<output_iterator_type_>(reservoir));
    }

    /** @brief Hands @p callback every member equal to @p comparable, which for a unique-key store is one or none. */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept
        requires inner_matches_equals_k
    {
        shared_lock _ {mutex_};
        return inner_store_.equal_range(std::forward<comparable_type_>(comparable),
                                        std::forward<callback_type_>(callback));
    }

    /** @brief How many keys the ordinal surface indexes, which is what @c select counts against. */
    [[nodiscard]] std::size_t ranked_size() const noexcept
        requires inner_counts_ranked_k
    {
        shared_lock _ {mutex_};
        return inner_store_.ranked_size();
    }

    /** @brief How many versions the store holds across every key, published and staged alike. */
    [[nodiscard]] std::size_t versions_count() const noexcept
        requires inner_counts_versions_k
    {
        shared_lock _ {mutex_};
        return inner_store_.versions_count();
    }

    /** @brief How many versions of @p comparable the store still holds. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t versions_count(comparable_type_ const &comparable) const noexcept
        requires inner_counts_versions_k
    {
        shared_lock _ {mutex_};
        return inner_store_.versions_count(comparable);
    }

    /** @brief The newest snapshot no open transaction sits below, which is what @c vacuum prunes to. */
    [[nodiscard]] generation_t low_water_mark() const noexcept
        requires inner_marks_low_water_k
    {
        shared_lock _ {mutex_};
        return inner_store_.low_water_mark();
    }

    /** @brief Writes @p element whether or not its key is taken, spelled as the inner store spells it. */
    [[nodiscard]] status_t insert_or_assign(value_t &&element) noexcept
        requires inner_assigns_over_key_k
    {
        unique_lock _ {mutex_};
        return inner_store_.insert_or_assign(std::move(element));
    }

#pragma region Sharded Membership

    /**
     *  @brief Draws every stamp and every snapshot from @p clock rather than from the wrapped store's own.
     *    Called by an outer shard set while it owns this store outright, before any transaction is open,
     *    so it takes the mutex for symmetry rather than for a race it could lose.
     */
    void attach_clock(clock_t &clock) noexcept
        requires inner_shares_clock_k
    {
        unique_lock _ {mutex_};
        inner_store_.attach_clock(clock);
    }

    /** @brief Opens one part of a sharded transaction on a @p snapshot and @p generation drawn elsewhere. */
    [[nodiscard]] expected<transaction_t> transaction_at(generation_t snapshot, generation_t generation) noexcept
        requires inner_shares_clock_k
    {
        unique_lock _ {mutex_};
        auto opened = inner_store_.transaction_at(snapshot, generation);
        if (!opened) return opened.status();
        return transaction_t {*this, std::move(*opened)};
    }

#pragma endregion Sharded Membership
};

#pragma region Aliases

/**
 *  @brief A set-shaped store behind one shared mutex, spelled @c locked_set<monotonic_avl_set<key_t>>.
 *
 *  The wrapper is one class whichever shape it holds, so the shape is a constraint rather than a
 *  second spelling: naming the set alias over a store of mappings has no substitution, and neither
 *  does naming @c locked_map over a store of plain keys. Two aliases that accept the same arguments
 *  would name the same type and catch nothing.
 */
template <set_shaped_store set_store_type_, typename shared_mutex_type_ = spin_shared_mutex>
using locked_set = locked_store<set_store_type_, shared_mutex_type_>;

/** @brief A map-shaped store behind one shared mutex, spelled @c locked_map<monotonic_avl_map<key_t, value_t>>. */
template <map_shaped_store map_store_type_, typename shared_mutex_type_ = spin_shared_mutex>
using locked_map = locked_store<map_store_type_, shared_mutex_type_>;

#pragma endregion Aliases

} // namespace ashvardanian::smashtable
