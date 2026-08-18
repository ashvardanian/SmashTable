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
 *  that opened it, and its staged changes live outside the mutex entirely. Only the points where a
 *  transaction reaches the store - @c watch, @c stage, @c commit, @c rollback, @c find - take the lock.
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
    using self_t = locked_store;
    using inner_store_t = store_type_;
    using inner_transaction_t = typename inner_store_t::transaction_t;
    using mutex_t = shared_mutex_type_;

    using value_t = typename inner_store_t::value_t;
    using value_type = value_t; // ? STL style
    using key_type = typename mapping_key_type_or_itself<value_t>::type;
    using mapped_type = typename mapped_value_type_or_void<value_t>::type;
    using is_associative = std::bool_constant<is_mapping<value_t>>;
    using is_transactional = std::true_type;
    using callback_reads = std::true_type;

    /** @brief One mutex serializes whole transactions, so the inner store's promise carries over intact. */
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

    class transaction_t {
        friend class locked_store;
        /**
         *  @brief The store this transaction reaches through, held by pointer rather than reference.
         *    A reference member deletes the defaulted move assignment, and a transaction that moves but
         *    cannot be move-assigned is one no container can hold.
         */
        locked_store *store_;
        /** @brief The inner store's own transaction, which stages entirely outside the mutex. */
        inner_transaction_t unlocked_;
        static_assert(std::is_nothrow_move_constructible<inner_transaction_t>());

      public:
        transaction_t(locked_store &db, inner_transaction_t &&unlocked) noexcept
            : store_(&db), unlocked_(std::move(unlocked)) {}
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        generation_t generation() const noexcept { return unlocked_.generation(); }

        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            shared_lock _ {store_->mutex_};
            return unlocked_.watch(id);
        }

        [[nodiscard]] status_t reserve(std::size_t size) noexcept { return unlocked_.reserve(size); }
        [[nodiscard]] status_t upsert(value_t &&element) noexcept { return unlocked_.upsert(std::move(element)); }
        [[nodiscard]] status_t erase(identifier_t const &id) noexcept { return unlocked_.erase(id); }

        /**
         *  @brief Stages @p element only if its key is free, refusing rather than writing over it.
         *    Takes the lock, unlike @c upsert: a strict insert reads the store to learn whether the
         *    key is already there, and only staging itself lives outside the mutex.
         */
        [[nodiscard]] status_t insert(value_t &&element) noexcept
            requires inner_transaction_refuses_occupied_key_k
        {
            shared_lock _ {store_->mutex_};
            return unlocked_.insert(std::move(element));
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
            return unlocked_.insert(std::move(element), std::forward<callback_inserted_type_>(callback_inserted),
                                    std::forward<callback_existing_type_>(callback_existing));
        }

        /** @brief Stages @p element only if its key is already taken, refusing to create one. */
        [[nodiscard]] status_t update(value_t &&element) noexcept
            requires inner_transaction_refuses_absent_key_k
        {
            shared_lock _ {store_->mutex_};
            return unlocked_.update(std::move(element));
        }

        [[nodiscard]] status_t stage() noexcept {
            unique_lock _ {store_->mutex_};
            return unlocked_.stage();
        }

        [[nodiscard]] status_t reset() noexcept {
            unique_lock _ {store_->mutex_};
            return unlocked_.reset();
        }

        [[nodiscard]] status_t rollback() noexcept {
            unique_lock _ {store_->mutex_};
            return unlocked_.rollback();
        }

        [[nodiscard]] status_t commit() noexcept {
            unique_lock _ {store_->mutex_};
            return unlocked_.commit();
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
            shared_lock _ {store_->mutex_};
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

        /** @brief Whether @p comparable is there, including this transaction's own writes. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
            shared_lock _ {store_->mutex_};
            return unlocked_.contains(std::forward<comparable_type_>(comparable));
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
            return unlocked_.find_and_watch(std::forward<comparable_type_>(comparable),
                                            std::forward<callback_found_type_>(callback_found),
                                            std::forward<callback_missing_type_>(callback_missing));
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept
            requires inner_transaction_is_ordered_k
        {
            shared_lock _ {store_->mutex_};
            unlocked_.upper_bound(std::forward<comparable_type_>(comparable),
                                  std::forward<callback_found_type_>(callback_found),
                                  std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Hands @p callback_found the first member at or after @p comparable, or reports none. */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept
            requires inner_transaction_lower_bounds_k
        {
            shared_lock _ {store_->mutex_};
            unlocked_.lower_bound(std::forward<comparable_type_>(comparable),
                                  std::forward<callback_found_type_>(callback_found),
                                  std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Hands @p callback every member equal to @p comparable, this transaction's writes included. */
        template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
        void equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept
            requires inner_transaction_matches_equals_k
        {
            shared_lock _ {store_->mutex_};
            unlocked_.equal_range(std::forward<comparable_type_>(comparable), std::forward<callback_type_>(callback));
        }

        /** @brief Hands @p callback every member of [ @p lower, @p upper ), this transaction's writes included. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires inner_transaction_walks_range_k
        {
            shared_lock _ {store_->mutex_};
            unlocked_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                            std::forward<callback_type_>(callback));
        }

        /** @brief Hands @p callback every member this transaction reads, in whatever order the store keeps. */
        template <typename callback_type_ = no_op_t>
        void for_each(callback_type_ &&callback) const noexcept
            requires inner_transaction_enumerates_k
        {
            shared_lock _ {store_->mutex_};
            unlocked_.for_each(std::forward<callback_type_>(callback));
        }

        /** @brief Stages @p element only if its key is free, leaving an incumbent untouched. */
        [[nodiscard]] status_t insert_if_missing(value_t &&element) noexcept
            requires inner_transaction_skips_occupied_key_k
        {
            shared_lock _ {store_->mutex_};
            return unlocked_.insert_if_missing(std::move(element));
        }

        /** @brief Whether anything is staged, which is transaction-local and needs no lock. */
        [[nodiscard]] bool has_changes() const noexcept
            requires inner_transaction_reports_changes_k
        {
            return unlocked_.has_changes();
        }

        /** @brief How many writes are staged, which is transaction-local and needs no lock. */
        [[nodiscard]] std::size_t changes_count() const noexcept
            requires inner_transaction_reports_changes_k
        {
            return unlocked_.changes_count();
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
            requires inner_transaction_shards_k
        {
            shared_lock _ {store_->mutex_};
            return unlocked_.validate_for_commit();
        }

        /** @brief Publishes everything staged under @p stamp, which the shard set drew for the whole commit. */
        void publish_under(commit_stamp_t stamp) noexcept
            requires inner_transaction_shards_k
        {
            unique_lock _ {store_->mutex_};
            unlocked_.publish_under(stamp);
        }

        /** @brief Moves where this part reads, which is transaction-local and touches no store. */
        void adopt_snapshot(generation_t snapshot) noexcept
            requires inner_transaction_shards_k
        {
            unlocked_.adopt_snapshot(snapshot);
        }

        /** @brief Frees whatever the commit just published left unreachable. */
        void prune_committed() noexcept
            requires inner_transaction_shards_k
        {
            unique_lock _ {store_->mutex_};
            unlocked_.prune_committed();
        }

        /** @brief Discards everything staged and pending, at a snapshot the shard set drew once. */
        [[nodiscard]] status_t reset_at(generation_t snapshot) noexcept
            requires inner_transaction_shards_k
        {
            unique_lock _ {store_->mutex_};
            return unlocked_.reset_at(snapshot);
        }

#pragma endregion Sharded Commit
    };

  private:
    mutable mutex_t mutex_;
    inner_store_t unlocked_;

    locked_store(inner_store_t &&unlocked) noexcept : unlocked_(std::move(unlocked)) {}
    /** @warning @p other must have no open transaction and no other thread touching it; see the class note. */
    locked_store &operator=(locked_store &&other) noexcept {
        unique_lock _ {mutex_};
        unlocked_ = std::move(other.unlocked_);
        return *this;
    }

  public:
    locked_store() noexcept = default;
    /** @warning @p other must have no open transaction and no other thread touching it; see the class note. */
    locked_store(locked_store &&other) noexcept : unlocked_(std::move(other.unlocked_)) {}

    [[nodiscard]] std::size_t size() const noexcept {
        shared_lock _ {mutex_};
        return unlocked_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        shared_lock _ {mutex_};
        return unlocked_.empty();
    }

    /**
     *  @brief Builds the inner store from @p arguments and takes ownership of it.
     *    Forwards whatever the core needs - a comparator for a tree, a hasher and equality for a table -
     *    so a store whose comparator has no default constructor is still constructible.
     */
    template <typename... arguments_type_>
    [[nodiscard]] static expected<locked_store> make(arguments_type_ &&...arguments) noexcept {
        expected<locked_store> result;
        if (expected<inner_store_t> unlocked = inner_store_t::make(std::forward<arguments_type_>(arguments)...);
            unlocked)
            result = locked_store {std::move(*unlocked)};
        return result;
    }

    [[nodiscard]] expected<transaction_t> transaction() noexcept {
        expected<transaction_t> result;
        unique_lock _ {mutex_};
        if (auto unlocked = unlocked_.transaction(); unlocked) result = transaction_t {*this, std::move(*unlocked)};
        return result;
    }

    [[nodiscard]] status_t upsert(value_t &&element) noexcept {
        unique_lock _ {mutex_};
        return unlocked_.upsert(std::forward<value_t>(element));
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
        return unlocked_.erase(std::forward<comparable_type_>(comparable),
                               std::forward<callback_found_type_>(callback_found),
                               std::forward<callback_missing_type_>(callback_missing));
    }

    /** @brief Inserts @p element only if its key is absent, leaving an incumbent untouched. */
    [[nodiscard]] status_t insert_if_missing(value_t &&element) noexcept {
        unique_lock _ {mutex_};
        return unlocked_.insert_if_missing(std::move(element));
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
        return unlocked_.insert_if_missing(std::move(element), std::forward<callback_inserted_type_>(callback_inserted),
                                           std::forward<callback_existing_type_>(callback_existing));
    }

    /** @brief Inserts @p element only if its key is free, refusing with the inner store's own status. */
    [[nodiscard]] status_t insert(value_t &&element) noexcept
        requires inner_refuses_occupied_key_k
    {
        unique_lock _ {mutex_};
        return unlocked_.insert(std::move(element));
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
        return unlocked_.insert(std::move(element), std::forward<callback_inserted_type_>(callback_inserted),
                                std::forward<callback_existing_type_>(callback_existing));
    }

    /** @brief Writes @p element only if its key is already taken, refusing to create one. */
    [[nodiscard]] status_t update(value_t &&element) noexcept
        requires inner_refuses_absent_key_k
    {
        unique_lock _ {mutex_};
        return unlocked_.update(std::move(element));
    }

    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        unique_lock _ {mutex_};
        return unlocked_.upsert(begin, end);
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept {
        shared_lock _ {mutex_};
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
        unique_lock _ {mutex_};
        return unlocked_.insert_if_missing(begin, end);
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ordered_k
    {
        shared_lock _ {mutex_};
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
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
        requires inner_is_ordered_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    /** @brief Copies out the first element ordered strictly after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
        requires inner_is_ordered_k
    {
        expected<value_t> result {status_t::key_not_found_k};
        upper_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, []() noexcept {});
        return result;
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ordered_k
    {
        shared_lock _ {mutex_};
        unlocked_.upper_bound(std::forward<comparable_type_>(comparable),
                              std::forward<callback_found_type_>(callback_found),
                              std::forward<callback_missing_type_>(callback_missing));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
        requires inner_is_ordered_k
    {
        shared_lock _ {mutex_};
        unlocked_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                        std::forward<callback_type_>(callback));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                       callback_type_ &&callback = {}) noexcept
        requires inner_is_ordered_k
    {
        unique_lock _ {mutex_};
        return unlocked_.erase_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                     std::forward<callback_type_>(callback));
    }

    /** @brief Erases every element at or after @p lower, reporting each to @p callback. */
    template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback = {}) noexcept
        requires inner_erases_open_range_k
    {
        unique_lock _ {mutex_};
        return unlocked_.erase_from(std::forward<lower_type_>(lower), std::forward<callback_type_>(callback));
    }

    /** @brief Erases every element before @p upper, reporting each to @p callback. */
    template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires inner_erases_open_range_k
    {
        unique_lock _ {mutex_};
        return unlocked_.erase_up_to(std::forward<upper_type_>(upper), std::forward<callback_type_>(callback));
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
        // The inner walk answers either @c status_t or nothing at all, and the wrapper has to name one
        // return type. The wider of the two loses nothing: a walk that cannot refuse always succeeds.
        if constexpr (std::is_void<decltype(unlocked_.update_range(std::forward<lower_type_>(lower),
                                                                   std::forward<upper_type_>(upper),
                                                                   std::forward<callback_type_>(callback)))>()) {
            unlocked_.update_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                   std::forward<callback_type_>(callback));
            return success_k;
        }
        else
            return unlocked_.update_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
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
    void for_each(callback_type_ &&callback) const noexcept
        requires inner_enumerates_k
    {
        shared_lock _ {mutex_};
        unlocked_.for_each(std::forward<callback_type_>(callback));
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
        return unlocked_.vacuum();
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
        return unlocked_.vacuum(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper));
    }

    /**
     *  @brief Hands @p callback_found the element at zero-based position @p ordinal.
     *  @param[in] callback_missing Fires when fewer elements are there. Must be @c noexcept.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    void select(std::size_t ordinal, callback_found_type_ &&callback_found,
                callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ranked_k
    {
        shared_lock _ {mutex_};
        unlocked_.select(ordinal, std::forward<callback_found_type_>(callback_found),
                         std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief Hands @p callback_found how many elements the store orders before @p comparable.
     *  @param[in] callback_missing Fires when @p comparable is not there at all. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept
        requires inner_is_ranked_k
    {
        shared_lock _ {mutex_};
        unlocked_.rank(std::forward<comparable_type_>(comparable), std::forward<callback_found_type_>(callback_found),
                       std::forward<callback_missing_type_>(callback_missing));
    }

    [[nodiscard]] status_t clear() noexcept {
        unique_lock _ {mutex_};
        return unlocked_.clear();
    }

    [[nodiscard]] status_t reserve(std::size_t size) noexcept {
        unique_lock _ {mutex_};
        return unlocked_.reserve(size);
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    void sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                    callback_type_ &&callback) const noexcept
        requires inner_samples_one_k
    {
        shared_lock _ {mutex_};
        unlocked_.sample_one(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                             std::forward<generator_type_>(generator), std::forward<callback_type_>(callback));
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    void sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator, std::size_t &seen,
                          std::size_t reservoir_capacity, output_iterator_type_ &&reservoir) const noexcept
        requires inner_samples_reservoir_k
    {
        shared_lock _ {mutex_};
        unlocked_.sample_reservoir(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                   std::forward<generator_type_>(generator), seen, reservoir_capacity,
                                   std::forward<output_iterator_type_>(reservoir));
    }

    /** @brief Hands @p callback every member equal to @p comparable, which for a unique-key store is one or none. */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
    void equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept
        requires inner_matches_equals_k
    {
        shared_lock _ {mutex_};
        unlocked_.equal_range(std::forward<comparable_type_>(comparable), std::forward<callback_type_>(callback));
    }

    /** @brief How many keys the ordinal surface indexes, which is what @c select counts against. */
    [[nodiscard]] std::size_t ranked_size() const noexcept
        requires inner_counts_ranked_k
    {
        shared_lock _ {mutex_};
        return unlocked_.ranked_size();
    }

    /** @brief How many versions the store holds across every key, published and staged alike. */
    [[nodiscard]] std::size_t versions_count() const noexcept
        requires inner_counts_versions_k
    {
        shared_lock _ {mutex_};
        return unlocked_.versions_count();
    }

    /** @brief How many versions of @p comparable the store still holds. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t versions_count(comparable_type_ const &comparable) const noexcept
        requires inner_counts_versions_k
    {
        shared_lock _ {mutex_};
        return unlocked_.versions_count(comparable);
    }

    /** @brief The newest snapshot no open transaction sits below, which is what @c vacuum prunes to. */
    [[nodiscard]] generation_t low_water_mark() const noexcept
        requires inner_marks_low_water_k
    {
        shared_lock _ {mutex_};
        return unlocked_.low_water_mark();
    }

    /** @brief Writes @p element whether or not its key is taken, spelled as the inner store spells it. */
    [[nodiscard]] status_t insert_or_assign(value_t &&element) noexcept
        requires inner_assigns_over_key_k
    {
        unique_lock _ {mutex_};
        return unlocked_.insert_or_assign(std::move(element));
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
        unlocked_.attach_clock(clock);
    }

    /** @brief Opens one part of a sharded transaction on a @p snapshot and @p generation drawn elsewhere. */
    [[nodiscard]] expected<transaction_t> transaction_at(generation_t snapshot, generation_t generation) noexcept
        requires inner_shares_clock_k
    {
        expected<transaction_t> result;
        unique_lock _ {mutex_};
        if (auto unlocked = unlocked_.transaction_at(snapshot, generation); unlocked)
            result = transaction_t {*this, std::move(*unlocked)};
        return result;
    }

#pragma endregion Sharded Membership
};

#pragma region Aliases

/**
 *  @brief A set-shaped store behind one shared mutex, spelled @c locked_set<monotonic_avl_set<key_t>>.
 *    The element type comes from the store being wrapped, so this alias and @c locked_map build the
 *    same thing; what they add is the shape at the point of use, which the store families already name.
 */
template <typename set_store_type_, typename shared_mutex_type_ = spin_shared_mutex>
using locked_set = locked_store<set_store_type_, shared_mutex_type_>;

/** @brief A map-shaped store behind one shared mutex, spelled @c locked_map<monotonic_avl_map<key_t, value_t>>. */
template <typename map_store_type_, typename shared_mutex_type_ = spin_shared_mutex>
using locked_map = locked_store<map_store_type_, shared_mutex_type_>;

#pragma endregion Aliases

} // namespace ashvardanian::smashtable
