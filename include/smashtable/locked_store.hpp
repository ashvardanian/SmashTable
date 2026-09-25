/**
 *  @file include/smashtable/locked_store.hpp
 *  @author Ash Vardanian
 *  @date October 13, 2022
 *  @brief Wraps any transactional store behind one shared mutex, making the store thread-safe while
 *      its transactions stay single-threaded.
 */
#pragma once
#include <cassert> // `assert`

#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief Wraps and protects any transactional store under a shared mutex.
 *
 *  The store becomes @b thread-safe; a single transaction does @b not - it belongs to the thread
 *  that opened it, and its staged changes live outside the mutex entirely. The rule is one line and
 *  holds for every entry point, present and future: a call takes the lock exactly when it reaches
 *  the wrapped store, shared where it only reads that store and exclusive where it writes to it,
 *  and a call answered from the transaction's own staged state takes no lock at all.
 *
 *  One mutex means one commit in flight, so the order this store makes for itself is sized for
 *  exactly that, and the wrapped store is rebound to whichever order this one is a member of.
 *
 *  @c stage and @c commit take the mutex separately and drop it in between, so the lock is not what
 *  spans them. The store-side reservation is: staging claims every key the transaction wrote, and
 *  no other transaction can take those keys until this one publishes or unwinds, which is what
 *  carries the inner store's isolation level across the gap.
 *
 *  @warning Every callback runs with the mutex held, and it is not recursive, so a callback that
 *      calls back into this store - or into anything that eventually does - deadlocks against
 *      itself. Copy out what a callback needs and do the rest after it returns.
 *  @warning Moving a store leaves every open transaction pointing at the husk, whose contents are
 *      empty and whose mutex guards nothing, so commits land nowhere and report success. Move only
 *      a store no transaction is open on and no other thread is touching.
 *
 *  A value's destructor also runs under the mutex wherever the store displaces or reclaims what it
 *  holds, so a value whose teardown can re-enter must record the drop and perform it afterwards.
 */
template <typename store_type_, typename shared_mutex_type_ = spin_shared_mutex_t,
          typename order_type_ = basic_commit_order<1>>
class locked_store {

  public:
    using store_t = locked_store;
    using order_t = order_type_;

    /** The same store in another order, which is how a shard set builds its partitions into its
     *  own. */
    template <typename other_order_type_>
    using rebind_order = locked_store<store_type_, shared_mutex_type_, other_order_type_>;

    using inner_store_t = typename rebound_order_of<store_type_, order_t>::type;
    using inner_transaction_t = typename inner_store_t::transaction_t;
    using mutex_t = shared_mutex_type_;

    using value_t = typename inner_store_t::value_t;
    using value_type = value_t; // ? STL style
    using key_type = typename mapping_key_type_or_itself<value_t>::type;
    using mapped_type = typename mapped_value_type_or_void<value_t>::type;
    using is_associative = std::bool_constant<is_mapping<value_t>>;
    using is_transactional = std::true_type;

    /** The staging reservation carries the inner store's promise over, not the mutex; see the class
     *  note. */
    static constexpr isolation_t isolation_k = inner_store_t::isolation_k;

    using comparator_t = typename inner_store_t::comparator_t;
    using identifier_t = typename inner_store_t::identifier_t;
    using generation_t = typename inner_store_t::generation_t;

    /** Whether the wrapped transaction can be driven as one part of a commit under a stamp drawn
     *  elsewhere. */
    static constexpr bool inner_transaction_shards_k = shards_its_commit<inner_transaction_t>;

    /** Whether the wrapped transaction decides and writes in two steps rather than one. */
    static constexpr bool inner_transaction_splits_commit_k = splits_its_commit<inner_transaction_t>;

    class transaction_t {
        friend class locked_store;

        /** The store this transaction reaches through, never null after construction. A pointer
         *  rather than a reference, which would delete the defaulted move assignment. */
        locked_store *store_;

        /** The inner store's own transaction, which stages entirely outside the mutex. */
        inner_transaction_t inner_transaction_;

        /** The hold @c validate_for_commit leaves on the mutex for the publication, rollback or
         *  reset that answers it; empty between such pairs, and released with the transaction. */
        unique_lock<mutex_t> validation_;
        static_assert(std::is_nothrow_move_constructible<inner_transaction_t>());

        /** The mutex for one call: the hold a validation left, or a fresh one. */
        unique_lock<mutex_t> held_() noexcept {
            return validation_ ? std::move(validation_) : unique_lock<mutex_t> {store_->mutex_};
        }

      public:
        transaction_t(locked_store &db, inner_transaction_t &&inner_transaction) noexcept
            : store_(&db), inner_transaction_(std::move(inner_transaction)) {}
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        generation_t generation() const noexcept { return inner_transaction_.generation(); }

        /** The snapshot the wrapped transaction reads at, which is transaction-local. */
        [[nodiscard]] generation_t snapshot() const noexcept
            requires requires(inner_transaction_t const &transaction) { transaction.snapshot(); }
        {
            return inner_transaction_.snapshot();
        }

        /** The stamp the wrapped transaction's last commit published under, which is
         *  transaction-local. */
        [[nodiscard]] generation_t commit_stamp() const noexcept
            requires requires(inner_transaction_t const &transaction) { transaction.commit_stamp(); }
        {
            return inner_transaction_.commit_stamp();
        }

        status_t watch(identifier_t const &id) noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.watch(id);
        }

        status_t reserve(std::size_t size) noexcept { return inner_transaction_.reserve(size); }
        status_t upsert(value_t &&element) noexcept { return inner_transaction_.upsert(std::move(element)); }

        /** Stages an erase, reporting @c key_not_found_k when this transaction has no such key. */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        status_t erase(identifier_t const &id, callback_found_type_ &&callback_found = {},
                       callback_missing_type_ &&callback_missing = {}) noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.erase(id, std::forward<callback_found_type_>(callback_found),
                                            std::forward<callback_missing_type_>(callback_missing));
        }

        /** Stages @p element only if its key is free, refusing rather than writing over it. Takes
         *  the lock, unlike @c upsert: a strict insert reads the store to learn whether the key is
         *  already there, and only staging itself lives outside the mutex. */
        status_t insert(value_t &&element) noexcept
            requires transaction_offers_insert<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.insert(std::move(element));
        }

        /**
         *  @brief Stages @p element only if its key is free, and says which branch was taken.
         *  @param[in] callback_inserted Receives the element once staged.
         *  @param[in] callback_existing Receives the element present, which refused the insert.
         */
        template <typename callback_inserted_type_, typename callback_existing_type_>
        status_t insert(value_t &&element, callback_inserted_type_ &&callback_inserted,
                        callback_existing_type_ &&callback_existing) noexcept
            requires transaction_offers_insert_naming_occupant<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.insert(std::move(element),
                                             std::forward<callback_inserted_type_>(callback_inserted),
                                             std::forward<callback_existing_type_>(callback_existing));
        }

        /** Stages @p element only if its key is already taken, refusing to create one. */
        status_t update(value_t &&element) noexcept
            requires transaction_offers_update<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.update(std::move(element));
        }

        status_t stage() noexcept {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.stage();
        }

        status_t reset() noexcept {
            unique_lock<mutex_t> const _ = held_();
            return inner_transaction_.reset();
        }

        status_t rollback() noexcept {
            unique_lock<mutex_t> const _ = held_();
            return inner_transaction_.rollback();
        }

        status_t commit() noexcept {
            unique_lock<mutex_t> const _ = held_();
            return inner_transaction_.commit();
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                      callback_missing_type_ &&callback_missing = {}) const noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.find(std::forward<comparable_type_>(comparable),
                                           std::forward<callback_found_type_>(callback_found),
                                           std::forward<callback_missing_type_>(callback_missing));
        }

        /** Copies out the member equal to @p comparable, including this transaction's writes. */
        template <typename comparable_type_ = identifier_t>
        expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            status_t const looked_up = find(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            if (failed(looked_up)) return looked_up;
            return result;
        }

        /** Whether @p comparable is there, including this transaction's own writes. */
        template <typename comparable_type_ = identifier_t>
        expected<bool> contains(comparable_type_ &&comparable) const noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.contains(std::forward<comparable_type_>(comparable));
        }

        /** Finds the member equal to @p comparable and records what it saw into the read set. The
         *  watching twin of @c find, and the one that can fail because a read set is memory. */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        status_t find_and_watch(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) noexcept {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.find_and_watch(std::forward<comparable_type_>(comparable),
                                                     std::forward<callback_found_type_>(callback_found),
                                                     std::forward<callback_missing_type_>(callback_missing));
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                             callback_missing_type_ &&callback_missing = {}) const noexcept
            requires transaction_offers_upper_bound<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.upper_bound(std::forward<comparable_type_>(comparable),
                                                  std::forward<callback_found_type_>(callback_found),
                                                  std::forward<callback_missing_type_>(callback_missing));
        }

        /** Hands @p callback_found the smallest member this transaction reads, or reports none. */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        status_t smallest(callback_found_type_ &&callback_found,
                          callback_missing_type_ &&callback_missing = {}) const noexcept
            requires transaction_offers_smallest<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.smallest(std::forward<callback_found_type_>(callback_found),
                                               std::forward<callback_missing_type_>(callback_missing));
        }

        /**
         *  @brief Hands @p callback_found the member this transaction reads at zero-based
         *      @p ordinal.
         *  @param[in] callback_missing Fires when fewer members are there. Must be @c noexcept.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                        callback_missing_type_ &&callback_missing = {}) const noexcept
            requires transaction_offers_select<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.select(ordinal, std::forward<callback_found_type_>(callback_found),
                                             std::forward<callback_missing_type_>(callback_missing));
        }

        /**
         *  @brief Hands @p callback_found how many members this transaction orders before
         *      @p comparable.
         *  @param[in] callback_missing Fires when @p comparable is not there at all. Must be
         *      @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                      callback_missing_type_ &&callback_missing = {}) const noexcept
            requires transaction_offers_rank<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.rank(std::forward<comparable_type_>(comparable),
                                           std::forward<callback_found_type_>(callback_found),
                                           std::forward<callback_missing_type_>(callback_missing));
        }

        /** Hands @p callback_found the first member at or after @p comparable, or reports none. */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                             callback_missing_type_ &&callback_missing = {}) const noexcept
            requires transaction_offers_lower_bound<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.lower_bound(std::forward<comparable_type_>(comparable),
                                                  std::forward<callback_found_type_>(callback_found),
                                                  std::forward<callback_missing_type_>(callback_missing));
        }

        /**
         *  @brief Hands @p callback every member equal to @p comparable, writes included.
         *  @note A callback answering @c walk_control_t stops the walk where it says to.
         */
        template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
        status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept
            requires transaction_offers_equal_range<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.equal_range(std::forward<comparable_type_>(comparable),
                                                  std::forward<callback_type_>(callback));
        }

        /**
         *  @brief Hands @p callback every member of [ @p lower, @p upper ), writes included.
         *  @note A callback answering @c walk_control_t stops the walk where it says to.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires transaction_offers_range<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                            std::forward<callback_type_>(callback));
        }

        /**
         *  @brief Hands @p callback every member at or after @p lower, with no upper end.
         *  @note A callback answering @c walk_control_t stops the walk where it says to.
         */
        template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
        status_t range_from(lower_type_ &&lower, callback_type_ &&callback) const noexcept
            requires transaction_offers_range_from<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.range_from(std::forward<lower_type_>(lower),
                                                 std::forward<callback_type_>(callback));
        }

        /**
         *  @brief Hands @p callback every member before @p upper, excluded, with no lower end.
         *  @note A callback answering @c walk_control_t stops the walk where it says to.
         */
        template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
        status_t range_up_to(upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires transaction_offers_range_up_to<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.range_up_to(std::forward<upper_type_>(upper),
                                                  std::forward<callback_type_>(callback));
        }

        /**
         *  @brief Hands @p callback every member this transaction reads, in the store's own order.
         *  @note A callback answering @c walk_control_t stops the walk where it says to.
         */
        template <typename callback_type_ = no_op_t>
        status_t for_each(callback_type_ &&callback) const noexcept
            requires transaction_offers_for_each<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.for_each(std::forward<callback_type_>(callback));
        }

        /** Stages @p element only if its key is free, leaving an incumbent untouched. */
        status_t insert_if_missing(value_t &&element) noexcept
            requires transaction_offers_insert_if_missing<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.insert_if_missing(std::move(element));
        }

#pragma region Transaction Range Operations

        /** How many members equal @p comparable, this transaction's own writes included. */
        template <typename comparable_type_ = identifier_t>
        expected<std::size_t> count(comparable_type_ &&comparable) const noexcept
            requires transaction_offers_count<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.count(std::forward<comparable_type_>(comparable));
        }

        /** Copies out the first member at or after @p comparable. */
        template <typename comparable_type_ = identifier_t>
        expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
            requires transaction_offers_lower_bound_copy<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.lower_bound_copy(std::forward<comparable_type_>(comparable));
        }

        /** Copies out the first member after @p comparable. */
        template <typename comparable_type_ = identifier_t>
        expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
            requires transaction_offers_upper_bound_copy<inner_store_t>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.upper_bound_copy(std::forward<comparable_type_>(comparable));
        }

        /** Stages a tombstone for every member in [ @p lower, @p upper ). */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires transaction_offers_erase_range<inner_store_t>
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.erase_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                                  std::forward<callback_type_>(callback));
        }

        /** Stages a tombstone for every member at or after @p lower. */
        template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
        status_t erase_from(lower_type_ &&lower, callback_type_ &&callback) noexcept
            requires transaction_offers_erase_from<inner_store_t>
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.erase_from(std::forward<lower_type_>(lower),
                                                 std::forward<callback_type_>(callback));
        }

        /** Stages a tombstone for every member before @p upper. */
        template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
        status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires transaction_offers_erase_up_to<inner_store_t>
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.erase_up_to(std::forward<upper_type_>(upper),
                                                  std::forward<callback_type_>(callback));
        }

        /** Stages a tombstone for every member this transaction reads. */
        status_t clear() noexcept
            requires transaction_offers_clear<inner_store_t>
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.clear();
        }

        /** Hands @p callback each member in [ @p lower, @p upper ) to revise, and stages the
         *  result. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        status_t update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires transaction_offers_update_range<inner_store_t>
        {
            unique_lock _ {store_->mutex_};
            return inner_transaction_.update_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                                   std::forward<callback_type_>(callback));
        }

        /** Draws one member uniformly from [ @p lower, @p upper ). */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename callback_type_ = no_op_t>
        status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                            callback_type_ &&callback) const noexcept
            requires transaction_offers_sample_one<inner_store_t> &&
                     uniform_random_bits<std::remove_cvref_t<generator_type_>>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.sample_one(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                                 std::forward<generator_type_>(generator),
                                                 std::forward<callback_type_>(callback));
        }

        /** Fills @p reservoir with up to @p capacity members drawn from [ @p lower, @p upper ). */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename output_iterator_type_ = no_op_t>
        status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                  std::size_t &seen, std::size_t capacity,
                                  output_iterator_type_ &&reservoir) const noexcept
            requires transaction_offers_sample_reservoir<inner_store_t> &&
                     uniform_random_bits<std::remove_cvref_t<generator_type_>>
        {
            shared_lock _ {store_->mutex_};
            return inner_transaction_.sample_reservoir(std::forward<lower_type_>(lower),
                                                       std::forward<upper_type_>(upper),
                                                       std::forward<generator_type_>(generator), seen, capacity,
                                                       std::forward<output_iterator_type_>(reservoir));
        }

#pragma endregion Transaction Range Operations

        /** Whether anything is staged, which is transaction-local and needs no lock. */
        [[nodiscard]] bool has_changes() const noexcept
            requires transaction_offers_has_changes<inner_store_t>
        {
            return inner_transaction_.has_changes();
        }

        /** How many writes are staged, which is transaction-local and needs no lock. */
        [[nodiscard]] std::size_t changes_count() const noexcept
            requires transaction_offers_changes_count<inner_store_t>
        {
            return inner_transaction_.changes_count();
        }

#pragma region Sharded Commit

        /**
         *  @brief The part of a commit an outer shard set or a transaction group drives itself, one
         *      call per phase.
         *
         *  A caller spanning several stores asks every part whether it may proceed, publishes all
         *  of them, then reclaims - so the phases are separate calls rather than one @c commit. The
         *  validation takes this store's mutex exclusively and keeps it until the publication, the
         *  rollback or the reset that follows: a writer slipping in between would move a watched
         *  key after the answer, which is the lost update the watch was taken against. Callers take
         *  the stores in ascending address, so two of them holding across the phases cannot
         *  deadlock; a shard set holds its partition locks around all of this, and the two orders
         *  never cross, since a partition lock is always taken first.
         */
        status_t validate_for_commit() noexcept
            requires(inner_transaction_shards_k || inner_transaction_splits_commit_k)
        {
            validation_ = held_();
            return inner_transaction_.validate_for_commit();
        }

        /** Publishes everything staged under @p stamp, which the shard set drew for the whole
         *  commit. */
        void publish_under(commit_stamp_t stamp) noexcept
            requires inner_transaction_shards_k
        {
            unique_lock<mutex_t> const _ = held_();
            inner_transaction_.publish_under(stamp);
        }

        /** Publishes everything staged, the wrapped store stamping its own versions. */
        void publish_under() noexcept
            requires inner_transaction_splits_commit_k
        {
            unique_lock<mutex_t> const _ = held_();
            inner_transaction_.publish_under();
        }

        /** Moves where this part reads, which is transaction-local and touches no store. */
        void adopt_snapshot(generation_t snapshot) noexcept
            requires inner_transaction_shards_k
        {
            inner_transaction_.adopt_snapshot(snapshot);
        }

        /** Frees whatever the commit just published left unreachable. */
        void prune_committed() noexcept
            requires inner_transaction_shards_k
        {
            unique_lock _ {store_->mutex_};
            inner_transaction_.prune_committed();
        }

        /** Discards everything staged and pending, at a snapshot the shard set drew once. */
        status_t reset_at(generation_t snapshot) noexcept
            requires inner_transaction_shards_k
        {
            unique_lock<mutex_t> const _ = held_();
            return inner_transaction_.reset_at(snapshot);
        }

#pragma endregion Sharded Commit
    };

  private:
    mutable mutex_t mutex_;

    /** The order this store makes for itself when nobody hands it one, declared before the store it
     *  seats. */
    SMASHTABLE_NO_UNIQUE_ADDRESS_ mutable order_t own_order_ {};

    /** The order the wrapped store is a member of, which is the one above unless a caller named
     *  another. */
    order_t *order_ {&own_order_};
    inner_store_t inner_store_;

    /** Seats whichever order this store is a member of in the store it wraps, where there is one to
     *  seat. */
    void seat_order_() noexcept {
        if constexpr (requires { inner_store_.join_order(*order_); }) inner_store_.join_order(*order_);
    }

    locked_store(inner_store_t &&inner_store) noexcept : inner_store_(std::move(inner_store)) { seat_order_(); }

    locked_store(order_t &order, inner_store_t &&inner_store) noexcept
        : order_(&order), inner_store_(std::move(inner_store)) {
        seat_order_();
    }

    /** Takes over whichever order @p other was a member of, and seats the wrapped store in it. */
    void adopt_order_of_(locked_store &other) noexcept {
        if (other.order_ == &other.own_order_) own_order_.adopt(other.own_order_);
        else order_ = other.order_;
        seat_order_();
    }

    /**
     *  @warning @p other must be untouched: no open transaction, no other thread; see class notes.
     */
    locked_store &operator=(locked_store &&other) noexcept {
        unique_lock _ {mutex_};
        inner_store_ = std::move(other.inner_store_);
        adopt_order_of_(other);
        return *this;
    }

  public:
    locked_store() noexcept { seat_order_(); }

    /** Builds a store that is a member of @p order rather than of one it made for itself. */
    explicit locked_store(order_t &order) noexcept : order_(&order) { seat_order_(); }

    /**
     *  @brief Makes this store, and the store it wraps, a member of @p order.
     *
     *  Called once, by the wrapper that owns both, before any transaction opens. A shard set seats
     *  every partition this way, which is what makes one stamp mean the same thing across them.
     */
    void join_order(order_t &order) noexcept {
        assert(order_->open_snapshots() == 0 && "a reader would leave its claim on the order being left");
        order_ = &order;
        seat_order_();
    }

    /**
     *  @warning @p other must be untouched: no open transaction, no other thread; see class notes.
     */
    locked_store(locked_store &&other) noexcept : inner_store_(std::move(other.inner_store_)) {
        adopt_order_of_(other);
    }

    [[nodiscard]] std::size_t size() const noexcept {
        shared_lock _ {mutex_};
        return inner_store_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        shared_lock _ {mutex_};
        return inner_store_.empty();
    }

    /** Builds the inner store from @p arguments and takes ownership of it. Forwards whatever the
     *  core needs - a comparator for a tree, a hasher and equality for a table - so a store whose
     *  comparator has no default constructor is still constructible. */
    template <typename... arguments_type_>
    static expected<locked_store> make(arguments_type_ &&...arguments) noexcept {
        expected<inner_store_t> inner_store = inner_store_t::make(std::forward<arguments_type_>(arguments)...);
        if (!inner_store) return inner_store.status();
        return locked_store {std::move(*inner_store)};
    }

    expected<transaction_t> transaction() noexcept {
        unique_lock _ {mutex_};
        auto opened = inner_store_.transaction();
        if (!opened) return opened.status();
        return transaction_t {*this, std::move(*opened)};
    }

    status_t upsert(value_t &&element) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.upsert(std::forward<value_t>(element));
    }

    /** Erases whatever equals @p comparable, reporting through the callbacks so presence needs no
     *  second probe. Takes any type the inner store compares against, not only an identifier. */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    status_t erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found = {},
                   callback_missing_type_ &&callback_missing = {}) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.erase(std::forward<comparable_type_>(comparable),
                                  std::forward<callback_found_type_>(callback_found),
                                  std::forward<callback_missing_type_>(callback_missing));
    }

    /** Inserts @p element only if its key is absent, leaving an incumbent untouched. */
    status_t insert_if_missing(value_t &&element) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.insert_if_missing(std::move(element));
    }

    /**
     *  @brief Inserts @p element only if its key is absent, and says which branch was taken.
     *  @param[in] callback_inserted Receives the element once stored.
     *  @param[in] callback_existing Receives the element present, which declined the insert.
     */
    template <typename callback_inserted_type_, typename callback_existing_type_>
    status_t insert_if_missing(value_t &&element, callback_inserted_type_ &&callback_inserted,
                               callback_existing_type_ &&callback_existing) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.insert_if_missing(std::move(element),
                                              std::forward<callback_inserted_type_>(callback_inserted),
                                              std::forward<callback_existing_type_>(callback_existing));
    }

    /** Inserts @p element only if its key is free, refusing with the inner store's own status. */
    status_t insert(value_t &&element) noexcept
        requires offers_insert<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.insert(std::move(element));
    }

    /**
     *  @brief Inserts @p element only if its key is free, and says which branch was taken.
     *  @param[in] callback_inserted Receives the element once stored.
     *  @param[in] callback_existing Receives the already-present element that refused the insert.
     */
    template <typename callback_inserted_type_, typename callback_existing_type_>
    status_t insert(value_t &&element, callback_inserted_type_ &&callback_inserted,
                    callback_existing_type_ &&callback_existing) noexcept
        requires offers_insert_naming_occupant<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.insert(std::move(element), std::forward<callback_inserted_type_>(callback_inserted),
                                   std::forward<callback_existing_type_>(callback_existing));
    }

    /** Writes @p element only if its key is already taken, refusing to create one. */
    status_t update(value_t &&element) noexcept
        requires offers_update<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.update(std::move(element));
    }

    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.upsert(begin, end);
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
        shared_lock _ {mutex_};
        return inner_store_.find(std::forward<comparable_type_>(comparable),
                                 std::forward<callback_found_type_>(callback_found),
                                 std::forward<callback_missing_type_>(callback_missing));
    }

    /** Existence check, expressed through @c find so the lock discipline stays in one place. */
    template <typename comparable_type_ = identifier_t>
    expected<bool> contains(comparable_type_ &&comparable) const noexcept {
        bool present = false;
        status_t const answered = find(
            std::forward<comparable_type_>(comparable), [&](value_t const &) noexcept { present = true; }, no_op_t {});
        if (failed(answered)) return answered;
        return present;
    }

    /** Number of elements matching @p comparable, which is 0 or 1 for unique keys. */
    template <typename comparable_type_ = identifier_t>
    expected<std::size_t> count(comparable_type_ &&comparable) const noexcept {
        expected<bool> const present = contains(std::forward<comparable_type_>(comparable));
        if (!present) return present.status();
        return *present ? std::size_t {1} : std::size_t {0};
    }

    /** Inherits the inner store's all-or-nothing batch, held under this store's write lock
     *  throughout. */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    status_t insert_if_missing(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.insert_if_missing(begin, end);
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept
        requires offers_lower_bound<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.lower_bound(std::forward<comparable_type_>(comparable),
                                        std::forward<callback_found_type_>(callback_found),
                                        std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief A resumable walk in ascending key order, holding no lock between its steps.
     *
     *  The position is a key rather than an iterator, so a write between two steps cannot
     *  invalidate it: the next step re-probes from the last key handed over. That is what lets the
     *  walk survive concurrent change, and it is why a step costs a lookup, not an increment.
     */
    class ordered_cursor_t {
        friend class locked_store;

        locked_store const *store_ {nullptr};

        /** The bound the next step is taken from - the one given, then the last key handed over. */
        identifier_t position_;

        /** The key the walk stops before, meaningful only under @c up_to_the_bound_k. */
        identifier_t bound_;
        cursor_seed_t seed_ {cursor_seed_t::the_smallest_k};
        cursor_limit_t limit_ {cursor_limit_t::the_whole_keyspace_k};

        /** Latched by the step that runs out, so @c exhausted answers without probing again. */
        cursor_reach_t reach_ {cursor_reach_t::walking_k};

        explicit ordered_cursor_t(locked_store const &store, cursor_seed_t seed, cursor_limit_t limit) noexcept
            : store_(&store), seed_(seed), limit_(limit) {}

      public:
        ordered_cursor_t() noexcept = default;

        /** Whether the walk is over, which includes having passed the bound it was given. */
        [[nodiscard]] bool exhausted() const noexcept { return !store_ || reach_ == cursor_reach_t::spent_k; }

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
                reach_ = cursor_reach_t::spent_k;
                callback_missing();
                return;
            }
            seed_ = cursor_seed_t::past_the_last_k;
        }
    };

    /** A walk of every member in ascending order, resumable and holding no lock between steps. */
    [[nodiscard]] ordered_cursor_t cursor() const noexcept
        requires offers_both_bounds<inner_store_t> && offers_smallest<inner_store_t>
    {
        return ordered_cursor_t {*this, cursor_seed_t::the_smallest_k, cursor_limit_t::the_whole_keyspace_k};
    }

    /** The same walk, begun at the first member ordered at or after @p from. */
    [[nodiscard]] ordered_cursor_t cursor_from(identifier_t from) const noexcept
        requires offers_both_bounds<inner_store_t>
    {
        ordered_cursor_t walking {*this, cursor_seed_t::the_given_bound_k, cursor_limit_t::the_whole_keyspace_k};
        walking.position_ = std::move(from);
        return walking;
    }

    /** The same walk, stopping before @p upper. */
    [[nodiscard]] ordered_cursor_t cursor_up_to(identifier_t upper) const noexcept
        requires offers_both_bounds<inner_store_t> && offers_smallest<inner_store_t>
    {
        ordered_cursor_t walking {*this, cursor_seed_t::the_smallest_k, cursor_limit_t::up_to_the_bound_k};
        walking.bound_ = std::move(upper);
        return walking;
    }

    /** The same walk over [ @p from, @p upper ). */
    [[nodiscard]] ordered_cursor_t cursor_range(identifier_t from, identifier_t upper) const noexcept
        requires offers_both_bounds<inner_store_t>
    {
        ordered_cursor_t walking {*this, cursor_seed_t::the_given_bound_k, cursor_limit_t::up_to_the_bound_k};
        walking.position_ = std::move(from);
        walking.bound_ = std::move(upper);
        return walking;
    }

    /** Hands @p callback_found the smallest member, or reports the store is empty. */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    status_t smallest(callback_found_type_ &&callback_found,
                      callback_missing_type_ &&callback_missing = {}) const noexcept
        requires offers_smallest<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.smallest(std::forward<callback_found_type_>(callback_found),
                                     std::forward<callback_missing_type_>(callback_missing));
    }

    /** Removes the smallest member and hands it over, or reports the store is empty. The choice and
     *  the removal happen under one exclusive hold, so no writer can slip in between. */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    status_t pop_smallest(callback_found_type_ &&callback_found = {},
                          callback_missing_type_ &&callback_missing = {}) noexcept
        requires offers_pop_smallest<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.pop_smallest(std::forward<callback_found_type_>(callback_found),
                                         std::forward<callback_missing_type_>(callback_missing));
    }

    /** Copies out the smallest member and removes it, or reports @c key_not_found_k. */
    expected<value_t> pop_smallest_copy() noexcept
        requires offers_pop_smallest<inner_store_t>
    {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const popped =
            pop_smallest([&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(popped)) return popped;
        return result;
    }

    /** Copies out the smallest member, or reports @c key_not_found_k. */
    expected<value_t> smallest_copy() const noexcept
        requires offers_smallest<inner_store_t>
    {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const looked_up =
            smallest([&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(looked_up)) return looked_up;
        return result;
    }

    /** Copies out the member equal to @p comparable, or reports @c key_not_found_k. */
    template <typename comparable_type_ = identifier_t>
    expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const looked_up = find(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(looked_up)) return looked_up;
        return result;
    }

    /** Copies out the first element ordered at or after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
        requires offers_lower_bound<inner_store_t>
    {
        expected<value_t> result {status_t::key_not_found_k};
        status_t const bounded = lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        if (failed(bounded)) return bounded;
        return result;
    }

    /** Copies out the first element ordered strictly after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
        requires offers_upper_bound<inner_store_t>
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
    status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept
        requires offers_upper_bound<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.upper_bound(std::forward<comparable_type_>(comparable),
                                        std::forward<callback_found_type_>(callback_found),
                                        std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief Hands @p callback every member of [ @p lower, @p upper ), one shared lock held.
     *  @note A callback answering @c walk_control_t stops the walk where it says to.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
        requires offers_range<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                  std::forward<callback_type_>(callback));
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires offers_erase_range<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.erase_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                        std::forward<callback_type_>(callback));
    }

    /** Erases every element at or after @p lower, reporting each to @p callback. */
    template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
    status_t erase_from(lower_type_ &&lower, callback_type_ &&callback = {}) noexcept
        requires offers_erase_from<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.erase_from(std::forward<lower_type_>(lower), std::forward<callback_type_>(callback));
    }

    /** Erases every element before @p upper, reporting each to @p callback. */
    template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
    status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires offers_erase_up_to<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.erase_up_to(std::forward<upper_type_>(upper), std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Rewrites the mapped side of every element in [ @p lower, @p upper ).
     *  @param[in] callback Invoked with (key const &, mapped &) per element. Must be @c noexcept.
     *  @return Success, or an allocation failure. A store whose walk cannot fail always succeeds.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    status_t update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires offers_update_range<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.update_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                         std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Hands @p callback every member the store holds, in whatever order it keeps them.
     *
     *  The one walk an unordered core can offer, and the whole of it runs under one shared lock, so
     *  a writer is held off for its length and the enumeration is a consistent snapshot: every
     *  element present when the call began is visited exactly once, and no element inserted during
     *  it is seen.
     *
     *  @note A callback answering @c walk_control_t stops the walk where it says to, which also
     *      ends the shared hold early.
     */
    template <typename callback_type_ = no_op_t>
    status_t for_each(callback_type_ &&callback) const noexcept
        requires offers_for_each<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.for_each(std::forward<callback_type_>(callback));
    }

    /**
     *  @brief Frees every version no reader can still reach.
     *  @return How many versions were reclaimed, or why none could be. A store whose sweep cannot
     *      refuse always answers with a count.
     */
    expected<std::size_t> vacuum() noexcept
        requires offers_vacuum<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.vacuum();
    }

    /**
     *  @brief Frees the unreachable versions of every key in [ @p lower, @p upper ), so a caller
     *      can step through the keyspace instead of paying for one pass over all of it.
     *  @return How many versions were reclaimed, or why none could be.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t>
    expected<std::size_t> vacuum(lower_type_ &&lower, upper_type_ &&upper) noexcept
        requires offers_vacuum_window<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.vacuum(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper));
    }

    /**
     *  @brief Hands @p callback_found the element at zero-based position @p ordinal.
     *  @param[in] callback_missing Fires when fewer elements are there. Must be @c noexcept.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                    callback_missing_type_ &&callback_missing = {}) const noexcept
        requires offers_select<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.select(ordinal, std::forward<callback_found_type_>(callback_found),
                                   std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief Hands @p callback_found how many elements the store orders before @p comparable.
     *  @param[in] callback_missing Fires when @p comparable is not there at all. Must be
     *      @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept
        requires offers_rank<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.rank(std::forward<comparable_type_>(comparable),
                                 std::forward<callback_found_type_>(callback_found),
                                 std::forward<callback_missing_type_>(callback_missing));
    }

    status_t clear() noexcept {
        unique_lock _ {mutex_};
        return inner_store_.clear();
    }

    status_t reserve(std::size_t size) noexcept {
        unique_lock _ {mutex_};
        return inner_store_.reserve(size);
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                        callback_type_ &&callback) const noexcept
        requires offers_sample_one<inner_store_t> && uniform_random_bits<std::remove_cvref_t<generator_type_>>
    {
        shared_lock _ {mutex_};
        return inner_store_.sample_one(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                       std::forward<generator_type_>(generator),
                                       std::forward<callback_type_>(callback));
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator, std::size_t &seen,
                              std::size_t reservoir_capacity, output_iterator_type_ &&reservoir) const noexcept
        requires offers_sample_reservoir<inner_store_t> && uniform_random_bits<std::remove_cvref_t<generator_type_>>
    {
        shared_lock _ {mutex_};
        return inner_store_.sample_reservoir(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                             std::forward<generator_type_>(generator), seen, reservoir_capacity,
                                             std::forward<output_iterator_type_>(reservoir));
    }

    /**
     *  @brief Hands @p callback every member equal to @p comparable, which for a unique-key store
     *      is one or none.
     *  @note A callback answering @c walk_control_t stops the walk where it says to.
     */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
    status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept
        requires offers_equal_range<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.equal_range(std::forward<comparable_type_>(comparable),
                                        std::forward<callback_type_>(callback));
    }

    /** How many keys the ordinal surface indexes, which is what @c select counts against. */
    [[nodiscard]] std::size_t ranked_size() const noexcept
        requires offers_ranked_size<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.ranked_size();
    }

    /** How many versions the store holds across every key, published and staged alike. */
    [[nodiscard]] std::size_t versions_count() const noexcept
        requires offers_versions_count<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.versions_count();
    }

    /** How many versions of @p comparable the store still holds. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t versions_count(comparable_type_ const &comparable) const noexcept
        requires offers_versions_count<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.versions_count(comparable);
    }

    /** The newest snapshot no open transaction sits below, which is what @c vacuum prunes to. */
    [[nodiscard]] generation_t low_water_mark() const noexcept
        requires offers_low_water_mark<inner_store_t>
    {
        shared_lock _ {mutex_};
        return inner_store_.low_water_mark();
    }

    /** Writes @p element whether or not its key is taken, spelled as the inner store spells it. */
    status_t insert_or_assign(value_t &&element) noexcept
        requires offers_insert_or_assign<inner_store_t>
    {
        unique_lock _ {mutex_};
        return inner_store_.insert_or_assign(std::move(element));
    }

#pragma region Sharded Membership

    /** Opens one part of a sharded transaction on @p snapshot and @p generation drawn elsewhere. */
    expected<transaction_t> transaction_at(opened_at_t opened) noexcept
        requires draws_from_a_shared_order<inner_store_t>
    {
        unique_lock _ {mutex_};
        auto inner = inner_store_.transaction_at(opened);
        if (!inner) return inner.status();
        return transaction_t {*this, std::move(*inner)};
    }

    /** The order the wrapped store is a member of, which is fixed at construction and needs no
     *  lock. */
    [[nodiscard]] order_t &order() noexcept { return *order_; }
    [[nodiscard]] order_t const &order() const noexcept { return *order_; }

#pragma endregion Sharded Membership
};

#pragma region Aliases

/**
 *  @brief A set-shaped store behind one shared mutex, spelled
 *      @c locked_set<monotonic_avl_set<key_t>>.
 *
 *  The wrapper is one class whichever shape it holds, so the shape is a constraint rather than a
 *  second spelling: naming the set alias over a store of mappings has no substitution, and neither
 *  does naming @c locked_map over a store of plain keys. Two aliases that accept the same arguments
 *  would name the same type and catch nothing.
 */
template <set_shaped_store set_store_type_, typename shared_mutex_type_ = spin_shared_mutex_t>
using locked_set = locked_store<set_store_type_, shared_mutex_type_>;

/** A map-shaped store behind one shared mutex, spelled @c locked_map<monotonic_avl_map<key_t,
 *  value_t>>. */
template <map_shaped_store map_store_type_, typename shared_mutex_type_ = spin_shared_mutex_t>
using locked_map = locked_store<map_store_type_, shared_mutex_type_>;

#pragma endregion Aliases

} // namespace ashvardanian::smashtable
