/**
 *  @brief Generic transactional container with ACID semantics, providing 2-phase commit transactions. Can be
 *      instantiated with any key-addressable core - AVL trees, weight-balanced trees, open-addressed tables -
 *      and gates its ordered surface behind the cores that supply an ordering.
 *      All operations use callback-based APIs and are exception-free via @c noexcept constraints.
 *  @author Ash Vardanian
 *  @file include/smashtable/transactional_store.hpp
 *  @date October 13, 2022
 */
#pragma once
#include <cassert> // `assert`

#include <memory>   // `std::allocator`, `std::construct_at`, `std::destroy_at`
#include <optional> // `std::optional`
#include <utility>  // `std::exchange`

#include "basic_avl_tree.hpp"
#include "basic_hash_table.hpp"
#include "basic_vector.hpp"
#include "basic_wb_tree.hpp"
#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief  Generic transactional store providing 2-phase commits and "watch" operations.
 *    Can be instantiated with AVL trees, weight-balanced trees, or open-addressed hash tables.
 *    Not thread-safe by itself. Entirely exception-free, with all methods marked @c noexcept.
 *
 *  @section transactional_store_design_goals Design Goals
 *
 *  All operations are atomic. When updating multiple values, you don't want to break in an intermediate state
 *  where only some updates succeeded. With two-phase commit transactions, you can stage many changes and commit
 *  them all at once, or rollback if something goes wrong. Common use case: synchronizing updates across multiple
 *  data stores while maintaining consistency guarantees.
 *
 *  The API is simple, generalizable, and lightweight. This collection @b doesn't provide snapshots or full MVCC
 *  (Multi-Version Concurrency Control). If you start a transaction and "watch" values through it, there's no
 *  guarantee the value hasn't been updated before the transaction began and the entry was added to the watched
 *  list. Only "Monotonic Atomic View" consistency is guaranteed, including its inferior "Read Committed" and
 *  "Read Uncommitted" levels. Transactions cannot observe writes from other uncommitted transactions.
 *
 *  The state management doesn't rely on entry pointers or iterators. Those could simplify the implementation,
 *  but introduce require validity constraints for re-allocations and modifications of the tree and underlying
 *  allocator behavior. So all entry IDs must either be nothrow-copyable or provide a @c copy() member method
 *  returning an @c std::optional<T> to support safe copying for watch bookkeeping.
 *
 *  @see https://jepsen.io/consistency/models/monotonic-atomic-view
 *  @see https://jepsen.io/consistency/models/read-committed
 *
 *  @section transactional_store_api_overview API Overview
 *
 *  - All lookups are heterogeneous: you can provide any type comparable to the element type. Your comparator
 *    MUST define @code using is_transparent = void; @endcode to enable this, just like std::map and std::set.
 *  - No iterators are provided to keep the implementation simple and avoid complexity of maintaining persistent
 *    iterator validity across transactions and modifications.
 *  - All operations use callback-based APIs for consistency, with all callbacks expected to be @c noexcept.
 *
 *  @subsection Insert Strategies
 *
 *  Four distinct modification strategies with different failure handling:
 *
 *  | Method               | Key Exists?     | Returns          | Use Case                                      |
 *  |----------------------|-----------------|------------------|-----------------------------------------------|
 *  | insert()             | Fails (error)   | invalid_arg_k    | Strict: ensure key is new                     |
 *  | insert_if_missing()  | Skips (success) | success_k        | Lenient: insert only if absent, else no-op    |
 *  | upsert()             | Overwrites      | success_k        | Always update regardless of existence         |
 *  | update()             | Fails (error)   | key_not_found_k  | Strict: ensure key exists before updating     |
 *
 *  @tparam collection_type_ The underlying core, satisfying @c key_addressable_collection and providing a
 *    @c rebind template alias. Cores that also satisfy @c ordered_collection unlock the ordered surface -
 *    bounds, ranges, and, with order statistics, @c select and @c rank.
 */
template <typename collection_type_>
class transactional_store {

  public:
#pragma region Type Definitions

    using value_t = typename owned_value_of<collection_type_>::type;
    using value_type = value_t; // ? STL style

    using key_t = typename collection_type_::key_type;
    using key_type = key_t; // ? STL style

    using mapped_t = typename collection_type_::mapped_type;
    using mapped_type = mapped_t; // ? STL style

    using is_associative = typename collection_type_::is_associative;
    using is_transactional = std::true_type;
    using callback_reads = std::true_type;

    /** @brief A commit publishes every staged version before any reader can run, and takes no snapshot. */
    static constexpr isolation_t isolation_k = isolation_t::monotonic_atomic_view_k;

    using allocator_t = typename collection_type_::allocator_type;

    /** @brief What this core family rebinds on, and how a decorated entry is reached inside it. */
    using storage_shape_t = versioned_storage_for<collection_type_, value_t>;

    using comparator_t = typename storage_shape_t::addressing_source_t;
    using versioning_t = typename storage_shape_t::versioning_t;
    using identifier_t = typename versioning_t::identifier_t;
    using generation_t = typename versioning_t::generation_t;
    using watch_t = typename versioning_t::watch_t;
    using watched_identifier_t = typename versioning_t::watched_identifier_t;
    using dated_identifier_t = typename versioning_t::dated_identifier_t;
    using versioned_t = typename versioning_t::versioned_t;
    using versioned_entry_t = versioned_t;

  private:
    /** @brief One further version of a key, reached by pointer from that key's chain head. */
    struct version_node_t {
        /** @brief The version itself, exactly as it was stored. */
        versioned_t entry;
        /** @brief Another version of the same key, or null at the tail. */
        version_node_t *next {nullptr};
    };

    /** @brief Where every overflow version node comes from and goes back to. */
    using version_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<version_node_t>;

    /**
     *  @brief Every version of one key: one held inline, any others in a pointer chain.
     *
     *  Outside a staging window a key carries exactly one version, so @c others is null and the whole
     *  chain costs one pointer beyond the version itself. A chain frees its own overflow nodes, so it
     *  carries the allocator they came from - which a stateless allocator makes free.
     *
     *  The chain is deliberately unordered. Ordering it would mean displacing the inline version to
     *  make room for a newer one, and moving a version out of a live slot and then assigning a
     *  replacement into it leaves that slot momentarily moved-from - which a value type is entitled to
     *  reject. Versions are therefore only ever appended as nodes, and the two queries that care about
     *  order walk the chain, which almost always holds one entry.
     */
    struct versioned_chain_t {
        using is_version_chain = void;

        /** @brief One version, always present, in no particular position. */
        versioned_t head;
        /** @brief Further versions of the same key, usually null. */
        version_node_t *others {nullptr};
        /** @brief Where @c others came from, so the chain can hand them back when it dies. */
        [[no_unique_address]] version_allocator_t allocator {};

        versioned_chain_t() = default;
        explicit versioned_chain_t(versioned_t &&only) noexcept : head(std::move(only)) {}
        versioned_chain_t(versioned_chain_t const &) = delete;
        versioned_chain_t &operator=(versioned_chain_t const &) = delete;
        versioned_chain_t(versioned_chain_t &&other) noexcept
            : head(std::move(other.head)), others(std::exchange(other.others, nullptr)), allocator(other.allocator) {}
        versioned_chain_t &operator=(versioned_chain_t &&other) noexcept {
            if (this == &other) return *this;
            release_others();
            head = std::move(other.head);
            others = std::exchange(other.others, nullptr);
            allocator = other.allocator;
            return *this;
        }
        ~versioned_chain_t() noexcept { release_others(); }

        /** @brief Frees every chained node, leaving the inline one behind. */
        void release_others() noexcept {
            while (others) {
                version_node_t *const next = others->next;
                others->~version_node_t();
                allocator.deallocate(others, 1);
                others = next;
            }
        }
    };

    /** @brief What @c chain_detach_ did, so the caller knows whether the tree node is now empty. */
    enum class detach_outcome_t {
        not_found_k,
        detached_k,
        detached_and_emptied_k,
    };

    // The store keys one entry per identifier and hangs that identifier's versions off it. The
    // transaction stages into a plain tree of versions, where its own generation makes every key
    // unique, so only the store needs chains.
    using versioned_chains_t = typename storage_shape_t::template rebind<versioned_chain_t>;
    using versioned_set_t = typename storage_shape_t::template rebind<versioned_t>;

    using watches_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<watched_identifier_t>;
    using watches_vector_t = basic_vector<watched_identifier_t, watches_allocator_t>;

    using changed_ids_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<identifier_t>;
    using changed_ids_vector_t = basic_vector<identifier_t, changed_ids_allocator_t>;

    using store_t = transactional_store;

  public:
    class transaction_t {

        friend store_t;
        store_t *store_ {nullptr};
        versioned_set_t changes_ {};
        watches_vector_t watches_ {};
        changed_ids_vector_t changed_ids_ {};
        generation_t generation_ {0};
        staging_t staging_ {staging_t::pending_k};

        transaction_t(store_t &set) noexcept
            : store_(&set), changes_(storage_shape_t::template sibling<versioned_t>(set.entries_)),
              watches_(watches_allocator_t(storage_shape_t::allocator_of(set.entries_))),
              changed_ids_(changed_ids_allocator_t(storage_shape_t::allocator_of(set.entries_))),
              generation_(set.new_generation_()) {}
        /** @brief What this transaction saw when it looked at a key that was not there. */
        static watch_t missing_watch() noexcept { return watch_t {absent_generation_k, presence_t::erased_k}; }

        /**
         *  @brief Drops this transaction's version of the first @p processed changed identifiers.
         *    Nothing else can carry this generation, so a key that has no other version disappears.
         *  @param[in] processed How many leading entries of the changed-identifier list to undo.
         */
        void unstage_(std::size_t processed) noexcept {
            auto &store = store_ref();
            for (std::size_t index = 0; index != processed; ++index)
                // ! Don't materialize a new copy of the identifier here, use a reference
                store.chain_discard_(changed_ids_[index], generation_, nullptr);
        }

        store_t &store_ref() noexcept { return *store_; }
        store_t const &store_ref() const noexcept { return *store_; }

        /**
         *  @brief Drops anything staged and never published, and gives up this transaction's claim.
         *
         *  Abandoning a staged transaction is the only way the store can accumulate versions that no
         *  generation can ever name again, so the unwind is not optional. A null @c store_ marks a
         *  moved-from transaction, which owns nothing and must undo nothing.
         */
        void unwind_() noexcept {
            if (!store_) return;
            if (staging_ == staging_t::staged_k) unstage_(changed_ids_.size());
            staging_ = staging_t::pending_k;
            store_ = nullptr;
        }

      public:
        /**
         *  @brief Takes over @p other entirely, leaving it owning nothing.
         *
         *  Written out rather than defaulted: a defaulted move copies @c store_ into the new object
         *  and leaves the old one pointing at the same store, so the destructor below would unstage
         *  the same versions twice.
         */
        transaction_t(transaction_t &&other) noexcept
            : store_(std::exchange(other.store_, nullptr)), changes_(std::move(other.changes_)),
              watches_(std::move(other.watches_)), changed_ids_(std::move(other.changed_ids_)),
              generation_(other.generation_), staging_(std::exchange(other.staging_, staging_t::pending_k)) {}

        transaction_t &operator=(transaction_t &&other) noexcept {
            if (this == &other) return *this;
            unwind_();
            store_ = std::exchange(other.store_, nullptr);
            changes_ = std::move(other.changes_);
            watches_ = std::move(other.watches_);
            changed_ids_ = std::move(other.changed_ids_);
            generation_ = other.generation_;
            staging_ = std::exchange(other.staging_, staging_t::pending_k);
            return *this;
        }

        ~transaction_t() noexcept { unwind_(); }

        transaction_t(transaction_t const &) = delete;
        transaction_t &operator=(transaction_t const &) = delete;

        /**
         *  @brief Returns the generation (sequence number) of this transaction.
         *  @return The transaction's generation identifier.
         */
        generation_t generation() const noexcept { return generation_; }
        /**
         *  @brief Checks if this transaction has any pending changes (upserts or erases).
         *  @return True if there are pending changes, false otherwise.
         */

        bool has_changes() const noexcept { return changes_.size() != 0; }
        /**
         *  @brief Returns the number of pending changes in this transaction.
         *  @return The count of staged changes (including both upserts and erases).
         */
        std::size_t changes_count() const noexcept { return changes_.size(); }

      public:
        /**
         *  @brief Stages an insert operation only if the key doesn't exist. Fails if key exists.
         *    Checks both transaction changes and main store for existence.
         *
         *  @param[in] value Element to insert (moved into the transaction).
         *  @return Success, or @c key_already_exists_k if key exists, or OOM error.
         */
        [[nodiscard]] status_t insert(value_t &&value) noexcept {
            auto local_it = changes_.find(value);
            if (local_it != changes_.end() && (*local_it).presence == presence_t::present_k)
                return {key_already_exists_k};
            if (store_ref().contains(value)) return {key_already_exists_k};

            auto maybe_id = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
            if (!maybe_id) return status_t {out_of_memory_heap_k};
            auto reserve_status = changed_ids_.reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            versioned_t versioned(std::move(value));
            versioned.generation = generation_;
            versioned.presence = presence_t::present_k;
            versioned.publication = publication_t::staged_k;
            auto result = storage_shape_t::upsert(changes_, std::move(versioned));
            if (!result) return status_t {out_of_memory_heap_k};

            changed_ids_.push_back(assume_reserved, std::move(*maybe_id));
            return status_t {success_k};
        }

        /**
         *  @brief Stages an insert operation only if key is missing. Silently skips if key exists (no error).
         *    Checks both transaction changes and main store for existence.
         *
         *  @param[in] value Element to insert (moved into the transaction).
         *  @return Always succeeds (unless OOM). Returns success even if key exists.
         */
        [[nodiscard]] status_t insert_if_missing(value_t &&value) noexcept {
            auto local_it = changes_.find(value);
            if (local_it != changes_.end() && (*local_it).presence == presence_t::present_k) return {success_k};
            if (store_ref().contains(value)) return {success_k};

            auto maybe_id = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
            if (!maybe_id) return status_t {out_of_memory_heap_k};
            auto reserve_status = changed_ids_.reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            versioned_t versioned(std::move(value));
            versioned.generation = generation_;
            versioned.presence = presence_t::present_k;
            versioned.publication = publication_t::staged_k;
            auto result = storage_shape_t::upsert(changes_, std::move(versioned));
            if (!result) return status_t {out_of_memory_heap_k};

            changed_ids_.push_back(assume_reserved, std::move(*maybe_id));
            return status_t {success_k};
        }

        /**
         *  @brief Stages an upsert operation for the given element (insert or update). Always succeeds.
         *    Overwrites existing element if key exists. Changes visible after @c stage() and @c commit().
         *
         *  @param[in] value Element to upsert (moved into the transaction).
         *  @return Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t upsert(value_t &&value) noexcept {
            auto maybe_id = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
            if (!maybe_id) return status_t {out_of_memory_heap_k};
            auto reserve_status = changed_ids_.reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            versioned_t versioned(std::move(value));
            versioned.generation = generation_;
            versioned.presence = presence_t::present_k;
            versioned.publication = publication_t::staged_k;
            auto result = storage_shape_t::upsert(changes_, std::move(versioned));
            if (!result) return status_t {out_of_memory_heap_k};

            changed_ids_.push_back(assume_reserved, std::move(*maybe_id));
            return status_t {success_k};
        }

        /**
         *  @brief Stages an update operation for existing keys only.
         *    Fails if key doesn't exist anywhere (local changes or main store).
         *
         *  @param[in] value Element to update (moved into the transaction).
         *  @return Success, or @c key_not_found_k if key doesn't exist.
         */
        [[nodiscard]] status_t update(value_t &&value) noexcept {
            auto local_it = changes_.find(value);
            if (local_it != changes_.end() && (*local_it).presence == presence_t::present_k)
                return upsert(std::move(value));
            if (!store_ref().contains(value)) return {key_not_found_k};
            return upsert(std::move(value));
        }

        /**
         *  @brief Stages an erase operation for the given identifier.
         *    Marks the entry as deleted in the transaction. Actual removal happens on commit.
         *
         *  @param[in] id Identifier of the element to erase.
         *  @return Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t erase(identifier_t const &id) noexcept {
            auto maybe_id = copy_safely<identifier_t>(id);
            if (!maybe_id) return status_t {out_of_memory_heap_k};
            auto reserve_status = changed_ids_.reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            // The tombstone owns its own identifier, and the list of changed identifiers owns another,
            // so a move-only key needs two safe copies rather than one copy and one implicit one.
            auto maybe_payload = copy_safely<identifier_t>(id);
            if (!maybe_payload) return status_t {out_of_memory_heap_k};

            versioned_t versioned(value_t {std::move(*maybe_payload)});
            versioned.generation = generation_;
            versioned.presence = presence_t::erased_k;
            versioned.publication = publication_t::staged_k;
            auto result = storage_shape_t::upsert(changes_, std::move(versioned));
            if (!result) return status_t {out_of_memory_heap_k};

            changed_ids_.push_back(assume_reserved, std::move(*maybe_id));
            return status_t {success_k};
        }

        [[nodiscard]] status_t reserve(std::size_t size) noexcept { return watches_.reserve(size); }

        [[nodiscard]] status_t watch(identifier_t id) noexcept {
            status_t result {success_k};
            auto found = [&](versioned_t const &versioned) noexcept {
                // A committed tombstone is reported as found, since the public `find` has to see it
                // to hide it. Validation resolves that same key to "missing", so a watch on it must
                // record the missing shape or it could never match itself.
                watch_t const observed = versioned.presence == presence_t::erased_k
                                             ? missing_watch()
                                             : watch_t {versioned.generation, versioned.presence};
                result = watches_.push_back({std::move(id), observed});
            };
            auto missing = [&]() noexcept { result = watches_.push_back({std::move(id), missing_watch()}); };
            store_ref().find_visible_entry_(id, found, missing);
            return result;
        }

        [[nodiscard]] status_t watch(versioned_t const &versioned) noexcept {
            auto maybe_id = copy_safely<identifier_t>(identifier_t {versioned.unversioned});
            if (!maybe_id) return status_t {out_of_memory_heap_k};
            return watches_.push_back({std::move(*maybe_id), watch_t {versioned.generation, versioned.presence}});
        }

        /**
         *  @brief Finds a member @b equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c transactional_store::find(), will include the entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
            if (auto iterator = changes_.find(std::forward<comparable_type_>(comparable)); iterator != changes_.end())
                (*iterator).presence == presence_t::present_k ? callback_found((*iterator).unversioned)
                                                              : callback_missing();
            else
                store_ref().find(std::forward<comparable_type_>(comparable),
                                 std::forward<callback_found_type_>(callback_found),
                                 std::forward<callback_missing_type_>(callback_missing));
        }

        /**
         *  @brief Finds and returns a copy of an element equal to @p comparable, including transaction changes.
         *
         *  @param[in] comparable Object comparable to @c value_t.
         *  @return Result with copied element if found, or failure status.
         */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result;
            result.status.errc = errc_t::key_not_found_k;

            // Only the fall-through branch forwards: a lookup that misses `changes_` is the last use, and
            // forwarding at both sites would hand the second one an already moved-from object.
            if (auto it = changes_.find(comparable); it != changes_.end()) {
                if ((*it).presence == presence_t::present_k) {
                    auto copy_result = copy_safely((*it).unversioned);
                    if (copy_result) {
                        result.outcome = std::move(*copy_result);
                        result.status = status_t {success_k};
                    }
                    else { result.status = copy_result.status; }
                }
            }
            else { return store_ref().find_copy(std::forward<comparable_type_>(comparable)); }

            return result;
        }

        /**
         *  @brief Checks if a member @b equal to the given @p comparable exists, including transaction changes.
         *    Convenience wrapper around @c find() for existence checks.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @return True if the element exists, false otherwise.
         */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
            bool found = false;
            find(
                std::forward<comparable_type_>(comparable), [&](auto const &) noexcept { found = true; },
                []() noexcept {});
            return found;
        }

        /**
         *  @brief Finds the first member @b greater or equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c transactional_store::lower_bound(), will include entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_collection<versioned_chains_t>
        {

            auto const &store = store_ref();
            versioned_t const *store_visible = nullptr;
            for (auto cursor = store.entries_.lower_bound(comparable); cursor != store.entries_.end(); ++cursor) {
                if (changes_.contains(*cursor)) continue;
                if ((store_visible = store_t::visible_version_(*cursor)) != nullptr) break;
            }

            auto changed_lb = changes_.lower_bound(comparable);
            while (changed_lb != changes_.end() && changed_lb->presence == presence_t::erased_k) { ++changed_lb; }

            if (!store_visible && changed_lb == changes_.end()) { callback_missing(); }
            else if (!store_visible) { callback_found(changed_lb->unversioned); }
            else if (changed_lb == changes_.end()) { callback_found(store_visible->unversioned); }
            else {
                auto const &ordering = changes_.key_comp();
                if (ordering.less(store_visible->unversioned, changed_lb->unversioned)) {
                    callback_found(store_visible->unversioned);
                }
                else { callback_found(changed_lb->unversioned); }
            }
        }

        /**
         *  @brief Finds all elements equal to a single key. Invokes callback for each matching element.
         *    For sets with unique keys, returns at most one element (0 or 1).
         *    Includes transaction changes.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_fn_t>
        void equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
            find(
                std::forward<comparable_type_>(comparable),
                [&](versioned_t const &versioned) noexcept { callback(versioned.unversioned); }, []() noexcept {});
        }

        /**
         *  @brief Iterates over all entries in the range [ @p lower, @p upper), including transaction changes.
         *
         *  @param[in] lower Lower bound of the range (inclusive).
         *  @param[in] upper Upper bound of the range (exclusive).
         *  @param[in] callback Callback invoked for each element in range. Must be @c noexcept.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_fn_t>
        void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires ordered_collection<versioned_chains_t>
        {
            auto const less = changes_.key_comp();
            auto lower_internal = changes_.lower_bound(std::forward<lower_type_>(lower));
            auto const upper_internal_bound = identifier_t(upper);
            for (auto it = lower_internal; it != changes_.end() && less(*it, upper_internal_bound); ++it)
                if (it->presence == presence_t::present_k) callback(it->unversioned);

            store_ref().range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                              [&](value_t const &external_value) {
                                  auto local_state = changes_.find(external_value);
                                  if (local_state == changes_.end()) callback(external_value);
                              });
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_collection<versioned_chains_t>
        {

            auto const &store = store_ref();
            versioned_t const *store_visible = nullptr;
            for (auto cursor = store.entries_.upper_bound(comparable); cursor != store.entries_.end(); ++cursor) {
                if (changes_.contains(*cursor)) continue;
                if ((store_visible = store_t::visible_version_(*cursor)) != nullptr) break;
            }

            auto changed_ub = changes_.upper_bound(comparable);
            while (changed_ub != changes_.end() && changed_ub->presence == presence_t::erased_k) { ++changed_ub; }

            if (!store_visible && changed_ub == changes_.end()) { callback_missing(); }
            else if (!store_visible) { callback_found(changed_ub->unversioned); }
            else if (changed_ub == changes_.end()) { callback_found(store_visible->unversioned); }
            else {
                auto const &ordering = changes_.key_comp();
                if (ordering.less(store_visible->unversioned, changed_ub->unversioned)) {
                    callback_found(store_visible->unversioned);
                }
                else { callback_found(changed_ub->unversioned); }
            }
        }

        /**
         *  @brief Finds the k-th smallest element including transaction changes.
         *    Only available for tree implementations that support order statistics (e.g., WB trees).
         *    Merges view of local staged changes with committed entries from main store.
         *
         *  @param[in] k Zero-based index (0 = smallest element).
         *  @param[in] callback_found Callback to receive the k-th element. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if k >= size(). Must be @c noexcept.
         */
        template <typename callback_found_type_ = no_op_fn_t, typename callback_missing_type_ = no_op_fn_t>
        void select(std::size_t k, callback_found_type_ &&callback_found,
                    callback_missing_type_ &&callback_missing = {}) const noexcept
            requires supports_order_statistics<versioned_chains_t>
        {
            std::size_t visible_index = 0;
            bool found = false;
            auto const less = changes_.key_comp();

            auto local_it = changes_.begin();
            chain_node_t *store_node = chain_node_t::find_min(store_ref().entries_.root());

            while ((local_it != changes_.end() || store_node) && !found) {
                bool take_local = false;

                if (local_it == changes_.end()) { take_local = false; }
                else if (!store_node) { take_local = true; }
                else { take_local = less(local_it->unversioned, store_node->fruit); }

                if (take_local) {
                    if (local_it->presence == presence_t::present_k) {
                        if (visible_index == k) {
                            callback_found(local_it->unversioned);
                            found = true;
                        }
                        ++visible_index;
                    }
                    ++local_it;
                }
                else {
                    versioned_t const *store_visible = store_t::visible_version_(store_node->fruit);
                    if (store_visible && store_visible->presence == presence_t::present_k &&
                        changes_.find(mapping_key_or_itself<value_t>(store_visible->unversioned)) == changes_.end()) {
                        if (visible_index == k) {
                            callback_found(store_visible->unversioned);
                            found = true;
                        }
                        ++visible_index;
                    }

                    store_node = chain_node_t::find_successor(store_node);
                }
            }

            if (!found) callback_missing();
        }

        /**
         *  @brief Finds the rank (position) of an element including transaction changes.
         *    Only available for tree implementations that support order statistics (e.g., WB trees).
         *    Includes both staged changes and committed entries from main store.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive the rank (size_t). Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if element not found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept
            requires supports_order_statistics<versioned_chains_t>
        {
            identifier_t target_id(comparable);
            std::size_t rank_value = 0;
            bool found = false;
            auto const less = changes_.key_comp();

            auto local_target = changes_.find(target_id);
            if (local_target != changes_.end() && local_target->presence == presence_t::present_k) { found = true; }
            else if (store_ref().contains(target_id)) { found = true; }

            if (!found) {
                callback_missing();
                return;
            }

            auto local_it = changes_.begin();
            chain_node_t *store_node = chain_node_t::find_min(store_ref().entries_.root());

            while (local_it != changes_.end() || store_node) {
                bool take_local = false;

                if (local_it == changes_.end()) { take_local = false; }
                else if (!store_node) { take_local = true; }
                else { take_local = less(local_it->unversioned, store_node->fruit); }

                if (take_local) {
                    if (local_it->presence == presence_t::present_k && less(local_it->unversioned, target_id))
                        ++rank_value;
                    ++local_it;
                }
                else {
                    versioned_t const *store_visible = store_t::visible_version_(store_node->fruit);
                    if (store_visible && store_visible->presence == presence_t::present_k &&
                        changes_.find(mapping_key_or_itself<value_t>(store_visible->unversioned)) == changes_.end() &&
                        less(store_visible->unversioned, target_id)) {
                        ++rank_value;
                    }

                    store_node = chain_node_t::find_successor(store_node);
                }
            }

            callback_found(rank_value);
        }

        /**
         *  If the staging fails, the transaction contents remain unchanged. On success, all
         *  changes are merged into the main store but remain invisible until @c commit() is called.
         *  Otherwise, the user may call @c rollback() to pull back the staged changes into the
         *  transaction itself, or @c reset() to discard all changes and start fresh.
         */
        [[nodiscard]] status_t stage() noexcept {
            auto &store = store_ref();
            auto const prepaid = storage_shape_t::prepare(store.entries_, changed_ids_.size());
            if (!prepaid) return prepaid;
            auto const entry_missing = missing_watch();
            for (auto const &id_and_watch : watches_) {
                auto consistency_violated = false;
                store.find_latest_entry_(
                    id_and_watch.id,
                    [&](versioned_t const &versioned) noexcept {
                        consistency_violated = versioned != id_and_watch.watch;
                    },
                    [&]() noexcept { consistency_violated = entry_missing != id_and_watch.watch; });
                if (consistency_violated) return {errc_t::consistency_k};
            }

            // Every change needs somewhere to land before any of them moves, or a transaction could
            // run out of memory with half of itself already in the store. So this pass reserves a
            // chain slot for each new key and one spare version node for each key that already has
            // one, and only then does the second pass move the versions across, where it cannot fail.
            std::size_t spare_versions_needed = 0;
            for (std::size_t reserved = 0; reserved != changed_ids_.size(); ++reserved) {
                identifier_t const &id = changed_ids_[reserved];
                if (store.entries_.find(id) != store.entries_.end()) {
                    ++spare_versions_needed;
                    continue;
                }
                auto reserved_id = copy_safely<identifier_t>(id);
                if (!reserved_id) {
                    unstage_(reserved);
                    return status_t {out_of_memory_heap_k};
                }
                versioned_t reservation(value_t {std::move(*reserved_id)});
                reservation.generation = generation_;
                reservation.presence = presence_t::present_k;
                reservation.publication = publication_t::staged_k;
                auto result = storage_shape_t::upsert(store.entries_, versioned_chain_t {std::move(reservation)});
                if (!result) {
                    unstage_(reserved);
                    return status_t {out_of_memory_heap_k};
                }
            }
            if (!store.reserve_spare_versions_(spare_versions_needed)) {
                unstage_(changed_ids_.size());
                return status_t {out_of_memory_heap_k};
            }

            // The visibility is updated later, in `commit`. A reserved slot already carries this
            // generation, so the version replaces it rather than lengthening the chain.
            for (auto staged = changes_.begin(); staged != changes_.end(); ++staged) {
                versioned_t &version = store_t::mutable_ref_(*staged);
                store.chain_attach_(mapping_key_or_itself<value_t>(version.unversioned), std::move(version));
            }

            changes_.clear();
            staging_ = staging_t::staged_k;
            return {success_k};
        }

        [[nodiscard]] status_t reset() noexcept {
            auto &store = store_ref();
            if (staging_ == staging_t::staged_k) unstage_(changed_ids_.size());

            watches_.clear();
            changes_.clear();
            changed_ids_.clear();
            staging_ = staging_t::pending_k;
            generation_ = store.new_generation_();
            return {success_k};
        }

        [[nodiscard]] status_t rollback() noexcept {
            if (staging_ != staging_t::staged_k) return {operation_not_permitted_k};

            auto &store = store_ref();
            status_t result {success_k};

            // The recovered versions carry the generation they were staged under, and this
            // transaction is about to take a new one, so each is re-stamped on the way back. The
            // changed identifiers are deliberately kept: these keys are still going to be written,
            // and a later stage reserves a chain for each one before moving anything into it.
            generation_t const resumed = store.new_generation_();
            for (identifier_t const &id : changed_ids_) {
                versioned_t recovered;
                // ! Don't materialize a new copy of `id` here, use a reference
                if (store.chain_discard_(id, generation_, &recovered) == detach_outcome_t::not_found_k) continue;
                recovered.generation = resumed;
                recovered.publication = publication_t::staged_k;
                auto reinstated = storage_shape_t::upsert(changes_, std::move(recovered));
                if (!reinstated) result = status_t {out_of_memory_heap_k};
            }

            staging_ = staging_t::pending_k;
            generation_ = resumed;
            return result;
        }

        [[nodiscard]] status_t commit() noexcept {
            if (staging_ != staging_t::staged_k) return {operation_not_permitted_k};

            // Once we make an entry visible, if there are more than one with the same key,
            // the older generation must die.
            auto &store = store_ref();
            for (auto const &id : changed_ids_) store.unmask_and_compact_(id, generation_);

            changes_.clear();
            changed_ids_.clear();
            staging_ = staging_t::pending_k;
            return {success_k};
        }
    };

  private:
    /** @brief Version nodes reserved by a staging pass and not yet filed into a chain. */
    version_node_t *spare_versions_ {nullptr};
    versioned_chains_t entries_;
    generation_t generation_ {0};
    std::size_t visible_count_ {0};
    std::size_t visible_deleted_count_ {0};

    friend class transaction_t;
    using chain_node_t = typename storage_node_of<versioned_chains_t>::type;

    /**
     *  @brief Hands out the next version stamp, which must be unique across concurrent openers.
     *
     *  Relaxed suffices because one location has a total modification order, which is all uniqueness
     *  needs; a stamp is compared by value - @c > when walking a chain, @c == when validating a watch -
     *  and never used to order memory. Publishing what a generation tags is the partition lock's job.
     */
    generation_t new_generation_() noexcept { return atomic_add_fetch<generation_t>(generation_, 1); }

    /** @brief A writable reference to a stored element, which every core hands out as immutable. */
    template <typename element_type_>
    static element_type_ &mutable_ref_(element_type_ const &element) noexcept {
        return const_cast<element_type_ &>(element);
    }

    /** @brief The one version a reader may see, or null while every version of the key is staged. */
    static versioned_t const *visible_version_(versioned_chain_t const &chain) noexcept {
        if (chain.head.publication == publication_t::published_k) return &chain.head;
        for (version_node_t const *node = chain.others; node; node = node->next)
            if (node->entry.publication == publication_t::published_k) return &node->entry;
        return nullptr;
    }

    /** @brief The highest-generation version of the key, whether or not anything has committed it. */
    static versioned_t const *latest_version_(versioned_chain_t const &chain) noexcept {
        versioned_t const *latest = &chain.head;
        for (version_node_t const *node = chain.others; node; node = node->next)
            if (node->entry.generation > latest->generation) latest = &node->entry;
        return latest;
    }

    /** @brief Hands back every spare the staging pass reserved and did not use. */
    void release_spare_versions_() noexcept {
        version_allocator_t allocator(storage_shape_t::allocator_of(entries_));
        while (spare_versions_) {
            version_node_t *const next = spare_versions_->next;
            allocator.deallocate(spare_versions_, 1);
            spare_versions_ = next;
        }
    }

    /**
     *  @brief Reserves @p count version nodes up front, so the pass that files them cannot fail.
     *    Staging is two passes for this reason: everything that can fail happens in the first, and
     *    the second only moves versions into slots already paid for.
     *  @param[in] count How many chains will lengthen by one.
     *  @return Whether every node was obtained; on failure nothing is left reserved.
     */
    bool reserve_spare_versions_(std::size_t count) noexcept {
        version_allocator_t allocator(storage_shape_t::allocator_of(entries_));
        for (std::size_t index = 0; index != count; ++index) {
            version_node_t *const node = allocator.allocate(1);
            if (!node) {
                release_spare_versions_();
                return false;
            }
            node->next = spare_versions_;
            spare_versions_ = node;
        }
        return true;
    }

    /** @brief Takes one reserved node, which the caller has already guaranteed to be there. */
    [[nodiscard]] version_node_t *take_spare_version_() noexcept {
        version_node_t *const node = spare_versions_;
        assert(node && "The staging pass reserved one spare version node for this key");
        if (node) spare_versions_ = node->next;
        return node;
    }

    /**
     *  @brief Files @p version into the chain of @p comparable, which the caller has already reserved.
     *    A version already carrying that generation is overwritten, which is how a slot reserved by the
     *    first staging pass receives its payload without a second allocation.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[inout] version The version to file, left empty once it has been taken.
     */
    template <typename comparable_type_>
    void chain_attach_(comparable_type_ &&comparable, versioned_t &&version) noexcept {
        auto found = entries_.find(comparable);
        assert(found != entries_.end() && "The staging pass reserves a chain before anything moves into it");
        auto &chain = mutable_ref_(*found);

        // A slot reserved by the first pass already carries this generation, so the payload lands on
        // top of it rather than lengthening the chain.
        if (chain.head.generation == version.generation) {
            chain.head = std::move(version);
            return;
        }
        for (version_node_t *node = chain.others; node; node = node->next)
            if (node->entry.generation == version.generation) {
                node->entry = std::move(version);
                return;
            }

        version_node_t *const reserved = take_spare_version_();
        chain.others = new (reserved) version_node_t {std::move(version), chain.others};
        chain.allocator = version_allocator_t(storage_shape_t::allocator_of(entries_));
    }

    /**
     *  @brief Unlinks the version of @p chain carrying @p generation.
     *  @param[inout] chain The key's version chain, shortened by one entry when the generation is present.
     *  @param[in] generation The generation to unlink.
     *  @param[out] destination Receives the unlinked version, or null to discard it.
     *  @return Whether anything was unlinked, and whether the chain is now empty.
     */
    detach_outcome_t chain_detach_(versioned_chain_t &chain, generation_t generation,
                                   versioned_t *destination) noexcept {
        // The chained nodes go first, so a version this store never committed - which is where a
        // caller asking for the version back is always looking - never touches the inline slot.
        for (version_node_t **link = &chain.others; *link; link = &(*link)->next) {
            if ((*link)->entry.generation != generation) continue;
            version_node_t *const doomed = *link;
            if (destination) *destination = std::move(doomed->entry);
            *link = doomed->next;
            doomed->~version_node_t();
            chain.allocator.deallocate(doomed, 1);
            return detach_outcome_t::detached_k;
        }

        if (chain.head.generation != generation) return detach_outcome_t::not_found_k;
        if (!chain.others) {
            if (destination) *destination = std::move(chain.head);
            return detach_outcome_t::detached_and_emptied_k;
        }

        // A key first staged by one transaction takes that version in the inline slot, so a second
        // transaction staging the same key leaves the recoverable version inline with others behind
        // it. Swapping rather than moving the inline version out and refilling the hole keeps both
        // slots holding a live object at every step, which a value type is entitled to require.
        version_node_t *const promoted = chain.others;
        if (destination) {
            std::swap(chain.head, promoted->entry);
            *destination = std::move(promoted->entry);
        }
        else { chain.head = std::move(promoted->entry); }
        chain.others = promoted->next;
        promoted->~version_node_t();
        chain.allocator.deallocate(promoted, 1);
        return detach_outcome_t::detached_k;
    }

    /**
     *  @brief Looks @p id up and unlinks its version of @p generation, dropping the key when none remain.
     *  @param[in] id Identifier whose chain is shortened.
     *  @param[in] generation The generation to unlink.
     *  @param[out] destination Receives the unlinked version, or null to discard it.
     *  @return Whether anything was unlinked, and whether the key disappeared with it.
     */
    detach_outcome_t chain_discard_(identifier_t const &id, generation_t generation,
                                    versioned_t *destination) noexcept {
        auto found = entries_.find(id);
        if (found == entries_.end()) return detach_outcome_t::not_found_k;
        auto &chain = mutable_ref_(*found);

        // Handing the version back empties the inline slot, and the key the index orders this entry
        // by lives inside it. So the entry leaves the index first, while it still compares correctly,
        // and only then is the version harvested out of the detached node.
        if (destination && !chain.others && chain.head.generation == generation) {
            [[maybe_unused]] bool const taken = storage_shape_t::extract_payload(entries_, id, *destination);
            return detach_outcome_t::detached_and_emptied_k;
        }

        auto const outcome = chain_detach_(chain, generation, destination);
        if (outcome == detach_outcome_t::detached_and_emptied_k) entries_.erase(id);
        return outcome;
    }

    /**
     *  @brief Internal API: Finds the latest visible entry and invokes callback with @c versioned_t const &.
     *    Used by internal methods that need access to generation, presence and publication fields.
     *    Only considers VISIBLE entries (committed/staged).
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c versioned_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void find_visible_entry_(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                             callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, versioned_t const &>,
                      "callback_found must be noexcept invocable with versioned_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        auto found = entries_.find(std::forward<comparable_type_>(comparable));
        versioned_t const *visible = found != entries_.end() ? visible_version_(*found) : nullptr;
        if (visible) callback_found(*visible);
        else callback_missing();
    }

    /**
     *  @brief Internal API: Finds the latest entry regardless of visibility for watch validation.
     *    Checks ALL entries including staged (invisible) ones.
     *    Critical for detecting write-write conflicts with concurrent transactions.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c versioned_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void find_latest_entry_(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                            callback_missing_type_ &&callback_missing = {}) const noexcept {

        auto found = entries_.find(std::forward<comparable_type_>(comparable));
        versioned_t const *latest = found != entries_.end() ? latest_version_(*found) : nullptr;
        if (latest && latest->presence == presence_t::present_k) callback_found(*latest);
        else callback_missing();
    }

    void unmask_and_compact_(identifier_t const &id, generation_t generation_to_unmask) noexcept {
        // ! Don't materialize a new copy of `id` here, use a reference
        auto found = entries_.find(id);
        if (found == entries_.end()) return;
        auto &chain = mutable_ref_(*found);

        // Every other visible version of this key gives way to the one being unmasked, whether it is
        // older or newer - commit order decides, not generation order. Only this key's chain is
        // walked, and outside a staging window it holds a single version, so this usually finds none.
        while (versioned_t const *doomed = visible_version_(chain)) {
            if (doomed->generation == generation_to_unmask) break;
            generation_t const doomed_generation = doomed->generation;
            bool const doomed_was_erased = doomed->presence == presence_t::erased_k;
            auto const outcome = chain_detach_(chain, doomed_generation, nullptr);
            --visible_count_;
            visible_deleted_count_ -= doomed_was_erased;
            if (outcome == detach_outcome_t::detached_and_emptied_k) {
                entries_.erase(id);
                return;
            }
        }

        versioned_t *unmasked = nullptr;
        if (chain.head.generation == generation_to_unmask) { unmasked = &chain.head; }
        else
            for (version_node_t *node = chain.others; node && !unmasked; node = node->next)
                if (node->entry.generation == generation_to_unmask) unmasked = &node->entry;

        if (unmasked && unmasked->publication == publication_t::staged_k) {
            unmasked->publication = publication_t::published_k;
            ++visible_count_;
            if (unmasked->presence == presence_t::erased_k) visible_deleted_count_++;
        }
    }

#pragma endregion Type Definitions

#pragma region Constructors and Assignment

  public:
    transactional_store() noexcept {}

    /** @brief Seeds the underlying tree's allocator, which a stateful allocator needs. */
    explicit transactional_store(allocator_t const &allocator) noexcept
        : entries_(storage_shape_t::template build<versioned_chain_t>(allocator)) {}

    /** @brief Seeds the comparator as well, which a comparator carrying state or a dispatch pointer needs. */
    transactional_store(comparator_t const &comparator, allocator_t const &allocator = {}) noexcept
        : entries_(storage_shape_t::template build<versioned_chain_t>(comparator, allocator)) {}
    transactional_store(transactional_store &&other) noexcept
        : spare_versions_(std::exchange(other.spare_versions_, nullptr)), entries_(std::move(other.entries_)),
          generation_(other.generation_), visible_count_(other.visible_count_),
          visible_deleted_count_(other.visible_deleted_count_) {}

    transactional_store &operator=(transactional_store &&other) noexcept {
        if (this == &other) return *this;
        // The chains being dropped free their own version nodes, so the tree is emptied before the
        // spares are, and neither depends on the other surviving.
        entries_.clear();
        release_spare_versions_();
        spare_versions_ = std::exchange(other.spare_versions_, nullptr);
        entries_ = std::move(other.entries_);
        generation_ = other.generation_;
        visible_count_ = other.visible_count_;
        visible_deleted_count_ = other.visible_deleted_count_;
        return *this;
    }

    ~transactional_store() noexcept {
        entries_.clear();
        release_spare_versions_();
    }

#pragma endregion Constructors and Assignment

#pragma region Capacity

    /**
     *  @brief Returns the number of visible (committed) non-deleted elements in the tree.
     *  @return Number of elements.
     */
    [[nodiscard]] std::size_t size() const noexcept { return visible_count_ - visible_deleted_count_; }

    /**
     *  @brief Checks if the tree has no visible elements.
     *  @return True if empty, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    /**
     *  @brief Returns the number of elements with key equal to the specified argument.
     *    For unique-key containers like this, returns either 0 or 1.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @return Number of elements with key equal to @p comparable (0 or 1).
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t count(comparable_type_ &&comparable) const noexcept {
        return contains(std::forward<comparable_type_>(comparable)) ? 1 : 0;
    }

    /**
     *  @brief Checks if a member @b equal to the given @p comparable exists in the tree.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @return True if element exists, false otherwise.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
        bool found = false;
        find(std::forward<comparable_type_>(comparable), [&](value_t const &) noexcept { found = true; });
        return found;
    }

    /**
     *  @brief Finds and returns a copy of an element equal to @p comparable.
     *    Convenience method to avoid callback-based access in tests and simple use cases.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] allocator Allocator for copying the element (reserved for future use).
     *  @return Result with copied element if found, or failure status.
     */
    template <typename comparable_type_ = identifier_t, typename copy_allocator_>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable,
                                              [[maybe_unused]] copy_allocator_ &&allocator) const noexcept {
        expected<value_t> result {};
        result.status.errc = errc_t::unknown_k;
        find(std::forward<comparable_type_>(comparable), [&](value_t const &v) noexcept {
            auto copy_result = copy_safely(v);
            result.outcome = std::move(*copy_result);
            result.status = copy_result.status;
        });
        return result;
    }

    /**
     *  @brief Finds and returns a copy of an element equal to @p comparable using container's allocator.
     *    Convenience method to avoid callback-based access in tests and simple use cases.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @return Result with copied element if found, or failure status.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        return find_copy(std::forward<comparable_type_>(comparable), storage_shape_t::allocator_of(entries_));
    }

    /**
     *  @brief Finds and returns a copy of the first element not less than (>=) @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Result with copied element if found, or failure status.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
        requires ordered_collection<versioned_chains_t>
    {
        expected<value_t> result;
        result.status.errc = errc_t::key_not_found_k;
        lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &v) noexcept {
                auto copy_result = copy_safely(v);
                if (copy_result) {
                    result.outcome = std::move(*copy_result);
                    result.status = status_t {success_k};
                }
                else { result.status = copy_result.status; }
            },
            [&]() noexcept {});
        return result;
    }

    /**
     *  @brief Finds and returns a copy of the first element greater than (>) @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Result with copied element if found, or failure status.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
        requires ordered_collection<versioned_chains_t>
    {
        expected<value_t> result;
        result.status.errc = errc_t::key_not_found_k;
        upper_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &v) noexcept {
                auto copy_result = copy_safely(v);
                if (copy_result) {
                    result.outcome = std::move(*copy_result);
                    result.status = status_t {success_k};
                }
                else { result.status = copy_result.status; }
            },
            [&]() noexcept {});
        return result;
    }

#pragma endregion Capacity

#pragma region Observers

    /**
     *  @brief Factory method to create a new transactional binary tree without throwing exceptions.
     *    Returns an empty optional on allocation failure.
     *
     *  @param[in] allocator Optional allocator instance.
     *  @return Container instance or empty optional on failure.
     */
    [[nodiscard]] static std::optional<store_t> make(allocator_t const &allocator = {}) noexcept {
        return store_t {allocator};
    }

    /**
     *  @brief Builds a container around a specific comparator, for comparators that carry state.
     *  @param[in] comparator The instance every comparison will consult.
     *  @param[in] allocator Optional allocator instance.
     *  @return Container instance or empty optional on failure.
     */
    [[nodiscard]] static std::optional<store_t> make(comparator_t const &comparator,
                                                     allocator_t const &allocator) noexcept {
        return store_t {comparator, allocator};
    }

#pragma endregion Observers

#pragma region Transaction Management

    /**
     *  @brief Creates a new transaction with a fresh generation number.
     *    Transaction can be reset and reused after commit/rollback to avoid reallocations.
     *    Returns empty optional on allocation failure.
     *
     *  @return Transaction instance or empty optional on failure.
     */
    [[nodiscard]] std::optional<transaction_t> transaction() noexcept { return transaction_t {*this}; }

#pragma endregion Transaction Management

#pragma region Modifiers

    /**
     *  @brief Atomically inserts an element only if the key doesn't exist. Fails if key exists.
     *    This is the strict insert semantics matching @c std::set::insert().
     *
     *  @param[in] value Element to insert (moved into the tree).
     *  @return Success, or @c invalid_argument_k if key exists, or OOM error.
     */
    [[nodiscard]] status_t insert(value_t &&value) noexcept {
        if (contains(value)) return {invalid_argument_k};
        return upsert(std::move(value));
    }

    /**
     *  @brief Atomically inserts an element only if missing. Silently skips if key exists (no error).
     *    This is the "silent no-op" insert semantics.
     *
     *  @param[in] value Element to insert (moved into the tree).
     *  @return Always succeeds (unless OOM). Returns success even if key exists.
     *  @sa The three-argument overload reports which of the two happened, which the status cannot.
     */
    [[nodiscard]] status_t insert_if_missing(value_t &&value) noexcept {
        if (contains(value)) return {success_k};
        return upsert(std::move(value));
    }

    /**
     *  @brief Inserts only if the key is absent, reporting which of the two outcomes occurred.
     *    Declining to overwrite is a success, so the status alone cannot distinguish the cases and a
     *    caller that needs to know would otherwise pay for a membership probe of its own.
     *
     *  @param[in] value Element to insert, moved into the tree when the key is absent.
     *  @param[in] callback_inserted Invoked with the stored element after a fresh insert. Must be @c noexcept.
     *  @param[in] callback_existing Invoked with the element already present, which keeps its value.
     *    Must be @c noexcept.
     *  @return Success, or @c out_of_memory_heap_k when a fresh insert cannot allocate.
     */
    template <typename callback_inserted_type_ = no_op_fn_t, typename callback_existing_type_ = no_op_fn_t>
    [[nodiscard]] status_t insert_if_missing(value_t &&value, callback_inserted_type_ &&callback_inserted,
                                             callback_existing_type_ &&callback_existing) noexcept {
        bool exists = false;
        find(
            mapping_key_or_itself<value_t>(value),
            [&](value_t const &present) noexcept {
                exists = true;
                callback_existing(present);
            },
            []() noexcept {});
        if (exists) return {success_k};

        // The identifier has to be taken before the move, or the lookup below searches by a
        // moved-from key - which compares wrongly rather than failing loudly.
        auto maybe_id = copy_safely(mapping_key_or_itself<value_t>(value));
        if (!maybe_id) return maybe_id.status;

        auto status = upsert(std::move(value));
        if (!status) return status;
        find(maybe_id.outcome, [&](value_t const &stored) noexcept { callback_inserted(stored); }, []() noexcept {});
        return status;
    }

    /**
     *  @brief Atomically upserts an element (insert or update). Always succeeds (unless OOM).
     *    Overwrites existing element if key exists (upsert semantics).
     *
     *  @param[in] element Element to upsert (moved into the tree).
     *  @return Success or error code (e.g., out of memory).
     */
    [[nodiscard]] status_t upsert(value_t &&value) noexcept {
        versioned_t versioned(std::move(value));
        versioned.generation = new_generation_();
        versioned.presence = presence_t::present_k;
        versioned.publication = publication_t::published_k;

        // A direct write carries the newest generation there is, so nothing this key already holds can
        // outrank it, staged versions included. The whole chain gives way to the single new version.
        auto found = entries_.find(mapping_key_or_itself<value_t>(versioned.unversioned));
        if (found != entries_.end()) {
            auto &chain = mutable_ref_(*found);
            if (versioned_t const *displaced = visible_version_(chain)) {
                --visible_count_;
                visible_deleted_count_ -= displaced->presence == presence_t::erased_k;
            }
            chain.release_others();
            chain.head = std::move(versioned);
            ++visible_count_;
            return {success_k};
        }

        auto result = storage_shape_t::upsert(entries_, versioned_chain_t {std::move(versioned)});
        if (!result) return {out_of_memory_heap_k};
        ++visible_count_;
        return {success_k};
    }

    /**
     *  @brief Atomically updates an existing element. Fails if key doesn't exist.
     *    Unlike @c upsert(), this will NOT insert new keys.
     *
     *  @param[in] element Element to update (moved into the tree).
     *  @return Success, @c key_not_found_k if key doesn't exist, or OOM error.
     */
    [[nodiscard]] status_t update(value_t &&value) noexcept {
        if (!contains(value)) return {key_not_found_k};
        return upsert(std::move(value));
    }

    /**
     *  @brief Bulk insert from iterator range (atomic strict semantics).
     *    Fails if ANY key already exists. All elements inserted atomically - if any allocation fails
     *    or any key exists, no changes are made.
     *
     *  @param[in] first Beginning of range to insert.
     *  @param[in] last End of range to insert.
     *  @return Success, invalid_argument_k if any key exists, or out_of_memory_heap_k.
     *    Operation is atomic (all-or-nothing).
     */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t insert(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        if (first == last) return {success_k};

        // Staging is what makes this all-or-nothing: bailing out before `stage()` leaves the store
        // untouched, and duplicates inside the range collapse onto a single staged version.
        auto opened = transaction();
        if (!opened) return {out_of_memory_heap_k};

        for (; first != last; ++first) {
            value_t candidate(*first);
            if (contains(mapping_key_or_itself<value_t>(candidate))) return {invalid_argument_k};
            auto status = opened->insert_if_missing(std::move(candidate));
            if (!status) return status;
        }

        auto stage_status = opened->stage();
        if (!stage_status) return stage_status;
        return opened->commit();
    }

    /**
     *  @brief Bulk insert from iterator range (atomic lenient semantics).
     *    Skips keys that already exist. All new elements inserted atomically - if any allocation fails,
     *    no changes are made.
     *
     *  @param[in] first Beginning of range to insert.
     *  @param[in] last End of range to insert.
     *  @return Success or out_of_memory_heap_k. Operation is atomic (all-or-nothing).
     */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t insert_if_missing(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        auto opened = transaction();
        if (!opened) return {out_of_memory_heap_k};

        for (; first != last; ++first) {
            auto status = opened->insert_if_missing(value_t(*first));
            if (!status) return status;
        }

        auto stage_status = opened->stage();
        if (!stage_status) return stage_status;
        return opened->commit();
    }

    /**
     *  @brief Bulk upsert from iterator range (atomic insert or update semantics).
     *    All elements inserted/updated atomically - if any allocation fails, no changes are made.
     *
     *  @param[in] first Beginning of range to upsert.
     *  @param[in] last End of range to upsert.
     *  @return Success or out_of_memory_heap_k. Operation is atomic (all-or-nothing).
     */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t upsert(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        auto opened = transaction();
        if (!opened) return {out_of_memory_heap_k};

        for (; first != last; ++first) {
            auto status = opened->upsert(value_t(*first));
            if (!status) return status;
        }

        auto stage_status = opened->stage();
        if (!stage_status) return stage_status;
        return opened->commit();
    }

    /**
     *  @brief Updates elements with given range. All keys must exist, operation fails atomically if any missing.
     *    All elements updated atomically - if any key missing, no changes are made.
     *
     *  @param[in] first Beginning of range to update.
     *  @param[in] last End of range to update.
     *  @return Success or key_not_found_k if any key missing. Operation is atomic (all-or-nothing).
     */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t update(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        if (first == last) return {success_k};

        auto opened = transaction();
        if (!opened) return {out_of_memory_heap_k};

        for (; first != last; ++first) {
            value_t candidate(*first);
            if (!contains(mapping_key_or_itself<value_t>(candidate))) return {key_not_found_k};
            auto status = opened->upsert(std::move(candidate));
            if (!status) return status;
        }

        auto stage_status = opened->stage();
        if (!stage_status) return stage_status;
        return opened->commit();
    }

#pragma endregion Modifiers

#pragma region Lookup

    /**
     *  @brief Finds a member @b equal to the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept {

        // A committed erase leaves a visible tombstone, which `find_visible_entry_` still reports
        // because `watch` needs to see it. To a reader the key is gone, so it reads as missing here.
        find_visible_entry_(
            std::forward<comparable_type_>(comparable),
            [&](versioned_t const &versioned) noexcept {
                if (versioned.presence == presence_t::erased_k) callback_missing();
                else callback_found(versioned.unversioned);
            },
            [&]() noexcept { callback_missing(); });
    }

    /**
     *  @brief Finds the first member @b greater or equal to the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept
        requires ordered_collection<versioned_chains_t>
    {

        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        // Inclusive on the first probe, then strictly-greater steps over keys that are staged but not
        // yet committed, so a cursor walk sees every key exactly once and never skips the smallest.
        chain_node_t *cursor = chain_node_t::lower_bound(entries_.root(), comparable, entries_.key_comp());
        versioned_t const *visible = nullptr;
        while (cursor && !(visible = visible_version_(cursor->fruit)))
            cursor = chain_node_t::upper_bound(entries_.root(), cursor->fruit, entries_.key_comp());

        if (visible) callback_found(visible->unversioned);
        else callback_missing();
    }

    /**
     *  @brief Finds all elements equal to a single key. Invokes callback for each matching element.
     *    For trees with unique keys, this returns at most one element (0 or 1).
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_fn_t>
    void equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
        find(std::forward<comparable_type_>(comparable), std::forward<callback_type_>(callback), []() noexcept {});
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept
        requires ordered_collection<versioned_chains_t>
    {

        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        // One entry per key means the strict successor lands on the next key, never on another
        // version of the one the caller just held, so a cursor walk cannot repeat itself.
        chain_node_t *cursor = chain_node_t::upper_bound(entries_.root(), comparable, entries_.key_comp());
        versioned_t const *visible = nullptr;
        while (cursor && !(visible = visible_version_(cursor->fruit)))
            cursor = chain_node_t::upper_bound(entries_.root(), cursor->fruit, entries_.key_comp());

        if (visible) callback_found(visible->unversioned);
        else callback_missing();
    }

#pragma endregion Lookup

#pragma region Range Operations

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) const noexcept
        requires ordered_collection<versioned_chains_t>
    {
        chain_node_t::range(entries_.root(), std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                            entries_.key_comp(), [&](chain_node_t *node) noexcept {
                                if (versioned_t const *visible = visible_version_(node->fruit))
                                    callback(visible->unversioned);
                            });
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires is_mapping<value_t> && ordered_collection<versioned_chains_t>
    {
        generation_t generation = new_generation_();
        chain_node_t::range(entries_.root(), std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                            entries_.key_comp(), [&](chain_node_t *node) noexcept {
                                auto &chain = node->fruit;
                                auto *visible = const_cast<versioned_t *>(visible_version_(chain));
                                if (!visible) return;
                                callback(visible->unversioned.key, visible->unversioned.mapped);
                                visible->generation = generation;
                            });
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires ordered_collection<versioned_chains_t>
    {
        // The whole chain goes, staged versions included, so the counters only owe the visible one.
        entries_.erase_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                             [&](versioned_chain_t const &chain) noexcept {
                                 if (versioned_t const *visible = visible_version_(chain)) {
                                     callback(visible->unversioned);
                                     --visible_count_;
                                     visible_deleted_count_ -= visible->presence == presence_t::erased_k;
                                 }
                             });
        return status_t {success_k};
    }

#pragma endregion Range Operations

#pragma region Sampling

    template <typename lower_type_, typename upper_type_, typename generator_type_,
              typename callback_type_ = no_op_fn_t>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                      callback_type_ &&callback) const noexcept
        requires ordered_collection<versioned_chains_t>
    {

        auto node = chain_node_t::sample_range( //
            entries_.root(), lower, upper, entries_.key_comp(), std::forward<generator_type_>(generator),
            [](chain_node_t *candidate) noexcept { return visible_version_(candidate->fruit) != nullptr; });
        // Callers see the stored value; the version metadata never leaves this class.
        if (node) callback(visible_version_(node->fruit)->unversioned);
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator, std::size_t &seen,
                      std::size_t reservoir_capacity, output_iterator_type_ &&reservoir) const noexcept
        requires ordered_collection<versioned_chains_t>
    {

        using output_iterator_t = std::remove_reference_t<output_iterator_type_>;
        using output_category_t = typename std::iterator_traits<output_iterator_t>::iterator_category;
        static_assert(std::is_same<std::random_access_iterator_tag, output_category_t>(), "Must be random access!");

        auto sampler = [&](value_t const &value) noexcept {
            if (seen < reservoir_capacity) reservoir[seen] = value;

            else {
                auto slot_to_replace = draw_below(generator, seen + 1);
                if (slot_to_replace < reservoir_capacity) reservoir[slot_to_replace] = value;
            }

            ++seen;
        };
        range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), sampler);
    }

#pragma endregion Sampling

#pragma region Order Statistics

    template <typename callback_found_type_ = no_op_fn_t, typename callback_missing_type_ = no_op_fn_t>
    void select(std::size_t k, callback_found_type_ &&callback_found,
                callback_missing_type_ &&callback_missing = {}) const noexcept
        requires supports_order_statistics<versioned_chains_t>
    {
        std::size_t visible_index = 0;
        bool found = false;

        chain_node_t::for_each_left_right(entries_.root(), [&](chain_node_t *node) noexcept {
            versioned_t const *visible = visible_version_(node->fruit);
            if (!visible || visible->presence == presence_t::erased_k) return;
            if (visible_index == k) {
                callback_found(visible->unversioned);
                found = true;
                return;
            }
            ++visible_index;
        });

        if (!found) callback_missing();
    }

    /**
     *  @brief Finds the rank (position) of an element among visible (committed) elements.
     *    Only available for tree implementations that support order statistics (e.g., WB trees).
     *    Requires O(log n) time for weight-balanced trees.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive the rank (size_t). Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if element not found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    void rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept
        requires supports_order_statistics<versioned_chains_t>
    {
        std::size_t rank_value = 0;
        bool found = false;
        identifier_t target_id(comparable);
        auto const less = entries_.key_comp();

        chain_node_t::for_each_left_right(entries_.root(), [&](chain_node_t *node) noexcept {
            versioned_t const *visible = visible_version_(node->fruit);
            if (!visible || visible->presence == presence_t::erased_k) return;

            if (less.same(visible->unversioned, target_id)) {
                callback_found(rank_value);
                found = true;
                return;
            }

            if (less(visible->unversioned, target_id)) ++rank_value;
        });

        if (!found) callback_missing();
    }

    /**
     *  @brief Erases a single entry matching the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive the erased entry. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     *  @return @c key_not_found_k if no visible entry matched, otherwise success.
     *  @note Both channels agree: an absent key fires @p callback_missing @b and reports
     *    @c key_not_found_k, so a caller may read the answer from whichever suits it and never
     *    needs a membership probe of its own beforehand.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
              typename callback_missing_type_ = no_op_fn_t>
    status_t erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found = {},
                   callback_missing_type_ &&callback_missing = {}) noexcept {
        // `find` already unwraps to the stored value, and `comparable` is read twice below,
        // so it stays an lvalue rather than being forwarded away on the first use.
        bool found = false;
        find(
            comparable,
            [&](value_t const &value) noexcept {
                found = true;
                callback_found(value);
            },
            []() noexcept {});

        if (!found) {
            callback_missing();
            return status_t {errc_t::key_not_found_k};
        }

        // One entry holds every version of the key, so dropping it drops them all.
        auto doomed = entries_.find(comparable);
        if (doomed != entries_.end())
            if (versioned_t const *visible = visible_version_(*doomed)) {
                --visible_count_;
                visible_deleted_count_ -= visible->presence == presence_t::erased_k;
            }
        entries_.erase(comparable);
        return status_t {success_k};
    }

    /**
     *  @brief Hints to the tree to pre-allocate memory. No-op for tree implementations.
     *    Provided for API consistency with other containers. Doesn't guarantee subsequent
     *    insertions won't fail with "out of memory".
     *
     *  @param[in] size Suggested capacity (ignored for tree structures).
     *  @return Always succeeds.
     *
     *  @note This is a no-op because tree structures don't support reserving capacity efficiently.
     */
    [[nodiscard]] status_t reserve(std::size_t) noexcept { return {success_k}; }

    /**
     *  @brief Removes all elements from the tree and resets generation counter.
     *    Always succeeds here, but reports a status to match the partitioned collection, which allocates.
     */
    [[nodiscard]] status_t clear() noexcept {
        entries_.clear();
        generation_ = 0;
        visible_count_ = 0;
        visible_deleted_count_ = 0;
        return {success_k};
    }

    /**
     *  @brief Debug utility to print tree contents.
     *    Templated on the sink so the header never pulls in a stream of its own.
     */
    template <typename stream_type_>
    void print(stream_type_ &stream)
        requires ordered_collection<versioned_chains_t>
    {
        stream << "Items: " << entries_.size() << "\n";
        stream << "Imbalance: " << entries_.total_imbalance() << "\n";
        auto show = [&](versioned_t const &version) {
            char const *marker = version.publication == publication_t::published_k ? "✓" : "✗";
            stream << identifier_t {version.unversioned} << " @" << version.generation;
            stream << marker << " ";
        };
        chain_node_t::for_each_left_right(entries_.root(), [&](chain_node_t *node) {
            show(node->fruit.head);
            for (version_node_t const *other = node->fruit.others; other; other = other->next) show(other->entry);
        });
        stream << "\n";
    }
};

/**
 *  @brief STL-style transactional set using AVL tree.
 *    Stores unique elements in sorted order with ACID transaction semantics.
 *
 *  @tparam value_type_ Type of elements stored in the set.
 *  @tparam comparator_type_ Comparator for ordering elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<value_type_>>
using transactional_avl_set = transactional_store<basic_avl_tree<value_type_, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional map using AVL tree.
 *    Stores key-value pairs in sorted order with ACID transaction semantics.
 *
 *  @tparam key_type_ Type of keys stored in the map.
 *  @tparam value_type_ Type of values stored in the map.
 *  @tparam comparator_type_ Comparator for ordering keys. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using transactional_avl_map =
    transactional_store<basic_avl_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional set using weight-balanced tree with order statistics support.
 *    Stores unique elements in sorted order with ACID transaction semantics and O(log n) rank/select operations.
 *
 *  @tparam value_type_ Type of elements stored in the set.
 *  @tparam comparator_type_ Comparator for ordering elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<value_type_>>
using transactional_wb_set = transactional_store<basic_wb_tree<value_type_, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional map using weight-balanced tree with order statistics support.
 *    Stores key-value pairs in sorted order with ACID transaction semantics and O(log n) rank/select operations.
 *
 *  @tparam key_type_ Type of keys stored in the map.
 *  @tparam value_type_ Type of values stored in the map.
 *  @tparam comparator_type_ Comparator for ordering keys. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using transactional_wb_map =
    transactional_store<basic_wb_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional set using an open-addressed hash table.
 *    Point access only - no bounds, ranges, or order statistics, as the core supplies no ordering.
 *
 *  @tparam key_type_ Type of elements stored in the set.
 *  @tparam hasher_type_ Hasher for placing elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam equals_type_ Equality for resolving collisions. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for the table's slabs, defaults to @c std::allocator.
 */
template <typename key_type_, typename hasher_type_ = default_hash_t, typename equals_type_ = equal_to_t,
          typename allocator_type_ = std::allocator<std::byte>>
using transactional_hash_set =
    transactional_store<basic_hash_table<key_type_, hasher_type_, equals_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional map using an open-addressed hash table.
 *    Point access only - no bounds, ranges, or order statistics, as the core supplies no ordering.
 *
 *  @tparam key_type_ Type of keys stored in the map.
 *  @tparam value_type_ Type of values stored in the map.
 *  @tparam hasher_type_ Hasher for placing keys. Define @c is_transparent for heterogeneous lookups.
 *  @tparam equals_type_ Equality for resolving collisions. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for the table's slabs, defaults to @c std::allocator.
 */
template <typename key_type_, typename value_type_, typename hasher_type_ = default_hash_t,
          typename equals_type_ = equal_to_t, typename allocator_type_ = std::allocator<std::byte>>
using transactional_hash_map =
    transactional_store<basic_hash_table<mapping<key_type_, value_type_>, hasher_type_, equals_type_, allocator_type_>>;

#pragma endregion Order Statistics

} // namespace ashvardanian::smashtable
