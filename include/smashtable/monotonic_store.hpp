/**
 *  @brief Generic transactional container with ACID semantics, providing 2-phase commit transactions. Can be
 *      instantiated with any key-addressable core - AVL trees, weight-balanced trees, open-addressed tables -
 *      and gates its ordered surface behind the cores that supply an ordering.
 *      All operations use callback-based APIs and are exception-free via @c noexcept constraints.
 *  @author Ash Vardanian
 *  @file include/smashtable/monotonic_store.hpp
 *  @date October 13, 2022
 */
#pragma once
#include <cassert> // `assert`

#include <memory>  // `std::allocator`, `std::construct_at`, `std::destroy_at`
#include <utility> // `std::exchange`

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
 *  @section monotonic_store_design_goals Design Goals
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
 *  allocator behavior. So all entry identifiers must either be nothrow-copyable or provide a @c copy() member method
 *  returning an @c expected<T> to support safe copying for watch bookkeeping.
 *
 *  @see https://jepsen.io/consistency/models/monotonic-atomic-view
 *  @see https://jepsen.io/consistency/models/read-committed
 *
 *  @section monotonic_store_api_overview API Overview
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
 *  | Method               | Key Exists?     | Returns              | Use Case                                   |
 *  |----------------------|-----------------|----------------------|--------------------------------------------|
 *  | insert()             | Fails (error)   | key_already_exists_k | Strict: ensure key is new                  |
 *  | insert_if_missing()  | Skips (success) | success_k            | Lenient: insert only if absent, else no-op |
 *  | upsert()             | Overwrites      | success_k            | Always update regardless of existence      |
 *  | update()             | Fails (error)   | key_not_found_k      | Strict: ensure key exists before updating  |
 *
 *  @tparam collection_type_ The underlying core, satisfying @c key_addressable_collection and providing a
 *    @c rebind template alias. Cores that also satisfy @c ordered_collection unlock the ordered surface -
 *    bounds, ranges, and, with order statistics, @c select and @c rank.
 */
template <typename collection_type_>
class monotonic_store {

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

    /** @brief What this store calls itself, so generic code spells an engine and a wrapper alike. */
    using store_t = monotonic_store;

    /**
     *  @brief The one shape a watch records, whatever the read that produced it looked like.
     *
     *  A committed tombstone is handed out as found - the public @c find has to see it in order to
     *  hide it - so a watch on one has to record the same shape a watch on an absent key records, or
     *  it could never match itself. Validation re-derives the shape here too, so the two cannot drift.
     *
     *  @param[in] resolved The version a read resolved to, or null when the key resolves to nothing.
     */
    [[nodiscard]] static watch_t watch_shape_of(versioned_t const *resolved) noexcept {
        if (!resolved || resolved->presence != presence_t::present_k) return missing_watch();
        return watch_t {resolved->generation, resolved->presence};
    }

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
        ST_NO_UNIQUE_ADDRESS_ version_allocator_t allocator {};

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
    enum class detach_outcome_t : std::uint8_t {
        /** @brief No version carried that generation, so nothing was taken out. */
        not_found_k,
        /** @brief The version left the chain, which still holds others. */
        detached_k,
        /** @brief The version left the chain, which is now empty and no longer worth an entry. */
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

    using changed_identifiers_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<identifier_t>;
    using changed_identifiers_vector_t = basic_vector<identifier_t, changed_identifiers_allocator_t>;

    /** @brief Holds the members a revising walk collected before it stages any of them back. */
    using values_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<value_t>;
    using values_vector_t = basic_vector<value_t, values_allocator_t>;

  public:
    class transaction_t {

        friend store_t;
        store_t *store_ {nullptr};
        versioned_set_t changes_ {};
        watches_vector_t watches_ {};
        changed_identifiers_vector_t changed_identifiers_ {};
        generation_t generation_ {0};
        staging_t staging_ {staging_t::pending_k};

        transaction_t(store_t &set) noexcept
            : store_(&set), changes_(storage_shape_t::template sibling<versioned_t>(set.entries_)),
              watches_(watches_allocator_t(storage_shape_t::allocator_of(set.entries_))),
              changed_identifiers_(changed_identifiers_allocator_t(storage_shape_t::allocator_of(set.entries_))),
              generation_(set.next_generation_()) {}

        /**
         *  @brief Drops this transaction's version of the first @p processed changed identifiers.
         *    Nothing else can carry this generation, so a key that has no other version disappears.
         *  @param[in] processed How many leading entries of the changed-identifier list to undo.
         */
        void unstage_(std::size_t processed) noexcept {
            auto &store = store_ref();
            for (std::size_t index = 0; index != processed; ++index)
                // ! Don't materialize a new copy of the identifier here, use a reference
                store.chain_discard_(changed_identifiers_[index], generation_, nullptr);
        }

        store_t &store_ref() noexcept { return *store_; }
        store_t const &store_ref() const noexcept { return *store_; }

        /** @brief Stages @p versioned under this transaction's generation, recording @p identifier as changed. */
        [[nodiscard]] status_t stage_(identifier_t &&identifier, versioned_t &&versioned) noexcept {
            auto reserve_status = changed_identifiers_.reserve(changed_identifiers_.size() + 1);
            if (failed(reserve_status)) return reserve_status;
            versioned.generation = generation_;
            auto result = storage_shape_t::upsert(changes_, std::move(versioned));
            if (failed(result)) return out_of_memory_heap_k;
            [[maybe_unused]] status_t const recorded =
                changed_identifiers_.push_back(assume_reserved, std::move(identifier));
            return success_k;
        }

        /**
         *  @brief Hands the smaller of a staged and a committed candidate to @p callback_found.
         *    Either may be null, and two nulls mean the merged view has nothing left to show.
         */
        template <typename callback_found_type_, typename callback_missing_type_>
        void merge_first_(versioned_t const *staged, versioned_t const *committed,
                          callback_found_type_ &&callback_found,
                          callback_missing_type_ &&callback_missing) const noexcept {
            if (!staged && !committed) {
                callback_missing();
                return;
            }
            if (!staged) {
                callback_found(committed->payload);
                return;
            }
            if (!committed) {
                callback_found(staged->payload);
                return;
            }
            auto const &ordering = changes_.key_comp();
            if (ordering.less(committed->payload, staged->payload)) callback_found(committed->payload);
            else callback_found(staged->payload);
        }

        /**
         *  @brief Whether every watched key still carries the version this transaction read.
         *    Asked twice - once when reserving, once when publishing - because a commit landing in
         *    between is the only thing that can invalidate a read after it was validated.
         */
        [[nodiscard]] status_t validate_watches_() const noexcept {
            auto const &store = store_ref();
            for (watched_identifier_t const &watched : watches_) {
                watch_t latest = missing_watch();
                store.find_committed_entry_(
                    watched.identifier, [&](versioned_t const &entry) noexcept { latest = watch_shape_of(&entry); });
                if (latest != watched.watch) return status_t::consistency_k;
            }
            return success_k;
        }

        /**
         *  @brief Drops anything staged and never published, and gives up this transaction's claim.
         *
         *  Abandoning a staged transaction is the only way the store can accumulate versions that no
         *  generation can ever name again, so the unwind is not optional. A null @c store_ marks a
         *  moved-from transaction, which owns nothing and must undo nothing.
         */
        void unwind_() noexcept {
            if (!store_) return;
            if (staging_ == staging_t::staged_k) unstage_(changed_identifiers_.size());
            staging_ = staging_t::pending_k;
            store_ = nullptr;
        }

        /**
         *  @brief Whether this transaction already stages a version of @p comparable.
         *
         *  The change set is a core this transaction owns, so its lookup answers without refusing;
         *  the status it returns is folded away here rather than at each of the walks that ask.
         */
        template <typename comparable_type_>
        [[nodiscard]] bool stages_(comparable_type_ const &comparable) const noexcept {
            expected<bool> const staged = changes_.contains(comparable);
            return staged && *staged;
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
              watches_(std::move(other.watches_)), changed_identifiers_(std::move(other.changed_identifiers_)),
              generation_(other.generation_), staging_(std::exchange(other.staging_, staging_t::pending_k)) {}

        transaction_t &operator=(transaction_t &&other) noexcept {
            if (this == &other) return *this;
            unwind_();
            store_ = std::exchange(other.store_, nullptr);
            changes_ = std::move(other.changes_);
            watches_ = std::move(other.watches_);
            changed_identifiers_ = std::move(other.changed_identifiers_);
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
            auto staged_iterator = changes_.find(value);
            if (staged_iterator != changes_.end() && (*staged_iterator).presence == presence_t::present_k)
                return key_already_exists_k;
            expected<bool> const key_is_present = store_ref().contains(value);
            if (!key_is_present) return key_is_present.status();
            if (*key_is_present) return key_already_exists_k;
            return upsert(std::move(value));
        }

        /**
         *  @brief Stages an insert operation only if key is missing. Silently skips if key exists (no error).
         *    Checks both transaction changes and main store for existence.
         *
         *  @param[in] value Element to insert (moved into the transaction).
         *  @return Always succeeds (unless OOM). Returns success even if key exists.
         */
        [[nodiscard]] status_t insert_if_missing(value_t &&value) noexcept {
            auto staged_iterator = changes_.find(value);
            if (staged_iterator != changes_.end() && (*staged_iterator).presence == presence_t::present_k)
                return success_k;
            expected<bool> const key_is_present = store_ref().contains(value);
            if (!key_is_present) return key_is_present.status();
            if (*key_is_present) return success_k;
            return upsert(std::move(value));
        }

        /**
         *  @brief Stages an upsert operation for the given element (insert or update). Always succeeds.
         *    Overwrites existing element if key exists. Changes visible after @c stage() and @c commit().
         *
         *  @param[in] value Element to upsert (moved into the transaction).
         *  @return Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t upsert(value_t &&value) noexcept {
            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
            if (!maybe_identifier) return out_of_memory_heap_k;
            versioned_t versioned(std::move(value));
            versioned.presence = presence_t::present_k;
            return stage_(std::move(*maybe_identifier), std::move(versioned));
        }

        /**
         *  @brief Stages an update operation for existing keys only.
         *    Fails if key doesn't exist anywhere (local changes or main store).
         *
         *  @param[in] value Element to update (moved into the transaction).
         *  @return Success, or @c key_not_found_k if key doesn't exist.
         */
        [[nodiscard]] status_t update(value_t &&value) noexcept {
            auto staged_iterator = changes_.find(value);
            if (staged_iterator != changes_.end() && (*staged_iterator).presence == presence_t::present_k)
                return upsert(std::move(value));
            expected<bool> const key_is_present = store_ref().contains(value);
            if (!key_is_present) return key_is_present.status();
            if (!*key_is_present) return key_not_found_k;
            return upsert(std::move(value));
        }

        /**
         *  @brief Stages an erase operation for the given identifier.
         *    Marks the entry as deleted in the transaction. Actual removal happens on commit.
         *
         *  @param[in] identifier Identifier of the element to erase.
         *  @return Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t erase(identifier_t const &identifier) noexcept {
            // The tombstone owns its own identifier, and the list of changed identifiers owns another,
            // so a move-only key needs two safe copies rather than one copy and one implicit one.
            auto maybe_identifier = copy_safely<identifier_t>(identifier);
            if (!maybe_identifier) return out_of_memory_heap_k;
            auto maybe_payload = copy_safely<identifier_t>(identifier);
            if (!maybe_payload) return out_of_memory_heap_k;

            versioned_t versioned(value_t {std::move(*maybe_payload)});
            versioned.presence = presence_t::erased_k;
            return stage_(std::move(*maybe_identifier), std::move(versioned));
        }

        [[nodiscard]] status_t reserve(std::size_t size) noexcept { return watches_.reserve(size); }

        /**
         *  @brief Records what this transaction resolves @p identifier to, so a later commit can refuse
         *    if anything published over it in the meantime.
         *
         *  @param[in] identifier Identifier to watch, borrowed and copied into the read set - the read
         *    set outlives the call, and a caller's identifier is never consumed by a read.
         *  @return Success unless the read set could not grow, or the identifier could not be copied.
         */
        [[nodiscard]] status_t watch(identifier_t const &identifier) noexcept {
            auto maybe_identifier = copy_safely<identifier_t>(identifier);
            if (!maybe_identifier) return maybe_identifier.status();

            watch_t shape = watch_shape_of(nullptr);
            store_ref().find_visible_entry_(
                identifier, [&](versioned_t const &versioned) noexcept { shape = watch_shape_of(&versioned); },
                no_op_t {});
            return watches_.push_back({std::move(*maybe_identifier), shape});
        }

        /**
         *  @brief Finds a member equal to @p comparable and records what it saw into the read set.
         *    The watching counterpart to @c find: this one can fail, because a read set is memory.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find_and_watch(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                              callback_missing_type_ &&callback_missing = {}) noexcept {
            auto maybe_identifier = copy_safely<identifier_t>(comparable);
            if (!maybe_identifier) return maybe_identifier.status();
            status_t const recorded = watch(*maybe_identifier);
            if (failed(recorded)) return recorded;
            return find(std::forward<comparable_type_>(comparable), std::forward<callback_found_type_>(callback_found),
                        std::forward<callback_missing_type_>(callback_missing));
        }

        /** @brief Records @p versioned as the version this transaction read of its own key. */
        [[nodiscard]] status_t watch(versioned_t const &versioned) noexcept {
            // Through `copy_safely` rather than a braced `identifier_t`, so a move-only identifier
            // compiles here and an identifier that allocates reports instead of throwing.
            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(versioned.payload));
            if (!maybe_identifier) return maybe_identifier.status();
            return watches_.push_back({std::move(*maybe_identifier), watch_shape_of(&versioned)});
        }

        /**
         *  @brief Finds a member @b equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c monotonic_store::find(), will include the entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept {
            if (auto iterator = changes_.find(std::forward<comparable_type_>(comparable)); iterator != changes_.end()) {
                (*iterator).presence == presence_t::present_k ? callback_found((*iterator).payload)
                                                              : callback_missing();
                return success_k;
            }
            return store_ref().find(std::forward<comparable_type_>(comparable),
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
            expected<value_t> result {status_t::key_not_found_k};

            // Only the fall-through branch forwards: a lookup that misses `changes_` is the last use, and
            // forwarding at both sites would hand the second one an already moved-from object.
            if (auto it = changes_.find(comparable); it != changes_.end()) {
                if ((*it).presence == presence_t::present_k) { result = copy_safely((*it).payload); }
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
        [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
            bool present = false;
            status_t const answered = find(
                std::forward<comparable_type_>(comparable), [&](auto const &) noexcept { present = true; }, no_op_t {});
            if (failed(answered)) return answered;
            return present;
        }

        /**
         *  @brief Finds the first member @b greater or equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c monotonic_store::lower_bound(), will include entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        /**
         *  @brief Hands @p callback_found the smallest member visible here, staged writes included.
         *
         *  Walks from the first entry until one is readable, so a store whose front is all tombstones pays
         *  for that prefix - the cost a caller should assume is the run of unreadable entries, not a descent.
         *
         *  The unbounded case of @c lower_bound, which is what a merged walk over several stores needs to
         *  open with: it asks for a first key rather than an ordinal, so a core keeping no subtree counts
         *  can answer it.
         *
         *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered when nothing is readable. Must be @c noexcept.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t smallest(callback_found_type_ &&callback_found,
                                        callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_collection<versioned_chains_t>
        {
            auto const &store = store_ref();
            versioned_t const *store_visible = nullptr;
            for (auto cursor = store.entries_.begin(); cursor != store.entries_.end(); ++cursor) {
                if (stages_(*cursor)) continue;
                if ((store_visible = store_t::readable_version_(*cursor)) != nullptr) break;
            }

            auto staged = changes_.begin();
            while (staged != changes_.end() && staged->presence == presence_t::erased_k) ++staged;

            merge_first_(staged != changes_.end() ? &*staged : nullptr, store_visible,
                         std::forward<callback_found_type_>(callback_found),
                         std::forward<callback_missing_type_>(callback_missing));
            return success_k;
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_collection<versioned_chains_t>
        {

            auto const &store = store_ref();
            versioned_t const *store_visible = nullptr;
            for (auto cursor = store.entries_.lower_bound(comparable); cursor != store.entries_.end(); ++cursor) {
                if (stages_(*cursor)) continue;
                if ((store_visible = store_t::readable_version_(*cursor)) != nullptr) break;
            }

            auto changed_lb = changes_.lower_bound(comparable);
            while (changed_lb != changes_.end() && changed_lb->presence == presence_t::erased_k) { ++changed_lb; }

            merge_first_(changed_lb != changes_.end() ? &*changed_lb : nullptr, store_visible,
                         std::forward<callback_found_type_>(callback_found),
                         std::forward<callback_missing_type_>(callback_missing));
            return success_k;
        }

        /**
         *  @brief Finds all elements equal to a single key. Invokes callback for each matching element.
         *    For sets with unique keys, returns at most one element (0 or 1).
         *    Includes transaction changes.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
            return find(std::forward<comparable_type_>(comparable), std::forward<callback_type_>(callback), no_op_t {});
        }

        /**
         *  @brief Iterates over all entries in the range [ @p lower, @p upper), including transaction changes.
         *    The two sides are merged as the walk goes, so the output is sorted however they interleave.
         *
         *  @param[in] lower Lower bound of the range (inclusive).
         *  @param[in] upper Upper bound of the range (exclusive).
         *  @param[in] callback Callback invoked for each element in range. Must be @c noexcept.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept
            requires ordered_collection<versioned_chains_t>
        {
            auto const less = changes_.key_comp();
            return walk_from_(
                std::forward<lower_type_>(lower),
                [&](auto const &candidate) noexcept { return less(candidate, upper); },
                std::forward<callback_type_>(callback));
        }

        /** @brief Copies the lowest member this transaction reads, so a walk has a key to start from. */
        [[nodiscard]] expected<value_t> smallest_copy_() const noexcept
            requires ordered_collection<versioned_chains_t>
        {
            expected<value_t> result {status_t::key_not_found_k};
            if (status_t const answered =
                    smallest([&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
                failed(answered))
                return answered;
            return result;
        }

        /**
         *  @brief Walks from @p lower while @p within accepts, hands every member to @p callback, and
         *    stages a tombstone for each.
         *
         *  The walk finishes before the first tombstone is staged, because a staged write changes what the
         *  merged view answers and a walk revising itself would step over its own neighbours.
         */
        template <typename within_type_, typename lower_type_, typename callback_type_>
        [[nodiscard]] status_t erase_walked_(within_type_ &&within, lower_type_ &&lower,
                                             callback_type_ &&callback) noexcept
            requires ordered_collection<versioned_chains_t>
        {
            changed_identifiers_vector_t doomed(
                changed_identifiers_allocator_t(storage_shape_t::allocator_of(store_ref().entries_)));
            status_t collecting = success_k;
            [[maybe_unused]] status_t const walked = walk_from_(
                std::forward<lower_type_>(lower), std::forward<within_type_>(within),
                [&](value_t const &value) noexcept {
                    if (failed(collecting)) return;
                    auto owned = copy_safely<identifier_t>(identifier_t {mapping_key_or_itself<value_t>(value)});
                    if (!owned) {
                        collecting = owned.status();
                        return;
                    }
                    callback(value);
                    if (status_t const kept = doomed.push_back(std::move(*owned)); failed(kept)) collecting = kept;
                });
            if (failed(collecting)) return collecting;

            for (std::size_t index = 0; index != doomed.size(); ++index)
                if (status_t const staged = erase(doomed[index]); failed(staged)) return staged;
            return success_k;
        }

        /**
         *  @brief Merges the staged and committed sides from @p lower, while @p within accepts the candidate.
         *
         *  The bounds are compared through the transparent comparator rather than materialized as an
         *  @c identifier_t, which a move-only key cannot copy into. A key this transaction touched is
         *  answered from its own version, so the committed side skips whatever @c changes_ speaks for and
         *  no key is emitted twice.
         */
        template <typename lower_type_, typename within_type_, typename callback_type_>
        [[nodiscard]] status_t walk_from_(lower_type_ &&lower, within_type_ &&within,
                                          callback_type_ &&callback) const noexcept
            requires ordered_collection<versioned_chains_t>
        {
            auto const &store = store_ref();
            auto staged_cursor = changes_.lower_bound(lower);
            auto committed_cursor = store.entries_.lower_bound(lower);

            // A key this transaction touched is answered from its own version, so the committed side
            // skips whatever `changes_` already speaks for and never emits a key twice.
            auto const less = changes_.key_comp();
            for (;;) {
                versioned_t const *staged = nullptr;
                while (!staged && staged_cursor != changes_.end() && within(*staged_cursor)) {
                    if (staged_cursor->presence == presence_t::present_k) staged = &*staged_cursor;
                    else ++staged_cursor;
                }
                versioned_t const *committed = nullptr;
                while (!committed && committed_cursor != store.entries_.end() && within(*committed_cursor)) {
                    if (!stages_(*committed_cursor)) committed = store_t::readable_version_(*committed_cursor);
                    if (!committed) ++committed_cursor;
                }

                if (!staged && !committed) break;
                if (!staged || (committed && less(committed->payload, staged->payload))) {
                    callback(committed->payload);
                    ++committed_cursor;
                }
                else {
                    callback(staged->payload);
                    ++staged_cursor;
                }
            }
            return success_k;
        }

        /**
         *  @brief Hands @p callback every element this transaction reads, in whatever order the core holds them.
         *
         *  The one walk an unordered core can offer, so it promises no ordering even where the core has
         *  one. Every element a @c find of this transaction would answer with at the moment of the call is
         *  visited @b exactly @b once - its own staged writes included, its own tombstones and every version
         *  another transaction has not committed excluded. Nothing may write to the transaction or its store
         *  while the walk runs.
         *
         *  @param[in] callback Callback invoked for each element. Must be @c noexcept.
         */
        template <typename callback_type_ = no_op_t>
        [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept {
            static_assert(is_safe_callback_for<callback_type_, value_t const &>,
                          "callback must be noexcept invocable with value_t const &");

            for (auto staged = changes_.begin(); staged != changes_.end(); ++staged)
                if ((*staged).presence == presence_t::present_k) callback((*staged).payload);

            // A key this transaction touched is answered from its own version above, so the committed
            // side skips whatever `changes_` already speaks for and never emits a key twice.
            auto const &store = store_ref();
            for (auto committed = store.entries_.begin(); committed != store.entries_.end(); ++committed) {
                if (stages_(*committed)) continue;
                if (versioned_t const *readable = store_t::readable_version_(*committed)) callback(readable->payload);
            }
            return success_k;
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept
            requires ordered_collection<versioned_chains_t>
        {

            auto const &store = store_ref();
            versioned_t const *store_visible = nullptr;
            for (auto cursor = store.entries_.upper_bound(comparable); cursor != store.entries_.end(); ++cursor) {
                if (stages_(*cursor)) continue;
                if ((store_visible = store_t::readable_version_(*cursor)) != nullptr) break;
            }

            auto changed_ub = changes_.upper_bound(comparable);
            while (changed_ub != changes_.end() && changed_ub->presence == presence_t::erased_k) { ++changed_ub; }

            merge_first_(changed_ub != changes_.end() ? &*changed_ub : nullptr, store_visible,
                         std::forward<callback_found_type_>(callback_found),
                         std::forward<callback_missing_type_>(callback_missing));
            return success_k;
        }

        /**
         *  @brief Finds the @p ordinal -th smallest element, merging staged changes with the store.
         *    Walks both sides in order, since a subtree weight counts versions rather than visible values.
         *    Instantiates only for a core carrying order statistics, which excludes the AVL aliases.
         *
         *  @param[in] ordinal Zero-based position among the elements this transaction can see.
         *  @param[in] callback_found Callback to receive the element. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered when fewer elements are visible. Must be @c noexcept.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                                      callback_missing_type_ &&callback_missing = {}) const noexcept
            requires supports_order_statistics<versioned_chains_t>
        {
            std::size_t visible_index = 0;
            bool found = false;
            auto const less = changes_.key_comp();

            auto const &store = store_ref();
            auto staged_iterator = changes_.begin();
            auto committed_iterator = store.entries_.begin();

            while ((staged_iterator != changes_.end() || committed_iterator != store.entries_.end()) && !found) {
                bool take_local = false;

                if (staged_iterator == changes_.end()) { take_local = false; }
                else if (committed_iterator == store.entries_.end()) { take_local = true; }
                else { take_local = less(staged_iterator->payload, *committed_iterator); }

                if (take_local) {
                    if (staged_iterator->presence == presence_t::present_k) {
                        if (visible_index == ordinal) {
                            callback_found(staged_iterator->payload);
                            found = true;
                        }
                        ++visible_index;
                    }
                    ++staged_iterator;
                }
                else {
                    versioned_t const *store_visible = store_t::readable_version_(*committed_iterator);
                    if (store_visible &&
                        changes_.find(mapping_key_or_itself<value_t>(store_visible->payload)) == changes_.end()) {
                        if (visible_index == ordinal) {
                            callback_found(store_visible->payload);
                            found = true;
                        }
                        ++visible_index;
                    }

                    ++committed_iterator;
                }
            }

            if (!found) callback_missing();
            return success_k;
        }

        /**
         *  @brief Finds the rank (position) of an element, merging staged changes with the store.
         *    Walks both sides in order, since a subtree weight counts versions rather than visible values.
         *    Instantiates only for a core carrying order statistics, which excludes the AVL aliases.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive the rank (size_t). Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if element not found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept
            requires supports_order_statistics<versioned_chains_t>
        {
            std::size_t rank_value = 0;
            bool found = false;
            auto const less = changes_.key_comp();

            auto local_target = changes_.find(comparable);
            if (local_target != changes_.end() && local_target->presence == presence_t::present_k) { found = true; }
            else {
                expected<bool> const seen = store_ref().contains(comparable);
                if (!seen) return seen.status();
                found = *seen;
            }

            if (!found) {
                callback_missing();
                return success_k;
            }

            auto const &store = store_ref();
            auto staged_iterator = changes_.begin();
            auto committed_iterator = store.entries_.begin();

            while (staged_iterator != changes_.end() || committed_iterator != store.entries_.end()) {
                bool take_local = false;

                if (staged_iterator == changes_.end()) { take_local = false; }
                else if (committed_iterator == store.entries_.end()) { take_local = true; }
                else { take_local = less(staged_iterator->payload, *committed_iterator); }

                if (take_local) {
                    if (staged_iterator->presence == presence_t::present_k &&
                        less(staged_iterator->payload, comparable))
                        ++rank_value;
                    ++staged_iterator;
                }
                else {
                    versioned_t const *store_visible = store_t::readable_version_(*committed_iterator);
                    if (store_visible &&
                        changes_.find(mapping_key_or_itself<value_t>(store_visible->payload)) == changes_.end() &&
                        less(store_visible->payload, comparable)) {
                        ++rank_value;
                    }

                    ++committed_iterator;
                }
            }

            callback_found(rank_value);
            return success_k;
        }

#pragma region Transaction Range Operations

        /** @brief How many members equal @p comparable, which for a unique key is nought or one. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept {
            expected<bool> const present = contains(std::forward<comparable_type_>(comparable));
            if (!present) return present.status();
            return *present ? std::size_t {1} : std::size_t {0};
        }

        /** @brief Copies out the first member at or after @p comparable, this transaction's writes included. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept
            requires ordered_collection<versioned_chains_t>
        {
            expected<value_t> result {status_t::key_not_found_k};
            [[maybe_unused]] status_t const answered = lower_bound(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            return result;
        }

        /** @brief Copies out the first member after @p comparable, this transaction's writes included. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept
            requires ordered_collection<versioned_chains_t>
        {
            expected<value_t> result {status_t::key_not_found_k};
            [[maybe_unused]] status_t const answered = upper_bound(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            return result;
        }

        /** @brief Stages a tombstone for every member this transaction reads in [ @p lower, @p upper ). */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires ordered_collection<versioned_chains_t>
        {
            auto const less = changes_.key_comp();
            return erase_walked_([&](auto const &candidate) noexcept { return less(candidate, upper); },
                                 std::forward<lower_type_>(lower), std::forward<callback_type_>(callback));
        }

        /** @brief Stages a tombstone for every member at or after @p lower, @p lower included. */
        template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback) noexcept
            requires ordered_collection<versioned_chains_t>
        {
            return erase_walked_([](auto const &) noexcept { return true; }, std::forward<lower_type_>(lower),
                                 std::forward<callback_type_>(callback));
        }

        /** @brief Stages a tombstone for every member before @p upper, @p upper excluded. */
        template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback) noexcept
            requires ordered_collection<versioned_chains_t>
        {
            // The walk needs a key to start from and there is no smallest key to name, so the lowest
            // member this transaction reads is asked for and used as the bound.
            expected<value_t> lowest = smallest_copy_();
            if (!lowest) return lowest.status() == key_not_found_k ? success_k : lowest.status();
            auto const less = changes_.key_comp();
            return erase_walked_([&](auto const &candidate) noexcept { return less(candidate, upper); },
                                 mapping_key_or_itself<value_t>(*lowest), std::forward<callback_type_>(callback));
        }

        /** @brief Hands @p callback each member in [ @p lower, @p upper ) to revise, and stages the result. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper,
                                            callback_type_ &&callback) noexcept
            requires is_mapping<value_t> && ordered_collection<versioned_chains_t>
        {
            values_vector_t revised(values_allocator_t(storage_shape_t::allocator_of(store_ref().entries_)));
            status_t collecting = success_k;
            [[maybe_unused]] status_t const walked = range(
                std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), [&](value_t const &value) noexcept {
                    if (failed(collecting)) return;
                    auto duplicate = copy_safely(value);
                    if (!duplicate) {
                        collecting = duplicate.status();
                        return;
                    }
                    if (status_t const kept = revised.push_back(std::move(*duplicate)); failed(kept)) collecting = kept;
                });
            if (failed(collecting)) return collecting;

            for (std::size_t index = 0; index != revised.size(); ++index) {
                value_t &revision = revised[index];
                callback(revision.key, revision.mapped);
                if (status_t const staged = upsert(std::move(revision)); failed(staged)) return staged;
            }
            return success_k;
        }

        /** @brief Draws one member uniformly from [ @p lower, @p upper ), this transaction's writes included. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                          callback_type_ &&callback) const noexcept
            requires ordered_collection<versioned_chains_t>
        {
            std::size_t counted = 0;
            if (status_t const measured = range(lower, upper, [&](value_t const &) noexcept { ++counted; });
                failed(measured))
                return measured;
            if (!counted) return success_k;

            std::size_t skipped = draw_below(generator, counted);
            bool drawn = false;
            return range(lower, upper, [&](value_t const &value) noexcept {
                if (drawn) return;
                if (skipped) --skipped;
                else {
                    callback(value);
                    drawn = true;
                }
            });
        }

        /** @brief Fills @p reservoir with up to @p capacity members drawn from [ @p lower, @p upper ). */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename output_iterator_type_ = no_op_t>
        [[nodiscard]] status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                                std::size_t &seen, std::size_t capacity,
                                                output_iterator_type_ &&reservoir) const noexcept
            requires ordered_collection<versioned_chains_t>
        {
            static_assert(std::is_nothrow_copy_assignable_v<value_t>,
                          "a reservoir copies into the caller's buffer, so the member must copy without throwing");
            return range(lower, upper, [&](value_t const &value) noexcept {
                if (seen < capacity) reservoir[seen] = value;
                else if (std::size_t const slot = draw_below(generator, seen + 1); slot < capacity)
                    reservoir[slot] = value;
                ++seen;
            });
        }

#pragma endregion Transaction Range Operations
        /**
         *  If the staging fails, the transaction contents remain unchanged. On success, all
         *  changes are merged into the main store but remain invisible until @c commit() is called.
         *  Otherwise, the user may call @c rollback() to pull back the staged changes into the
         *  transaction itself, or @c reset() to discard all changes and start fresh.
         *
         *  @return Success, @c operation_not_permitted_k when already staged, or an allocation failure.
         */
        [[nodiscard]] status_t stage() noexcept {
            if (staging_ == staging_t::staged_k) return operation_not_permitted_k;
            auto &store = store_ref();
            auto const prepaid = storage_shape_t::prepare(store.entries_, changed_identifiers_.size());
            if (failed(prepaid)) return prepaid;
            if (status_t const validated = validate_watches_(); failed(validated)) return validated;

            // Every change needs somewhere to land before any of them moves, or a transaction could
            // run out of memory with half of itself already in the store. So this pass reserves a
            // chain slot for each new key and one spare version node for each key that already has
            // one, and only then does the second pass move the versions across, where it cannot fail.
            std::size_t spare_versions_needed = 0;
            for (std::size_t reserved = 0; reserved != changed_identifiers_.size(); ++reserved) {
                identifier_t const &identifier = changed_identifiers_[reserved];
                if (store.entries_.find(identifier) != store.entries_.end()) {
                    ++spare_versions_needed;
                    continue;
                }
                auto reserved_identifier = copy_safely<identifier_t>(identifier);
                if (!reserved_identifier) {
                    unstage_(reserved);
                    return out_of_memory_heap_k;
                }
                versioned_t reservation(value_t {std::move(*reserved_identifier)});
                reservation.generation = generation_;
                reservation.presence = presence_t::present_k;
                auto result = storage_shape_t::upsert(store.entries_, versioned_chain_t {std::move(reservation)});
                if (failed(result)) {
                    unstage_(reserved);
                    return out_of_memory_heap_k;
                }
            }
            if (!store.reserve_spare_versions_(spare_versions_needed)) {
                unstage_(changed_identifiers_.size());
                return out_of_memory_heap_k;
            }

            // The visibility is updated later, in `commit`. A reserved slot already carries this
            // generation, so the version replaces it rather than lengthening the chain.
            for (auto staged = changes_.begin(); staged != changes_.end(); ++staged) {
                versioned_t &version = store_t::mutable_ref_(*staged);
                store.chain_attach_(mapping_key_or_itself<value_t>(version.payload), std::move(version));
            }

            changes_.clear();
            staging_ = staging_t::staged_k;
            return success_k;
        }

        [[nodiscard]] status_t reset() noexcept {
            auto &store = store_ref();
            if (staging_ == staging_t::staged_k) unstage_(changed_identifiers_.size());

            watches_.clear();
            changes_.clear();
            changed_identifiers_.clear();
            staging_ = staging_t::pending_k;
            generation_ = store.next_generation_();
            return success_k;
        }

        [[nodiscard]] status_t rollback() noexcept {
            if (staging_ != staging_t::staged_k) return operation_not_permitted_k;

            auto &store = store_ref();
            status_t result = success_k;

            // The recovered versions carry the generation they were staged under, and this
            // transaction is about to take a new one, so each is re-stamped on the way back. The
            // changed identifiers are deliberately kept: these keys are still going to be written,
            // and a later stage reserves a chain for each one before moving anything into it.
            generation_t const resumed = store.next_generation_();
            for (identifier_t const &identifier : changed_identifiers_) {
                versioned_t recovered;
                // ! Don't materialize a new copy of `identifier` here, use a reference
                // A version that is absent cannot be handed back, and the pending write it
                // carried is gone, so the rollback reports that rather than quietly dropping it.
                if (store.chain_discard_(identifier, generation_, &recovered) == detach_outcome_t::not_found_k) {
                    result = status_t::consistency_k;
                    continue;
                }
                recovered.generation = resumed;
                auto reinstated = storage_shape_t::upsert(changes_, std::move(recovered));
                if (failed(reinstated)) result = out_of_memory_heap_k;
            }

            staging_ = staging_t::pending_k;
            generation_ = resumed;
            return result;
        }

        /**
         *  @brief Whether this transaction may still publish what it staged, refusing before it writes anything.
         *
         *  A watch is asked again here as well as at staging: another transaction may have committed
         *  over a watched key while this one sat staged, and publishing on top of it would be the lost
         *  update the watch was taken to prevent. Nothing is written until this answers, so a refusal
         *  leaves the transaction staged and retryable - and a caller spreading one commit across
         *  several stores asks every one of them before any of them writes.
         *
         *  @return Success, @c consistency_k when a watched key moved under this transaction, or
         *    @c operation_not_permitted_k when nothing was staged.
         */
        [[nodiscard]] status_t validate_for_commit() const noexcept {
            if (staging_ != staging_t::staged_k) return operation_not_permitted_k;
            return validate_watches_();
        }

        /**
         *  @brief Makes every staged version visible, which cannot fail and cannot refuse.
         *
         *  The second half of a commit, split out because a caller spanning several stores has to know
         *  that once the first of them writes, none of the rest can turn back. It draws its own stamp
         *  rather than taking one, because these stores keep no shared clock - each orders its own
         *  versions, and a caller spanning several of them is buying atomicity of the decision, not one
         *  instant across all of them.
         *
         *  Unmasking cannot report a missing version here. Every path that removes an entry leaves a
         *  staged one alone: an erase retires only the published version, reclamation passes over any
         *  chain still carrying one, another transaction's unmasking walks only visible versions, and
         *  @c clear refuses outright while one is outstanding. So a version staged by this transaction
         *  is still where staging put it, and the outcome is an assertion rather than a status.
         *
         *  @warning Only ever called after @c validate_for_commit answered success, with nothing since.
         */
        void publish_under() noexcept {
            assert(staging_ == staging_t::staged_k && "publishing what was never staged");
            auto &store = store_ref();
            commit_stamp_t const stamp = store.next_commit_stamp_();
            for (auto const &identifier : changed_identifiers_) {
                [[maybe_unused]] unmask_outcome_t const outcome =
                    store.unmask_and_compact_(identifier, generation_, stamp);
                assert(outcome != unmask_outcome_t::version_missing_k &&
                       "a staged version vanished before it published");
            }

            changed_identifiers_.clear();
            staging_ = staging_t::pending_k;
        }

        /** @brief Validates and then publishes, which is the whole commit for a caller spanning one store. */
        [[nodiscard]] status_t commit() noexcept {
            if (status_t const permitted = validate_for_commit(); failed(permitted)) return permitted;
            publish_under();
            return success_k;
        }
    };

  private:
    /** @brief Version nodes reserved by a staging pass and not yet moved into a chain. */
    version_node_t *spare_versions_ {nullptr};
    versioned_chains_t entries_;
    alignas(atomic_alignment<generation_t>) generation_t generation_ {0};
    alignas(atomic_alignment<generation_t>) generation_t commits_ {0};
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
    generation_t next_generation_() noexcept { return atomic_add_fetch<generation_t>(generation_, 1); }

    /**
     *  @brief Hands out the stamp a write becomes visible under, which is the store's only ordering.
     *
     *  Drawn once per commit rather than once per write, so every key one transaction touched becomes
     *  visible under the same number and a reader can never see half of it. Counted apart from
     *  @c generation_, which dates a transaction's opening and says nothing about what is current.
     */
    commit_stamp_t next_commit_stamp_() noexcept {
        return static_cast<commit_stamp_t>(atomic_add_fetch<generation_t>(commits_, 1));
    }

    /** @brief A writable reference to a stored element, which every core hands out as immutable. */
    template <typename element_type_>
    static element_type_ &mutable_ref_(element_type_ const &element) noexcept {
        return const_cast<element_type_ &>(element);
    }

    /** @brief The one version a reader may see, or null while every version of the key is staged. */
    static versioned_t const *visible_version_(versioned_chain_t const &chain) noexcept {
        if (visible_now(chain.head.committed)) return &chain.head;
        for (version_node_t const *node = chain.others; node; node = node->next)
            if (visible_now(node->entry.committed)) return &node->entry;
        return nullptr;
    }

    /**
     *  @brief The version a reader may @b see the payload of, which excludes a committed tombstone.
     *
     *  A committed erase stays in the tree as a published tombstone, because a watch has to be able to
     *  observe it. Every surface that hands a payload out asks this instead of @c visible_version_, so
     *  the rule that a tombstone is not an element lives in one place.
     */
    static versioned_t const *readable_version_(versioned_chain_t const &chain) noexcept {
        versioned_t const *visible = visible_version_(chain);
        return visible && visible->presence == presence_t::present_k ? visible : nullptr;
    }

    /** @brief Whether nothing but a committed tombstone is left, so the whole entry can be freed. */
    static bool is_reclaimable_(versioned_chain_t const &chain) noexcept {
        return !chain.others && visible_now(chain.head.committed) && chain.head.presence == presence_t::erased_k;
    }

    /** @brief Whether @p chain carries a version an open transaction staged and has not yet published. */
    static bool holds_staged_version_(versioned_chain_t const &chain) noexcept {
        if (!visible_now(chain.head.committed)) return true;
        for (version_node_t const *node = chain.others; node; node = node->next)
            if (!visible_now(node->entry.committed)) return true;
        return false;
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
     *  @param[inout] version The version to store, left empty once it has been taken.
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
     *  @brief Looks @p identifier up and unlinks its version of @p generation, dropping the key when none remain.
     *  @param[in] identifier Identifier whose chain is shortened.
     *  @param[in] generation The generation to unlink.
     *  @param[out] destination Receives the unlinked version, or null to discard it.
     *  @return Whether anything was unlinked, and whether the key disappeared with it.
     */
    detach_outcome_t chain_discard_(identifier_t const &identifier, generation_t generation,
                                    versioned_t *destination) noexcept {
        auto found = entries_.find(identifier);
        if (found == entries_.end()) return detach_outcome_t::not_found_k;
        auto &chain = mutable_ref_(*found);

        // Handing the version back empties the inline slot, and the key the index orders this entry
        // by lives inside it. So the entry leaves the index first, while it still compares correctly,
        // and only then is the version harvested out of the detached node.
        if (destination && !chain.others && chain.head.generation == generation) {
            [[maybe_unused]] bool const taken = storage_shape_t::extract_payload(entries_, identifier, *destination);
            return detach_outcome_t::detached_and_emptied_k;
        }

        auto const outcome = chain_detach_(chain, generation, destination);
        if (outcome == detach_outcome_t::detached_and_emptied_k) entries_.erase(identifier);
        return outcome;
    }

    /**
     *  @brief Drops the published version of @p chain, leaving every staged one where it is.
     *
     *  A staged version belongs to a transaction that has not committed, so no direct write is
     *  entitled to decide its fate - only the commit that owns it is. Generations are unique within
     *  a chain, so the published one is named by its own stamp.
     *
     *  @param[inout] chain The key's version chain, shortened by its published version if it has one.
     *  @return Whether anything was dropped, and whether the entry must now leave the index.
     */
    detach_outcome_t retire_visible_(versioned_chain_t &chain) noexcept {
        versioned_t const *const visible = visible_version_(chain);
        if (!visible) return detach_outcome_t::not_found_k;
        --visible_count_;
        visible_deleted_count_ -= visible->presence == presence_t::erased_k;
        return chain_detach_(chain, visible->generation, nullptr);
    }

    /**
     *  @brief Frees the unreachable entries between @p cursor and @p stop, which an ordered core lets
     *    the walk erase in place.
     *  @return How many entries were reclaimed.
     */
    template <typename iterator_type_>
    std::size_t vacuum_between_(iterator_type_ cursor, iterator_type_ const stop) noexcept {
        std::size_t reclaimed = 0;
        while (cursor != stop) {
            if (!is_reclaimable_(*cursor)) {
                ++cursor;
                continue;
            }
            --visible_count_;
            --visible_deleted_count_;
            cursor = entries_.erase(cursor).next;
            ++reclaimed;
        }
        return reclaimed;
    }

    /**
     *  @brief Retires the published version of every entry between @p cursor and @p stop, handing each
     *    to @p callback before it goes.
     *
     *  Only published versions go, so the span cannot be split out whole: a chain that still holds a
     *  version staged by an open transaction keeps its entry, and the walk steps past it.
     */
    template <typename iterator_type_, typename callback_type_>
    void retire_between_(iterator_type_ cursor, iterator_type_ const stop, callback_type_ &&callback) noexcept {
        while (cursor != stop) {
            auto &chain = mutable_ref_(*cursor);
            if (versioned_t const *readable = readable_version_(chain)) callback(readable->payload);
            if (retire_visible_(chain) == detach_outcome_t::detached_and_emptied_k)
                cursor = entries_.erase(cursor).next;
            else ++cursor;
        }
    }

    /**
     *  @brief Files @p version into @p chain as its one published version, retiring the one it replaces.
     *    Overwriting the published slot in place costs nothing; a chain that holds only staged versions
     *    has to lengthen, which is the one way a direct write can run out of memory.
     *
     *  @param[inout] chain The key's version chain, which already exists in the index.
     *  @param[in] version The published version to store.
     *  @return Success, or @c out_of_memory_heap_k when the chain cannot lengthen.
     */
    [[nodiscard]] status_t publish_directly_(versioned_chain_t &chain, versioned_t &&version) noexcept {
        if (versioned_t const *const displaced = visible_version_(chain)) {
            visible_deleted_count_ -= displaced->presence == presence_t::erased_k;
            mutable_ref_(*displaced) = std::move(version);
            return success_k;
        }

        version_allocator_t allocator(storage_shape_t::allocator_of(entries_));
        version_node_t *const node = allocator.allocate(1);
        if (!node) return out_of_memory_heap_k;
        chain.others = new (node) version_node_t {std::move(version), chain.others};
        chain.allocator = allocator;
        ++visible_count_;
        return success_k;
    }

    /**
     *  @brief Internal API: Finds the latest visible entry and invokes callback with @c versioned_t const &.
     *    Used by internal methods that need access to the generation, presence and commit stamp.
     *    Only considers VISIBLE entries (committed/staged).
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c versioned_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
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
     *  @brief The version a watch is validated against: the one carrying the newest commit stamp.
     *
     *  A version nobody has committed carries no stamp, so it cannot answer here - which is what lets
     *  a transaction validate through another's staging window instead of being turned away by a
     *  write that may yet be rolled back. A committed tombstone is handed over as it stands, and
     *  @c watch_shape_of is what turns it into the shape a watch is compared against.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c versioned_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void find_committed_entry_(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                               callback_missing_type_ &&callback_missing = {}) const noexcept {

        auto found = entries_.find(std::forward<comparable_type_>(comparable));
        versioned_t const *committed = found != entries_.end() ? visible_version_(*found) : nullptr;
        if (committed) callback_found(*committed);
        else callback_missing();
    }

    /**
     *  @brief Publishes the version of @p identifier carrying @p generation_to_unmask, retiring the one it replaces.
     *  @param[in] stamp The stamp this commit publishes under, shared by every key it touched.
     *  @return Whether the version was found and published, which a caller reports as its own success.
     */
    unmask_outcome_t unmask_and_compact_(identifier_t const &identifier, generation_t generation_to_unmask,
                                         commit_stamp_t stamp) noexcept {
        // ! Don't materialize a new copy of `identifier` here, use a reference
        auto found = entries_.find(identifier);
        if (found == entries_.end()) return unmask_outcome_t::version_missing_k;
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
                entries_.erase(identifier);
                return unmask_outcome_t::version_missing_k;
            }
        }

        versioned_t *unmasked = nullptr;
        if (chain.head.generation == generation_to_unmask) { unmasked = &chain.head; }
        else
            for (version_node_t *node = chain.others; node && !unmasked; node = node->next)
                if (node->entry.generation == generation_to_unmask) unmasked = &node->entry;

        if (!unmasked) return unmask_outcome_t::version_missing_k;
        // Listing one key twice makes the second pass find a version this commit already published,
        // which is settled rather than missing.
        if (visible_now(unmasked->committed)) return unmask_outcome_t::already_published_k;
        unmasked->committed = stamp;
        ++visible_count_;
        if (unmasked->presence == presence_t::erased_k) visible_deleted_count_++;
        return unmask_outcome_t::unmasked_k;
    }

    /**
     *  @brief Stages every element of [ @p first, @p last ) through @p stage_one and commits them together.
     *    Bailing out before @c stage leaves the store untouched, which is what makes a bulk write
     *    all-or-nothing.
     *
     *  @param[in] first Beginning of the range to write.
     *  @param[in] last End of the range to write.
     *  @param[in] stage_one Invoked with the open transaction and one element, reporting what it staged.
     *  @return The first failure any step reported, or the status of the commit.
     */
    template <typename input_iterator_type_, typename stage_one_type_>
    [[nodiscard]] status_t commit_each_(input_iterator_type_ first, input_iterator_type_ last,
                                        stage_one_type_ &&stage_one) noexcept {
        if (first == last) return success_k;
        auto opened = transaction();
        if (!opened) return out_of_memory_heap_k;

        for (; first != last; ++first)
            if (status_t const staged = stage_one(*opened, value_t(*first)); failed(staged)) return staged;
        if (status_t const staged = opened->stage(); failed(staged)) return staged;
        return opened->commit();
    }

#pragma endregion Type Definitions

#pragma region Constructors and Assignment

  public:
    monotonic_store() noexcept {}

    /** @brief Seeds the underlying tree's allocator, which a stateful allocator needs. */
    explicit monotonic_store(allocator_t const &allocator) noexcept
        : entries_(storage_shape_t::template build<versioned_chain_t>(allocator)) {}

    /** @brief Seeds the comparator as well, which a comparator carrying state or a dispatch pointer needs. */
    monotonic_store(comparator_t const &comparator, allocator_t const &allocator = {}) noexcept
        : entries_(storage_shape_t::template build<versioned_chain_t>(comparator, allocator)) {}
    monotonic_store(monotonic_store &&other) noexcept
        : spare_versions_(std::exchange(other.spare_versions_, nullptr)), entries_(std::move(other.entries_)),
          generation_(other.generation_), commits_(other.commits_), visible_count_(other.visible_count_),
          visible_deleted_count_(other.visible_deleted_count_) {}

    monotonic_store &operator=(monotonic_store &&other) noexcept {
        if (this == &other) return *this;
        // The chains being dropped free their own version nodes, so the tree is emptied before the
        // spares are, and neither depends on the other surviving.
        entries_.clear();
        release_spare_versions_();
        spare_versions_ = std::exchange(other.spare_versions_, nullptr);
        entries_ = std::move(other.entries_);
        generation_ = other.generation_;
        commits_ = other.commits_;
        visible_count_ = other.visible_count_;
        visible_deleted_count_ = other.visible_deleted_count_;
        return *this;
    }

    ~monotonic_store() noexcept {
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
    [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept {
        expected<bool> const present = contains(std::forward<comparable_type_>(comparable));
        if (!present) return present.status();
        return *present ? std::size_t {1} : std::size_t {0};
    }

    /**
     *  @brief Checks if a member @b equal to the given @p comparable exists in the tree.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @return True if element exists, false otherwise.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
        bool present = false;
        status_t const answered =
            find(std::forward<comparable_type_>(comparable), [&](value_t const &) noexcept { present = true; });
        if (failed(answered)) return answered;
        return present;
    }

    /**
     *  @brief Finds and returns a copy of an element equal to @p comparable.
     *    Convenience method to avoid callback-based access in tests and simple use cases.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @return Result with copied element if found, or failure status.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        [[maybe_unused]] status_t const looked_up =
            find(std::forward<comparable_type_>(comparable),
                 [&](value_t const &found) noexcept { result = copy_safely(found); });
        return result;
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
        expected<value_t> result {status_t::key_not_found_k};
        [[maybe_unused]] status_t const bounded = lower_bound(
            std::forward<comparable_type_>(comparable), [&](value_t const &v) noexcept { result = copy_safely(v); },
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
        expected<value_t> result {status_t::key_not_found_k};
        [[maybe_unused]] status_t const bounded = upper_bound(
            std::forward<comparable_type_>(comparable), [&](value_t const &v) noexcept { result = copy_safely(v); },
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
    [[nodiscard]] static expected<store_t> make(allocator_t const &allocator = {}) noexcept {
        return store_t {allocator};
    }

    /**
     *  @brief Builds a container around a specific comparator, for comparators that carry state.
     *  @param[in] comparator The instance every comparison will consult.
     *  @param[in] allocator Optional allocator instance.
     *  @return Container instance or empty optional on failure.
     */
    [[nodiscard]] static expected<store_t> make(comparator_t const &comparator, allocator_t const &allocator) noexcept {
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
    [[nodiscard]] expected<transaction_t> transaction() noexcept { return transaction_t {*this}; }

#pragma endregion Transaction Management

#pragma region Modifiers

    /**
     *  @brief Atomically inserts an element only if the key doesn't exist. Fails if key exists.
     *    This is the strict insert semantics matching @c std::set::insert().
     *
     *  @param[in] value Element to insert (moved into the tree).
     *  @return Success, or @c key_already_exists_k if key exists, or OOM error.
     */
    [[nodiscard]] status_t insert(value_t &&value) noexcept {
        expected<bool> const key_is_present = contains(value);
        if (!key_is_present) return key_is_present.status();
        if (*key_is_present) return key_already_exists_k;
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
     *  @note Reporting the fresh insert costs a copy of the identifier and a second lookup, so a caller
     *    that leaves @p callback_inserted at its default pays for neither.
     */
    template <typename callback_inserted_type_ = no_op_t, typename callback_existing_type_ = no_op_t>
    [[nodiscard]] status_t insert_if_missing(value_t &&value, callback_inserted_type_ &&callback_inserted = {},
                                             callback_existing_type_ &&callback_existing = {}) noexcept {
        bool exists = false;
        [[maybe_unused]] status_t const looked_up = find(
            mapping_key_or_itself<value_t>(value),
            [&](value_t const &present) noexcept {
                exists = true;
                callback_existing(present);
            },
            no_op_t {});
        if (exists) return success_k;

        if constexpr (std::is_same_v<std::remove_cvref_t<callback_inserted_type_>, no_op_t>)
            return upsert(std::move(value));

        // The identifier has to be taken before the move, or the lookup below searches by a
        // moved-from key - which compares wrongly rather than failing loudly.
        auto maybe_identifier = copy_safely(mapping_key_or_itself<value_t>(value));
        if (!maybe_identifier) return maybe_identifier.status();

        auto status = upsert(std::move(value));
        if (failed(status)) return status;
        [[maybe_unused]] status_t const reported = find(*maybe_identifier, callback_inserted, no_op_t {});
        return status;
    }

    /**
     *  @brief Atomically upserts an element (insert or update). Always succeeds (unless OOM).
     *    Overwrites existing element if key exists (upsert semantics).
     *
     *  @param[in] element Element to upsert (moved into the tree).
     *  @return Success or error code (e.g., out of memory).
     *  @note Only the published version gives way. A version staged by an open transaction is that
     *    transaction's to publish or discard, and a direct write leaves it untouched.
     */
    [[nodiscard]] status_t upsert(value_t &&value) noexcept {
        versioned_t versioned(std::move(value));
        versioned.generation = next_generation_();
        versioned.presence = presence_t::present_k;
        versioned.committed = next_commit_stamp_();

        auto found = entries_.find(mapping_key_or_itself<value_t>(versioned.payload));
        if (found != entries_.end()) return publish_directly_(mutable_ref_(*found), std::move(versioned));

        auto result = storage_shape_t::upsert(entries_, versioned_chain_t {std::move(versioned)});
        if (failed(result)) return out_of_memory_heap_k;
        ++visible_count_;
        return success_k;
    }

    /**
     *  @brief Atomically updates an existing element. Fails if key doesn't exist.
     *    Unlike @c upsert(), this will NOT insert new keys.
     *
     *  @param[in] element Element to update (moved into the tree).
     *  @return Success, @c key_not_found_k if key doesn't exist, or OOM error.
     */
    [[nodiscard]] status_t update(value_t &&value) noexcept {
        expected<bool> const key_is_present = contains(value);
        if (!key_is_present) return key_is_present.status();
        if (!*key_is_present) return key_not_found_k;
        return upsert(std::move(value));
    }

    /**
     *  @brief Bulk insert from iterator range (atomic strict semantics).
     *    Fails if ANY key already exists. All elements inserted atomically - if any allocation fails
     *    or any key exists, no changes are made.
     *
     *  @param[in] first Beginning of range to insert.
     *  @param[in] last End of range to insert.
     *  @return Success, @c key_already_exists_k if any key exists, or out_of_memory_heap_k.
     *    Operation is atomic (all-or-nothing).
     */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t insert(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        // Duplicates inside the range collapse onto a single staged version, so the strictness is
        // about the store's own keys rather than about the range repeating itself.
        return commit_each_(first, last, [this](transaction_t &staging, value_t &&candidate) noexcept {
            expected<bool> const key_is_present = contains(mapping_key_or_itself<value_t>(candidate));
            if (!key_is_present) return key_is_present.status();
            if (*key_is_present) return key_already_exists_k;
            return staging.insert_if_missing(std::move(candidate));
        });
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
        return commit_each_(first, last, [](transaction_t &staging, value_t &&candidate) noexcept {
            return staging.insert_if_missing(std::move(candidate));
        });
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
        return commit_each_(first, last, [](transaction_t &staging, value_t &&candidate) noexcept {
            return staging.upsert(std::move(candidate));
        });
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
        return commit_each_(first, last, [this](transaction_t &staging, value_t &&candidate) noexcept {
            expected<bool> const key_is_present = contains(mapping_key_or_itself<value_t>(candidate));
            if (!key_is_present) return key_is_present.status();
            if (!*key_is_present) return key_not_found_k;
            return staging.upsert(std::move(candidate));
        });
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
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        auto found = entries_.find(std::forward<comparable_type_>(comparable));
        versioned_t const *readable = found != entries_.end() ? readable_version_(*found) : nullptr;
        if (readable) callback_found(readable->payload);
        else callback_missing();
        return success_k;
    }

    /**
     *  @brief Finds the first member @b greater or equal to the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    /**
     *  @brief Hands @p callback_found the smallest member any committed write left visible.
     *
     *  The unbounded case of @c lower_bound, and the one a merged walk over several stores opens with:
     *  it asks for a first key rather than an ordinal, so a core keeping no subtree counts can answer.
     *
     *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered when nothing is readable. Must be @c noexcept.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t smallest(callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept
        requires ordered_collection<versioned_chains_t>
    {
        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        for (auto cursor = entries_.begin(); cursor != entries_.end(); ++cursor)
            if (versioned_t const *readable = readable_version_(*cursor)) {
                callback_found(readable->payload);
                return success_k;
            }
        callback_missing();
        return success_k;
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
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
        while (cursor && !(visible = readable_version_(cursor->fruit)))
            cursor = chain_node_t::upper_bound(entries_.root(), cursor->fruit, entries_.key_comp());

        if (visible) callback_found(visible->payload);
        else callback_missing();
        return success_k;
    }

    /**
     *  @brief Finds all elements equal to a single key. Invokes callback for each matching element.
     *    For trees with unique keys, this returns at most one element (0 or 1).
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
        return find(std::forward<comparable_type_>(comparable), std::forward<callback_type_>(callback), no_op_t {});
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
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
        while (cursor && !(visible = readable_version_(cursor->fruit)))
            cursor = chain_node_t::upper_bound(entries_.root(), cursor->fruit, entries_.key_comp());

        if (visible) callback_found(visible->payload);
        else callback_missing();
        return success_k;
    }

#pragma endregion Lookup

#pragma region Enumeration

    /**
     *  @brief Hands @p callback every element the store shows, in whatever order the core holds them.
     *
     *  The one walk an unordered core can offer, so it promises no ordering even where the core has one.
     *  Every element a @c find would answer with at the moment of the call is visited @b exactly @b once -
     *  a committed tombstone and every version no commit has published yet are both left out. Nothing may
     *  write to the store while the walk runs.
     *
     *  @param[in] callback Callback invoked for each element. Must be @c noexcept.
     */
    template <typename callback_type_ = no_op_t>
    [[nodiscard]] status_t for_each(callback_type_ &&callback) const noexcept {
        static_assert(is_safe_callback_for<callback_type_, value_t const &>,
                      "callback must be noexcept invocable with value_t const &");

        for (auto cursor = entries_.begin(); cursor != entries_.end(); ++cursor)
            if (versioned_t const *readable = readable_version_(*cursor)) callback(readable->payload);
        return success_k;
    }

#pragma endregion Enumeration

#pragma region Range Operations

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper,
                                 callback_type_ &&callback = {}) const noexcept
        requires ordered_collection<versioned_chains_t>
    {
        chain_node_t::range(entries_.root(), std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                            entries_.key_comp(), [&](chain_node_t *node) noexcept {
                                if (versioned_t const *readable = readable_version_(node->fruit))
                                    callback(readable->payload);
                            });
        return success_k;
    }

    /**
     *  @brief Hands @p callback the mapped half of every visible element in [ @p lower, @p upper ).
     *
     *  @param[in] lower Lower bound, inclusive.
     *  @param[in] upper Upper bound, exclusive.
     *  @param[in] callback Callback invoked with (key const &, mapped &) per element. Must be @c noexcept.
     *  @return Always success here, since the revision is written in place; the status is reported so a
     *    sibling engine that has to copy the value first has the same channel.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires is_mapping<value_t> && ordered_collection<versioned_chains_t>
    {
        generation_t generation = next_generation_();
        chain_node_t::range(entries_.root(), std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                            entries_.key_comp(), [&](chain_node_t *node) noexcept {
                                auto &chain = node->fruit;
                                versioned_t const *readable = readable_version_(chain);
                                if (!readable) return;
                                versioned_t &mutable_version = mutable_ref_(*readable);
                                callback(mutable_version.payload.key, mutable_version.payload.mapped);
                                mutable_version.generation = generation;
                            });
        return success_k;
    }

    /**
     *  @brief Retires every published version ordered in [ @p lower, @p upper), handing each to @p callback.
     *  @return Always success; the status is reported so a wrapper that allocates can answer alike.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                       callback_type_ &&callback = {}) noexcept
        requires ordered_collection<versioned_chains_t>
    {
        retire_between_(entries_.lower_bound(std::forward<lower_type_>(lower)),
                        entries_.lower_bound(std::forward<upper_type_>(upper)), std::forward<callback_type_>(callback));
        return success_k;
    }

    /**
     *  @brief Retires every published version ordered at or after @p lower, with no upper bound at all.
     *
     *  @param[in] lower Lower bound of the range, inclusive - a key equal to it is erased, which is the
     *    same end @c erase_range() includes.
     *  @param[in] callback Callback handed each element about to be retired. Must be @c noexcept.
     *  @return Always success; the status is reported so a wrapper that allocates can answer alike.
     */
    template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback = {}) noexcept
        requires ordered_collection<versioned_chains_t>
    {
        retire_between_(entries_.lower_bound(std::forward<lower_type_>(lower)), entries_.end(),
                        std::forward<callback_type_>(callback));
        return success_k;
    }

    /**
     *  @brief Retires every published version ordered before @p upper, with no lower bound at all.
     *
     *  @param[in] upper Upper bound of the range, exclusive - a key equal to it is kept, which is the
     *    same end @c erase_range() excludes.
     *  @param[in] callback Callback handed each element about to be retired. Must be @c noexcept.
     *  @return Always success; the status is reported so a wrapper that allocates can answer alike.
     */
    template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires ordered_collection<versioned_chains_t>
    {
        retire_between_(entries_.begin(), entries_.lower_bound(std::forward<upper_type_>(upper)),
                        std::forward<callback_type_>(callback));
        return success_k;
    }

#pragma endregion Range Operations

#pragma region Vacuuming

    /**
     *  @brief Frees every entry no reader can reach - a committed tombstone with nothing staged behind it.
     *
     *  A committed erase leaves the key in the index so that a watch can still observe it, and nothing
     *  ever takes it back out. Reclaiming one changes nothing observable: @c size() already discounted
     *  it, no lookup ever saw it, and on the unordered core it stops holding a slot against the load
     *  factor.
     *
     *  @return How many entries were reclaimed, or @c out_of_memory_heap_k when the unordered core has
     *    no room for the list of keys the walk has to erase after it.
     */
    [[nodiscard]] expected<std::size_t> vacuum() noexcept {
        if constexpr (ordered_collection<versioned_chains_t>) {
            return vacuum_between_(entries_.begin(), entries_.end());
        }
        else {
            // An open-addressed table shifts its neighbours back when a slot is freed, so the walk only
            // names what it will reclaim and the erasures follow it.
            changed_identifiers_vector_t doomed(
                changed_identifiers_allocator_t(storage_shape_t::allocator_of(entries_)));
            for (auto cursor = entries_.begin(); cursor != entries_.end(); ++cursor) {
                if (!is_reclaimable_(*cursor)) continue;
                auto reclaimed_identifier =
                    copy_safely<identifier_t>(mapping_key_or_itself<value_t>((*cursor).head.payload));
                if (!reclaimed_identifier) return out_of_memory_heap_k;
                if (failed(doomed.push_back(std::move(*reclaimed_identifier)))) return out_of_memory_heap_k;
            }
            for (identifier_t const &identifier : doomed) {
                --visible_count_;
                --visible_deleted_count_;
                entries_.erase(identifier);
            }
            return doomed.size();
        }
    }

    /**
     *  @brief Frees the unreachable entries ordered in [ @p lower, @p upper), so a caller can step through
     *    the keyspace instead of paying for one pass over all of it.
     *  @return How many entries were reclaimed.
     */
    template <typename lower_type_, typename upper_type_>
    [[nodiscard]] expected<std::size_t> vacuum(lower_type_ &&lower, upper_type_ &&upper) noexcept
        requires ordered_collection<versioned_chains_t>
    {
        return vacuum_between_(entries_.lower_bound(std::forward<lower_type_>(lower)),
                               entries_.lower_bound(std::forward<upper_type_>(upper)));
    }

#pragma endregion Vacuuming

#pragma region Sampling

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                      callback_type_ &&callback) const noexcept
        requires ordered_collection<versioned_chains_t>
    {

        auto node = chain_node_t::sample_range( //
            entries_.root(), lower, upper, entries_.key_comp(), std::forward<generator_type_>(generator),
            [](chain_node_t *candidate) noexcept { return readable_version_(candidate->fruit) != nullptr; });
        // Callers see the stored value; the version metadata never leaves this class.
        if (node) callback(readable_version_(node->fruit)->payload);
        return success_k;
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    [[nodiscard]] status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                            std::size_t &seen, std::size_t reservoir_capacity,
                                            output_iterator_type_ &&reservoir) const noexcept
        requires ordered_collection<versioned_chains_t>
    {
        static_assert(std::is_nothrow_copy_assignable_v<value_t>,
                      "sampling copies each drawn element into the caller's buffer, so that copy must not throw");

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
        return range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), sampler);
    }

#pragma endregion Sampling

#pragma region Order Statistics

    /**
     *  @brief Finds the @p ordinal -th smallest visible (committed) element, counting from zero.
     *    Walks every entry in order, since a subtree weight counts versions rather than visible values.
     *    Instantiates only for a core carrying order statistics, which excludes the AVL aliases.
     *
     *  @param[in] ordinal Zero-based position among the visible elements.
     *  @param[in] callback_found Callback to receive the element. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered when fewer elements are visible. Must be @c noexcept.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                                  callback_missing_type_ &&callback_missing = {}) const noexcept
        requires supports_order_statistics<versioned_chains_t>
    {
        std::size_t visible_index = 0;
        bool found = false;

        chain_node_t::for_each_left_right(entries_.root(), [&](chain_node_t *node) noexcept {
            // The walk has no early exit, so the answer has to guard itself against every entry after it.
            if (found) return;
            versioned_t const *readable = readable_version_(node->fruit);
            if (!readable) return;
            if (visible_index == ordinal) {
                callback_found(readable->payload);
                found = true;
                return;
            }
            ++visible_index;
        });

        if (!found) callback_missing();
        return success_k;
    }

    /**
     *  @brief Finds the rank (position) of an element among visible (committed) elements.
     *    Walks every entry in order, since a subtree weight counts versions rather than visible values.
     *    Instantiates only for a core carrying order statistics, which excludes the AVL aliases.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive the rank (size_t). Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if element not found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept
        requires supports_order_statistics<versioned_chains_t>
    {
        std::size_t rank_value = 0;
        bool found = false;
        auto const less = entries_.key_comp();

        chain_node_t::for_each_left_right(entries_.root(), [&](chain_node_t *node) noexcept {
            versioned_t const *readable = readable_version_(node->fruit);
            if (!readable) return;

            if (less.same(readable->payload, comparable)) {
                callback_found(rank_value);
                found = true;
                return;
            }

            if (less(readable->payload, comparable)) ++rank_value;
        });

        if (!found) callback_missing();
        return success_k;
    }

#pragma endregion Order Statistics

#pragma region Modifiers and Diagnostics

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
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found = {},
                                 callback_missing_type_ &&callback_missing = {}) noexcept {
        // `find` already unwraps to the stored value, and `comparable` is read twice below,
        // so it stays an lvalue rather than being forwarded away on the first use.
        bool found = false;
        [[maybe_unused]] status_t const looked_up = find(
            comparable,
            [&](value_t const &value) noexcept {
                found = true;
                callback_found(value);
            },
            no_op_t {});

        if (!found) {
            callback_missing();
            return status_t::key_not_found_k;
        }

        // Only the published version goes; a version an open transaction staged is not this call's to
        // decide, so the entry survives whenever one is still hanging off it.
        auto doomed = entries_.find(comparable);
        if (doomed != entries_.end())
            if (retire_visible_(mutable_ref_(*doomed)) == detach_outcome_t::detached_and_emptied_k)
                entries_.erase(comparable);
        return success_k;
    }

    /**
     *  @brief Asks the core to make room for @p size more entries.
     *
     *  Honoured wherever the core has one allocation to grow - the open-addressed table reserves its
     *  slab here, and a later write lands in a slot already paid for. A node-based core allocates one
     *  node per write and has nothing to prepay, so there the hint is accepted and nothing happens.
     *  Either way a later write may still report @c out_of_memory_heap_k.
     *
     *  @param[in] size How many more entries to make room for.
     *  @return Success, or @c out_of_memory_heap_k when the core could not grow.
     */
    [[nodiscard]] status_t reserve(std::size_t size) noexcept { return storage_shape_t::prepare(entries_, size); }

    /**
     *  @brief Removes all elements from the tree, refusing while any transaction has something staged.
     *
     *  @return Success, or @c operation_not_permitted_k when a staged version would be dropped from
     *    under the transaction that is about to publish it.
     *  @note The generation counter keeps running. Rewinding it would hand a future transaction a stamp
     *    an open one already carries, and a watch compares stamps by value.
     */
    [[nodiscard]] status_t clear() noexcept {
        // Refused while anything is staged, because dropping a version an open transaction is about to
        // publish would make that publication fail - and the second half of a commit is the one step a
        // caller spanning several stores is promised cannot turn back.
        for (auto cursor = entries_.begin(); cursor != entries_.end(); ++cursor)
            if (holds_staged_version_(*cursor)) return operation_not_permitted_k;

        entries_.clear();
        visible_count_ = 0;
        visible_deleted_count_ = 0;
        return success_k;
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
            char const *marker = visible_now(version.committed) ? "✓" : "✗";
            stream << identifier_t {version.payload} << " @" << version.generation;
            stream << marker << " ";
        };
        chain_node_t::for_each_left_right(entries_.root(), [&](chain_node_t *node) {
            show(node->fruit.head);
            for (version_node_t const *other = node->fruit.others; other; other = other->next) show(other->entry);
        });
        stream << "\n";
    }
};

#pragma endregion Modifiers and Diagnostics

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
using monotonic_avl_set = monotonic_store<basic_avl_tree<value_type_, comparator_type_, allocator_type_>>;

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
using monotonic_avl_map =
    monotonic_store<basic_avl_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional set using weight-balanced tree with order statistics support.
 *    Stores unique elements in sorted order with ACID transaction semantics, and answers @c rank and
 *    @c select - by an ordered walk, not a descent: a subtree weight here counts versions rather than
 *    visible values, so the counts cannot be indexed into.
 *
 *  @tparam value_type_ Type of elements stored in the set.
 *  @tparam comparator_type_ Comparator for ordering elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<value_type_>>
using monotonic_wb_set = monotonic_store<basic_wb_tree<value_type_, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional map using weight-balanced tree with order statistics support.
 *    Stores key-value pairs in sorted order with ACID transaction semantics, and answers @c rank and
 *    @c select - by an ordered walk, not a descent, for the reason @c monotonic_wb_set names.
 *
 *  @tparam key_type_ Type of keys stored in the map.
 *  @tparam value_type_ Type of values stored in the map.
 *  @tparam comparator_type_ Comparator for ordering keys. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using monotonic_wb_map =
    monotonic_store<basic_wb_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>>;

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
using monotonic_hash_set = monotonic_store<basic_hash_table<key_type_, hasher_type_, equals_type_, allocator_type_>>;

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
using monotonic_hash_map =
    monotonic_store<basic_hash_table<mapping<key_type_, value_type_>, hasher_type_, equals_type_, allocator_type_>>;

} // namespace ashvardanian::smashtable
