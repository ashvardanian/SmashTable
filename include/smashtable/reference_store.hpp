/**
 *  @brief Transactional set container with ACID semantics, built on @c std::set for baseline reference. Provides
 *      2-phase commit transactions with optimistic concurrency control through watch/CAS operations. All operations
 *      use callback-based APIs and are exception-safe via @c noexcept wrappers.
 *  @author Ash Vardanian
 *  @file include/smashtable/reference_store.hpp
 *  @date October 12, 2022
 */
#pragma once
#include <cassert> // `assert`
#include <cstdint> // `std::uint8_t`

#include <memory>      // `std::allocator` as default
#include <set>         // `std::set` for inner versioned entries
#include <stdexcept>   // `std::length_error`, which MSVC does not reach through `<set>`
#include <type_traits> // `std::is_nothrow_invocable_v`
#include <vector>      // `std::vector` for watches

#include "shared.hpp"

namespace ashvardanian::smashtable {

template <typename callable_type_>
[[nodiscard]] status_t invoke_safely(callable_type_ &&callable) noexcept {
    if constexpr (noexcept(callable())) {
        callable();
        return success_k;
    }
    else {
        try {
            callable();
            return success_k;
        }
        catch (std::bad_alloc const &) {
            return status_t::out_of_memory_heap_k;
        }
        catch (std::length_error const &) {
            return status_t::out_of_memory_heap_k;
        }
        catch (...) {
            return status_t::unknown_k;
        }
    }
}

/**
 *  @brief  Transactional set providing ACID semantics with 2-phase commit and watch/CAS operations.
 *    Built on @c std::set as a baseline reference implementation. Not thread-safe by itself.
 *
 *  @section reference_store_design_goals Design Goals
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
 *  @see https://jepsen.io/consistency/models/monotonic-atomic-view
 *  @see https://jepsen.io/consistency/models/read-committed
 *
 *  @section reference_store_api_overview API Overview
 *
 *  - All lookups are heterogeneous: you can provide any type comparable to @p value_type_. Your comparator
 *    MUST define @code using is_transparent = void; @endcode to enable this, just like std::map and std::set.
 *  - No iterators are provided to keep the implementation simple and avoid complexity of maintaining persistent
 *    iterator validity across transactions and modifications.
 *  - All operations use callback-based APIs for consistency and exception safety via @c noexcept wrappers.
 *
 *  @subsection Insert Strategies
 *
 *  Three distinct insert strategies with different failure handling:
 *
 *  | Method               | Key Exists?     | Returns               | Use Case                              |
 *  |----------------------|-----------------|-----------------------|---------------------------------------|
 *  | insert()             | Fails (error)   | key_already_exists_k  | Strict: ensure key is new             |
 *  | insert_if_missing()  | Skips (success) | success_k             | Lenient: insert only if absent        |
 *  | insert_or_assign()   | Overwrites      | success_k             | Upsert: always update                 |
 *
 *  The @c upsert method is an alias for @c insert_or_assign, following a more DBMS-like naming convention.
 *  The @c try_emplace is deleted in favor of the more explicit @c insert_if_missing.
 *
 *  @section reference_store_requirements Requirements
 *
 *  @par Element Type
 *  - Nothrow default-constructible and nothrow move constructible/assignable (required)
 *  - For watch operations: Nothrow copy-constructible OR provides
 *      @code .copy() const -> expected<T> @endcode for safe copying of identifiers
 *
 *  @par Comparator Type
 *  - Must define @code bool operator()(value_type const &, value_type const &) const @endcode
 *  - For heterogeneous lookups: Define @code using is_transparent = void; @endcode and
 *      @code using value_type = ...; @endcode (optional but recommended)
 *
 *  @par Allocator Type
 *  - Must be "rebindable" for internal structures (@c std::set nodes, @c std::vector arrays)
 *  - Standard allocator propagation traits respected for move/copy operations
 *
 *  @tparam value_type_ Type of the elements stored in the set.
 *  @tparam comparator_type_ Ideally heterogeneous comparator for @c value_type_.
 *  @tparam allocator_type_ Arbitrary "rebindable" allocator for all internal structures.
 */
template < //
    typename value_type_, typename comparator_type_ = less_t, typename allocator_type_ = std::allocator<std::uint8_t>>
class reference_store {

#pragma region Type Definitions

  public:
    using value_t = value_type_;
    using value_type = value_t; // ? STL style
    using comparator_t = comparator_type_;
    using allocator_t = allocator_type_;

    using key_type = typename mapping_key_type_or_itself<value_t>::type;
    using mapped_type = typename mapped_value_type_or_void<value_t>::type;
    using is_associative = std::bool_constant<is_mapping<value_t>>;
    using is_transactional = std::true_type;

    /** @brief A commit publishes every staged version before any reader can run, and takes no snapshot. */
    static constexpr isolation_t isolation_k = isolation_t::monotonic_atomic_view_k;

    using versioning_t = versioning_for<value_t, comparator_t>;
    using identifier_t = typename versioning_t::identifier_t;
    using generation_t = typename versioning_t::generation_t;
    using dated_identifier_t = typename versioning_t::dated_identifier_t;
    using watch_t = typename versioning_t::watch_t;
    using watched_identifier_t = typename versioning_t::watched_identifier_t;
    using versioned_entry_t = typename versioning_t::versioned_entry_t;
    using versioned_comparator_t = typename versioning_t::versioned_comparator_t;

    /** @brief What this store calls itself, so generic code spells an engine and a wrapper alike. */
    using store_t = reference_store;

    /**
     *  @brief The one shape a watch records, whatever the read that produced it looked like.
     *
     *  A committed tombstone is handed out as found - the public @c find has to see it in order to
     *  hide it - so a watch on one has to record the same shape a watch on an absent key records, or
     *  it could never match itself. Validation re-derives the shape here too, so the two cannot drift.
     *
     *  @param[in] resolved The entry a read resolved to, or null when the key resolves to nothing.
     */
    [[nodiscard]] static watch_t watch_shape_of(versioned_entry_t const *resolved) noexcept {
        if (!resolved || resolved->presence != presence_t::present_k) return missing_watch();
        return watch_t {resolved->generation, resolved->presence};
    }

  private:
    using entry_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<versioned_entry_t>;
    using entry_set_t = std::set< //
        versioned_entry_t, versioned_comparator_t, entry_allocator_t>;
    using entry_iterator_t = typename entry_set_t::iterator;

    using watches_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<watched_identifier_t>;
    using watches_array_t = std::vector<watched_identifier_t, watches_allocator_t>;

    /** @brief Holds the members a revising walk collected before it stages any of them back. */
    using values_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<value_t>;
    using values_array_t = std::vector<value_t, values_allocator_t>;

    using changed_identifiers_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<identifier_t>;
    using changed_identifiers_vector_t = std::vector<identifier_t, changed_identifiers_allocator_t>;

  public:
    class transaction_t {

        friend store_t;
        store_t *store_ {nullptr};
        entry_set_t changes_ {};
        watches_array_t watches_ {};
        changed_identifiers_vector_t changed_identifiers_ {};
        generation_t generation_ {0};
        staging_t staging_ {staging_t::pending_k};

        // The local change set must order exactly as the store does, so it borrows the store's comparator
        // rather than default-constructing one - otherwise a stateful comparator would sort the two apart.
        transaction_t(store_t &set) noexcept(false)
            : store_(&set), changes_(set.entries_.key_comp()), generation_(set.next_generation_()) {
            // A stamp of `absent_generation_k` would make a staged entry indistinguishable from the
            // watch that stands for a key which is not there.
            assert(generation_ != absent_generation_k);
        }

        /**
         *  @brief Files @p element under this transaction's generation, replacing whatever it staged before.
         *
         *  The generation is part of the set's key, so it is stamped onto the entry @b before insertion
         *  and never touched afterwards. Every entry in one change set carries the same generation, so an
         *  existing entry for this key already carries the right one and only its payload is rewritten.
         *
         *  @param[in] callback_staged Invoked with the staged element once it is in the change set,
         *    so a caller can report what it wrote without copying it. Must be @c noexcept.
         */
        template <typename callback_staged_type_ = no_op_t>
        [[nodiscard]] status_t stage_(value_t &&element, presence_t presence, identifier_t &&identifier,
                                      callback_staged_type_ &&callback_staged = {}) noexcept {
            return invoke_safely([&]() {
                changed_identifiers_.reserve(changed_identifiers_.size() + 1);
                auto iterator = changes_.lower_bound(element);
                if (iterator != changes_.end() && changes_.key_comp().same(iterator->payload, element)) {
                    assert(iterator->generation == generation_);
                    const_cast<value_t &>(iterator->payload) = std::move(element);
                    const_cast<presence_t &>(iterator->presence) = presence;
                }
                else {
                    auto stamped = versioned_entry_t {std::move(element)};
                    stamped.generation = generation_;
                    stamped.presence = presence;
                    iterator = changes_.emplace_hint(iterator, std::move(stamped));
                }
                changed_identifiers_.push_back(std::move(identifier));
                callback_staged(iterator->payload);
            });
        }

        /**
         *  @brief Reports whether this transaction can see @p identifier, handing the element to @p callback_present.
         *  @note A key this transaction erased reads as absent, exactly as its own @c find sees it.
         */
        template <typename callback_present_type_>
        [[nodiscard]] bool transaction_sees_key_(identifier_t const &identifier,
                                                 callback_present_type_ &&callback_present) const noexcept {
            bool present = false;
            [[maybe_unused]] status_t const looked_up = find(
                identifier,
                [&](value_t const &element) noexcept {
                    present = true;
                    callback_present(element);
                },
                no_op_t {});
            return present;
        }

        /**
         *  @brief Reports the first element of the merged order, staged writes taking precedence.
         *
         *  @param[in] internal_iterator Where this transaction's own bound landed in the change set.
         *  @param[in] probe_store Invoked as @c probe_store(previous,on_found,on_missing) , with a null
         *    @p previous asking for the caller's own bound in the store and a non-null one asking for the
         *    first element strictly after it. Must be @c noexcept.
         *  @note A store element this transaction erased is not skipped by re-asking for the same bound -
         *    that returns the same element forever - but by asking for the one after it.
         */
        template <typename probe_type_, typename callback_found_type_, typename callback_missing_type_>
        void merge_first_(entry_iterator_t internal_iterator, probe_type_ &&probe_store,
                          callback_found_type_ &&callback_found,
                          callback_missing_type_ &&callback_missing) const noexcept {

            while (internal_iterator != changes_.end() && internal_iterator->presence == presence_t::erased_k)
                ++internal_iterator;

            // The element handed to the callback lives in the store's ordered core, which nothing here
            // mutates, so pointing at it across a restart is cheaper than copying an identifier out -
            // and works for identifiers that cannot be copied at all.
            value_t const *external_previous = nullptr;
            bool faced_deleted_entry = false;
            auto on_external_found = [&](value_t const &external_element) noexcept {
                if (internal_iterator == changes_.end()) return callback_found(external_element);

                value_t const &internal_element = internal_iterator->payload;
                if (!changes_.key_comp().less(external_element, internal_element))
                    return callback_found(internal_element);

                auto const staged = changes_.find(external_element);
                if (staged != changes_.end() && staged->presence == presence_t::erased_k) {
                    faced_deleted_entry = true;
                    external_previous = &external_element;
                }
                else callback_found(external_element);
            };
            auto on_external_missing = [&]() noexcept {
                if (internal_iterator == changes_.end()) callback_missing();
                else callback_found(internal_iterator->payload);
            };

            do {
                faced_deleted_entry = false;
                probe_store(external_previous, on_external_found, on_external_missing);
            } while (faced_deleted_entry);
        }

        store_t &store_ref() noexcept { return *store_; }
        store_t const &store_ref() const noexcept { return *store_; }

        /**
         *  @brief Whether every watched key still carries the version this transaction read.
         *    Asked twice - once when staging, once when publishing - because a commit landing in
         *    between is the only thing that can invalidate a read after it was validated.
         */
        [[nodiscard]] status_t validate_watches_() const noexcept {
            auto const &store = store_ref();
            for (watched_identifier_t const &watched : watches_) {
                watch_t latest = missing_watch();
                store.find_committed_entry_(watched.identifier, [&](versioned_entry_t const &entry) noexcept {
                    latest = watch_shape_of(&entry);
                });
                if (latest != watched.watch) return status_t::consistency_k;
            }
            return success_k;
        }

        /** @brief Erases every entry this transaction staged under its own generation. */
        void unstage_() noexcept {
            auto &store = store_ref();
            for (auto const &identifier : changed_identifiers_) {
                // Heterogeneous `erase` is only coming in C++23.
                dated_identifier<identifier_t const &> dated {identifier, generation_};
                if (auto iterator = store.entries_.find(dated); iterator != store.entries_.end())
                    store.entries_.erase(iterator);
            }
        }

        /**
         *  @brief Drops anything staged and never published, and gives up this transaction's claim.
         *
         *  Abandoning a staged transaction is the only way the store can accumulate entries that no
         *  generation can ever name again, so the unwind is not optional. A null @c store_ marks a
         *  moved-from transaction, which owns nothing and must undo nothing.
         */
        void unwind_() noexcept {
            if (!store_) return;
            if (staging_ == staging_t::staged_k) unstage_();
            staging_ = staging_t::pending_k;
            store_ = nullptr;
        }

        /**
         *  @brief Hands @p callback every element this transaction reads, in the container's own order.
         *
         *  Both sides are ordered sets, so they are merged as the walk goes rather than concatenated -
         *  which is what an ordinal needs and what the unordered @c for_each cannot promise. A key this
         *  transaction touched is answered from its own staged version, and the committed side skips
         *  whatever @c changes_ already speaks for, so no key is visited twice.
         *
         *  @param[in] callback Callback invoked for each element. Must be @c noexcept.
         */
        template <typename callback_type_>
        void for_each_ordered_(callback_type_ &&callback) const noexcept {
            auto const &store = store_ref();
            auto const ordering = changes_.key_comp();
            auto staged = changes_.begin();
            auto committed = store.entries_.begin();

            while (staged != changes_.end() || committed != store.entries_.end()) {
                if (committed == store.entries_.end() ||
                    (staged != changes_.end() && ordering.per_key_compare(*staged, *committed))) {
                    if (staged->presence == presence_t::present_k) callback(staged->payload);
                    ++staged;
                    continue;
                }
                if (visible_now(committed->committed) && committed->presence == presence_t::present_k &&
                    changes_.find(committed->payload) == changes_.end())
                    callback(committed->payload);
                ++committed;
            }
        }

      public:
        /**
         *  @brief Takes over @p other entirely, leaving it owning nothing.
         *
         *  Written out rather than defaulted: a defaulted move copies @c store_ into the new object
         *  and leaves the old one pointing at the same store, so the destructor below would erase the
         *  same staged entries twice.
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
        bool has_changes() const noexcept { return !changes_.empty(); }

        /**
         *  @brief Returns the number of pending changes in this transaction.
         *  @return Count of staged changes (upserts and erases).
         */
        std::size_t changes_count() const noexcept { return changes_.size(); }

        /**
         *  @brief Stages an insert, refusing the key that is already there.
         *
         *  @param[in] element Element to insert (moved into the transaction).
         *  @param[in] callback_inserted Invoked with the staged element. Must be @c noexcept.
         *  @param[in] callback_existing Invoked with the element already under the key, which keeps its value.
         *    Must be @c noexcept.
         *  @return Success, @c key_already_exists_k when the key is taken, or an allocation failure.
         */
        template <typename callback_inserted_type_ = no_op_t, typename callback_existing_type_ = no_op_t>
        [[nodiscard]] status_t insert(value_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                      callback_existing_type_ &&callback_existing = {}) noexcept {

            static_assert(is_safe_callback_for<callback_existing_type_, value_t const &>,
                          "callback_existing must be noexcept invocable with value_t const &");
            static_assert(is_safe_callback_for<callback_inserted_type_, value_t const &>,
                          "callback_inserted must be noexcept invocable with value_t const &");

            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself(element));
            if (!maybe_identifier) return maybe_identifier.status();
            if (transaction_sees_key_(*maybe_identifier, callback_existing)) return key_already_exists_k;
            return stage_(std::move(element), presence_t::present_k, std::move(*maybe_identifier), callback_inserted);
        }

        /**
         *  @brief Stages an insert, leaving the key that is already there untouched.
         *
         *  @param[in] element Element to insert (moved into the transaction).
         *  @param[in] callback_inserted Invoked with the staged element. Must be @c noexcept.
         *  @param[in] callback_existing Invoked with the element already under the key, which keeps its value.
         *    Must be @c noexcept.
         *  @return Success even when nothing was staged, or an allocation failure.
         */
        template <typename callback_inserted_type_ = no_op_t, typename callback_existing_type_ = no_op_t>
        [[nodiscard]] status_t insert_if_missing(value_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                                 callback_existing_type_ &&callback_existing = {}) noexcept {

            static_assert(is_safe_callback_for<callback_existing_type_, value_t const &>,
                          "callback_existing must be noexcept invocable with value_t const &");
            static_assert(is_safe_callback_for<callback_inserted_type_, value_t const &>,
                          "callback_inserted must be noexcept invocable with value_t const &");

            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself(element));
            if (!maybe_identifier) return maybe_identifier.status();
            if (transaction_sees_key_(*maybe_identifier, callback_existing)) return success_k;
            return stage_(std::move(element), presence_t::present_k, std::move(*maybe_identifier), callback_inserted);
        }

        /**
         *  @brief Stages a write that takes the key whether or not it was there.
         *
         *  @param[in] element Element to insert or assign (moved into the transaction).
         *  @param[in] callback_inserted Invoked with the staged element when the key was free. Must be @c noexcept.
         *  @param[in] callback_existing Invoked with the element about to be replaced. Must be @c noexcept.
         *  @return Success or an allocation failure.
         */
        template <typename callback_inserted_type_ = no_op_t, typename callback_existing_type_ = no_op_t>
        [[nodiscard]] status_t insert_or_assign_(value_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                                 callback_existing_type_ &&callback_existing = {}) noexcept {

            static_assert(is_safe_callback_for<callback_existing_type_, value_t const &>,
                          "callback_existing must be noexcept invocable with value_t const &");
            static_assert(is_safe_callback_for<callback_inserted_type_, value_t const &>,
                          "callback_inserted must be noexcept invocable with value_t const &");

            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself(element));
            if (!maybe_identifier) return maybe_identifier.status();
            // The element being replaced is reported before the write, which is the last moment it
            // is still there to report - staging overwrites this transaction's own version in place.
            bool const replacing = transaction_sees_key_(*maybe_identifier, callback_existing);
            return stage_(std::move(element), presence_t::present_k, std::move(*maybe_identifier),
                          [&](value_t const &staged) noexcept {
                              if (!replacing) callback_inserted(staged);
                          });
        }

        /**
         *  @brief Alias for @c insert_or_assign(). Stages an insert or assign operation.
         *  @param[in] element Element to insert or assign (moved into the transaction).
         *  @return Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t upsert(value_t &&element) noexcept { return insert_or_assign_(std::move(element)); }

        /**
         *  @brief Stages a write that refuses a key this transaction cannot already see.
         *    The strict counterpart to @c upsert(), which takes the key either way.
         *
         *  @param[in] element Element to write (moved into the transaction when the key is there).
         *  @return Success, @c key_not_found_k when the key is absent, or an allocation failure.
         */
        [[nodiscard]] status_t update(value_t &&element) noexcept {
            expected<bool> const key_is_present = contains(mapping_key_or_itself(element));
            if (!key_is_present) return key_is_present.status();
            if (!*key_is_present) return key_not_found_k;
            return insert_or_assign_(std::move(element));
        }

        /**
         *  @brief Stages a delete operation for the element with the given identifier.
         *    The deletion is not visible until after @c stage() and @c commit().
         *
         *  @param[in] identifier Identifier of the element to erase.
         *  @return Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t erase(identifier_t const &identifier) noexcept {
            // The tombstone and the change list each keep their own identifier, and a heavy key
            // only copies through `.copy()`.
            auto maybe_identifier = copy_safely<identifier_t>(identifier);
            if (!maybe_identifier) return maybe_identifier.status();
            auto maybe_tombstone = copy_safely<identifier_t>(identifier);
            if (!maybe_tombstone) return maybe_tombstone.status();
            return stage_(value_t {std::move(*maybe_tombstone)}, presence_t::erased_k, std::move(*maybe_identifier));
        }

        /**
         *  @brief Pre-allocates memory for watch operations to reduce allocation failures during transaction.
         *
         *  @param[in] size Expected number of watches.
         *  @return Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t reserve(std::size_t size) noexcept {
            return invoke_safely([&]() { watches_.reserve(size); });
        }

        /**
         *  @brief Registers a watch on the element with the given identifier for optimistic concurrency control.
         *    The transaction will fail at @c stage() if the watched element changes.
         *
         *  @param[in] identifier Identifier of the element to watch.
         *  @return Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t watch(identifier_t const &identifier) noexcept {
            status_t status = success_k;
            // A watch outlives the entry it refers to, so the identifier must be owned, not referenced.
            auto remember = [&](identifier_t &&owned, watch_t watch) noexcept {
                status = invoke_safely([&]() { watches_.push_back({std::move(owned), watch}); });
            };
            store_ref().find_visible_entry_(
                identifier,
                [&](versioned_entry_t const &entry) noexcept {
                    auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself(entry.payload));
                    if (!maybe_identifier) status = maybe_identifier.status();
                    else remember(std::move(*maybe_identifier), watch_shape_of(&entry));
                },
                [&]() noexcept {
                    auto maybe_identifier = copy_safely<identifier_t>(identifier);
                    if (!maybe_identifier) status = maybe_identifier.status();
                    else remember(std::move(*maybe_identifier), watch_shape_of(nullptr));
                });
            return status;
        }

        /**
         *  @brief Registers a watch on an already-fetched entry for optimistic concurrency control.
         *    The transaction will fail at @c stage() if the watched element changes.
         *
         *  @param[in] entry Entry to watch (typically from a previous find/lookup).
         *  @return Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t watch(versioned_entry_t const &entry) noexcept {
            auto maybe_identifier = copy_safely<identifier_t>(mapping_key_or_itself(entry.payload));
            if (!maybe_identifier) return maybe_identifier.status();
            return invoke_safely([&] { watches_.push_back({std::move(*maybe_identifier), watch_shape_of(&entry)}); });
        }

        /**
         *  @brief Finds a member @b equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c reference_store::find(), will include entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c value_t const &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept {

            static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                          "callback_found must be noexcept invocable with value_t const &");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            if (auto iterator = changes_.find(std::forward<comparable_type_>(comparable)); iterator != changes_.end()) {
                if (iterator->presence == presence_t::present_k) callback_found(iterator->payload);
                else callback_missing();
            }
            else
                return store_ref().find(std::forward<comparable_type_>(comparable),
                                        std::forward<callback_found_type_>(callback_found),
                                        std::forward<callback_missing_type_>(callback_missing));
            return success_k;
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

        /**
         *  @brief Copies out the member equal to @p comparable, including this transaction's own writes.
         *
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @return The copied element, or @c key_not_found_k when absent.
         */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            [[maybe_unused]] status_t const looked_up = find(
                std::forward<comparable_type_>(comparable),
                [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
            return result;
        }

        /**
         *  @brief Checks if a member @b equal to the given @p comparable exists, including transaction changes.
         *    Convenience wrapper around @c find() for existence checks.
         *
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @return True if the element exists, false otherwise.
         */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
            bool present = false;
            status_t const answered = find(
                std::forward<comparable_type_>(comparable), [&](value_t const &) noexcept { present = true; },
                no_op_t {});
            if (failed(answered)) return answered;
            return present;
        }

        /**
         *  @brief Finds every member equal to @p comparable, including this transaction's own writes.
         *    For a unique-key container that is at most one element.
         *
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
            return find(std::forward<comparable_type_>(comparable), std::forward<callback_type_>(callback), no_op_t {});
        }

        /**
         *  @brief Finds the first member @b greater or equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c reference_store::lower_bound(), includes entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c value_t const &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept {

            static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                          "callback_found must be noexcept invocable with value_t const &");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            auto &store = store_ref();
            merge_first_(
                changes_.lower_bound(comparable),
                [&](value_t const *previous, auto &&on_found, auto &&on_missing) noexcept {
                    [[maybe_unused]] status_t const answered =
                        previous ? store.upper_bound(mapping_key_or_itself(*previous), on_found, on_missing)
                                 : store.lower_bound(comparable, on_found, on_missing);
                },
                std::forward<callback_found_type_>(callback_found),
                std::forward<callback_missing_type_>(callback_missing));
            return success_k;
        }

        /**
         *  @brief Finds the first member @b greater than the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c reference_store::upper_bound(), includes entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c value_t const &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept {

            static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                          "callback_found must be noexcept invocable with value_t const &");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            auto &store = store_ref();
            merge_first_(
                changes_.upper_bound(comparable),
                [&](value_t const *previous, auto &&on_found, auto &&on_missing) noexcept {
                    [[maybe_unused]] status_t const answered =
                        previous ? store.upper_bound(mapping_key_or_itself(*previous), on_found, on_missing)
                                 : store.upper_bound(comparable, on_found, on_missing);
                },
                std::forward<callback_found_type_>(callback_found),
                std::forward<callback_missing_type_>(callback_missing));
            return success_k;
        }

        /**
         *  @brief Iterates over all entries in the range [ @p lower, @p upper), including transaction changes.
         *    Degrades to @c equal_range() if @p lower and @p upper are the same.
         *
         *  @param[in] lower Lower bound of the range (inclusive).
         *  @param[in] upper Upper bound of the range (exclusive).
         *  @param[in] callback Callback invoked for each element in range. Must be @c noexcept.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper,
                                     callback_type_ &&callback = {}) const noexcept {
            // First, iterate over local changes
            auto lower_internal = changes_.lower_bound(std::forward<lower_type_>(lower));
            auto upper_internal = changes_.lower_bound(std::forward<upper_type_>(upper));
            for (auto it = lower_internal; it != upper_internal; ++it)
                if (it->presence == presence_t::present_k) callback(it->payload);

            // Then, iterate over external store, skipping entries that were modified or deleted locally
            return store_ref().range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                     [&](value_t const &external_element) noexcept {
                                         // Check if this entry exists in local changes
                                         auto local_state = changes_.find(external_element);
                                         // Not modified locally, include it
                                         if (local_state == changes_.end()) callback(external_element);
                                         // If modified locally, we already processed it above
                                     });
        }

        /**
         *  @brief Finds the @p ordinal -th smallest element this transaction reads, counting from zero.
         *    Merges the staged and committed sides as it walks, which is linear and meant to be: this
         *    is the oracle the ranked engines are checked against.
         *
         *  @param[in] ordinal Zero-based position among the elements this transaction can see.
         *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered when fewer elements are visible. Must be @c noexcept.
         */
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                                      callback_missing_type_ &&callback_missing = {}) const noexcept {

            static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                          "callback_found must be noexcept invocable with value_t const &");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            std::size_t visible_index = 0;
            bool found = false;
            for_each_ordered_([&](value_t const &element) noexcept {
                // The merge has no early exit, so the answer guards itself against every element after it.
                if (found) return;
                if (visible_index == ordinal) {
                    callback_found(element);
                    found = true;
                    return;
                }
                ++visible_index;
            });
            if (!found) callback_missing();
            return success_k;
        }

        /**
         *  @brief Hands @p callback_found how many elements this transaction orders before @p comparable.
         *    A linear merge, for the reason @c select() takes one.
         *
         *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive a @c std::size_t. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered when @p comparable is not visible. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept {

            static_assert(is_safe_callback_for<callback_found_type_, std::size_t>,
                          "callback_found must be noexcept invocable with std::size_t");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            auto const ordering = changes_.key_comp();
            identifier_t const target(comparable);
            std::size_t preceding = 0;
            bool found = false;
            for_each_ordered_([&](value_t const &element) noexcept {
                if (ordering.same(element, target)) found = true;
                else if (!found && ordering.per_key_compare(element, target)) ++preceding;
            });
            if (found) callback_found(preceding);
            else callback_missing();
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

            for (auto iterator = changes_.begin(); iterator != changes_.end(); ++iterator)
                if (iterator->presence == presence_t::present_k) callback(iterator->payload);

            // A key this transaction touched is answered from its own version above, so the committed
            // side skips whatever `changes_` already speaks for and never emits a key twice.
            return store_ref().for_each([&](value_t const &external_element) noexcept {
                if (changes_.find(external_element) == changes_.end()) callback(external_element);
            });
        }

#pragma region Transaction Range Operations

        /**
         *  @brief Walks every member this transaction reads, hands the accepted ones to @p callback, and
         *    stages a tombstone for each.
         *
         *  Linear over the whole transaction rather than seeking to a bound, which is what this store is
         *  for - it is the oracle the other engines are checked against, so it is written to be obviously
         *  right rather than fast.
         */
        template <typename accepts_type_, typename callback_type_>
        [[nodiscard]] status_t erase_accepted_(accepts_type_ &&accepts, callback_type_ &&callback) noexcept {
            std::vector<identifier_t, changed_identifiers_allocator_t> doomed;
            status_t collecting = success_k;
            if (status_t const visited = for_each([&](value_t const &member) noexcept {
                    if (failed(collecting) || !accepts(member)) return;
                    callback(member);
                    collecting = invoke_safely([&]() { doomed.push_back(mapping_key_or_itself<value_t>(member)); });
                });
                failed(visited))
                return visited;
            if (failed(collecting)) return collecting;

            for (identifier_t const &identifier : doomed)
                if (status_t const staged = erase(identifier); failed(staged)) return staged;
            return success_k;
        }

        /** @brief How many members equal @p comparable, which for a unique key is nought or one. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept {
            expected<bool> const present = contains(std::forward<comparable_type_>(comparable));
            if (!present) return present.status();
            return *present ? std::size_t {1} : std::size_t {0};
        }

        /** @brief Copies out the first member at or after @p comparable, this transaction's writes included. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            if (status_t const bounded = lower_bound(
                    std::forward<comparable_type_>(comparable),
                    [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
                failed(bounded))
                return bounded;
            return result;
        }

        /** @brief Copies out the first member after @p comparable, this transaction's writes included. */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept {
            expected<value_t> result {status_t::key_not_found_k};
            if (status_t const bounded = upper_bound(
                    std::forward<comparable_type_>(comparable),
                    [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
                failed(bounded))
                return bounded;
            return result;
        }

        /**
         *  @brief Stages a tombstone for every member this transaction reads in [ @p lower, @p upper ).
         *
         *  The window is walked first and staged afterwards, because a staged tombstone changes what the
         *  merged walk answers and a walk revising itself would step over its own neighbours.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                           callback_type_ &&callback) noexcept {
            return erase_accepted_(
                [&](value_t const &member) noexcept {
                    auto const &ordering = changes_.key_comp();
                    return !ordering(member, lower) && ordering(member, upper);
                },
                std::forward<callback_type_>(callback));
        }

        /** @brief Stages a tombstone for every member at or after @p lower, @p lower included. */
        template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback) noexcept {
            return erase_accepted_([&](value_t const &member) noexcept { return !changes_.key_comp()(member, lower); },
                                   std::forward<callback_type_>(callback));
        }

        /** @brief Stages a tombstone for every member before @p upper, @p upper excluded. */
        template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback) noexcept {
            return erase_accepted_([&](value_t const &member) noexcept { return changes_.key_comp()(member, upper); },
                                   std::forward<callback_type_>(callback));
        }

        /** @brief Hands @p callback each member in [ @p lower, @p upper ) to revise, and stages the result. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper,
                                            callback_type_ &&callback) noexcept
            requires is_mapping<value_t>
        {
            values_array_t revised;
            status_t collecting = success_k;
            if (status_t const walked = range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                              [&](value_t const &value) noexcept {
                                                  if (failed(collecting)) return;
                                                  collecting = invoke_safely([&]() { revised.push_back(value); });
                                              });
                failed(walked))
                return walked;
            if (failed(collecting)) return collecting;

            for (value_t &revision : revised) {
                callback(revision.key, revision.mapped);
                if (status_t const staged = upsert(value_t {revision}); failed(staged)) return staged;
            }
            return success_k;
        }

        /** @brief Draws one member uniformly from [ @p lower, @p upper ), this transaction's writes included. */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename generator_type_ = no_op_t, typename callback_type_ = no_op_t>
        [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                          callback_type_ &&callback) const noexcept {
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
                                                output_iterator_type_ &&reservoir) const noexcept {
            return range(lower, upper, [&](value_t const &value) noexcept {
                if (seen < capacity) reservoir[seen] = value;
                else if (std::size_t const slot = draw_below(generator, seen + 1); slot < capacity)
                    reservoir[slot] = value;
                ++seen;
            });
        }

#pragma endregion Transaction Range Operations

        /**
         *  @brief Validates watches and stages all changes to the main store, making them visible but uncommitted.
         *    Fails with @c status_t::consistency_k if any watched elements changed.
         *
         *  @return Success, @c operation_not_permitted_k if the transaction is already staged, or a
         *    consistency error if a watch fails to match.
         */
        [[nodiscard]] status_t stage() noexcept {
            if (staging_ == staging_t::staged_k) return operation_not_permitted_k;

            // First, check if we have any collisions by validating watches.
            auto &store = store_ref();
            if (status_t const validated = validate_watches_(); failed(validated)) return validated;

            // Merge our current nodes into the store.
            // The visibility will be updated later in the `commit`.
            store.entries_.merge(changes_);
            staging_ = staging_t::staged_k;
            return success_k;
        }

        /**
         *  @brief Resets the transaction to a clean state, discarding all changes and watches.
         *    A staged transaction has its staged changes taken back out of the main store.
         *    A new generation is assigned for reuse of this transaction.
         *
         *  @return Always succeeds.
         */
        [[nodiscard]] status_t reset() noexcept {
            // If the transaction was "staged",
            // we must delete all the entries.
            auto &store = store_ref();
            if (staging_ == staging_t::staged_k) unstage_();

            watches_.clear();
            changes_.clear();
            changed_identifiers_.clear();
            staging_ = staging_t::pending_k;
            generation_ = store.next_generation_();
            return success_k;
        }

        /**
         *  @brief Rolls back a previously staged transaction, moving changes from the store back to the transaction.
         *    Can only be called on staged transactions. Watches are preserved (read concerns persist across rollback),
         *    allowing retry patterns. New generation is assigned.
         *
         *  @return Success, @c operation_not_permitted_k if transaction is not staged, or
         *    @c consistency_k if a staged version was destroyed before the rollback reached it.
         */
        [[nodiscard]] status_t rollback() noexcept {
            if (staging_ != staging_t::staged_k) return operation_not_permitted_k;

            // Transaction was staged, we must extract all the entries back
            auto &store = store_ref();

            // Entries are keyed by identifier and generation, and this transaction is about to take a
            // new generation, so each is re-stamped while it is out of any container. The changed
            // identifiers are deliberately kept: these keys are still going to be written, and commit
            // walks that list to find what to publish.
            generation_t const resumed = store.next_generation_();
            status_t result = success_k;
            for (auto const &identifier : changed_identifiers_) {
                dated_identifier<identifier_t const &> dated {identifier, generation_};
                auto source = store.entries_.find(dated);
                // A version that is absent cannot be handed back, and the pending write it
                // carried is gone, so the rollback reports that rather than quietly dropping it.
                if (source == store.entries_.end()) {
                    // Touching one key twice lists it twice, and the second pass finds the version
                    // already pulled back into the change set rather than a loss.
                    if (changes_.find(dated_identifier<identifier_t const &> {identifier, resumed}) == changes_.end())
                        result = status_t::consistency_k;
                    continue;
                }
                auto detached = store.entries_.extract(source);
                detached.value().generation = resumed;
                changes_.insert(std::move(detached));
            }

            // Preserve watches_ for future stage operations (read watches persist across rollback)
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
         *  entry still carrying one, another transaction's unmasking walks only visible versions, and
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
                auto range = store.entries_.equal_range(identifier);
                [[maybe_unused]] auto const outcome =
                    store.unmask_and_compact_(range.first, range.second, generation_, stamp);
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
    entry_set_t entries_ {};
    alignas(atomic_alignment<generation_t>) generation_t generation_ {0};
    alignas(atomic_alignment<generation_t>) generation_t commits_ {0};
    std::size_t visible_count_ {0};
    std::size_t visible_deleted_count_ {0};

    friend class transaction_t;

    /**
     *  @brief Hands out the next version stamp, which must be unique across concurrent openers.
     *
     *  Relaxed suffices because one location has a total modification order, which is all uniqueness
     *  needs; a stamp is compared by value - @c > when walking a chain, @c == when validating a watch -
     *  and never used to order memory. Publishing what a generation tags is the partition lock's job.
     */
    generation_t next_generation_() noexcept {
        generation_t const stamp = atomic_add_fetch<generation_t>(generation_, 1);
        // Every entry is stamped from here, so this is the one place that can promise no stored
        // entry carries `absent_generation_k` - the value a watch on a missing key is anchored to.
        assert(stamp != absent_generation_k);
        return stamp;
    }

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

    /**
     *  @brief Internal API: Finds the latest visible entry and invokes callback with @c versioned_entry_t const &.
     *    Used by internal methods that need access to the generation, presence and commit stamp.
     *    Only considers VISIBLE entries (committed/staged).
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c versioned_entry_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void find_visible_entry_(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                             callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, versioned_entry_t const &>,
                      "callback_found must be noexcept invocable with versioned_entry_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));

        // Skip all the invisible entries
        while (range.first != range.second && !visible_now(range.first->committed)) ++range.first;

        if (range.first != range.second && range.first->presence == presence_t::present_k) callback_found(*range.first);
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
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c versioned_entry_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void find_committed_entry_(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                               callback_missing_type_ &&callback_missing = {}) const noexcept {
        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));

        entry_iterator_t committed = range.second;
        for (auto it = range.first; it != range.second; ++it) {
            if (!visible_now(it->committed)) continue;
            if (committed == range.second || it->committed > committed->committed) committed = it;
        }

        // Invoke whichever callback matches the outcome.
        if (committed == range.second) callback_missing();
        else callback_found(*committed);
    }

    /**
     *  @brief Drops every published entry in [ @p begin, @p end), leaving other transactions' staged ones.
     *  @note Only entries a reader could have seen reach @p callback: a published tombstone is erased
     *    silently, since reporting it would hand the caller a key it just told them was gone.
     */
    template <typename callback_type_ = no_op_t>
    void erase_visible_(entry_iterator_t begin, entry_iterator_t end, callback_type_ &&callback = {}) noexcept {
        entry_iterator_t current = begin;
        while (current != end)
            if (visible_now(current->committed)) {
                if (current->presence == presence_t::present_k) callback(current->payload);
                --visible_count_;
                visible_deleted_count_ -= current->presence == presence_t::erased_k;
                current = entries_.erase(current);
            }
            else ++current;
    }

    /**
     *  @brief Erases the committed tombstones in [ @p begin, @p end), which nothing else reclaims.
     *
     *  A published tombstone answers every read as absence, yet a single-key @c erase declines to
     *  touch it and only a later write to the same key overwrites it. Staged entries belong to
     *  transactions still running and are left alone. The ordered core erases in place, so no
     *  intermediate buffer of doomed keys is needed.
     */
    [[nodiscard]] std::size_t vacuum_between_(entry_iterator_t begin, entry_iterator_t end) noexcept {
        std::size_t reclaimed = 0;
        entry_iterator_t current = begin;
        while (current != end)
            if (visible_now(current->committed) && current->presence == presence_t::erased_k) {
                --visible_count_;
                --visible_deleted_count_;
                current = entries_.erase(current);
                ++reclaimed;
            }
            else ++current;
        return reclaimed;
    }

    /**
     *  @brief Publishes the version stamped @p generation_to_unmask and drops every other visible one.
     *  @return Whether that version was there to publish, which a commit reports to its caller.
     */
    unmask_outcome_t unmask_and_compact_(entry_iterator_t begin, entry_iterator_t end,
                                         generation_t generation_to_unmask, commit_stamp_t stamp) noexcept {
        // The entry being unmasked is the one that survives, wherever it sits in the set's order.
        // Keeping whichever came last instead would erase it again the moment a revision with a
        // higher generation is already published, which happens whenever a transaction opens early
        // and commits late. Every other published revision gives way; other transactions' staged
        // ones are untouched, since they are not this commit's to decide.
        auto outcome = unmask_outcome_t::version_missing_k;
        entry_iterator_t current = begin;
        while (current != end) {
            entry_iterator_t next = current;
            ++next;
            if (current->generation == generation_to_unmask) {
                if (!visible_now(current->committed)) {
                    ++visible_count_;
                    visible_deleted_count_ += current->presence == presence_t::erased_k;
                    const_cast<commit_stamp_t &>(current->committed) = stamp;
                    outcome = unmask_outcome_t::unmasked_k;
                }
                // Listing one key twice makes the second pass find a version this commit already
                // published, which is settled rather than missing.
                else outcome = unmask_outcome_t::already_published_k;
            }
            else if (visible_now(current->committed)) {
                --visible_count_;
                visible_deleted_count_ -= current->presence == presence_t::erased_k;
                entries_.erase(current);
            }
            current = next;
        }
        return outcome;
    }

    /** @brief The visible element under @p comparable, or @c entries_.end() when the key is not there. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] entry_iterator_t find_visible_present_(comparable_type_ &&comparable) noexcept {
        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));
        while (range.first != range.second && !visible_now(range.first->committed)) ++range.first;
        if (range.first == range.second || range.first->presence != presence_t::present_k) return entries_.end();
        return range.first;
    }

    /**
     *  @brief Writes @p element as a commit of its own and drops the visible version it replaces.
     *
     *  @param[in] callback_stored Invoked with the stored element once it is visible. Must be @c noexcept.
     *  @param[in] callback_replaced Invoked with the element it displaced, if a reader could see one.
     *    Must be @c noexcept.
     *  @return Success, or an allocation failure that left the store untouched.
     */
    template <typename callback_stored_type_ = no_op_t, typename callback_replaced_type_ = no_op_t>
    [[nodiscard]] status_t publish_alone_(value_t &&element, callback_stored_type_ &&callback_stored = {},
                                          callback_replaced_type_ &&callback_replaced = {}) noexcept {
        generation_t const generation = next_generation_();
        commit_stamp_t const stamp = next_commit_stamp_();
        return invoke_safely([&]() {
            auto entry = versioned_entry_t {std::move(element)};
            entry.generation = generation;
            entry.presence = presence_t::present_k;
            entry.committed = stamp;
            auto range_end = entries_.insert(std::move(entry)).first;
            auto range_start = entries_.lower_bound(range_end->payload);
            ++visible_count_;
            erase_visible_(range_start, range_end, callback_replaced);
            callback_stored(range_end->payload);
        });
    }

    /**
     *  @brief Atomically inserts or assigns a collection of entries.
     *
     *  Either all entries will be inserted/assigned, or all will fail.
     *  This operation is identical to creating and committing
     *  a transaction with all the same elements put into it.
     *
     *  @section reference_store_why_not_take_r_value Why not take R-Value?
     *  We want this operation to be consistent, as the rest of the container,
     *  so we need a place to return all the objects, if the operation fails.
     *  With R-Value, the batch would be lost.
     *
     *  The import itself cannot fail: every node was already allocated into @p sources, and splicing
     *  it across only relinks pointers. Whatever can run out of memory has done so by now, which is
     *  what makes the all-or-nothing claim above hold.
     *
     *  @param[inout] sources Collection of entries to import, left empty.
     */
    void insert_or_assign_(entry_set_t &sources) noexcept {
        for (auto source = sources.begin(); source != sources.end();) {
            bool const should_compact = visible_now(source->committed);
            visible_count_ += should_compact;
            visible_deleted_count_ += should_compact && source->presence == presence_t::erased_k;
            auto source_node = sources.extract(source++);
            auto range_end = entries_.insert(std::move(source_node)).position;
            if (should_compact) {
                auto range_start = entries_.lower_bound(range_end->payload);
                erase_visible_(range_start, range_end);
            }
        }
    }

#pragma endregion Type Definitions

#pragma region Constructors and Assignment

  public:
    reference_store() noexcept(false) {}

    /** @brief Seeds the comparator, which an instance carrying state or a dispatch pointer needs. */
    explicit reference_store(comparator_t const &comparator) noexcept(false)
        : entries_(versioned_comparator_t(comparator)) {}
    reference_store(reference_store const &) = delete;
    reference_store(reference_store &&) noexcept = default;
    reference_store &operator=(reference_store const &) = delete;
    reference_store &operator=(reference_store &&) noexcept = default;

#pragma endregion Constructors and Assignment

#pragma region Capacity

    /**
     *  @brief Returns the number of visible (committed) non-deleted elements in the container.
     *  @return Number of elements.
     */
    [[nodiscard]] std::size_t size() const noexcept { return visible_count_ - visible_deleted_count_; }

    /**
     *  @brief Checks if the container has no visible elements.
     *  @return True if empty, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    /**
     *  @brief Returns the number of elements with key equal to the specified argument.
     *    For unique-key containers like this, returns either 0 or 1.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @return Number of elements with key equal to @p comparable (0 or 1).
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<std::size_t> count(comparable_type_ &&comparable) const noexcept {
        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));
        while (range.first != range.second && !visible_now(range.first->committed)) ++range.first;
        bool const present = range.first != range.second && range.first->presence == presence_t::present_k;
        return present ? std::size_t {1} : std::size_t {0};
    }

#pragma endregion Capacity

#pragma region Observers

    /** @brief Builds a store around a specific comparator, for comparators that carry state. */
    [[nodiscard]] static expected<store_t> make(comparator_t const &comparator) noexcept {
        return store_t {comparator};
    }

    /**
     *  @brief Factory method to create a new transactional set without throwing exceptions.
     *    Returns error status on allocation failure.
     *
     *  @return Container instance, or empty on allocation failure.
     */
    [[nodiscard]] static expected<store_t> make() noexcept {
        expected<store_t> opt_store;
        auto status = invoke_safely([&]() { opt_store = store_t {}; });
        if (failed(status)) return status_t::out_of_memory_heap_k;
        return opt_store;
    }

#pragma endregion Observers

#pragma region Transaction Management

    /**
     *  @brief Creates a new transaction with a fresh generation number.
     *    Transaction can be reset and reused after commit/rollback to avoid reallocations.
     *    Returns error status on allocation failure.
     *
     *  @return Transaction instance, or empty on allocation failure.
     */
    [[nodiscard]] expected<transaction_t> transaction() noexcept {
        expected<transaction_t> opt_txn;
        // The constructor is private, so `emplace` cannot reach it; build here, where we are a friend.
        auto status = invoke_safely([&]() { opt_txn = transaction_t {*this}; });
        if (failed(status)) return status_t::out_of_memory_heap_k;
        return opt_txn;
    }

#pragma endregion Transaction Management

#pragma region Modifiers

    /**
     *  @brief Atomically inserts an element, refusing the key that is already there.
     *
     *  @param[in] element Element to insert (moved into the container).
     *  @param[in] callback_inserted Invoked with the stored element. Must be @c noexcept.
     *  @param[in] callback_existing Invoked with the element already under the key, which keeps its value.
     *    Must be @c noexcept.
     *  @return Success, @c key_already_exists_k when the key is taken, or an allocation failure.
     */
    template <typename callback_inserted_type_ = no_op_t, typename callback_existing_type_ = no_op_t>
    [[nodiscard]] status_t insert(value_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                  callback_existing_type_ &&callback_existing = {}) noexcept {

        static_assert(is_safe_callback_for<callback_existing_type_, value_t const &>,
                      "callback_existing must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback_for<callback_inserted_type_, value_t const &>,
                      "callback_inserted must be noexcept invocable with value_t const &");

        if (auto const existing = find_visible_present_(mapping_key_or_itself(element)); existing != entries_.end()) {
            callback_existing(existing->payload);
            return key_already_exists_k;
        }
        return publish_alone_(std::move(element), callback_inserted);
    }

    /**
     *  @brief Atomically inserts an element, leaving the key that is already there untouched.
     *
     *  @param[in] element Element to insert (moved into the container).
     *  @param[in] callback_inserted Invoked with the stored element. Must be @c noexcept.
     *  @param[in] callback_existing Invoked with the element already under the key, which keeps its value.
     *    Must be @c noexcept.
     *  @return Success even when nothing was stored, or an allocation failure.
     */
    template <typename callback_inserted_type_ = no_op_t, typename callback_existing_type_ = no_op_t>
    [[nodiscard]] status_t insert_if_missing(value_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                             callback_existing_type_ &&callback_existing = {}) noexcept {

        static_assert(is_safe_callback_for<callback_existing_type_, value_t const &>,
                      "callback_existing must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback_for<callback_inserted_type_, value_t const &>,
                      "callback_inserted must be noexcept invocable with value_t const &");

        if (auto const existing = find_visible_present_(mapping_key_or_itself(element)); existing != entries_.end()) {
            callback_existing(existing->payload);
            return success_k;
        }
        return publish_alone_(std::move(element), callback_inserted);
    }

    /**
     *  @brief Atomically writes an element, taking the key whether or not it was there.
     *
     *  @param[in] element Element to insert or assign (moved into the container).
     *  @param[in] callback_inserted Invoked with the stored element when the key was free. Must be @c noexcept.
     *  @param[in] callback_existing Invoked with the element that was replaced. Must be @c noexcept.
     *  @return Success or an allocation failure.
     */
    template <typename callback_inserted_type_ = no_op_t, typename callback_existing_type_ = no_op_t>
    [[nodiscard]] status_t insert_or_assign(value_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                            callback_existing_type_ &&callback_existing = {}) noexcept {

        static_assert(is_safe_callback_for<callback_existing_type_, value_t const &>,
                      "callback_existing must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback_for<callback_inserted_type_, value_t const &>,
                      "callback_inserted must be noexcept invocable with value_t const &");

        // The element that gives way is reported by the compaction itself, which is also what decides
        // whether there was one - so nothing has to probe for the key before the write.
        bool replaced = false;
        return publish_alone_(
            std::move(element),
            [&](value_t const &stored) noexcept {
                if (!replaced) callback_inserted(stored);
            },
            [&](value_t const &previous) noexcept {
                replaced = true;
                callback_existing(previous);
            });
    }

    /**
     *  @brief Alias for @c insert_or_assign(). Atomically inserts or updates an element.
     *  @param[in] element Element to insert or assign (moved into the container).
     *  @return Success or error code (e.g., out of memory).
     */
    [[nodiscard]] status_t upsert(value_t &&element) noexcept { return insert_or_assign(std::move(element)); }

    /**
     *  @brief Atomically writes an element, refusing a key that is not already there.
     *    The strict counterpart to @c insert_or_assign(), which takes the key either way.
     *
     *  @param[in] element Element to write (moved into the container when the key is present).
     *  @return Success, @c key_not_found_k when no visible element carries the key, or an allocation failure.
     */
    [[nodiscard]] status_t update(value_t &&element) noexcept {
        if (find_visible_present_(mapping_key_or_itself(element)) == entries_.end()) return key_not_found_k;
        return insert_or_assign(std::move(element));
    }

    /**
     *  @brief Deleted: Use @c insert_if_missing() instead for "insert only if missing" semantics.
     *    The @c std::map::try_emplace() name doesn't clearly communicate insert failure strategies.
     *    We provide three explicit alternatives:
     *      - @c insert(): Fails with error if key exists
     *      - @c insert_if_missing(): Silently skips if key exists (use this instead of try_emplace)
     *      - @c insert_or_assign(): Always overwrites if key exists
     */
    template <typename... args_types_>
    status_t try_emplace(args_types_ &&...) noexcept = delete;

    /**
     *  @brief Atomically inserts or assigns a batch of elements. Either all succeed or all fail.
     *    Iterator dereferencing should return R-value references (use @c std::make_move_iterator()).
     *
     *  @param[in] begin Iterator to the first element.
     *  @param[in] end Iterator past the last element.
     *  @return Success or error code (e.g., out of memory).
     */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t insert_or_assign(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        generation_t const generation = next_generation_();
        commit_stamp_t const stamp = next_commit_stamp_();

        // Staging and importing share one guarded scope, so the batch never has to outlive it and
        // never has to be handed to `expected`, whose values must move without throwing - something
        // only libstdc++ and libc++ promise for `std::set`.
        return invoke_safely([&]() {
            entry_set_t batch;
            for (; begin != end; ++begin) {
                auto iterator = batch.emplace(*begin).first;
                const_cast<generation_t &>(iterator->generation) = generation;
                const_cast<commit_stamp_t &>(iterator->committed) = stamp;
                const_cast<presence_t &>(iterator->presence) = presence_t::present_k;
            }
            insert_or_assign_(batch);
        });
    }

    /**
     *  @brief Alias for batch @c insert_or_assign(). Atomically inserts or assigns a batch of elements.
     *  @param[in] begin Iterator to the first element.
     *  @param[in] end Iterator past the last element.
     *  @return Success or error code (e.g., out of memory).
     */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        return insert_or_assign(begin, end);
    }

    /**
     *  @brief Atomically inserts a batch, leaving already-present keys untouched.
     *  @param[in] begin Iterator to the first element.
     *  @param[in] end Iterator past the last element.
     *  @return Success or error code (e.g., out of memory).
     */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t insert_if_missing(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        auto maybe_txn = transaction();
        if (!maybe_txn) return status_t::out_of_memory_heap_k;
        auto &transaction = *maybe_txn;
        for (; begin != end; ++begin)
            if (auto status = transaction.insert_if_missing(value_t(*begin)); failed(status)) return status;
        if (auto status = transaction.stage(); failed(status)) return status;
        return transaction.commit();
    }

    /**
     *  @brief Atomically writes a batch, refusing the group if any key is absent.
     *  @param[in] begin Iterator to the first element.
     *  @param[in] end Iterator past the last element.
     *  @return Success, @c key_not_found_k if any key is absent, or an allocation failure. Nothing is
     *    written unless all of it is.
     */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t update(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        auto maybe_txn = transaction();
        if (!maybe_txn) return status_t::out_of_memory_heap_k;
        auto &transaction = *maybe_txn;
        for (; begin != end; ++begin) {
            value_t candidate(*begin);
            if (find_visible_present_(mapping_key_or_itself(candidate)) == entries_.end()) return key_not_found_k;
            if (auto status = transaction.upsert(std::move(candidate)); failed(status)) return status;
        }
        if (auto status = transaction.stage(); failed(status)) return status;
        return transaction.commit();
    }

#pragma endregion Modifiers

#pragma region Lookup

    /**
     *  @brief Finds a member @b equal to the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c value_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        find_visible_entry_(
            std::forward<comparable_type_>(comparable),
            [&](versioned_entry_t const &entry) noexcept { callback_found(entry.payload); },
            std::forward<callback_missing_type_>(callback_missing));
        return success_k;
    }

    /**
     *  @brief Copies out the member equal to @p comparable, for callers that cannot use a callback.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @return The copied element, or @c key_not_found_k when absent.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        [[maybe_unused]] status_t const looked_up = find(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        return result;
    }

    /**
     *  @brief Checks if a member @b equal to the given @p comparable exists.
     *    Convenience wrapper around @c find() for existence checks.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @return True if the element exists, false otherwise.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<bool> contains(comparable_type_ &&comparable) const noexcept {
        bool present = false;
        status_t const answered = find(
            std::forward<comparable_type_>(comparable), [&](value_t const &) noexcept { present = true; }, no_op_t {});
        if (failed(answered)) return answered;
        return present;
    }

    /**
     *  @brief Finds the first member @b greater or equal to the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c value_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        auto iterator = entries_.lower_bound(std::forward<comparable_type_>(comparable));

        // Skip all the invisible entries
        while (iterator != entries_.end() &&
               (!visible_now(iterator->committed) || iterator->presence == presence_t::erased_k))
            ++iterator;

        iterator != entries_.end() ? callback_found(iterator->payload) : callback_missing();
        return success_k;
    }

    /**
     *  @brief Finds the first member @b greater than the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c value_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        auto iterator = entries_.upper_bound(std::forward<comparable_type_>(comparable));

        // Skip all the invisible entries
        while (iterator != entries_.end() &&
               (!visible_now(iterator->committed) || iterator->presence == presence_t::erased_k))
            ++iterator;

        iterator != entries_.end() ? callback_found(iterator->payload) : callback_missing();
        return success_k;
    }

    /** @brief Copies out the first visible element ordered at or after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        [[maybe_unused]] status_t const bounded = lower_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        return result;
    }

    /** @brief Copies out the first visible element ordered strictly after @p comparable. */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept {
        expected<value_t> result {status_t::key_not_found_k};
        [[maybe_unused]] status_t const bounded = upper_bound(
            std::forward<comparable_type_>(comparable),
            [&](value_t const &value) noexcept { result = copy_safely(value); }, no_op_t {});
        return result;
    }

    /**
     *  @brief Finds all elements equal to a single key. Invokes callback for each matching element.
     *    For sets with unique keys, this returns at most one element (0 or 1).
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));

        // Iterate through all entries with this key (should be at most one visible)
        for (auto it = range.first; it != range.second; ++it)
            if (visible_now(it->committed) && it->presence == presence_t::present_k) callback(it->payload);
        return success_k;
    }

#pragma endregion Lookup

#pragma region Order Statistics

    /**
     *  @brief Hands @p callback_found the @p ordinal -th smallest visible element, counting from zero.
     *
     *  A linear walk of the whole store, and deliberately so: this is the oracle the ranked engines are
     *  checked against, so it is written to be obviously right rather than fast. Nothing here is
     *  augmented, and an ordinal read from a plain ordered walk cannot disagree with the walk.
     *
     *  @param[in] ordinal Zero-based position among the visible elements.
     *  @param[in] callback_found Callback to receive a @c value_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered when fewer elements are visible. Must be @c noexcept.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t select(std::size_t ordinal, callback_found_type_ &&callback_found,
                                  callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        std::size_t visible_index = 0;
        for (auto iterator = entries_.begin(); iterator != entries_.end(); ++iterator) {
            if (!visible_now(iterator->committed) || iterator->presence != presence_t::present_k) continue;
            if (visible_index == ordinal) {
                callback_found(iterator->payload);
                return success_k;
            }
            ++visible_index;
        }
        callback_missing();
        return success_k;
    }

    /**
     *  @brief Hands @p callback_found how many visible elements the store orders before @p comparable.
     *    A linear walk, for the reason @c select() takes one.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c std::size_t. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered when @p comparable is not visible. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, std::size_t>,
                      "callback_found must be noexcept invocable with std::size_t");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        // `comparable` is read twice below, so it stays an lvalue rather than being forwarded away.
        auto const boundary = entries_.lower_bound(comparable);
        std::size_t preceding = 0;
        for (auto iterator = entries_.begin(); iterator != boundary; ++iterator)
            if (visible_now(iterator->committed) && iterator->presence == presence_t::present_k) ++preceding;

        auto const matches = entries_.equal_range(comparable);
        for (auto iterator = matches.first; iterator != matches.second; ++iterator)
            if (visible_now(iterator->committed) && iterator->presence == presence_t::present_k) {
                callback_found(preceding);
                return success_k;
            }
        callback_missing();
        return success_k;
    }

#pragma endregion Order Statistics

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

        for (auto iterator = entries_.begin(); iterator != entries_.end(); ++iterator)
            if (visible_now(iterator->committed) && iterator->presence == presence_t::present_k)
                callback(iterator->payload);
        return success_k;
    }

#pragma endregion Enumeration

#pragma region Range Operations

    /**
     *  @brief Iterates over all entries in the range [ @p lower, @p upper). Const version.
     *    Unlike @c equal_range(), this takes TWO keys and returns all entries between them.
     *
     *  @param[in] lower Lower bound (inclusive).
     *  @param[in] upper Upper bound (exclusive).
     *  @param[in] callback Callback invoked for each element in range. Must be @c noexcept.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper,
                                 callback_type_ &&callback = {}) const noexcept {
        auto lower_iterator = entries_.lower_bound(std::forward<lower_type_>(lower));
        auto const upper_iterator = entries_.lower_bound(std::forward<upper_type_>(upper));
        for (; lower_iterator != upper_iterator; ++lower_iterator)
            if (visible_now(lower_iterator->committed) && lower_iterator->presence == presence_t::present_k)
                callback(lower_iterator->payload);
        return success_k;
    }

    /**
     *  @brief Iterates over key-value associations in [ @p lower, @p upper), providing mutable value access.
     *    Only enabled for mapping types. Callback receives (key_type const&, value_type&).
     *    Updates generation for each accessed element.
     *
     *  @param[in] lower Lower bound (inclusive).
     *  @param[in] upper Upper bound (exclusive).
     *  @param[in] callback Callback invoked with (key_type const&, value_type&) for each element. Must be @c noexcept.
     *  @return Always success here; the status is reported so a mutator that can fail on a sibling
     *    engine has the same channel on all three.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires is_mapping<value_t>
    {
        generation_t generation = next_generation_();
        auto lower_iterator = entries_.lower_bound(std::forward<lower_type_>(lower));
        auto const upper_iterator = entries_.lower_bound(std::forward<upper_type_>(upper));
        for (; lower_iterator != upper_iterator; ++lower_iterator) {
            if (!visible_now(lower_iterator->committed) || lower_iterator->presence == presence_t::erased_k) continue;
            // ! STL's `std::set::iterator` dereferencing operator returns immutable references
            // ! to isolate keys from possible modifications, corrupting the ordered layout.
            auto &entry = const_cast<versioned_entry_t &>(*lower_iterator);
            callback(entry.payload.key, entry.payload.mapped);
            entry.generation = generation;
        }
        return success_k;
    }

    /**
     *  @brief Erases a single entry matching the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive the erased @c value_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     *  @return @c key_not_found_k if no visible entry matched, otherwise success.
     *  @note A committed tombstone is not a match, but it is still reclaimed here - nothing else
     *    would ever reach it, and leaving it behind is how a key that was erased twice keeps its
     *    memory forever.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found = {},
                                 callback_missing_type_ &&callback_missing = {}) noexcept {

        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));

        // Skip all the invisible entries
        while (range.first != range.second && !visible_now(range.first->committed)) ++range.first;

        // Check if there are no visible entries at all
        if (range.first == range.second || range.first->presence == presence_t::erased_k) {
            if (range.first != range.second) {
                --visible_count_;
                --visible_deleted_count_;
                entries_.erase(range.first);
            }
            callback_missing();
            return status_t::key_not_found_k;
        }

        // Invoke callback before erasing
        callback_found(range.first->payload);

        // Erase the visible entry
        --visible_count_;
        visible_deleted_count_ -= range.first->presence == presence_t::erased_k;
        entries_.erase(range.first);
        return success_k;
    }

    /**
     *  @brief Erases all the entries falling in between the @p lower and the @p upper.
     *
     *  @param[in] lower Lower bound of the range.
     *  @param[in] upper Upper bound of the range (exclusive).
     *  @param[in] callback Optional callback invoked for each erased element. Must be @c noexcept.
     *  @return Always succeeds.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                       callback_type_ &&callback = {}) noexcept {

        static_assert(is_safe_callback_for<callback_type_, value_t const &>,
                      "callback must be noexcept invocable with value_t const &");

        auto lower_iterator = entries_.lower_bound(std::forward<lower_type_>(lower));
        auto const upper_iterator = entries_.lower_bound(std::forward<upper_type_>(upper));
        erase_visible_(lower_iterator, upper_iterator, std::forward<callback_type_>(callback));
        return success_k;
    }

    /**
     *  @brief Erases every entry ordered at or after @p lower, with no upper bound at all.
     *
     *  @param[in] lower Lower bound of the range, inclusive - an entry equal to it is erased, which is
     *    the same end @c erase_range() includes.
     *  @param[in] callback Optional callback invoked for each erased element. Must be @c noexcept.
     *  @return Always succeeds.
     */
    template <typename lower_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback = {}) noexcept {

        static_assert(is_safe_callback_for<callback_type_, value_t const &>,
                      "callback must be noexcept invocable with value_t const &");

        erase_visible_(entries_.lower_bound(std::forward<lower_type_>(lower)), entries_.end(),
                       std::forward<callback_type_>(callback));
        return success_k;
    }

    /**
     *  @brief Erases every entry ordered before @p upper, with no lower bound at all.
     *
     *  @param[in] upper Upper bound of the range, exclusive - an entry equal to it is kept, which is
     *    the same end @c erase_range() excludes.
     *  @param[in] callback Optional callback invoked for each erased element. Must be @c noexcept.
     *  @return Always succeeds.
     */
    template <typename upper_type_ = identifier_t, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback = {}) noexcept {

        static_assert(is_safe_callback_for<callback_type_, value_t const &>,
                      "callback must be noexcept invocable with value_t const &");

        erase_visible_(entries_.begin(), entries_.lower_bound(std::forward<upper_type_>(upper)),
                       std::forward<callback_type_>(callback));
        return success_k;
    }

    /**
     *  @brief Removes all elements from the container.
     *
     *  @return Always succeeds.
     *  @note The generation and stamp counters keep running. Rewinding either would hand a future
     *    transaction a number an open one already carries, and both are compared by value.
     */
    [[nodiscard]] status_t clear() noexcept {
        // Refused while anything is staged, because dropping a version an open transaction is about to
        // publish would make that publication fail - and the second half of a commit is the one step a
        // caller spanning several stores is promised cannot turn back.
        for (auto cursor = entries_.begin(); cursor != entries_.end(); ++cursor)
            if (!visible_now(cursor->committed)) return operation_not_permitted_k;

        entries_.clear();
        visible_count_ = 0;
        visible_deleted_count_ = 0;
        return success_k;
    }

    /**
     *  @brief Accepts a capacity hint and does nothing with it.
     *
     *  Every allocation this store makes is one node at a time, drawn when the write happens: the
     *  entries live in a node-based ordered set, and the vectors it also holds belong to transactions
     *  rather than to the store, which is why the transaction-level @c reserve() does real work while
     *  this one has nothing to reserve. Reserving here would therefore promise a later insert cannot
     *  run out of memory, which it still can.
     *
     *  @param[in] size Suggested capacity, ignored.
     *  @return Always succeeds.
     */
    [[nodiscard]] status_t reserve(std::size_t) noexcept { return {}; }

#pragma endregion Range Operations

#pragma region Vacuuming

    /** @brief Frees every entry no reader can reach. Returns how many were reclaimed. */
    [[nodiscard]] expected<std::size_t> vacuum() noexcept { return vacuum_between_(entries_.begin(), entries_.end()); }

    /**
     *  @brief Frees the unreachable entries ordered in [ @p lower, @p upper), so a caller can step through
     *    the keyspace instead of paying for one pass over all of it.
     *  @return How many entries were reclaimed.
     */
    template <typename lower_type_, typename upper_type_>
    [[nodiscard]] expected<std::size_t> vacuum(lower_type_ &&lower, upper_type_ &&upper) noexcept {
        return vacuum_between_(entries_.lower_bound(std::forward<lower_type_>(lower)),
                               entries_.lower_bound(std::forward<upper_type_>(upper)));
    }

#pragma endregion Vacuuming

#pragma region Sampling

    /**
     *  @brief Draws one visible entry uniformly from [ @p lower, @p upper), handing it to @p callback.
     *    Counts the window, then walks it again to the drawn offset - linear both times, and meant to
     *    be: this is the oracle the descending engines are checked against.
     *
     *  @param[in] lower Lower bound (inclusive).
     *  @param[in] upper Upper bound (exclusive).
     *  @param[inout] generator Random number generator (e.g., @c std::mt19937).
     *  @param[in] callback Callback to receive the sampled element. Must be @c noexcept. Invoked at most
     *    once, and never when the window shows nothing.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t sample_one(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                      callback_type_ &&callback) const noexcept {

        std::size_t count = 0;
        [[maybe_unused]] status_t const walked = range(lower, upper, [&](value_t const &) noexcept { ++count; });

        if (!count) return success_k;

        std::size_t matches_to_skip = draw_below(generator, count);
        bool drawn = false;
        return range(lower, upper, [&](value_t const &element) noexcept {
            // The walk has no early exit, so the draw has to guard itself against every element after it.
            if (drawn) return;
            if (matches_to_skip) --matches_to_skip;
            else {
                callback(element);
                drawn = true;
            }
        });
    }

    /**
     *  @brief Uniformly samples entries from [ @p lower, @p upper) using reservoir sampling algorithm.
     *    Fills a reservoir buffer with up to @p reservoir_capacity randomly selected elements.
     *
     *  @param[in] lower Lower bound (inclusive).
     *  @param[in] upper Upper bound (exclusive).
     *  @param[inout] generator Random number generator (e.g., @c std::mt19937).
     *  @param[inout] seen Count of entries processed (can span multiple calls).
     *  @param[in] reservoir_capacity Maximum number of samples to collect.
     *  @param[out] reservoir Random access iterator to output buffer.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    [[nodiscard]] status_t sample_reservoir(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                            std::size_t &seen, std::size_t reservoir_capacity,
                                            output_iterator_type_ &&reservoir) const noexcept {
        static_assert(std::is_nothrow_copy_assignable_v<value_t>,
                      "sampling copies each drawn element into the caller's buffer, so that copy must not throw");

        using output_iterator_t = std::remove_reference_t<output_iterator_type_>;
        using output_category_t = typename std::iterator_traits<output_iterator_t>::iterator_category;
        static_assert(std::is_same<std::random_access_iterator_tag, output_category_t>(), "Must be random access!");

        auto sampler = [&](value_t const &element) noexcept {
            if (seen < reservoir_capacity) reservoir[seen] = element;

            else {
                auto slot_to_replace = draw_below(generator, seen + 1);
                if (slot_to_replace < reservoir_capacity) reservoir[slot_to_replace] = element;
            }

            ++seen;
        };
        return range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), sampler);
    }

#pragma endregion Sampling
};

template < //
    typename value_type_, typename comparator_type_ = less_t, typename allocator_type_ = std::allocator<std::uint8_t>>
using reference_set = reference_store<value_type_, comparator_type_, allocator_type_>;

template < //
    typename key_type_, typename value_type_, typename comparator_type_ = less_t,
    typename allocator_type_ = std::allocator<std::uint8_t>>
using reference_map = reference_store<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>;

} // namespace ashvardanian::smashtable
