/**
 *  @brief  Transactional set container with ACID semantics, built on @c std::set for baseline reference.
 *    Provides 2-phase commit transactions with optimistic concurrency control through watch/CAS operations.
 *    All operations use callback-based APIs and are exception-safe via @c noexcept wrappers.
 *
 *  @file   transactional_std_store.hpp
 *  @author Ash Vardanian
 */
#pragma once
#include <functional>  // `std::less` as default
#include <memory>      // `std::allocator` as default
#include <optional>    // `std::optional` for "expected"
#include <random>      // `std::uniform_int_distribution` for sampling
#include <set>         // `std::set` for inner versioned entries
#include <type_traits> // `std::is_nothrow_invocable_v`
#include <vector>      // `std::vector` for watches

#include "shared.hpp"

namespace ashvardanian::smashtable {

template <typename callable_type_>
status_t invoke_safely(callable_type_ &&callable) noexcept {
    if constexpr (noexcept(callable())) {
        callable();
        return {success_k};
    }
    else {
        try {
            callable();
            return {success_k};
        }
        catch (std::bad_alloc const &) {
            return {errc_t::out_of_memory_heap_k};
        }
        catch (...) {
            return {errc_t::unknown_k};
        }
    }
}

/**
 *  @brief  Transactional set providing ACID semantics with 2-phase commit and watch/CAS operations.
 *    Built on @c std::set as a baseline reference implementation. Not thread-safe by itself.
 *
 *  @section Design Goals
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
 *  @section API Overview
 *
 *  - All lookups are heterogeneous: you can provide any type comparable to @p element_type_. Your comparator
 *    MUST define @code using is_transparent = void; @endcode to enable this, just like std::map and std::set.
 *  - No iterators are provided to keep the implementation simple and avoid complexity of maintaining persistent
 *    iterator validity across transactions and modifications.
 *  - All operations use callback-based APIs for consistency and exception safety via @c noexcept wrappers.
 *
 *  @subsection Insert Strategies
 *
 *  Three distinct insert strategies with different failure handling:
 *
 *  | Method               | Key Exists?     | Returns       | Use Case                                      |
 *  |----------------------|-----------------|---------------|-----------------------------------------------|
 *  | insert()             | Fails (error)   | invalid_arg_k | Strict: ensure key is new                     |
 *  | insert_if_missing()  | Skips (success) | success_k     | Lenient: insert only if absent, else no-op    |
 *  | insert_or_assign()   | Overwrites      | success_k     | Upsert: always update regardless of existence |
 *
 *  The @c upsert method is an alias for @c insert_or_assign, following a more DBMS-like naming convention.
 *  The @c try_emplace is deleted in favor of the more explicit @c insert_if_missing.
 *
 *  @tparam element_type_ Type of the elements stored in the set.
 *  @tparam comparator_type_ Ideally heterogeneous comparator for @c element_type_.
 *  @tparam allocator_type_ Arbitrary "rebindable" allocator for all internal structures.
 */
template < //
    typename element_type_, typename comparator_type_ = std::less<element_type_>,
    typename allocator_type_ = std::allocator<std::uint8_t>>
class transactional_std_store {

#pragma mark - Type Definitions

  public:
    using element_t = element_type_;
    using comparator_t = comparator_type_;
    using allocator_t = allocator_type_;

    using versioning_t = versioning_for<element_t, comparator_t>;
    using identifier_t = typename versioning_t::identifier_t;
    using generation_t = typename versioning_t::generation_t;
    using dated_identifier_t = typename versioning_t::dated_identifier_t;
    using watch_t = typename versioning_t::watch_t;
    using watched_identifier_t = typename versioning_t::watched_identifier_t;
    using versioned_entry_t = typename versioning_t::versioned_entry_t;
    using versioned_comparator_t = typename versioning_t::versioned_comparator_t;

  private:
    using entry_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<versioned_entry_t>;
    using entry_set_t = std::set< //
        versioned_entry_t, versioned_comparator_t, entry_allocator_t>;
    using entry_iterator_t = typename entry_set_t::iterator;

    using watches_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<watched_identifier_t>;
    using watches_array_t = std::vector<watched_identifier_t, watches_allocator_t>;
    using watch_iterator_t = typename watches_array_t::iterator;

    using store_t = transactional_std_store;

  public:
    class transaction_t {

        friend store_t;
        enum class stage_t {
            created_k,
            staged_k,
            committed_k,
        };

        store_t *store_ {nullptr};
        entry_set_t changes_ {};
        watches_array_t watches_ {};
        generation_t generation_ {0};
        stage_t stage_ {stage_t::created_k};

        transaction_t(store_t &set) noexcept(false) : store_(&set), generation_(set.new_generation_()) {}
        watch_t missing_watch() const noexcept { return watch_t {generation_, true}; }
        store_t &store_ref() noexcept { return *store_; }
        store_t const &store_ref() const noexcept { return *store_; }

      public:
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        transaction_t(transaction_t const &) = delete;
        transaction_t &operator=(transaction_t const &) = delete;

        /**
         *  @brief Returns the generation (sequence number) of this transaction.
         *  @return generation_t The transaction's generation identifier.
         */
        generation_t generation() const noexcept { return generation_; }

        /**
         *  @brief Checks if this transaction has any pending changes (upserts or erases).
         *  @return bool True if there are pending changes, false otherwise.
         */
        bool has_changes() const noexcept { return !changes_.empty(); }

        /**
         *  @brief Returns the number of pending changes in this transaction.
         *  @return std::size_t Count of staged changes (upserts and erases).
         */
        std::size_t changes_count() const noexcept { return changes_.size(); }

        /**
         *  @brief Stages an insert operation only if the key doesn't exist. Fails if key exists.
         *    Checks both transaction changes and main store for existence.
         *
         *  @param[in] element Element to insert (moved into the transaction).
         *  @param[in] callback_inserted Callback invoked if element will be inserted.
         *  @param[in] callback_exists Callback invoked if key already exists (insertion failed).
         *  @return status_t Success, or @c invalid_argument_k if key exists, or OOM error.
         */
        template <typename callback_inserted_type_ = no_op_t, typename callback_exists_type_ = no_op_t,
                  typename... tags_types_>
        [[nodiscard]] status_t insert(element_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                      callback_exists_type_ &&callback_exists = {}, tags_types_...) noexcept {

            static_assert(is_safe_callback_for<callback_exists_type_, element_t const &>,
                          "callback_exists must be noexcept invocable with element_t const &");
            static_assert(is_safe_callback<callback_inserted_type_>, "callback_inserted must be noexcept invocable");

            // Check local changes first
            identifier_t id {element};
            auto local_it = changes_.find(id);
            if (local_it != changes_.end() && !local_it->deleted) {
                callback_exists(local_it->element);
                return {invalid_argument_k};
            }

            // Check main store if not in local changes or was deleted locally
            bool exists_in_store = false;
            store_ref().find(
                id,
                [&](versioned_entry_t const &entry) noexcept {
                    exists_in_store = true;
                    callback_exists(entry.element);
                },
                []() noexcept {});

            if (exists_in_store) return {invalid_argument_k};

            // Key doesn't exist anywhere, proceed with insertion
            auto status = invoke_safely([&]() {
                auto iterator = changes_.lower_bound(element);
                if (iterator == changes_.end() || !versioned_comparator_t {}.same(iterator->element, element))
                    iterator = changes_.emplace_hint(iterator, std::move(element));
                else const_cast<element_t &>(iterator->element) = std::move(element);
                const_cast<generation_t &>(iterator->generation) = generation_;
                const_cast<bool &>(iterator->deleted) = false;
                const_cast<bool &>(iterator->visible) = false;
            });

            if (status) callback_inserted();
            return status;
        }

        /**
         *  @brief Stages an insert operation only if key is missing. Silently skips if key exists (no error).
         *    Checks both transaction changes and main store for existence.
         *
         *  @param[in] element Element to insert (moved into the transaction).
         *  @param[in] callback_inserted Callback invoked if element will be inserted (key doesn't exist).
         *  @param[in] callback_skipped Callback invoked if key already exists (insertion skipped).
         *  @return status_t Always succeeds (unless OOM). Success returned even if key exists.
         */
        template <typename callback_inserted_type_ = no_op_t, typename callback_skipped_type_ = no_op_t,
                  typename... tags_types_>
        [[nodiscard]] status_t insert_if_missing(element_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                                 callback_skipped_type_ &&callback_skipped = {},
                                                 tags_types_...) noexcept {

            static_assert(is_safe_callback<callback_inserted_type_>, "callback_inserted must be noexcept invocable");
            static_assert(is_safe_callback_for<callback_skipped_type_, element_t const &>,
                          "callback_skipped must be noexcept invocable with element_t const &");

            // Check local changes first
            identifier_t id {element};
            auto local_it = changes_.find(id);
            if (local_it != changes_.end() && !local_it->deleted) {
                callback_skipped(local_it->element);
                return {success_k}; // Success, just didn't insert
            }

            // Check main store if not in local changes or was deleted locally
            bool exists_in_store = false;
            store_ref().find(
                id,
                [&](versioned_entry_t const &entry) noexcept {
                    exists_in_store = true;
                    callback_skipped(entry.element);
                },
                []() noexcept {});

            if (exists_in_store) return {success_k}; // Success, just didn't insert

            // Key doesn't exist anywhere, proceed with insertion
            auto status = invoke_safely([&]() {
                auto iterator = changes_.lower_bound(element);
                if (iterator == changes_.end() || !versioned_comparator_t {}.same(iterator->element, element))
                    iterator = changes_.emplace_hint(iterator, std::move(element));
                else const_cast<element_t &>(iterator->element) = std::move(element);
                const_cast<generation_t &>(iterator->generation) = generation_;
                const_cast<bool &>(iterator->deleted) = false;
                const_cast<bool &>(iterator->visible) = false;
            });

            if (status) callback_inserted();
            return status;
        }

        /**
         *  @brief Stages an insert or assign operation for the given element. Always succeeds.
         *    Overwrites existing element if key exists. Changes visible after @c stage() and @c commit().
         *
         *  @param[in] element Element to insert or assign (moved into the transaction).
         *  @param[in] callback_inserted Callback invoked if element will be inserted (key doesn't exist).
         *  @param[in] callback_assigned Callback invoked if element will be assigned (key exists, value updated).
         *  @return status_t Success or error code (e.g., out of memory).
         */
        template <typename callback_inserted_type_ = no_op_t, typename callback_assigned_type_ = no_op_t,
                  typename... tags_types_>
        [[nodiscard]] status_t insert_or_assign_(element_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                                 callback_assigned_type_ &&callback_assigned = {},
                                                 tags_types_...) noexcept {

            static_assert(is_safe_callback<callback_inserted_type_>, "callback_inserted must be noexcept invocable");
            static_assert(is_safe_callback<callback_assigned_type_>, "callback_assigned must be noexcept invocable");

            // Check if key exists in local changes or store
            identifier_t id {element};
            auto local_it = changes_.find(id);
            bool key_exists = (local_it != changes_.end() && !local_it->deleted);

            // Check in main store
            if (!key_exists) key_exists = store_ref().contains(id);

            auto status = invoke_safely([&]() {
                auto iterator = changes_.lower_bound(element);
                if (iterator == changes_.end() || !versioned_comparator_t {}.same(iterator->element, element))
                    iterator = changes_.emplace_hint(iterator, std::move(element));
                else const_cast<element_t &>(iterator->element) = std::move(element);
                const_cast<generation_t &>(iterator->generation) = generation_;
                const_cast<bool &>(iterator->deleted) = false;
                const_cast<bool &>(iterator->visible) = false;
            });

            if (!status) return status;

            if (key_exists) callback_assigned();
            else callback_inserted();
            return status;
        }

        /**
         *  @brief Alias for @c insert_or_assign(). Stages an insert or assign operation.
         *  @param[in] element Element to insert or assign (moved into the transaction).
         *  @return status_t Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t upsert(element_t &&element) noexcept { return insert_or_assign(std::move(element)); }

        /**
         *  @brief Stages a delete operation for the element with the given identifier.
         *    The deletion is not visible until after @c stage() and @c commit().
         *
         *  @param[in] id Identifier of the element to erase.
         *  @return status_t Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t erase(identifier_t const &id) noexcept {
            return invoke_safely([&]() {
                auto iterator = changes_.lower_bound(id);
                if (iterator == changes_.end() || !versioned_comparator_t {}.same(iterator->element, id))
                    iterator = changes_.emplace_hint(iterator, element_t {id});
                else const_cast<element_t &>(iterator->element) = element_t {id};
                const_cast<generation_t &>(iterator->generation) = generation_;
                const_cast<bool &>(iterator->deleted) = true;
                const_cast<bool &>(iterator->visible) = false;
            });
        }

        /**
         *  @brief Pre-allocates memory for watch operations to reduce allocation failures during transaction.
         *
         *  @param[in] size Expected number of watches.
         *  @return status_t Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t reserve(std::size_t size) noexcept {
            return invoke_safely([&]() { watches_.reserve(size); });
        }

        /**
         *  @brief Registers a watch on the element with the given identifier for optimistic concurrency control.
         *    The transaction will fail at @c stage() if the watched element changes.
         *
         *  @param[in] id Identifier of the element to watch.
         *  @return status_t Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            status_t status;
            store_ref().find_visible_entry_(
                id,
                [&](versioned_entry_t const &entry) noexcept {
                    status = invoke_safely([&]() {
                        watches_.push_back({identifier_t {entry.element}, watch_t {entry.generation, entry.deleted}});
                    });
                },
                [&]() noexcept { status = invoke_safely([&]() { watches_.push_back({id, missing_watch()}); }); });
            return status;
        }

        /**
         *  @brief Registers a watch on an already-fetched entry for optimistic concurrency control.
         *    The transaction will fail at @c stage() if the watched element changes.
         *
         *  @param[in] entry Entry to watch (typically from a previous find/lookup).
         *  @return status_t Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t watch(versioned_entry_t const &entry) noexcept {
            return invoke_safely(
                [&] { watches_.push_back({identifier_t {entry.element}, watch_t {entry.generation, entry.deleted}}); });
        }

        /**
         *  @brief Finds a member @b equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c transactional_std_store::find(), will include entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t const &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {

            static_assert(is_safe_callback_for<callback_found_type_, element_t const &>,
                          "callback_found must be noexcept invocable with element_t const &");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            if (auto iterator = changes_.find(std::forward<comparable_type_>(comparable)); iterator != changes_.end()) {
                if (!iterator->deleted) callback_found(iterator->element);
                else callback_missing();
            }
            else
                store_ref().find(std::forward<comparable_type_>(comparable),
                                 std::forward<callback_found_type_>(callback_found),
                                 std::forward<callback_missing_type_>(callback_missing));
        }

        /**
         *  @brief Checks if a member @b equal to the given @p comparable exists, including transaction changes.
         *    Convenience wrapper around @c find() for existence checks.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @return bool True if the element exists, false otherwise.
         */
        template <typename comparable_type_ = identifier_t>
        [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
            bool found = false;
            find(
                std::forward<comparable_type_>(comparable), [&](element_t const &) noexcept { found = true; },
                []() noexcept {});
            return found;
        }

        /**
         *  @brief Finds the first member @b greater or equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c transactional_std_store::lower_bound(), includes entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t const &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept {

            static_assert(is_safe_callback_for<callback_found_type_, element_t const &>,
                          "callback_found must be noexcept invocable with element_t const &");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            auto external_previous_id = identifier_t(comparable);
            auto internal_iterator = changes_.lower_bound(std::forward<comparable_type_>(comparable));
            while (internal_iterator != changes_.end() && internal_iterator->deleted) ++internal_iterator;

            // Once picking the next smallest element from the global store,
            // we might face an entry, that was already deleted from here,
            // so this might become a multi-step process.
            auto faced_deleted_entry = false;
            auto callback_external_found = [&](element_t const &external_element) noexcept {
                // The simplest case is when we have an external object.
                if (internal_iterator == changes_.end()) return callback_found(external_element);

                element_t const &internal_element = internal_iterator->element;
                if (!versioned_comparator_t {}(external_element, internal_element))
                    return callback_found(internal_element);

                // Check if this entry was deleted and we should try again.
                auto external_id = identifier_t(external_element);
                auto external_element_internal_state = changes_.find(external_element);
                if (external_element_internal_state != changes_.end() && external_element_internal_state->deleted) {
                    faced_deleted_entry = true;
                    external_previous_id = external_id;
                }
                else callback_found(external_element);
            };
            auto callback_external_missing = [&]() noexcept {
                if (internal_iterator == changes_.end()) callback_missing();
                else callback_found(internal_iterator->element);
            };

            // Iterate until we find the a non-deleted external value
            auto &store = store_ref();
            do {
                faced_deleted_entry = false;
                store.lower_bound(external_previous_id, callback_external_found, callback_external_missing);
            } while (faced_deleted_entry);
        }

        /**
         *  @brief Finds the first member @b greater than the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c transactional_std_store::upper_bound(), includes entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t const &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept {

            static_assert(is_safe_callback_for<callback_found_type_, element_t const &>,
                          "callback_found must be noexcept invocable with element_t const &");
            static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

            auto external_previous_id = identifier_t(comparable);
            auto internal_iterator = changes_.upper_bound(std::forward<comparable_type_>(comparable));
            while (internal_iterator != changes_.end() && internal_iterator->deleted) ++internal_iterator;

            // Once picking the next smallest element from the global store,
            // we might face an entry, that was already deleted from here,
            // so this might become a multi-step process.
            auto faced_deleted_entry = false;
            auto callback_external_found = [&](element_t const &external_element) noexcept {
                // The simplest case is when we have an external object.
                if (internal_iterator == changes_.end()) return callback_found(external_element);

                element_t const &internal_element = internal_iterator->element;
                if (!versioned_comparator_t {}(external_element, internal_element))
                    return callback_found(internal_element);

                // Check if this entry was deleted and we should try again.
                auto external_id = identifier_t(external_element);
                auto external_element_internal_state = changes_.find(external_element);
                if (external_element_internal_state != changes_.end() && external_element_internal_state->deleted) {
                    faced_deleted_entry = true;
                    external_previous_id = external_id;
                }
                else callback_found(external_element);
            };
            auto callback_external_missing = [&]() noexcept {
                if (internal_iterator == changes_.end()) callback_missing();
                else callback_found(internal_iterator->element);
            };

            // Iterate until we find the a non-deleted external value
            auto &store = store_ref();
            do {
                faced_deleted_entry = false;
                store.upper_bound(external_previous_id, callback_external_found, callback_external_missing);
            } while (faced_deleted_entry);
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
        void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
            // First, iterate over local changes
            auto lower_internal = changes_.lower_bound(std::forward<lower_type_>(lower));
            auto upper_internal = changes_.lower_bound(std::forward<upper_type_>(upper));
            for (auto it = lower_internal; it != upper_internal; ++it)
                if (!it->deleted) callback(it->element);

            // Then, iterate over external store, skipping entries that were modified or deleted locally
            store_ref().range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                              [&](element_t const &external_element) noexcept {
                                  // Check if this entry exists in local changes
                                  auto local_state = changes_.find(external_element);
                                  // Not modified locally, include it
                                  if (local_state == changes_.end()) callback(external_element);
                                  // If modified locally, we already processed it above
                              });
        }

        /**
         *  @brief Validates watches and stages all changes to the main store, making them visible but uncommitted.
         *    Fails with @c errc_t::consistency_k if any watched elements changed.
         *
         *  @return status_t Success, or consistency error if watches failed validation.
         */
        template <typename... tags_types_>
        [[nodiscard]] status_t stage(tags_types_... tags) noexcept {
            // First, check if we have any collisions by validating watches.
            auto &store = store_ref();
            auto entry_missing = missing_watch();
            for (auto const &id_and_watch : watches_) {
                auto consistency_violated = false;
                store.find_latest_entry_(
                    id_and_watch.id,
                    [&](versioned_entry_t const &entry) noexcept {
                        consistency_violated = entry != id_and_watch.watch;
                    },
                    [&]() noexcept { consistency_violated = entry_missing != id_and_watch.watch; });
                if (consistency_violated) return {errc_t::consistency_k};
            }

            // Now all of our watches will be replaced with "links" to entries
            // we are merging into the main tree.
            watches_.clear();
            auto status = invoke_safely([&]() { watches_.reserve(changes_.size()); });
            if (!status) return status;

            // No new memory allocations or failures are possible after that.
            // It is all safe.
            for (auto const &entry : changes_)
                watches_.push_back({identifier_t {entry.element}, watch_t {generation_, entry.deleted}});

            // Than just merge our current nodes.
            // The visibility will be updated later in the `commit`.
            store.entries_.merge(changes_);
            stage_ = stage_t::staged_k;

            // Support return_new_size to export staged count
            get_type_or<return_new_size_t, black_hole_t>(tags...) = watches_.size();
            return {success_k};
        }

        /**
         *  @brief Resets the transaction to a clean state, discarding all changes and watches.
         *    If transaction was staged, all staged changes are removed from the main store.
         *    A new generation is assigned for reuse of this transaction.
         *
         *  @return status_t Always succeeds.
         */
        [[nodiscard]] status_t reset() noexcept {
            // If the transaction was "staged",
            // we must delete all the entries.
            auto &store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const &id_and_watch : watches_) {
                    // Heterogeneous `erase` is only coming in C++23.
                    dated_identifier_t dated {id_and_watch.id, id_and_watch.watch.generation};
                    if (auto iterator = store.entries_.find(dated); iterator != store.entries_.end())
                        store.entries_.erase(iterator);
                }

            watches_.clear();
            changes_.clear();
            stage_ = stage_t::created_k;
            generation_ = store.new_generation_();
            return {success_k};
        }

        /**
         *  @brief Rolls back a previously staged transaction, moving changes from the store back to the transaction.
         *    Can only be called on staged transactions. Watches are cleared, new generation is assigned.
         *
         *  @return status_t Success, or @c operation_not_permitted_k if transaction is not staged.
         */
        [[nodiscard]] status_t rollback() noexcept {
            if (stage_ != stage_t::staged_k) return {operation_not_permitted_k};

            // Transaction was staged, we must extract all the entries back
            auto &store = store_ref();
            for (auto const &id_and_watch : watches_) {
                dated_identifier_t dated {id_and_watch.id, id_and_watch.watch.generation};
                auto source = store.entries_.find(dated);
                auto node = store.entries_.extract(source);
                changes_.insert(std::move(node));
            }

            watches_.clear();
            stage_ = stage_t::created_k;
            generation_ = store.new_generation_();
            return {success_k};
        }

        /**
         *  @brief Commits a previously staged transaction, making all changes permanently visible.
         *    Can only be called on staged transactions. Removes older versions of modified entries.
         *
         *  @tparam tags_types_ Optional tag types (@c return_new_size_t to export visible count).
         *  @return status_t Success, or @c operation_not_permitted_k if transaction is not staged.
         */
        template <typename... tags_types_>
        [[nodiscard]] status_t commit(tags_types_... tags) noexcept {
            if (stage_ != stage_t::staged_k) return {operation_not_permitted_k};

            // Once we make an entry visible,
            // if there are more than one with the same key,
            // the older generation must die.
            auto &store = store_ref();
            for (auto const &id_and_watch : watches_) {
                auto range = store.entries_.equal_range(id_and_watch.id);
                store.unmask_and_compact_(range.first, range.second, id_and_watch.watch.generation);
            }

            stage_ = stage_t::created_k;
            get_type_or<return_new_size_t, black_hole_t>(tags...) = store.visible_count_;
            return {success_k};
        }
    };

  private:
    entry_set_t entries_ {};
    generation_t generation_ {0};
    std::size_t visible_count_ {0};
    std::size_t visible_deleted_count_ {0};

    friend class transaction_t;

    transactional_std_store() noexcept(false) {}
    generation_t new_generation_() noexcept { return ++generation_; }

    /**
     *  @brief Internal API: Finds the latest visible entry and invokes callback with @c versioned_entry_t const &.
     *    Used by internal methods that need access to generation/deleted/visible fields.
     *    Only considers VISIBLE entries (committed/staged).
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
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
        while (range.first != range.second && !range.first->visible) ++range.first;

        if (range.first != range.second && !range.first->deleted) callback_found(*range.first);
        else callback_missing();
    }

    /**
     *  @brief Internal API: Finds the latest entry regardless of visibility for watch validation.
     *    Checks ALL entries including staged (invisible) ones.
     *    Critical for detecting write-write conflicts with concurrent transactions.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive a @c versioned_entry_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void find_latest_entry_(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                            callback_missing_type_ &&callback_missing = {}) const noexcept {
        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));

        // Locate the most recent revision regardless of visibility.
        entry_iterator_t latest = range.second;
        for (auto it = range.first; it != range.second; ++it)
            if (latest == range.second || it->generation > latest->generation) latest = it;

        // Invoke whichever callback matches the outcome.
        if (latest == range.second || latest->deleted) callback_missing();
        else callback_found(*latest);
    }

    template <typename callback_type_ = no_op_t>
    void erase_visible_(entry_iterator_t begin, entry_iterator_t end, callback_type_ &&callback = {}) noexcept {
        entry_iterator_t current = begin;
        while (current != end)
            if (current->visible) {
                callback(current->element);
                --visible_count_;
                visible_deleted_count_ -= current->deleted;
                current = entries_.erase(current);
            }
            else ++current;
    }

    void unmask_and_compact_(entry_iterator_t begin, entry_iterator_t end, generation_t generation_to_unmask) noexcept {
        entry_iterator_t current = begin;
        entry_iterator_t last_visible_entry = end;
        for (; current != end; ++current) {
            auto keep_this = current->generation == generation_to_unmask;
            if (keep_this) {
                visible_count_ += !current->visible;
                visible_deleted_count_ += !current->visible && current->deleted;
                const_cast<bool &>(current->visible) = true;
            }

            if (!current->visible) continue;

            // Older revisions must die
            if (last_visible_entry != end) {
                --visible_count_;
                visible_deleted_count_ -= last_visible_entry->deleted;
                entries_.erase(last_visible_entry);
            }
            last_visible_entry = current;
        }
    }

    /**
     *  @brief Atomically inserts or assigns a collection of entries.
     *
     *  Either all entries will be inserted/assigned, or all will fail.
     *  This operation is identical to creating and committing
     *  a transaction with all the same elements put into it.
     *
     *  @section Why not take R-Value?
     *  We want this operation to be consistent, as the rest of the container,
     *  so we need a place to return all the objects, if the operation fails.
     *  With R-Value, the batch would be lost.
     *
     *  @param[inout] sources Collection of entries to import.
     *  @return status_t Can fail, if out of memory.
     */
    [[nodiscard]] status_t insert_or_assign_(entry_set_t &sources) noexcept {
        for (auto source = sources.begin(); source != sources.end();) {
            bool should_compact = source->visible;
            visible_count_ += source->visible;
            visible_deleted_count_ += source->visible && source->deleted;
            auto source_node = sources.extract(source++);
            auto range_end = entries_.insert(std::move(source_node)).position;
            if (should_compact) {
                auto range_start = entries_.lower_bound(range_end->element);
                erase_visible_(range_start, range_end);
            }
        }
        return {success_k};
    }

#pragma mark - Constructors and Assignment

  public:
    transactional_std_store(transactional_std_store const &) = delete;
    transactional_std_store(transactional_std_store &&) noexcept = default;
    transactional_std_store &operator=(transactional_std_store const &) = delete;
    transactional_std_store &operator=(transactional_std_store &&) noexcept = default;

#pragma mark - Capacity

    /**
     *  @brief Returns the number of visible (committed) non-deleted elements in the container.
     *  @return std::size_t Number of elements.
     */
    [[nodiscard]] std::size_t size() const noexcept { return visible_count_ - visible_deleted_count_; }

    /**
     *  @brief Checks if the container has no visible elements.
     *  @return bool True if empty, false otherwise.
     */
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    /**
     *  @brief Returns the number of elements with key equal to the specified argument.
     *    For unique-key containers like this, returns either 0 or 1.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @return std::size_t Number of elements with key equal to @p comparable (0 or 1).
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] std::size_t count(comparable_type_ &&comparable) const noexcept {
        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));

        // Skip invisible entries
        while (range.first != range.second && !range.first->visible) ++range.first;

        // Return 1 if we found a visible, non-deleted entry, otherwise 0
        return (range.first != range.second && !range.first->deleted) ? 1 : 0;
    }

#pragma mark - Observers

    /**
     *  @brief Factory method to create a new transactional set without throwing exceptions.
     *    Returns an empty optional on allocation failure.
     *
     *  @return std::optional<store_t> Container instance or empty optional on failure.
     */
    [[nodiscard]] static std::optional<store_t> make() noexcept {
        std::optional<store_t> result;
        invoke_safely([&]() { result.emplace(store_t {}); });
        return result;
    }

#pragma mark - Transaction Management

    /**
     *  @brief Creates a new transaction with a fresh generation number.
     *    Transaction can be reset and reused after commit/rollback to avoid reallocations.
     *    Returns empty optional on allocation failure.
     *
     *  @return std::optional<transaction_t> Transaction instance or empty optional on failure.
     */
    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        std::optional<transaction_t> result;
        invoke_safely([&]() { result.emplace(transaction_t {*this}); });
        return result;
    }

#pragma mark - Modifiers

    /**
     *  @brief Atomically inserts an element only if the key doesn't exist. Fails if key exists.
     *    This is the strict insert semantics matching @c std::map::insert().
     *
     *  @param[in] element Element to insert (moved into the container).
     *  @param[in] callback_inserted Callback invoked if element was inserted.
     *  @param[in] callback_exists Callback invoked if key already exists (insertion failed).
     *  @return status_t Success, or @c invalid_argument_k if key exists, or OOM error.
     */
    template <typename callback_inserted_type_ = no_op_t, typename callback_exists_type_ = no_op_t>
    [[nodiscard]] status_t insert(element_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                  callback_exists_type_ &&callback_exists = {}) noexcept {

        static_assert(is_safe_callback_for<callback_exists_type_, element_t const &>,
                      "callback_exists must be noexcept invocable with element_t const &");
        static_assert(is_safe_callback<callback_inserted_type_>, "callback_inserted must be noexcept invocable");

        // First check if key already exists
        identifier_t id {element};
        auto range = entries_.equal_range(id);

        // Skip invisible entries
        while (range.first != range.second && !range.first->visible) ++range.first;

        // If we found a visible, non-deleted entry, key exists - fail
        if (range.first != range.second && !range.first->deleted) {
            callback_exists(range.first->element);
            return {invalid_argument_k};
        }

        // Key doesn't exist, proceed with insertion
        generation_t generation = new_generation_();
        auto status = invoke_safely([&]() {
            auto entry = versioned_entry_t {std::move(element)};
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            auto range_end = entries_.insert(std::move(entry)).first;
            auto range_start = entries_.lower_bound(range_end->element);
            ++visible_count_;
            erase_visible_(range_start, range_end);
        });

        if (status) callback_inserted();
        return status;
    }

    /**
     *  @brief Atomically inserts an element only if missing. Silently skips if key exists (no error).
     *    This is the "silent no-op" insert semantics.
     *
     *  @param[in] element Element to insert (moved into the container).
     *  @param[in] callback_inserted Callback invoked if element was inserted (key didn't exist).
     *  @param[in] callback_skipped Callback invoked if key already exists (insertion skipped).
     *  @return status_t Always succeeds (unless OOM). Success returned even if key exists.
     */
    template <typename callback_inserted_type_ = no_op_t, typename callback_skipped_type_ = no_op_t>
    [[nodiscard]] status_t insert_if_missing(element_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                             callback_skipped_type_ &&callback_skipped = {}) noexcept {

        static_assert(is_safe_callback<callback_inserted_type_>, "callback_inserted must be noexcept invocable");
        static_assert(is_safe_callback_for<callback_skipped_type_, element_t const &>,
                      "callback_skipped must be noexcept invocable with element_t const &");

        // Check if key already exists
        identifier_t id {element};
        auto range = entries_.equal_range(id);

        // Skip invisible entries
        while (range.first != range.second && !range.first->visible) ++range.first;

        // If we found a visible, non-deleted entry, key exists - skip silently
        if (range.first != range.second && !range.first->deleted) {
            callback_skipped(range.first->element);
            return {success_k}; // Success, just didn't insert
        }

        // Key doesn't exist, proceed with insertion
        generation_t generation = new_generation_();
        auto status = invoke_safely([&]() {
            auto entry = versioned_entry_t {std::move(element)};
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            auto range_end = entries_.insert(std::move(entry)).first;
            auto range_start = entries_.lower_bound(range_end->element);
            ++visible_count_;
            erase_visible_(range_start, range_end);
        });

        if (status) callback_inserted();
        return status;
    }

    /**
     *  @brief Atomically inserts or updates an element. Always succeeds (unless OOM).
     *    Overwrites existing element if key exists. Matches @c std::map::insert_or_assign() semantics.
     *
     *  @param[in] element Element to insert or assign (moved into the container).
     *  @param[in] callback_inserted Callback invoked if element was inserted (key didn't exist).
     *  @param[in] callback_assigned Callback invoked if element was assigned (key existed, value updated).
     *  @return status_t Success or error code (e.g., out of memory).
     */
    template <typename callback_inserted_type_ = no_op_t, typename callback_assigned_type_ = no_op_t>
    [[nodiscard]] status_t insert_or_assign(element_t &&element, callback_inserted_type_ &&callback_inserted = {},
                                            callback_assigned_type_ &&callback_assigned = {}) noexcept {

        static_assert(is_safe_callback<callback_inserted_type_>, "callback_inserted must be noexcept invocable");
        static_assert(is_safe_callback<callback_assigned_type_>, "callback_assigned must be noexcept invocable");

        // Check if key exists
        identifier_t id {element};
        auto range = entries_.equal_range(id);
        while (range.first != range.second && !range.first->visible) ++range.first;
        bool key_exists = (range.first != range.second && !range.first->deleted);

        generation_t generation = new_generation_();
        auto status = invoke_safely([&]() {
            auto entry = versioned_entry_t {std::move(element)};
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            auto range_end = entries_.insert(std::move(entry)).first;
            auto range_start = entries_.lower_bound(range_end->element);
            ++visible_count_;
            erase_visible_(range_start, range_end);
        });

        if (!status) return status;

        if (key_exists) callback_assigned();
        else callback_inserted();
        return status;
    }

    /**
     *  @brief Alias for @c insert_or_assign(). Atomically inserts or updates an element.
     *  @param[in] element Element to insert or assign (moved into the container).
     *  @return status_t Success or error code (e.g., out of memory).
     */
    [[nodiscard]] status_t upsert(element_t &&element) noexcept { return insert_or_assign(std::move(element)); }

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
     *  @return status_t Success or error code (e.g., out of memory).
     */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t insert_or_assign(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        generation_t generation = new_generation_();
        std::optional<entry_set_t> batch;
        auto batch_construction_status = invoke_safely([&]() {
            batch = entry_set_t {};
            for (; begin != end; ++begin) {
                auto iterator = batch->emplace(*begin).first;
                const_cast<generation_t &>(iterator->generation) = generation;
                const_cast<bool &>(iterator->visible) = true;
                const_cast<bool &>(iterator->deleted) = false;
            }
        });
        if (!batch_construction_status) return batch_construction_status;

        return insert_or_assign(batch.value());
    }

    /**
     *  @brief Alias for batch @c insert_or_assign(). Atomically inserts or assigns a batch of elements.
     *  @param[in] begin Iterator to the first element.
     *  @param[in] end Iterator past the last element.
     *  @return status_t Success or error code (e.g., out of memory).
     */
    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {
        return insert_or_assign(begin, end);
    }

#pragma mark - Lookup

    /**
     *  @brief Finds a member @b equal to the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c element_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, element_t const &>,
                      "callback_found must be noexcept invocable with element_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        find_visible_entry_(
            std::forward<comparable_type_>(comparable),
            [&](versioned_entry_t const &entry) noexcept { callback_found(entry.element); },
            std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief Checks if a member @b equal to the given @p comparable exists.
     *    Convenience wrapper around @c find() for existence checks.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @return bool True if the element exists, false otherwise.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
        bool found = false;
        find(
            std::forward<comparable_type_>(comparable), [&](element_t const &) noexcept { found = true; },
            []() noexcept {});
        return found;
    }

    /**
     *  @brief Finds the first member @b greater or equal to the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c element_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, element_t const &>,
                      "callback_found must be noexcept invocable with element_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        auto iterator = entries_.lower_bound(std::forward<comparable_type_>(comparable));

        // Skip all the invisible entries
        while (iterator != entries_.end() && (!iterator->visible || iterator->deleted)) ++iterator;

        iterator != entries_.end() ? callback_found(iterator->element) : callback_missing();
    }

    /**
     *  @brief Finds the first member @b greater than the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c element_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, element_t const &>,
                      "callback_found must be noexcept invocable with element_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        auto iterator = entries_.upper_bound(std::forward<comparable_type_>(comparable));

        // Skip all the invisible entries
        while (iterator != entries_.end() && (!iterator->visible || iterator->deleted)) ++iterator;

        iterator != entries_.end() ? callback_found(iterator->element) : callback_missing();
    }

    /**
     *  @brief Finds all elements equal to a single key. Invokes callback for each matching element.
     *    For sets with unique keys, this returns at most one element (0 or 1).
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
    void equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));

        // Iterate through all entries with this key (should be at most one visible)
        for (auto it = range.first; it != range.second; ++it)
            if (it->visible && !it->deleted) callback(it->element);
    }

#pragma mark - Range Operations

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
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        auto lower_iterator = entries_.lower_bound(std::forward<lower_type_>(lower));
        auto const upper_iterator = entries_.lower_bound(std::forward<upper_type_>(upper));
        for (; lower_iterator != upper_iterator; ++lower_iterator)
            if (lower_iterator->visible && !lower_iterator->deleted) callback(lower_iterator->element);
    }

    /**
     *  @brief Iterates over key-value associations in [ @p lower, @p upper), providing mutable value access.
     *    Only enabled for association types. Callback receives (key_type const&, value_type&).
     *    Updates generation for each accessed element.
     *
     *  @param[in] lower Lower bound (inclusive).
     *  @param[in] upper Upper bound (exclusive).
     *  @param[in] callback Callback invoked with (key_type const&, value_type&) for each element. Must be @c noexcept.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    void update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires is_association<element_t>
    {
        generation_t generation = new_generation_();
        auto lower_iterator = entries_.lower_bound(std::forward<lower_type_>(lower));
        auto const upper_iterator = entries_.lower_bound(std::forward<upper_type_>(upper));
        for (; lower_iterator != upper_iterator; ++lower_iterator) {
            if (!lower_iterator->visible || lower_iterator->deleted) continue;
            // ! STL's `std::set::iterator` dereferencing operator returns immutable references
            // ! to isolate keys from possible modifications, corrupting the ordered layout.
            auto &entry = const_cast<versioned_entry_t &>(*lower_iterator);
            callback(entry.element.key, entry.element.value);
            entry.generation = generation;
        }
    }

    /**
     *  @brief Erases a single entry matching the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive the erased @c element_t const &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     *  @return status_t Always succeeds.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    status_t erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found = {},
                   callback_missing_type_ &&callback_missing = {}) noexcept {

        auto range = entries_.equal_range(std::forward<comparable_type_>(comparable));

        // Skip all the invisible entries
        while (range.first != range.second && !range.first->visible) ++range.first;

        // Check if there are no visible entries at all
        if (range.first == range.second || range.first->deleted) {
            callback_missing();
            return status_t {success_k};
        }

        // Invoke callback before erasing
        callback_found(range.first->element);

        // Erase the visible entry
        --visible_count_;
        visible_deleted_count_ -= range.first->deleted;
        entries_.erase(range.first);
        return status_t {success_k};
    }

    /**
     *  @brief Erases all the entries falling in between the @p lower and the @p upper.
     *
     *  @param[in] lower Lower bound of the range.
     *  @param[in] upper Upper bound of the range (exclusive).
     *  @param[in] callback Optional callback invoked for each erased element. Must be @c noexcept.
     *  @return status_t Always succeeds.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) noexcept {

        static_assert(is_safe_callback_for<callback_type_, element_t const &>,
                      "callback must be noexcept invocable with element_t const &");

        auto lower_iterator = entries_.lower_bound(std::forward<lower_type_>(lower));
        auto const upper_iterator = entries_.lower_bound(std::forward<upper_type_>(upper));
        erase_visible_(lower_iterator, upper_iterator, std::forward<callback_type_>(callback));
        return status_t {success_k};
    }

    /**
     *  @brief Removes all elements from the container and resets generation counter.
     *
     *  @return status_t Always succeeds.
     */
    [[nodiscard]] status_t clear() noexcept {
        entries_.clear();
        generation_ = 0;
        visible_count_ = 0;
        visible_deleted_count_ = 0;
        return {success_k};
    }

    /**
     *  @brief Hints to the container to pre-allocate memory. No-op.
     *
     *  @param[in] size Suggested capacity (ignored for std::set).
     *  @return status_t Always succeeds.
     *
     *  @note This is a no-op because @c std::set doesn't support reserving capacity.
     */
    [[nodiscard]] status_t reserve(std::size_t) noexcept { return {}; }

#pragma mark - Sampling

    /**
     *  @brief Uniformly samples a single random entry from the range [ @p lower, @p upper).
     *    Uses a two-pass algorithm: first counts entries, then selects random offset.
     *
     *  @param[in] lower Lower bound (inclusive).
     *  @param[in] upper Upper bound (exclusive).
     *  @param[inout] generator Random number generator (e.g., @c std::mt19937).
     *  @param[in] callback Callback to receive the sampled element. Must be @c noexcept.
     *
     *  @note Inefficient for large ranges. Use reservoir sampling overload for multiple samples.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                      callback_type_ &&callback) const noexcept {

        std::size_t count = 0;
        range(lower, upper, [&](element_t const &) noexcept { ++count; });

        if (!count) return;

        std::uniform_int_distribution<std::size_t> distribution {0, count - 1};
        std::size_t matches_to_skip = distribution(generator);
        range(lower, upper, [&](element_t const &element) noexcept {
            if (matches_to_skip) --matches_to_skip;
            else callback(element);
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
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator, std::size_t &seen,
                      std::size_t reservoir_capacity, output_iterator_type_ &&reservoir) const noexcept {

        using output_iterator_t = std::remove_reference_t<output_iterator_type_>;
        using output_category_t = typename std::iterator_traits<output_iterator_t>::iterator_category;
        static_assert(std::is_same<std::random_access_iterator_tag, output_category_t>(), "Must be random access!");

        auto sampler = [&](element_t const &element) noexcept {
            if (seen < reservoir_capacity) reservoir[seen] = element;

            else {
                std::uniform_int_distribution<std::size_t> distribution {0, seen};
                auto slot_to_replace = distribution(generator);
                if (slot_to_replace < reservoir_capacity) reservoir[slot_to_replace] = element;
            }

            ++seen;
        };
        range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), sampler);
    }
};

/**
 *  @brief Unlike @c std::set<>::merge, this function overwrites existing values.
 *
 *  @see https://en.cppreference.com/w/cpp/container/set#Member_types
 *  @see https://en.cppreference.com/w/cpp/container/set/insert
 */
template <typename keys_type_, typename compare_type_, typename allocator_type_>
void merge_overwrite(std::set<keys_type_, compare_type_, allocator_type_> &target,
                     std::set<keys_type_, compare_type_, allocator_type_> &source) noexcept {
    for (auto source_it = source.begin(); source_it != source.end();) {
        auto node = source.extract(source_it++);
        auto result = target.insert(std::move(node));
        if (!result.inserted) std::swap(*result.position, result.node.value());
    }
}

template < //
    typename element_type_, typename comparator_type_ = std::less<element_type_>,
    typename allocator_type_ = std::allocator<std::uint8_t>>
using transactional_std_set = transactional_std_store<element_type_, comparator_type_, allocator_type_>;

template < //
    typename key_type_, typename value_type_, typename comparator_type_ = std::less<key_type_>,
    typename allocator_type_ = std::allocator<std::uint8_t>>
using transactional_std_map =
    transactional_std_store<association<key_type_, value_type_>, comparator_type_, allocator_type_>;

} // namespace ashvardanian::smashtable
