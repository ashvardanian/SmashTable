/**
 *  @brief  Generic transactional binary tree container with ACID semantics, providing 2-phase commit transactions.
 *    Can be instantiated with any binary search tree implementation (AVL, WB, etc.) supporting the required interface.
 *    All operations use callback-based APIs and are exception-free via @c noexcept constraints.
 *
 *  @file   transactional_binary_tree.hpp
 *  @author Ash Vardanian
 */
#pragma once
#include <cassert>   // `assert`
#include <algorithm> // `std::max`
#include <memory>    // `std::allocator`
#include <optional>  // `std::optional`
#include <random>    // `std::uniform_int_distribution`
#include <utility>   // `std::exchange`

#include "shared.hpp"
#include "basic_vector.hpp"
#include "basic_avl_tree.hpp"
#include "basic_wb_tree.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief  Generic transactional binary tree providing 2-phase commits and "watch" operations.
 *    Can be instantiated with AVL trees, weight-balanced trees, or other binary search tree implementations.
 *    Not thread-safe by itself. Entirely exception-free, with all methods marked @c noexcept.
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
 *  The state management doesn't rely on entry pointers or iterators. Those could simplify the implementation,
 *  but introduce require validity constraints for re-allocations and modifications of the tree and underlying
 *  allocator behavior. So all entry IDs must either be nothrow-copyable or provide a @c copy() member method
 *  returning an @c std::optional<T> to support safe copying for watch bookkeeping.
 *
 *  @see https://jepsen.io/consistency/models/monotonic-atomic-view
 *  @see https://jepsen.io/consistency/models/read-committed
 *
 *  @section API Overview
 *
 *  - All lookups are heterogeneous: you can provide any type comparable to the element type. Your comparator
 *    MUST define @code using is_transparent = void; @endcode to enable this, just like std::map and std::set.
 *  - No iterators are provided to keep the implementation simple and avoid complexity of maintaining persistent
 *    iterator validity across transactions and modifications.
 *  - All operations use callback-based APIs for consistency, with all callbacks expected to be @c noexcept.
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
 *
 *  @tparam basic_tree_type_ The underlying binary search tree type (basic_avl_tree, basic_wb_tree, etc.).
 *    Must provide @c rebind template alias for type transformations.
 */
template <typename basic_tree_type_>
class transactional_binary_tree {

  public:
#pragma mark - Type Definitions

    using element_t = typename basic_tree_type_::entry_t;
    using comparator_t = typename basic_tree_type_::comparator_t;
    using allocator_t = typename basic_tree_type_::allocator_t;

    using versioning_t = versioning_for<element_t, comparator_t>;
    using identifier_t = typename versioning_t::identifier_t;
    using generation_t = typename versioning_t::generation_t;
    using dated_identifier_t = typename versioning_t::dated_identifier_t;
    using watch_t = typename versioning_t::watch_t;
    using watched_identifier_t = typename versioning_t::watched_identifier_t;
    using versioned_entry_t = typename versioning_t::versioned_entry_t;
    using versioned_comparator_t = typename versioning_t::versioned_comparator_t;

  private:
    // Use tree's rebind to create versioned tree - clean 1-step type transformation!
    using versioned_entry_set_t = typename basic_tree_type_::template rebind<versioned_entry_t, versioned_comparator_t>;
    using versioned_entry_node_t = typename versioned_entry_set_t::node_t;
    using versioned_entry_iterator_t = versioned_entry_node_t *;

    using watches_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<watched_identifier_t>;
    using watches_vector_t = basic_vector<watched_identifier_t, watches_allocator_t>;

    using changed_ids_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<identifier_t>;
    using changed_ids_vector_t = basic_vector<identifier_t, changed_ids_allocator_t>;

    using store_t = transactional_binary_tree;
    using extract_result_t = typename versioned_entry_set_t::extract_result_t;

  public:
    class transaction_t {

        friend store_t;
        enum class stage_t {
            created_k,
            staged_k,
            commited_k,
        };

        store_t *store_ {nullptr};
        versioned_entry_set_t changed_entries_ {};
        watches_vector_t watches_ {};
        changed_ids_vector_t changed_ids_ {};
        generation_t generation_ {0};
        stage_t stage_ {stage_t::created_k};
        bool is_snapshot_ {false};

        transaction_t(store_t &set) noexcept : store_(&set), generation_(set.new_generation_()) {}
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
        bool has_changes() const noexcept { return changed_entries_.size() != 0; }

        /**
         *  @brief Returns the number of pending changes in this transaction.
         *  @return std::size_t The count of staged changes (including both upserts and erases).
         */
        std::size_t changed_entries_count() const noexcept { return changed_entries_.size(); }

      public:
        /**
         *  @brief Stages an insert operation only if the key doesn't exist. Fails if key exists.
         *    Checks both transaction changes and main store for existence.
         *
         *  @param[in] element Element to insert (moved into the transaction).
         *  @return status_t Success, or @c invalid_argument_k if key exists, or OOM error.
         */
        [[nodiscard]] status_t insert(element_t &&element) noexcept {
            // Check local changes first
            auto local_it = changed_entries_.find(element);
            if (local_it != changed_entries_.end() && !local_it->deleted) return {invalid_argument_k};

            // Check main store if not in local changes or was deleted locally
            if (store_ref().contains(element)) return {invalid_argument_k};

            // Key doesn't exist anywhere, proceed with insertion
            auto maybe_id = try_copy_identifier(id);
            if (!maybe_id) return status_t {out_of_memory_heap_k};

            auto reserve_status = changed_ids_.try_reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            versioned_entry_t entry;
            entry.element = std::move(element);
            entry.generation = generation_;
            entry.deleted = false;
            entry.visible = false;
            auto result = changed_entries_.upsert(std::move(entry));
            if (result.failed()) return status_t {out_of_memory_heap_k};

            changed_ids_.push_back(std::move(*maybe_id), assume_reserved);
            return status_t {success_k};
        }

        /**
         *  @brief Stages an insert operation only if key is missing. Silently skips if key exists (no error).
         *    Checks both transaction changes and main store for existence.
         *
         *  @param[in] element Element to insert (moved into the transaction).
         *  @return status_t Always succeeds (unless OOM). Returns success even if key exists.
         */
        [[nodiscard]] status_t insert_if_missing(element_t &&element) noexcept {
            // Check local changes first
            auto local_it = changed_entries_.find(element);
            if (local_it != changed_entries_.end() && !local_it->deleted) return {success_k};

            // Check main store if not in local changes or was deleted locally
            if (store_ref().contains(element)) return {success_k};

            // Key doesn't exist anywhere, proceed with insertion
            auto maybe_id = try_copy_identifier(id);
            if (!maybe_id) return status_t {out_of_memory_heap_k};

            auto reserve_status = changed_ids_.try_reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            versioned_entry_t entry;
            entry.element = std::move(element);
            entry.generation = generation_;
            entry.deleted = false;
            entry.visible = false;
            auto result = changed_entries_.upsert(std::move(entry));
            if (result.failed()) return status_t {out_of_memory_heap_k};

            changed_ids_.try_push_back(std::move(*maybe_id), assume_reserved);
            return status_t {success_k};
        }

        /**
         *  @brief Stages an insert or assign operation for the given element. Always succeeds.
         *    Overwrites existing element if key exists. Changes visible after @c stage() and @c commit().
         *
         *  @param[in] element Element to insert or assign (moved into the transaction).
         *  @return status_t Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t insert_or_assign(element_t &&element) noexcept {
            auto maybe_id = try_copy_identifier(element);
            if (!maybe_id) return status_t {out_of_memory_heap_k};

            auto reserve_status = changed_ids_.try_reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            versioned_entry_t entry;
            entry.element = std::move(element);
            entry.generation = generation_;
            entry.deleted = false;
            entry.visible = false;
            auto result = changed_entries_.upsert(std::move(entry));
            if (result.failed()) return status_t {out_of_memory_heap_k};

            changed_ids_.try_push_back(std::move(*maybe_id), assume_reserved);
            return status_t {success_k};
        }

        /**
         *  @brief Alias for @c insert_or_assign(). Stages an insert or assign operation.
         *  @param[in] element Element to insert or assign (moved into the transaction).
         *  @return status_t Success or error code (e.g., out of memory).
         */
        [[nodiscard]] status_t upsert(element_t &&element) noexcept { return insert_or_assign(std::move(element)); }

        [[nodiscard]] status_t erase(identifier_t const &id) noexcept {
            auto maybe_id = try_copy_identifier(id);
            if (!maybe_id) return status_t {out_of_memory_heap_k};

            auto reserve_status = changed_ids_.try_reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            versioned_entry_t entry;
            entry.element = element_t {id};
            entry.generation = generation_;
            entry.deleted = true;
            entry.visible = false;
            auto result = changed_entries_.upsert(std::move(entry));
            if (result.failed()) return status_t {out_of_memory_heap_k};

            changed_ids_.try_push_back(std::move(*maybe_id), assume_reserved);
            return status_t {success_k};
        }

        [[nodiscard]] status_t reserve(std::size_t size) noexcept { return watches_.try_reserve(size); }

        [[nodiscard]] status_t watch(identifier_t id) noexcept {
            status_t result {success_k};
            auto found = [&](versioned_entry_t const &entry) noexcept {
                result = watches_.try_push_back({std::move(id), watch_t {entry.generation, entry.deleted}});
            };
            auto missing = [&]() noexcept { result = watches_.try_push_back({std::move(id), missing_watch()}); };
            store_ref().find_visible_entry_(id, found, missing);
            return result;
        }

        [[nodiscard]] status_t watch(versioned_entry_t const &entry) noexcept {
            auto maybe_id = try_copy_identifier(identifier_t {entry.element});
            if (!maybe_id) return status_t {out_of_memory_heap_k};
            return watches_.try_push_back({std::move(*maybe_id), watch_t {entry.generation, entry.deleted}});
        }

        /**
         *  @brief Finds a member @b equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c transactional_binary_tree::find(), will include the entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
            if (auto iterator = changed_entries_.find(std::forward<comparable_type_>(comparable));
                iterator != changed_entries_.end())
                !iterator->deleted ? callback_found(*iterator) : callback_missing();
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
                std::forward<comparable_type_>(comparable), [&](auto const &) noexcept { found = true; },
                []() noexcept {});
            return found;
        }

        /**
         *  @brief Finds the first member @b greater or equal to the given @p comparable.
         *    You may want to @c watch() the received object, it's not done by default.
         *    Unlike @c transactional_binary_tree::lower_bound(), will include entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept {
            auto external_previous_id = identifier_t(comparable);
            auto internal_iterator = changed_entries_.lower_bound(std::forward<comparable_type_>(comparable));
            while (internal_iterator != changed_entries_.end() && internal_iterator->deleted) ++internal_iterator;

            // Once picking the next smallest element from the global store,
            // we might face an entry that was already deleted from here,
            // so this might become a multi-step process.
            auto faced_deleted_entry = false;
            auto callback_external_found = [&](element_t const &external_element) {
                // The simplest case is when we have an external object.
                if (internal_iterator == changed_entries_.end()) return callback_found(external_element);

                element_t const &internal_element = *internal_iterator;
                if (!versioned_comparator_t {}(external_element, internal_element))
                    return callback_found(internal_element);

                // Check if this entry was deleted and we should try again.
                auto external_id = identifier_t(external_element);
                auto external_element_internal_state = changed_entries_.find(external_element);
                if (external_element_internal_state != changed_entries_.end() &&
                    external_element_internal_state->deleted) {
                    faced_deleted_entry = true;
                    external_previous_id = external_id;
                }
                else callback_found(external_element);
            };
            auto callback_external_missing = [&] {
                if (internal_iterator == changed_entries_.end()) callback_missing();
                else callback_found(*internal_iterator);
            };

            // Iterate until we find a non-deleted external value
            auto &store = store_ref();
            do {
                faced_deleted_entry = false;
                store.lower_bound(external_previous_id, callback_external_found, callback_external_missing);
            } while (faced_deleted_entry);
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
        void equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
            find(
                std::forward<comparable_type_>(comparable),
                [&](versioned_entry_t const &entry) noexcept { callback(entry.element); }, []() noexcept {});
        }

        /**
         *  @brief Iterates over all entries in the range [ @p lower, @p upper), including transaction changes.
         *
         *  @param[in] lower Lower bound of the range (inclusive).
         *  @param[in] upper Upper bound of the range (exclusive).
         *  @param[in] callback Callback invoked for each element in range. Must be @c noexcept.
         */
        template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
                  typename callback_type_ = no_op_t>
        void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
            // First, iterate over local changes
            auto less = versioned_comparator_t {};
            auto lower_internal = changed_entries_.lower_bound(std::forward<lower_type_>(lower));
            auto const upper_internal_bound = identifier_t(upper);
            for (auto it = lower_internal; it != changed_entries_.end() && less(*it, upper_internal_bound); ++it)
                if (!it->deleted) callback(it->element);

            // Then, iterate over external store, skipping entries that were modified or deleted locally
            store_ref().range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                              [&](element_t const &external_element) {
                                  // Check if this entry exists in local changes
                                  auto local_state = changed_entries_.find(external_element);
                                  // Not modified locally, include it
                                  if (local_state == changed_entries_.end())
                                      callback(external_element); // Not modified locally
                                  // If modified locally, we already processed it above
                              });
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept {
            auto external_previous_id = identifier_t(comparable);
            auto internal_iterator = changed_entries_.upper_bound(std::forward<comparable_type_>(comparable));
            while (internal_iterator != changed_entries_.end() && internal_iterator->deleted) ++internal_iterator;

            // Once picking the next smallest element from the global store,
            // we might face an entry that was already deleted from here,
            // so this might become a multi-step process.
            auto faced_deleted_entry = false;
            auto callback_external_found = [&](element_t const &external_element) {
                // The simplest case is when we have an external object.
                if (internal_iterator == changed_entries_.end()) {
                    callback_found(external_element);
                    return;
                }

                element_t const &internal_element = *internal_iterator;
                if (!versioned_comparator_t {}(external_element, internal_element)) {
                    callback_found(internal_element);
                    return;
                }

                // Check if this entry was deleted and we should try again.
                auto external_id = identifier_t(external_element);
                auto external_element_internal_state = changed_entries_.find(external_element);
                if (external_element_internal_state != changed_entries_.end() &&
                    external_element_internal_state->deleted) {
                    faced_deleted_entry = true;
                    external_previous_id = external_id;
                }
                else { callback_found(external_element); }
            };
            auto callback_external_missing = [&] {
                if (internal_iterator == changed_entries_.end()) callback_missing();
                else callback_found(*internal_iterator);
            };

            // Iterate until we find the a non-deleted external value
            auto &store = store_ref();
            do {
                faced_deleted_entry = false;
                store.upper_bound(external_previous_id, callback_external_found, callback_external_missing);
            } while (faced_deleted_entry);
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
        template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
        void select(std::size_t k, callback_found_type_ &&callback_found,
                    callback_missing_type_ &&callback_missing = {}) const noexcept
            requires supports_order_statistics<versioned_entry_set_t>
        {
            // Collect all visible elements (both local changes and committed entries)
            std::size_t visible_index = 0;
            bool found = false;
            versioned_comparator_t less;

            // Build merged sorted view: iterate both trees in sorted order
            auto local_it = changed_entries_.begin();
            versioned_entry_node_t *store_node = versioned_entry_node_t::find_min(store_ref().entries_.root());

            while ((local_it != changed_entries_.end() || store_node) && !found) {
                // Determine which element comes next in sorted order
                bool take_local = false;

                if (local_it == changed_entries_.end()) { take_local = false; }
                else if (!store_node) { take_local = true; }
                else { take_local = less(local_it->element, store_node->entry.element); }

                if (take_local) {
                    // Process local change
                    if (!local_it->deleted) {
                        if (visible_index == k) {
                            callback_found(local_it->element);
                            found = true;
                        }
                        ++visible_index;
                    }
                    ++local_it;
                }
                else {
                    // Process store entry
                    identifier_t store_id {store_node->entry.element};
                    auto local_state = changed_entries_.find(store_id);

                    // Only count if visible and not overridden/deleted locally
                    if (store_node->entry.visible && !store_node->entry.deleted &&
                        local_state == changed_entries_.end()) {
                        if (visible_index == k) {
                            callback_found(store_node->entry.element);
                            found = true;
                        }
                        ++visible_index;
                    }

                    // Move to next store node
                    store_node = versioned_entry_node_t::find_successor(store_ref().entries_.root(), store_node,
                                                                        store_ref().entries_.key_comp());
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
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        void rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept
            requires supports_order_statistics<versioned_entry_set_t>
        {
            identifier_t target_id(comparable);
            std::size_t rank_value = 0;
            bool found = false;
            versioned_comparator_t less;

            // Check if target exists in local changes or main store
            auto local_target = changed_entries_.find(target_id);
            if (local_target != changed_entries_.end() && !local_target->deleted) { found = true; }
            else if (store_ref().contains(target_id)) { found = true; }

            if (!found) {
                callback_missing();
                return;
            }

            // Count visible elements less than target
            auto local_it = changed_entries_.begin();
            versioned_entry_node_t *store_node = versioned_entry_node_t::find_min(store_ref().entries_.root());

            while (local_it != changed_entries_.end() || store_node) {
                bool take_local = false;

                if (local_it == changed_entries_.end()) { take_local = false; }
                else if (!store_node) { take_local = true; }
                else { take_local = less(local_it->element, store_node->entry.element); }

                if (take_local) {
                    // Check if this local entry is less than target
                    if (!local_it->deleted && less(local_it->element, target_id)) ++rank_value;
                    ++local_it;
                }
                else {
                    identifier_t store_id {store_node->entry.element};
                    auto local_state = changed_entries_.find(store_id);

                    // Count if visible, not locally overridden, and less than target
                    if (store_node->entry.visible && !store_node->entry.deleted &&
                        local_state == changed_entries_.end() && less(store_node->entry.element, target_id)) {
                        ++rank_value;
                    }

                    store_node = versioned_entry_node_t::find_successor(store_ref().entries_.root(), store_node,
                                                                        store_ref().entries_.key_comp());
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
            // First, check if we have any collisions by validating watches.
            auto &store = store_ref();
            auto const entry_missing = missing_watch();
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

            // Merge our current nodes into the store. The visibility will be updated later in the `commit`.
            // We can mark all new entries as unique, because no other set of entries can have the same
            // generation as contents of this transaction.
            store.entries_.merge(changed_entries_, assume_unique);
            stage_ = stage_t::staged_k;
            return {success_k};
        }

        [[nodiscard]] status_t reset() noexcept {
            // If the transaction was "staged", we must delete all the entries.
            auto &store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const &id : changed_ids_) store.entries_.erase(dated_identifier_t {id, generation_});

            watches_.clear();
            changed_entries_.clear();
            changed_ids_.clear();
            stage_ = stage_t::created_k;
            generation_ = store.new_generation_();
            return {success_k};
        }

        [[nodiscard]] status_t rollback() noexcept {
            if (stage_ != stage_t::staged_k) return {operation_not_permitted_k};

            // Extract staged entries back into the transaction
            auto &store = store_ref();
            for (auto const &id : changed_ids_)
                changed_entries_.merge(store.entries_.extract(dated_identifier_t {id, generation_}));

            // Preserve watches_ for future stage operations (read watches persist across rollback)
            changed_ids_.clear();
            stage_ = stage_t::created_k;
            generation_ = store.new_generation_();
            return {success_k};
        }

        [[nodiscard]] status_t commit() noexcept {
            if (stage_ != stage_t::staged_k) return {operation_not_permitted_k};

            // Once we make an entry visible, if there are more than one with the same key,
            // the older generation must die.
            auto &store = store_ref();
            for (auto const &id : changed_ids_) store.unmask_and_compact_(id, generation_);

            changed_ids_.clear();
            stage_ = stage_t::created_k;
            return {success_k};
        }
    };

  private:
    versioned_entry_set_t entries_;
    generation_t generation_ {0};
    std::size_t visible_count_ {0};
    std::size_t visible_deleted_count_ {0};

    friend class transaction_t;
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

        versioned_entry_node_t *largest_visible = nullptr;
        versioned_entry_node_t::range(
            entries_.root(), comparable, comparable, entries_.key_comp(), [&](versioned_entry_node_t *node) noexcept {
                if ((node->entry.visible) &&
                    (!largest_visible || node->entry.generation > largest_visible->entry.generation))
                    largest_visible = node;
            });

        static_assert(is_safe_callback_for<callback_found_type_, versioned_entry_t const &>,
                      "callback_found must be noexcept invocable with versioned_entry_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");
        if (largest_visible) callback_found(largest_visible->entry);
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

        versioned_entry_node_t *latest = nullptr;
        versioned_entry_node_t::range(entries_.root(), comparable, comparable, entries_.key_comp(),
                                      [&](versioned_entry_node_t *node) noexcept {
                                          // Find HIGHEST generation, regardless of visibility
                                          if (!latest || node->entry.generation > latest->entry.generation)
                                              latest = node;
                                      });

        if (latest && !latest->entry.deleted) callback_found(latest->entry);
        else callback_missing();
    }

    void unmask_and_compact_(identifier_t const &id, generation_t generation_to_unmask) noexcept {
        // This is similar to the public `erase_range()`, but adds generation-matching conditions.
        auto current = entries_.lower_bound(id);
        if (current == entries_.end()) return;

        auto less = versioned_comparator_t {};
        auto last_visible_entry = std::optional<dated_identifier_t> {};
        while (current != entries_.end() && less.same(id, (*current).element)) {
            auto next = entries_.upper_bound(*current);
            auto was_visible = (*current).visible;
            (*current).visible |= (*current).generation == generation_to_unmask;

            // Update counters if visibility changed
            if (!was_visible && (*current).visible) {
                ++visible_count_;
                visible_deleted_count_ += (*current).deleted;
            }

            if (!(*current).visible) {
                current = next;
                continue;
            }

            // Older revisions must die
            if (last_visible_entry) {
                auto to_erase = entries_.find(*last_visible_entry);
                if (to_erase != entries_.end() && (*to_erase).visible) {
                    --visible_count_;
                    visible_deleted_count_ -= (*to_erase).deleted;
                }
                entries_.extract(*last_visible_entry);
            }
            last_visible_entry = dated_identifier_t {id, (*current).generation};
            current = next;
        }
    }

#pragma mark - Constructors and Assignment

  public:
    transactional_binary_tree() noexcept {}
    transactional_binary_tree(transactional_binary_tree &&other) noexcept
        : entries_(std::move(other.entries_)), generation_(other.generation_), visible_count_(other.visible_count_),
          visible_deleted_count_(other.visible_deleted_count_) {}

    transactional_binary_tree &operator=(transactional_binary_tree &&other) noexcept {
        entries_ = std::move(other.entries_);
        generation_ = other.generation_;
        visible_count_ = other.visible_count_;
        visible_deleted_count_ = other.visible_deleted_count_;
        return *this;
    }

#pragma mark - Capacity

    /**
     *  @brief Returns the number of visible (committed) non-deleted elements in the tree.
     *  @return std::size_t Number of elements.
     */
    [[nodiscard]] std::size_t size() const noexcept { return visible_count_ - visible_deleted_count_; }

    /**
     *  @brief Checks if the tree has no visible elements.
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
        return contains(std::forward<comparable_type_>(comparable)) ? 1 : 0;
    }

    /**
     *  @brief Checks if a member @b equal to the given @p comparable exists in the tree.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @return bool True if element exists, false otherwise.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
        bool found = false;
        find(std::forward<comparable_type_>(comparable), [&](element_t const &) noexcept { found = true; });
        return found;
    }

#pragma mark - Observers

    /**
     *  @brief Factory method to create a new transactional binary tree without throwing exceptions.
     *    Returns an empty optional on allocation failure.
     *
     *  @param[in] allocator Optional allocator instance.
     *  @return std::optional<store_t> Container instance or empty optional on failure.
     */
    [[nodiscard]] static std::optional<store_t> make(allocator_t &&allocator = {}) noexcept { return store_t {}; }

#pragma mark - Transaction Management

    /**
     *  @brief Creates a new transaction with a fresh generation number.
     *    Transaction can be reset and reused after commit/rollback to avoid reallocations.
     *    Returns empty optional on allocation failure.
     *
     *  @return std::optional<transaction_t> Transaction instance or empty optional on failure.
     */
    [[nodiscard]] std::optional<transaction_t> transaction() noexcept { return transaction_t {*this}; }

#pragma mark - Modifiers

    /**
     *  @brief Atomically inserts an element only if the key doesn't exist. Fails if key exists.
     *    This is the strict insert semantics matching @c std::set::insert().
     *
     *  @param[in] element Element to insert (moved into the tree).
     *  @return status_t Success, or @c invalid_argument_k if key exists, or OOM error.
     */
    [[nodiscard]] status_t insert(element_t &&element) noexcept {
        if (contains(element)) return {invalid_argument_k};
        return insert_or_assign(std::move(element));
    }

    /**
     *  @brief Atomically inserts an element only if missing. Silently skips if key exists (no error).
     *    This is the "silent no-op" insert semantics.
     *
     *  @param[in] element Element to insert (moved into the tree).
     *  @return status_t Always succeeds (unless OOM). Returns success even if key exists.
     */
    [[nodiscard]] status_t insert_if_missing(element_t &&element) noexcept {
        if (contains(element)) return {success_k};
        return insert_or_assign(std::move(element));
    }

    /**
     *  @brief Atomically inserts or updates an element. Always succeeds (unless OOM).
     *    Overwrites existing element if key exists. Matches @c std::map::insert_or_assign() semantics.
     *
     *  @param[in] element Element to insert or assign (moved into the tree).
     *  @return status_t Success or error code (e.g., out of memory).
     */
    [[nodiscard]] status_t insert_or_assign(element_t &&element) noexcept {
        auto node = entries_.allocator().allocate(1);
        if (!node) return {out_of_memory_heap_k};

        generation_t generation = new_generation_();
        auto &entry = node->entry;
        new (&entry.element) element_t(std::move(element));
        entry.generation = generation;
        entry.deleted = false;
        entry.visible = true;
        entries_.merge(extract_result_t {&entries_, node});
        ++visible_count_;
        assert(!entry.deleted && "entry.deleted is always false here, otherwise update visible_deleted_count_");

        erase_range(entry.element, dated_identifier_t {entry.element, generation});
        return {success_k};
    }

    /**
     *  @brief Alias for @c insert_or_assign(). Atomically inserts or updates an element.
     *  @param[in] element Element to insert or assign (moved into the tree).
     *  @return status_t Success or error code (e.g., out of memory).
     */
    [[nodiscard]] status_t upsert(element_t &&element) noexcept { return insert_or_assign(std::move(element)); }

    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {

        // To make such batch insertions cheaper and easier until we have fast joins,
        // we can build a linked-list of pre-allocated nodes. Populate them and insert
        // one-by-one with the same generation.
        std::size_t const count = end - begin;
        std::size_t count_remaining = count;
        versioned_entry_node_t *last_node = nullptr;
        while (count_remaining) {
            versioned_entry_node_t *next_node = entries_.allocator().allocate(1);
            if (!next_node) break;
            // Reset the state
            next_node->right = nullptr;
            // Link for future iteration
            if (last_node) last_node->right = next_node;
            next_node->left = last_node;
            // Update state for next loop cycle
            last_node = next_node;
            count_remaining--;
        }

        // We have failed to allocate all the needed nodes.
        if (count_remaining) {
            while (count_remaining != count) {
                versioned_entry_node_t *prev_node = last_node->left;
                entries_.allocator().deallocate(last_node, 1);
                // Update state for next loop cycle
                last_node = prev_node;
                ++count_remaining;
            }
            return {out_of_memory_heap_k};
        }

        // Populate the allocated nodes and merge into the tree.
        generation_t generation = new_generation_();
        while (count_remaining != count) {
            versioned_entry_node_t *prev_node = last_node->left;
            last_node->left = nullptr;
            last_node->right = nullptr;

            auto &entry = last_node->entry;
            new (&entry.element) element_t(*begin);
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            entries_.merge(extract_result_t {&entries_, last_node});
            ++visible_count_;
            assert(!entry.deleted && "entry.deleted is always false here, otherwise update visible_deleted_count_");

            // Remove older revisions
            identifier_t id {entry.element};
            erase_range(id, dated_identifier_t {id, generation});

            // Update state for next loop cycle
            last_node = prev_node;
            ++count_remaining;
            ++begin;
        }

        return {success_k};
    }

#pragma mark - Lookup

    /**
     *  @brief Finds a member @b equal to the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept {

        find_visible_entry_(
            std::forward<comparable_type_>(comparable),
            [&](versioned_entry_t const &entry) noexcept { callback_found(entry.element); },
            std::forward<callback_missing_type_>(callback_missing));
    }

    /**
     *  @brief Finds the first member @b greater or equal to the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, element_t const &>,
                      "callback_found must be noexcept invocable with element_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        // Skip all the invisible entries
        versioned_entry_node_t *next_visible =
            versioned_entry_node_t::lower_bound(entries_.root(), comparable, entries_.key_comp());
        while (next_visible && !next_visible->entry.visible)
            next_visible =
                versioned_entry_node_t::upper_bound(entries_.root(), next_visible->entry, entries_.key_comp());

        if (next_visible) callback_found(next_visible->entry.element);
        else callback_missing();
    }

    /**
     *  @brief Finds all elements equal to a single key. Invokes callback for each matching element.
     *    For trees with unique keys, this returns at most one element (0 or 1).
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
     */
    template <typename comparable_type_ = identifier_t, typename callback_type_ = no_op_t>
    void equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
        find(std::forward<comparable_type_>(comparable), std::forward<callback_type_>(callback), []() noexcept {});
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                     callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, element_t const &>,
                      "callback_found must be noexcept invocable with element_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        // Skip all the invisible entries
        versioned_entry_node_t *next_visible =
            versioned_entry_node_t::upper_bound(entries_.root(), comparable, entries_.key_comp());
        while (next_visible && !next_visible->entry.visible)
            next_visible =
                versioned_entry_node_t::upper_bound(entries_.root(), next_visible->entry, entries_.key_comp());

        if (next_visible) callback_found(next_visible->entry.element);
        else callback_missing();
    }

#pragma mark - Range Operations

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        versioned_entry_node_t::range(entries_.root(), std::forward<lower_type_>(lower),
                                      std::forward<upper_type_>(upper), entries_.key_comp(),
                                      [&](versioned_entry_node_t *node) noexcept {
                                          if (node->entry.visible) callback(node->entry.element);
                                      });
    }

    /**
     *  @brief Iterates over key-value associations in [ @p lower, @p upper), providing mutable value access.
     *    Only enabled for association types. Callback receives (const key_type&, value_type&).
     *    Updates generation for each accessed element.
     *
     *  @param[in] lower Lower bound (inclusive).
     *  @param[in] upper Upper bound (exclusive).
     *  @param[in] callback Callback invoked with (const Key&, Value&) for each element. Must be @c noexcept.
     */
    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    void update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires is_association<element_t>
    {
        generation_t generation = new_generation_();
        versioned_entry_node_t::range(entries_.root(), std::forward<lower_type_>(lower),
                                      std::forward<upper_type_>(upper), entries_.key_comp(),
                                      [&](versioned_entry_node_t *node) noexcept {
                                          if (!node->entry.visible) return;
                                          callback(node->entry.element.key, node->entry.element.value);
                                          node->entry.generation = generation;
                                      });
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
        // Use split/join for O(log n + k) complexity instead of O(k log n)
        // Split at lower bound
        auto first_split = entries_.split(std::forward<lower_type_>(lower));

        // Split the right part at upper bound to isolate the range
        auto second_split = first_split.right.split(std::forward<upper_type_>(upper));

        // middle_tree contains elements in [lower, upper) that need to be deleted
        // Traverse it to invoke callbacks and update counters
        versioned_entry_node_t::for_each_left_right(second_split.left.root(),
                                                    [&](versioned_entry_node_t *node) noexcept {
                                                        if (node->entry.visible) {
                                                            callback(node->entry.element);
                                                            --visible_count_;
                                                            visible_deleted_count_ -= node->entry.deleted;
                                                        }
                                                    });

        // Deallocate all nodes in the middle tree (second_split.left)
        second_split.left.clear();

        // Join the left and far_right trees back together
        first_split.left.join(second_split.right);

        // Move the result back to entries_
        entries_ = std::move(first_split.left);

        return status_t {success_k};
    }

#pragma mark - Sampling

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                      callback_type_ &&callback) const noexcept {

        auto node = versioned_entry_node_t::sample_range( //
            entries_.root(), lower, upper, entries_.key_comp(), std::forward<generator_type_>(generator),
            [](versioned_entry_node_t *node) noexcept { return node->entry.visible; });
        if (node) callback(node->entry);
    }

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

#pragma mark - Order Statistics

    /**
     *  @brief Finds the k-th smallest visible (committed) element using order statistics.
     *    Only available for tree implementations that support order statistics (e.g., WB trees).
     *    Requires O(log n) time for weight-balanced trees.
     *
     *  @param[in] k Zero-based index (0 = smallest element).
     *  @param[in] callback_found Callback to receive the k-th element. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if k >= size(). Must be @c noexcept.
     */
    template <typename callback_found_type_ = no_op_t, typename callback_missing_type_ = no_op_t>
    void select(std::size_t k, callback_found_type_ &&callback_found,
                callback_missing_type_ &&callback_missing = {}) const noexcept
        requires supports_order_statistics<versioned_entry_set_t>
    {
        // Count visible entries until we reach the k-th one
        std::size_t visible_index = 0;
        bool found = false;

        // Iterate in sorted order, counting only visible entries
        versioned_entry_node_t::for_each_left_right(entries_.root(), [&](versioned_entry_node_t *node) noexcept {
            if (!node->entry.visible || node->entry.deleted) return;
            if (visible_index == k) {
                callback_found(node->entry.element);
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
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void rank(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
              callback_missing_type_ &&callback_missing = {}) const noexcept
        requires supports_order_statistics<versioned_entry_set_t>
    {
        // Count visible entries before the target
        std::size_t rank_value = 0;
        bool found = false;
        identifier_t target_id(comparable);
        versioned_comparator_t less;

        versioned_entry_node_t::for_each_left_right(entries_.root(), [&](versioned_entry_node_t *node) noexcept {
            if (!node->entry.visible || node->entry.deleted) return;

            // Check if this is our target
            if (less.same(node->entry.element, target_id)) {
                callback_found(rank_value);
                found = true;
                return;
            }

            // If this node is less than target, increment rank
            if (less(node->entry.element, target_id)) ++rank_value;
        });

        if (!found) callback_missing();
    }

    /**
     *  @brief Erases a single entry matching the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive the erased entry. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     *  @return status_t Always succeeds.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    status_t erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found = {},
                   callback_missing_type_ &&callback_missing = {}) noexcept {
        // Find the entry
        bool found = false;
        find(
            std::forward<comparable_type_>(comparable),
            [&](versioned_entry_t const &entry) noexcept {
                found = true;
                callback_found(entry.element);
            },
            []() noexcept {});

        if (!found) {
            callback_missing();
            return status_t {success_k};
        }

        // Erase the entry
        erase_range(std::forward<comparable_type_>(comparable),
                    dated_identifier_t {identifier_t(comparable), generation_ + 1});
        return status_t {success_k};
    }

    /**
     *  @brief Hints to the tree to pre-allocate memory. No-op for tree implementations.
     *    Provided for API consistency with other containers. Doesn't guarantee subsequent
     *    insertions won't fail with "out of memory".
     *
     *  @param[in] size Suggested capacity (ignored for tree structures).
     *  @return status_t Always succeeds.
     *
     *  @note This is a no-op because tree structures don't support reserving capacity efficiently.
     */
    [[nodiscard]] status_t reserve(std::size_t) noexcept { return {success_k}; }

    /**
     *  @brief Removes all elements from the tree and resets generation counter.
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
     *  @brief Debug utility to print tree contents.
     *  @note Requires @c #include <ostream> (not included by default to reduce header weight)
     */
    template <typename dont_instantiate_me_type_>
    void print(dont_instantiate_me_type_ &cout) {
        cout << "Items: " << entries_.size() << "\n";
        cout << "Imbalance: " << entries_.total_imbalance() << "\n";
        versioned_entry_node_t::for_each_left_right(entries_.root(), [&](versioned_entry_node_t *node) {
            char const *marker = node->entry.visible ? "✓" : "✗";
            cout << identifier_t {node->entry.element} << " @" << node->entry.generation << marker << " ";
        });
        cout << "\n";
    }
};

/**
 *  @brief STL-style transactional set using AVL tree.
 *    Stores unique elements in sorted order with ACID transaction semantics.
 *
 *  @tparam element_type_ Type of elements stored in the set.
 *  @tparam comparator_type_ Comparator for ordering elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename element_type_, typename comparator_type_ = std::less<element_type_>,
          typename allocator_type_ = std::allocator<element_type_>>
using transactional_avl_set =
    transactional_binary_tree<basic_avl_tree<element_type_, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional map using AVL tree.
 *    Stores key-value pairs in sorted order with ACID transaction semantics.
 *
 *  @tparam key_type_ Type of keys stored in the map.
 *  @tparam value_type_ Type of values stored in the map.
 *  @tparam comparator_type_ Comparator for ordering keys. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename key_type_, typename value_type_, typename comparator_type_ = std::less<key_type_>,
          typename allocator_type_ = std::allocator<association<key_type_, value_type_>>>
using transactional_avl_map =
    transactional_binary_tree<basic_avl_tree<association<key_type_, value_type_>, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional set using weight-balanced tree with order statistics support.
 *    Stores unique elements in sorted order with ACID transaction semantics and O(log n) rank/select operations.
 *
 *  @tparam element_type_ Type of elements stored in the set.
 *  @tparam comparator_type_ Comparator for ordering elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename element_type_, typename comparator_type_ = std::less<element_type_>,
          typename allocator_type_ = std::allocator<element_type_>>
using transactional_wb_set = transactional_binary_tree<basic_wb_tree<element_type_, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional map using weight-balanced tree with order statistics support.
 *    Stores key-value pairs in sorted order with ACID transaction semantics and O(log n) rank/select operations.
 *
 *  @tparam key_type_ Type of keys stored in the map.
 *  @tparam value_type_ Type of values stored in the map.
 *  @tparam comparator_type_ Comparator for ordering keys. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename key_type_, typename value_type_, typename comparator_type_ = std::less<key_type_>,
          typename allocator_type_ = std::allocator<association<key_type_, value_type_>>>
using transactional_wb_map =
    transactional_binary_tree<basic_wb_tree<association<key_type_, value_type_>, comparator_type_, allocator_type_>>;

} // namespace ashvardanian::smashtable
