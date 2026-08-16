/**
 *  @brief Generic transactional binary tree container with ACID semantics, providing 2-phase commit transactions. Can
 *      be instantiated with any binary search tree implementation (AVL, WB, etc.) supporting the required interface.
 *      All operations use callback-based APIs and are exception-free via @c noexcept constraints.
 *  @author Ash Vardanian
 *  @file include/smashtable/transactional_binary_tree.hpp
 *  @date October 13, 2022
 */
#pragma once
#include <cassert> // `assert`

#include <algorithm> // `std::max`
#include <memory>    // `std::allocator`
#include <optional>  // `std::optional`
#include <random>    // `std::uniform_int_distribution`
#include <utility>   // `std::exchange`

#include "basic_avl_tree.hpp"
#include "basic_vector.hpp"
#include "basic_wb_tree.hpp"
#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief  Generic transactional binary tree providing 2-phase commits and "watch" operations.
 *    Can be instantiated with AVL trees, weight-balanced trees, or other binary search tree implementations.
 *    Not thread-safe by itself. Entirely exception-free, with all methods marked @c noexcept.
 *
 *  @section transactional_binary_tree_design_goals Design Goals
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
 *  @section transactional_binary_tree_api_overview API Overview
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
 *  @tparam basic_tree_type_ The underlying binary search tree type (basic_avl_tree, basic_wb_tree, etc.).
 *    Must provide @c rebind template alias for type transformations.
 */
template <typename basic_tree_type_>
class transactional_binary_tree {

  public:
#pragma region Type Definitions

    using value_t = typename basic_tree_type_::value_type;
    using value_type = value_t; // ? STL style

    using key_t = typename basic_tree_type_::key_type;
    using key_type = key_t; // ? STL style

    using mapped_t = typename basic_tree_type_::mapped_type;
    using mapped_type = mapped_t; // ? STL style

    using is_associative = typename basic_tree_type_::is_associative;
    using is_transactional = std::true_type;
    using callback_reads = std::true_type;

    using comparator_t = typename basic_tree_type_::comparator_t;
    using allocator_t = typename basic_tree_type_::allocator_t;

    using versioning_t = versioning_for<value_t, comparator_t>;
    using identifier_t = typename versioning_t::identifier_t;
    using generation_t = typename versioning_t::generation_t;
    using watch_t = typename versioning_t::watch_t;
    using watched_identifier_t = typename versioning_t::watched_identifier_t;
    using dated_identifier_t = typename versioning_t::dated_identifier_t;
    using versioned_t = typename versioning_t::versioned_t;
    using versioned_entry_t = versioned_t;

  private:
    using versioned_comparator_t = typename versioning_t::versioned_comparator_t;

    // Use tree's rebind to create versioned tree - clean 1-step type transformation!
    using versioned_set_t = typename basic_tree_type_::template rebind<versioned_t, versioned_comparator_t>;

    using watches_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<watched_identifier_t>;
    using watches_vector_t = basic_vector<watched_identifier_t, watches_allocator_t>;

    using changed_ids_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<identifier_t>;
    using changed_ids_vector_t = basic_vector<identifier_t, changed_ids_allocator_t>;

    using store_t = transactional_binary_tree;
    using extract_result_t = typename versioned_set_t::extract_result_t;

  public:
    class transaction_t {

        friend store_t;
        enum class stage_t {
            created_k,
            staged_k,
            commited_k,
        };

        store_t *store_ {nullptr};
        versioned_set_t changes_ {};
        watches_vector_t watches_ {};
        changed_ids_vector_t changed_ids_ {};
        generation_t generation_ {0};
        stage_t stage_ {stage_t::created_k};
        bool is_snapshot_ {false};

        transaction_t(store_t &set) noexcept
            : store_(&set), changes_(set.entries_.key_comp(), set.entries_.allocator()),
              watches_(watches_allocator_t(set.entries_.allocator())),
              changed_ids_(changed_ids_allocator_t(set.entries_.allocator())), generation_(set.new_generation_()) {}
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
            if (local_it != changes_.end() && !local_it->deleted) return {key_already_exists_k};
            if (store_ref().contains(value)) return {key_already_exists_k};

            auto maybe_id = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
            if (!maybe_id) return status_t {out_of_memory_heap_k};
            auto reserve_status = changed_ids_.reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            versioned_t versioned(std::move(value));
            versioned.generation = generation_;
            versioned.deleted = false;
            versioned.visible = false;
            auto result = changes_.upsert(std::move(versioned));
            if (result.failed()) return status_t {out_of_memory_heap_k};

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
            if (local_it != changes_.end() && !local_it->deleted) return {success_k};
            if (store_ref().contains(value)) return {success_k};

            auto maybe_id = copy_safely<identifier_t>(mapping_key_or_itself<value_t>(value));
            if (!maybe_id) return status_t {out_of_memory_heap_k};
            auto reserve_status = changed_ids_.reserve(changed_ids_.size() + 1);
            if (!reserve_status) return reserve_status;

            versioned_t versioned(std::move(value));
            versioned.generation = generation_;
            versioned.deleted = false;
            versioned.visible = false;
            auto result = changes_.upsert(std::move(versioned));
            if (result.failed()) return status_t {out_of_memory_heap_k};

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
            versioned.deleted = false;
            versioned.visible = false;
            auto result = changes_.upsert(std::move(versioned));
            if (result.failed()) return status_t {out_of_memory_heap_k};

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
            if (local_it != changes_.end() && !local_it->deleted) return upsert(std::move(value));
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

            versioned_t versioned(value_t {id});
            versioned.generation = generation_;
            versioned.deleted = true;
            versioned.visible = false;
            auto result = changes_.upsert(std::move(versioned));
            if (result.failed()) return status_t {out_of_memory_heap_k};

            changed_ids_.push_back(assume_reserved, std::move(*maybe_id));
            return status_t {success_k};
        }

        [[nodiscard]] status_t reserve(std::size_t size) noexcept { return watches_.reserve(size); }

        [[nodiscard]] status_t watch(identifier_t id) noexcept {
            status_t result {success_k};
            auto found = [&](versioned_t const &versioned) noexcept {
                result = watches_.push_back({std::move(id), watch_t {versioned.generation, versioned.deleted}});
            };
            auto missing = [&]() noexcept { result = watches_.push_back({std::move(id), missing_watch()}); };
            store_ref().find_visible_entry_(id, found, missing);
            return result;
        }

        [[nodiscard]] status_t watch(versioned_t const &versioned) noexcept {
            auto maybe_id = copy_safely<identifier_t>(identifier_t {versioned.unversioned});
            if (!maybe_id) return status_t {out_of_memory_heap_k};
            return watches_.push_back({std::move(*maybe_id), watch_t {versioned.generation, versioned.deleted}});
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
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                  callback_missing_type_ &&callback_missing = {}) const noexcept {
            if (auto iterator = changes_.find(std::forward<comparable_type_>(comparable)); iterator != changes_.end())
                !iterator->deleted ? callback_found(iterator->unversioned) : callback_missing();
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

            if (auto it = changes_.find(std::forward<comparable_type_>(comparable)); it != changes_.end()) {
                if (!it->deleted) {
                    auto copy_result = copy_safely(it->unversioned);
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
         *    Unlike @c transactional_binary_tree::lower_bound(), will include entries added to this transaction.
         *
         *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
         *  @param[in] callback_found Callback to receive an @c element_t @c const @c &. Must be @c noexcept.
         *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
         */
        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void lower_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept {

            auto store_lb = store_ref().entries_.lower_bound(comparable);
            while (store_lb != store_ref().entries_.end() && changes_.contains(*store_lb)) { ++store_lb; }

            auto changed_lb = changes_.lower_bound(comparable);
            while (changed_lb != changes_.end() && changed_lb->deleted) { ++changed_lb; }

            if (store_lb == store_ref().entries_.end() && changed_lb == changes_.end()) { callback_missing(); }
            else if (store_lb == store_ref().entries_.end()) { callback_found(changed_lb->unversioned); }
            else if (changed_lb == changes_.end()) { callback_found(store_lb->unversioned); }
            else {
                if (versioned_comparator_t {}(store_lb->unversioned, changed_lb->unversioned)) {
                    callback_found(store_lb->unversioned);
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
        void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
            auto less = versioned_comparator_t {};
            auto lower_internal = changes_.lower_bound(std::forward<lower_type_>(lower));
            auto const upper_internal_bound = identifier_t(upper);
            for (auto it = lower_internal; it != changes_.end() && less(*it, upper_internal_bound); ++it)
                if (!it->deleted) callback(it->unversioned);

            store_ref().range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                              [&](value_t const &external_value) {
                                  auto local_state = changes_.find(external_value);
                                  if (local_state == changes_.end()) callback(external_value);
                              });
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_fn_t,
                  typename callback_missing_type_ = no_op_fn_t>
        void upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                         callback_missing_type_ &&callback_missing = {}) const noexcept {

            auto store_ub = store_ref().entries_.upper_bound(comparable);
            while (store_ub != store_ref().entries_.end() && changes_.contains(*store_ub)) { ++store_ub; }

            auto changed_ub = changes_.upper_bound(comparable);
            while (changed_ub != changes_.end() && changed_ub->deleted) { ++changed_ub; }

            if (store_ub == store_ref().entries_.end() && changed_ub == changes_.end()) { callback_missing(); }
            else if (store_ub == store_ref().entries_.end()) { callback_found(changed_ub->unversioned); }
            else if (changed_ub == changes_.end()) { callback_found(store_ub->unversioned); }
            else {
                if (versioned_comparator_t {}(store_ub->unversioned, changed_ub->unversioned)) {
                    callback_found(store_ub->unversioned);
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
            requires supports_order_statistics<versioned_set_t>
        {
            std::size_t visible_index = 0;
            bool found = false;
            versioned_comparator_t less;

            auto local_it = changes_.begin();
            using node_t = typename versioned_set_t::node_t;
            node_t *store_node = node_t::find_min(store_ref().entries_.root());

            while ((local_it != changes_.end() || store_node) && !found) {
                bool take_local = false;

                if (local_it == changes_.end()) { take_local = false; }
                else if (!store_node) { take_local = true; }
                else { take_local = less(local_it->unversioned, store_node->fruit); }

                if (take_local) {
                    if (!local_it->deleted) {
                        if (visible_index == k) {
                            callback_found(local_it->unversioned);
                            found = true;
                        }
                        ++visible_index;
                    }
                    ++local_it;
                }
                else {
                    identifier_t store_id {store_node->fruit};
                    auto local_state = changes_.find(store_id);

                    if (store_node->fruit.visible && !store_node->fruit.deleted && local_state == changes_.end()) {
                        if (visible_index == k) {
                            callback_found(store_node->fruit.unversioned);
                            found = true;
                        }
                        ++visible_index;
                    }

                    store_node = node_t::find_successor(store_node);
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
            requires supports_order_statistics<versioned_set_t>
        {
            identifier_t target_id(comparable);
            std::size_t rank_value = 0;
            bool found = false;
            versioned_comparator_t less;

            auto local_target = changes_.find(target_id);
            if (local_target != changes_.end() && !local_target->deleted) { found = true; }
            else if (store_ref().contains(target_id)) { found = true; }

            if (!found) {
                callback_missing();
                return;
            }

            auto local_it = changes_.begin();
            using node_t = typename versioned_set_t::node_t;
            node_t *store_node = node_t::find_min(store_ref().entries_.root());

            while (local_it != changes_.end() || store_node) {
                bool take_local = false;

                if (local_it == changes_.end()) { take_local = false; }
                else if (!store_node) { take_local = true; }
                else { take_local = less(local_it->unversioned, store_node->fruit); }

                if (take_local) {
                    if (!local_it->deleted && less(local_it->unversioned, target_id)) ++rank_value;
                    ++local_it;
                }
                else {
                    identifier_t store_id {store_node->fruit};
                    auto local_state = changes_.find(store_id);

                    if (store_node->fruit.visible && !store_node->fruit.deleted && local_state == changes_.end() &&
                        less(store_node->fruit.unversioned, target_id)) {
                        ++rank_value;
                    }

                    store_node = node_t::find_successor(store_node);
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

            // Merge our current nodes into the store. The visibility will be updated later in the `commit`.
            // We can mark all new entries as unique, because no other set of entries can have the same
            // generation as contents of this transaction.
            store.entries_.merge(changes_, assume_unique);
            stage_ = stage_t::staged_k;
            return {success_k};
        }

        [[nodiscard]] status_t reset() noexcept {
            auto &store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (identifier_t const &id : changed_ids_)
                    // ! Don't materialize a new copy of `id` here, use a reference
                    store.entries_.erase(dated_identifier<identifier_t const &> {id, generation_});

            watches_.clear();
            changes_.clear();
            changed_ids_.clear();
            stage_ = stage_t::created_k;
            generation_ = store.new_generation_();
            return {success_k};
        }

        [[nodiscard]] status_t rollback() noexcept {
            if (stage_ != stage_t::staged_k) return {operation_not_permitted_k};

            auto &store = store_ref();
            for (identifier_t const &id : changed_ids_)
                // ! Don't materialize a new copy of `id` here, use a reference
                changes_.merge(store.entries_.extract(dated_identifier<identifier_t const &> {id, generation_}));

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
    versioned_set_t entries_;
    generation_t generation_ {0};
    std::size_t visible_count_ {0};
    std::size_t visible_deleted_count_ {0};

    friend class transaction_t;
    generation_t new_generation_() noexcept { return ++generation_; }

    /**
     *  @brief Internal API: Finds the latest visible entry and invokes callback with @c versioned_t const &.
     *    Used by internal methods that need access to generation/deleted/visible fields.
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

        using node_t = typename versioned_set_t::node_t;
        node_t *largest_visible = nullptr;
        node_t::range(entries_.root(), comparable, comparable, entries_.key_comp(), [&](node_t *node) noexcept {
            if ((node->fruit.visible) &&
                (!largest_visible || node->fruit.generation > largest_visible->fruit.generation))
                largest_visible = node;
        });

        static_assert(is_safe_callback_for<callback_found_type_, versioned_t const &>,
                      "callback_found must be noexcept invocable with versioned_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");
        if (largest_visible) callback_found(largest_visible->fruit);
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

        using node_t = typename versioned_set_t::node_t;
        node_t *latest = nullptr;
        node_t::range(entries_.root(), comparable, comparable, entries_.key_comp(), [&](node_t *node) noexcept {
            if (!latest || node->fruit.generation > latest->fruit.generation) latest = node;
        });

        if (latest && !latest->fruit.deleted) callback_found(latest->fruit);
        else callback_missing();
    }

    void unmask_and_compact_(identifier_t const &id, generation_t generation_to_unmask) noexcept {
        std::size_t removed_deleted = 0;
        auto should_remove = [&](versioned_t const &versioned) noexcept {
            if (!versioned_comparator_t {}.same(versioned.unversioned, id)) return false;
            if (versioned.generation == generation_to_unmask) return false;
            if (!versioned.visible) return false;
            removed_deleted += versioned.deleted;
            return true;
        };

        auto removed_count = entries_.remove_if(should_remove);
        visible_count_ -= removed_count;
        visible_deleted_count_ -= removed_deleted;

        // ! Don't materialize a new copy of `id` here, use a reference
        auto it = entries_.find(dated_identifier<identifier_t const &> {id, generation_to_unmask});
        if (it != entries_.end()) {
            auto &versioned = const_cast<versioned_t &>(*it);
            if (!versioned.visible) {
                versioned.visible = true;
                ++visible_count_;
                if (versioned.deleted) visible_deleted_count_++;
            }
        }
    }

#pragma endregion Type Definitions

#pragma region Constructors and Assignment

  public:
    transactional_binary_tree() noexcept {}

    /** @brief Seeds the underlying tree's allocator, which a stateful allocator needs. */
    explicit transactional_binary_tree(allocator_t const &allocator) noexcept
        : entries_(typename versioned_set_t::allocator_t(allocator)) {}
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
        return find_copy(std::forward<comparable_type_>(comparable), entries_.allocator());
    }

    /**
     *  @brief Finds and returns a copy of the first element not less than (>=) @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Result with copied element if found, or failure status.
     */
    template <typename comparable_type_ = identifier_t>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept {
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
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept {
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
     */
    [[nodiscard]] status_t insert_if_missing(value_t &&value) noexcept {
        if (contains(value)) return {success_k};
        return upsert(std::move(value));
    }

    /**
     *  @brief Atomically upserts an element (insert or update). Always succeeds (unless OOM).
     *    Overwrites existing element if key exists (upsert semantics).
     *
     *  @param[in] element Element to upsert (moved into the tree).
     *  @return Success or error code (e.g., out of memory).
     */
    [[nodiscard]] status_t upsert(value_t &&value) noexcept {
        auto node = entries_.allocator().allocate(1);
        if (!node) return {out_of_memory_heap_k};

        // The allocator hands back raw storage, so the links and the subtree bookkeeping are
        // garbage until the node itself is constructed - not just its payload.
        using node_t = typename versioned_set_t::node_t;
        new (node) node_t {};

        generation_t generation = new_generation_();
        auto &versioned = node->fruit;
        versioned.unversioned = std::move(value);
        versioned.generation = generation;
        versioned.deleted = false;
        versioned.visible = true;
        entries_.merge(extract_result_t {&entries_, node});
        ++visible_count_;
        assert(!versioned.deleted && "versioned.deleted is always false here, otherwise update visible_deleted_count_");

        if constexpr (std::is_copy_constructible_v<value_t>) {
            erase_range(
                mapping_key_or_itself<value_t>(versioned.unversioned),
                dated_identifier<identifier_t> {mapping_key_or_itself<value_t>(versioned.unversioned), generation});
        }
        else {
            auto upper_bound_key_copy = copy_safely(mapping_key_or_itself<value_t>(versioned.unversioned));
            if (!upper_bound_key_copy) return {errc_t::out_of_memory_heap_k};
            erase_range(mapping_key_or_itself<value_t>(versioned.unversioned),
                        dated_identifier_t {std::move(*upper_bound_key_copy), generation});
        }
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
        std::size_t const count = std::distance(first, last);
        if (count == 0) return {success_k};

        // Build temporary tree from range with this generation
        versioned_set_t temp_tree(entries_.key_comp(), entries_.allocator());
        generation_t generation = new_generation_();

        for (; first != last; ++first) {
            versioned_t versioned;
            versioned.unversioned = *first;
            versioned.generation = generation;
            versioned.deleted = false;
            versioned.visible = true;

            auto result = temp_tree.insert_if_missing(std::move(versioned));
            if (result.first == temp_tree.end() && !result.second) return {out_of_memory_heap_k};
        }

        bool conflict = false;
        for (auto const &temp_entry : temp_tree) {
            identifier_t id {temp_entry.unversioned};
            find_visible_entry_(id, [&](versioned_t const &) noexcept { conflict = true; }, {});
            if (conflict) break;
        }

        if (conflict) return {invalid_argument_k};

        entries_.merge(temp_tree, assume_unique_t {});
        visible_count_ += count;

        return {success_k};
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
        auto txn = transaction();
        if (!txn) return {out_of_memory_heap_k};

        for (; first != last; ++first) {
            auto status = txn->insert_if_missing(value_t(*first));
            if (!status) return status;
        }

        auto stage_status = txn->stage();
        if (!stage_status) return stage_status;
        return txn->commit();
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
        auto txn = transaction();
        if (!txn) return {out_of_memory_heap_k};

        for (; first != last; ++first) {
            auto status = txn->upsert(value_t(*first));
            if (!status) return status;
        }

        auto stage_status = txn->stage();
        if (!stage_status) return stage_status;
        return txn->commit();
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
        std::size_t const count = std::distance(first, last);
        if (count == 0) return {success_k};

        // Build temporary tree from range
        versioned_set_t temp_tree(entries_.key_comp(), entries_.allocator());
        generation_t generation = new_generation_();

        for (; first != last; ++first) {
            versioned_t versioned;
            versioned.unversioned = *first;
            versioned.generation = generation;
            versioned.deleted = false;
            versioned.visible = true;

            auto result = temp_tree.insert_if_missing(std::move(versioned));
            if (result.first == temp_tree.end() && !result.second) return {out_of_memory_heap_k};
        }

        bool all_exist = true;
        for (auto const &temp_entry : temp_tree) {
            identifier_t id {temp_entry.unversioned};
            bool found = false;
            find_visible_entry_(id, [&](versioned_t const &) noexcept { found = true; }, {});
            if (!found) {
                all_exist = false;
                break;
            }
        }
        if (!all_exist) return {key_not_found_k};

        entries_.merge_with_upsert(temp_tree);

        for (auto const &versioned : temp_tree) {
            erase_range(versioned.unversioned, dated_identifier_t {versioned.unversioned, generation});
        }

        return {success_k};
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

        find_visible_entry_(
            std::forward<comparable_type_>(comparable),
            [&](versioned_t const &versioned) noexcept { callback_found(versioned.unversioned); },
            std::forward<callback_missing_type_>(callback_missing));
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
                     callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        using node_t = typename versioned_set_t::node_t;
        node_t *next_visible = node_t::lower_bound(entries_.root(), comparable, entries_.key_comp());
        while (next_visible && !next_visible->fruit.visible)
            next_visible = node_t::upper_bound(entries_.root(), next_visible->fruit, entries_.key_comp());

        if (next_visible) callback_found(next_visible->fruit.unversioned);
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
                     callback_missing_type_ &&callback_missing = {}) const noexcept {

        static_assert(is_safe_callback_for<callback_found_type_, value_t const &>,
                      "callback_found must be noexcept invocable with value_t const &");
        static_assert(is_safe_callback<callback_missing_type_>, "callback_missing must be noexcept invocable");

        using node_t = typename versioned_set_t::node_t;
        node_t *next_visible = node_t::upper_bound(entries_.root(), comparable, entries_.key_comp());
        while (next_visible && !next_visible->fruit.visible)
            next_visible = node_t::upper_bound(entries_.root(), next_visible->fruit, entries_.key_comp());

        if (next_visible) callback_found(next_visible->fruit.unversioned);
        else callback_missing();
    }

#pragma endregion Lookup

#pragma region Range Operations

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) const noexcept {
        using node_t = typename versioned_set_t::node_t;
        node_t::range(entries_.root(), std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                      entries_.key_comp(), [&](node_t *node) noexcept {
                          if (node->fruit.visible) callback(node->fruit.unversioned);
                      });
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    void update_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept
        requires is_mapping<value_t>
    {
        generation_t generation = new_generation_();
        using node_t = typename versioned_set_t::node_t;
        node_t::range(entries_.root(), std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                      entries_.key_comp(), [&](node_t *node) noexcept {
                          if (!node->fruit.visible) return;
                          callback(node->fruit.unversioned.key, node->fruit.unversioned.mapped);
                          node->fruit.generation = generation;
                      });
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_fn_t>
    status_t erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) noexcept {
        entries_.erase_range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                             [&](versioned_t const &versioned) noexcept {
                                 if (versioned.visible) {
                                     callback(versioned.unversioned);
                                     --visible_count_;
                                     visible_deleted_count_ -= versioned.deleted;
                                 }
                             });
        return status_t {success_k};
    }

#pragma endregion Range Operations

#pragma region Sampling

    template <typename lower_type_, typename upper_type_, typename generator_type_,
              typename callback_type_ = no_op_fn_t>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                      callback_type_ &&callback) const noexcept {

        using node_t = typename versioned_set_t::node_t;
        auto node = node_t::sample_range( //
            entries_.root(), lower, upper, entries_.key_comp(), std::forward<generator_type_>(generator),
            [](node_t *node) noexcept { return node->fruit.visible; });
        // Callers see the stored value; the version metadata never leaves this class.
        if (node) callback(node->fruit.unversioned);
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator, std::size_t &seen,
                      std::size_t reservoir_capacity, output_iterator_type_ &&reservoir) const noexcept {

        using output_iterator_t = std::remove_reference_t<output_iterator_type_>;
        using output_category_t = typename std::iterator_traits<output_iterator_t>::iterator_category;
        static_assert(std::is_same<std::random_access_iterator_tag, output_category_t>(), "Must be random access!");

        auto sampler = [&](value_t const &value) noexcept {
            if (seen < reservoir_capacity) reservoir[seen] = value;

            else {
                std::uniform_int_distribution<std::size_t> distribution {0, seen};
                auto slot_to_replace = distribution(generator);
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
        requires supports_order_statistics<versioned_set_t>
    {
        std::size_t visible_index = 0;
        bool found = false;

        using node_t = typename versioned_set_t::node_t;
        node_t::for_each_left_right(entries_.root(), [&](node_t *node) noexcept {
            if (!node->fruit.visible || node->fruit.deleted) return;
            if (visible_index == k) {
                callback_found(node->fruit.unversioned);
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
        requires supports_order_statistics<versioned_set_t>
    {
        std::size_t rank_value = 0;
        bool found = false;
        identifier_t target_id(comparable);
        versioned_comparator_t less;

        using node_t = typename versioned_set_t::node_t;
        node_t::for_each_left_right(entries_.root(), [&](node_t *node) noexcept {
            if (!node->fruit.visible || node->fruit.deleted) return;

            if (less.same(node->fruit.unversioned, target_id)) {
                callback_found(rank_value);
                found = true;
                return;
            }

            if (less(node->fruit.unversioned, target_id)) ++rank_value;
        });

        if (!found) callback_missing();
    }

    /**
     *  @brief Erases a single entry matching the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c element_t and convertible to @c identifier_t.
     *  @param[in] callback_found Callback to receive the erased entry. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     *  @return Always succeeds.
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
            return status_t {success_k};
        }

        erase_range(comparable, dated_identifier_t {identifier_t(comparable), generation_ + 1});
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
    void print(stream_type_ &stream) {
        stream << "Items: " << entries_.size() << "\n";
        stream << "Imbalance: " << entries_.total_imbalance() << "\n";
        using node_t = typename versioned_set_t::node_t;
        node_t::for_each_left_right(entries_.root(), [&](node_t *node) {
            char const *marker = node->fruit.visible ? "✓" : "✗";
            stream << identifier_t {node->fruit.unversioned} << " @" << node->fruit.generation;
            stream << marker << " ";
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
template <typename value_type_, typename comparator_type_ = std::less<value_type_>,
          typename allocator_type_ = std::allocator<value_type_>>
using transactional_avl_set = transactional_binary_tree<basic_avl_tree<value_type_, comparator_type_, allocator_type_>>;

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
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using transactional_avl_map =
    transactional_binary_tree<basic_avl_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>>;

/**
 *  @brief STL-style transactional set using weight-balanced tree with order statistics support.
 *    Stores unique elements in sorted order with ACID transaction semantics and O(log n) rank/select operations.
 *
 *  @tparam value_type_ Type of elements stored in the set.
 *  @tparam comparator_type_ Comparator for ordering elements. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes, defaults to @c std::allocator.
 */
template <typename value_type_, typename comparator_type_ = std::less<value_type_>,
          typename allocator_type_ = std::allocator<value_type_>>
using transactional_wb_set = transactional_binary_tree<basic_wb_tree<value_type_, comparator_type_, allocator_type_>>;

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
          typename allocator_type_ = std::allocator<mapping<key_type_, value_type_>>>
using transactional_wb_map =
    transactional_binary_tree<basic_wb_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>>;

#pragma endregion Order Statistics

} // namespace ashvardanian::smashtable
