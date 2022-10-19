#pragma once
#include <algorithm> // `std::max`
#include <memory>    // `std::allocator`

#include "status.hpp"

namespace av {

/**
 * @brief AVL-Trees are some of the simplest yet performant Binary Search Trees.
 * This "node" class implements the primary logic, but doesn't take part in
 * memory management.
 *
 * > Never throws! Even if new node allocation had failed.
 * > Implements `find_next` for faster and lighter iterators.
 *   Alternative would be - Binary Threaded Search Tree.
 * > Implements sampling methods.
 *
 * @tparam content_at       Type of contents to store in this tree.
 * @tparam comparator_at    A comparator function object, that overload
 *                          @code
 *                              bool operator ()(content_at, content_at) const
 *                          @endcode
 */
template <typename content_at, typename comparator_at>
class avl_node_gt {
  public:
    using content_t = content_at;
    using comparator_t = comparator_at;
    using height_t = std::int16_t;
    using node_t = avl_node_gt;

    content_t content;
    node_t* left = nullptr;
    node_t* right = nullptr;
    height_t height = 0;

    static height_t get_height(node_t* node) noexcept { return node ? node->height : 0; }
    static height_t get_balance(node_t* node) noexcept {
        return node ? get_height(node->left) - get_height(node->right) : 0;
    }

#pragma mark - Search

    template <typename callback_at>
    static void for_each_top_down(node_t* node, callback_at&& callback) noexcept {
        if (!node)
            return;
        callback(node);
        for_each(node->left, callback);
        for_each(node->right, callback);
    }

    template <typename callback_at>
    static void for_each_bottom_up(node_t* node, callback_at&& callback) noexcept {
        if (!node)
            return;
        for_each(node->left, callback);
        for_each(node->right, callback);
        callback(node);
    }

    static node_t* find_min(node_t* node) noexcept {
        while (node->left)
            node = node->left;
        return node;
    }

    static node_t* find_max(node_t* node) noexcept {
        while (node->right)
            node = node->right;
        return node;
    }

    /**
     * @brief Searches for equal content in this subtree.
     * @param comparable Any key comparable with stored contents.
     * @return NULL if nothing was found.
     */
    template <typename comparable_at>
    static node_t* find(node_t* node, comparable_at&& comparable) noexcept {
        auto less = comparator_t {};
        while (node) {
            if (less(comparable, node->content))
                node = node->left;
            else if (less(node->content, comparable))
                node = node->right;
            else
                break;
        }
        return node;
    }

    /**
     * @brief Searches for the shortest node, that is ancestor of both provided keys.
     * @return NULL if nothing was found.
     * ! Recursive implementation is suboptimal.
     */
    template <typename comparable_a_at, typename comparable_b_at>
    static node_t* lowest_common_ancestor(node_t* node, comparable_a_at&& a, comparable_b_at&& b) noexcept {
        if (!node)
            return nullptr;

        auto less = comparator_t {};
        // If both `a` and `b` are smaller than `node`, then LCA lies in left
        if (less(a, node->content) && less(b, node->content))
            return lowest_common_ancestor(node->left, a, b);

        // If both `a` and `b` are greater than `node`, then LCA lies in right
        else if (less(node->content, a) && less(node->content, b))
            return lowest_common_ancestor(node->right, a, b);

        else
            return node;
    }

    /**
     * @brief Searches for the first/smallest content that compares equal to the provided content.
     */
    template <typename comparable_at>
    static node_t* lower_bound(node_t* node, comparable_at&& comparable) noexcept;

    /**
     * @brief Searches for the last/biggest content that compares equal to the provided content.
     */
    template <typename comparable_at>
    static node_t* upper_bound(node_t* node, comparable_at&& comparable) noexcept;

    struct node_interval_t {
        node_t* lower_bound = nullptr;
        node_t* upper_bound = nullptr;
        node_t* lowest_common_ancestor = nullptr;
    };

    /**
     * @brief Complex method, that detects the left-most and right-most nodes
     * containing keys in a provided intervals, as well as their lowest common ancestors.
     * ! Has a recursive implementation for now.
     */
    template <typename lower_at, typename upper_at, typename callback_at>
    static node_interval_t find_equals_interval(node_t* node,
                                                lower_at&& low,
                                                upper_at&& high,
                                                callback_at&& callback) noexcept {
        if (!node)
            return {};

        // If this node fits into the interval - analyze its children.
        // The first call to reach this branch in the call-stack
        // will be by definition the Lowest Common Ancestor.
        auto less = comparator_t {};
        if (!less(high, node->content) && !less(node->content, low)) {
            callback(node);
            auto left_sub_interval = find_equals_interval(node->left, low, high, callback);
            auto right_sub_interval = find_equals_interval(node->right, low, high, callback);

            auto result = node_interval_t {};
            result.lower_bound = left_sub_interval.lower_bound ?: node;
            result.upper_bound = right_sub_interval.upper_bound ?: node;
            result.lowest_common_ancestor = node;
            return result;
        }

        else if (less(node->content, low))
            return find_equals_interval(node->right, low, high, callback);

        else
            return find_equals_interval(node->left, low, high, callback);
    }

    template <typename comparable_at>
    static node_interval_t equal_interval(node_t* node, comparable_at&& comparable) noexcept {
        return find_equals_interval(node, comparable, comparable);
    }

    /**
     * @brief Find the smallest content, bigger than the provided one.
     * @param comparable Any key comparable with stored contents.
     * @return NULL if nothing was found.
     *
     * Is used for an atomic implementation of iterators.
     * Alternatively one can:
     * > store a stack for path, which is ~O(logN) space.
     * > store parents in nodes and have complex logic.
     * This implementation has no recursion and no
     */
    template <typename comparable_at>
    static node_t* find_next(node_t* node, comparable_at&& comparable) noexcept {
        node_t* succ = nullptr;
        auto less = comparator_t {};
        while (node) {
            // If the given key is less than the root node, visit the left
            // subtree, taking current node as potential successor.
            if (less(comparable, node->content)) {
                succ = node;
                node = node->left;
            }

            // Of the given key is more than the root node, visit the right
            // subtree.
            else if (less(node->content, comparable)) {
                node = node->right;
            }

            // If a node with the desired value is found, the successor is the
            // minimum value node in its right subtree (if any).
            else {
                if (node->right)
                    succ = find_min(node->right);
                break;
            }
        }
        return succ;
    }

#pragma mark - Insertions

    static node_t* rotate_right(node_t* y) noexcept {
        node_t* x = y->left;
        node_t* z = x->right;

        // Perform rotation
        x->right = y;
        y->left = z;

        // Update heights
        y->height = std::max(get_height(y->left), get_height(y->right)) + 1;
        x->height = std::max(get_height(x->left), get_height(x->right)) + 1;
        return x;
    }

    static node_t* rotate_left(node_t* x) noexcept {
        node_t* y = x->right;
        node_t* z = y->left;

        // Perform rotation
        y->left = x;
        x->right = z;

        // Update heights
        x->height = std::max(get_height(x->left), get_height(x->right)) + 1;
        y->height = std::max(get_height(y->left), get_height(y->right)) + 1;
        return y;
    }

    struct find_or_make_result_t {
        node_t* root = nullptr;
        node_t* match = nullptr;
        bool inserted = false;

        /**
         * @return True if the allocation of the new node has failed.
         */
        bool failed() const noexcept { return !inserted && !match; }
    };

    template <typename comparable_at>
    inline static node_t* rebalance_after_insert(node_t* node, comparable_at&& comparable) noexcept {
        // Update height and check if branches aren't balanced
        node->height = std::max(get_height(node->left), get_height(node->right)) + 1;
        auto balance = get_balance(node);
        auto less = comparator_t {};

        // Left Left Case
        if (balance > 1 && less(comparable, node->left->content))
            return rotate_right(node);

        // Right Right Case
        else if (balance < -1 && less(node->right->content, comparable))
            return rotate_left(node);

        // Left Right Case
        else if (balance > 1 && less(node->left->content, comparable)) {
            node->left = rotate_left(node->left);
            return rotate_right(node);
        }
        // Right Left Case
        else if (balance < -1 && less(comparable, node->right->content)) {
            node->right = rotate_right(node->right);
            return rotate_left(node);
        }
        else
            return node;
    }

    template <typename comparable_at, typename node_allocator_at>
    static find_or_make_result_t find_or_make(node_t* node,
                                              comparable_at&& comparable,
                                              node_allocator_at&& node_allocator) noexcept {
        if (!node) {
            node = node_allocator();
            if (node) {
                node->left = nullptr;
                node->right = nullptr;
                node->height = 1;
            }
            return {node, node, true};
        }

        auto less = comparator_t {};
        if (less(comparable, node->content)) {
            auto downstream = find_or_make(node->left, comparable, node_allocator);
            node->left = downstream.root;
            if (downstream.inserted)
                node = rebalance_after_insert(node);
            return {node, downstream.match, downstream.inserted};
        }
        else if (less(node->content, comparable)) {
            auto downstream = find_or_make(node->right, comparable, node_allocator);
            node->right = downstream.root;
            if (downstream.inserted)
                node = rebalance_after_insert(node);
            return {node, downstream.match, downstream.inserted};
        }
        else {
            // Equal keys are not allowed in BST
            return {node, node, false};
        }
    }

    template <typename node_allocator_at>
    static find_or_make_result_t insert(node_t* node,
                                        content_t&& content,
                                        node_allocator_at&& node_allocator) noexcept {
        auto result = find_or_make(node, content, node_allocator);
        if (result.inserted)
            result.match->content = std::move(content);
        return result.root;
    }

    template <typename node_allocator_at>
    static find_or_make_result_t upsert(node_t* node,
                                        content_t&& content,
                                        node_allocator_at&& node_allocator) noexcept {
        auto result = find_or_make(node, content, node_allocator);
        if (result.match)
            result.match->content = std::move(content);
        return result.root;
    }

#pragma mark - Removals

    struct pop_result_t {
        node_t* root = nullptr;
        std::unique_ptr<node_t> popped;
    };

    inline static node_t* rebalance_after_pop(node_t* node) noexcept {
        node->height = 1 + std::max(get_height(node->left), get_height(node->right));
        auto balance = get_balance(node);

        // Left Left Case
        if (balance > 1 && get_balance(node->left) >= 0)
            return rotate_right(node);

        // Left Right Case
        else if (balance > 1 && get_balance(node->left) < 0) {
            node->left = rotate_left(node->left);
            return rotate_right(node);
        }

        // Right Right Case
        else if (balance < -1 && get_balance(node->right) <= 0)
            return rotate_left(node);

        // Right Left Case
        else if (balance < -1 && get_balance(node->right) > 0) {
            node->right = rotate_right(node->right);
            return rotate_left(node);
        }
        else
            return node;
    }

    /**
     * @brief Pops the root replacing it with one of descendants, if present.
     * @param comparable Any key comparable with stored contents.
     */
    static pop_result_t pop(node_t* node) noexcept {

        // If the node has two children, replace it with the
        // smallest entry in the right branch.
        if (node->left && node->right) {
            node_t* midpoint = find_min(node->right);
            auto downstream = pop(midpoint->right, midpoint->content);
            midpoint = downstream.popped.release();
            midpoint->left = node->left;
            midpoint->right = downstream.root;
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            return {midpoint, {node}};
        }
        // Just one child is present, so it is the natural successor.
        else if (node->left || node->right) {
            node_t* replacement = node->left ? node->left : node->right;
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            return {replacement, {node}};
        }
        // No children are present.
        else {
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            return {nullptr, {node}};
        }
    }

    /**
     * @brief Searches for a matching ancestor and pops it out.
     * @param comparable Any key comparable with stored contents.
     */
    template <typename comparable_at>
    static pop_result_t pop(node_t* node, comparable_at&& comparable) noexcept {
        if (!node)
            return {node, {}};

        auto less = comparator_t {};
        if (less(comparable, node->content)) {
            auto downstream = pop(node->left, comparable);
            node->left = downstream.node;
            if (downstream.popped)
                node = rebalance_after_pop(node);
            return {node, std::move(downstream.popped)};
        }

        else if (less(node->content, comparable)) {
            auto downstream = pop(node->right, comparable);
            node->right = downstream.node;
            if (downstream.popped)
                node = rebalance_after_pop(node);
            return {node, std::move(downstream.popped)};
        }

        else
            // We have found the node to pop!
            return pop(node);
    }

    struct remove_if_result_t {
        node_t* root = nullptr;
        std::size_t count = 0;
    };

    template <typename predicate_at, typename node_deallocator_at>
    static remove_if_result_t remove_if(node_t* node,
                                        predicate_at&& predicate,
                                        node_deallocator_at&& node_deallocator) noexcept {
        return {};
    }

    static node_t* remove_interval(node_t* node, node_interval_t&& interval) noexcept {
        return node;
    }
};

template <typename content_at,
          typename comparator_at,
          typename node_allocator_at = std::allocator<avl_node_gt<content_at, comparator_at>>>
class avl_tree_gt {
  public:
    using node_t = avl_node_gt<content_at, comparator_at>;
    using node_allocator_t = node_allocator_at;
    using comparator_t = comparator_at;
    using content_t = content_at;

  private:
    node_t* root_ = nullptr;
    std::size_t size_ = 0;
    node_allocator_t allocator_;

  public:
    avl_tree_gt() noexcept = default;
    avl_tree_gt(avl_tree_gt&& other) noexcept
        : root_(std::exchange(other.root_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    avl_tree_gt& operator=(avl_tree_gt& other) noexcept {
        std::swap(root_, other.root_);
        std::swap(size_, other.size_);
        return *this;
    }

    ~avl_tree_gt() { clear(); }
    std::size_t size() const noexcept { return size_; }
    node_t* root() const noexcept { return root_; }

    template <typename comparable_at>
    node_t* find(comparable_at&& comparable) noexcept {
        return node_t::find_first_of(root_, std::forward<comparable_at>(comparable));
    }

    template <typename comparable_at>
    node_t* find_next(comparable_at&& comparable) noexcept {
        return node_t::find_next(root_, std::forward<comparable_at>(comparable));
    }

    struct upsert_result_t {
        node_t* node = nullptr;
        bool inserted = false;

        /**
         * @return True if the allocation of the new node has failed.
         */
        bool failed() const noexcept { return !inserted && !node; }
        upsert_result_t& operator=(content_t&& content) noexcept {
            node->content = content;
            return *this;
        }
    };

    template <typename comparable_at>
    upsert_result_t insert(comparable_at&& comparable) noexcept {
        auto result =
            node_t::insert(root_, std::forward<comparable_at>(comparable), [&] { return allocator_.allocate(1); });
        root_ = result.root;
        size_ += result.inserted;
        return {result.match, result.inserted};
    }

    template <typename comparable_at>
    upsert_result_t upsert(comparable_at&& comparable) noexcept {
        auto result =
            node_t::upsert(root_, std::forward<comparable_at>(comparable), [&] { return allocator_.allocate(1); });
        root_ = result.root;
        size_ += result.inserted;
        return {result.match, result.inserted};
    }

    struct pop_result_t {
        avl_tree_gt* tree_ = nullptr;
        node_t* node_ptr_ = nullptr;

        ~pop_result_t() noexcept {
            if (node_ptr_)
                tree_->allocator_.deallocate(node_ptr_);
        }
        explicit operator bool() const noexcept { return node_ptr_; }
    };

    template <typename comparable_at>
    pop_result_t pop(comparable_at&& comparable) noexcept {
        auto result = node_t::pop(root_, std::forward(comparable));
        root_ = result.root;
        size_ -= result.popped != nullptr;
        return pop_result_t {this, result.popped.release()};
    }

    template <typename comparable_at>
    bool erase(comparable_at&& comparable) noexcept {
        return pop(std::forward(comparable));
    }

    template <typename comparable_at>
    std::size_t erase_equals_interval(comparable_at&& comparable) noexcept {
        return 0;
    }

    void clear() noexcept {
        node_t::for_each_bottom_up(root_, [&](node_t* node) { return allocator_.deallocate(node, 1); });
        root_ = nullptr;
        size_ = 0;
    }
};

/**
 * @brief Transactional Concurrent In-Memory Container with Snapshots support.
 *
 * @section Writes Consistency
 * Writing one entry or a batch is logically different.
 * Either all fail or all succeed. Thats why `set` and `set_many`
 * are implemented separately. Transactions write only on `submit`,
 * thus they don't need `set_many`.
 *
 * @section Read Consistency
 * Reading a batch of entries is same as reading one by one.
 * The received items might not be consistent with each other.
 * If such behaviour is needed - you must create snapshot.
 *
 * @section Pitfalls with WATCH-ing missing values
 * If an entry was missing. Then:
 *      1. WATCH-ed in a transaction.
 *      2. added in the second transaction.
 *      3. removed in the third transaction.
 * The first transaction will succeed, if we try to commit it.
 *
 */
template < //
    typename element_at,
    typename comparator_at = std::less<element_at>,
    typename allocator_at = std::allocator<std::uint8_t>>
class consistent_avl_gt {

  public:
    using element_t = element_at;
    using comparator_t = comparator_at;
    using allocator_t = allocator_at;

    using versioning_t = element_versioning_gt<element_t, comparator_t>;
    using identifier_t = typename versioning_t::identifier_t;
    using generation_t = typename versioning_t::generation_t;
    using status_t = typename versioning_t::status_t;
    using dated_identifier_t = typename versioning_t::dated_identifier_t;
    using watch_t = typename versioning_t::watch_t;
    using watched_identifier_t = typename versioning_t::watched_identifier_t;
    using entry_t = typename versioning_t::entry_t;
    using entry_comparator_t = typename versioning_t::entry_comparator_t;

  private:
    using entry_node_t = avl_node_gt<entry_t, entry_comparator_t>;
    using entry_allocator_t = typename allocator_t::template rebind<entry_node_t>::other;
    using entry_set_t = avl_tree_gt<entry_t, entry_comparator_t, entry_allocator_t>;
    using entry_iterator_t = entry_node_t*;

    using watches_allocator_t = typename allocator_t::template rebind<watched_identifier_t>::other;
    using watches_array_t = std::vector<watched_identifier_t, watches_allocator_t>;
    using watch_iterator_t = typename watches_array_t::iterator;

    using store_t = consistent_avl_gt;

  public:
    class transaction_t {

        friend store_t;
        enum class stage_t {
            created_k,
            staged_k,
            commited_k,
        };

        store_t* store_ {nullptr};
        entry_set_t changes_ {};
        watches_array_t watches_ {};
        generation_t generation_ {0};
        stage_t stage_ {stage_t::created_k};

        transaction_t(store_t& set) noexcept(false) : store_(&set) {}
        void date(generation_t generation) noexcept { generation_ = generation; }
        watch_t missing_watch() const noexcept { return watch_t {generation_, true}; }
        store_t& store_ref() noexcept { return *store_; }
        store_t const& store_ref() const noexcept { return *store_; }

      public:
        transaction_t(transaction_t&&) noexcept = default;
        transaction_t& operator=(transaction_t&&) noexcept = default;
        transaction_t(transaction_t const&) = delete;
        transaction_t& operator=(transaction_t const&) = delete;

        [[nodiscard]] status_t upsert(element_t&& element) noexcept {
            entry_t entry;
            entry.element = std::move(element);
            entry.generation = generation_;
            entry.deleted = false;
            entry.visible = false;
            auto result = changes_.upsert(std::move(entry));
            return result.failed() ? status_t {out_of_memory_heap_k} : status_t {success_k};
        }

        [[nodiscard]] status_t erase(identifier_t const& id) noexcept {
            entry_t entry;
            entry.element = id;
            entry.generation = generation_;
            entry.deleted = true;
            entry.visible = false;
            auto result = changes_.upsert(std::move(entry));
            return result.failed() ? status_t {out_of_memory_heap_k} : status_t {success_k};
        }

        [[nodiscard]] status_t watch(identifier_t const& id) noexcept {
            return store_ref().find_first_of(
                id,
                [&](entry_t const& entry) {
                    watches_.insert_or_assign(identifier_t(entry.element), watch_t {entry.generation, entry.deleted});
                },
                [&] { watches_.insert_or_assign(id, missing_watch()); });
        }

        [[nodiscard]] status_t watch(entry_t const& entry) noexcept {
            return invoke_safely([&] {
                watches_.insert_or_assign(identifier_t(entry.element), watch_t {entry.generation, entry.deleted});
            });
        }

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t find_first_of(comparable_at&& comparable,
                                             callback_found_at&& callback_found,
                                             callback_missing_at&& callback_missing = {}) const noexcept {
            if (auto iterator = changes_.find_first_of(std::forward<comparable_at>(comparable));
                iterator != changes_.end())
                return !iterator->deleted ? invoke_safely([&callback_found, &iterator] { callback_found(*iterator); })
                                          : invoke_safely(callback_missing);
            else
                return store_ref().find_first_of(std::forward<comparable_at>(comparable),
                                                 std::forward<callback_found_at>(callback_found),
                                                 std::forward<callback_missing_at>(callback_missing));
        }

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t find_next(comparable_at&& comparable,
                                         callback_found_at&& callback_found,
                                         callback_missing_at&& callback_missing = {}) const noexcept {
            auto external_previous_id = identifier_t(comparable);
            auto internal_iterator = changes_.upper_bound(std::forward<comparable_at>(comparable));
            while (internal_iterator != changes_.end() && internal_iterator->deleted)
                ++internal_iterator;

            // Once picking the next smallest element from the global store,
            // we might face an entry, that was already deleted from here,
            // so this might become a multi-step process.
            auto faced_deleted_entry = false;
            auto callback_external_found = [&](element_t const& external_element) {
                // The simplest case is when we have an external object.
                if (internal_iterator == changes_.end())
                    return callback_found(external_element);

                element_t const& internal_element = internal_iterator->content;
                if (!entry_comparator_t {}(external_element, internal_element))
                    return callback_found(internal_element);

                // Check if this entry was deleted and we should try again.
                auto external_id = identifier_t(external_element);
                auto external_element_internal_state = changes_.find_first_of(external_element);
                if (external_element_internal_state != changes_.end() && external_element_internal_state->deleted) {
                    faced_deleted_entry = true;
                    external_previous_id = external_id;
                    return;
                }
                else
                    return callback_found(external_element);
            };
            auto callback_external_missing = [&] {
                if (internal_iterator == changes_.end())
                    return callback_missing();
                else {
                    element_t const& internal_element = internal_iterator->content;
                    return callback_found(internal_element);
                }
            };

            // Iterate until we find the a non-deleted external value
            auto& store = store_ref();
            auto status = status_t {};
            do {
                status = store.find_next(external_previous_id, callback_external_found, callback_external_missing);
            } while (faced_deleted_entry && status);
            return status;
        }

        [[nodiscard]] status_t stage() noexcept {
            // First, check if we have any collisions.
            auto& store = store_ref();
            auto entry_missing = missing_watch();
            for (auto const& [id, watch] : watches_) {
                auto consistency_violated = false;
                auto status = store.find_first_of(
                    id,
                    [&](entry_t const& entry) noexcept { consistency_violated = entry != watch; },
                    [&]() noexcept { consistency_violated = entry_missing != watch; });
                if (consistency_violated)
                    return {consistent_set_errc_t::consistency_k};
                if (!status)
                    return status;
            }

            // Now all of our watches will be replaced with "links" to entries
            // we are merging into the main tree.
            watches_.clear();
            auto status = invoke_safely([&] { watches_.reserve(changes_.size()); });
            if (!status)
                return status;

            // No new memory allocations or failures are possible after that.
            // It is all safe.
            for (auto const& entry : changes_)
                watches_.insert_or_assign(identifier_t(entry.element), watch_t {generation_, entry.deleted});

            // Than just merge our current nodes.
            // The visibility will be updated later in the `commit`.
            store.entries_.merge(changes_);
            stage_ = stage_t::staged_k;
            return {success_k};
        }

        /**
         * @brief Resets the state of the transaction.
         * > All the updates staged in DB will be reverted.
         * > All the updates will in this Transaction will be lost.
         * > All the watches will be lost.
         */
        [[nodiscard]] status_t reset() noexcept {
            // If the transaction was "staged",
            // we must delete all the entries.
            auto& store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const& [id, watch] : watches_)
                    // Heterogeneous `erase` is only coming in C++23.
                    if (auto iterator = store.entries_.find_first_of(dated_identifier_t {id, watch.generation});
                        iterator != store.entries_.end())
                        store.entries_.erase(iterator);

            watches_.clear();
            changes_.clear();
            stage_ = stage_t::created_k;
            return {success_k};
        }

        /**
         * @brief Rolls-back a previously "staged" transaction.
         * > All the updates will be reverted in the DB.
         * > All the updates will re-emerge in this Transaction.
         * > All the watches will be lost.
         */
        [[nodiscard]] status_t rollback() noexcept {
            if (stage_ != stage_t::staged_k)
                return {operation_not_permitted_k};

            // If the transaction was "staged",
            // we must delete all the entries.
            auto& store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const& [id, watch] : watches_) {
                    auto source = store.entries_.find_first_of(dated_identifier_t {id, watch.generation});
                    auto node = store.entries_.extract(source);
                    changes_.insert(std::move(node));
                }

            watches_.clear();
            stage_ = stage_t::created_k;
            return {success_k};
        }

        [[nodiscard]] status_t commit() noexcept {
            if (stage_ != stage_t::staged_k)
                return {operation_not_permitted_k};

            // Once we make an entry visible,
            // if there are more than one with the same key,
            // the older generation must die.
            auto& store = store_ref();
            for (auto const& [id, watch] : watches_) {
                auto range = store.entries_.equal_range(id);
                store.unmask_and_compact(range.first, range.second, watch.generation);
            }

            stage_ = stage_t::created_k;
            return {success_k};
        }
    };

  private:
    entry_set_t entries_;
    generation_t generation_ {0};

    consistent_avl_gt() noexcept(false) {}
    generation_t new_generation() noexcept { return ++generation_; }

    template <typename callback_at = no_op_t>
    void erase_visible(entry_iterator_t begin, entry_iterator_t end, callback_at&& callback = {}) noexcept {
        entry_iterator_t& current = begin;
        while (current != end)
            if (current->visible) {
                callback(*current);
                current = entries_.erase(current);
            }
            else
                ++current;
    }

    void unmask_and_compact(entry_iterator_t begin, entry_iterator_t end, generation_t generation_to_unmask) noexcept {
        entry_iterator_t& current = begin;
        entry_iterator_t last_visible_entry = end;
        for (; current != end; ++current) {
            auto keep_this = current->generation == generation_to_unmask;
            current->visible |= keep_this;
            if (!current->visible)
                continue;

            // Older revisions must die
            if (last_visible_entry != end)
                entries_.erase(last_visible_entry);
            last_visible_entry = current;
        }
    }

  public:
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    [[nodiscard]] static std::optional<consistent_avl_gt> make() noexcept {
        std::optional<consistent_avl_gt> result;
        invoke_safely([&] { result.emplace(consistent_avl_gt {}); });
        return result;
    }

    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        generation_t generation = new_generation();
        std::optional<transaction_t> result;
        invoke_safely([&] { result.emplace(transaction_t {*this}).date(generation); });
        return result;
    }

    [[nodiscard]] status_t upsert(element_t&& element) noexcept {
        generation_t generation = new_generation();
        return invoke_safely([&] {
            auto entry = entry_t {std::move(element)};
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            entries_.insert(std::move(entry));
            // auto range_end = .first;
            // auto range_start = entries_.lower_bound(range_end->content);
            // erase_visible(range_start, range_end);
        });
    }

    [[nodiscard]] status_t upsert(entry_set_t& sources) noexcept {
        for (auto source = sources.begin(); source != sources.end();) {
            bool should_compact = source->visible;
            auto source_node = sources.extract(source++);
            entries_.insert(std::move(source_node));
            if (should_compact) {
                // auto range_start = entries_.lower_bound(range_end->content);
                // erase_visible(range_start, range_end);
            }
        }
        return {success_k};
    }

    template <typename elements_begin_at, typename elements_end_at = elements_begin_at>
    [[nodiscard]] status_t upsert(elements_begin_at begin, elements_end_at end) noexcept {}

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find_first_of(comparable_at&& comparable,
                                         callback_found_at&& callback_found,
                                         callback_missing_at&& callback_missing = {}) const noexcept {

        entry_node_t* largest_visible = nullptr;
        entry_node_t::find_equals_interval(entries_.root(), comparable, comparable, [&](entry_node_t* node) {
            if ((node->content.visible) &&
                (!largest_visible || node->content.generation > largest_visible->content.generation))
                largest_visible = node;
        });

        static_assert(noexcept(callback_found(largest_visible->content.element)));
        static_assert(noexcept(callback_missing()));
        largest_visible ? callback_found(largest_visible->content.element) : callback_missing();
        return {success_k};
    }

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find_next(comparable_at&& comparable,
                                     callback_found_at&& callback_found,
                                     callback_missing_at&& callback_missing = {}) const noexcept {

        // Skip all the invisible entries
        entry_node_t* next_visible = entry_node_t::find_next(entries_.root(), comparable);
        while (next_visible && !next_visible->content.visible) {
            next_visible = entry_node_t::find_next(entries_.root(), next_visible->content);
            // The logic is more complex if we start doing multi-versioning
            // TODO:
        }

        static_assert(noexcept(callback_found(next_visible->content.element)));
        static_assert(noexcept(callback_missing()));
        next_visible ? callback_found(next_visible->content.element) : callback_missing();
        return {success_k};
    }

    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t find_equals_interval(comparable_at&& comparable, callback_at&& callback) const noexcept {
        entry_node_t::find_equals_interval(entries_.root(), comparable, comparable, [&](entry_node_t* node) {
            if (node->content.visible)
                callback(node->content.element);
            static_assert(noexcept(callback(node->content.element)));
        });
        return {success_k};
    }

    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t find_equals_interval(comparable_at&& comparable, callback_at&& callback) noexcept {
        generation_t generation = new_generation();
        entry_node_t::find_equals_interval(entries_.root(), comparable, comparable, [&](entry_node_t* node) {
            if (node->content.visible)
                callback(node->content.element), node->content.generation = generation;
            static_assert(noexcept(callback(node->content.element)));
        });
        return {success_k};
    }

    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t erase_equals_interval(comparable_at&& comparable, callback_at&& callback = {}) noexcept {
        // TODO:
        return {success_k};
    }

    [[nodiscard]] status_t clear() noexcept {
        entries_.clear();
        generation_ = 0;
        return {success_k};
    }
};

} // namespace av