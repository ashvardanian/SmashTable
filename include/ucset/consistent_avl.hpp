#pragma once
#include <algorithm> // `std::max`
#include <memory>    // `std::allocator`
#include <optional>  // `std::optional`

#include "status.hpp"

namespace unum::ucset {

/**
 * @brief AVL-Trees are some of the simplest yet performant Binary Search Trees.
 * This "node" class implements the primary logic, but doesn't take part in
 * memory management.
 *
 * > Never throws! Even if new node allocation had failed.
 * > Implements `upper_bound` for faster and lighter iterators.
 *   Alternative would be - Binary Threaded Search Tree.
 * > Implements sampling methods.
 *
 * @tparam entry_at         Type of entries to store in this tree.
 * @tparam comparator_at    A comparator function object, that overload
 *                          @code
 *                              bool operator ()(entry_at, entry_at) const
 *                          @endcode
 */
template <typename entry_at, typename comparator_at>
class avl_node_gt {
  public:
    using entry_t = entry_at;
    using comparator_t = comparator_at;
    using height_t = std::int16_t;
    using node_t = avl_node_gt;

    entry_t entry;
    node_t* left = nullptr;
    node_t* right = nullptr;
    /**
     * @brief Root has the biggest `height` in the tree.
     * Zero is possible only in the uninitialized detached state.
     * A non-NULL node would have height of one.
     * Allows you to guess the upper bound of branch size, as `1 << height`.
     */
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
        for_each_top_down(node->left, callback);
        for_each_top_down(node->right, callback);
    }

    template <typename callback_at>
    static void for_each_bottom_up(node_t* node, callback_at&& callback) noexcept {
        if (!node)
            return;
        for_each_bottom_up(node->left, callback);
        for_each_bottom_up(node->right, callback);
        callback(node);
    }

    template <typename callback_at>
    static void for_each_left_right(node_t* node, callback_at&& callback) noexcept {
        if (!node)
            return;
        for_each_left_right(node->left, callback);
        callback(node);
        for_each_left_right(node->right, callback);
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
     * @brief Searches for equal entry in this subtree.
     * @param comparable Any key comparable with stored entries.
     * @return NULL if nothing was found.
     */
    template <typename comparable_at>
    static node_t* find(node_t* node, comparable_at&& comparable) noexcept {
        auto less = comparator_t {};
        while (node) {
            if (less(comparable, node->entry))
                node = node->left;
            else if (less(node->entry, comparable))
                node = node->right;
            else
                break;
        }
        return node;
    }

    /**
     * @brief Find the smallest entry, bigger than or equal to the provided one.
     * @param comparable Any key comparable with stored entries.
     * @return NULL if nothing was found.
     */
    template <typename comparable_at>
    static node_t* lower_bound(node_t* node, comparable_at&& comparable) noexcept {
        node_t* successor = nullptr;
        comparator_t less;
        while (node) {
            // If the given key is less than the root node, visit the left
            // subtree, taking current node as potential successor.
            if (less(comparable, node->entry)) {
                successor = node;
                node = node->left;
            }

            // Of the given key is more than the root node, visit the right
            // subtree.
            else if (less(node->entry, comparable)) {
                node = node->right;
            }

            // If a node with the desired value is found, the successor is the
            // minimum value node in its right subtree (if any).
            else {
                successor = node;
                node = node->left;
            }
        }
        return successor;
    }

    /**
     * @brief Find the smallest entry, bigger than the provided one.
     * @param comparable Any key comparable with stored entries.
     * @return NULL if nothing was found.
     *
     * Is used for an atomic implementation of iterators.
     * Alternatively one can:
     * > store a stack for path, which is ~O(logN) space.
     * > store parents in nodes and have complex logic.
     */
    template <typename comparable_at>
    static node_t* upper_bound(node_t* node, comparable_at&& comparable) noexcept {
        node_t* successor = nullptr;
        comparator_t less;
        while (node) {
            // If the given key is less than the root node, visit the left
            // subtree, taking current node as potential successor.
            if (less(comparable, node->entry)) {
                successor = node;
                node = node->left;
            }

            // Of the given key is more than the root node, visit the right
            // subtree.
            else if (less(node->entry, comparable)) {
                node = node->right;
            }

            // If a node with the desired value is found, the successor is the
            // minimum value node in its right subtree (if any).
            else {
                if (node->right)
                    successor = find_min(node->right);
                node = nullptr;
            }
        }
        return successor;
    }

    /**
     * @brief Searches for the shortest node, that is ancestor of both provided keys.
     * @return NULL if nothing was found.
     * @warning Current recursive implementation is suboptimal.
     */
    template <typename comparable_a_at, typename comparable_b_at>
    static node_t* lowest_common_ancestor(node_t* node, comparable_a_at&& a, comparable_b_at&& b) noexcept {
        if (!node)
            return nullptr;

        auto less = comparator_t {};
        // If both `a` and `b` are smaller than `node`, then LCA lies in left
        if (less(a, node->entry) && less(b, node->entry))
            return lowest_common_ancestor(node->left, a, b);

        // If both `a` and `b` are greater than `node`, then LCA lies in right
        else if (less(node->entry, a) && less(node->entry, b))
            return lowest_common_ancestor(node->right, a, b);

        else
            return node;
    }

    struct node_interval_t {
        node_t* lower_bound = nullptr;
        node_t* upper_bound = nullptr;
        node_t* lowest_common_ancestor = nullptr;
    };

    /**
     * @brief Complex method, that detects the left-most and right-most nodes
     * containing keys in a provided intervals, as well as their lowest common ancestors.
     * @warning Current recursive implementation is suboptimal.
     */
    template <typename lower_at, typename upper_at, typename callback_at>
    static node_interval_t range(node_t* node, lower_at&& low, upper_at&& high, callback_at&& callback) noexcept {
        if (!node)
            return {};

        // If this node fits into the interval - analyze its children.
        // The first call to reach this branch in the call-stack
        // will be by definition the Lowest Common Ancestor.
        auto less = comparator_t {};
        if (!less(high, node->entry) && !less(node->entry, low)) {
            callback(node);
            auto left_sub_interval = range(node->left, low, high, callback);
            auto right_sub_interval = range(node->right, low, high, callback);

            auto result = node_interval_t {};
            result.lower_bound = left_sub_interval.lower_bound ? left_sub_interval.lower_bound : node;
            result.upper_bound = right_sub_interval.upper_bound ? right_sub_interval.upper_bound : node;
            result.lowest_common_ancestor = node;
            return result;
        }

        else if (less(node->entry, low))
            return range(node->right, low, high, callback);

        else
            return range(node->left, low, high, callback);
    }

    template <typename comparable_at>
    static node_interval_t equal_range(node_t* node, comparable_at&& comparable) noexcept {
        return range(node, comparable, comparable);
    }

    /**
     * @brief Random samples nodes.
     * @param generator Any STL-compatible random number generator.
     * @return NULL if nothing was found.
     * @warning Resulting distribution is inaccurate, as we only have the upper bound of the branch size.
     */
    template <typename generator_at>
    static node_t* sample(node_t* node, generator_at&& generator) noexcept {
        auto less = comparator_t {};
        while (node) {
            auto count_left = node->left ? 1ul << node->left->height : 0ul;
            auto count_right = node->right ? 1ul << node->right->height : 0ul;
            auto count_total = count_left + count_right + 1ul;
            std::uniform_int_distribution<std::size_t> distribution {0, count_total + 1};
            auto choice = distribution(generator);
            if (choice == 0)
                break;

            node = choice > (count_left + 1ul) ? node->right : node->left;
        }
        return node;
    }

    /**
     * @brief Random samples nodes within a given range of keys.
     * @param generator Any STL-compatible random number generator.
     * @return NULL if nothing was found.
     * @warning Without additional stored metadata or dynamic memory, this algorithm performs two passes.
     */
    template <typename generator_at, typename lower_at, typename upper_at, typename predicate_at>
    static node_t* sample_range( //
        node_t* node,
        lower_at&& low,
        upper_at&& high,
        generator_at&& generator,
        predicate_at&& predicate) noexcept {

        std::size_t count_matches = 0;
        range(node, low, high, [&](node_t* node) noexcept { count_matches += predicate(node); });

        node_t* result = node;
        std::uniform_int_distribution<std::size_t> distribution {0, count_matches + 1};
        auto choice = distribution(generator);
        if (choice != 0)
            range(node, low, high, [&](node_t* node) noexcept {
                choice -= predicate(node);
                result = choice != 0 ? result : node;
            });

        return result;
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
        if (balance > 1 && less(comparable, node->left->entry))
            return rotate_right(node);

        // Right Right Case
        else if (balance < -1 && less(node->right->entry, comparable))
            return rotate_left(node);

        // Left Right Case
        else if (balance > 1 && less(node->left->entry, comparable)) {
            node->left = rotate_left(node->left);
            return rotate_right(node);
        }
        // Right Left Case
        else if (balance < -1 && less(comparable, node->right->entry)) {
            node->right = rotate_right(node->right);
            return rotate_left(node);
        }
        else
            return node;
    }

    template <typename comparable_at, typename callback_found_at, typename callback_make_at>
    static find_or_make_result_t find_or_make(node_t* node,
                                              comparable_at&& comparable,
                                              callback_found_at&& callback_found,
                                              callback_make_at&& callback_make) noexcept {
        if (!node) {
            node = callback_make();
            if (node) {
                node->left = nullptr;
                node->right = nullptr;
                node->height = 1;
            }
            return {node, node, true};
        }

        auto less = comparator_t {};
        if (less(comparable, node->entry)) {
            auto downstream = find_or_make(node->left, comparable, callback_found, callback_make);
            node->left = downstream.root;
            if (downstream.inserted)
                node = rebalance_after_insert(node, downstream.match->entry);
            return {node, downstream.match, downstream.inserted};
        }
        else if (less(node->entry, comparable)) {
            auto downstream = find_or_make(node->right, comparable, callback_found, callback_make);
            node->right = downstream.root;
            if (downstream.inserted)
                node = rebalance_after_insert(node, downstream.match->entry);
            return {node, downstream.match, downstream.inserted};
        }
        else {
            // Equal keys are not allowed in BST
            callback_found(node);
            return {node, node, false};
        }
    }

    template <typename node_allocator_at>
    static find_or_make_result_t insert(node_t* node, entry_t&& entry, node_allocator_at&& node_allocator) noexcept {
        auto found = [&](node_t* node) noexcept {
        };
        auto make = [&]() noexcept -> node_t* {
            auto node = node_allocator();
            if (node)
                new (&node->entry) entry_t(std::move(entry));
            return node;
        };
        auto result = find_or_make(node, entry, found, make);
        return result;
    }

    template <typename node_allocator_at>
    static find_or_make_result_t upsert(node_t* node, entry_t&& entry, node_allocator_at&& node_allocator) noexcept {
        auto found = [&](node_t* node) noexcept {
            node->entry = std::move(entry);
        };
        auto make = [&]() noexcept -> node_t* {
            auto node = node_allocator();
            if (node)
                new (&node->entry) entry_t(std::move(entry));
            return node;
        };
        auto result = find_or_make(node, entry, found, make);
        return result;
    }

    static find_or_make_result_t insert(node_t* node, node_t* new_child) noexcept {
        return find_or_make(
            node,
            new_child->entry,
            [](node_t*) noexcept {},
            [=]() noexcept { return new_child; });
    }

#pragma mark - Removals

    struct extract_result_t {
        node_t* root = nullptr;
        std::unique_ptr<node_t> extracted;

        node_t* release() noexcept { return extracted.release(); }
    };

    static node_t* rebalance_after_extract(node_t* node) noexcept {
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
     * @param comparable Any key comparable with stored entries.
     */
    static extract_result_t extract(node_t* node) noexcept {

        // If the node has two children, replace it with the
        // smallest entry in the right branch.
        if (node->left && node->right) {
            node_t* midpoint = find_min(node->right);
            auto downstream = extract(node->right, midpoint->entry);
            midpoint = downstream.extracted.release();
            midpoint->left = node->left;
            midpoint->right = downstream.root;
            midpoint->height = 1 + std::max(get_height(midpoint->left), get_height(midpoint->right));
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            node->height = 1;
            return {midpoint, std::unique_ptr<node_t> {node}};
        }
        // Just one child is present, so it is the natural successor.
        else if (node->left || node->right) {
            node_t* replacement = node->left ? node->left : node->right;
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            node->height = 1;
            return {replacement, std::unique_ptr<node_t> {node}};
        }
        // No children are present.
        else {
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            node->height = 1;
            return {nullptr, std::unique_ptr<node_t> {node}};
        }
    }

    /**
     * @brief Searches for a matching ancestor and extracts it out.
     * @param comparable Any key comparable with stored entries.
     */
    template <typename comparable_at>
    static extract_result_t extract(node_t* node, comparable_at&& comparable) noexcept {
        if (!node)
            return {node, {}};

        auto less = comparator_t {};
        if (less(comparable, node->entry)) {
            auto downstream = extract(node->left, comparable);
            node->left = downstream.root;
            if (downstream.extracted)
                node = rebalance_after_extract(node);
            return {node, std::move(downstream.extracted)};
        }

        else if (less(node->entry, comparable)) {
            auto downstream = extract(node->right, comparable);
            node->right = downstream.root;
            if (downstream.extracted)
                node = rebalance_after_extract(node);
            return {node, std::move(downstream.extracted)};
        }

        else
            // We have found the node to extract!
            return extract(node);
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
};

template <typename entry_at,
          typename comparator_at,
          typename node_allocator_at = std::allocator<avl_node_gt<entry_at, comparator_at>>>
class avl_tree_gt {
  public:
    using node_t = avl_node_gt<entry_at, comparator_at>;
    using node_allocator_t = node_allocator_at;
    using comparator_t = comparator_at;
    using entry_t = entry_at;
    using avl_tree_t = avl_tree_gt;

  private:
    node_t* root_ = nullptr;
    std::size_t size_ = 0;
    node_allocator_t allocator_;

  public:
    avl_tree_gt() noexcept = default;
    avl_tree_gt(avl_tree_gt&& other) noexcept
        : root_(std::exchange(other.root_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    avl_tree_gt& operator=(avl_tree_gt&& other) noexcept {
        std::swap(root_, other.root_);
        std::swap(size_, other.size_);
        return *this;
    }

    ~avl_tree_gt() { clear(); }
    std::size_t size() const noexcept { return size_; }
    std::size_t height() noexcept { return root_ ? root_->height : 0; }
    node_t* root() const noexcept { return root_; }
    node_t* end() const noexcept { return nullptr; }
    node_allocator_t& allocator() noexcept { return allocator_; }
    node_allocator_t const& allocator() const noexcept { return allocator_; }

    std::size_t total_imbalance() const noexcept {
        std::size_t abs_sum = 0;
        node_t::for_each_top_down(root_,
                                  [&](node_t* node) noexcept { abs_sum += std::abs(node_t::get_balance(node)); });
        return abs_sum;
    }

    template <typename comparable_at>
    node_t* find(comparable_at&& comparable) noexcept {
        return node_t::find(root_, std::forward<comparable_at>(comparable));
    }

    template <typename comparable_at>
    node_t* lower_bound(comparable_at&& comparable) noexcept {
        return node_t::lower_bound(root_, std::forward<comparable_at>(comparable));
    }

    template <typename comparable_at>
    node_t* upper_bound(comparable_at&& comparable) noexcept {
        return node_t::upper_bound(root_, std::forward<comparable_at>(comparable));
    }

    template <typename comparable_at>
    node_t const* find(comparable_at&& comparable) const noexcept {
        return node_t::find(root_, std::forward<comparable_at>(comparable));
    }

    template <typename comparable_at>
    node_t const* lower_bound(comparable_at&& comparable) const noexcept {
        return node_t::lower_bound(root_, std::forward<comparable_at>(comparable));
    }

    template <typename comparable_at>
    node_t const* upper_bound(comparable_at&& comparable) const noexcept {
        return node_t::upper_bound(root_, std::forward<comparable_at>(comparable));
    }

    struct upsert_result_t {
        node_t* node = nullptr;
        bool inserted = false;

        /**
         * @return True if the allocation of the new node has failed.
         */
        bool failed() const noexcept { return !inserted && !node; }
        upsert_result_t& operator=(entry_t&& entry) noexcept {
            node->entry = entry;
            return *this;
        }
    };

    template <typename comparable_at>
    upsert_result_t insert(comparable_at&& comparable) noexcept {
        auto result = node_t::insert(root_, std::forward<comparable_at>(comparable), [&]() noexcept {
            return allocator_.allocate(1);
        });
        root_ = result.root;
        size_ += result.inserted;
        return {result.match, result.inserted};
    }

    template <typename comparable_at>
    upsert_result_t upsert(comparable_at&& comparable) noexcept {
        auto result = node_t::upsert(root_, std::forward<comparable_at>(comparable), [&]() noexcept {
            return allocator_.allocate(1);
        });
        root_ = result.root;
        size_ += result.inserted;
        return {result.match, result.inserted};
    }

    struct extract_result_t {
        avl_tree_gt* tree_ = nullptr;
        node_t* node_ptr_ = nullptr;

        ~extract_result_t() noexcept {
            if (node_ptr_)
                tree_->allocator_.deallocate(node_ptr_, 1);
        }
        extract_result_t(extract_result_t const&) = delete;
        extract_result_t& operator=(extract_result_t const&) = delete;
        explicit operator bool() const noexcept { return node_ptr_; }
        node_t* release() noexcept { return std::exchange(node_ptr_, nullptr); }
    };

    template <typename comparable_at>
    extract_result_t extract(comparable_at&& comparable) noexcept {
        auto result = node_t::extract(root_, std::forward<comparable_at>(comparable));
        root_ = result.root;
        size_ -= result.extracted != nullptr;
        return extract_result_t {this, result.extracted.release()};
    }

    template <typename comparable_at>
    bool erase(comparable_at&& comparable) noexcept {
        return !!extract(std::forward<comparable_at>(comparable));
    }

    void clear() noexcept {
        node_t::for_each_bottom_up(root_, [&](node_t* node) noexcept { return allocator_.deallocate(node, 1); });
        root_ = nullptr;
        size_ = 0;
    }

    template <typename callback_at>
    void for_each(callback_at&& callback) noexcept {
        node_t::for_each_bottom_up(root_, [&](node_t* node) noexcept { callback(node->entry); });
    }

    void merge(avl_tree_t& other) noexcept {
        node_t::for_each_bottom_up(other.root_, [&](node_t* node) noexcept {
            auto result = node_t::insert(root_, node);
            root_ = result.root;
            size_ += result.inserted;
        });
        other.root_ = nullptr;
        other.size_ = 0;
    }

    void merge(extract_result_t other) noexcept {
        if (!other.node_ptr_)
            return;
        auto result = node_t::insert(root_, other.release());
        root_ = result.root;
        size_ += result.inserted;
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
    using extract_result_t = typename entry_set_t::extract_result_t;

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
        bool is_snapshot_ {false};

        transaction_t(store_t& set) noexcept : store_(&set), generation_(set.new_generation()) {}
        watch_t missing_watch() const noexcept { return watch_t {generation_, true}; }
        store_t& store_ref() noexcept { return *store_; }
        store_t const& store_ref() const noexcept { return *store_; }

      public:
        transaction_t(transaction_t&&) noexcept = default;
        transaction_t& operator=(transaction_t&&) noexcept = default;
        transaction_t(transaction_t const&) = delete;
        transaction_t& operator=(transaction_t const&) = delete;
        generation_t generation() const noexcept { return generation_; }

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

        [[nodiscard]] status_t reserve(std::size_t size) noexcept {
            return invoke_safely([&] { watches_.reserve(size); });
        }

        [[nodiscard]] status_t watch(identifier_t const& id) noexcept {
            auto found = [&](entry_t const& entry) noexcept {
                watches_.push_back({identifier_t {entry.element}, watch_t {entry.generation, entry.deleted}});
            };
            auto missing = [&]() noexcept {
                watches_.push_back({id, missing_watch()});
            };
            return store_ref().find(id, found, missing);
        }

        [[nodiscard]] status_t watch(entry_t const& entry) noexcept {
            watches_.push_back({identifier_t {entry.element}, watch_t {entry.generation, entry.deleted}});
        }

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t find(comparable_at&& comparable,
                                    callback_found_at&& callback_found,
                                    callback_missing_at&& callback_missing = {}) const noexcept {
            if (auto iterator = changes_.find(std::forward<comparable_at>(comparable)); iterator != changes_.end()) {
                !iterator->entry.deleted ? callback_found(iterator->entry) : callback_missing();
                return {success_k};
            }
            else
                return store_ref().find(std::forward<comparable_at>(comparable),
                                        std::forward<callback_found_at>(callback_found),
                                        std::forward<callback_missing_at>(callback_missing));
        }

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_at&& comparable,
                                           callback_found_at&& callback_found,
                                           callback_missing_at&& callback_missing = {}) const noexcept {
            auto external_previous_id = identifier_t(comparable);
            auto internal_iterator = changes_.upper_bound(std::forward<comparable_at>(comparable));
            while (internal_iterator != changes_.end() && internal_iterator->entry.deleted)
                ++internal_iterator;

            // Once picking the next smallest element from the global store,
            // we might face an entry, that was already deleted from here,
            // so this might become a multi-step process.
            auto faced_deleted_entry = false;
            auto callback_external_found = [&](element_t const& external_element) {
                // The simplest case is when we have an external object.
                if (internal_iterator == changes_.end())
                    return callback_found(external_element);

                element_t const& internal_element = internal_iterator->entry;
                if (!entry_comparator_t {}(external_element, internal_element))
                    return callback_found(internal_element);

                // Check if this entry was deleted and we should try again.
                auto external_id = identifier_t(external_element);
                auto external_element_internal_state = changes_.find(external_element);
                if (external_element_internal_state != changes_.end() &&
                    external_element_internal_state->entry.deleted) {
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
                    element_t const& internal_element = internal_iterator->entry;
                    return callback_found(internal_element);
                }
            };

            // Iterate until we find the a non-deleted external value
            auto& store = store_ref();
            auto status = status_t {};
            do {
                status = store.upper_bound(external_previous_id, callback_external_found, callback_external_missing);
            } while (faced_deleted_entry && status);
            return status;
        }

        [[nodiscard]] status_t stage() noexcept {
            // First, check if we have any collisions.
            auto& store = store_ref();
            auto entry_missing = missing_watch();
            for (auto const& id_and_watch : watches_) {
                auto consistency_violated = false;
                auto status = store.find(
                    id_and_watch.id,
                    [&](entry_t const& entry) noexcept { consistency_violated = entry != id_and_watch.watch; },
                    [&]() noexcept { consistency_violated = entry_missing != id_and_watch.watch; });
                if (consistency_violated)
                    return {errc_t::consistency_k};
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
            changes_.for_each([&](entry_t const& entry) noexcept {
                watches_.push_back({identifier_t {entry.element}, watch_t {generation_, entry.deleted}});
            });

            // Than just merge our current nodes.
            // The visibility will be updated later in the `commit`.
            store.entries_.merge(changes_);
            stage_ = stage_t::staged_k;
            return {success_k};
        }

        [[nodiscard]] status_t reset() noexcept {
            // If the transaction was "staged",
            // we must delete all the entries.
            auto& store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const& id_and_watch : watches_)
                    store.entries_.erase(dated_identifier_t {id_and_watch.id, id_and_watch.watch.generation});

            watches_.clear();
            changes_.clear();
            stage_ = stage_t::created_k;
            generation_ = store.new_generation();
            return {success_k};
        }

        [[nodiscard]] status_t rollback() noexcept {
            if (stage_ != stage_t::staged_k)
                return {operation_not_permitted_k};

            // If the transaction was "staged",
            // we must delete all the entries.
            auto& store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const& id_and_watch : watches_)
                    changes_.merge(
                        store.entries_.extract(dated_identifier_t {id_and_watch.id, id_and_watch.watch.generation}));

            watches_.clear();
            stage_ = stage_t::created_k;
            generation_ = store.new_generation();
            return {success_k};
        }

        [[nodiscard]] status_t commit() noexcept {
            if (stage_ != stage_t::staged_k)
                return {operation_not_permitted_k};

            // Once we make an entry visible,
            // if there are more than one with the same key,
            // the older generation must die.
            auto& store = store_ref();
            for (auto const& id_and_watch : watches_)
                store.unmask_and_compact(id_and_watch.id, id_and_watch.watch.generation);

            stage_ = stage_t::created_k;
            return {success_k};
        }
    };

  private:
    entry_set_t entries_;
    generation_t generation_ {0};
    std::size_t visible_count_ {0};

    friend class transaction_t;
    generation_t new_generation() noexcept { return ++generation_; }

    void unmask_and_compact(identifier_t const& id, generation_t generation_to_unmask) noexcept {
        // This is similar to the public `erase_range()`, but adds generation-matching conditions.
        auto current = entries_.lower_bound(id);
        auto less = entry_comparator_t {};
        auto last_visible_entry = std::optional<dated_identifier_t> {};
        while (current && less.same(id, current->entry.element)) {
            auto next = entries_.upper_bound(current->entry);
            current->entry.visible |= current->entry.generation == generation_to_unmask;
            if (!current->entry.visible) {
                current = next;
                continue;
            }

            // Older revisions must die
            if (last_visible_entry)
                entries_.extract(*last_visible_entry);
            last_visible_entry = dated_identifier_t {id, current->entry.generation};
            current = next;
        }
    }

  public:
    consistent_avl_gt() noexcept {}
    consistent_avl_gt(consistent_avl_gt&& other) noexcept
        : entries_(std::move(other.entries_)), generation_(other.generation_), visible_count_(other.visible_count_) {}

    consistent_avl_gt& operator=(consistent_avl_gt&& other) noexcept {
        entries_ = std::move(other.entries_);
        generation_ = other.generation_;
        visible_count_ = other.visible_count_;
        return *this;
    }

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] static std::optional<store_t> make(allocator_t&& allocator = {}) noexcept { return store_t {}; }
    [[nodiscard]] std::optional<transaction_t> transaction() noexcept { return transaction_t {*this}; }

    [[nodiscard]] status_t upsert(element_t&& element) noexcept {
        auto node = entries_.allocator().allocate(1);
        if (!node)
            return {out_of_memory_heap_k};

        identifier_t id {element};
        generation_t generation = new_generation();
        auto& entry = node->entry;
        new (&entry.element) element_t(std::move(element));
        entry.generation = generation;
        entry.deleted = false;
        entry.visible = true;
        entries_.merge(extract_result_t {&entries_, node});
        ++visible_count_;

        return erase_range(id, dated_identifier_t {id, generation});
    }

    template <typename elements_begin_at, typename elements_end_at = elements_begin_at>
    [[nodiscard]] status_t upsert(elements_begin_at begin, elements_end_at end) noexcept {

        // To make such batch insertions cheaper and easier until we have fast joins,
        // we can build a linked-list of pre-allocated nodes. Populate them and insert
        // one-by-one with the same generation.
        std::size_t const count = end - begin;
        std::size_t count_remaining = count;
        entry_node_t* last_node = nullptr;
        while (count_remaining) {
            entry_node_t* next_node = entries_.allocator().allocate(1);
            if (!next_node)
                break;
            // Reset the state
            next_node->right = nullptr;
            // Link for future iteration
            if (last_node)
                last_node->right = next_node;
            next_node->left = last_node;
            // Update state for next loop cycle
            last_node = next_node;
            count_remaining--;
        }

        // We have failed to allocate all the needed nodes.
        if (count_remaining) {
            while (count_remaining != count) {
                entry_node_t* prev_node = last_node->left;
                entries_.allocator().deallocate(last_node, 1);
                // Update state for next loop cycle
                last_node = prev_node;
                ++count_remaining;
            }
            return {out_of_memory_heap_k};
        }

        // Populate the allocated nodes and merge into the tree.
        generation_t generation = new_generation();
        while (count_remaining != count) {
            entry_node_t* prev_node = last_node->left;
            last_node->left = nullptr;
            last_node->right = nullptr;

            auto& entry = last_node->entry;
            new (&entry.element) element_t(*begin);
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            entries_.merge(extract_result_t {&entries_, last_node});
            ++visible_count_;

            // Remove older revisions
            identifier_t id {entry.element};
            auto status = erase_range(id, dated_identifier_t {id, generation});
            if (!status)
                return status;

            // Update state for next loop cycle
            last_node = prev_node;
            ++count_remaining;
            ++begin;
        }

        return {success_k};
    }

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find(comparable_at&& comparable,
                                callback_found_at&& callback_found,
                                callback_missing_at&& callback_missing = {}) const noexcept {

        entry_node_t* largest_visible = nullptr;
        entry_node_t::range(entries_.root(), comparable, comparable, [&](entry_node_t* node) noexcept {
            if ((node->entry.visible) &&
                (!largest_visible || node->entry.generation > largest_visible->entry.generation))
                largest_visible = node;
        });

        // static_assert(noexcept(callback_found(largest_visible->entry)));
        // static_assert(noexcept(callback_missing()));
        largest_visible ? callback_found(largest_visible->entry) : callback_missing();
        return {success_k};
    }

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_at&& comparable,
                                       callback_found_at&& callback_found,
                                       callback_missing_at&& callback_missing = {}) const noexcept {

        // Skip all the invisible entries
        entry_node_t* next_visible = entry_node_t::upper_bound(entries_.root(), comparable);
        while (next_visible && !next_visible->entry.visible)
            next_visible = entry_node_t::upper_bound(entries_.root(), next_visible->entry);

        // static_assert(noexcept(callback_found(next_visible->entry)));
        // static_assert(noexcept(callback_missing()));
        next_visible ? callback_found(next_visible->entry) : callback_missing();
        return {success_k};
    }

    template <typename lower_at = identifier_t, typename upper_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t range(lower_at&& lower, upper_at&& upper, callback_at&& callback) const noexcept {
        entry_node_t::range(entries_.root(),
                            std::forward<lower_at>(lower),
                            std::forward<upper_at>(upper),
                            [&](entry_node_t* node) noexcept {
                                if (node->entry.visible)
                                    callback(node->entry.element);
                                static_assert(noexcept(callback(node->entry.element)));
                            });
        return {success_k};
    }

    template <typename lower_at = identifier_t, typename upper_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t range(lower_at&& lower, upper_at&& upper, callback_at&& callback) noexcept {
        generation_t generation = new_generation();
        entry_node_t::range(entries_.root(),
                            std::forward<lower_at>(lower),
                            std::forward<upper_at>(upper),
                            [&](entry_node_t* node) noexcept {
                                if (node->entry.visible)
                                    callback(node->entry.element), node->entry.generation = generation;
                                static_assert(noexcept(callback(node->entry.element)));
                            });
        return {success_k};
    }

    template <typename lower_at = identifier_t, typename upper_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t erase_range(lower_at&& lower, upper_at&& upper, callback_at&& callback = {}) noexcept {
        // Implementing Splits and Joins for AVL can be tricky.
        // Let's start with deleting them one by one.
        // TODO: Implement range-removals.
        auto last = entries_.lower_bound(std::forward<lower_at>(lower));
        auto less = entry_comparator_t {};
        while (last != entries_.end() && less(last->entry, upper)) {
            auto next = entries_.upper_bound(last->entry);
            if (last->entry.visible)
                entries_.extract(last->entry);
            last = next;
        }
        return {success_k};
    }

    template <typename lower_at, typename upper_at, typename generator_at, typename callback_at = no_op_t>
    [[nodiscard]] status_t sample_range(lower_at&& lower,
                                        upper_at&& upper,
                                        generator_at&& generator,
                                        callback_at&& callback) const noexcept {

        auto node = entry_node_t::sample_range( //
            entries_.root(),
            lower,
            upper,
            std::forward<generator_at>(generator),
            [](entry_node_t* node) noexcept { return node->entry.visible; });
        if (node)
            callback(node->entry);
        return {success_k};
    }

    template <typename lower_at, typename upper_at, typename generator_at, typename output_iterator_at>
    [[nodiscard]] status_t sample_range(lower_at&& lower,
                                        upper_at&& upper,
                                        generator_at&& generator,
                                        std::size_t& seen,
                                        std::size_t reservoir_capacity,
                                        output_iterator_at&& reservoir) const noexcept {

        using output_iterator_t = std::remove_reference_t<output_iterator_at>;
        using output_category_t = typename std::iterator_traits<output_iterator_t>::iterator_category;
        static_assert(std::is_same<std::random_access_iterator_tag, output_category_t>(), "Must be random access!");

        auto sampler = [&](element_t const& element) noexcept {
            if (seen < reservoir_capacity)
                reservoir[seen] = element;

            else {
                std::uniform_int_distribution<std::size_t> distribution {0, seen};
                auto slot_to_replace = distribution(generator);
                if (slot_to_replace < reservoir_capacity)
                    reservoir[slot_to_replace] = element;
            }

            ++seen;
        };
        return range(std::forward<lower_at>(lower), std::forward<upper_at>(upper), sampler);
    }

    [[nodiscard]] status_t clear() noexcept {
        entries_.clear();
        generation_ = 0;
        return {success_k};
    }

    template <typename dont_instantiate_me_at>
    void print(dont_instantiate_me_at& cout) {
        cout << "Items: " << entries_.size() << std::endl;
        cout << "Imbalance: " << entries_.total_imbalance() << std::endl;
        entry_node_t::for_each_left_right(entries_.root(), [&](entry_node_t* node) {
            char const* marker = node->entry.visible ? "✓" : "✗";
            cout << identifier_t {node->entry.element} << " @" << node->entry.generation << marker << " ";
        });
        cout << std::endl;
    }
};

} // namespace unum::ucset