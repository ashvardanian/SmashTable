/**
 *  @brief  Ordered "Adelson-Velsky and Landis" @b (AVL) Binary Search Tree implementation.
 *    Provides exception-free, allocator-aware ordered collection similar to @c std::set.
 *    Not thread-safe by itself. Doesn't raise any exceptions unlike STL-based alternatives.
 *
 *  @section Design Characteristics
 *
 *  AVL trees maintain strict balance (height difference ≤ 1), providing O(log n) worst-case lookups, insertions,
 *  and deletions. Compared to Red-Black trees (used in std::set), AVL trees are more rigidly balanced, offering
 *  faster lookups at the cost of slightly slower insertions and deletions due to more frequent rebalancing.
 *
 *  This implementation supports heterogeneous lookups (searching with types other than @c entry_type_), custom
 *  allocators for all internal nodes, and callback-based iteration to avoid iterator invalidation complexity.
 *  All methods are @c noexcept and use status codes instead of exceptions for error handling.
 *
 *  @file   basic_avl_tree.hpp
 *  @author Ash Vardanian
 *  @see    https://en.wikipedia.org/wiki/AVL_tree
 */
#pragma once
#include <cassert>   // `assert`
#include <algorithm> // `std::max`
#include <memory>    // `std::allocator`
#include <optional>  // `std::optional`
#include <random>    // `std::uniform_int_distribution`
#include <utility>   // `std::exchange`

#include "status.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief  Tree node structure for the AVL-Tree implementation.
 *    Provides static methods for all tree operations: search, insertion, deletion, and traversal.
 *    All operations are exception-free and @c noexcept.
 *
 *  @section Implementation Notes
 *
 *  AVL trees are self-balancing binary search trees where the height difference between left and right subtrees
 *  is at most 1. This "node" class implements the core tree logic including rotations and rebalancing, but
 *  doesn't participate in memory management or provide atomicity guarantees - those are handled by @c basic_avl_tree.
 *
 *  Features:
 *  - Never throws exceptions, even on allocation failure
 *  - Implements @c lower_bound and @c upper_bound for iterator-free navigation
 *  - Supports random sampling within ranges for statistical operations
 *  - All methods are static and work on raw node pointers for flexibility
 *
 *  @tparam entry_type_      Type of entries to store in this tree.
 *  @tparam comparator_type_ A comparator function object that overloads
 *    @code bool operator()(entry_type_, entry_type_) const @endcode.
 *    For heterogeneous lookups, define @code using is_transparent = void; @endcode inside the comparator.
 */
template <typename entry_type_, typename comparator_type_>
class basic_avl_node {
  public:
    using entry_t = entry_type_;
    using comparator_t = comparator_type_;
    using height_t = std::int16_t;
    using node_t = basic_avl_node;

    entry_t entry;
    node_t *left = nullptr;
    node_t *right = nullptr;

    /**
     *  @brief Root has the biggest @c height in the tree.
     *  Zero is possible only in the uninitialized detached state.
     *  A non-NULL node would have height of one.
     *  Allows you to guess the upper bound of branch size, as @c 1 << height.
     */
    height_t height = 0;

    static height_t get_height(node_t *node) noexcept { return node ? node->height : 0; }
    static height_t get_balance(node_t *node) noexcept {
        return node ? get_height(node->left) - get_height(node->right) : 0;
    }

#pragma mark - Search

    template <typename callback_type_>
    static void for_each_top_down(node_t *node, callback_type_ &&callback) noexcept {
        if (!node) return;
        callback(node);
        for_each_top_down(node->left, callback);
        for_each_top_down(node->right, callback);
    }

    template <typename callback_type_>
    static void for_each_bottom_up(node_t *node, callback_type_ &&callback) noexcept {
        if (!node) return;
        for_each_bottom_up(node->left, callback);
        for_each_bottom_up(node->right, callback);
        callback(node);
    }

    template <typename callback_type_>
    static void for_each_left_right(node_t *node, callback_type_ &&callback) noexcept {
        if (!node) return;
        for_each_left_right(node->left, callback);
        callback(node);
        for_each_left_right(node->right, callback);
    }

    static node_t *find_min(node_t *node) noexcept {
        if (!node) return nullptr;
        while (node->left) node = node->left;
        return node;
    }

    static node_t *find_max(node_t *node) noexcept {
        if (!node) return nullptr;
        while (node->right) node = node->right;
        return node;
    }

    /**
     *  @brief Finds the next node in in-order traversal (successor).
     *  @param[in] root Root of the tree.
     *  @param[in] node Current node.
     *  @return node_t* Successor node, or nullptr if node is the maximum.
     */
    static node_t *find_successor(node_t *root, node_t *node) noexcept {
        if (!node) return find_min(root);
        return upper_bound(root, node->entry);
    }

    /**
     *  @brief Finds the previous node in in-order traversal (predecessor).
     *  @param[in] root  Root of the tree.
     *  @param[in] node  Current node.
     *  @return node_t* Predecessor node, or nullptr if node is the minimum.
     */
    static node_t *find_predecessor(node_t *root, node_t *node) noexcept {
        if (!node) return find_max(root);

        node_t *predecessor = nullptr;
        comparator_t less;
        node_t *current = root;

        while (current) {
            // Current is less than target, it's a candidate predecessor
            if (less(current->entry, node->entry)) {
                predecessor = current;
                current = current->right;
            }
            // Current is >= target, search left subtree
            else current = current->left;
        }
        return predecessor;
    }

    /**
     *  @brief Searches for equal entry in this subtree.
     *  @param comparable Any key comparable with stored entries.
     *  @return NULL if nothing was found.
     */
    template <typename comparable_type_>
    static node_t *find(node_t *node, comparable_type_ &&comparable) noexcept {
        auto less = comparator_t {};
        while (node) {
            if (less(comparable, node->entry)) node = node->left;
            else if (less(node->entry, comparable)) node = node->right;
            else break;
        }
        return node;
    }

    /**
     *  @brief Find the smallest entry, bigger than or equal to the provided one.
     *  @param comparable Any key comparable with stored entries.
     *  @return NULL if nothing was found.
     */
    template <typename comparable_type_>
    static node_t *lower_bound(node_t *node, comparable_type_ &&comparable) noexcept {
        node_t *successor = nullptr;
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
            else if (less(node->entry, comparable)) { node = node->right; }

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
     *  @brief Find the smallest entry, bigger than the provided one.
     *  @param comparable Any key comparable with stored entries.
     *  @return NULL if nothing was found.
     *
     *  Is used for an atomic implementation of iterators.
     *  Alternatively one can:
     *  > store a stack for path, which is ~O(logN) space.
     *  > store parents in nodes and have complex logic.
     */
    template <typename comparable_type_>
    static node_t *upper_bound(node_t *node, comparable_type_ &&comparable) noexcept {
        node_t *successor = nullptr;
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
            else if (less(node->entry, comparable)) node = node->right;

            // If a node with the desired value is found, the successor is the
            // minimum value node in its right subtree (if any).
            else {
                if (node->right) successor = find_min(node->right);
                node = nullptr;
            }
        }
        return successor;
    }

    /**
     *  @brief Searches for the shortest node, that is ancestor of both provided keys.
     *  @return NULL if nothing was found.
     *  @warning Current recursive implementation is suboptimal.
     */
    template <typename comparable_a_type_, typename comparable_b_type_>
    static node_t *lowest_common_ancestor(node_t *node, comparable_a_type_ &&a, comparable_b_type_ &&b) noexcept {
        if (!node) return nullptr;

        auto less = comparator_t {};
        // If both `a` and `b` are smaller than `node`, then LCA lies in left
        if (less(a, node->entry) && less(b, node->entry)) return lowest_common_ancestor(node->left, a, b);

        // If both `a` and `b` are greater than `node`, then LCA lies in right
        if (less(node->entry, a) && less(node->entry, b)) return lowest_common_ancestor(node->right, a, b);

        return node;
    }

    struct node_interval_t {
        node_t *lower_bound = nullptr;
        node_t *upper_bound = nullptr;
        node_t *lowest_common_ancestor = nullptr;
    };

    /**
     *  @brief Complex method, that detects the left-most and right-most nodes
     *  containing keys in a provided intervals, as well as their lowest common ancestors.
     *  @warning Current recursive implementation is suboptimal.
     */
    template <typename lower_type_, typename upper_type_, typename callback_type_>
    static node_interval_t range(node_t *node, lower_type_ &&low, upper_type_ &&high,
                                 callback_type_ &&callback) noexcept {
        if (!node) return {};

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

        if (less(node->entry, low)) return range(node->right, low, high, callback);

        return range(node->left, low, high, callback);
    }

    template <typename comparable_type_>
    static node_interval_t equal_range(node_t *node, comparable_type_ &&comparable) noexcept {
        return range(node, comparable, comparable);
    }

    /**
     *  @brief Random samples nodes.
     *  @param generator Any STL-compatible random number generator.
     *  @return NULL if nothing was found.
     *  @warning Resulting distribution is inaccurate, as we only have the upper bound of the branch size.
     */
    template <typename generator_type_>
    static node_t *sample(node_t *node, generator_type_ &&generator) noexcept {
        auto less = comparator_t {};
        while (node) {
            auto count_left = node->left ? 1ul << node->left->height : 0ul;
            auto count_right = node->right ? 1ul << node->right->height : 0ul;
            auto count_total = count_left + count_right + 1ul;
            std::uniform_int_distribution<std::size_t> distribution {0, count_total + 1};
            auto choice = distribution(generator);
            if (choice == 0) break;

            node = choice > (count_left + 1ul) ? node->right : node->left;
        }
        return node;
    }

    /**
     *  @brief Random samples nodes within a given range of keys.
     *  @param generator Any STL-compatible random number generator.
     *  @return NULL if nothing was found.
     *  @warning Without additional stored metadata or dynamic memory, this algorithm performs two passes.
     */
    template <typename generator_type_, typename lower_type_, typename upper_type_, typename predicate_type_>
    static node_t *sample_range( //
        node_t *node, lower_type_ &&low, upper_type_ &&high, generator_type_ &&generator,
        predicate_type_ &&predicate) noexcept {

        std::size_t count_matches = 0;
        range(node, low, high, [&](node_t *node) noexcept { count_matches += predicate(node); });

        if (count_matches == 0) return nullptr;

        node_t *result = nullptr;
        std::uniform_int_distribution<std::size_t> distribution {0, count_matches - 1};
        auto choice = distribution(generator);
        range(node, low, high, [&](node_t *node) noexcept {
            if (!predicate(node)) return;
            if (choice == 0) result = node;
            --choice;
        });

        return result;
    }

#pragma mark - Insertions

    static node_t *rotate_right(node_t *y) noexcept {
        node_t *x = y->left;
        node_t *z = x->right;

        // Perform rotation
        x->right = y;
        y->left = z;

        // Update heights
        y->height = std::max(get_height(y->left), get_height(y->right)) + 1;
        x->height = std::max(get_height(x->left), get_height(x->right)) + 1;
        return x;
    }

    static node_t *rotate_left(node_t *x) noexcept {
        node_t *y = x->right;
        node_t *z = y->left;

        // Perform rotation
        y->left = x;
        x->right = z;

        // Update heights
        x->height = std::max(get_height(x->left), get_height(x->right)) + 1;
        y->height = std::max(get_height(y->left), get_height(y->right)) + 1;
        return y;
    }

    struct find_or_make_result_t {
        node_t *root = nullptr;
        node_t *match = nullptr;
        bool inserted = false;

        /**
         *  @return True if the allocation of the new node has failed.
         */
        bool failed() const noexcept { return !inserted && !match; }
    };

    template <typename comparable_type_>
    inline static node_t *rebalance_after_insert(node_t *node, comparable_type_ &&comparable) noexcept {
        // Update height and check if branches aren't balanced
        node->height = std::max(get_height(node->left), get_height(node->right)) + 1;
        auto balance = get_balance(node);
        auto less = comparator_t {};

        // Left Left Case
        if (balance > 1 && less(comparable, node->left->entry)) return rotate_right(node);

        // Right Right Case
        else if (balance < -1 && less(node->right->entry, comparable)) return rotate_left(node);

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
        else return node;
    }

    template <typename comparable_type_, typename callback_found_type_, typename callback_make_type_>
    static find_or_make_result_t find_or_make(node_t *node, comparable_type_ &&comparable,
                                              callback_found_type_ &&callback_found,
                                              callback_make_type_ &&callback_make) noexcept {
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
            if (downstream.inserted) node = rebalance_after_insert(node, downstream.match->entry);
            return {node, downstream.match, downstream.inserted};
        }
        else if (less(node->entry, comparable)) {
            auto downstream = find_or_make(node->right, comparable, callback_found, callback_make);
            node->right = downstream.root;
            if (downstream.inserted) node = rebalance_after_insert(node, downstream.match->entry);
            return {node, downstream.match, downstream.inserted};
        }
        else {
            // Equal keys are not allowed in BST
            callback_found(node);
            return {node, node, false};
        }
    }

    template <typename node_allocator_type_>
    static find_or_make_result_t insert(node_t *node, entry_t &&entry, node_allocator_type_ &&node_allocator) noexcept {
        auto found = [&](node_t *node) noexcept {};
        auto make = [&]() noexcept -> node_t * {
            auto node = node_allocator();
            if (node) new (&node->entry) entry_t(std::move(entry));
            return node;
        };
        auto result = find_or_make(node, entry, found, make);
        return result;
    }

    template <typename node_allocator_type_>
    static find_or_make_result_t upsert(node_t *node, entry_t &&entry, node_allocator_type_ &&node_allocator) noexcept {
        auto found = [&](node_t *node) noexcept { node->entry = std::move(entry); };
        auto make = [&]() noexcept -> node_t * {
            auto node = node_allocator();
            if (node) new (&node->entry) entry_t(std::move(entry));
            return node;
        };
        auto result = find_or_make(node, entry, found, make);
        return result;
    }

    static find_or_make_result_t insert(node_t *node, node_t *new_child) noexcept {
        return find_or_make(node, new_child->entry, [](node_t *) noexcept {}, [=]() noexcept { return new_child; });
    }

#pragma mark - Removals

    struct extract_result_t {
        node_t *root = nullptr;
        std::unique_ptr<node_t> extracted;

        node_t *release() noexcept { return extracted.release(); }
    };

    static node_t *rebalance_after_extract(node_t *node) noexcept {
        node->height = 1 + std::max(get_height(node->left), get_height(node->right));
        auto balance = get_balance(node);

        // Left Left Case
        if (balance > 1 && get_balance(node->left) >= 0) return rotate_right(node);

        // Left Right Case
        else if (balance > 1 && get_balance(node->left) < 0) {
            node->left = rotate_left(node->left);
            return rotate_right(node);
        }

        // Right Right Case
        else if (balance < -1 && get_balance(node->right) <= 0) return rotate_left(node);

        // Right Left Case
        else if (balance < -1 && get_balance(node->right) > 0) {
            node->right = rotate_right(node->right);
            return rotate_left(node);
        }
        else return node;
    }

    /**
     *  @brief Pops the root replacing it with one of descendants, if present.
     *  @param comparable Any key comparable with stored entries.
     */
    static extract_result_t extract(node_t *node) noexcept {

        // If the node has two children, replace it with the
        // smallest entry in the right branch.
        if (node->left && node->right) {
            node_t *midpoint = find_min(node->right);
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
            node_t *replacement = node->left ? node->left : node->right;
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
     *  @brief Searches for a matching ancestor and extracts it out.
     *  @param comparable Any key comparable with stored entries.
     */
    template <typename comparable_type_>
    static extract_result_t extract(node_t *node, comparable_type_ &&comparable) noexcept {
        if (!node) return {node, {}};

        auto less = comparator_t {};
        if (less(comparable, node->entry)) {
            auto downstream = extract(node->left, comparable);
            node->left = downstream.root;
            if (downstream.extracted) node = rebalance_after_extract(node);
            return {node, std::move(downstream.extracted)};
        }

        else if (less(node->entry, comparable)) {
            auto downstream = extract(node->right, comparable);
            node->right = downstream.root;
            if (downstream.extracted) node = rebalance_after_extract(node);
            return {node, std::move(downstream.extracted)};
        }

        else
            // We have found the node to extract!
            return extract(node);
    }

    struct remove_if_result_t {
        node_t *root = nullptr;
        std::size_t count = 0;
    };

    template <typename predicate_type_, typename node_deallocator_type_>
    static remove_if_result_t remove_if(node_t *node, predicate_type_ &&predicate,
                                        node_deallocator_type_ &&node_deallocator) noexcept {
        return {};
    }
};

/**
 *  @brief  Exception-free AVL tree container providing ordered storage similar to @c std::set.
 *    Manages memory allocation and provides high-level tree operations with status-based error handling.
 *
 *  @section API Design
 *
 *  This AVL tree provides an STL-compatible interface:
 *  - Bidirectional iterators for in-order traversal (begin/end/rbegin/rend)
 *  - Returns @c status_t for some operations instead of throwing exceptions
 *  - Supports heterogeneous lookups when comparator defines @c is_transparent
 *  - Provides both @c insert (fails if exists) and @c upsert (always succeeds) semantics
 *  - Allows custom allocators for all internal nodes
 *  - Iterator invalidation: only iterators to erased elements are invalidated
 *
 *  Callback-based APIs are also provided for convenience (find, range, equal_range).
 *  The tree maintains O(log n) height through automatic rebalancing after modifications.
 *
 *  @tparam entry_type_ Type of entries stored in the tree.
 *  @tparam comparator_type_ Comparator for ordering entries. Define @c is_transparent for heterogeneous lookups.
 *  @tparam node_allocator_type_ Allocator for tree nodes. Must be rebindable to @c basic_avl_node.
 */
template <typename entry_type_, typename comparator_type_,
          typename node_allocator_type_ = std::allocator<basic_avl_node<entry_type_, comparator_type_>>>
class basic_avl_tree {
  public:
    using node_t = basic_avl_node<entry_type_, comparator_type_>;
    using node_allocator_t = node_allocator_type_;
    using comparator_t = comparator_type_;
    using entry_t = entry_type_;
    using avl_tree_t = basic_avl_tree;

    // Forward declare iterator
    class iterator;
    class const_iterator;

    /**
     *  @brief Result of an erase operation on an iterator.
     *    Combines iterator to next element with operation status.
     */
    struct erase_result_t {
        iterator next;   //!< Iterator to the element following the erased element (or end()).
        status_t status; //!< Status of the erase operation.
    };

    /**
     *  @brief Bidirectional iterator for AVL tree.
     *    Provides in-order traversal of tree elements.
     */
    class iterator {
        friend class basic_avl_tree;
        friend class const_iterator;

      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = entry_t;
        using difference_type = std::ptrdiff_t;
        using pointer = entry_t *;
        using reference = entry_t &;

      private:
        basic_avl_tree const *tree_;
        node_t *node_;

        iterator(basic_avl_tree const *tree, node_t *node) noexcept : tree_(tree), node_(node) {}

      public:
        iterator() noexcept : tree_(nullptr), node_(nullptr) {}

        reference operator*() const noexcept { return node_->entry; }
        pointer operator->() const noexcept { return &node_->entry; }

        iterator &operator++() noexcept {
            node_ = node_t::find_successor(tree_->root_, node_);
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        iterator &operator--() noexcept {
            node_ = node_t::find_predecessor(tree_->root_, node_);
            return *this;
        }

        iterator operator--(int) noexcept {
            iterator tmp = *this;
            --(*this);
            return tmp;
        }

        bool operator==(iterator const &other) const noexcept { return node_ == other.node_; }
        bool operator!=(iterator const &other) const noexcept { return node_ != other.node_; }
    };

    /**
     *  @brief Const bidirectional iterator for AVL tree.
     *    Provides in-order traversal of tree elements (read-only).
     */
    class const_iterator {
        friend class basic_avl_tree;

      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = entry_t const;
        using difference_type = std::ptrdiff_t;
        using pointer = entry_t const *;
        using reference = entry_t const &;

      private:
        basic_avl_tree const *tree_;
        node_t const *node_;

        const_iterator(basic_avl_tree const *tree, node_t const *node) noexcept : tree_(tree), node_(node) {}

      public:
        const_iterator() noexcept : tree_(nullptr), node_(nullptr) {}
        const_iterator(iterator const &it) noexcept : tree_(it.tree_), node_(it.node_) {}

        reference operator*() const noexcept { return node_->entry; }
        pointer operator->() const noexcept { return &node_->entry; }

        const_iterator &operator++() noexcept {
            node_ = node_t::find_successor(tree_->root_, const_cast<node_t *>(node_));
            return *this;
        }

        const_iterator operator++(int) noexcept {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        const_iterator &operator--() noexcept {
            node_ = node_t::find_predecessor(tree_->root_, const_cast<node_t *>(node_));
            return *this;
        }

        const_iterator operator--(int) noexcept {
            const_iterator tmp = *this;
            --(*this);
            return tmp;
        }

        bool operator==(const_iterator const &other) const noexcept { return node_ == other.node_; }
        bool operator!=(const_iterator const &other) const noexcept { return node_ != other.node_; }
    };

    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  private:
    node_t *root_ = nullptr;
    std::size_t size_ = 0;
    node_allocator_t allocator_;

  public:
    basic_avl_tree() noexcept = default;
    basic_avl_tree(basic_avl_tree &&other) noexcept
        : root_(std::exchange(other.root_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    basic_avl_tree &operator=(basic_avl_tree &&other) noexcept {
        std::swap(root_, other.root_);
        std::swap(size_, other.size_);
        return *this;
    }

    ~basic_avl_tree() { clear(); }

    /**
     *  @brief Returns the number of elements in the tree.
     *  @return std::size_t Number of elements.
     */
    std::size_t size() const noexcept { return size_; }

    /**
     *  @brief Returns the height of the tree (number of edges in longest path from root to leaf).
     *  @return std::size_t Height of the tree, 0 if empty.
     */
    std::size_t height() noexcept { return root_ ? root_->height : 0; }

    /**
     *  @brief Returns raw pointer to the root node. For internal use.
     *  @return node_t* Pointer to root, or nullptr if empty.
     */
    node_t *root() const noexcept { return root_; }

    /**
     *  @brief Returns the allocator associated with the tree.
     *  @return node_allocator_t& Reference to the allocator.
     */
    node_allocator_t &allocator() noexcept { return allocator_; }

    /**
     *  @brief Returns the allocator associated with the tree (const version).
     *  @return node_allocator_t const& Const reference to the allocator.
     */
    node_allocator_t const &allocator() const noexcept { return allocator_; }

    /**
     *  @brief Returns an iterator to the first element (minimum) in the tree.
     *  @return iterator Iterator to the minimum element, or end() if empty.
     */
    iterator begin() noexcept { return iterator(this, node_t::find_min(root_)); }

    /**
     *  @brief Returns a const iterator to the first element (minimum) in the tree.
     *  @return const_iterator Const iterator to the minimum element, or end() if empty.
     */
    const_iterator begin() const noexcept { return const_iterator(this, node_t::find_min(root_)); }

    /**
     *  @brief Returns a const iterator to the first element (minimum) in the tree.
     *  @return const_iterator Const iterator to the minimum element, or end() if empty.
     */
    const_iterator cbegin() const noexcept { return const_iterator(this, node_t::find_min(root_)); }

    /**
     *  @brief Returns an iterator to one past the last element.
     *  @return iterator End iterator (points to nullptr).
     */
    iterator end() noexcept { return iterator(this, nullptr); }

    /**
     *  @brief Returns a const iterator to one past the last element.
     *  @return const_iterator End iterator (points to nullptr).
     */
    const_iterator end() const noexcept { return const_iterator(this, nullptr); }

    /**
     *  @brief Returns a const iterator to one past the last element.
     *  @return const_iterator End iterator (points to nullptr).
     */
    const_iterator cend() const noexcept { return const_iterator(this, nullptr); }

    /**
     *  @brief Returns a reverse iterator to the first element of the reversed tree (maximum element).
     *  @return reverse_iterator Reverse iterator to the maximum element, or rend() if empty.
     */
    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

    /**
     *  @brief Returns a const reverse iterator to the first element of the reversed tree (maximum element).
     *  @return const_reverse_iterator Const reverse iterator to the maximum element, or rend() if empty.
     */
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

    /**
     *  @brief Returns a const reverse iterator to the first element of the reversed tree (maximum element).
     *  @return const_reverse_iterator Const reverse iterator to the maximum element, or rend() if empty.
     */
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }

    /**
     *  @brief Returns a reverse iterator to one past the last element of the reversed tree (before minimum).
     *  @return reverse_iterator Reverse end iterator.
     */
    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

    /**
     *  @brief Returns a const reverse iterator to one past the last element of the reversed tree (before minimum).
     *  @return const_reverse_iterator Const reverse end iterator.
     */
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

    /**
     *  @brief Returns a const reverse iterator to one past the last element of the reversed tree (before minimum).
     *  @return const_reverse_iterator Const reverse end iterator.
     */
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

    std::size_t total_imbalance() const noexcept {
        std::size_t abs_sum = 0;
        node_t::for_each_top_down(root_,
                                  [&](node_t *node) noexcept { abs_sum += std::abs(node_t::get_balance(node)); });
        return abs_sum;
    }

    /**
     *  @brief Finds an element equal to the given @p comparable.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return iterator Iterator to found element, or end() if not found.
     */
    template <typename comparable_type_>
    iterator find(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::find(root_, std::forward<comparable_type_>(comparable)));
    }

    /**
     *  @brief Finds an element equal to the given @p comparable (const version).
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return const_iterator Const iterator to found element, or end() if not found.
     */
    template <typename comparable_type_>
    const_iterator find(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this, node_t::find(root_, std::forward<comparable_type_>(comparable)));
    }

    /**
     *  @brief Finds the first element not less than (>=) the given @p comparable.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return iterator Iterator to found element, or end() if all elements are less.
     */
    template <typename comparable_type_>
    iterator lower_bound(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::lower_bound(root_, std::forward<comparable_type_>(comparable)));
    }

    /**
     *  @brief Finds the first element not less than (>=) the given @p comparable (const version).
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return const_iterator Const iterator to found element, or end() if all elements are less.
     */
    template <typename comparable_type_>
    const_iterator lower_bound(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this, node_t::lower_bound(root_, std::forward<comparable_type_>(comparable)));
    }

    /**
     *  @brief Finds the first element greater than (>) the given @p comparable.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return iterator Iterator to found element, or end() if no element is greater.
     */
    template <typename comparable_type_>
    iterator upper_bound(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::upper_bound(root_, std::forward<comparable_type_>(comparable)));
    }

    /**
     *  @brief Finds the first element greater than (>) the given @p comparable (const version).
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return const_iterator Const iterator to found element, or end() if no element is greater.
     */
    template <typename comparable_type_>
    const_iterator upper_bound(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this, node_t::upper_bound(root_, std::forward<comparable_type_>(comparable)));
    }

    /**
     *  @brief Checks if a member @b equal to the given @p comparable exists in the tree.
     *
     *  @param[in] comparable Object comparable to @c entry_t and convertible to search key.
     *  @return bool True if element exists, false otherwise.
     */
    template <typename comparable_type_>
    bool contains(comparable_type_ &&comparable) const noexcept {
        return find(std::forward<comparable_type_>(comparable)) != end();
    }

    /**
     *  @brief Returns the number of elements with key equal to the specified argument.
     *    For unique-key containers like this, returns either 0 or 1.
     *
     *  @param[in] comparable Object comparable to @c entry_t and convertible to search key.
     *  @return std::size_t Number of elements with key equal to @p comparable (0 or 1).
     */
    template <typename comparable_type_>
    std::size_t count(comparable_type_ &&comparable) const noexcept {
        return find(std::forward<comparable_type_>(comparable)) != end() ? 1 : 0;
    }

    /**
     *  @brief Checks if the tree has no elements.
     *  @return bool True if empty, false otherwise.
     */
    bool empty() const noexcept { return size_ == 0; }

    /**
     *  @brief Returns a range of elements matching a specific key.
     *    For unique-key containers, returns range containing at most one element.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return std::pair<iterator, iterator> Pair of iterators [first, last) where all elements are equal to key.
     *    If key not found, both iterators equal end().
     */
    template <typename comparable_type_>
    std::pair<iterator, iterator> equal_range(comparable_type_ &&comparable) noexcept {
        auto it = find(std::forward<comparable_type_>(comparable));
        if (it == end()) return {it, it};
        auto next = it;
        ++next;
        return {it, next};
    }

    /**
     *  @brief Returns a range of elements matching a specific key (const version).
     *    For unique-key containers, returns range containing at most one element.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return std::pair<const_iterator, const_iterator> Pair of const iterators [first, last).
     */
    template <typename comparable_type_>
    std::pair<const_iterator, const_iterator> equal_range(comparable_type_ &&comparable) const noexcept {
        auto it = find(std::forward<comparable_type_>(comparable));
        if (it == end()) return {it, it};
        auto next = it;
        ++next;
        return {it, next};
    }

    /**
     *  @brief Finds all elements equal to a single key. Invokes callback for each matching element.
     *    For trees with unique keys, this returns at most one element (0 or 1).
     *    Callback-based alternative to iterator-returning equal_range().
     *
     *  @param[in] comparable Object comparable to @c entry_t and convertible to search key.
     *  @param[in] callback   Callback invoked for each element equal to the key. Must be @c noexcept.
     */
    template <typename comparable_type_ = entry_t, typename callback_type_ = no_op_t>
    void equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
        auto it = find(std::forward<comparable_type_>(comparable));
        if (it != end()) { callback(*it); }
    }

    /**
     *  @brief Iterates over all entries in the range [@p lower, @p upper). Const version.
     *    Invokes callback for each element in the specified range.
     *
     *  @param[in] lower Lower bound of the range (inclusive).
     *  @param[in] upper Upper bound of the range (exclusive).
     *  @param[in] callback Callback invoked for each element in range. Must be @c noexcept.
     */
    template <typename lower_type_ = entry_t, typename upper_type_ = entry_t, typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        node_t::range(root_, std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                      [&](node_t *node) noexcept { callback(node->entry); });
    }

    /**
     *  @brief Iterates over all entries in the range [@p lower, @p upper), allowing in-place modification.
     *    Invokes callback for each mutable element in the specified range. Non-const version.
     *
     *  @param[in] lower Lower bound of the range (inclusive).
     *  @param[in] upper Upper bound of the range (exclusive).
     *  @param[inout] callback Callback invoked for each mutable element in range. Must be @c noexcept.
     */
    template <typename lower_type_ = entry_t, typename upper_type_ = entry_t, typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        node_t::range(root_, std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                      [&](node_t *node) noexcept { callback(node->entry); });
    }

    struct upsert_result_t {
        node_t *node = nullptr;
        bool inserted = false;

        /**
         *  @return True if the allocation of the new node has failed.
         */
        bool failed() const noexcept { return !inserted && !node; }
        upsert_result_t &operator=(entry_t &&entry) noexcept {
            node->entry = entry;
            return *this;
        }
    };

    /**
     *  @brief Inserts an entry only if the key doesn't exist. Matches @c std::set::insert() semantics.
     *    Fails if key already exists (does not overwrite).
     *
     *  @param[in] entry Entry to insert (moved into the tree).
     *  @return std::pair<iterator, bool> Pair of iterator to inserted/existing element and bool indicating success.
     *    Returns {iterator, true} if inserted successfully.
     *    Returns {iterator, false} if key already exists (iterator points to existing).
     *    Returns {end(), false} if allocation failed.
     */
    template <typename comparable_type_>
    std::pair<iterator, bool> insert(comparable_type_ &&comparable) noexcept {
        auto result = node_t::insert(root_, std::forward<comparable_type_>(comparable),
                                     [&]() noexcept { return allocator_.allocate(1); });
        root_ = result.root;
        size_ += result.inserted;
        if (result.failed()) return {end(), false};
        return {iterator(this, result.match), result.inserted};
    }

    /**
     *  @brief Atomically inserts an entry only if missing. Silently skips if key exists (no error).
     *    This is the "silent no-op" insert semantics.
     *
     *  @param[in] entry Entry to insert (moved into the tree).
     *  @return upsert_result_t Result containing pointer to node and insertion status.
     *    Always returns a valid @c node pointer (existing or new).
     *    @c inserted is true only if a new node was created.
     */
    template <typename comparable_type_>
    upsert_result_t insert_if_missing(comparable_type_ &&comparable) noexcept {
        return insert(std::forward<comparable_type_>(comparable));
    }

    /**
     *  @brief Atomically inserts or updates an entry. Always succeeds (unless OOM).
     *    Overwrites existing entry if key exists. Matches @c std::map::insert_or_assign() semantics.
     *
     *  @param[in] entry Entry to insert or assign (moved into the tree).
     *  @return upsert_result_t Result containing pointer to node and insertion status.
     *    @c inserted is true if new node was created, false if existing was updated.
     */
    template <typename comparable_type_>
    upsert_result_t insert_or_assign(comparable_type_ &&comparable) noexcept {
        auto result = node_t::upsert(root_, std::forward<comparable_type_>(comparable),
                                     [&]() noexcept { return allocator_.allocate(1); });
        root_ = result.root;
        size_ += result.inserted;
        return {result.match, result.inserted};
    }

    /**
     *  @brief Alias for @c insert_or_assign(). Atomically inserts or updates an entry.
     *  @param[in] entry Entry to insert or assign (moved into the tree).
     *  @return upsert_result_t Result containing pointer to node and insertion status.
     */
    template <typename comparable_type_>
    upsert_result_t upsert(comparable_type_ &&comparable) noexcept {
        return insert_or_assign(std::forward<comparable_type_>(comparable));
    }

    /**
     *  @brief Constructs an element in-place. Matches @c std::set::emplace() semantics.
     *         Does not insert if key already exists.
     *
     *  @tparam Args Types of arguments to forward to entry_t constructor.
     *  @param[in] args Arguments to forward to entry_t constructor.
     *  @return std::pair<iterator, bool> Pair of iterator to inserted/existing element and bool indicating success.
     */
    template <typename... Args>
    std::pair<iterator, bool> emplace(Args &&...args) noexcept {
        return insert(entry_t(std::forward<Args>(args)...));
    }

    /**
     *  @brief Deleted: Hint-based emplace is not supported.
     *    AVL trees don't benefit from position hints, and providing unused hints is misleading.
     *    Use @c emplace() instead.
     */
    template <typename... Args>
    iterator emplace_hint(const_iterator, Args &&...) noexcept = delete;

    /**
     *  @brief Deleted: Hint-based insert is not supported.
     *    AVL trees don't benefit from position hints, and providing unused hints is misleading.
     *    Use @c insert(value) instead.
     */
    iterator insert(const_iterator, entry_t const &) noexcept = delete;

    /**
     *  @brief Deleted: Hint-based insert is not supported.
     *    AVL trees don't benefit from position hints, and providing unused hints is misleading.
     *    Use @c insert(value) instead.
     */
    iterator insert(const_iterator, entry_t &&) noexcept = delete;

    /**
     *  @brief Inserts a range of entries.
     *    Inserts elements one-by-one until completion or first failure.
     *    On failure, some elements may have been inserted (partial insertion).
     *    This matches STL's basic exception guarantee semantics.
     *
     *  @tparam input_iterator_type_ Type of input iterator.
     *  @param[in] first Beginning of range to insert.
     *  @param[in] last End of range to insert.
     *  @return status_t First error encountered, or success if all elements inserted.
     */
    template <typename input_iterator_type_>
    status_t insert(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        for (; first != last; ++first) {
            auto result = insert(*first);
            if (result.first == end() && !result.second) return {errc_t::out_of_memory_heap_k};
        }
        return {success_k};
    }

    /**
     *  @brief Inserts elements from an initializer list.
     *    Inserts elements one-by-one until completion or first failure.
     *    On failure, some elements may have been inserted (partial insertion).
     *
     *  @param[in] ilist Initializer list of entries to insert.
     *  @return status_t First error encountered, or success if all elements inserted.
     */
    status_t insert(std::initializer_list<entry_t> ilist) noexcept { return insert(ilist.begin(), ilist.end()); }

    /**
     *  @brief Returns the function object that compares keys.
     *  @return comparator_t The comparison function object.
     */
    comparator_t key_comp() const noexcept { return comparator_t {}; }

    /**
     *  @brief Returns the function object that compares values.
     *         For sets, this is the same as key_comp().
     *  @return comparator_t The comparison function object.
     */
    comparator_t value_comp() const noexcept { return comparator_t {}; }

    /**
     *  @brief Returns the maximum possible number of elements.
     *  @return std::size_t Theoretical maximum size.
     */
    std::size_t max_size() const noexcept {
        return std::min(allocator_.max_size(), std::numeric_limits<std::size_t>::max() / sizeof(node_t));
    }

    /**
     *  @brief Exchanges the contents of this tree with another.
     *  @param[inout] other Tree to swap with.
     */
    void swap(basic_avl_tree &other) noexcept {
        std::swap(root_, other.root_);
        std::swap(size_, other.size_);
        std::swap(allocator_, other.allocator_);
    }

    struct extract_result_t {
        basic_avl_tree *tree_ = nullptr;
        node_t *node_ptr_ = nullptr;

        extract_result_t() = default;
        extract_result_t(basic_avl_tree *tree, node_t *node) noexcept : tree_(tree), node_ptr_(node) {}

        ~extract_result_t() noexcept {
            if (node_ptr_) tree_->allocator_.deallocate(node_ptr_, 1);
        }
        extract_result_t(extract_result_t const &) = delete;
        extract_result_t &operator=(extract_result_t const &) = delete;
        extract_result_t(extract_result_t &&other) noexcept
            : tree_(other.tree_), node_ptr_(std::exchange(other.node_ptr_, nullptr)) {}
        extract_result_t &operator=(extract_result_t &&other) noexcept {
            if (this != &other) {
                if (node_ptr_) tree_->allocator_.deallocate(node_ptr_, 1);
                tree_ = other.tree_;
                node_ptr_ = std::exchange(other.node_ptr_, nullptr);
            }
            return *this;
        }
        explicit operator bool() const noexcept { return node_ptr_; }
        node_t *release() noexcept { return std::exchange(node_ptr_, nullptr); }
    };

    template <typename comparable_type_>
    extract_result_t extract(comparable_type_ &&comparable) noexcept {
        auto result = node_t::extract(root_, std::forward<comparable_type_>(comparable));
        root_ = result.root;
        size_ -= result.extracted != nullptr;
        return extract_result_t {this, result.extracted.release()};
    }

    /**
     *  @brief Erases a single entry matching the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c entry_t and convertible to search key.
     *  @param[in] callback_found Callback to receive the erased entry. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = entry_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
               callback_missing_type_ &&callback_missing) noexcept {
        auto node = find(std::forward<comparable_type_>(comparable));
        if (!node) {
            callback_missing();
            return;
        }

        callback_found(node->entry);
        auto result = node_t::extract(root_, std::forward<comparable_type_>(comparable));
        root_ = result.root;
        size_ -= result.extracted != nullptr;
    }

    /**
     *  @brief Erases a single entry matching the given @p comparable. No callbacks.
     *
     *  @param[in] comparable Object comparable to @c entry_t and convertible to search key.
     *  @return bool True if element was erased, false if not found.
     */
    template <typename comparable_type_>
    bool erase(comparable_type_ &&comparable) noexcept {
        return !!extract(std::forward<comparable_type_>(comparable));
    }

    /**
     *  @brief Erases the element at the specified iterator position.
     *    Unlike STL, returns both the next iterator and a status code for error reporting.
     *
     *  @param[in] pos Iterator to element to erase. Must be valid and dereferenceable.
     *  @return erase_result_t Contains iterator to element following the erased element and operation status.
     *
     *  @note If @p pos is end(), returns {end(), success} without modifying the tree.
     *    If erase fails (e.g., tree corruption), returns {end(), error}.
     */
    erase_result_t erase(iterator pos) noexcept {
        if (pos == end()) return {end(), {success_k}};
        auto next = std::next(pos);
        bool erased = erase(*pos);
        return {next, erased ? status_t {success_k} : status_t {errc_t::unknown_k}};
    }

    /**
     *  @brief Erases the element at the specified const_iterator position.
     *    Unlike STL, returns both the next iterator and a status code for error reporting.
     *
     *  @param[in] pos Const iterator to element to erase. Must be valid and dereferenceable.
     *  @return erase_result_t Contains iterator to element following the erased element and operation status.
     *
     *  @note If @p pos is end(), returns {end(), success} without modifying the tree.
     *    If erase fails (e.g., tree corruption), returns {end(), error}.
     */
    erase_result_t erase(const_iterator pos) noexcept { return erase(iterator(this, pos.node_)); }

    /**
     *  @brief Erases all elements in the range [first, last).
     *    Unlike STL, returns both the iterator following the last erased element and a status code.
     *    On error, some elements may have been erased (partial erase, matches STL's basic guarantee).
     *
     *  @param[in] first Beginning of range to erase.
     *  @param[in] last End of range to erase (not erased).
     *  @return erase_result_t Contains iterator equal to @p last and status of the operation.
     *    Returns first error encountered, or success if all elements erased.
     */
    erase_result_t erase(iterator first, iterator last) noexcept {
        while (first != last) {
            auto result = erase(first);
            if (result.status.failed()) return {result.next, result.status};
            first = result.next;
        }
        return {last, {success_k}};
    }

    /**
     *  @brief Erases all elements in the range [first, last) using const_iterators.
     *    Unlike STL, returns both the iterator following the last erased element and a status code.
     *    On error, some elements may have been erased (partial erase, matches STL's basic guarantee).
     *
     *  @param[in] first Beginning of range to erase.
     *  @param[in] last End of range to erase (not erased).
     *  @return erase_result_t Contains iterator equal to @p last and status of the operation.
     *    Returns first error encountered, or success if all elements erased.
     */
    erase_result_t erase(const_iterator first, const_iterator last) noexcept {
        return erase(iterator(this, first.node_), iterator(this, last.node_));
    }

    /**
     *  @brief Hints to the tree to pre-allocate memory. No-op for AVL tree implementation.
     *    Provided for API consistency with other containers. Doesn't guarantee subsequent
     *    insertions won't fail with "out of memory".
     *
     *  @param[in] size Suggested capacity (ignored for AVL trees).
     *
     *  @note This is a no-op because AVL trees don't support reserving capacity efficiently.
     */
    void reserve(std::size_t) noexcept {}

    /**
     *  @brief Removes all elements from the tree and frees their memory.
     *  @return status_t Always succeeds.
     */
    void clear() noexcept {
        node_t::for_each_bottom_up(root_, [&](node_t *node) noexcept { return allocator_.deallocate(node, 1); });
        root_ = nullptr;
        size_ = 0;
    }

    template <typename callback_type_>
    void for_each(callback_type_ &&callback) noexcept {
        node_t::for_each_bottom_up(root_, [&](node_t *node) noexcept { callback(node->entry); });
    }

    /**
     *  @brief Merges another tree into this one, transferring all nodes.
     *    Elements with keys that already exist in this tree are deallocated (not kept in source).
     *
     *  @param[inout] other Tree to merge from. Will be empty after merge.
     *
     *  @note Unlike @c std::set::merge(), nodes with duplicate keys are deallocated rather than
     *    remaining in the source container. This ensures no memory leaks in a noexcept context.
     */
    void merge(avl_tree_t &other) noexcept {
        node_t::for_each_bottom_up(other.root_, [&](node_t *node) noexcept {
            auto result = node_t::insert(root_, node);
            root_ = result.root;
            size_ += result.inserted;
            if (!result.inserted) {
                // Key conflict - node wasn't inserted, deallocate it
                allocator_.deallocate(node, 1);
            }
        });
        other.root_ = nullptr;
        other.size_ = 0;
    }

    /**
     *  @brief Merges a single extracted node into this tree.
     *    If the key already exists, the node is deallocated.
     *
     *  @param[in] other Extracted node to merge.
     *
     *  @note Unlike @c std::set::merge(), a node with a duplicate key is deallocated rather than
     *    being returned. This ensures no memory leaks in a noexcept context.
     */
    void merge(extract_result_t other) noexcept {
        if (!other.node_ptr_) return;
        node_t *node_to_insert = other.release();
        auto result = node_t::insert(root_, node_to_insert);
        root_ = result.root;
        size_ += result.inserted;
        if (!result.inserted) {
            // Key conflict - node wasn't inserted, deallocate it
            allocator_.deallocate(node_to_insert, 1);
        }
    }

    /**
     *  @brief Uniformly samples a single random entry from the entire tree.
     *    Uses a probabilistic algorithm based on tree height (upper bound of subtree size).
     *
     *  @param[in] generator Random number generator (e.g., @c std::mt19937).
     *  @param[in] callback Callback to receive the sampled element. Must be @c noexcept.
     *
     *  @note Distribution is approximate due to relying on height-based size estimates.
     */
    template <typename generator_type_, typename callback_type_ = no_op_t>
    void sample(generator_type_ &&generator, callback_type_ &&callback) const noexcept {
        auto node = node_t::sample(root_, std::forward<generator_type_>(generator));
        if (node) { callback(node->entry); }
    }

    /**
     *  @brief Uniformly samples a single random entry from the range [@p lower, @p upper).
     *    Uses a two-pass algorithm: first counts entries, then selects random offset.
     *
     *  @param[in] lower Lower bound of the range (inclusive).
     *  @param[in] upper Upper bound of the range (exclusive).
     *  @param[in] generator Random number generator (e.g., @c std::mt19937).
     *  @param[in] callback Callback to receive the sampled element. Must be @c noexcept.
     *
     *  @note Requires two tree traversals. For multiple samples, use reservoir sampling overload.
     */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    void sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                      callback_type_ &&callback) const noexcept {
        auto node =
            node_t::sample_range(root_, std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                                 std::forward<generator_type_>(generator), [](node_t *) noexcept { return true; });
        if (node) { callback(node->entry); }
    }

    /**
     *  @brief Uniformly samples entries from [ @p lower, @p upper) using reservoir sampling algorithm.
     *    Fills a reservoir buffer with up to @p reservoir_capacity randomly selected elements.
     *
     *  @param[in] lower Lower bound of the range (inclusive).
     *  @param[in] upper Upper bound of the range (exclusive).
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

        auto sampler = [&](entry_t const &entry) noexcept {
            if (seen < reservoir_capacity) reservoir[seen] = entry;
            else {
                std::uniform_int_distribution<std::size_t> distribution {0, seen};
                auto slot_to_replace = distribution(generator);
                if (slot_to_replace < reservoir_capacity) reservoir[slot_to_replace] = entry;
            }
            ++seen;
        };
        range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), sampler);
    }
};

} // namespace ashvardanian::smashtable
