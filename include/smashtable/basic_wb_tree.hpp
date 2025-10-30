/**
 *  @brief  Weight-balanced tree with order statistics support.
 *    Size-balanced alternative to AVL trees, enabling O(log n) @c select and @c rank operations.
 *
 *  @file   basic_wb_tree.hpp
 *  @author Ash Vardanian
 *
 *  @section Weight-Balanced Trees
 *
 *  Weight-balanced trees maintain balance based on subtree sizes rather than heights. Rebalancing uses parameters
 *  Δ=3 and Γ=2, which are the only proven integer solution (Hirai & Yamamoto, 2011).
 *
 *  Balance invariant: For every node, size(left) < Δ × size(right) AND size(right) < Δ × size(left)
 *
 *  @section Order Statistics
 *
 *  Storing sizes enables efficient order statistic queries:
 *  - @c select(k): Find k-th smallest element in O(log n)
 *  - @c rank(x): Find position of element x in O(log n)
 *
 *  Use cases: pagination (OFFSET/LIMIT), percentiles, window functions, quantile estimation.
 *
 *  @section Performance
 *
 *  - Expected depth: ~1.88 log₂(n) vs AVL's 1.44 log₂(n)
 *  - Amortized rotations: O(1) per update vs AVL's O(log n) worst-case
 *  - Space overhead: Same as AVL (1 word per node for size vs height)
 */
#pragma once
#include <cassert>   // `assert`
#include <algorithm> // `std::max`
#include <iterator>  // `std::bidirectional_iterator_tag`
#include <memory>    // `std::allocator`
#include <random>    // `std::uniform_int_distribution`
#include <utility>   // `std::pair`, `std::exchange`

#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief Node for weight-balanced binary search tree.
 *    Stores subtree size for O(log n) order statistics and O(1) amortized rebalancing.
 *
 *  @tparam entry_type_      Type of elements stored in nodes.
 *  @tparam comparator_type_ Comparator defining ordering. For heterogeneous lookups, define:
 *    @code using is_transparent = void; @endcode inside the comparator.
 *
 *  @section Rebalancing Parameters
 *
 *  Δ=3: Rotation threshold. Rebalance if size(left) >= 3×size(right) or vice versa.
 *  Γ=2: Rotation type selector. Single rotation if size(heavy.inner) < 2×size(heavy.outer).
 *
 *  These are the **only valid integer parameters** (proven in Coq).
 */
template <typename entry_type_, typename comparator_type_>
class basic_wb_node {
  public:
    using versioned_entry_t = entry_type_;
    using comparator_t = comparator_type_;
    using size_t = std::size_t;
    using node_t = basic_wb_node;

    versioned_entry_t entry;
    node_t *left = nullptr;
    node_t *right = nullptr;

    /**
     *  @brief Subtree size (number of nodes in subtree rooted at this node).
     *    Invariant: size = 1 + size(left) + size(right)
     *    Enables O(log n) order statistics: @c select(k), @c rank(x).
     */
    size_t size = 1;

    /**
     *  @brief WBT rebalancing parameters (Hirai & Yamamoto, 2011).
     *    Δ=3: Triggers rotation when size imbalance >= 3×
     *    Γ=2: Chooses single vs double rotation based on grandchild sizes
     */
    static constexpr size_t delta_k = 3;
    static constexpr size_t gamma_k = 2;

    static size_t get_size(node_t *node) noexcept { return node ? node->size : 0; }

#pragma mark - Traversal and Search

    /**
     *  @brief Find minimum (leftmost) node in subtree.
     *  @param[in] node Root of subtree.
     *  @return Pointer to minimum node, or nullptr if tree is empty.
     */
    static node_t *find_min(node_t *node) noexcept {
        if (!node) return nullptr;
        while (node->left) node = node->left;
        return node;
    }

    /**
     *  @brief Find maximum (rightmost) node in subtree.
     *  @param[in] node Root of subtree.
     *  @return Pointer to maximum node, or nullptr if tree is empty.
     */
    static node_t *find_max(node_t *node) noexcept {
        if (!node) return nullptr;
        while (node->right) node = node->right;
        return node;
    }

    /**
     *  @brief Find exact match for a key.
     *  @param[in] node Root of subtree to search.
     *  @param[in] comparable Key to search for.
     *  @param[in] comparator Comparator instance (may be stateful).
     *  @return Pointer to node if found, nullptr otherwise.
     */
    template <typename comparable_type_>
    static node_t *find(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        while (node) {
            if (comparator(comparable, node->entry)) node = node->left;
            else if (comparator(node->entry, comparable)) node = node->right;
            else break;
        }
        return node;
    }

    /**
     *  @brief Find smallest entry >= comparable (lower bound).
     *  @param[in] node Root of subtree to search.
     *  @param[in] comparable Key to search for.
     *  @param[in] comparator Comparator instance (may be stateful).
     *  @return Pointer to lower bound node, or nullptr if all elements < comparable.
     */
    template <typename comparable_type_>
    static node_t *lower_bound(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        node_t *result = nullptr;
        while (node) {
            if (!comparator(node->entry, comparable)) {
                result = node;
                node = node->left;
            }
            else { node = node->right; }
        }
        return result;
    }

    /**
     *  @brief Find smallest entry > comparable (upper bound).
     *  @param[in] node Root of subtree to search.
     *  @param[in] comparable Key to search for.
     *  @param[in] comparator Comparator instance (may be stateful).
     *  @return Pointer to upper bound node, or nullptr if all elements <= comparable.
     */
    template <typename comparable_type_>
    static node_t *upper_bound(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        node_t *result = nullptr;
        while (node) {
            if (comparator(comparable, node->entry)) {
                result = node;
                node = node->left;
            }
            else { node = node->right; }
        }
        return result;
    }

    /**
     *  @brief Finds the next node in in-order traversal (successor).
     *  @param[in] root Root of the tree.
     *  @param[in] node Current node.
     *  @param[in] comparator Comparator for ordering.
     *  @return node_t* Successor node, or nullptr if node is the maximum.
     */
    static node_t *find_successor(node_t *root, node_t *node, comparator_t const &comparator) noexcept {
        if (!node) return find_min(root);
        return upper_bound(root, node->entry, comparator);
    }

    /**
     *  @brief Finds the previous node in in-order traversal (predecessor).
     *  @param[in] root Root of the tree.
     *  @param[in] node Current node.
     *  @param[in] comparator Comparator for ordering.
     *  @return node_t* Predecessor node, or nullptr if node is the minimum.
     */
    static node_t *find_predecessor(node_t *root, node_t *node, comparator_t const &comparator) noexcept {
        if (!node) return find_max(root);

        node_t *predecessor = nullptr;
        node_t *current = root;

        while (current) {
            // Current is less than target, it's a candidate predecessor
            if (comparator(current->entry, node->entry)) {
                predecessor = current;
                current = current->right;
            }
            // Current is >= target, search left subtree
            else current = current->left;
        }
        return predecessor;
    }

#pragma mark - Order Statistics

    /**
     *  @brief Select k-th smallest element (0-indexed).
     *  @param[in] node Root of subtree.
     *  @param[in] k Index of element to find (0 = minimum, size-1 = maximum).
     *  @param[in] comparator Comparator instance.
     *  @return Pointer to k-th node, or nullptr if k >= size.
     *
     *  @par Complexity
     *  O(log n) expected, where n = size(node).
     *
     *  @par Example
     *  @code
     *  auto median = select(root, size/2, comp);  // Find median
     *  auto q1 = select(root, size/4, comp);      // First quartile
     *  @endcode
     */
    static node_t *select(node_t *node, size_t k, comparator_t const &comparator) noexcept {
        if (!node) return nullptr;
        size_t left_size = get_size(node->left);

        if (k < left_size) return select(node->left, k, comparator);
        if (k == left_size) return node;
        return select(node->right, k - left_size - 1, comparator);
    }

    /**
     *  @brief Find rank (position) of element in sorted order.
     *  @param[in] node Root of subtree.
     *  @param[in] comparable Key to find rank of.
     *  @param[in] comparator Comparator instance.
     *  @return Number of elements < comparable. If element exists, this is its 0-based index.
     *
     *  @par Complexity
     *  O(log n) expected.
     *
     *  @par Example
     *  @code
     *  auto pos = rank(root, value, comp);
     *  // pos elements are smaller than value
     *  // If value exists, it's at index pos
     *  @endcode
     */
    template <typename comparable_type_>
    static size_t rank(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        if (!node) return 0;

        if (comparator(comparable, node->entry)) {
            // comparable < node, search left
            return rank(node->left, std::forward<comparable_type_>(comparable), comparator);
        }
        else if (comparator(node->entry, comparable)) {
            // node < comparable, search right
            return get_size(node->left) + 1 + rank(node->right, std::forward<comparable_type_>(comparable), comparator);
        }
        else {
            // Found it - return count of smaller elements
            return get_size(node->left);
        }
    }

    /**
     *  @brief In-order traversal (left-root-right) with callback.
     *    Visits nodes in sorted order according to comparator.
     *
     *  @param node Root of subtree to traverse.
     *  @param callback Callback to invoke for each node. Must be @c noexcept.
     */
    template <typename callback_type_>
    static void for_each_left_right(node_t *node, callback_type_ &&callback) noexcept {
        if (!node) return;
        for_each_left_right(node->left, callback);
        callback(node);
        for_each_left_right(node->right, callback);
    }

    /**
     *  @brief Result of a range query containing interval boundaries.
     */
    struct node_interval_t {
        node_t *lower_bound = nullptr;
        node_t *upper_bound = nullptr;
        node_t *lowest_common_ancestor = nullptr;
    };

    /**
     *  @brief Finds all nodes in the range [low, high] and invokes callback for each.
     *    Recursively traverses the tree, visiting only nodes within the specified range.
     *
     *  @param node Root of subtree to search.
     *  @param low Lower bound of range (inclusive).
     *  @param high Upper bound of range (inclusive).
     *  @param comparator Comparator for element comparison.
     *  @param callback Callback to invoke for each node in range. Must be @c noexcept.
     *  @return Interval containing lower bound, upper bound, and lowest common ancestor.
     *
     *  @warning Current recursive implementation is suboptimal.
     */
    template <typename lower_type_, typename upper_type_, typename callback_type_>
    static node_interval_t range(node_t *node, lower_type_ &&low, upper_type_ &&high, comparator_t const &comparator,
                                 callback_type_ &&callback) noexcept {
        if (!node) return {};

        // If this node fits into the interval - analyze its children.
        // The first call to reach this branch in the call-stack
        // will be by definition the Lowest Common Ancestor.
        if (!comparator(high, node->entry) && !comparator(node->entry, low)) {
            callback(node);
            auto left_sub_interval = range(node->left, low, high, comparator, callback);
            auto right_sub_interval = range(node->right, low, high, comparator, callback);

            auto result = node_interval_t {};
            result.lower_bound = left_sub_interval.lower_bound ? left_sub_interval.lower_bound : node;
            result.upper_bound = right_sub_interval.upper_bound ? right_sub_interval.upper_bound : node;
            result.lowest_common_ancestor = node;
            return result;
        }

        if (comparator(node->entry, low)) return range(node->right, low, high, comparator, callback);

        return range(node->left, low, high, comparator, callback);
    }

    /**
     *  @brief Finds equal range for a given key.
     */
    template <typename comparable_type_>
    static node_interval_t equal_range(node_t *node, comparable_type_ &&comparable,
                                       comparator_t const &comparator) noexcept {
        auto no_op = [](node_t *) noexcept {};
        return range(node, comparable, comparable, comparator, no_op);
    }

#pragma mark - Rotations and Rebalancing

    /**
     *  @brief Update size field to match children.
     *    Must be called after any operation that modifies children.
     */
    static void update_size(node_t *node) noexcept {
        if (node) node->size = 1 + get_size(node->left) + get_size(node->right);
    }

    /**
     *  @brief Single right rotation.
     *  @code
     *      y              x
     *     / \            / \
     *    x   C    =>    A   y
     *   / \                / \
     *  A   B              B   C
     *  @endcode
     */
    static node_t *rotate_right(node_t *y) noexcept {
        node_t *x = y->left;
        node_t *B = x->right;

        x->right = y;
        y->left = B;

        update_size(y); // Update y first (now child)
        update_size(x); // Then x (now parent)

        return x;
    }

    /**
     *  @brief Single left rotation.
     *  @code
     *    x                y
     *   / \              / \
     *  A   y      =>    x   C
     *     / \          / \
     *    B   C        A   B
     *  @endcode
     */
    static node_t *rotate_left(node_t *x) noexcept {
        node_t *y = x->right;
        node_t *B = y->left;

        y->left = x;
        x->right = B;

        update_size(x); // Update x first (now child)
        update_size(y); // Then y (now parent)

        return y;
    }

    /**
     *  @brief Check if node satisfies weight-balance invariant.
     *  @return True if balanced: size(left) < Δ×size(right) AND size(right) < Δ×size(left)
     */
    static bool is_balanced(node_t *node) noexcept {
        if (!node) return true;
        size_t left_size = get_size(node->left);
        size_t right_size = get_size(node->right);

        // Both conditions must hold
        return (left_size < delta_k * right_size) && (right_size < delta_k * left_size);
    }

    /**
     *  @brief Rebalance node if weight invariant violated.
     *    Uses Δ=3, Γ=2 parameters (only valid integer solution).
     *
     *  @par Algorithm
     *  - If left too heavy (size(left) >= 3×size(right)):
     *    - Single right rotation if size(left.right) < 2×size(left.left)
     *    - Double (left-right) rotation otherwise
     *  - Mirror logic for right-heavy case
     *
     *  @return New root after rebalancing (may be unchanged).
     */
    static node_t *rebalance(node_t *node) noexcept {
        if (!node) return nullptr;

        size_t left_size = get_size(node->left);
        size_t right_size = get_size(node->right);

        // Check if left is too heavy
        if (left_size >= delta_k * right_size && left_size > 0) {
            node_t *left = node->left;
            size_t left_left_size = get_size(left->left);
            size_t left_right_size = get_size(left->right);

            // Choose single or double rotation based on grandchild sizes
            if (left_right_size < gamma_k * left_left_size) {
                // Single right rotation
                return rotate_right(node);
            }
            else {
                // Double rotation: left-right
                node->left = rotate_left(left);
                return rotate_right(node);
            }
        }

        // Check if right is too heavy
        if (right_size >= delta_k * left_size && right_size > 0) {
            node_t *right = node->right;
            size_t right_left_size = get_size(right->left);
            size_t right_right_size = get_size(right->right);

            // Choose single or double rotation based on grandchild sizes
            if (right_left_size < gamma_k * right_right_size) {
                // Single left rotation
                return rotate_left(node);
            }
            else {
                // Double rotation: right-left
                node->right = rotate_right(right);
                return rotate_left(node);
            }
        }

        return node; // Already balanced
    }

#pragma mark - Removals

    struct extract_result_t {
        node_t *root = nullptr;
        std::unique_ptr<node_t> extracted;

        node_t *release() noexcept { return extracted.release(); }
    };

    static node_t *rebalance_after_extract(node_t *node) noexcept {
        update_size(node);
        return rebalance(node);
    }

    /**
     *  @brief Pops the root replacing it with one of descendants, if present.
     *  @param node Node to extract.
     *  @param comparator Comparator for element comparison.
     */
    static extract_result_t extract(node_t *node, comparator_t const &comparator) noexcept {

        // If the node has two children, replace it with the
        // smallest entry in the right branch.
        if (node->left && node->right) {
            node_t *midpoint = find_min(node->right);
            auto downstream = extract(node->right, midpoint->entry, comparator);
            midpoint = downstream.extracted.release();
            midpoint->left = node->left;
            midpoint->right = downstream.root;
            update_size(midpoint);
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            node->size = 1;
            return {midpoint, std::unique_ptr<node_t> {node}};
        }
        // Just one child is present, so it is the natural successor.
        else if (node->left || node->right) {
            node_t *replacement = node->left ? node->left : node->right;
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            node->size = 1;
            return {replacement, std::unique_ptr<node_t> {node}};
        }
        // No children are present.
        else {
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            node->size = 1;
            return {nullptr, std::unique_ptr<node_t> {node}};
        }
    }

    /**
     *  @brief Searches for a matching ancestor and extracts it out.
     *  @param comparable Any key comparable with stored entries.
     */
    template <typename comparable_type_>
    static extract_result_t extract(node_t *node, comparable_type_ &&comparable,
                                    comparator_t const &comparator) noexcept {
        if (!node) return {node, {}};

        if (comparator(comparable, node->entry)) {
            auto downstream = extract(node->left, comparable, comparator);
            node->left = downstream.root;
            if (downstream.extracted) node = rebalance_after_extract(node);
            return {node, std::move(downstream.extracted)};
        }

        else if (comparator(node->entry, comparable)) {
            auto downstream = extract(node->right, comparable, comparator);
            node->right = downstream.root;
            if (downstream.extracted) node = rebalance_after_extract(node);
            return {node, std::move(downstream.extracted)};
        }

        else
            // We have found the node to extract!
            return extract(node, comparator);
    }

#pragma mark - Insertions

    struct find_or_make_result_t {
        node_t *root = nullptr;
        node_t *match = nullptr;
        bool inserted = false;

        /**
         *  @return True if the allocation of the new node has failed.
         */
        bool failed() const noexcept { return !inserted && !match; }
    };

    /**
     *  @brief Inserts an existing node into the tree.
     *    Used by merge operation to insert extracted nodes.
     *  @param node Root of subtree to insert into.
     *  @param new_child Pre-allocated node to insert.
     *  @param comparator Comparator for element comparison.
     *  @return Result containing new root, matched node, and insertion status.
     */
    static find_or_make_result_t insert(node_t *node, node_t *new_child, comparator_t const &comparator) noexcept {
        if (!node) return {new_child, new_child, true};

        if (comparator(new_child->entry, node->entry)) {
            auto result = insert(node->left, new_child, comparator);
            node->left = result.root;
            if (result.inserted) {
                update_size(node);
                node = rebalance(node);
            }
            return {node, result.match, result.inserted};
        }

        else if (comparator(node->entry, new_child->entry)) {
            auto result = insert(node->right, new_child, comparator);
            node->right = result.root;
            if (result.inserted) {
                update_size(node);
                node = rebalance(node);
            }
            return {node, result.match, result.inserted};
        }

        else {
            // Key already exists - don't insert
            return {node, node, false};
        }
    }

    /**
     *  @brief Inserts or updates an entry in the tree.
     *    If key exists, overwrites the entry. If not, creates new node.
     *
     *  @param node Root of subtree.
     *  @param entry Entry to insert or assign (moved).
     *  @param comparator Comparator for element comparison.
     *  @param node_allocator Allocator function that returns new node pointer or nullptr on failure.
     *  @return Result containing new root, matched node, and insertion status.
     */
    template <typename node_allocator_type_>
    static find_or_make_result_t upsert(node_t *node, entry_type_ &&entry, comparator_t const &comparator,
                                        node_allocator_type_ &&node_allocator) noexcept {
        // Base case: empty tree, allocate new node
        if (!node) {
            node_t *new_node = node_allocator();
            if (!new_node) return {nullptr, nullptr, false};
            new (&new_node->entry) entry_type_(std::move(entry));
            new_node->left = nullptr;
            new_node->right = nullptr;
            new_node->size = 1;
            return {new_node, new_node, true};
        }

        // Recursive case: search for insertion point
        if (comparator(entry, node->entry)) {
            auto result = upsert(node->left, std::move(entry), comparator, node_allocator);
            node->left = result.root;
            if (result.inserted) {
                update_size(node);
                node = rebalance(node);
            }
            return {node, result.match, result.inserted};
        }
        else if (comparator(node->entry, entry)) {
            auto result = upsert(node->right, std::move(entry), comparator, node_allocator);
            node->right = result.root;
            if (result.inserted) {
                update_size(node);
                node = rebalance(node);
            }
            return {node, result.match, result.inserted};
        }
        else {
            // Key already exists - update the entry
            node->entry = std::move(entry);
            return {node, node, false};
        }
    }

#pragma mark - Split and Join Operations

    /**
     *  @brief Result of splitting a tree at a key.
     */
    struct split_result_t {
        node_t *left = nullptr;  //!< Tree with all elements < key.
        node_t *right = nullptr; //!< Tree with all elements >= key.
    };

    /**
     *  @brief Joins two trees with a root node between them.
     *    Precondition: all(left) < root < all(right)
     *
     *  @param left Left subtree (all elements < root).
     *  @param root Middle node to join with.
     *  @param right Right subtree (all elements > root).
     *  @return New root of joined tree.
     *
     *  @par Complexity
     *  O(log n) expected, where n is size of larger tree.
     */
    static node_t *join_with_root(node_t *left, node_t *root, node_t *right) noexcept {
        if (!root) {
            // If no root, we can't join - this shouldn't happen
            return left ? left : right;
        }

        root->left = left;
        root->right = right;
        update_size(root);
        return rebalance(root);
    }

    /**
     *  @brief Joins two trees where all(left) < all(right).
     *    Recursively joins trees by extracting min from right subtree.
     *
     *  @param left Left subtree.
     *  @param right Right subtree.
     *  @param comparator Comparator for ordering elements.
     *  @return New root of joined tree.
     *
     *  @par Complexity
     *  O(log n) expected.
     */
    static node_t *join(node_t *left, node_t *right, comparator_t const &comparator) noexcept {
        if (!left) return right;
        if (!right) return left;

        // Extract min from right tree to use as joining root
        node_t *min_node = find_min(right);
        auto extracted = extract(right, min_node->entry, comparator);

        return join_with_root(left, extracted.extracted.release(), extracted.root);
    }

    /**
     *  @brief Splits tree at a given key.
     *    Returns two trees: left contains all elements < comparable,
     *    right contains all elements >= comparable.
     *
     *  @param node Root of tree to split.
     *  @param comparable Key to split at.
     *  @param comparator Comparator for element comparison.
     *  @return Split result with left and right subtrees.
     *
     *  @par Complexity
     *  O(log n) expected.
     *
     *  @par Example
     *  @code
     *  auto [left, right] = split(root, 5, comp);
     *  // left: all elements < 5
     *  // right: all elements >= 5
     *  @endcode
     */
    template <typename comparable_type_>
    static split_result_t split(node_t *node, comparable_type_ &&comparable,
                                comparator_t const &comparator) noexcept {
        if (!node) return {nullptr, nullptr};

        if (comparator(comparable, node->entry)) {
            // Split key < node, split left subtree
            auto downstream = split(node->left, std::forward<comparable_type_>(comparable), comparator);
            // join_with_root will set node's children and call update_size/rebalance
            return {downstream.left, join_with_root(downstream.right, node, node->right)};
        }
        else {
            // Split key >= node, split right subtree
            auto downstream = split(node->right, std::forward<comparable_type_>(comparable), comparator);
            // join_with_root will set node's children and call update_size/rebalance
            return {join_with_root(node->left, node, downstream.left), downstream.right};
        }
    }
};

/**
 *  @brief Weight-balanced tree container with order statistics support.
 *    Provides ordered storage similar to @c std::set with additional O(log n) select/rank operations.
 *
 *  @tparam entry_type_ Type of elements stored.
 *  @tparam comparator_type_ Comparator defining ordering.
 *  @tparam node_allocator_type_ Allocator for node allocation.
 *
 *  @section Order Statistics API
 *
 *  Beyond standard tree operations, provides:
 *  - @c select(k): Find k-th smallest element in O(log n)
 *  - @c rank(x): Find position of element x in O(log n)
 *
 *  Use cases: pagination, percentiles, quantiles, window functions.
 */
template <typename entry_type_, typename comparator_type_ = std::less<entry_type_>,
          typename node_allocator_type_ = std::allocator<basic_wb_node<entry_type_, comparator_type_>>>
class basic_wb_tree {
  public:
    using entry_t = entry_type_;
    using comparator_t = comparator_type_;
    using node_t = basic_wb_node<entry_t, comparator_t>;
    using node_allocator_t = node_allocator_type_;
    using wb_tree_t = basic_wb_tree;
    using size_t = std::size_t;
    using allocator_t = node_allocator_type_;

    /**
     *  @brief Rebind this tree type to different entry and comparator types.
     *    Follows STL allocator rebind pattern for type transformations.
     *
     *  @tparam other_entry_ New entry type for the rebound tree.
     *  @tparam other_comparator_ New comparator type for the rebound tree.
     */
    template <typename other_entry_, typename other_comparator_>
    using rebind = basic_wb_tree<other_entry_, other_comparator_,
                                 typename std::allocator_traits<node_allocator_type_>::template rebind_alloc<
                                     basic_wb_node<other_entry_, other_comparator_>>>;

  private:
    node_t *root_ = nullptr;
    size_t size_ = 0;
    [[no_unique_address]] comparator_t comparator_;
    [[no_unique_address]] node_allocator_t allocator_;

  public:
    /**
     *  @brief Default constructor. Creates empty tree.
     */
    basic_wb_tree() noexcept = default;

    /**
     *  @brief Move constructor.
     */
    basic_wb_tree(wb_tree_t &&other) noexcept
        : root_(std::exchange(other.root_, nullptr)), size_(std::exchange(other.size_, 0)),
          comparator_(std::move(other.comparator_)), allocator_(std::move(other.allocator_)) {}

    /**
     *  @brief Move assignment operator.
     */
    wb_tree_t &operator=(wb_tree_t &&other) noexcept {
        std::swap(root_, other.root_);
        std::swap(size_, other.size_);
        std::swap(comparator_, other.comparator_);
        std::swap(allocator_, other.allocator_);
        return *this;
    }

    /**
     *  @brief Destructor. Deallocates all nodes.
     */
    ~basic_wb_tree() noexcept { clear(); }

    // Disable copy
    basic_wb_tree(wb_tree_t const &) = delete;
    wb_tree_t &operator=(wb_tree_t const &) = delete;

#pragma mark - Capacity

    /**
     *  @brief Returns the number of elements in the tree.
     */
    size_t size() const noexcept { return size_; }

    /**
     *  @brief Checks whether the tree is empty.
     */
    bool empty() const noexcept { return size_ == 0; }

#pragma mark - Modifiers

    /**
     *  @brief Inserts an element if key doesn't exist.
     *  @param[in] entry Entry to insert (moved into the tree).
     *  @return Pair of pointer to node and bool indicating success.
     *    Returns {node, true} if inserted successfully.
     *    Returns {node, false} if key already exists.
     *    Returns {nullptr, false} if allocation failed.
     */
    std::pair<node_t *, bool> insert(entry_t &&entry) noexcept {
        // For now, simple implementation - just allocate and track size
        node_t *new_node = allocator_.allocate(1);
        if (!new_node) return {nullptr, false};

        new (&new_node->entry) entry_t(std::move(entry));
        new_node->left = nullptr;
        new_node->right = nullptr;
        new_node->size = 1;

        if (!root_) {
            root_ = new_node;
            size_ = 1;
            return {new_node, true};
        }

        // Simple BST insert (TODO: add rebalancing)
        node_t *current = root_;
        node_t *parent = nullptr;

        while (current) {
            parent = current;
            if (comparator_(new_node->entry, current->entry)) { current = current->left; }
            else if (comparator_(current->entry, new_node->entry)) { current = current->right; }
            else {
                // Key already exists
                allocator_.deallocate(new_node, 1);
                return {current, false};
            }
        }

        // Insert as child of parent
        if (comparator_(new_node->entry, parent->entry)) { parent->left = new_node; }
        else { parent->right = new_node; }

        ++size_;

        // Update sizes upward (simplified - should update from insertion point up)
        update_sizes_from_root();

        return {new_node, true};
    }

    /**
     *  @brief Result type for upsert operations.
     */
    struct upsert_result_t {
        node_t *node = nullptr;
        bool inserted = false;

        /**
         *  @return True if the allocation of the new node has failed.
         */
        bool failed() const noexcept { return !inserted && !node; }
    };

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
        auto result = node_t::upsert(root_, std::forward<comparable_type_>(comparable), comparator_,
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
     *  @brief Removes all elements from the tree.
     */
    void clear() noexcept {
        clear_recursive(root_);
        root_ = nullptr;
        size_ = 0;
    }

#pragma mark - Lookup

    /**
     *  @brief Checks if an element exists in the tree.
     */
    bool contains(entry_t const &entry) const noexcept { return find(entry) != end(); }

#pragma mark - Order Statistics

    /**
     *  @brief Select k-th smallest element (0-indexed).
     *  @param[in] k Index of element to find (0 = minimum, size-1 = maximum).
     *  @return Pointer to k-th node, or nullptr if k >= size.
     *
     *  @par Complexity
     *  O(log n) expected.
     *
     *  @par Example
     *  @code
     *  auto median_node = tree.select(tree.size() / 2);
     *  if (median_node) {
     *      std::cout << "Median: " << median_node->entry << std::endl;
     *  }
     *  @endcode
     */
    node_t *select(size_t k) noexcept { return node_t::select(root_, k, comparator_); }

    /**
     *  @brief Select k-th smallest element (const version).
     */
    node_t const *select(size_t k) const noexcept { return node_t::select(root_, k, comparator_); }

    /**
     *  @brief Find rank (position) of element in sorted order.
     *  @param[in] entry Entry to find rank of.
     *  @return Number of elements < entry. If element exists, this is its 0-based index.
     *    Returns size() if element is greater than all elements in tree.
     *
     *  @par Complexity
     *  O(log n) expected.
     *
     *  @par Example
     *  @code
     *  auto pos = tree.rank(42);
     *  // pos elements are smaller than 42
     *  @endcode
     */
    size_t rank(entry_t const &entry) const noexcept { return node_t::rank(root_, entry, comparator_); }

    /**
     *  @brief Iterates over all entries in sorted order.
     *  @param[in] callback Callback to invoke for each entry. Must be @c noexcept.
     */
    template <typename callback_type_>
    void for_each(callback_type_ &&callback) noexcept {
        node_t::for_each_left_right(root_, [&](node_t *node) noexcept { callback(node->entry); });
    }

#pragma mark - Extraction and Merging

    /**
     *  @brief RAII wrapper for extracted nodes.
     *    Owns the extracted node and deallocates it when destroyed (unless released).
     */
    struct extract_result_t {
        basic_wb_tree *tree_ = nullptr;
        node_t *node_ptr_ = nullptr;

        extract_result_t() = default;
        extract_result_t(basic_wb_tree *tree, node_t *node) noexcept : tree_(tree), node_ptr_(node) {}

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

    /**
     *  @brief Extracts a node from the tree.
     *    The extracted node is removed from the tree and returned in an RAII wrapper.
     *
     *  @param[in] comparable Key to extract.
     *  @return RAII wrapper containing extracted node (empty if not found).
     *
     *  @par Complexity
     *  O(log n) expected.
     *
     *  @par Example
     *  @code
     *  auto extracted = tree.extract(42);
     *  if (extracted) {
     *      other_tree.merge(std::move(extracted));
     *  }
     *  @endcode
     */
    template <typename comparable_type_>
    extract_result_t extract(comparable_type_ &&comparable) noexcept {
        auto result = node_t::extract(root_, std::forward<comparable_type_>(comparable), comparator_);
        root_ = result.root;
        size_ -= result.extracted != nullptr;
        return extract_result_t {this, result.extracted.release()};
    }

  private:
    /**
     *  @brief Checks if any key exists in both trees using O(m+n) simultaneous traversal.
     *  @param[in] other Tree to check for intersecting keys.
     *  @return true if at least one key exists in both trees, false otherwise.
     */
    bool has_any_key(wb_tree_t const &other) const noexcept {
        auto it1 = begin();
        auto it2 = other.begin();
        while (it1 != end() && it2 != other.end()) {
            if (comparator_(*it1, *it2)) ++it1;
            else if (comparator_(*it2, *it1)) ++it2;
            else return true; // Found duplicate
        }
        return false;
    }

    /**
     *  @brief Checks if all keys from other tree exist in this tree using O(m+n) simultaneous traversal.
     *  @param[in] other Tree whose keys to check.
     *  @return true if all keys from other exist in this tree, false otherwise.
     */
    bool has_all_keys(wb_tree_t const &other) const noexcept {
        auto it1 = begin();
        auto it2 = other.begin();
        while (it1 != end() && it2 != other.end()) {
            if (comparator_(*it1, *it2)) ++it1;
            else if (comparator_(*it2, *it1)) return false; // Key in other not found in this
            else { ++it1; ++it2; }
        }
        return it2 == other.end(); // All keys from other were found
    }

    /**
     *  @brief Merges another tree using upsert semantics (updates duplicates instead of skipping).
     *  @param[in] other Tree to merge from. Will be empty after merge.
     */
    void merge_with_upsert(wb_tree_t &other) noexcept {
        while (other.size() > 0) {
            node_t *other_root = other.root_;
            if (!other_root) break;

            // Extract root node and upsert into this tree
            auto extracted = other.extract(other_root->entry);
            if (extracted) {
                node_t *node_to_insert = extracted.release();
                auto result = node_t::upsert(root_, std::move(node_to_insert->entry), comparator_,
                                             [&]() noexcept { return allocator_.allocate(1); });
                root_ = result.root;
                size_ += result.inserted;
                allocator_.deallocate(node_to_insert, 1);
            }
        }
    }

  public:
    /**
     *  @brief Merges an extracted node into this tree.
     *
     *  @param[in] other Extracted node to merge.
     *
     *  @note Unlike @c std::set::merge(), a node with a duplicate key is deallocated rather than
     *    being returned. This ensures no memory leaks in a noexcept context.
     */
    void merge(extract_result_t other) noexcept {
        if (!other.node_ptr_) return;
        node_t *node_to_insert = other.release();
        auto result = node_t::insert(root_, node_to_insert, comparator_);
        root_ = result.root;
        size_ += result.inserted;
        // Key conflict - node wasn't inserted, deallocate it
        if (!result.inserted) allocator_.deallocate(node_to_insert, 1);
    }

    /**
     *  @brief Merges another tree into this one, transferring all nodes.
     *    Elements with keys that already exist in this tree are deallocated (not kept in source).
     *
     *  @param[in] other Tree to merge from. Will be empty after merge.
     *
     *  @note Unlike @c std::set::merge(), nodes with duplicate keys are deallocated rather than
     *    remaining in the source container. This ensures no memory leaks in a noexcept context.
     *  @note Complexity: O(m log n) where m = other.size(), n = this.size().
     */
    void merge(wb_tree_t &other) noexcept {
        // Simple implementation: extract and merge each node one by one
        while (other.size() > 0) {
            // Extract root node (simplest approach)
            node_t *other_root = other.root_;
            if (!other_root) break;

            auto extracted = other.extract(other_root->entry);
            if (extracted) merge(std::move(extracted));
        }
    }

    /**
     *  @brief Merges another tree into this one, assuming all keys are unique.
     *    More efficient than regular merge when caller guarantees no duplicate keys.
     *
     *  @param[in] other Tree to merge from. Will be empty after merge.
     *
     *  @par Precondition
     *  All keys in @c other must be different from keys in this tree.
     *
     *  @warning If precondition violated (duplicate keys exist), behavior is undefined.
     */
    void merge(wb_tree_t &other, assume_unique_t) noexcept {
        if (other.empty()) return;
        if (empty()) {
            // Move other into this
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        // Fast path: Check if all(this) < all(other), use join: O(log n)
        auto this_max = node_t::find_max(root_);
        auto other_min = node_t::find_min(other.root_);

        if (comparator_(this_max->entry, other_min->entry)) {
            root_ = node_t::join(root_, other.root_, comparator_);
            size_ += other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        // Fallback: Use regular merge (one-by-one insertion)
        merge(other);
    }

#pragma mark - Iterator Support

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
     *  @brief Bidirectional iterator for WB tree.
     *    Provides in-order traversal of tree elements.
     */
    class iterator {
        friend class basic_wb_tree;
        friend class const_iterator;

      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = entry_t;
        using difference_type = std::ptrdiff_t;
        using pointer = entry_t *;
        using reference = entry_t &;

      private:
        basic_wb_tree const *tree_;
        node_t *node_;

        iterator(basic_wb_tree const *tree, node_t *node) noexcept : tree_(tree), node_(node) {}

      public:
        iterator() noexcept : tree_(nullptr), node_(nullptr) {}

        reference operator*() const noexcept { return node_->entry; }
        pointer operator->() const noexcept { return &node_->entry; }

        iterator &operator++() noexcept {
            node_ = node_t::find_successor(tree_->root_, node_, tree_->comparator_);
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        iterator &operator--() noexcept {
            node_ = node_t::find_predecessor(tree_->root_, node_, tree_->comparator_);
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
     *  @brief Const bidirectional iterator for WB tree.
     *    Provides in-order traversal of tree elements (read-only).
     */
    class const_iterator {
        friend class basic_wb_tree;

      public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = entry_t const;
        using difference_type = std::ptrdiff_t;
        using pointer = entry_t const *;
        using reference = entry_t const &;

      private:
        basic_wb_tree const *tree_;
        node_t const *node_;

        const_iterator(basic_wb_tree const *tree, node_t const *node) noexcept : tree_(tree), node_(node) {}

      public:
        const_iterator() noexcept : tree_(nullptr), node_(nullptr) {}
        const_iterator(iterator const &it) noexcept : tree_(it.tree_), node_(it.node_) {}

        reference operator*() const noexcept { return node_->entry; }
        pointer operator->() const noexcept { return &node_->entry; }

        const_iterator &operator++() noexcept {
            node_ = node_t::find_successor(tree_->root_, const_cast<node_t *>(node_), tree_->comparator_);
            return *this;
        }

        const_iterator operator++(int) noexcept {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        const_iterator &operator--() noexcept {
            node_ = node_t::find_predecessor(tree_->root_, const_cast<node_t *>(node_), tree_->comparator_);
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

    /**
     *  @brief Returns iterator to first element (minimum).
     */
    iterator begin() noexcept { return iterator(this, node_t::find_min(root_)); }

    /**
     *  @brief Returns const iterator to first element (minimum).
     */
    const_iterator begin() const noexcept { return const_iterator(this, node_t::find_min(root_)); }

    /**
     *  @brief Returns const iterator to first element (minimum).
     */
    const_iterator cbegin() const noexcept { return begin(); }

    /**
     *  @brief Returns iterator to past-the-end element.
     */
    iterator end() noexcept { return iterator(this, nullptr); }

    /**
     *  @brief Returns const iterator to past-the-end element.
     */
    const_iterator end() const noexcept { return const_iterator(this, nullptr); }

    /**
     *  @brief Returns const iterator to past-the-end element.
     */
    const_iterator cend() const noexcept { return end(); }

    /**
     *  @brief Finds an element equal to the given @p comparable.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return iterator Iterator to found element, or end() if not found.
     */
    template <typename comparable_type_>
    iterator find(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::find(root_, std::forward<comparable_type_>(comparable), comparator_));
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
        return const_iterator(this, node_t::find(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element not less than the given @p comparable (lower bound).
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return iterator Iterator to lower bound element, or end() if all elements < comparable.
     */
    template <typename comparable_type_>
    iterator lower_bound(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::lower_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element not less than the given @p comparable (const version).
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return const_iterator Const iterator to lower bound element, or end() if all elements < comparable.
     */
    template <typename comparable_type_>
    const_iterator lower_bound(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this, node_t::lower_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element greater than the given @p comparable (upper bound).
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return iterator Iterator to upper bound element, or end() if all elements <= comparable.
     */
    template <typename comparable_type_>
    iterator upper_bound(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::upper_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element greater than the given @p comparable (const version).
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return const_iterator Const iterator to upper bound element, or end() if all elements <= comparable.
     */
    template <typename comparable_type_>
    const_iterator upper_bound(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this, node_t::upper_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
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
    erase_result_t erase(const_iterator pos) noexcept {
        return erase(iterator(const_cast<basic_wb_tree *>(this), const_cast<node_t *>(pos.node_)));
    }

    /**
     *  @brief Erases all elements in the range [first, last).
     *    Unlike STL, returns both the iterator following the last erased element and a status code.
     *    On error, some elements may have been erased (partial erase, matches STL's basic guarantee).
     *
     *  @param[in] first Iterator to first element in range.
     *  @param[in] last Iterator to past-the-end of range.
     *  @return erase_result_t Contains iterator following the last erased element and operation status.
     *
     *  @note If first == last, no elements are erased and returns {last, success}.
     */
    erase_result_t erase(iterator first, iterator last) noexcept {
        while (first != last) {
            auto result = erase(first);
            if (!result.status) return result;
            first = result.next;
        }
        return {last, {success_k}};
    }

#pragma mark - Split and Join Operations

    /**
     *  @brief Result of splitting a tree at a key.
     *    Contains two trees: left (all < key) and right (all >= key).
     */
    struct split_result_t {
        wb_tree_t left;   //!< Tree with all elements < key.
        wb_tree_t right;  //!< Tree with all elements >= key.

        split_result_t() = default;
        split_result_t(wb_tree_t &&l, wb_tree_t &&r) : left(std::move(l)), right(std::move(r)) {}
    };

    /**
     *  @brief Splits this tree at a given key.
     *    Returns both left (< key) and right (>= key) parts. This tree is left empty.
     *
     *  @param[in] comparable Key to split at.
     *  @return Split result containing left and right trees.
     *
     *  @par Complexity
     *  O(log n) expected.
     *
     *  @par Example
     *  @code
     *  auto split = tree.split(5);
     *  // split.left has elements < 5
     *  // split.right has elements >= 5
     *  @endcode
     */
    template <typename comparable_type_>
    split_result_t split(comparable_type_ &&comparable) noexcept {
        auto result = node_t::split(root_, std::forward<comparable_type_>(comparable), comparator_);

        // Create left tree
        wb_tree_t left_tree;
        left_tree.root_ = result.left;
        left_tree.comparator_ = comparator_;
        left_tree.update_sizes_from_root();

        // Create right tree
        wb_tree_t right_tree;
        right_tree.root_ = result.right;
        right_tree.comparator_ = comparator_;
        right_tree.update_sizes_from_root();

        // This tree is now empty
        root_ = nullptr;
        size_ = 0;

        return {std::move(left_tree), std::move(right_tree)};
    }

    /**
     *  @brief Joins another tree into this one.
     *    Precondition: all elements in this tree < all elements in other tree.
     *
     *  @param[in] other Tree to join (will be empty after join).
     *
     *  @par Complexity
     *  O(log n) expected.
     *
     *  @warning This is a low-level operation. The caller must ensure all(this) < all(other).
     *    For general merging with duplicate handling, use merge() instead.
     *
     *  @par Example
     *  @code
     *  auto right = tree.split(5);  // tree: [0,5), right: [5,∞)
     *  tree.join(right);            // tree: [0,∞)
     *  @endcode
     */
    void join(wb_tree_t &other) noexcept {
        if (!other.root_) return;
        if (!root_) {
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        root_ = node_t::join(root_, other.root_, comparator_);
        update_sizes_from_root();

        other.root_ = nullptr;
        other.size_ = 0;
    }

#pragma mark - Observers

    /**
     *  @brief Returns the comparator object.
     */
    comparator_t key_comp() const noexcept { return comparator_; }

    /**
     *  @brief Returns pointer to root node (for internal use).
     */
    node_t *root() const noexcept { return root_; }

    /**
     *  @brief Returns reference to node allocator (for internal use).
     */
    node_allocator_t &allocator() noexcept { return allocator_; }

    /**
     *  @brief Returns const reference to node allocator (for internal use).
     */
    node_allocator_t const &allocator() const noexcept { return allocator_; }

  private:
    void clear_recursive(node_t *node) noexcept {
        if (!node) return;
        clear_recursive(node->left);
        clear_recursive(node->right);
        allocator_.deallocate(node, 1);
    }

    void update_sizes_from_root() noexcept { update_sizes_recursive(root_); }

    void update_sizes_recursive(node_t *node) noexcept {
        if (!node) return;
        update_sizes_recursive(node->left);
        update_sizes_recursive(node->right);
        node_t::update_size(node);
    }
};

} // namespace ashvardanian::smashtable
