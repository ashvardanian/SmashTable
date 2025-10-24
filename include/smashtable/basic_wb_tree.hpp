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

    // Additional methods (insert, erase, etc.) to be added...
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
     *  @brief Removes all elements from the tree.
     */
    void clear() noexcept {
        clear_recursive(root_);
        root_ = nullptr;
        size_ = 0;
    }

#pragma mark - Lookup

    /**
     *  @brief Finds an element equal to the given entry.
     *  @param[in] entry Entry to search for.
     *  @return Pointer to node if found, nullptr otherwise.
     */
    node_t *find(entry_t const &entry) noexcept { return node_t::find(root_, entry, comparator_); }

    /**
     *  @brief Finds an element equal to the given entry (const version).
     */
    node_t const *find(entry_t const &entry) const noexcept { return node_t::find(root_, entry, comparator_); }

    /**
     *  @brief Checks if an element exists in the tree.
     */
    bool contains(entry_t const &entry) const noexcept { return find(entry) != nullptr; }

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

#pragma mark - Observers

    /**
     *  @brief Returns the comparator object.
     */
    comparator_t key_comp() const noexcept { return comparator_; }

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
