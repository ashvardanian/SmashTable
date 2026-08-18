/**
 *  @brief Ordered "Adelson-Velsky and Landis" @b AVL Binary Search Tree implementation. Provides exception-free,
 *      allocator-aware ordered collection similar to @c std::set. Not thread-safe by itself. Doesn't raise any
 *      exceptions unlike STL-based alternatives.
 *  @author Ash Vardanian
 *  @file include/smashtable/basic_avl_tree.hpp
 *  @date October 13, 2022
 *
 *  @section basic_avl_tree_design_characteristics Design Characteristics
 *
 *  AVL trees maintain strict balance (height difference ≤ 1), providing O(log n) worst-case lookups, insertions,
 *  and deletions. Compared to Red-Black trees (used in @c std::set), AVL trees are more rigidly balanced, offering
 *  faster lookups at the cost of slightly slower insertions and deletions due to more frequent rebalancing.
 *
 *  This implementation supports heterogeneous lookups (searching with types other than @c value_type), custom
 *  allocators for all internal nodes, and callback-based iteration to avoid iterator invalidation complexity.
 *  All methods are @c noexcept and use status codes instead of exceptions for error handling.
 *
 *  @section basic_avl_tree_stl_interface_compatibility STL Interface Compatibility
 *
 *  Provides standard @c std::set interface:
 *  - @b Lookup: @c find, @c lower_bound, @c upper_bound, @c equal_range, @c contains, @c count
 *  - @b Modification: @c insert, @c emplace, @c erase, @c extract, @c clear, @c swap
 *  - @b Iteration: Bidirectional iterators (@c begin, @c end), range-based for loops, callback-based traversal
 *  - @b Capacity: @c size, @c empty, @c max_size
 *  - @b Observers: @c key_comp, @c value_comp
 *  - @b Bulk Operations: Range @c insert, @c merge
 *
 *  @section basic_avl_tree_advanced_operations Advanced Operations
 *
 *  Beyond STL, provides high-performance set operations and advanced variants:
 *
 *  @b Split/Join Operations:
 *  - @c split(): O(log n) partition into two trees at arbitrary key
 *  - @c join(): O(log n) concatenation of ordered disjoint trees (precondition: all(this) < all(other))
 *
 *  @b Adaptive Merge Algorithms:
 *  - @c merge(other): O(m log n) union with duplicate handling (deallocates duplicates)
 *  - @c merge(other, @b assume_unique_t): Optimized O(m log(n/m+1)) or O(m+n) for disjoint trees
 *    - Selects optimal algorithm based on tree characteristics:
 *    - O(log n) join for fully ordered trees
 *    - O(m log(n/m+1)) split-based merge (Blelloch et al., "Just Join for Parallel Ordered Sets", 2016)
 *    - O(m+n) Day-Stout-Warren spine merge for large similarly-sized trees
 *
 *  @b Bulk Construction:
 *  - @c insert(first, last, @b assume_sorted_t): O(n) perfect balancing from sorted ranges
 *
 *  @b Move-Only Type Support:
 *  - Compatible with @c std::move_iterator for bulk operations
 *  - @c emplace for in-place construction of move-only types
 *
 *  @section basic_avl_tree_requirements Requirements
 *
 *  @par Entry Type
 *  - Nothrow default-constructible and nothrow move constructible/assignable (required)
 *  - For @c copy(): Nothrow copy-constructible OR provides
 *      @code .copy() const -> expected<T> @endcode
 *
 *  @par Comparator Type
 *  - Must define @code bool operator()(entry_type const &, entry_type const &) const @endcode
 *  - For heterogeneous lookups, define @code using is_transparent = void; @endcode
 *
 *  @par Allocator Type
 *  - Must propagate on move assignment @c propagate_on_container_move_assignment==true
 *  - For @c swap(): If non-propagating, both trees must use equal allocators,
 *    otherwise @c invalid_argument_k is returned
 *
 *  @see https://en.wikipedia.org/wiki/AVL_tree
 */
#pragma once
#include <cassert> // `assert`

#include <limits>  // `std::numeric_limits`
#include <memory>  // `std::allocator`
#include <utility> // `std::exchange`

#include "shared.hpp"

namespace ashvardanian::smashtable {

/**
 *  @brief Non-owning tree node structure for the AVL-Tree implementation, unaware of memory management.
 *    Provides static methods for all tree operations: search, insertion, deletion, and traversal.
 *    All operations are exception-free and @c noexcept.
 *
 *  @section basic_avl_tree_implementation_notes Implementation Notes
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
 *  Layout: @c fruit is the stored entry, @c left, @c right and @c parent the links, and @c height the
 *  longest downward path to a leaf. The root carries the biggest @c height in the tree, a non-NULL node
 *  has at least one, and zero occurs only in the uninitialized detached state - which makes @c 1 << height
 *  an upper bound on the branch size.
 *
 *  @tparam value_type_ Type of entries to store in this tree.
 *    Those must be @c noexcept move-constructible.
 *
 *  @tparam comparator_type_ A comparator function object that overloads
 *    @code bool operator()(value_type_ const &, value_type_ const &) const @endcode.
 *    For heterogeneous lookups, define @code using is_transparent = void; @endcode inside the comparator.
 */
template <typename value_type_, typename comparator_type_>
class basic_avl_node {
  public:
    using value_t = value_type_;
    using comparator_t = comparator_type_;
    using height_t = std::ptrdiff_t;
    using node_t = basic_avl_node;

    value_t fruit;
    node_t *left = nullptr;
    node_t *right = nullptr;
    node_t *parent = nullptr;

    height_t height = 0;

    static height_t get_height(node_t *node) noexcept { return node ? node->height : 0; }
    static height_t get_balance(node_t *node) noexcept {
        return node ? get_height(node->left) - get_height(node->right) : 0;
    }

#pragma region Traversal and Search

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
     *  @param[in] node The node to find the successor for.
     *  @return The successor node, or @c nullptr if the node is the maximum element.
     */
    static node_t *find_successor(node_t *node) noexcept {
        if (!node) return nullptr;

        // If there is a right subtree, the successor is the minimum element in it.
        if (node->right) return find_min(node->right);

        // Otherwise, the successor is one of the ancestors.
        // Go up until we find a node that is a left child of its parent.
        node_t *parent = node->parent;
        while (parent && node == parent->right) {
            node = parent;
            parent = parent->parent;
        }
        return parent;
    }

    /**
     *  @brief Finds the previous node in in-order traversal (predecessor).
     *  @param[in] node The node to find the predecessor for.
     *  @return The predecessor node, or @c nullptr if the node is the minimum element.
     */
    static node_t *find_predecessor(node_t *node) noexcept {
        if (!node) return nullptr;

        // If there is a left subtree, the predecessor is the maximum element in it.
        if (node->left) return find_max(node->left);

        // Otherwise, the predecessor is one of the ancestors.
        // Go up until we find a node that is a right child of its parent.
        node_t *parent = node->parent;
        while (parent && node == parent->left) {
            node = parent;
            parent = parent->parent;
        }
        return parent;
    }

    /**
     *  @brief Searches for an entry in this subtree.
     *  @param[in] node The root of the subtree to search.
     *  @param[in] comparable Any key comparable with stored entries.
     *  @param[in] comparator The comparator instance (may be stateful).
     *  @return A pointer to the found node, or @c nullptr if nothing was found.
     */
    template <typename comparable_type_>
    static node_t *find(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        while (node) {
            if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit))) node = node->left;
            else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable)))
                node = node->right;
            else break;
        }
        return node;
    }

    /**
     *  @brief Finds the smallest entry that is bigger than or equal to the provided key.
     *  @param[in] node The root of the subtree to search.
     *  @param[in] comparable Any key comparable with stored entries.
     *  @param[in] comparator The comparator instance (may be stateful).
     *  @return A pointer to the found node, or @c nullptr if no such element was found.
     */
    template <typename comparable_type_>
    static node_t *lower_bound(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        node_t *successor = nullptr;
        while (node) {
            // If the given key is less than the root node, visit the left
            // subtree, taking current node as potential successor.
            if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit))) {
                successor = node;
                node = node->left;
            }

            // Of the given key is more than the root node, visit the right
            // subtree.
            else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable))) {
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
     *  @brief Finds the smallest entry that is bigger than the provided key.
     *  @param[in] node The root of the subtree to search.
     *  @param[in] comparable Any key comparable with stored entries.
     *  @param[in] comparator The comparator instance (may be stateful).
     *  @return A pointer to the found node, or @c nullptr if no such element was found.
     */
    template <typename comparable_type_>
    static node_t *upper_bound(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        node_t *successor = nullptr;
        while (node) {
            // If the given key is less than the root node, visit the left
            // subtree, taking current node as potential successor.
            if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit))) {
                successor = node;
                node = node->left;
            }
            // Otherwise, we must go right. This covers both the case where
            // the node's entry is less than the comparable, and the case where
            // they are equal.
            else { node = node->right; }
        }
        return successor;
    }

    /**
     *  @brief Searches for the lowest common ancestor of two keys.
     *  @param[in] node The root of the subtree to search.
     *  @param[in] a The first key.
     *  @param[in] b The second key.
     *  @param[in] comparator The comparator instance (may be stateful).
     *  @return A pointer to the lowest common ancestor, or @c nullptr if the tree is empty.
     */
    template <typename comparable_a_type_, typename comparable_b_type_>
    static node_t *lowest_common_ancestor(node_t *node, comparable_a_type_ &&a, comparable_b_type_ &&b,
                                          comparator_t const &comparator) noexcept {
        while (node) {
            if (comparator(mapping_key_or_itself(a), mapping_key_or_itself(node->fruit)) &&
                comparator(mapping_key_or_itself(b), mapping_key_or_itself(node->fruit)))
                node = node->left;
            else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(a)) &&
                     comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(b)))
                node = node->right;
            else return node;
        }
        return nullptr;
    }

    struct node_interval_t {
        node_t *lower_bound = nullptr;
        node_t *upper_bound = nullptr;
        node_t *lowest_common_ancestor = nullptr;
    };

    /**
     *  @brief Iterates over a range of entries.
     *  @param[in] node The root of the subtree to search.
     *  @param[in] low The lower bound of the range, included.
     *  @param[in] high The upper bound of the range, excluded.
     *  @param[in] comparator The comparator instance (may be stateful).
     *  @param[in] callback The function to call for each node in the range.
     */
    template <typename lower_type_, typename upper_type_, typename callback_type_>
    static void range(node_t *node, lower_type_ &&low, upper_type_ &&high, comparator_t const &comparator,
                      callback_type_ &&callback) noexcept {
        node_t *current = lower_bound(node, low, comparator);
        while (current && comparator(mapping_key_or_itself(current->fruit), mapping_key_or_itself(high))) {
            callback(current);
            current = find_successor(current);
        }
    }

    /**
     *  @brief Locates the nodes bounding all entries equal to @p comparable.
     *  @return Half-open bounds plus the lowest ancestor covering them, all @c nullptr on an empty tree.
     */
    template <typename comparable_type_>
    static node_interval_t equal_range(node_t *node, comparable_type_ &&comparable,
                                       comparator_t const &comparator) noexcept {
        return {lower_bound(node, comparable, comparator), upper_bound(node, comparable, comparator),
                lowest_common_ancestor(node, comparable, comparable, comparator)};
    }

#pragma endregion Traversal and Search

#pragma region Sampling

    /**
     *  @brief Random samples a single node using reservoir sampling for uniform distribution.
     *  @param[inout] generator Any STL-compatible random number generator.
     *  @return Pointer to randomly selected node, or @c nullptr if tree is empty.
     *  @note Uses single-pass reservoir sampling for O(n) time with true uniform distribution.
     */
    template <typename generator_type_>
    static node_t *sample(node_t *node, generator_type_ &&generator) noexcept {
        if (!node) return nullptr;

        node_t *result = nullptr;
        std::size_t count = 0;

        for_each_left_right(node, [&](node_t *current) noexcept {
            ++count;
            if (draw_below(generator, count) == 0) result = current;
        });

        return result;
    }

    /**
     *  @brief Random samples nodes within a given range of keys using reservoir sampling.
     *  @param[in] node Root of subtree.
     *  @param[in] low Lower bound.
     *  @param[in] high Upper bound.
     *  @param[in] comparator Comparator instance (may be stateful).
     *  @param[inout] generator Any STL-compatible random number generator.
     *  @param[in] predicate Predicate to filter nodes.
     *  @return NULL if nothing was found.
     *  @note Uses single-pass reservoir sampling for O(n) time with uniform distribution.
     */
    template <typename generator_type_, typename lower_type_, typename upper_type_, typename predicate_type_>
    static node_t *sample_range( //
        node_t *node, lower_type_ &&low, upper_type_ &&high, comparator_t const &comparator,
        generator_type_ &&generator, predicate_type_ &&predicate) noexcept {

        node_t *result = nullptr;
        std::size_t count = 0;
        range(node, low, high, comparator, [&](node_t *node) noexcept {
            if (!predicate(node)) return;
            ++count;
            if (draw_below(generator, count) == 0) result = node;
        });

        return result;
    }

    /**
     *  @brief Multi-sample reservoir sampling for k uniform random nodes in range.
     *  @param[in] node Root of the subtree to sample from.
     *  @param[in] low Lower bound for the range (inclusive).
     *  @param[in] high Upper bound for the range (exclusive).
     *  @param[inout] generator Random number generator (e.g., @c std::mt19937).
     *  @param[in] predicate Predicate to filter nodes.
     *  @param[inout] seen Count of matching nodes processed (can span multiple calls).
     *  @param[in] reservoir_capacity Maximum number of samples to collect.
     *  @param[out] reservoir Random access iterator to output buffer.
     *  @note Uses Algorithm R (reservoir sampling) for uniform k-sample in single pass.
     */
    template <typename generator_type_, typename lower_type_, typename upper_type_, typename predicate_type_,
              typename output_iterator_type_>
    static void sample_range( //
        node_t *node, lower_type_ &&low, upper_type_ &&high, comparator_t const &comparator,
        generator_type_ &&generator, predicate_type_ &&predicate, std::size_t &seen, std::size_t reservoir_capacity,
        output_iterator_type_ &&reservoir) noexcept {

        using output_iterator_t = std::remove_reference_t<output_iterator_type_>;
        using output_category_t = typename std::iterator_traits<output_iterator_t>::iterator_category;
        static_assert(std::is_same<std::random_access_iterator_tag, output_category_t>(), "Must be random access!");

        range(node, low, high, comparator, [&](node_t *node) noexcept {
            if (!predicate(node)) return;

            if (seen < reservoir_capacity) { reservoir[seen] = node; }
            else {
                auto slot_to_replace = draw_below(generator, seen + 1);
                if (slot_to_replace < reservoir_capacity) reservoir[slot_to_replace] = node;
            }

            ++seen;
        });
    }

#pragma endregion Sampling

#pragma region Rotations and Balancing

    static node_t *rotate_right(node_t *y) noexcept {
        node_t *x = y->left;
        node_t *z = x->right;

        // Perform rotation
        x->right = y;
        if (z) z->parent = y;
        y->left = z;

        // Update parent pointers
        x->parent = y->parent;
        y->parent = x;

        // Update heights
        y->height = larger_of(get_height(y->left), get_height(y->right)) + 1;
        x->height = larger_of(get_height(x->left), get_height(x->right)) + 1;
        return x;
    }

    static node_t *rotate_left(node_t *x) noexcept {
        node_t *y = x->right;
        node_t *z = y->left;

        // Perform rotation
        y->left = x;
        if (z) z->parent = x;
        x->right = z;

        // Update parent pointers
        y->parent = x->parent;
        x->parent = y;

        // Update heights
        x->height = larger_of(get_height(x->left), get_height(x->right)) + 1;
        y->height = larger_of(get_height(y->left), get_height(y->right)) + 1;
        return y;
    }

#pragma endregion Rotations and Balancing

#pragma region Insertions

    /** @brief What a walk that may create a node did with the key it was handed. */
    enum class node_placement_t : std::uint8_t {
        /** @brief No node was available to store the key, so nothing was stored. */
        refused_k,
        /** @brief An equal key was already there, and @c match names the node holding it. */
        matched_k,
        /** @brief A fresh node took the key, and @c match names it. */
        made_k,
    };

    struct find_or_make_result_t {
        node_t *root = nullptr;
        node_t *match = nullptr;
        node_placement_t placement = node_placement_t::refused_k;

        /** @brief Whether nothing was stored, which is the only way this walk fails. */
        bool failed() const noexcept { return placement == node_placement_t::refused_k; }
    };

    template <typename comparable_type_>
    inline static node_t *rebalance_on_insert(node_t *node, comparable_type_ &&comparable,
                                              comparator_t const &comparator) noexcept {
        // Update height and check if branches aren't balanced
        node->height = larger_of(get_height(node->left), get_height(node->right)) + 1;
        auto balance = get_balance(node);

        // Left Left Case
        if (balance > 1 && comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->left->fruit)))
            return rotate_right(node);

        // Right Right Case
        else if (balance < -1 &&
                 comparator(mapping_key_or_itself(node->right->fruit), mapping_key_or_itself(comparable)))
            return rotate_left(node);

        // Left Right Case
        else if (balance > 1 &&
                 comparator(mapping_key_or_itself(node->left->fruit), mapping_key_or_itself(comparable))) {
            node->left = rotate_left(node->left);
            return rotate_right(node);
        }
        // Right Left Case
        else if (balance < -1 &&
                 comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->right->fruit))) {
            node->right = rotate_right(node->right);
            return rotate_left(node);
        }
        else return node;
    }

    template <typename comparable_type_, typename callback_found_type_>
    static find_or_make_result_t find_or_make(node_t *node, comparable_type_ &&comparable,
                                              comparator_t const &comparator, callback_found_type_ &&callback_found,
                                              node_t *new_node) noexcept {
        if (!node) {
            if (new_node) {
                new_node->left = nullptr;
                new_node->right = nullptr;
                new_node->parent = nullptr;
                new_node->height = 1;
            }
            return {new_node, new_node, new_node ? node_placement_t::made_k : node_placement_t::refused_k};
        }

        if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit))) {
            auto subtree_result = find_or_make(node->left, comparable, comparator, callback_found, new_node);
            node->left = subtree_result.root;
            if (subtree_result.root) subtree_result.root->parent = node;
            if (subtree_result.placement == node_placement_t::made_k)
                node = rebalance_on_insert(node, subtree_result.match->fruit, comparator);
            return {node, subtree_result.match, subtree_result.placement};
        }
        else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable))) {
            auto subtree_result = find_or_make(node->right, comparable, comparator, callback_found, new_node);
            node->right = subtree_result.root;
            if (subtree_result.root) subtree_result.root->parent = node;
            if (subtree_result.placement == node_placement_t::made_k)
                node = rebalance_on_insert(node, subtree_result.match->fruit, comparator);
            return {node, subtree_result.match, subtree_result.placement};
        }
        else {
            // Equal keys are not allowed in BST
            callback_found(node);
            return {node, node, node_placement_t::matched_k};
        }
    }

    static find_or_make_result_t insert(node_t *node, node_t *new_child, comparator_t const &comparator) noexcept {
        return find_or_make(node, new_child->fruit, comparator, [](node_t *) noexcept {}, new_child);
    }

    /**
     *  @brief Builds a balanced AVL tree from sorted range in O(n) time.
     *    Precondition: Range [first, first+count) must be sorted according to comparator.
     *
     *  @param[in] first Iterator to beginning of sorted range.
     *  @param[in] count Number of elements in range.
     *  @param[in] allocate_node Allocator function that returns new node pointer or nullptr on failure.
     *  @return The root of the balanced tree, or @c nullptr if an allocation failed.
     *
     *  @note Complexity: O(n) time, O(log n) recursion depth.
     *    Builds perfectly balanced tree by recursively picking middle element as root.
     *  @warning If precondition violated (unsorted input), resulting tree has undefined structure.
     */
    template <typename iterator_type_, typename allocator_func_>
    static node_t *build_from_sorted(iterator_type_ first, std::size_t count,
                                     allocator_func_ &&allocate_node) noexcept {
        if (count == 0) return nullptr;

        // Find middle element
        std::size_t mid = count / 2;
        auto mid_iter = first;
        std::advance(mid_iter, mid);

        // Allocate root node
        node_t *root = allocate_node();
        if (!root) return nullptr;
        new (&root->fruit) value_t(*mid_iter);
        // Raw memory, so the link to the parent is only correct once the caller overwrites it.
        root->parent = nullptr;

        // Build left subtree from [first, mid)
        root->left = build_from_sorted(first, mid, allocate_node);
        if (root->left) root->left->parent = root;

        // Build right subtree from (mid, last)
        auto right_first = mid_iter;
        ++right_first;
        root->right = build_from_sorted(right_first, count - mid - 1, allocate_node);
        if (root->right) root->right->parent = root;

        // Set height (no balancing needed for perfectly balanced construction)
        root->height = 1 + larger_of(get_height(root->left), get_height(root->right));

        return root;
    }

#pragma endregion Insertions

#pragma region Removals

    struct extract_result_t {
        node_t *root = nullptr;
        std::unique_ptr<node_t> extracted;

        node_t *release() noexcept { return extracted.release(); }
    };

    static node_t *rebalance_after_extract(node_t *node) noexcept {
        node->height = 1 + larger_of(get_height(node->left), get_height(node->right));
        auto balance = get_balance(node);

        // Left Left Case
        if (balance > 1 && get_balance(node->left) >= 0) return rotate_right(node);

        // Left Right Case
        else if (balance > 1 && get_balance(node->left) < 0) {
            node->left = rotate_left(node->left);
            if (node->left) node->left->parent = node;
            return rotate_right(node);
        }

        // Right Right Case
        else if (balance < -1 && get_balance(node->right) <= 0) return rotate_left(node);

        // Right Left Case
        else if (balance < -1 && get_balance(node->right) > 0) {
            node->right = rotate_right(node->right);
            if (node->right) node->right->parent = node;
            return rotate_left(node);
        }
        else return node;
    }

    /**
     *  @brief Pops the root replacing it with one of descendants, if present.
     *  @param[in] node Node to extract.
     *  @param[in] comparator Comparator for element comparison.
     */
    static extract_result_t extract(node_t *node, comparator_t const &comparator) noexcept {

        // If the node has two children, replace it with the
        // smallest entry in the right branch.
        if (node->left && node->right) {
            node_t *successor = find_min(node->right);
            auto subtree_result = extract(node->right, successor->fruit, comparator);
            successor = subtree_result.extracted.release();
            successor->left = node->left;
            if (successor->left) successor->left->parent = successor;
            successor->right = subtree_result.root;
            if (successor->right) successor->right->parent = successor;
            successor->parent = node->parent;
            successor->height = 1 + larger_of(get_height(successor->left), get_height(successor->right));
            // Detach the `node` from the descendants.
            node->left = node->right = node->parent = nullptr;
            node->height = 1;
            // The promoted node inherits the shrunk right branch, so it may itself be off balance,
            // and no ancestor rebalances it when the extracted node was the root.
            return {rebalance_after_extract(successor), std::unique_ptr<node_t> {node}};
        }
        // Just one child is present, so it is the natural successor.
        else if (node->left || node->right) {
            node_t *replacement = node->left ? node->left : node->right;
            replacement->parent = node->parent;
            // Detach the `node` from the descendants.
            node->left = node->right = node->parent = nullptr;
            node->height = 1;
            return {replacement, std::unique_ptr<node_t> {node}};
        }
        // No children are present.
        else {
            // Detach the `node` from the descendants.
            node->left = node->right = node->parent = nullptr;
            node->height = 1;
            return {nullptr, std::unique_ptr<node_t> {node}};
        }
    }

    /**
     *  @brief Searches for a matching ancestor and extracts it out.
     *  @param[in] comparable Any key comparable with stored entries.
     */
    template <typename comparable_type_>
    static extract_result_t extract(node_t *node, comparable_type_ &&comparable,
                                    comparator_t const &comparator) noexcept {
        if (!node) return {node, {}};

        if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit))) {
            auto subtree_result = extract(node->left, comparable, comparator);
            node->left = subtree_result.root;
            if (subtree_result.root) subtree_result.root->parent = node;
            if (subtree_result.extracted) node = rebalance_after_extract(node);
            return {node, std::move(subtree_result.extracted)};
        }

        else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable))) {
            auto subtree_result = extract(node->right, comparable, comparator);
            node->right = subtree_result.root;
            if (subtree_result.root) subtree_result.root->parent = node;
            if (subtree_result.extracted) node = rebalance_after_extract(node);
            return {node, std::move(subtree_result.extracted)};
        }

        else
            // We have found the node to extract!
            return extract(node, comparator);
    }

    struct erase_if_result_t {
        node_t *root = nullptr;
        std::size_t count = 0;
    };

    template <typename predicate_type_, typename node_deallocator_type_>
    static erase_if_result_t erase_if(node_t *node, predicate_type_ &&predicate,
                                      node_deallocator_type_ &&node_deallocator,
                                      comparator_t const &comparator) noexcept {
        if (!node) return {nullptr, 0};

        auto left_result = erase_if(node->left, predicate, node_deallocator, comparator);
        node->left = left_result.root;
        if (node->left) node->left->parent = node;

        auto right_result = erase_if(node->right, predicate, node_deallocator, comparator);
        node->right = right_result.root;
        if (node->right) node->right->parent = node;

        std::size_t count = left_result.count + right_result.count;

        if (predicate(node->fruit)) {
            auto extract_result = extract(node, comparator);
            node_deallocator(extract_result.extracted.release());
            return {extract_result.root, count};
        }
        else {
            node->height = 1 + larger_of(get_height(node->left), get_height(node->right));
            return {node, count + 1};
        }
    }

#pragma endregion Removals

#pragma region Split and Join

    /**
     *  @brief Result of a split operation.
     */
    struct split_result_t {
        /** @brief Subtree with all elements ordered before the split key. */
        node_t *left = nullptr;
        /** @brief Subtree with all elements not ordered before the split key. */
        node_t *right = nullptr;
    };

    /**
     *  @brief Joins two trees with a root node between them.
     *    Precondition: All elements in left < root < all elements in right.
     *    Maintains AVL balance property.
     *
     *  @param[in] left Left subtree (all elements < root).
     *  @param[in] root_node Root node to insert between subtrees.
     *  @param[in] right Right subtree (all elements > root).
     *  @return The root of the joined tree.
     */
    static node_t *join_with_root(node_t *left, node_t *root_node, node_t *right,
                                  comparator_t const &comparator) noexcept {
        if (!root_node) return join(left, right, comparator);
        node_t *const joined = join_with_root_(left, root_node, right, comparator);
        joined->parent = nullptr;
        return joined;
    }

    /**
     *  @brief Descends the right spine of the taller left tree - or the left spine of the taller right
     *    tree - until both sides match in height, hangs @p root_node there, and rebalances on the way out.
     *    The returned root keeps whatever parent the recursion frame above it will overwrite.
     */
    static node_t *join_with_root_(node_t *left, node_t *root_node, node_t *right,
                                   comparator_t const &comparator) noexcept {
        height_t const left_height = get_height(left);
        height_t const right_height = get_height(right);

        if (left_height > right_height + 1) {
            left->right = join_with_root_(left->right, root_node, right, comparator);
            left->right->parent = left;
            left->height = 1 + larger_of(get_height(left->left), get_height(left->right));
            return rebalance_after_extract(left);
        }

        if (right_height > left_height + 1) {
            right->left = join_with_root_(left, root_node, right->left, comparator);
            right->left->parent = right;
            right->height = 1 + larger_of(get_height(right->left), get_height(right->right));
            return rebalance_after_extract(right);
        }

        root_node->left = left;
        if (left) left->parent = root_node;
        root_node->right = right;
        if (right) right->parent = root_node;
        root_node->height = 1 + larger_of(left_height, right_height);
        return root_node;
    }

    /**
     *  @brief Joins two AVL trees into one.
     *    Precondition: All elements in left < all elements in right.
     *    Maintains AVL balance property.
     *
     *  @param[in] left Left tree (smaller elements).
     *  @param[in] right Right tree (larger elements).
     *  @param[in] comparator Comparator for element comparison.
     *  @return The root of the joined tree.
     */
    static node_t *join(node_t *left, node_t *right, comparator_t const &comparator) noexcept {
        if (!left) return right;
        if (!right) return left;

        auto left_height = get_height(left);
        auto right_height = get_height(right);

        // If left tree is taller, join with right subtree of left
        if (left_height > right_height + 1) {
            left->right = join(left->right, right, comparator);
            if (left->right) left->right->parent = left;
            left->height = 1 + larger_of(get_height(left->left), get_height(left->right));
            return rebalance_after_extract(left);
        }
        // If right tree is taller, join with left subtree of right
        else if (right_height > left_height + 1) {
            right->left = join(left, right->left, comparator);
            if (right->left) right->left->parent = right;
            right->height = 1 + larger_of(get_height(right->left), get_height(right->right));
            return rebalance_after_extract(right);
        }
        // Heights are balanced, extract min from right and use as root
        else {
            node_t *min_right = find_min(right);
            auto extract_result = extract(right, min_right->fruit, comparator);
            node_t *new_root = extract_result.release();
            return join_with_root(left, new_root, extract_result.root, comparator);
        }
    }

    /**
     *  @brief Splits an AVL tree at a given key.
     *    Elements less than the key go to left tree, elements >= key go to right tree.
     *    Maintains AVL balance property in both resulting trees.
     *
     *  @param[in] node Root of the tree to split.
     *  @param[in] comparable Key to split at.
     *  @return Contains left tree (< key) and right tree (>= key).
     */
    template <typename comparable_type_>
    static split_result_t split(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        if (!node) return {nullptr, nullptr};

        // If node < key, put node in left tree and split right subtree
        if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable))) {
            auto subtree_result = split(node->right, comparable, comparator);
            if (subtree_result.right) subtree_result.right->parent = nullptr;
            auto new_left = join_with_root(node->left, node, subtree_result.left, comparator);
            new_left->parent = nullptr;
            return {new_left, subtree_result.right};
        }
        // If key <= node, put node in right tree and split left subtree
        else {
            auto subtree_result = split(node->left, comparable, comparator);
            if (subtree_result.left) subtree_result.left->parent = nullptr;
            auto new_right = join_with_root(subtree_result.right, node, node->right, comparator);
            new_right->parent = nullptr;
            return {subtree_result.left, new_right};
        }
    }

#pragma endregion Split and Join

#pragma region Merge Algorithms

    /**
     *  @brief Merges two AVL trees using split-based divide-and-conquer.
     *    Optimal for unbalanced sizes: O(m log(n/m + 1)) where m <= n.
     *    Takes root of smaller tree, splits larger tree around it, recursively merges.
     *
     *  @param[in] small Smaller tree to merge (will be consumed).
     *  @param[in] large Larger tree to merge into (will be consumed).
     *  @return The root of the merged tree, holding every node from both inputs.
     *
     *  @note Both input trees are consumed (ownership transferred).
     *    Maintains AVL balance property throughout.
     *  @see Blelloch et al., "Just Join for Parallel Ordered Sets" (2016)
     */
    static node_t *merge_split_based(node_t *small, node_t *large, comparator_t const &comparator) noexcept {
        if (!small) return large;
        if (!large) return small;

        // Extract root of smaller tree as pivot
        node_t *pivot = small;
        node_t *small_left = pivot->left;
        node_t *small_right = pivot->right;

        // Split larger tree around pivot's key: O(log n)
        auto split_result = split(large, pivot->fruit, comparator);

        // Recursively merge subtrees
        node_t *merged_left = merge_split_based(small_left, split_result.left, comparator);
        node_t *merged_right = merge_split_based(small_right, split_result.right, comparator);

        // Join with pivot as root: O(log height_diff)
        return join_with_root(merged_left, pivot, merged_right, comparator);
    }

    /**
     *  @brief Converts AVL tree to right-leaning spine (degenerate vine).
     *    A spine is a sorted linked list using right pointers, all left pointers null.
     *
     *  @param[in] root Tree to convert.
     *  @param[out] count Number of nodes in resulting spine.
     *  @return The head of the spine, which is the smallest element.
     *
     *  @note Uses rotations to flatten tree: O(n) time, O(1) space.
     */
    static node_t *tree_to_spine(node_t *root, std::size_t &count) noexcept {
        node_t *spine_head = nullptr;
        node_t *spine_tail = nullptr;
        count = 0;

        // Stack-free Morris-like traversal using rotations
        while (root) {
            root->parent = nullptr;
            if (root->left) {
                // Rotate right to bring left child up
                node_t *temp = root->left;
                root->left = temp->right;
                if (root->left) root->left->parent = root;
                temp->right = root;
                root->parent = temp;
                root = temp;
            }
            else {
                // Link current node to spine
                if (!spine_head) spine_head = root;
                if (spine_tail) spine_tail->right = root;
                spine_tail = root;
                ++count;

                // Move to right subtree
                node_t *next = root->right;
                root->left = nullptr;
                root = next;
            }
        }

        if (spine_tail) spine_tail->right = nullptr;
        return spine_head;
    }

    /**
     *  @brief Merges two sorted spines into one sorted spine.
     *
     *  @param[in] first_spine First sorted spine.
     *  @param[in] second_spine Second sorted spine.
     *  @return The head of the merged spine.
     *
     *  @note O(n+m) time, O(1) space. Just like merging sorted linked lists.
     */
    static node_t *merge_spines(node_t *first_spine, node_t *second_spine, comparator_t const &comparator) noexcept {
        if (!first_spine) return second_spine;
        if (!second_spine) return first_spine;

        node_t *merged_head = nullptr;
        node_t *merged_tail = nullptr;

        while (first_spine && second_spine) {
            node_t *next_node;
            if (comparator(mapping_key_or_itself(first_spine->fruit), mapping_key_or_itself(second_spine->fruit))) {
                next_node = first_spine;
                first_spine = first_spine->right;
            }
            else {
                next_node = second_spine;
                second_spine = second_spine->right;
            }

            if (!merged_head) merged_head = next_node;
            if (merged_tail) merged_tail->right = next_node;
            merged_tail = next_node;
        }

        // Append remaining nodes
        node_t *remaining = first_spine ? first_spine : second_spine;
        if (merged_tail) merged_tail->right = remaining;
        else merged_head = remaining;

        return merged_head;
    }

    /**
     *  @brief Converts sorted spine back to balanced AVL tree (Day-Stout-Warren algorithm).
     *
     *  @param[in] spine Head of spine.
     *  @param[in] count Number of nodes in spine.
     *  @return The root of the balanced tree.
     *
     *  @note O(n) time using rotations to build perfectly balanced tree.
     *  @see Stout & Warren, "Tree Rebalancing in Optimal Time and Space" (1986)
     */
    static node_t *spine_to_balanced(node_t *spine, std::size_t count) noexcept {
        if (count == 0) return nullptr;
        if (count == 1) {
            spine->left = nullptr;
            spine->right = nullptr;
            spine->parent = nullptr;
            spine->height = 1;
            return spine;
        }

        // Build perfectly balanced tree recursively from spine
        auto build_tree = [](node_t *&current, std::size_t n, auto &build_ref) -> node_t * {
            if (n == 0) return nullptr;

            // Build left subtree with n/2 nodes
            node_t *left = build_ref(current, n / 2, build_ref);

            // Current node becomes root
            node_t *root = current;
            current = current->right;

            // Build right subtree with remaining nodes
            node_t *right = build_ref(current, n - n / 2 - 1, build_ref);

            root->left = left;
            if (left) left->parent = root;
            root->right = right;
            if (right) right->parent = root;
            root->height = 1 + larger_of(get_height(left), get_height(right));
            return root;
        };

        auto root = build_tree(spine, count, build_tree);
        if (root) root->parent = nullptr;
        return root;
    }

    /**
     *  @brief Merges two AVL trees using Day-Stout-Warren spine algorithm.
     *    Best for large similarly-sized trees: O(n+m) time, O(1) space.
     *    Flattens both trees to spines, merges spines, rebuilds balanced tree.
     *
     *  @param[in] first_tree First tree to merge (will be consumed).
     *  @param[in] second_tree Second tree to merge (will be consumed).
     *  @param[in] comparator Comparator for element comparison.
     *  @param[out] out_size Total number of nodes in result.
     *  @return The root of the merged, rebalanced tree.
     *
     *  @note High constant factor due to many rotations, but O(1) space.
     *    Both input trees are consumed (ownership transferred).
     *
     *  @see Stout & Warren, "Tree Rebalancing in Optimal Time and Space" (1986)
     *  @see https://en.wikipedia.org/wiki/Day%E2%80%93Stout%E2%80%93Warren_algorithm
     */
    static node_t *merge_dsw(node_t *first_tree, node_t *second_tree, comparator_t const &comparator,
                             std::size_t &out_size) noexcept {
        std::size_t first_count = 0, second_count = 0;

        // Convert both trees to spines: O(n + m)
        node_t *first_spine = tree_to_spine(first_tree, first_count);
        node_t *second_spine = tree_to_spine(second_tree, second_count);

        // Merge spines: O(n + m)
        node_t *merged_spine = merge_spines(first_spine, second_spine, comparator);

        out_size = first_count + second_count;

        // Rebuild balanced tree from spine: O(n + m)
        return spine_to_balanced(merged_spine, out_size);
    }
};

/**
 *  @brief  Exception-free AVL tree container providing ordered storage similar to @c std::set.
 *    Manages memory allocation and provides high-level tree operations with status-based error handling.
 *
 *  @section basic_avl_tree_api_design API Design
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
 *  @section basic_avl_tree_entry_constraints Entry Constraints
 *
 *  Entries stored in the tree must be comparable using the provided comparator.
 *  They must also be @c noexcept move-constructible and assignable to ensure safety.
 *  When dealing with bulk-insertions of non-copyable types, make sure to use a moving
 *  iterator, like the @c std::move_iterator.
 *
 *  @section basic_avl_tree_insert_strategies Insert Strategies
 *
 *  Four distinct modification strategies with different failure handling:
 *  | Method               | Key Exists?     | Returns          | Use Case                                      |
 *  |----------------------|-----------------|------------------|-----------------------------------------------|
 *  | insert()             | Fails (error)   | invalid_arg_k    | Strict: ensure key is new                     |
 *  | insert_if_missing()  | Skips (success) | success_k        | Lenient: insert only if absent, else no-op    |
 *  | upsert()             | Overwrites      | success_k        | Always update regardless of existence         |
 *  | update()             | Fails (error)   | key_not_found_k  | Strict: ensure key exists before updating     |
 *
 *  @tparam value_type_ Type of entries stored in the tree. Often a @c mapping for associative containers.
 *  @tparam comparator_type_ Comparator for ordering entries. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes. Must be rebindable to @c basic_avl_node.
 */
template <typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<basic_avl_node<value_type_, comparator_type_>>>
class basic_avl_tree {
  public:
    using node_t = basic_avl_node<value_type_, comparator_type_>;
    using node_type = node_t;

    using allocator_t = std::allocator_traits<allocator_type_>::template rebind_alloc<node_t>;
    using allocator_type = allocator_t; // ? STL compatibility

    using comparator_t = comparator_type_;
    using comparator_type = comparator_t; // ? STL compatibility

    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility

    // SFINAE to extract key_type for maps, or use value_type for sets.
    using key_t = typename mapping_key_type_or_itself<value_t>::type;
    using key_type = key_t; // ? STL compatibility

    // SFINAE to extract mapped_type for maps, or void for sets.
    using mapped_t = typename mapped_value_type_or_void<value_t>::type;
    using mapped_type = mapped_t; // ? STL compatibility

    // Trait to indicate this container uses iterator-based reads (not callbacks)
    using callback_reads = std::false_type;

    // Allocator propagation requirement for exception-free move assignment
    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "basic_avl_tree requires allocators that propagate on move assignment");

    using is_associative = std::bool_constant<is_mapping<value_t>>;

    /**
     *  @brief Rebind this tree type to different entry and comparator types.
     *    Follows STL allocator rebind pattern for type transformations.
     *
     *  @tparam other_value_type_ New entry type for the rebound tree.
     *  @tparam other_comparator_type_ New comparator type for the rebound tree.
     */
    template <typename other_value_type_, typename other_comparator_type_>
    using rebind = basic_avl_tree<other_value_type_, other_comparator_type_,
                                  typename std::allocator_traits<allocator_t>::template rebind_alloc<
                                      basic_avl_node<other_value_type_, other_comparator_type_>>>;

    class iterator;
    class const_iterator;

    /**
     *  @brief Result of an erase operation on an iterator.
     *    Combines iterator to next element with operation status.
     */
    struct erase_result_t {
        /** @brief Iterator to the element following the erased one, or @c end(). */
        iterator next;
        /** @brief Status of the erase operation. */
        status_t status = success_k;
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
        using value_type = value_t;
        using difference_type = std::ptrdiff_t;
        using pointer = value_t *;
        using reference = value_t &;

      private:
        basic_avl_tree const *tree_;
        node_t *node_;

        iterator(basic_avl_tree const *tree, node_t *node) noexcept : tree_(tree), node_(node) {}

      public:
        iterator() noexcept : tree_(nullptr), node_(nullptr) {}

        reference operator*() const noexcept { return node_->fruit; }
        pointer operator->() const noexcept { return &node_->fruit; }

        iterator &operator++() noexcept {
            node_ = node_t::find_successor(node_);
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator previous = *this;
            ++(*this);
            return previous;
        }

        iterator &operator--() noexcept {
            if (!node_) node_ = node_t::find_max(tree_->root_);
            else node_ = node_t::find_predecessor(node_);
            return *this;
        }

        iterator operator--(int) noexcept {
            iterator previous = *this;
            --(*this);
            return previous;
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
        using value_type = value_t const;
        using difference_type = std::ptrdiff_t;
        using pointer = value_t const *;
        using reference = value_t const &;

      private:
        basic_avl_tree const *tree_;
        node_t const *node_;

        const_iterator(basic_avl_tree const *tree, node_t const *node) noexcept : tree_(tree), node_(node) {}

      public:
        const_iterator() noexcept : tree_(nullptr), node_(nullptr) {}
        const_iterator(iterator const &it) noexcept : tree_(it.tree_), node_(it.node_) {}

        reference operator*() const noexcept { return node_->fruit; }
        pointer operator->() const noexcept { return &node_->fruit; }

        const_iterator &operator++() noexcept {
            node_ = node_t::find_successor(const_cast<node_t *>(node_));
            return *this;
        }

        const_iterator operator++(int) noexcept {
            const_iterator previous = *this;
            ++(*this);
            return previous;
        }

        const_iterator &operator--() noexcept {
            if (!node_) node_ = node_t::find_max(tree_->root_);
            else node_ = node_t::find_predecessor(const_cast<node_t *>(node_));
            return *this;
        }

        const_iterator operator--(int) noexcept {
            const_iterator previous = *this;
            --(*this);
            return previous;
        }

        bool operator==(const_iterator const &other) const noexcept { return node_ == other.node_; }
        bool operator!=(const_iterator const &other) const noexcept { return node_ != other.node_; }
    };

    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  private:
    node_t *root_ = nullptr;
    std::size_t size_ = 0;
    ST_NO_UNIQUE_ADDRESS_ comparator_t comparator_;
    ST_NO_UNIQUE_ADDRESS_ allocator_t allocator_;

    /**
     *  @brief Checks if ANY key from other tree exists in this tree.
     *    Uses simultaneous in-order traversal (merge-style algorithm).
     *
     *  @param[in] other Tree to check for intersection with.
     *  @return True if at least one key exists in both trees, false otherwise.
     *
     *  @note Complexity: O(m + n) - optimal for sorted set intersection check.
     */
    bool has_any_key(basic_avl_tree const &other) const noexcept {
        auto it1 = begin();
        auto it2 = other.begin();

        while (it1 != end() && it2 != other.end()) {
            if (comparator_(mapping_key_or_itself(*it1), mapping_key_or_itself(*it2))) ++it1;
            else if (comparator_(mapping_key_or_itself(*it2), mapping_key_or_itself(*it1))) ++it2;
            else return true;
        }
        return false; // No intersection
    }

    /**
     *  @brief Checks if ALL keys from other tree exist in this tree.
     *    Uses simultaneous in-order traversal (merge-style algorithm).
     *
     *  @param[in] other Tree whose keys to check for presence in this tree.
     *  @return True if all keys from other exist in this tree, false otherwise.
     *
     *  @note Complexity: O(m + n) - optimal for sorted subset check.
     */
    bool has_all_keys(basic_avl_tree const &other) const noexcept {
        auto it1 = begin();
        auto it2 = other.begin();

        while (it1 != end() && it2 != other.end()) {
            if (comparator_(mapping_key_or_itself(*it1), mapping_key_or_itself(*it2)))
                ++it1; // this < other, advance this
            else if (comparator_(mapping_key_or_itself(*it2), mapping_key_or_itself(*it1)))
                return false; // other key missing in this
            else {
                ++it1;
                ++it2; // Equal - match found, advance both
            }
        }

        return it2 == other.end(); // True if we matched all of other's keys
    }

    /**
     *  @brief Merges another tree into this one with upsert semantics.
     *    Inserts new keys and updates existing keys. Always empties the other tree.
     *
     *  Every node travels across by relinking, so nothing is allocated and nothing can be refused -
     *  which is what lets the callers promise all-or-nothing once their staging tree is built.
     *
     *  @param[inout] other Tree to merge from. Will be empty after merge.
     *
     *  @note Unlike @c merge(), this UPDATES nodes with duplicate keys instead of skipping them.
     *  @note Complexity: O(m log n) where m = other.size(), n = this.size().
     */
    void merge_with_upsert(basic_avl_tree &other) noexcept {
        if (other.empty()) return;
        if (empty()) {
            // Move other into this
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        node_t::for_each_bottom_up(other.root_, [&](node_t *node) noexcept {
            auto result = node_t::insert(root_, node, comparator_);
            root_ = result.root;
            if (root_) root_->parent = nullptr;
            size_ += result.placement == node_t::node_placement_t::made_k;
            // A key already here keeps its own node and takes the entry, so the traveller goes back.
            if (result.placement == node_t::node_placement_t::matched_k) {
                result.match->fruit = std::move(node->fruit);
                node->fruit.~value_t();
                other.allocator_.deallocate(node, 1);
            }
        });

        other.root_ = nullptr;
        other.size_ = 0;
    }

    /**
     *  @brief RAII guard for managing subtree cleanup on copy failure.
     *    Automatically cleans up allocated nodes if not explicitly released.
     */
    struct subtree_guard_t {
        allocator_t *allocator_;
        node_t *node_;

        subtree_guard_t(allocator_t *allocator, node_t *node) noexcept : allocator_(allocator), node_(node) {}

        ~subtree_guard_t() noexcept {
            if (node_) cleanup_();
        }

        node_t *release() noexcept { return std::exchange(node_, nullptr); }

        void cleanup_() noexcept {
            node_t::for_each_bottom_up(node_, [&](node_t *n) noexcept {
                n->fruit.~value_t();
                allocator_->deallocate(n, 1);
            });
        }
    };

    /**
     *  @brief Copies entry from source node into destination node.
     *  @param[in] source Source node to copy from.
     *  @param[in] destination Destination node (must have allocated memory, but entry not constructed).
     *  @return @c success_k when the copy completed, an error code otherwise.
     */
    static status_t copy_entry_into_(node_t *source, node_t *destination) noexcept {
        auto entry_copy = copy_safely(source->fruit);
        if (!entry_copy) return entry_copy.status();
        new (&destination->fruit) value_t(std::move(*entry_copy));
        return success_k;
    }

    /**
     *  @brief Recursively copies a subtree.
     *  @param[in] source Root of source subtree to copy.
     *  @param[inout] allocator Allocator for node allocation.
     *  @return The root of the copied subtree, or @c nullptr if a copy failed.
     */
    static node_t *copy_subtree_(node_t *source, allocator_t &allocator) noexcept {
        if (!source) return nullptr;

        // Allocate new node
        node_t *new_node = allocator.allocate(1);
        if (!new_node) return nullptr;

        // Copy entry into new node
        auto status = copy_entry_into_(source, new_node);
        if (failed(status)) {
            allocator.deallocate(new_node, 1);
            return nullptr;
        }

        // Initialize node structure. Nodes come from raw memory, so every link needs an explicit value.
        new_node->height = source->height;
        new_node->left = nullptr;
        new_node->right = nullptr;
        new_node->parent = nullptr;

        // Use RAII guard to ensure cleanup on failure
        subtree_guard_t guard(&allocator, new_node);

        // Recursively copy left subtree
        new_node->left = copy_subtree_(source->left, allocator);
        if (source->left && !new_node->left) return nullptr;
        if (new_node->left) new_node->left->parent = new_node;

        // Recursively copy right subtree
        new_node->right = copy_subtree_(source->right, allocator);
        if (source->right && !new_node->right) return nullptr;
        if (new_node->right) new_node->right->parent = new_node;

        // Success - release guard and return
        return guard.release();
    }

  public:
#pragma endregion Merge Algorithms

#pragma region Constructors and Assignment

    basic_avl_tree() noexcept = default;
    explicit basic_avl_tree(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}

    /** @brief Seeds both policies, which a stateful comparator needs to survive a rebind. */
    basic_avl_tree(comparator_t comparator, allocator_t allocator) noexcept
        : comparator_(std::move(comparator)), allocator_(std::move(allocator)) {}

    basic_avl_tree(basic_avl_tree &&other) noexcept
        : root_(std::exchange(other.root_, nullptr)), size_(std::exchange(other.size_, 0)),
          comparator_(std::move(other.comparator_)), allocator_(std::move(other.allocator_)) {}

    basic_avl_tree(basic_avl_tree const &) = delete;
    basic_avl_tree &operator=(basic_avl_tree const &) = delete;

    basic_avl_tree &operator=(basic_avl_tree &&other) noexcept {
        if (this == &other) return *this;
        clear();
        root_ = std::exchange(other.root_, nullptr);
        size_ = std::exchange(other.size_, 0);
        comparator_ = std::move(other.comparator_);
        allocator_ = std::move(other.allocator_);
        return *this;
    }

    ~basic_avl_tree() noexcept { clear(); }

    /**
     *  @brief Creates a deep copy of the tree.
     *  @return Copy of the tree, or error status on failure.
     *  @note This operation may fail due to allocation errors during node duplication.
     *    The returned tree uses a copy of this tree's allocator.
     */
    expected<basic_avl_tree> copy() const noexcept {
        basic_avl_tree result {allocator_};
        result.comparator_ = comparator_;

        if (!root_) return expected<basic_avl_tree>(std::move(result), success_k);

        result.root_ = copy_subtree_(root_, result.allocator_);
        if (!result.root_) return expected<basic_avl_tree>(basic_avl_tree(allocator_), out_of_memory_heap_k);

        result.size_ = size_;
        return expected<basic_avl_tree>(std::move(result), success_k);
    }

#pragma endregion Constructors and Assignment

#pragma region Capacity

    /**
     *  @brief Returns the number of elements in the tree.
     *  @return Number of elements.
     */
    std::size_t size() const noexcept { return size_; }

    /**
     *  @brief Returns the height of the tree (number of edges in longest path from root to leaf).
     *  @return Height of the tree, 0 if empty.
     */
    std::size_t height() noexcept { return root_ ? root_->height : 0; }

    /**
     *  @brief Returns raw pointer to the root node. For internal use.
     *  @return The root node, or @c nullptr when the tree is empty.
     */
    node_t *root() const noexcept { return root_; }

    /**
     *  @brief Returns the allocator associated with the tree.
     *  @return The tree's allocator, for callers that need to allocate alongside it.
     */
    allocator_t &allocator() noexcept { return allocator_; }

    /**
     *  @brief Returns the allocator associated with the tree (const version).
     *  @return The tree's allocator, for callers that need to inspect its state.
     */
    allocator_t const &allocator() const noexcept { return allocator_; }

#pragma endregion Capacity

#pragma region Iterators

    /**
     *  @brief Returns an iterator to the first element (minimum) in the tree.
     *  @return Iterator to the minimum element, or end() if empty.
     */
    iterator begin() noexcept { return iterator(this, node_t::find_min(root_)); }

    /**
     *  @brief Returns a const iterator to the first element (minimum) in the tree.
     *  @return Const iterator to the minimum element, or end() if empty.
     */
    const_iterator begin() const noexcept { return const_iterator(this, node_t::find_min(root_)); }

    /**
     *  @brief Returns a const iterator to the first element (minimum) in the tree.
     *  @return Const iterator to the minimum element, or end() if empty.
     */
    const_iterator cbegin() const noexcept { return const_iterator(this, node_t::find_min(root_)); }

    /**
     *  @brief Returns an iterator to one past the last element.
     *  @return End iterator (points to nullptr).
     */
    iterator end() noexcept { return iterator(this, nullptr); }

    /**
     *  @brief Returns a const iterator to one past the last element.
     *  @return End iterator (points to nullptr).
     */
    const_iterator end() const noexcept { return const_iterator(this, nullptr); }

    /**
     *  @brief Returns a const iterator to one past the last element.
     *  @return End iterator (points to nullptr).
     */
    const_iterator cend() const noexcept { return const_iterator(this, nullptr); }

    /**
     *  @brief Returns a reverse iterator to the first element of the reversed tree (maximum element).
     *  @return Reverse iterator to the maximum element, or rend() if empty.
     */
    reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }

    /**
     *  @brief Returns a const reverse iterator to the first element of the reversed tree (maximum element).
     *  @return Const reverse iterator to the maximum element, or rend() if empty.
     */
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }

    /**
     *  @brief Returns a const reverse iterator to the first element of the reversed tree (maximum element).
     *  @return Const reverse iterator to the maximum element, or rend() if empty.
     */
    const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }

    /**
     *  @brief Returns a reverse iterator to one past the last element of the reversed tree (before minimum).
     *  @return Reverse end iterator.
     */
    reverse_iterator rend() noexcept { return reverse_iterator(begin()); }

    /**
     *  @brief Returns a const reverse iterator to one past the last element of the reversed tree (before minimum).
     *  @return Const reverse end iterator.
     */
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

    /**
     *  @brief Returns a const reverse iterator to one past the last element of the reversed tree (before minimum).
     *  @return Const reverse end iterator.
     */
    const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

    std::size_t total_imbalance() const noexcept {
        std::size_t abs_sum = 0;
        node_t::for_each_top_down(root_,
                                  [&](node_t *node) noexcept { abs_sum += std::abs(node_t::get_balance(node)); });
        return abs_sum;
    }

#pragma endregion Iterators

#pragma region Lookup

    /**
     *  @brief Finds an element equal to the given @p comparable.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Iterator to found element, or end() if not found.
     */
    template <typename comparable_type_>
    iterator find(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::find(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds an element equal to the given @p comparable (const version).
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Const iterator to found element, or end() if not found.
     */
    template <typename comparable_type_>
    const_iterator find(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this, node_t::find(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds and returns a copy of an element equal to @p comparable.
     *    Convenience method to avoid callback-based access in tests and simple use cases.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Result with copied element if found, or failure status.
     */
    template <typename comparable_type_>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        auto it = find(std::forward<comparable_type_>(comparable));
        if (it == end()) return key_not_found_k;
        return copy_safely(*it);
    }

    /**
     *  @brief Finds and returns a copy of the first element not less than (>=) @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Result with copied element if found, or failure status.
     */
    template <typename comparable_type_>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept {
        auto it = lower_bound(std::forward<comparable_type_>(comparable));
        if (it == end()) return key_not_found_k;
        else return copy_safely(*it);
    }

    /**
     *  @brief Finds and returns a copy of the first element greater than (>) @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Result with copied element if found, or failure status.
     */
    template <typename comparable_type_>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept {
        auto it = upper_bound(std::forward<comparable_type_>(comparable));
        if (it == end()) return key_not_found_k;
        else return copy_safely(*it);
    }

    /**
     *  @brief Checks if an element equal to @p comparable exists in the tree.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return True if element found, false otherwise.
     */
    template <typename comparable_type_>
    bool contains(comparable_type_ &&comparable) const noexcept {
        return node_t::find(root_, std::forward<comparable_type_>(comparable), comparator_) != nullptr;
    }

    /**
     *  @brief Finds the first element not less than (>=) the given @p comparable.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Iterator to found element, or end() if all elements are less.
     */
    template <typename comparable_type_>
    iterator lower_bound(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::lower_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element not less than (>=) the given @p comparable (const version).
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Const iterator to found element, or end() if all elements are less.
     */
    template <typename comparable_type_>
    const_iterator lower_bound(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this,
                              node_t::lower_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element greater than (>) the given @p comparable.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Iterator to found element, or end() if no element is greater.
     */
    template <typename comparable_type_>
    iterator upper_bound(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::upper_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element greater than (>) the given @p comparable (const version).
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Const iterator to found element, or end() if no element is greater.
     */
    template <typename comparable_type_>
    const_iterator upper_bound(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this,
                              node_t::upper_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Returns the number of elements with key equal to the specified argument.
     *    For unique-key containers like this, returns either 0 or 1.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to search key.
     *  @return Number of elements with key equal to @p comparable (0 or 1).
     */
    template <typename comparable_type_>
    std::size_t count(comparable_type_ &&comparable) const noexcept {
        return find(std::forward<comparable_type_>(comparable)) != end() ? 1 : 0;
    }

    /**
     *  @brief Checks if the tree has no elements.
     *  @return True if empty, false otherwise.
     */
    bool empty() const noexcept { return size_ == 0; }

    /**
     *  @brief Returns a range of elements matching a specific key.
     *    For unique-key containers, returns range containing at most one element.
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Pair of iterators [first, last) where all elements are equal to key.
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
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Pair of const iterators [first, last).
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
     *  @param[in] comparable Object comparable to @c value_t and convertible to search key.
     *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
     */
    template <typename comparable_type_ = value_t, typename callback_type_ = no_op_t>
    void equal_range(comparable_type_ &&comparable, callback_type_ &&callback) const noexcept {
        auto it = find(std::forward<comparable_type_>(comparable));
        if (it != end()) { callback(*it); }
    }

    /**
     *  @brief Iterates over all entries in the range [ @p lower, @p upper). Const version.
     *    Invokes callback for each element in the specified range.
     *
     *  @param[in] lower Lower bound of the range (inclusive).
     *  @param[in] upper Upper bound of the range (exclusive).
     *  @param[in] callback Callback invoked for each element in range. Must be @c noexcept.
     */
    template <typename lower_type_ = value_t, typename upper_type_ = value_t, typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        node_t::range(root_, std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), comparator_,
                      [&](node_t *node) noexcept { callback(node->fruit); });
    }

    /**
     *  @brief Iterates over all entries in the range [ @p lower, @p upper), allowing in-place modification.
     *    Invokes callback for each mutable element in the specified range. Non-const version.
     *
     *  @param[in] lower Lower bound of the range (inclusive).
     *  @param[in] upper Upper bound of the range (exclusive).
     *  @param[inout] callback Callback invoked for each mutable element in range. Must be @c noexcept.
     */
    template <typename lower_type_ = value_t, typename upper_type_ = value_t, typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        node_t::range(root_, std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), comparator_,
                      [&](node_t *node) noexcept { callback(node->fruit); });
    }

#pragma endregion Lookup

#pragma region Modifiers

    template <typename predicate_type_>
    std::size_t erase_if(predicate_type_ &&predicate) noexcept {
        auto result = node_t::erase_if(
            root_, std::forward<predicate_type_>(predicate),
            [&](node_t *node) noexcept {
                node->fruit.~value_t();
                allocator_.deallocate(node, 1);
            },
            comparator_);
        root_ = result.root;
        if (root_) root_->parent = nullptr;
        auto removed_count = size_ - result.count;
        size_ = result.count;
        return removed_count;
    }

    /**
     *  @brief Erases a range of elements [lower, upper).
     *    Invokes callback for each erased element.
     *    Complexity: O(log N + K), where K is the number of elements in the range.
     *
     *  @param[in] lower Lower bound of the range (inclusive).
     *  @param[in] upper Upper bound of the range (exclusive).
     *  @param[in] callback Optional callback invoked for each erased element.
     */
    template <typename lower_type_, typename upper_type_, typename callback_type_ = no_op_t>
    void erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) noexcept {
        if (!root_) return;

        auto split1 = node_t::split(root_, std::forward<lower_type_>(lower), comparator_);
        auto split2 = node_t::split(split1.right, std::forward<upper_type_>(upper), comparator_);

        std::size_t deleted_count = 0;
        node_t::for_each_bottom_up(split2.left, [&](node_t *n) noexcept {
            callback(n->fruit);
            n->fruit.~value_t();
            allocator_.deallocate(n, 1);
            deleted_count++;
        });

        root_ = node_t::join(split1.left, split2.right, comparator_);
        if (root_) root_->parent = nullptr;

        size_ -= deleted_count;
    }

    /**
     *  @brief The node an upsert settled on, and how it got there.
     *    Assigning to the result overwrites that node's entry, which is what makes it usable as a
     *    handle rather than a report.
     */
    struct upserted_node_t {
        node_t *node = nullptr;
        typename node_t::node_placement_t placement = node_t::node_placement_t::refused_k;

        /** @brief Whether nothing was stored, which is the only way an upsert fails. */
        bool failed() const noexcept { return placement == node_t::node_placement_t::refused_k; }
        explicit operator bool() const noexcept { return !failed(); }
        upserted_node_t &operator=(value_t &&fruit) noexcept {
            node->fruit = std::move(fruit);
            return *this;
        }
    };

    /**
     *  @brief Where an insertion settled, and how it got there.
     *    Names the same three outcomes as @c upserted_node_t, one level up from the nodes.
     */
    struct inserted_iterator_t {
        /** @brief The element's position, which is @c end() when nothing was stored. */
        iterator position;
        /** @brief Whether the entry was made, matched an incumbent, or refused for want of memory. */
        typename node_t::node_placement_t placement = node_t::node_placement_t::refused_k;

        /** @brief Whether nothing was stored, which is the only way an insertion fails. */
        bool failed() const noexcept { return placement == node_t::node_placement_t::refused_k; }
        explicit operator bool() const noexcept { return !failed(); }
    };

    /**
     *  @brief Inserts an entry only if the key doesn't exist. Matches @c std::set::insert() semantics.
     *    Leaves any incumbent alone, and leaves @p comparable alone unless a node was made for it.
     *
     *  @param[in] comparable Entry to insert (moved into the tree).
     *  @return The position the key lives at, and whether it was made, matched, or refused.
     */
    template <typename comparable_type_>
    inserted_iterator_t insert_if_missing(comparable_type_ &&comparable) noexcept {
        // First, try to find the key without allocating memory.
        auto existing = find(comparable);
        if (existing != end()) return {existing, node_t::node_placement_t::matched_k};

        // If not found, allocate a new node and then insert it.
        node_t *new_node = allocator_.allocate(1);
        if (!new_node) return {end(), node_t::node_placement_t::refused_k};

        new (&new_node->fruit) value_t(std::forward<comparable_type_>(comparable));
        auto result = node_t::insert(root_, new_node, comparator_);
        root_ = result.root;
        if (root_) root_->parent = nullptr;
        size_ += result.placement == node_t::node_placement_t::made_k;

        if (result.failed()) {
            // Insertion failed, deallocate the node.
            new_node->fruit.~value_t();
            allocator_.deallocate(new_node, 1);
            return {end(), node_t::node_placement_t::refused_k};
        }

        return {iterator(this, result.match), result.placement};
    }

    /**
     *  @brief Atomically inserts an entry only if the key doesn't exist. Fails with error if key exists.
     *    This is strict insert semantics - ensures key is new.
     *
     *  A refusal carries only its reason: @c key_already_exists_k when an incumbent holds the key,
     *  @c out_of_memory_heap_k when no node was available. Callers that want the incumbent's
     *  position ask @c insert_if_missing() instead, which hands back both.
     *
     *  @param[in] comparable Entry to insert, moved into the tree only when a node is made for it.
     *  @return Iterator to the new element, or the reason there is none.
     */
    template <typename comparable_type_>
    expected<iterator> insert(comparable_type_ &&comparable) noexcept {
        // Probing first keeps @p comparable intact when the key is already present, and keeps an
        // allocation that was never needed from reporting an out-of-memory that never happened.
        if (find(comparable) != end()) return status_t::key_already_exists_k;

        node_t *new_node = allocator_.allocate(1);
        if (!new_node) return status_t::out_of_memory_heap_k;

        new (&new_node->fruit) value_t(std::forward<comparable_type_>(comparable));

        auto result = node_t::insert(root_, new_node, comparator_);

        root_ = result.root;
        if (root_) root_->parent = nullptr;
        size_ += result.placement == node_t::node_placement_t::made_k;

        if (result.failed()) {
            new_node->fruit.~value_t();
            allocator_.deallocate(new_node, 1);
            return status_t::out_of_memory_heap_k;
        }

        // Only an inconsistent comparator can disagree with the probe above, and the node it
        // refused to take still has to be given back.
        if (result.placement == node_t::node_placement_t::matched_k) {
            new_node->fruit.~value_t();
            allocator_.deallocate(new_node, 1);
            return status_t::key_already_exists_k;
        }

        return iterator(this, result.match);
    }

    /**
     *  @brief Atomically inserts or updates an entry. Always succeeds (unless OOM).
     *    Overwrites existing entry if key exists (upsert semantics).
     *
     *  @param[in] entry Entry to upsert (moved into the tree).
     *  @return The node the entry now lives in, and whether it was made or matched.
     */
    template <typename comparable_type_>
    upserted_node_t upsert(comparable_type_ &&comparable) noexcept {
        auto existing = find(comparable);
        if (existing != end()) {
            *existing = std::forward<comparable_type_>(comparable);
            return {existing.node_, node_t::node_placement_t::matched_k};
        }

        node_t *new_node = allocator_.allocate(1);
        if (!new_node) return {nullptr, node_t::node_placement_t::refused_k};

        new (&new_node->fruit) value_t(std::forward<comparable_type_>(comparable));
        auto result = node_t::insert(root_, new_node, comparator_);
        root_ = result.root;
        if (root_) root_->parent = nullptr;
        size_ += result.placement == node_t::node_placement_t::made_k;

        if (result.failed()) {
            new_node->fruit.~value_t();
            allocator_.deallocate(new_node, 1);
            return {nullptr, node_t::node_placement_t::refused_k};
        }

        return {result.match, result.placement};
    }

    /**
     *  @brief Constructs an element in-place. Matches @c std::set::emplace() semantics.
     *         Does not insert if key already exists.
     *
     *  @tparam args_types_ Types of arguments to forward to value_t constructor.
     *  @param[in] args Arguments to forward to value_t constructor.
     *  @return Pair of iterator to inserted/existing element and bool indicating success.
     *
     *  @note The STL shape collapses a refused allocation and a matched incumbent into the same
     *    @c false, so @c insert_if_missing() is the call that tells them apart.
     */
    template <typename... args_types_>
    std::pair<iterator, bool> emplace(args_types_ &&...args) noexcept {
        auto result = insert_if_missing(value_t(std::forward<args_types_>(args)...));
        return {result.position, result.placement == node_t::node_placement_t::made_k};
    }

    /**
     *  @brief Deleted: Hint-based emplace is not supported.
     *    AVL trees don't benefit from position hints, and providing unused hints is misleading.
     *    Use @c emplace() instead.
     */
    template <typename... args_types_>
    iterator emplace_hint(const_iterator, args_types_ &&...) noexcept = delete;

    /**
     *  @brief Deleted: Hint-based insert is not supported.
     *    AVL trees don't benefit from position hints, and providing unused hints is misleading.
     *    Use @c insert_if_missing(value) instead.
     */
    iterator insert(const_iterator, value_t const &) noexcept = delete;

    /**
     *  @brief Deleted: Hint-based insert is not supported.
     *    AVL trees don't benefit from position hints, and providing unused hints is misleading.
     *    Use @c insert_if_missing(value) instead.
     */
    iterator insert(const_iterator, value_t &&) noexcept = delete;

    /**
     *  @brief Inserts a range of entries only if ALL keys are new (transactional all-or-nothing).
     *    Builds a temporary tree from the range, validates no conflicts, then merges atomically.
     *    On allocation failure or key conflict, this tree remains unchanged.
     *
     *  @tparam input_iterator_type_ Type of input iterator.
     *  @tparam tags_types_ Optional tag types:
     *    - @c assume_sorted_t : Range is sorted, enables O(n) bulk construction
     *  @param[in] first Beginning of range to insert.
     *  @param[in] last End of range to insert.
     *  @param[in] tags Optional tags to control insertion behavior.
     *  @return Success if all keys inserted, @c out_of_memory_heap_k on OOM,
     *    or @c operation_not_permitted_k if any key already exists.
     *
     *  @note Complexity:
     *    - With @c assume_sorted_t : O(n) build + O(m+n) validation + O(merge) time
     *    - Without: O(n log n) build + O(m+n) validation + O(merge) time
     *  @note With @c assume_sorted_t : Range must be sorted (ascending order).
     *  @note Transactional semantics: If ANY key exists, entire operation fails atomically.
     *    Temp tree is destroyed via RAII, this tree remains unchanged.
     */
    template <typename input_iterator_type_, typename... tags_types_>
    [[nodiscard]] status_t insert_if_missing(input_iterator_type_ first, input_iterator_type_ last,
                                             tags_types_...) noexcept {

        auto count = std::distance(first, last);
        if (count == 0) return success_k;

        // Build temporary tree, then merge atomically
        basic_avl_tree temp_tree(allocator_);

        // O(n): Build perfectly balanced tree from sorted range
        if constexpr (contains_type<assume_sorted_t, tags_types_...>()) {
            temp_tree.root_ =
                node_t::build_from_sorted(first, count, [&]() noexcept { return temp_tree.allocator_.allocate(1); });

            if (!temp_tree.root_) return status_t::out_of_memory_heap_k;

            // Verify complete allocation
            std::size_t actual_count = 0;
            node_t::for_each_left_right(temp_tree.root_, [&](node_t *) noexcept { ++actual_count; });
            if (actual_count != static_cast<std::size_t>(count)) return status_t::out_of_memory_heap_k;

            temp_tree.size_ = count;
        }
        // O(n log n): Build tree from unsorted range
        else {
            for (; first != last; ++first)
                if (temp_tree.insert_if_missing(value_t(*first)).failed()) return status_t::out_of_memory_heap_k;
        }

        // TRANSACTIONAL VALIDATION: Check if ANY key already exists - O(m+n)
        if (has_any_key(temp_tree))
            return status_t::key_already_exists_k; // Temp tree auto-destructs, this tree unchanged

        // All keys are new - safe to merge with assume_unique optimization
        merge(temp_tree, assume_unique_t {});
        return success_k;
    }

    /**
     *  @brief Inserts elements from an initializer list if keys don't exist.
     *    Builds a temporary tree from the list, then merges atomically.
     *    On allocation failure during tree construction, this tree is unchanged.
     *
     *  @param[in] ilist Initializer list of entries to insert.
     *  @return First error encountered, or success if all elements inserted.
     */
    [[nodiscard]] status_t insert_if_missing(std::initializer_list<value_t> ilist) noexcept {
        return insert_if_missing(ilist.begin(), ilist.end());
    }

    /**
     *  @brief Upserts a range of entries (inserts new keys or updates existing ones).
     *    Builds a temporary tree from the range, then merges with upsert semantics.
     *    On allocation failure during tree construction, this tree remains unchanged.
     *
     *  @tparam input_iterator_type_ Type of input iterator.
     *  @tparam tags_types_ Optional tag types:
     *    - @c assume_sorted_t : Range is sorted, enables O(n) bulk construction
     *    - @c assume_unique_t : No duplicate keys with existing tree, enables optimized merge
     *  @param[in] first Beginning of range to upsert.
     *  @param[in] last End of range to upsert.
     *  @param[in] tags Optional tags to control insertion behavior.
     *  @return Success if all elements processed, or @c out_of_memory_heap_k on OOM.
     *
     *  @note Complexity:
     *    - With @c assume_sorted_t : O(n) build + O(merge) time
     *    - Without: O(n log n) build + O(merge) time
     *    - With @c assume_unique_t : O(log n) to O(m+n) optimized merge (assumes no conflicts)
     *    - Without @c assume_unique_t : O(m log n) upsert merge (updates duplicates)
     *  @note With @c assume_sorted_t : Range must be sorted (ascending order).
     *  @note With @c assume_unique_t : Range must have no duplicate keys with existing tree (UB if violated).
     *  @note Atomicity: every failure happens while the staging tree is built, leaving this tree
     *    unchanged; the merge itself only relinks nodes, so it cannot fail part-way.
     *    Duplicate keys are UPDATED during the merge, not skipped.
     */
    template <typename input_iterator_type_, typename... tags_types_>
    [[nodiscard]] status_t upsert(input_iterator_type_ first, input_iterator_type_ last, tags_types_...) noexcept {

        auto count = std::distance(first, last);
        if (count == 0) return success_k;

        // Build temporary tree from range
        basic_avl_tree temp_tree(allocator_);

        // O(n): Build perfectly balanced tree from sorted range
        if constexpr (contains_type<assume_sorted_t, tags_types_...>()) {
            temp_tree.root_ =
                node_t::build_from_sorted(first, count, [&]() noexcept { return temp_tree.allocator_.allocate(1); });

            if (!temp_tree.root_) return status_t::out_of_memory_heap_k;

            // Verify complete allocation (detect partial tree from mid-construction failure)
            std::size_t actual_count = 0;
            node_t::for_each_left_right(temp_tree.root_, [&](node_t *) noexcept { ++actual_count; });
            if (actual_count != static_cast<std::size_t>(count)) return status_t::out_of_memory_heap_k;

            temp_tree.size_ = count;
        }
        // O(n log n): Build tree from unsorted range
        else {
            for (; first != last; ++first)
                if (temp_tree.insert_if_missing(value_t(*first)).failed()) return status_t::out_of_memory_heap_k;
        }

        // Choose merge strategy based on uniqueness guarantee
        if constexpr (contains_type<assume_unique_t, tags_types_...>()) {
            // Optimized merge: O(log n) to O(m+n), assumes no duplicates (UB if violated)
            merge(temp_tree, assume_unique_t {});
        }
        else {
            // Upsert merge: O(m log n), updates duplicates instead of skipping
            merge_with_upsert(temp_tree);
        }

        return success_k;
    }

    /**
     *  @brief Upserts elements from initializer list (insert or update semantics).
     *
     *  @param[in] ilist Initializer list of entries to upsert.
     *  @return First error encountered, or success if all elements processed.
     */
    [[nodiscard]] status_t upsert(std::initializer_list<value_t> ilist) noexcept {
        return upsert(ilist.begin(), ilist.end());
    }

    /**
     *  @brief Updates a range of existing entries (transactional all-or-nothing).
     *    Builds a temporary tree from the range, validates all keys exist, then updates atomically.
     *    On allocation failure or any missing key, this tree remains unchanged.
     *
     *  @tparam input_iterator_type_ Type of input iterator.
     *  @tparam tags_types_ Optional tag types:
     *    - @c assume_sorted_t : Range is sorted, enables O(n) bulk construction
     *  @param[in] first Beginning of range to update.
     *  @param[in] last End of range to update.
     *  @param[in] tags Optional tags to control insertion behavior.
     *  @return Success if all keys updated, @c out_of_memory_heap_k on OOM,
     *    or @c key_not_found_k if any key doesn't exist.
     *
     *  @note Complexity:
     *    - With @c assume_sorted_t : O(n) build + O(m+n) validation + O(m log n) update
     *    - Without: O(n log n) build + O(m+n) validation + O(m log n) update
     *  @note With @c assume_sorted_t : Range must be sorted (ascending order).
     *  @note Transactional semantics: If ANY key missing, entire operation fails atomically.
     *    Temp tree is destroyed via RAII, this tree remains unchanged.
     */
    template <typename input_iterator_type_, typename... tags_types_>
    [[nodiscard]] status_t update(input_iterator_type_ first, input_iterator_type_ last, tags_types_...) noexcept {

        auto count = std::distance(first, last);
        if (count == 0) return success_k;

        // Build temporary tree from range
        basic_avl_tree temp_tree(allocator_);

        // O(n): Build perfectly balanced tree from sorted range
        if constexpr (contains_type<assume_sorted_t, tags_types_...>()) {
            temp_tree.root_ =
                node_t::build_from_sorted(first, count, [&]() noexcept { return temp_tree.allocator_.allocate(1); });

            if (!temp_tree.root_) return status_t::out_of_memory_heap_k;

            // Verify complete allocation (detect partial tree from mid-construction failure)
            std::size_t actual_count = 0;
            node_t::for_each_left_right(temp_tree.root_, [&](node_t *) noexcept { ++actual_count; });
            if (actual_count != static_cast<std::size_t>(count)) return status_t::out_of_memory_heap_k;

            temp_tree.size_ = count;
        }
        // O(n log n): Build tree from unsorted range
        else {
            for (; first != last; ++first)
                if (temp_tree.insert_if_missing(value_t(*first)).failed()) return status_t::out_of_memory_heap_k;
        }

        // TRANSACTIONAL VALIDATION: Check if ALL keys exist - O(m+n)
        if (!has_all_keys(temp_tree)) return status_t::key_not_found_k; // Temp tree auto-destructs, this tree unchanged

        // All keys exist - safe to upsert (will only update, never insert)
        merge_with_upsert(temp_tree);
        return success_k;
    }

    /**
     *  @brief Updates elements from an initializer list (all keys must exist).
     *
     *  @param[in] ilist Initializer list of entries to update.
     *  @return Success if all keys updated, or error if any key missing or OOM.
     */
    [[nodiscard]] status_t update(std::initializer_list<value_t> ilist) noexcept {
        return update(ilist.begin(), ilist.end());
    }

#pragma endregion Modifiers

#pragma region Observers

    /**
     *  @brief Returns the function object that compares keys.
     *  @return The comparison function object.
     */
    comparator_t key_comp() const noexcept { return comparator_; }

    /**
     *  @brief Returns the function object that compares values.
     *         For sets, this is the same as key_comp().
     *  @return The comparison function object.
     */
    comparator_t value_comp() const noexcept { return comparator_; }

    /**
     *  @brief Returns the maximum possible number of elements.
     *  @return Theoretical maximum size.
     */
    std::size_t max_size() const noexcept {
        return smaller_of(allocator_.max_size(), std::numeric_limits<std::size_t>::max() / sizeof(node_t));
    }

    /**
     *  @brief Exchanges the contents of this tree with another.
     *
     *  Swaps tree structure, size, comparator, and (if propagating) allocators.
     *  For non-propagating allocators (when @c propagate_on_container_swap is @c false),
     *  the allocators must be equal, otherwise @c invalid_argument_k is returned.
     *
     *  @param[inout] other Tree to swap with.
     *  @return @c success_k when the swap completed, or @c invalid_argument_k
     *    if allocators are incompatible and non-propagating.
     */
    [[nodiscard]] status_t swap(basic_avl_tree &other) noexcept {
        // For non-propagating allocators, they must be equal (C++ standard requirement)
        if constexpr (!std::allocator_traits<allocator_t>::propagate_on_container_swap::value)
            if (!(allocator_ == other.allocator_)) return invalid_argument_k;

        std::swap(root_, other.root_);
        std::swap(size_, other.size_);
        std::swap(comparator_, other.comparator_);

        // Only swap allocators if they propagate on swap
        if constexpr (std::allocator_traits<allocator_t>::propagate_on_container_swap::value)
            std::swap(allocator_, other.allocator_);

        return success_k;
    }

    struct extract_result_t {
        basic_avl_tree *tree_ = nullptr;
        node_t *node_ptr_ = nullptr;

        extract_result_t() = default;
        extract_result_t(basic_avl_tree *tree, node_t *node) noexcept : tree_(tree), node_ptr_(node) {}

        /** @brief Destroys the entry before releasing its node, as every other free site here does. */
        void discard_() noexcept {
            if (!node_ptr_) return;
            node_ptr_->fruit.~value_t();
            tree_->allocator_.deallocate(node_ptr_, 1);
            node_ptr_ = nullptr;
        }

        ~extract_result_t() noexcept { discard_(); }
        extract_result_t(extract_result_t const &) = delete;
        extract_result_t &operator=(extract_result_t const &) = delete;
        extract_result_t(extract_result_t &&other) noexcept
            : tree_(other.tree_), node_ptr_(std::exchange(other.node_ptr_, nullptr)) {}
        extract_result_t &operator=(extract_result_t &&other) noexcept {
            if (this != &other) {
                discard_();
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
        auto result = node_t::extract(root_, std::forward<comparable_type_>(comparable), comparator_);
        root_ = result.root;
        size_ -= result.extracted != nullptr;
        return extract_result_t {this, result.extracted.release()};
    }

    /**
     *  @brief Erases a single entry matching the given @p comparable.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to search key.
     *  @param[in] callback_found Callback to receive the erased entry. Must be @c noexcept.
     *  @param[in] callback_missing Callback triggered if nothing was found. Must be @c noexcept.
     */
    template <typename comparable_type_ = value_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    void erase(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
               callback_missing_type_ &&callback_missing) noexcept {
        auto position = find(comparable);
        if (position == end()) {
            callback_missing();
            return;
        }

        callback_found(*position);
        // The guard destroys the entry and returns the node to the allocator it came from.
        [[maybe_unused]] extract_result_t const erased = extract(std::forward<comparable_type_>(comparable));
    }

    /**
     *  @brief Erases a single entry matching the given @p comparable. No callbacks.
     *
     *  @param[in] comparable Object comparable to @c value_t and convertible to search key.
     *  @return True if element was erased, false if not found.
     */
    template <typename comparable_type_>
    bool erase(comparable_type_ &&comparable) noexcept {
        return !!extract(std::forward<comparable_type_>(comparable));
    }

    /**
     *  @brief Erases the element at the specified iterator position.
     *    Unlike STL, returns both the next iterator and a status code for error reporting.
     *
     *  @param[in] position Iterator to element to erase. Must be valid and dereferenceable.
     *  @return Contains iterator to element following the erased element and operation status.
     *
     *  @note If @p position is end(), returns {end(), success} without modifying the tree.
     *    If erase fails (e.g., tree corruption), returns {end(), error}.
     */
    erase_result_t erase(iterator position) noexcept {
        if (position == end()) return {end(), {success_k}};
        auto next = std::next(position);
        bool erased = erase(*position);
        return {next, erased ? success_k : status_t::unknown_k};
    }

    /**
     *  @brief Erases the element at the specified const_iterator position.
     *    Unlike STL, returns both the next iterator and a status code for error reporting.
     *
     *  @param[in] position Const iterator to element to erase. Must be valid and dereferenceable.
     *  @return Contains iterator to element following the erased element and operation status.
     *
     *  @note If @p position is end(), returns {end(), success} without modifying the tree.
     *    If erase fails (e.g., tree corruption), returns {end(), error}.
     */
    erase_result_t erase(const_iterator position) noexcept {
        // Erasing through a const iterator is the STL contract: the position is const, this tree is not.
        return erase(iterator(this, const_cast<node_t *>(position.node_)));
    }

    /**
     *  @brief Erases all elements in the range [first, last).
     *    Unlike STL, returns both the iterator following the last erased element and a status code.
     *    On error, some elements may have been erased (partial erase, matches STL's basic guarantee).
     *
     *  @param[in] first Beginning of range to erase.
     *  @param[in] last End of range to erase (not erased).
     *  @return Contains iterator equal to @p last and status of the operation.
     *    Returns first error encountered, or success if all elements erased.
     */
    erase_result_t erase(iterator first, iterator last) noexcept {
        while (first != last) {
            auto result = erase(first);
            if (failed(result.status)) return {result.next, result.status};
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
     *  @return Contains iterator equal to @p last and status of the operation.
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
     *  @return Always succeeds.
     */
    void clear() noexcept {
        node_t::for_each_bottom_up(root_, [&](node_t *node) noexcept {
            node->fruit.~value_t();
            allocator_.deallocate(node, 1);
        });
        root_ = nullptr;
        size_ = 0;
    }

    template <typename callback_type_>
    void for_each(callback_type_ &&callback) noexcept {
        node_t::for_each_left_right(root_, [&](node_t *node) noexcept { callback(node->fruit); });
    }

#pragma endregion Observers

#pragma region Merge Operations

    /**
     *  @brief Merges another tree into this one, transferring all nodes.
     *    Elements with keys that already exist in this tree are deallocated (not kept in source).
     *
     *  @param[inout] other Tree to merge from. Will be empty after merge.
     *
     *  @note Unlike @c std::set::merge(), nodes with duplicate keys are deallocated rather than
     *    remaining in the source container. This ensures no memory leaks in a noexcept context.
     *  @note Complexity: O(m log n) where m = other.size(), n = this.size().
     *    For disjoint trees, use @c merge(other, assume_unique) for faster O(m log(n/m+1)) or O(m+n).
     */
    void merge(basic_avl_tree &other) noexcept {
        node_t::for_each_bottom_up(other.root_, [&](node_t *node) noexcept {
            auto result = node_t::insert(root_, node, comparator_);
            root_ = result.root;
            size_ += result.placement == node_t::node_placement_t::made_k;
            // Key conflict - node wasn't inserted, so release the entry it carried
            if (result.placement == node_t::node_placement_t::matched_k) {
                node->fruit.~value_t();
                allocator_.deallocate(node, 1);
            }
        });
        other.root_ = nullptr;
        other.size_ = 0;
    }

    /**
     *  @brief Merges another tree into this one with optimized algorithm for disjoint trees.
     *    Precondition: Trees have no overlapping keys (disjoint).
     *
     *  @param[inout] other Tree to merge from. Will be empty after merge.
     *  @param[in] assume_unique Tag indicating trees are disjoint (no duplicate keys).
     *
     *  @note Complexity: O(m log(n/m + 1)) for unbalanced sizes, O(m+n) for similar sizes.
     *    Automatically selects optimal algorithm:
     *    - If all(this) < all(other): O(log n) join
     *    - Large similar sizes: O(m+n) Day-Stout-Warren @b (DSW) spine merge
     *    - Unbalanced sizes: O(m log(n/m+1)) split-based merge, optimal according to Blelloch
     *  @warning If precondition violated (duplicate keys exist), behavior is undefined.
     */
    void merge(basic_avl_tree &other, assume_unique_t) noexcept {
        if (other.empty()) return;
        if (empty()) {
            // Move other into this
            root_ = other.root_;
            size_ = other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        // Strategy 1: Check if fully ordered - O(log n) check, O(log n) join
        auto this_max = node_t::find_max(root_);
        auto this_min = node_t::find_min(root_);
        auto other_max = node_t::find_max(other.root_);
        auto other_min = node_t::find_min(other.root_);

        // Fast path: all(this) < all(other), use join: O(log n)
        if (comparator_(mapping_key_or_itself(this_max->fruit), mapping_key_or_itself(other_min->fruit))) {
            root_ = node_t::join(root_, other.root_, comparator_);
            size_ += other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        // Fast path: all(other) < all(this), use join: O(log n)
        if (comparator_(mapping_key_or_itself(other_max->fruit), mapping_key_or_itself(this_min->fruit))) {
            root_ = node_t::join(other.root_, root_, comparator_);
            size_ += other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        // Strategy 2: Adaptive selection between split-based and DSW
        std::size_t min_size = smaller_of(size_, other.size_);
        std::size_t max_size = larger_of(size_, other.size_);

        // Heuristic: Use DSW when both large and similar size
        constexpr std::size_t dsw_threshold_k = 10000;
        constexpr std::size_t size_ratio_threshold_k = 3;

        // DSW for large similarly-sized trees: O(m+n)
        if (min_size > dsw_threshold_k && max_size < min_size * size_ratio_threshold_k) {
            std::size_t new_size;
            root_ = node_t::merge_dsw(root_, other.root_, comparator_, new_size);
            size_ = new_size;
        }
        // Split-based for unbalanced sizes: O(m log(n/m+1))
        else {
            node_t *small = (size_ < other.size_) ? root_ : other.root_;
            node_t *large = (size_ < other.size_) ? other.root_ : root_;
            root_ = node_t::merge_split_based(small, large, comparator_);
            size_ += other.size_;
        }

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
        auto result = node_t::insert(root_, node_to_insert, comparator_);
        root_ = result.root;
        size_ += result.placement == node_t::node_placement_t::made_k;
        // Key conflict - node wasn't inserted, so release the entry it carried
        if (result.placement == node_t::node_placement_t::matched_k) {
            node_to_insert->fruit.~value_t();
            allocator_.deallocate(node_to_insert, 1);
        }
    }

#pragma endregion Merge Operations

#pragma region Split and Join

    /**
     *  @brief Result of a split operation on a tree.
     */
    struct split_result_t {
        /** @brief Tree with all elements ordered before the split key. */
        basic_avl_tree left;
        /** @brief Tree with all elements not ordered before the split key. */
        basic_avl_tree right;
    };

    /**
     *  @brief Splits the tree at a given key into two trees.
     *    Elements < key go to left tree, elements >= key go to right tree.
     *    This tree becomes empty after the split.
     *
     *  @param[in] comparable Key to split at.
     *  @return Contains left tree (< key) and right tree (>= key).
     *
     *  @note This operation is O(log n) and maintains AVL balance in both resulting trees.
     *    The current tree is emptied (moved-from state).
     */
    template <typename comparable_type_>
    split_result_t split(comparable_type_ &&comparable) noexcept {
        auto node_split_result = node_t::split(root_, std::forward<comparable_type_>(comparable), comparator_);

        // Count elements in each tree (need to traverse to get accurate size)
        std::size_t left_size = 0, right_size = 0;
        node_t::for_each_left_right(node_split_result.left, [&](node_t *) noexcept { ++left_size; });
        node_t::for_each_left_right(node_split_result.right, [&](node_t *) noexcept { ++right_size; });

        // Create result trees
        split_result_t result;
        result.left.root_ = node_split_result.left;
        result.left.size_ = left_size;
        result.left.allocator_ = allocator_;

        result.right.root_ = node_split_result.right;
        result.right.size_ = right_size;
        result.right.allocator_ = allocator_;

        // Empty current tree
        root_ = nullptr;
        size_ = 0;

        return result;
    }

    /**
     *  @brief Joins this tree with another tree.
     *    Precondition: All elements in this tree < all elements in other tree.
     *    The other tree becomes empty after the join.
     *
     *  @param[inout] other Tree to join with (must have all larger elements).
     *
     *  @note This operation is O(log n) and maintains AVL balance.
     *    If the precondition is violated, the resulting tree structure is undefined.
     */
    void join(basic_avl_tree &other) noexcept {
        root_ = node_t::join(root_, other.root_, comparator_);
        size_ += other.size_;
        other.root_ = nullptr;
        other.size_ = 0;
    }
};

template <typename value_type_, typename comparator_type_ = less_t, typename allocator_type_ = std::allocator<void>>
using avl_set = basic_avl_tree<value_type_, comparator_type_, allocator_type_>;

template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<void>>
using avl_map = basic_avl_tree<mapping<key_type_, value_type_>, comparator_type_, allocator_type_>;

#pragma endregion Split and Join

} // namespace ashvardanian::smashtable
