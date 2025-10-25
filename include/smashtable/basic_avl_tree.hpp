/**
 *  @brief  Ordered "Adelson-Velsky and Landis" @b AVL Binary Search Tree implementation.
 *    Provides exception-free, allocator-aware ordered collection similar to @c std::set.
 *    Not thread-safe by itself. Doesn't raise any exceptions unlike STL-based alternatives.
 *
 *  @section Design Characteristics
 *
 *  AVL trees maintain strict balance (height difference ≤ 1), providing O(log n) worst-case lookups, insertions,
 *  and deletions. Compared to Red-Black trees (used in @c std::set), AVL trees are more rigidly balanced, offering
 *  faster lookups at the cost of slightly slower insertions and deletions due to more frequent rebalancing.
 *
 *  This implementation supports heterogeneous lookups (searching with types other than @c entry_type_), custom
 *  allocators for all internal nodes, and callback-based iteration to avoid iterator invalidation complexity.
 *  All methods are @c noexcept and use status codes instead of exceptions for error handling.
 *
 *  @section STL Interface Compatibility
 *
 *  Provides standard @c std::set interface:
 *  - @b Lookup: @c find, @c lower_bound, @c upper_bound, @c equal_range, @c contains, @c count
 *  - @b Modification: @c insert, @c emplace, @c erase, @c extract, @c clear, @c swap
 *  - @b Iteration: Bidirectional iterators (@c begin, @c end), range-based for loops, callback-based traversal
 *  - @b Capacity: @c size, @c empty, @c max_size
 *  - @b Observers: @c key_comp, @c value_comp
 *  - @b Bulk Operations: Range @c insert, @c merge
 *
 *  @section Advanced Operations
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
 *  @section Requirements
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
 *  @file basic_avl_tree.hpp
 *  @date October 25, 2025
 *  @author Ash Vardanian
 *  @see https://en.wikipedia.org/wiki/AVL_tree
 */
#pragma once
#include <cassert>   // `assert`
#include <algorithm> // `std::max`
#include <memory>    // `std::allocator`
#include <random>    // `std::uniform_int_distribution`
#include <utility>   // `std::exchange`

#include "shared.hpp"

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
 *  @tparam entry_type_ Type of entries to store in this tree.
 *    Those must be @c noexcept move-constructible.
 *
 *  @tparam comparator_type_ A comparator function object that overloads
 *    @code bool operator()(entry_type_ const &, entry_type_ const &) const @endcode.
 *    For heterogeneous lookups, define @code using is_transparent = void; @endcode inside the comparator.
 */
template <typename entry_type_, typename comparator_type_>
class basic_avl_node {
  public:
    using entry_t = entry_type_;
    using comparator_t = comparator_type_;
    using height_t = std::ptrdiff_t;
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

#pragma mark - Traversal and Search

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

    /**
     *  @brief Searches for equal entry in this subtree.
     *  @param node Root of subtree to search.
     *  @param comparable Any key comparable with stored entries.
     *  @param comparator Comparator instance (may be stateful).
     *  @return NULL if nothing was found.
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
     *  @brief Find the smallest entry, bigger than or equal to the provided one.
     *  @param node Root of subtree to search.
     *  @param comparable Any key comparable with stored entries.
     *  @param comparator Comparator instance (may be stateful).
     *  @return NULL if nothing was found.
     */
    template <typename comparable_type_>
    static node_t *lower_bound(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        node_t *successor = nullptr;
        while (node) {
            // If the given key is less than the root node, visit the left
            // subtree, taking current node as potential successor.
            if (comparator(comparable, node->entry)) {
                successor = node;
                node = node->left;
            }

            // Of the given key is more than the root node, visit the right
            // subtree.
            else if (comparator(node->entry, comparable)) { node = node->right; }

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
     *  @param node Root of subtree to search.
     *  @param comparable Any key comparable with stored entries.
     *  @param comparator Comparator instance (may be stateful).
     *  @return NULL if nothing was found.
     *
     *  Is used for an atomic implementation of iterators.
     *  Alternatively one can:
     *  > store a stack for path, which is ~O(logN) space.
     *  > store parents in nodes and have complex logic.
     */
    template <typename comparable_type_>
    static node_t *upper_bound(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        node_t *successor = nullptr;
        while (node) {
            // If the given key is less than the root node, visit the left
            // subtree, taking current node as potential successor.
            if (comparator(comparable, node->entry)) {
                successor = node;
                node = node->left;
            }

            // Of the given key is more than the root node, visit the right
            // subtree.
            else if (comparator(node->entry, comparable)) node = node->right;

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
    static node_t *lowest_common_ancestor(node_t *node, comparable_a_type_ &&a, comparable_b_type_ &&b,
                                          comparator_t const &comparator) noexcept {
        if (!node) return nullptr;

        // If both `a` and `b` are smaller than `node`, then LCA lies in left
        if (comparator(a, node->entry) && comparator(b, node->entry))
            return lowest_common_ancestor(node->left, a, b, comparator);

        // If both `a` and `b` are greater than `node`, then LCA lies in right
        if (comparator(node->entry, a) && comparator(node->entry, b))
            return lowest_common_ancestor(node->right, a, b, comparator);

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
     *  @param node Root of subtree to search.
     *  @param low Lower bound of range.
     *  @param high Upper bound of range.
     *  @param comparator Comparator instance (may be stateful).
     *  @param callback Function to call for each node in range.
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

    template <typename comparable_type_>
    static node_interval_t equal_range(node_t *node, comparable_type_ &&comparable,
                                       comparator_t const &comparator) noexcept {
        auto no_op = [](node_t *) noexcept {};
        return range(node, comparable, comparable, comparator, no_op);
    }

#pragma mark - Sampling

    /**
     *  @brief Random samples a single node using reservoir sampling for uniform distribution.
     *  @param generator Any STL-compatible random number generator.
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
            std::uniform_int_distribution<std::size_t> distribution {0, count - 1};
            if (distribution(generator) == 0) result = current;
        });

        return result;
    }

    /**
     *  @brief Random samples nodes within a given range of keys using reservoir sampling.
     *  @param node Root of subtree.
     *  @param low Lower bound.
     *  @param high Upper bound.
     *  @param comparator Comparator instance (may be stateful).
     *  @param generator Any STL-compatible random number generator.
     *  @param predicate Predicate to filter nodes.
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
            std::uniform_int_distribution<std::size_t> distribution {0, count - 1};
            if (distribution(generator) == 0) result = node;
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
                std::uniform_int_distribution<std::size_t> distribution {0, seen};
                auto slot_to_replace = distribution(generator);
                if (slot_to_replace < reservoir_capacity) reservoir[slot_to_replace] = node;
            }

            ++seen;
        });
    }

#pragma mark - Rotations and Balancing

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

    template <typename comparable_type_>
    inline static node_t *rebalance_after_insert(node_t *node, comparable_type_ &&comparable,
                                                 comparator_t const &comparator) noexcept {
        // Update height and check if branches aren't balanced
        node->height = std::max(get_height(node->left), get_height(node->right)) + 1;
        auto balance = get_balance(node);

        // Left Left Case
        if (balance > 1 && comparator(comparable, node->left->entry)) return rotate_right(node);

        // Right Right Case
        else if (balance < -1 && comparator(node->right->entry, comparable)) return rotate_left(node);

        // Left Right Case
        else if (balance > 1 && comparator(node->left->entry, comparable)) {
            node->left = rotate_left(node->left);
            return rotate_right(node);
        }
        // Right Left Case
        else if (balance < -1 && comparator(comparable, node->right->entry)) {
            node->right = rotate_right(node->right);
            return rotate_left(node);
        }
        else return node;
    }

    template <typename comparable_type_, typename callback_found_type_, typename callback_make_type_>
    static find_or_make_result_t find_or_make(node_t *node, comparable_type_ &&comparable,
                                              comparator_t const &comparator, callback_found_type_ &&callback_found,
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

        if (comparator(comparable, node->entry)) {
            auto downstream = find_or_make(node->left, comparable, comparator, callback_found, callback_make);
            node->left = downstream.root;
            if (downstream.inserted) node = rebalance_after_insert(node, downstream.match->entry, comparator);
            return {node, downstream.match, downstream.inserted};
        }
        else if (comparator(node->entry, comparable)) {
            auto downstream = find_or_make(node->right, comparable, comparator, callback_found, callback_make);
            node->right = downstream.root;
            if (downstream.inserted) node = rebalance_after_insert(node, downstream.match->entry, comparator);
            return {node, downstream.match, downstream.inserted};
        }
        else {
            // Equal keys are not allowed in BST
            callback_found(node);
            return {node, node, false};
        }
    }

    template <typename allocator_type_>
    static find_or_make_result_t insert(node_t *node, entry_t &&entry, comparator_t const &comparator,
                                        allocator_type_ &&node_allocator) noexcept {
        auto found = [&](node_t *node) noexcept {};
        auto make = [&]() noexcept -> node_t * {
            auto node = node_allocator();
            if (node) new (&node->entry) entry_t(std::move(entry));
            return node;
        };
        auto result = find_or_make(node, entry, comparator, found, make);
        return result;
    }

    template <typename allocator_type_>
    static find_or_make_result_t upsert(node_t *node, entry_t &&entry, comparator_t const &comparator,
                                        allocator_type_ &&node_allocator) noexcept {
        auto found = [&](node_t *node) noexcept { node->entry = std::move(entry); };
        auto make = [&]() noexcept -> node_t * {
            auto node = node_allocator();
            if (node) new (&node->entry) entry_t(std::move(entry));
            return node;
        };
        auto result = find_or_make(node, entry, comparator, found, make);
        return result;
    }

    static find_or_make_result_t insert(node_t *node, node_t *new_child, comparator_t const &comparator) noexcept {
        return find_or_make(
            node, new_child->entry, comparator, [](node_t *) noexcept {}, [=]() noexcept { return new_child; });
    }

    /**
     *  @brief Builds a balanced AVL tree from sorted range in O(n) time.
     *    Precondition: Range [first, first+count) must be sorted according to comparator.
     *
     *  @param[in] first Iterator to beginning of sorted range.
     *  @param[in] count Number of elements in range.
     *  @param[in] alloc Allocator function that returns new node pointer or nullptr on failure.
     *  @return node_t* Root of balanced tree, or nullptr if allocation failed.
     *
     *  @note Complexity: O(n) time, O(log n) recursion depth.
     *    Builds perfectly balanced tree by recursively picking middle element as root.
     *  @warning If precondition violated (unsorted input), resulting tree has undefined structure.
     */
    template <typename iterator_type_, typename allocator_func_>
    static node_t *build_from_sorted(iterator_type_ first, std::size_t count, allocator_func_ &&alloc) noexcept {
        if (count == 0) return nullptr;

        // Find middle element
        std::size_t mid = count / 2;
        auto mid_iter = first;
        std::advance(mid_iter, mid);

        // Allocate root node
        node_t *root = alloc();
        if (!root) return nullptr;
        new (&root->entry) entry_t(*mid_iter);

        // Build left subtree from [first, mid)
        root->left = build_from_sorted(first, mid, alloc);

        // Build right subtree from (mid, last)
        auto right_first = mid_iter;
        ++right_first;
        root->right = build_from_sorted(right_first, count - mid - 1, alloc);

        // Set height (no balancing needed for perfectly balanced construction)
        root->height = 1 + std::max(get_height(root->left), get_height(root->right));

        return root;
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

    struct remove_if_result_t {
        node_t *root = nullptr;
        std::size_t count = 0;
    };

    template <typename predicate_type_, typename node_deallocator_type_>
    static remove_if_result_t remove_if(node_t *node, predicate_type_ &&predicate,
                                        node_deallocator_type_ &&node_deallocator) noexcept {
        return {};
    }

#pragma mark - Split and Join

    /**
     *  @brief Result of a split operation.
     */
    struct split_result_t {
        node_t *left = nullptr;  //!< Tree with all elements < key.
        node_t *right = nullptr; //!< Tree with all elements >= key.
    };

    /**
     *  @brief Joins two trees with a root node between them.
     *    Precondition: All elements in left < root < all elements in right.
     *    Maintains AVL balance property.
     *
     *  @param[in] left Left subtree (all elements < root).
     *  @param[in] root_node Root node to insert between subtrees.
     *  @param[in] right Right subtree (all elements > root).
     *  @return node_t* Root of the joined tree.
     */
    static node_t *join_with_root(node_t *left, node_t *root_node, node_t *right,
                                  comparator_t const &comparator) noexcept {
        if (!root_node) return join(left, right, comparator);

        root_node->left = left;
        root_node->right = right;
        root_node->height = 1 + std::max(get_height(left), get_height(right));

        // Rebalance if necessary
        auto balance = get_balance(root_node);

        // Left Left Case
        if (balance > 1 && get_balance(root_node->left) >= 0) return rotate_right(root_node);

        // Left Right Case
        else if (balance > 1 && get_balance(root_node->left) < 0) {
            root_node->left = rotate_left(root_node->left);
            return rotate_right(root_node);
        }

        // Right Right Case
        else if (balance < -1 && get_balance(root_node->right) <= 0) return rotate_left(root_node);

        // Right Left Case
        else if (balance < -1 && get_balance(root_node->right) > 0) {
            root_node->right = rotate_right(root_node->right);
            return rotate_left(root_node);
        }

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
     *  @return node_t* Root of the joined tree.
     */
    static node_t *join(node_t *left, node_t *right, comparator_t const &comparator) noexcept {
        if (!left) return right;
        if (!right) return left;

        auto left_height = get_height(left);
        auto right_height = get_height(right);

        // If left tree is taller, join with right subtree of left
        if (left_height > right_height + 1) {
            left->right = join(left->right, right, comparator);
            left->height = 1 + std::max(get_height(left->left), get_height(left->right));
            return rebalance_after_extract(left);
        }
        // If right tree is taller, join with left subtree of right
        else if (right_height > left_height + 1) {
            right->left = join(left, right->left, comparator);
            right->height = 1 + std::max(get_height(right->left), get_height(right->right));
            return rebalance_after_extract(right);
        }
        // Heights are balanced, extract min from right and use as root
        else {
            node_t *min_right = find_min(right);
            auto extract_result = extract(right, min_right->entry, comparator);
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
     *  @return split_result_t Contains left tree (< key) and right tree (>= key).
     */
    template <typename comparable_type_>
    static split_result_t split(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        if (!node) return {nullptr, nullptr};

        // If node < key, put node in left tree and split right subtree
        if (comparator(node->entry, comparable)) {
            auto downstream = split(node->right, comparable, comparator);
            return {join_with_root(node->left, node, downstream.left, comparator), downstream.right};
        }
        // If key <= node, put node in right tree and split left subtree
        else {
            auto downstream = split(node->left, comparable, comparator);
            return {downstream.left, join_with_root(downstream.right, node, node->right, comparator)};
        }
    }

#pragma mark - Merge Algorithms

    /**
     *  @brief Merges two AVL trees using split-based divide-and-conquer.
     *    Optimal for unbalanced sizes: O(m log(n/m + 1)) where m <= n.
     *    Takes root of smaller tree, splits larger tree around it, recursively merges.
     *
     *  @param[in] small Smaller tree to merge (will be consumed).
     *  @param[in] large Larger tree to merge into (will be consumed).
     *  @return node_t* Root of merged tree with all nodes from both inputs.
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
        auto split_result = split(large, pivot->entry, comparator);

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
     *  @return node_t* Head of spine (smallest element).
     *
     *  @note Uses rotations to flatten tree: O(n) time, O(1) space.
     */
    static node_t *tree_to_spine(node_t *root, std::size_t &count) noexcept {
        node_t *spine_head = nullptr;
        node_t *spine_tail = nullptr;
        count = 0;

        // Stack-free Morris-like traversal using rotations
        while (root) {
            if (root->left) {
                // Rotate right to bring left child up
                node_t *temp = root->left;
                root->left = temp->right;
                temp->right = root;
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
     *  @param[in] spine1 First sorted spine.
     *  @param[in] spine2 Second sorted spine.
     *  @return node_t* Head of merged spine.
     *
     *  @note O(n+m) time, O(1) space. Just like merging sorted linked lists.
     */
    static node_t *merge_spines(node_t *spine1, node_t *spine2, comparator_t const &comparator) noexcept {
        if (!spine1) return spine2;
        if (!spine2) return spine1;

        node_t *merged_head = nullptr;
        node_t *merged_tail = nullptr;

        while (spine1 && spine2) {
            node_t *next_node;
            if (comparator(spine1->entry, spine2->entry)) {
                next_node = spine1;
                spine1 = spine1->right;
            }
            else {
                next_node = spine2;
                spine2 = spine2->right;
            }

            if (!merged_head) merged_head = next_node;
            if (merged_tail) merged_tail->right = next_node;
            merged_tail = next_node;
        }

        // Append remaining nodes
        node_t *remaining = spine1 ? spine1 : spine2;
        if (merged_tail) merged_tail->right = remaining;
        else merged_head = remaining;

        return merged_head;
    }

    /**
     *  @brief Converts sorted spine back to balanced AVL tree (Day-Stout-Warren algorithm).
     *
     *  @param[in] spine Head of spine.
     *  @param[in] count Number of nodes in spine.
     *  @return node_t* Root of balanced AVL tree.
     *
     *  @note O(n) time using rotations to build perfectly balanced tree.
     *  @see Stout & Warren, "Tree Rebalancing in Optimal Time and Space" (1986)
     */
    static node_t *spine_to_balanced(node_t *spine, std::size_t count) noexcept {
        if (count == 0) return nullptr;
        if (count == 1) {
            spine->left = nullptr;
            spine->right = nullptr;
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
            root->right = right;
            root->height = 1 + std::max(get_height(left), get_height(right));
            return root;
        };

        return build_tree(spine, count, build_tree);
    }

    /**
     *  @brief Merges two AVL trees using Day-Stout-Warren spine algorithm.
     *    Best for large similarly-sized trees: O(n+m) time, O(1) space.
     *    Flattens both trees to spines, merges spines, rebuilds balanced tree.
     *
     *  @param[in] tree1 First tree to merge (will be consumed).
     *  @param[in] tree2 Second tree to merge (will be consumed).
     *  @param[in] comparator Comparator for element comparison.
     *  @param[out] out_size Total number of nodes in result.
     *  @return node_t* Root of merged balanced tree.
     *
     *  @note High constant factor due to many rotations, but O(1) space.
     *    Both input trees are consumed (ownership transferred).
     *
     *  @see Stout & Warren, "Tree Rebalancing in Optimal Time and Space" (1986)
     *  @see https://en.wikipedia.org/wiki/Day%E2%80%93Stout%E2%80%93Warren_algorithm
     */
    static node_t *merge_dsw(node_t *tree1, node_t *tree2, comparator_t const &comparator,
                             std::size_t &out_size) noexcept {
        std::size_t count1 = 0, count2 = 0;

        // Convert both trees to spines: O(n + m)
        node_t *spine1 = tree_to_spine(tree1, count1);
        node_t *spine2 = tree_to_spine(tree2, count2);

        // Merge spines: O(n + m)
        node_t *merged_spine = merge_spines(spine1, spine2, comparator);

        out_size = count1 + count2;

        // Rebuild balanced tree from spine: O(n + m)
        return spine_to_balanced(merged_spine, out_size);
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
 *  @section Entry Constraints
 *
 *  Entries stored in the tree must be comparable using the provided comparator.
 *  They must also be @c noexcept move-constructible and assignable to ensure safety.
 *  When dealing with bulk-insertions of non-copyable types, make sure to use a moving
 *  iterator, like the @c std::move_iterator.
 *
 *  @tparam entry_type_ Type of entries stored in the tree.
 *  @tparam comparator_type_ Comparator for ordering entries. Define @c is_transparent for heterogeneous lookups.
 *  @tparam allocator_type_ Allocator for tree nodes. Must be rebindable to @c basic_avl_node.
 */
template <typename entry_type_, typename comparator_type_,
          typename allocator_type_ = std::allocator<basic_avl_node<entry_type_, comparator_type_>>>
class basic_avl_tree {
  public:
    using node_t = basic_avl_node<entry_type_, comparator_type_>;
    using allocator_t = allocator_type_;
    using comparator_t = comparator_type_;
    using entry_t = entry_type_;

    // Trait to detect if this is a map-like (association-based) container
    using is_associative = std::conditional_t<requires {
        typename entry_t::key_t;
        typename entry_t::value_t;
    }, std::true_type, std::false_type>;

    // Allocator propagation requirement for exception-free move assignment
    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "basic_avl_tree requires allocators that propagate on move assignment");

    /**
     *  @brief Rebind this tree type to different entry and comparator types.
     *    Follows STL allocator rebind pattern for type transformations.
     *
     *  @tparam other_entry_ New entry type for the rebound tree.
     *  @tparam other_comparator_ New comparator type for the rebound tree.
     */
    template <typename other_entry_, typename other_comparator_>
    using rebind = basic_avl_tree<other_entry_, other_comparator_,
                                  typename std::allocator_traits<allocator_t>::template rebind_alloc<
                                      basic_avl_node<other_entry_, other_comparator_>>>;

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

        iterator operator-(int) noexcept {
            iterator tmp = *this;
            -(*this);
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

        const_iterator operator-(int) noexcept {
            const_iterator tmp = *this;
            -(*this);
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
    [[no_unique_address]] comparator_t comparator_;
    [[no_unique_address]] allocator_t allocator_;

    /**
     *  @brief RAII guard for managing subtree cleanup on copy failure.
     *    Automatically cleans up allocated nodes if not explicitly released.
     */
    struct subtree_guard_ {
        allocator_t *allocator_;
        node_t *node_;

        subtree_guard_(allocator_t *allocator, node_t *node) noexcept : allocator_(allocator), node_(node) {}

        ~subtree_guard_() noexcept {
            if (node_) cleanup_();
        }

        node_t *release() noexcept { return std::exchange(node_, nullptr); }

        void cleanup_() noexcept {
            node_t::for_each_bottom_up(node_, [&](node_t *n) noexcept {
                n->entry.~entry_t();
                allocator_->deallocate(n, 1);
            });
        }
    };

    /**
     *  @brief Copies entry from source node into destination node.
     *  @param[in] source Source node to copy from.
     *  @param[in] dest Destination node (must have allocated memory, but entry not constructed).
     *  @return status_t @c success_k if copy succeeded, error code otherwise.
     */
    static status_t copy_entry_into_(node_t *source, node_t *dest) noexcept {
        if constexpr (has_copy_method<entry_t>) {
            auto entry_copy = source->entry.copy();
            if (!entry_copy) return entry_copy.status;
            new (&dest->entry) entry_t(std::move(entry_copy.entry));
        }
        else if constexpr (std::is_nothrow_copy_constructible_v<entry_t>) { new (&dest->entry) entry_t(source->entry); }
        else {
            static_assert(std::is_nothrow_copy_constructible_v<entry_t> || has_copy_method<entry_t>,
                          "Entry type must be nothrow copy-constructible or provide .copy() method");
            return status_t {unknown_k};
        }
        return status_t {success_k};
    }

    /**
     *  @brief Recursively copies a subtree.
     *  @param[in] source Root of source subtree to copy.
     *  @param[inout] allocator Allocator for node allocation.
     *  @return node_t* Root of copied subtree, or nullptr on failure.
     */
    static node_t *copy_subtree_(node_t *source, allocator_t &allocator) noexcept {
        if (!source) return nullptr;

        // Allocate new node
        node_t *new_node = allocator.allocate(1);
        if (!new_node) return nullptr;

        // Copy entry into new node
        auto status = copy_entry_into_(source, new_node);
        if (!status) {
            allocator.deallocate(new_node, 1);
            return nullptr;
        }

        // Initialize node structure
        new_node->height = source->height;
        new_node->left = nullptr;
        new_node->right = nullptr;

        // Use RAII guard to ensure cleanup on failure
        subtree_guard_ guard(&allocator, new_node);

        // Recursively copy left subtree
        new_node->left = copy_subtree_(source->left, allocator);
        if (source->left && !new_node->left) return nullptr;

        // Recursively copy right subtree
        new_node->right = copy_subtree_(source->right, allocator);
        if (source->right && !new_node->right) return nullptr;

        // Success - release guard and return
        return guard.release();
    }

  public:
#pragma mark - Constructors and Assignment

    basic_avl_tree() noexcept = default;
    explicit basic_avl_tree(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}

    basic_avl_tree(basic_avl_tree &&other) noexcept
        : root_(std::exchange(other.root_, nullptr)), size_(std::exchange(other.size_, 0)),
          comparator_(std::move(other.comparator_)), allocator_(std::move(other.allocator_)) {}

    // Copy operations are explicitly deleted - use .copy() method for deep copies
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
     *  @return expected<basic_avl_tree> Copy of the tree, or error status on failure.
     *  @note This operation may fail due to allocation errors during node duplication.
     *    The returned tree uses a copy of this tree's allocator.
     */
    expected<basic_avl_tree> copy() const noexcept {
        basic_avl_tree result {allocator_};
        result.comparator_ = comparator_;

        if (!root_) return expected<basic_avl_tree>(std::move(result), status_t {success_k});

        result.root_ = copy_subtree_(root_, result.allocator_);
        if (!result.root_) {
            return expected<basic_avl_tree>(basic_avl_tree(allocator_), status_t {out_of_memory_heap_k});
        }

        result.size_ = size_;
        return expected<basic_avl_tree>(std::move(result), status_t {success_k});
    }

#pragma mark - Capacity

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
     *  @return allocator_t& Reference to the allocator.
     */
    allocator_t &allocator() noexcept { return allocator_; }

    /**
     *  @brief Returns the allocator associated with the tree (const version).
     *  @return allocator_t const& Const reference to the allocator.
     */
    allocator_t const &allocator() const noexcept { return allocator_; }

#pragma mark - Iterators

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

#pragma mark - Lookup

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
     *  @brief Checks if an element equal to @p comparable exists in the tree.
     *    Heterogeneous lookup supported if comparator defines @c is_transparent.
     *
     *  @param[in] comparable Object comparable to @c entry_t.
     *  @return bool True if element found, false otherwise.
     */
    template <typename comparable_type_>
    bool contains(comparable_type_ &&comparable) const noexcept {
        return node_t::find(root_, std::forward<comparable_type_>(comparable), comparator_) != nullptr;
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
        return iterator(this, node_t::lower_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
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
        return const_iterator(this,
                              node_t::lower_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
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
        return iterator(this, node_t::upper_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
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
        return const_iterator(this,
                              node_t::upper_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
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
     *  @param[in] callback Callback invoked for each element equal to the key. Must be @c noexcept.
     */
    template <typename comparable_type_ = entry_t, typename callback_type_ = no_op_t>
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
    template <typename lower_type_ = entry_t, typename upper_type_ = entry_t, typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        node_t::range(root_, std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), comparator_,
                      [&](node_t *node) noexcept { callback(node->entry); });
    }

    /**
     *  @brief Iterates over all entries in the range [ @p lower, @p upper), allowing in-place modification.
     *    Invokes callback for each mutable element in the specified range. Non-const version.
     *
     *  @param[in] lower Lower bound of the range (inclusive).
     *  @param[in] upper Upper bound of the range (exclusive).
     *  @param[inout] callback Callback invoked for each mutable element in range. Must be @c noexcept.
     */
    template <typename lower_type_ = entry_t, typename upper_type_ = entry_t, typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        node_t::range(root_, std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), comparator_,
                      [&](node_t *node) noexcept { callback(node->entry); });
    }

#pragma mark - Modifiers

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
        auto result = node_t::insert(root_, std::forward<comparable_type_>(comparable), comparator_,
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
     *  @brief Constructs an element in-place. Matches @c std::set::emplace() semantics.
     *         Does not insert if key already exists.
     *
     *  @tparam args_types_ Types of arguments to forward to entry_t constructor.
     *  @param[in] args Arguments to forward to entry_t constructor.
     *  @return std::pair<iterator, bool> Pair of iterator to inserted/existing element and bool indicating success.
     */
    template <typename... args_types_>
    std::pair<iterator, bool> emplace(args_types_ &&...args) noexcept {
        return insert(entry_t(std::forward<args_types_>(args)...));
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
     *  @tparam tags_types_ Optional tag types ( @c assume_sorted_t for O(n) bulk construction).
     *  @param[in] first Beginning of range to insert.
     *  @param[in] last End of range to insert.
     *  @param[in] tags Optional tags to control insertion behavior.
     *  @return status_t First error encountered, or success if all elements inserted.
     *
     *  @note With @c assume_sorted_t tag: O(n) time, builds perfectly balanced tree.
     *    Precondition: Tree must be empty and range must be sorted.
     *    Without tag: O(n log n) time, inserts elements one-by-one.
     */
    template <typename input_iterator_type_, typename... tags_types_>
    status_t insert(input_iterator_type_ first, input_iterator_type_ last, tags_types_... tags) noexcept {

        // O(n): Build balanced tree from sorted range
        if constexpr (contains_type<assume_sorted_t, tags_types_...>()) {
            if (root_) return {errc_t::operation_not_permitted_k}; // Must be empty

            auto count = std::distance(first, last);
            if (count == 0) return {success_k};

            root_ = node_t::build_from_sorted(first, count, [&]() noexcept { return allocator_.allocate(1); });

            if (!root_) {
                // Allocation failed - tree remains empty
                return {errc_t::out_of_memory_heap_k};
            }

            size_ = count;
            return {success_k};
        }
        // O(n log n): Insert elements one-by-one
        else {
            for (; first != last; ++first) {
                auto result = insert(entry_t(*first));
                if (result.first == end() && !result.second) return {errc_t::out_of_memory_heap_k};
            }
            return {success_k};
        }
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

#pragma mark - Observers

    /**
     *  @brief Returns the function object that compares keys.
     *  @return comparator_t The comparison function object.
     */
    comparator_t key_comp() const noexcept { return comparator_; }

    /**
     *  @brief Returns the function object that compares values.
     *         For sets, this is the same as key_comp().
     *  @return comparator_t The comparison function object.
     */
    comparator_t value_comp() const noexcept { return comparator_; }

    /**
     *  @brief Returns the maximum possible number of elements.
     *  @return std::size_t Theoretical maximum size.
     */
    std::size_t max_size() const noexcept {
        return std::min(allocator_.max_size(), std::numeric_limits<std::size_t>::max() / sizeof(node_t));
    }

    /**
     *  @brief Exchanges the contents of this tree with another.
     *
     *  Swaps tree structure, size, comparator, and (if propagating) allocators.
     *  For non-propagating allocators (when @c propagate_on_container_swap is @c false),
     *  the allocators must be equal, otherwise @c invalid_argument_k is returned.
     *
     *  @param[inout] other Tree to swap with.
     *  @return status_t @c success_k if swap succeeded, or @c invalid_argument_k
     *    if allocators are incompatible and non-propagating.
     */
    [[nodiscard]] status_t swap(basic_avl_tree &other) noexcept {
        // For non-propagating allocators, they must be equal (C++ standard requirement)
        if constexpr (!std::allocator_traits<allocator_t>::propagate_on_container_swap::value)
            if (!(allocator_ == other.allocator_)) return status_t {invalid_argument_k};

        std::swap(root_, other.root_);
        std::swap(size_, other.size_);
        std::swap(comparator_, other.comparator_);

        // Only swap allocators if they propagate on swap
        if constexpr (std::allocator_traits<allocator_t>::propagate_on_container_swap::value)
            std::swap(allocator_, other.allocator_);

        return status_t {success_k};
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
        auto result = node_t::extract(root_, std::forward<comparable_type_>(comparable), comparator_);
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
        auto result = node_t::extract(root_, std::forward<comparable_type_>(comparable), comparator_);
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
        node_t::for_each_bottom_up(root_, [&](node_t *node) noexcept {
            node->entry.~entry_t();
            allocator_.deallocate(node, 1);
        });
        root_ = nullptr;
        size_ = 0;
    }

    template <typename callback_type_>
    void for_each(callback_type_ &&callback) noexcept {
        node_t::for_each_left_right(root_, [&](node_t *node) noexcept { callback(node->entry); });
    }

#pragma mark - Merge Operations

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
            size_ += result.inserted;
            // Key conflict - node wasn't inserted, deallocate it
            if (!result.inserted) allocator_.deallocate(node, 1);
        });
        other.root_ = nullptr;
        other.size_ = 0;
    }

    /**
     *  @brief Merges another tree into this one with optimized algorithm for disjoint trees.
     *    Precondition: Trees have no overlapping keys (disjoint).
     *
     *  @param[inout] other Tree to merge from. Will be empty after merge.
     *  @param assume_unique Tag indicating trees are disjoint (no duplicate keys).
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
        if (comparator_(this_max->entry, other_min->entry)) {
            root_ = node_t::join(root_, other.root_, comparator_);
            size_ += other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        // Fast path: all(other) < all(this), use join: O(log n)
        if (comparator_(other_max->entry, this_min->entry)) {
            root_ = node_t::join(other.root_, root_, comparator_);
            size_ += other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        // Strategy 2: Adaptive selection between split-based and DSW
        std::size_t min_size = std::min(size_, other.size_);
        std::size_t max_size = std::max(size_, other.size_);

        // Heuristic: Use DSW when both large and similar size
        constexpr std::size_t DSW_THRESHOLD = 10000;
        constexpr std::size_t SIZE_RATIO_THRESHOLD = 3;

        // DSW for large similarly-sized trees: O(m+n)
        if (min_size > DSW_THRESHOLD && max_size < min_size * SIZE_RATIO_THRESHOLD) {
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
        size_ += result.inserted;
        // Key conflict - node wasn't inserted, deallocate it
        if (!result.inserted) allocator_.deallocate(node_to_insert, 1);
    }

#pragma mark - Split and Join

    /**
     *  @brief Result of a split operation on a tree.
     */
    struct split_result_t {
        basic_avl_tree left;  //!< Tree with all elements < key.
        basic_avl_tree right; //!< Tree with all elements >= key.
    };

    /**
     *  @brief Splits the tree at a given key into two trees.
     *    Elements < key go to left tree, elements >= key go to right tree.
     *    This tree becomes empty after the split.
     *
     *  @param[in] comparable Key to split at.
     *  @return split_result_t Contains left tree (< key) and right tree (>= key).
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

template <typename entry_type_, typename comparator_type_, typename allocator_type_>
using avl_set = basic_avl_tree<entry_type_, comparator_type_, allocator_type_>;

template <typename key_type_, typename value_type_, typename comparator_type_, typename allocator_type_>
using avl_map = basic_avl_tree<association<key_type_, value_type_>, comparator_type_, allocator_type_>;

} // namespace ashvardanian::smashtable
