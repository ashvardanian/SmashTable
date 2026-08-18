/**
 *  @brief Weight-balanced tree with order statistics support. Size-balanced alternative to AVL trees, enabling O(log n)
 *      @c select and @c rank operations.
 *  @author Ash Vardanian
 *  @file include/smashtable/basic_wb_tree.hpp
 *  @date October 24, 2025
 *
 *  @section basic_wb_tree_weight_balanced_trees Weight-Balanced Trees
 *
 *  Weight-balanced trees maintain balance based on subtree sizes rather than heights. Rebalancing uses parameters
 *  Δ=3 and Γ=2, which are the only proven integer solution (Hirai & Yamamoto, 2011).
 *
 *  Balance invariant: For every node, size(left) < Δ × size(right) AND size(right) < Δ × size(left)
 *
 *  @section basic_wb_tree_order_statistics Order Statistics
 *
 *  Storing sizes enables efficient order statistic queries:
 *  - @c select(k): Find k-th smallest element in O(log n)
 *  - @c rank(x): Find position of element x in O(log n)
 *
 *  Use cases: pagination (OFFSET/LIMIT), percentiles, window functions, quantile estimation.
 *
 *  A wrapper that answers for only some of the stored entries supplies an augmentation policy, and the tree
 *  keeps a second subtree count over that predicate - @c select_augmented and @c rank_augmented descend on it
 *  in O(log n). An unaugmented tree keeps neither the field nor the work.
 *
 *  @section basic_wb_tree_performance Performance
 *
 *  - Expected depth: ~1.88 log₂(n) vs AVL's 1.44 log₂(n)
 *  - Amortized rotations: O(1) per update vs AVL's O(log n) worst-case
 *  - Space overhead: Same as AVL (1 word per node for size vs height)
 */
#pragma once
#include <cassert> // `assert`

#include <concepts>    // `std::convertible_to`
#include <memory>      // `std::allocator`
#include <type_traits> // `std::conditional_t`, `std::is_same_v`
#include <utility>     // `std::pair`, `std::exchange`

#include "shared.hpp"

namespace ashvardanian::smashtable {

#pragma region Augmentation Policies

/**
 *  @brief Marks a tree that keeps only @c size, leaving nodes byte-identical to an unaugmented build.
 */
struct no_augmentation_t {};

/**
 *  @brief Contract a wrapper satisfies to keep a second subtree count beside @c size.
 *    @c augmented_count reads the entry and returns 0 or 1. The tree recomputes it from the entry alone on
 *    every rotation, join, and split, so a predicate that depends on anything else - a sibling entry, a
 *    global stamp - has to be materialized into the entry by the wrapper before the tree can aggregate it.
 */
template <typename augmentation_type_, typename value_type_>
concept wb_augmentation = requires(value_type_ const &fruit) {
    { augmentation_type_::augmented_count(fruit) } -> std::convertible_to<std::size_t>;
};

/** @brief Stand-in for the augmented counter in an unaugmented tree, occupying no bytes. */
struct no_augmented_count_t {};

#pragma endregion Augmentation Policies

/**
 *  @brief Node for weight-balanced binary search tree.
 *    Stores subtree size for O(log n) order statistics and O(1) amortized rebalancing.
 *
 *  @tparam value_type_ Type of elements stored in nodes.
 *  @tparam comparator_type_ Comparator defining ordering. For heterogeneous lookups, define:
 *    @code using is_transparent = void; @endcode inside the comparator.
 *
 *  @section basic_wb_tree_rebalancing_parameters Rebalancing Parameters
 *
 *  Δ=3: Rotation threshold. Rebalance if size(left) >= 3×size(right) or vice versa.
 *  Γ=2: Rotation type selector. Single rotation if size(heavy.inner) < 2×size(heavy.outer).
 *
 *  These are the @b only valid integer parameters, proven in Coq by Hirai and Yamamoto in 2011, and are
 *  exposed as @c delta_k and @c gamma_k.
 *
 *  Layout: @c fruit is the stored entry, @c left and @c right the links, and @c size the number of nodes
 *  in the subtree rooted here - invariant @c size @c = @c 1 @c + @c size(left) @c + @c size(right), which
 *  is what makes @c select and @c rank logarithmic.
 *
 *  @tparam augmentation_type_ Policy carrying a second per-subtree count, kept over a predicate of the
 *    wrapper's choosing. The default @c no_augmentation_t adds no field and no work.
 *
 *  @section basic_wb_tree_augmentation Augmentation
 *
 *  An augmented tree keeps @c augmented_size next to @c size, holding the number of entries in the
 *  subtree for which @c augmentation_type_::augmented_count returns 1. Because that count is read off the
 *  entry, every rotation, join, and split restores it from the moved node's own entry and its two children.
 *  An entry whose count flips while it sits in the tree is repaired by @c refresh_augmentation, which
 *  rewalks only the root-to-entry path and may be called any number of times per write - one wrapper
 *  operation is free to retag several entries.
 */
template <typename value_type_, typename comparator_type_, typename augmentation_type_ = no_augmentation_t>
class basic_wb_node {
  public:
    using value_t = value_type_;
    using comparator_t = comparator_type_;
    using augmentation_t = augmentation_type_;
    using size_t = std::size_t;
    using node_t = basic_wb_node;

    /** @brief True when a second count is maintained, which is what gates every augmented code path. */
    static constexpr bool is_augmented_k = !std::is_same_v<augmentation_t, no_augmentation_t>;
    static_assert(!is_augmented_k || wb_augmentation<augmentation_t, value_t>,
                  "An augmentation policy must expose `static std::size_t augmented_count(value_t const &)`");

    /** @brief The augmented counter, degenerating to an empty type when no augmentation is asked for. */
    using augmented_count_t = std::conditional_t<is_augmented_k, std::size_t, no_augmented_count_t>;

    value_t fruit;
    node_t *left = nullptr;
    node_t *right = nullptr;

    size_t size = 1;

    ST_NO_UNIQUE_ADDRESS_ augmented_count_t augmented_size {};

    static constexpr size_t delta_k = 3;
    static constexpr size_t gamma_k = 2;

    static size_t get_size(node_t *node) noexcept { return node ? node->size : 0; }

    /** @brief Number of entries in the subtree rooted at @p node that the policy counts. */
    static size_t get_augmented_size(node_t *node) noexcept {
        if constexpr (is_augmented_k) return node ? node->augmented_size : 0;
        else return 0;
    }

    /** @brief The entry's own contribution to the augmented count, 0 or 1. */
    static size_t get_own_augmented_count(node_t *node) noexcept {
        if constexpr (is_augmented_k) return augmentation_t::augmented_count(node->fruit);
        else return 0;
    }

#pragma region Traversal and Search

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
            if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit))) node = node->left;
            else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable)))
                node = node->right;
            else break;
        }
        return node;
    }

    /**
     *  @brief Find smallest fruit >= comparable (lower bound).
     *  @param[in] node Root of subtree to search.
     *  @param[in] comparable Key to search for.
     *  @param[in] comparator Comparator instance (may be stateful).
     *  @return Pointer to lower bound node, or nullptr if all elements < comparable.
     */
    template <typename comparable_type_>
    static node_t *lower_bound(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        node_t *result = nullptr;
        while (node) {
            if (!comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable))) {
                result = node;
                node = node->left;
            }
            else { node = node->right; }
        }
        return result;
    }

    /**
     *  @brief Find smallest fruit > comparable (upper bound).
     *  @param[in] node Root of subtree to search.
     *  @param[in] comparable Key to search for.
     *  @param[in] comparator Comparator instance (may be stateful).
     *  @return Pointer to upper bound node, or nullptr if all elements <= comparable.
     */
    template <typename comparable_type_>
    static node_t *upper_bound(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        node_t *result = nullptr;
        while (node) {
            if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit))) {
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
     *  @return The next node in order, or @c nullptr when there is none.
     */
    static node_t *find_successor(node_t *root, node_t *node, comparator_t const &comparator) noexcept {
        if (!node) return find_min(root);
        return upper_bound(root, node->fruit, comparator);
    }

    /**
     *  @brief Finds the previous node in in-order traversal (predecessor).
     *  @param[in] root Root of the tree.
     *  @param[in] node Current node.
     *  @param[in] comparator Comparator for ordering.
     *  @return The previous node in order, or @c nullptr when there is none.
     */
    static node_t *find_predecessor(node_t *root, node_t *node, comparator_t const &comparator) noexcept {
        if (!node) return find_max(root);

        node_t *predecessor = nullptr;
        node_t *current = root;

        while (current) {
            // Current is less than target, it's a candidate predecessor
            if (comparator(mapping_key_or_itself(current->fruit), mapping_key_or_itself(node->fruit))) {
                predecessor = current;
                current = current->right;
            }
            // Current is >= target, search left subtree
            else current = current->left;
        }
        return predecessor;
    }

#pragma endregion Traversal and Search

#pragma region Order Statistics

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
     *  auto position = rank(root, value, comp);
     *  // position elements are smaller than value
     *  // If value exists, it's at index position
     *  @endcode
     */
    template <typename comparable_type_>
    static size_t rank(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        if (!node) return 0;

        if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit))) {
            // comparable < node, search left
            return rank(node->left, std::forward<comparable_type_>(comparable), comparator);
        }
        else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable))) {
            // node < comparable, search right
            return get_size(node->left) + 1 + rank(node->right, std::forward<comparable_type_>(comparable), comparator);
        }
        else {
            // Found it - return count of smaller elements
            return get_size(node->left);
        }
    }

    /**
     *  @brief Select the k-th smallest entry among those the augmentation policy counts.
     *  @param[in] node Root of subtree.
     *  @param[in] k Index among counted entries, 0 for the smallest.
     *  @return Pointer to the k-th counted node, or nullptr when fewer than k+1 are counted.
     *
     *  @par Complexity
     *  O(log n), following one root-to-node path and reading only the stored counts.
     */
    static node_t *select_augmented(node_t *node, size_t k) noexcept {
        while (node) {
            size_t const left_count = get_augmented_size(node->left);
            if (k < left_count) {
                node = node->left;
                continue;
            }
            k -= left_count;
            size_t const own_count = get_own_augmented_count(node);
            if (own_count && k == 0) return node;
            k -= own_count;
            node = node->right;
        }
        return nullptr;
    }

    /**
     *  @brief Number of counted entries ordered before @p comparable.
     *  @param[in] node Root of subtree.
     *  @param[in] comparable Key to find the augmented rank of.
     *  @param[in] comparator Comparator instance.
     *
     *  @par Complexity
     *  O(log n).
     */
    template <typename comparable_type_>
    static size_t rank_augmented(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        size_t counted = 0;
        while (node) {
            if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit))) node = node->left;
            else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable))) {
                counted += get_augmented_size(node->left) + get_own_augmented_count(node);
                node = node->right;
            }
            else return counted + get_augmented_size(node->left);
        }
        return counted;
    }

    /**
     *  @brief Repairs the augmented counts on the path to @p comparable after its predicate flipped.
     *  @return True when the entry was found, which is when anything was repaired.
     *
     *  @par Complexity
     *  O(log n), touching only the ancestors whose counts could have moved.
     */
    template <typename comparable_type_>
    static bool refresh_augmentation(node_t *node, comparable_type_ &&comparable,
                                     comparator_t const &comparator) noexcept {
        if (!node) return false;

        bool found;
        if (comparator(mapping_key_or_itself(comparable), mapping_key_or_itself(node->fruit)))
            found = refresh_augmentation(node->left, comparable, comparator);
        else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable)))
            found = refresh_augmentation(node->right, comparable, comparator);
        else found = true;

        if (found) update_augmented_size(node);
        return found;
    }

    /**
     *  @brief In-order traversal (left-root-right) with callback.
     *    Visits nodes in sorted order according to comparator.
     *
     *  @param[in] node Root of subtree to traverse.
     *  @param[in] callback Callback to invoke for each node. Must be @c noexcept.
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
     *  @brief Finds all nodes in the half-open range [low, high) and invokes callback for each.
     *    Recursively traverses the tree, visiting only nodes within the specified range.
     *
     *  @param[in] node Root of subtree to search.
     *  @param[in] low Lower bound of range (inclusive).
     *  @param[in] high Upper bound of range (exclusive).
     *  @param[in] comparator Comparator for element comparison.
     *  @param[in] callback Callback to invoke for each node in range. Must be @c noexcept.
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
        if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(high)) &&
            !comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(low))) {
            auto left_sub_interval = range(node->left, low, high, comparator, callback);
            callback(node);
            auto right_sub_interval = range(node->right, low, high, comparator, callback);

            auto result = node_interval_t {};
            result.lower_bound = left_sub_interval.lower_bound ? left_sub_interval.lower_bound : node;
            result.upper_bound = right_sub_interval.upper_bound ? right_sub_interval.upper_bound : node;
            result.lowest_common_ancestor = node;
            return result;
        }

        if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(low)))
            return range(node->right, low, high, comparator, callback);

        return range(node->left, low, high, comparator, callback);
    }

#pragma endregion Order Statistics

#pragma region Rotations and Rebalancing

    /**
     *  @brief Update size field to match children.
     *    Must be called after any operation that modifies children.
     */
    static void update_size(node_t *node) noexcept {
        if (!node) return;
        node->size = 1 + get_size(node->left) + get_size(node->right);
        update_augmented_size(node);
    }

    /**
     *  @brief Recomputes only the augmented count, for a node whose shape is already correct.
     *    Used where an entry's own predicate changed but no link moved.
     */
    static void update_augmented_size(node_t *node) noexcept {
        if constexpr (is_augmented_k) {
            if (!node) return;
            node->augmented_size =
                get_own_augmented_count(node) + get_augmented_size(node->left) + get_augmented_size(node->right);
        }
    }

    /** @brief Restores a node whose links were just cleared, so both counts describe it alone. */
    static void reset_to_leaf(node_t *node) noexcept {
        node->size = 1;
        update_augmented_size(node);
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
     *  @brief Weight of a subtree, counting the empty tree as 1.
     *    Hirai and Yamamoto state the invariant over @c size+1; comparing raw sizes makes
     *    every node with an empty child look unbalanced and asks for rotations that cannot
     *    be performed, since the pivot's child is null.
     */
    static size_t get_weight(node_t *node) noexcept { return get_size(node) + 1; }

    /**
     *  @brief Check if node satisfies weight-balance invariant.
     *  @return True if balanced: size(left) < Δ×size(right) AND size(right) < Δ×size(left)
     */
    static bool is_balanced(node_t *node) noexcept {
        if (!node) return true;
        size_t left_weight = get_weight(node->left);
        size_t right_weight = get_weight(node->right);
        return left_weight <= delta_k * right_weight && right_weight <= delta_k * left_weight;
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

        size_t left_weight = get_weight(node->left);
        size_t right_weight = get_weight(node->right);

        // Left too heavy. The pivot's own children are non-empty here, so both rotations are safe.
        if (left_weight > delta_k * right_weight) {
            node_t *left = node->left;
            size_t inner_weight = get_weight(left->right);
            size_t outer_weight = get_weight(left->left);
            if (inner_weight < gamma_k * outer_weight) return rotate_right(node);
            node->left = rotate_left(left);
            return rotate_right(node);
        }

        // Right too heavy, mirrored.
        if (right_weight > delta_k * left_weight) {
            node_t *right = node->right;
            size_t inner_weight = get_weight(right->left);
            size_t outer_weight = get_weight(right->right);
            if (inner_weight < gamma_k * outer_weight) return rotate_left(node);
            node->right = rotate_right(right);
            return rotate_left(node);
        }

        return node; // Already balanced
    }

#pragma endregion Rotations and Rebalancing

#pragma region Removals

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
     *  @param[in] node Node to extract.
     *  @param[in] comparator Comparator for element comparison.
     */
    static extract_result_t extract(node_t *node, comparator_t const &comparator) noexcept {

        // If the node has two children, replace it with the
        // smallest fruit in the right branch.
        if (node->left && node->right) {
            node_t *midpoint = find_min(node->right);
            auto downstream = extract(node->right, midpoint->fruit, comparator);
            midpoint = downstream.extracted.release();
            midpoint->left = node->left;
            midpoint->right = downstream.root;
            // The right branch shrank by one under the promoted midpoint, so it may now be the
            // lighter side by more than Δ allows; only the midpoint itself can be out of balance.
            midpoint = rebalance_after_extract(midpoint);
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            reset_to_leaf(node);
            return {midpoint, std::unique_ptr<node_t> {node}};
        }
        // Just one child is present, so it is the natural successor.
        else if (node->left || node->right) {
            node_t *replacement = node->left ? node->left : node->right;
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            reset_to_leaf(node);
            return {replacement, std::unique_ptr<node_t> {node}};
        }
        // No children are present.
        else {
            // Detach the `node` from the descendants.
            node->left = node->right = nullptr;
            reset_to_leaf(node);
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
            auto downstream = extract(node->left, comparable, comparator);
            node->left = downstream.root;
            if (downstream.extracted) node = rebalance_after_extract(node);
            return {node, std::move(downstream.extracted)};
        }

        else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable))) {
            auto downstream = extract(node->right, comparable, comparator);
            node->right = downstream.root;
            if (downstream.extracted) node = rebalance_after_extract(node);
            return {node, std::move(downstream.extracted)};
        }

        else
            // We have found the node to extract!
            return extract(node, comparator);
    }

#pragma endregion Removals

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

    /**
     *  @brief Inserts an existing node into the tree.
     *    Used by merge operation to insert extracted nodes.
     *  @param[in] node Root of subtree to insert into.
     *  @param[in] new_child Pre-allocated node to insert.
     *  @param[in] comparator Comparator for element comparison.
     *  @return The new root, the node the key lives in, and whether it was made or matched.
     */
    static find_or_make_result_t insert(node_t *node, node_t *new_child, comparator_t const &comparator) noexcept {
        if (!node) return {new_child, new_child, new_child ? node_placement_t::made_k : node_placement_t::refused_k};

        if (comparator(mapping_key_or_itself(new_child->fruit), mapping_key_or_itself(node->fruit))) {
            auto result = insert(node->left, new_child, comparator);
            node->left = result.root;
            if (result.placement == node_placement_t::made_k) {
                update_size(node);
                node = rebalance(node);
            }
            return {node, result.match, result.placement};
        }

        else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(new_child->fruit))) {
            auto result = insert(node->right, new_child, comparator);
            node->right = result.root;
            if (result.placement == node_placement_t::made_k) {
                update_size(node);
                node = rebalance(node);
            }
            return {node, result.match, result.placement};
        }

        else {
            // Key already exists - don't insert
            return {node, node, node_placement_t::matched_k};
        }
    }

    /**
     *  @brief Inserts or updates an fruit in the tree.
     *    If key exists, overwrites the fruit. If not, creates new node.
     *
     *  @param[in] node Root of subtree.
     *  @param[in] fruit Entry to insert or assign (moved).
     *  @param[in] comparator Comparator for element comparison.
     *  @param[in] node_allocator Allocator function that returns new node pointer or nullptr on failure.
     *  @return The new root, the node the key lives in, and whether it was made or matched.
     */
    template <typename node_allocator_type_>
    static find_or_make_result_t upsert(node_t *node, value_type_ &&fruit, comparator_t const &comparator,
                                        node_allocator_type_ &&node_allocator) noexcept {
        // Base case: empty tree, allocate new node
        if (!node) {
            node_t *new_node = node_allocator();
            if (!new_node) return {nullptr, nullptr, node_placement_t::refused_k};
            new (&new_node->fruit) value_type_(std::move(fruit));
            new_node->left = nullptr;
            new_node->right = nullptr;
            reset_to_leaf(new_node);
            return {new_node, new_node, node_placement_t::made_k};
        }

        // Recursive case: search for insertion point
        if (comparator(mapping_key_or_itself(fruit), mapping_key_or_itself(node->fruit))) {
            auto result = upsert(node->left, std::move(fruit), comparator, node_allocator);
            node->left = result.root;
            if (result.placement == node_placement_t::made_k) {
                update_size(node);
                node = rebalance(node);
            }
            // Overwriting an existing entry leaves the shape alone but may flip its augmented count.
            else update_augmented_size(node);
            return {node, result.match, result.placement};
        }
        else if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(fruit))) {
            auto result = upsert(node->right, std::move(fruit), comparator, node_allocator);
            node->right = result.root;
            if (result.placement == node_placement_t::made_k) {
                update_size(node);
                node = rebalance(node);
            }
            else update_augmented_size(node);
            return {node, result.match, result.placement};
        }
        else {
            // Key already exists - update the fruit
            node->fruit = std::move(fruit);
            update_augmented_size(node);
            return {node, node, node_placement_t::matched_k};
        }
    }

    /** @brief Visits every node bottom-up, so a caller may free each one as it goes. */
    template <typename callback_type_>
    static void for_each_bottom_up(node_t *node, callback_type_ &&callback) noexcept {
        if (!node) return;
        for_each_bottom_up(node->left, callback);
        for_each_bottom_up(node->right, callback);
        callback(node);
    }

    struct erase_if_result_t {
        node_t *root = nullptr;
        std::size_t count = 0;
    };

    /**
     *  @brief Drops every node satisfying @p predicate, returning the new root and surviving count.
     *    Both halves may shed any number of nodes, so the two survivors are re-joined rather than
     *    stitched back in place - rotating once could not close a gap of many weight classes.
     */
    template <typename predicate_type_, typename node_deallocator_type_>
    static erase_if_result_t erase_if(node_t *node, predicate_type_ &&predicate,
                                      node_deallocator_type_ &&node_deallocator,
                                      comparator_t const &comparator) noexcept {
        if (!node) return {nullptr, 0};

        auto left_result = erase_if(node->left, predicate, node_deallocator, comparator);
        auto right_result = erase_if(node->right, predicate, node_deallocator, comparator);
        node->left = node->right = nullptr;
        reset_to_leaf(node);

        std::size_t const surviving = left_result.count + right_result.count;
        if (predicate(node->fruit)) {
            node_deallocator(node);
            return {join(left_result.root, right_result.root, comparator), surviving};
        }
        return {join_with_root(left_result.root, node, right_result.root), surviving + 1};
    }

    /** @brief Uniformly samples one node whose key lies in [lower, upper), or nullptr when empty. */
    template <typename lower_type_, typename upper_type_, typename generator_type_, typename predicate_type_ = no_op_t>
    static node_t *sample_range(node_t *node, lower_type_ &&lower, upper_type_ &&upper, comparator_t const &comparator,
                                generator_type_ &&generator, predicate_type_ &&predicate = {}) noexcept {

        // Reservoir sampling keeps this a single pass without materializing the range.
        node_t *chosen = nullptr;
        std::size_t seen = 0;
        range(node, lower, upper, comparator, [&](node_t *candidate) noexcept {
            if constexpr (!std::is_same_v<std::remove_cvref_t<predicate_type_>, no_op_t>)
                if (!predicate(candidate)) return;
            ++seen;
            if (draw_below(generator, seen) == 0) chosen = candidate;
        });
        return chosen;
    }

#pragma endregion Insertions

#pragma region Split and Join Operations

    /**
     *  @brief Result of splitting a tree at a key.
     */
    struct split_result_t {
        /** @brief Subtree with all elements ordered before the split key. */
        node_t *left = nullptr;
        /** @brief Subtree with all elements not ordered before the split key. */
        node_t *right = nullptr;
    };

    /**
     *  @brief Joins two trees with a root node between them.
     *    Precondition: @p root is non-null and all(left) < root < all(right).
     *
     *  @param[in] left Left subtree (all elements < root).
     *  @param[in] root Middle node to join with.
     *  @param[in] right Right subtree (all elements > root).
     *  @return New root of joined tree.
     *
     *  @par Complexity
     *  O(log n) expected, where n is size of larger tree.
     *
     *  The two sides may sit many weight classes apart, so hanging @p root between them and
     *  rotating once is not enough. Walking down the heavier side's inner spine until the two
     *  pieces are within Δ of each other, then rebalancing on the way back up, is what keeps the
     *  invariant - this is what makes @c split and @c erase_range produce balanced trees.
     */
    static node_t *join_with_root(node_t *left, node_t *root, node_t *right) noexcept {
        assert(root && "Joining without a middle node loses one of the sides");

        // Left is heavy enough that `root` belongs somewhere down its right spine.
        // The comparison itself proves `left` is non-null: its weight exceeds Δ ≥ 1.
        if (get_weight(left) > delta_k * get_weight(right)) {
            left->right = join_with_root(left->right, root, right);
            update_size(left);
            return rebalance(left);
        }

        // Right is heavy, mirrored.
        if (get_weight(right) > delta_k * get_weight(left)) {
            right->left = join_with_root(left, root, right->left);
            update_size(right);
            return rebalance(right);
        }

        root->left = left;
        root->right = right;
        update_size(root);
        return root;
    }

    /**
     *  @brief Joins two trees where all(left) < all(right).
     *    Recursively joins trees by extracting min from right subtree.
     *
     *  @param[in] left Left subtree.
     *  @param[in] right Right subtree.
     *  @param[in] comparator Comparator for ordering elements.
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
        auto extracted = extract(right, min_node->fruit, comparator);

        return join_with_root(left, extracted.extracted.release(), extracted.root);
    }

    /**
     *  @brief Splits tree at a given key.
     *    Returns two trees: left contains all elements < comparable,
     *    right contains all elements >= comparable.
     *
     *  @param[in] node Root of tree to split.
     *  @param[in] comparable Key to split at.
     *  @param[in] comparator Comparator for element comparison.
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
    static split_result_t split(node_t *node, comparable_type_ &&comparable, comparator_t const &comparator) noexcept {
        if (!node) return {nullptr, nullptr};

        // The left half holds keys strictly below the split point, so `split` composes into
        // half-open ranges the way `erase_range` and the AVL tree expect.
        if (comparator(mapping_key_or_itself(node->fruit), mapping_key_or_itself(comparable))) {
            auto downstream = split(node->right, comparable, comparator);
            return {join_with_root(node->left, node, downstream.left), downstream.right};
        }
        else {
            auto downstream = split(node->left, comparable, comparator);
            return {downstream.left, join_with_root(downstream.right, node, node->right)};
        }
    }
};

/**
 *  @brief Weight-balanced tree container with order statistics support.
 *    Provides ordered storage similar to @c std::set with additional O(log n) select/rank operations.
 *
 *  @tparam value_type_ Type of elements stored.
 *  @tparam comparator_type_ Comparator defining ordering.
 *  @tparam node_allocator_type_ Allocator for node allocation.
 *
 *  @section basic_wb_tree_order_statistics_api Order Statistics API
 *
 *  Beyond standard tree operations, provides:
 *  - @c select(k): Find k-th smallest element in O(log n)
 *  - @c rank(x): Find position of element x in O(log n)
 *
 *  Use cases: pagination, percentiles, quantiles, window functions.
 */
template <typename value_type_, typename comparator_type_ = less_t,
          typename node_allocator_type_ = std::allocator<basic_wb_node<value_type_, comparator_type_>>,
          typename augmentation_type_ = no_augmentation_t>
class basic_wb_tree {
  public:
    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility
    using comparator_t = comparator_type_;
    using comparator_type = comparator_t; // ? STL compatibility
    using augmentation_t = augmentation_type_;
    using node_t = basic_wb_node<value_t, comparator_t, augmentation_t>;
    using node_type = node_t;
    using node_allocator_t = node_allocator_type_;
    using wb_tree_t = basic_wb_tree;
    using size_t = std::size_t;
    using allocator_t = node_allocator_type_;
    using allocator_type = allocator_t; // ? STL compatibility

    // SFINAE to extract key_type for maps, or use value_type for sets.
    using key_t = typename mapping_key_type_or_itself<value_t>::type;
    using key_type = key_t; // ? STL compatibility

    // SFINAE to extract mapped_type for maps, or void for sets.
    using mapped_t = typename mapped_value_type_or_void<value_t>::type;
    using mapped_type = mapped_t; // ? STL compatibility

    // Trait to indicate this container uses iterator-based reads (not callbacks)
    using callback_reads = std::false_type;

    using is_associative = std::bool_constant<is_mapping<value_t>>;

    /**
     *  @brief Rebind this tree type to different fruit and comparator types.
     *    Follows STL allocator rebind pattern for type transformations.
     *
     *  @tparam other_value_type_ New fruit type for the rebound tree.
     *  @tparam other_comparator_ New comparator type for the rebound tree.
     */
    template <typename other_value_type_, typename other_comparator_>
    using rebind = basic_wb_tree<other_value_type_, other_comparator_,
                                 typename std::allocator_traits<node_allocator_type_>::template rebind_alloc<
                                     basic_wb_node<other_value_type_, other_comparator_, augmentation_type_>>,
                                 augmentation_type_>;

  private:
    node_t *root_ = nullptr;
    size_t size_ = 0;
    ST_NO_UNIQUE_ADDRESS_ comparator_t comparator_;
    ST_NO_UNIQUE_ADDRESS_ node_allocator_t allocator_;

  public:
    /**
     *  @brief Default constructor. Creates empty tree.
     */
    basic_wb_tree() noexcept = default;

    explicit basic_wb_tree(allocator_t allocator) noexcept : allocator_(std::move(allocator)) {}

    /** @brief Seeds both policies, which a stateful comparator needs to survive a rebind. */
    basic_wb_tree(comparator_t comparator, allocator_t allocator) noexcept
        : comparator_(std::move(comparator)), allocator_(std::move(allocator)) {}

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

#pragma endregion Split and Join Operations

#pragma region Capacity

    /**
     *  @brief Returns the number of elements in the tree.
     */
    size_t size() const noexcept { return size_; }

    /**
     *  @brief Checks whether the tree is empty.
     */
    bool empty() const noexcept { return size_ == 0; }

#pragma endregion Capacity

#pragma region Modifiers

    /**
     *  @brief Inserts an element if key doesn't exist.
     *  @param[in] fruit Entry to insert (moved into the tree).
     *  @return Pair of pointer to node and bool indicating success.
     *    Returns {node, true} if inserted successfully.
     *    Returns {node, false} if key already exists.
     *    Returns {nullptr, false} if allocation failed.
     */
    std::pair<node_t *, bool> insert(value_t &&fruit) noexcept {
        // Probing first keeps @p fruit intact when the key is already present - the rebalancing
        // descent below only moves it once the allocation has succeeded.
        if (node_t *existing = node_t::find(root_, fruit, comparator_)) return {existing, false};
        auto result =
            node_t::upsert(root_, std::move(fruit), comparator_, [&]() noexcept { return allocator_.allocate(1); });
        root_ = result.root;
        size_ += result.placement == node_t::node_placement_t::made_k;
        return {result.match, result.placement == node_t::node_placement_t::made_k};
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
     *  @brief Atomically inserts or updates an fruit. Always succeeds (unless OOM).
     *    Overwrites existing fruit if key exists. Matches @c std::map::insert_or_assign() semantics.
     *
     *  @param[in] fruit Entry to insert or assign (moved into the tree).
     *  @return The node the entry now lives in, and whether it was made or matched.
     */
    template <typename comparable_type_>
    upserted_node_t insert_or_assign(comparable_type_ &&comparable) noexcept {
        auto result = node_t::upsert(root_, std::forward<comparable_type_>(comparable), comparator_,
                                     [&]() noexcept { return allocator_.allocate(1); });
        root_ = result.root;
        size_ += result.placement == node_t::node_placement_t::made_k;
        return {result.match, result.placement};
    }

    /**
     *  @brief Alias for @c insert_or_assign(). Atomically inserts or updates an fruit.
     *  @param[in] fruit Entry to insert or assign (moved into the tree).
     *  @return The node the entry now lives in, and whether it was made or matched.
     */
    template <typename comparable_type_>
    upserted_node_t upsert(comparable_type_ &&comparable) noexcept {
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

#pragma endregion Modifiers

#pragma region Lookup

    /** @brief Existence check, heterogeneous when the comparator declares @c is_transparent. */
    template <typename comparable_type_ = value_t>
    [[nodiscard]] bool contains(comparable_type_ &&comparable) const noexcept {
        return find(std::forward<comparable_type_>(comparable)) != end();
    }

    /** @brief Number of elements matching @p comparable, which is 0 or 1 for unique keys. */
    template <typename comparable_type_ = value_t>
    [[nodiscard]] std::size_t count(comparable_type_ &&comparable) const noexcept {
        return contains(std::forward<comparable_type_>(comparable)) ? 1u : 0u;
    }

#pragma endregion Lookup

#pragma region Order Statistics

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
     *      std::cout << "Median: " << median_node->fruit << std::endl;
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
     *  @param[in] fruit Entry to find rank of.
     *  @return Number of elements < fruit. If element exists, this is its 0-based index.
     *    Returns size() if element is greater than all elements in tree.
     *
     *  @par Complexity
     *  O(log n) expected.
     *
     *  @par Example
     *  @code
     *  auto position = tree.rank(42);
     *  // position elements are smaller than 42
     *  @endcode
     */
    size_t rank(value_t const &fruit) const noexcept { return node_t::rank(root_, fruit, comparator_); }

    /**
     *  @brief Number of entries the augmentation policy counts across the whole tree.
     *    Exact for the counts as they stand now. One scalar per node cannot encode a function of a
     *    parameter, so a reader whose predicate differs from the one currently materialized in the entries
     *    - an older snapshot, say - gets no answer from this count and none of the descents built on it.
     */
    [[nodiscard]] size_t augmented_size() const noexcept { return node_t::get_augmented_size(root_); }

    /**
     *  @brief Select the k-th smallest entry among those the augmentation policy counts, in O(log n).
     *  @return Pointer to the node, or nullptr when @c augmented_size() is at most @p k.
     */
    node_t *select_augmented(size_t k) noexcept { return node_t::select_augmented(root_, k); }

    /** @brief Select over the augmented count, const version. */
    node_t const *select_augmented(size_t k) const noexcept { return node_t::select_augmented(root_, k); }

    /** @brief Number of counted entries ordered before @p comparable, in O(log n). */
    template <typename comparable_type_>
    size_t rank_augmented(comparable_type_ &&comparable) const noexcept {
        return node_t::rank_augmented(root_, std::forward<comparable_type_>(comparable), comparator_);
    }

    /**
     *  @brief Tells the tree that the entry matching @p comparable changed its augmented count.
     *    Repairs the root-to-entry path in O(log n) rather than rescanning. A wrapper calls this once per
     *    retagged entry, which may be several per write when one publish supersedes another entry.
     *  @return True when the entry was found.
     */
    template <typename comparable_type_>
    bool refresh_augmentation(comparable_type_ &&comparable) noexcept {
        return node_t::refresh_augmentation(root_, std::forward<comparable_type_>(comparable), comparator_);
    }

    /**
     *  @brief Iterates over all entries in sorted order.
     *  @param[in] callback Callback to invoke for each fruit. Must be @c noexcept.
     */
    template <typename callback_type_>
    void for_each(callback_type_ &&callback) noexcept {
        node_t::for_each_left_right(root_, [&](node_t *node) noexcept { callback(node->fruit); });
    }

    /** @brief Sum of absolute subtree-size differences, used by the tests to watch balance quality. */
    [[nodiscard]] std::size_t total_imbalance() const noexcept {
        std::size_t total = 0;
        node_t::for_each_left_right(root_, [&](node_t *node) noexcept {
            auto const left = node_t::get_size(node->left);
            auto const right = node_t::get_size(node->right);
            total += left > right ? left - right : right - left;
        });
        return total;
    }

    /** @brief Drops every element satisfying @p predicate and reports how many went. */
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
        auto const removed = size_ - result.count;
        size_ = result.count;
        return removed;
    }

    /**
     *  @brief Erases the half-open range [lower, upper), invoking @p callback for each element.
     *    Splitting twice and re-joining keeps this O(log N + K) rather than K separate erases.
     */
    template <typename lower_type_, typename upper_type_, typename callback_type_ = no_op_t>
    void erase_range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback = {}) noexcept {
        if (!root_) return;

        auto below = node_t::split(root_, std::forward<lower_type_>(lower), comparator_);
        auto within = node_t::split(below.right, std::forward<upper_type_>(upper), comparator_);

        std::size_t deleted = 0;
        node_t::for_each_bottom_up(within.left, [&](node_t *node) noexcept {
            callback(node->fruit);
            node->fruit.~value_t();
            allocator_.deallocate(node, 1);
            ++deleted;
        });

        root_ = node_t::join(below.left, within.right, comparator_);
        size_ -= deleted;
    }

#pragma endregion Order Statistics

#pragma region Extraction and Merging

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
            if (!node_ptr_) return;
            node_ptr_->fruit.~value_t();
            tree_->allocator_.deallocate(node_ptr_, 1);
        }
        extract_result_t(extract_result_t const &) = delete;
        extract_result_t &operator=(extract_result_t const &) = delete;
        extract_result_t(extract_result_t &&other) noexcept
            : tree_(other.tree_), node_ptr_(std::exchange(other.node_ptr_, nullptr)) {}
        extract_result_t &operator=(extract_result_t &&other) noexcept {
            if (this != &other) {
                if (node_ptr_) {
                    node_ptr_->fruit.~value_t();
                    tree_->allocator_.deallocate(node_ptr_, 1);
                }
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
     *  @return True if at least one key exists in both trees, false otherwise.
     */
    bool has_any_key(wb_tree_t const &other) const noexcept {
        auto it1 = begin();
        auto it2 = other.begin();
        while (it1 != end() && it2 != other.end()) {
            if (comparator_(mapping_key_or_itself(*it1), mapping_key_or_itself(*it2))) ++it1;
            else if (comparator_(mapping_key_or_itself(*it2), mapping_key_or_itself(*it1))) ++it2;
            else return true; // Found duplicate
        }
        return false;
    }

    /**
     *  @brief Checks if all keys from other tree exist in this tree using O(m+n) simultaneous traversal.
     *  @param[in] other Tree whose keys to check.
     *  @return True if all keys from other exist in this tree, false otherwise.
     */
    bool has_all_keys(wb_tree_t const &other) const noexcept {
        auto it1 = begin();
        auto it2 = other.begin();
        while (it1 != end() && it2 != other.end()) {
            if (comparator_(mapping_key_or_itself(*it1), mapping_key_or_itself(*it2))) ++it1;
            else if (comparator_(mapping_key_or_itself(*it2), mapping_key_or_itself(*it1)))
                return false; // Key in other not found in this
            else {
                ++it1;
                ++it2;
            }
        }
        return it2 == other.end(); // All keys from other were found
    }

    /**
     *  @brief Merges another tree using upsert semantics (updates duplicates instead of skipping).
     *  @param[inout] other Tree to merge from. Emptied on success, and on an allocation failure
     *    it keeps every entry that could not be moved across.
     */
    void merge_with_upsert(wb_tree_t &other) noexcept {
        while (other.size() > 0) {
            node_t *other_root = other.root_;
            if (!other_root) break;

            // The extracted node only carries its entry across; the guard frees the node itself
            // once the entry has been upserted into this tree.
            auto extracted = other.extract(other_root->fruit);
            if (!extracted) continue;

            auto result = node_t::upsert(root_, std::move(extracted.node_ptr_->fruit), comparator_,
                                         [&]() noexcept { return allocator_.allocate(1); });
            // Out of memory - hand the node back to the source tree rather than drop its entry
            if (result.failed()) {
                other.merge(std::move(extracted));
                return;
            }
            root_ = result.root;
            size_ += result.placement == node_t::node_placement_t::made_k;
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
        size_ += result.placement == node_t::node_placement_t::made_k;
        // Key conflict - node wasn't inserted, so release the entry it carried
        if (result.placement == node_t::node_placement_t::matched_k) {
            node_to_insert->fruit.~value_t();
            allocator_.deallocate(node_to_insert, 1);
        }
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

            auto extracted = other.extract(other_root->fruit);
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

        if (comparator_(mapping_key_or_itself(this_max->fruit), mapping_key_or_itself(other_min->fruit))) {
            root_ = node_t::join(root_, other.root_, comparator_);
            size_ += other.size_;
            other.root_ = nullptr;
            other.size_ = 0;
            return;
        }

        // Fallback: Use regular merge (one-by-one insertion)
        merge(other);
    }

#pragma endregion Extraction and Merging

#pragma region Iterator Support

    class iterator;
    class const_iterator;

    /** @brief Visits every element in [lower, upper) in sorted order. */
    template <typename lower_type_ = value_t, typename upper_type_ = value_t, typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        node_t::range(root_, std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), comparator_,
                      [&](node_t *node) noexcept { callback(node->fruit); });
    }

    /** @brief Same range walk, but the callback may modify each element in place. */
    template <typename lower_type_ = value_t, typename upper_type_ = value_t, typename callback_type_ = no_op_t>
    void range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        node_t::range(root_, std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), comparator_,
                      [&](node_t *node) noexcept { callback(node->fruit); });
    }

    /** @brief Copies out the element equal to @p comparable, or reports @c key_not_found_k. */
    template <typename comparable_type_>
    [[nodiscard]] expected<value_t> find_copy(comparable_type_ &&comparable) const noexcept {
        auto iterator = find(std::forward<comparable_type_>(comparable));
        if (iterator == end()) return key_not_found_k;
        return copy_safely(*iterator);
    }

    /** @brief Copies out the first element not less than @p comparable. */
    template <typename comparable_type_>
    [[nodiscard]] expected<value_t> lower_bound_copy(comparable_type_ &&comparable) const noexcept {
        auto iterator = lower_bound(std::forward<comparable_type_>(comparable));
        if (iterator == end()) return key_not_found_k;
        return copy_safely(*iterator);
    }

    /** @brief Copies out the first element greater than @p comparable. */
    template <typename comparable_type_>
    [[nodiscard]] expected<value_t> upper_bound_copy(comparable_type_ &&comparable) const noexcept {
        auto iterator = upper_bound(std::forward<comparable_type_>(comparable));
        if (iterator == end()) return key_not_found_k;
        return copy_safely(*iterator);
    }

    /** @brief Inserts one element only when its key is absent, leaving any existing element alone. */
    template <typename comparable_type_>
    std::pair<iterator, bool> insert_if_missing(comparable_type_ &&comparable) noexcept {
        if (auto existing = find(comparable); existing != end()) return {existing, false};
        auto result = upsert(std::forward<comparable_type_>(comparable));
        return {iterator {this, result.node}, result.node != nullptr};
    }

    /** @brief Inserts a range, skipping keys that are already present. */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t insert_if_missing(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        for (; first != last; ++first) {
            value_t element(*first);
            if (find(element) != end()) continue;
            if (upsert(std::move(element)).node == nullptr) return status_t::out_of_memory_heap_k;
        }
        return success_k;
    }

    /** @brief Upserts a range, overwriting any key already present. */
    template <typename input_iterator_type_>
    [[nodiscard]] status_t upsert(input_iterator_type_ first, input_iterator_type_ last) noexcept {
        for (; first != last; ++first)
            if (upsert(value_t(*first)).node == nullptr) return status_t::out_of_memory_heap_k;
        return success_k;
    }

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
     *  @brief Bidirectional iterator for WB tree.
     *    Provides in-order traversal of tree elements.
     */
    class iterator {
        friend class basic_wb_tree;
        friend class const_iterator;

      public:
        using value_type = value_t;
        using difference_type = std::ptrdiff_t;
        using pointer = value_t *;
        using reference = value_t &;

      private:
        basic_wb_tree const *tree_;
        node_t *node_;

        iterator(basic_wb_tree const *tree, node_t *node) noexcept : tree_(tree), node_(node) {}

      public:
        iterator() noexcept : tree_(nullptr), node_(nullptr) {}

        reference operator*() const noexcept { return node_->fruit; }
        pointer operator->() const noexcept { return &node_->fruit; }

        iterator &operator++() noexcept {
            node_ = node_t::find_successor(tree_->root_, node_, tree_->comparator_);
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator previous = *this;
            ++(*this);
            return previous;
        }

        iterator &operator--() noexcept {
            node_ = node_t::find_predecessor(tree_->root_, node_, tree_->comparator_);
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
     *  @brief Const bidirectional iterator for WB tree.
     *    Provides in-order traversal of tree elements (read-only).
     */
    class const_iterator {
        friend class basic_wb_tree;

      public:
        using value_type = value_t const;
        using difference_type = std::ptrdiff_t;
        using pointer = value_t const *;
        using reference = value_t const &;

      private:
        basic_wb_tree const *tree_;
        node_t const *node_;

        const_iterator(basic_wb_tree const *tree, node_t const *node) noexcept : tree_(tree), node_(node) {}

      public:
        const_iterator() noexcept : tree_(nullptr), node_(nullptr) {}
        const_iterator(iterator const &it) noexcept : tree_(it.tree_), node_(it.node_) {}

        reference operator*() const noexcept { return node_->fruit; }
        pointer operator->() const noexcept { return &node_->fruit; }

        const_iterator &operator++() noexcept {
            node_ = node_t::find_successor(tree_->root_, const_cast<node_t *>(node_), tree_->comparator_);
            return *this;
        }

        const_iterator operator++(int) noexcept {
            const_iterator previous = *this;
            ++(*this);
            return previous;
        }

        const_iterator &operator--() noexcept {
            node_ = node_t::find_predecessor(tree_->root_, const_cast<node_t *>(node_), tree_->comparator_);
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
     *  @brief Finds the first element not less than the given @p comparable (lower bound).
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Iterator to lower bound element, or end() if all elements < comparable.
     */
    template <typename comparable_type_>
    iterator lower_bound(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::lower_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element not less than the given @p comparable (const version).
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Const iterator to lower bound element, or end() if all elements < comparable.
     */
    template <typename comparable_type_>
    const_iterator lower_bound(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this,
                              node_t::lower_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element greater than the given @p comparable (upper bound).
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Iterator to upper bound element, or end() if all elements <= comparable.
     */
    template <typename comparable_type_>
    iterator upper_bound(comparable_type_ &&comparable) noexcept {
        return iterator(this, node_t::upper_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Finds the first element greater than the given @p comparable (const version).
     *
     *  @param[in] comparable Object comparable to @c value_t.
     *  @return Const iterator to upper bound element, or end() if all elements <= comparable.
     */
    template <typename comparable_type_>
    const_iterator upper_bound(comparable_type_ &&comparable) const noexcept {
        return const_iterator(this,
                              node_t::upper_bound(root_, std::forward<comparable_type_>(comparable), comparator_));
    }

    /**
     *  @brief Erases a single fruit matching the given @p comparable. No callbacks.
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
        auto next = position;
        ++next;
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
        return erase(iterator(const_cast<basic_wb_tree *>(this), const_cast<node_t *>(position.node_)));
    }

    /**
     *  @brief Erases all elements in the range [first, last).
     *    Unlike STL, returns both the iterator following the last erased element and a status code.
     *    On error, some elements may have been erased (partial erase, matches STL's basic guarantee).
     *
     *  @param[in] first Iterator to first element in range.
     *  @param[in] last Iterator to past-the-end of range.
     *  @return Contains iterator following the last erased element and operation status.
     *
     *  @note If first == last, no elements are erased and returns {last, success}.
     */
    erase_result_t erase(iterator first, iterator last) noexcept {
        while (first != last) {
            auto result = erase(first);
            if (failed(result.status)) return result;
            first = result.next;
        }
        return {last, {success_k}};
    }

#pragma endregion Iterator Support

#pragma region Split and Join Operations

    /**
     *  @brief Result of splitting a tree at a key.
     *    Contains two trees: left (all < key) and right (all >= key).
     */
    struct split_result_t {
        /** @brief Tree with all elements ordered before the split key. */
        wb_tree_t left;
        /** @brief Tree with all elements not ordered before the split key. */
        wb_tree_t right;

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

        // Both halves inherit this tree's allocator, since they own nodes it allocated.
        wb_tree_t left_tree(comparator_, allocator_);
        left_tree.root_ = result.left;
        left_tree.size_ = node_t::get_size(result.left);

        wb_tree_t right_tree(comparator_, allocator_);
        right_tree.root_ = result.right;
        right_tree.size_ = node_t::get_size(result.right);

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
        size_ += other.size_;

        other.root_ = nullptr;
        other.size_ = 0;
    }

#pragma endregion Split and Join Operations

#pragma region Observers

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
        node->fruit.~value_t();
        allocator_.deallocate(node, 1);
    }
};

template <typename value_type_, typename comparator_type_ = less_t, typename allocator_type_ = std::allocator<void>,
          typename augmentation_type_ = no_augmentation_t>
using wb_set = basic_wb_tree<value_type_, comparator_type_,
                             typename std::allocator_traits<allocator_type_>::template rebind_alloc<
                                 basic_wb_node<value_type_, comparator_type_, augmentation_type_>>,
                             augmentation_type_>;

template <typename key_type_, typename value_type_, typename comparator_type_ = less_t,
          typename allocator_type_ = std::allocator<void>, typename augmentation_type_ = no_augmentation_t>
using wb_map = basic_wb_tree<mapping<key_type_, value_type_>, comparator_type_,
                             typename std::allocator_traits<allocator_type_>::template rebind_alloc<
                                 basic_wb_node<mapping<key_type_, value_type_>, comparator_type_, augmentation_type_>>,
                             augmentation_type_>;

#pragma endregion Observers

} // namespace ashvardanian::smashtable
