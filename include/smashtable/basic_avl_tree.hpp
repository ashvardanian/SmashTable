/**
 *  @brief  Ordered "Adelson-Velsky and Landis" @b (AVL) Binary Search Tree implementation.
 *          Not thread-safe by itself. Doesn't raise any exceptions unlike STL-based alternatives.
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
 *  @sa     basic_avl_tree
 *
 *  AVL-Trees are some of the simplest yet performant Binary Search Trees.
 *  This "node" class implements the primary logic, but doesn't take part in
 *  memory management or any atomicity and consistency guarantees.
 *
 *  > Never throws! Even if new node allocation had failed.
 *  > Implements `lower_bound` and `upper_bound` for faster and lighter iterators.
 *  > Implements random sampling methods.
 *
 *  @tparam entry_type_         Type of entries to store in this tree.
 *  @tparam comparator_type_    A comparator function object, that overload
 *                           @code
 *                               bool operator ()(entry_type_, entry_type_) const
 *                           @endcode
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
     *  @brief Root has the biggest `height` in the tree.
     *  Zero is possible only in the uninitialized detached state.
     *  A non-NULL node would have height of one.
     *  Allows you to guess the upper bound of branch size, as `1 << height`.
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
     *  @brief Searches for equal entry in this subtree.
     *  @param comparable Any key comparable with stored entries.
     *  @return NULL if nothing was found.
     */
    template <typename comparable_type_>
    static node_t *find(node_t *node, comparable_type_ &&comparable) noexcept {
        auto less = comparator_t {};
        while (node) {
            if (less(comparable, node->entry)) node = node->left;
            else if (less(node->entry, comparable))
                node = node->right;
            else
                break;
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
            else if (less(node->entry, comparable)) { node = node->right; }

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
        else if (less(node->entry, a) && less(node->entry, b)) { return lowest_common_ancestor(node->right, a, b); }

        else { return node; }
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

        else if (less(node->entry, low))
            return range(node->right, low, high, callback);

        else
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

template <typename entry_type_, typename comparator_type_,
          typename node_allocator_type_ = std::allocator<basic_avl_node<entry_type_, comparator_type_>>>
class basic_avl_tree {
  public:
    using node_t = basic_avl_node<entry_type_, comparator_type_>;
    using node_allocator_t = node_allocator_type_;
    using comparator_t = comparator_type_;
    using entry_t = entry_type_;
    using avl_tree_t = basic_avl_tree;

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
    std::size_t size() const noexcept { return size_; }
    std::size_t height() noexcept { return root_ ? root_->height : 0; }
    node_t *root() const noexcept { return root_; }
    node_t *end() const noexcept { return nullptr; }
    node_allocator_t &allocator() noexcept { return allocator_; }
    node_allocator_t const &allocator() const noexcept { return allocator_; }

    std::size_t total_imbalance() const noexcept {
        std::size_t abs_sum = 0;
        node_t::for_each_top_down(root_,
                                  [&](node_t *node) noexcept { abs_sum += std::abs(node_t::get_balance(node)); });
        return abs_sum;
    }

    template <typename comparable_type_>
    node_t *find(comparable_type_ &&comparable) noexcept {
        return node_t::find(root_, std::forward<comparable_type_>(comparable));
    }

    template <typename comparable_type_>
    node_t *lower_bound(comparable_type_ &&comparable) noexcept {
        return node_t::lower_bound(root_, std::forward<comparable_type_>(comparable));
    }

    template <typename comparable_type_>
    node_t *upper_bound(comparable_type_ &&comparable) noexcept {
        return node_t::upper_bound(root_, std::forward<comparable_type_>(comparable));
    }

    template <typename comparable_type_>
    node_t const *find(comparable_type_ &&comparable) const noexcept {
        return node_t::find(root_, std::forward<comparable_type_>(comparable));
    }

    template <typename comparable_type_>
    node_t const *lower_bound(comparable_type_ &&comparable) const noexcept {
        return node_t::lower_bound(root_, std::forward<comparable_type_>(comparable));
    }

    template <typename comparable_type_>
    node_t const *upper_bound(comparable_type_ &&comparable) const noexcept {
        return node_t::upper_bound(root_, std::forward<comparable_type_>(comparable));
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

    template <typename comparable_type_>
    upsert_result_t insert(comparable_type_ &&comparable) noexcept {
        auto result = node_t::insert(root_, std::forward<comparable_type_>(comparable),
                                     [&]() noexcept { return allocator_.allocate(1); });
        root_ = result.root;
        size_ += result.inserted;
        return {result.match, result.inserted};
    }

    template <typename comparable_type_>
    upsert_result_t upsert(comparable_type_ &&comparable) noexcept {
        auto result = node_t::upsert(root_, std::forward<comparable_type_>(comparable),
                                     [&]() noexcept { return allocator_.allocate(1); });
        root_ = result.root;
        size_ += result.inserted;
        return {result.match, result.inserted};
    }

    struct extract_result_t {
        basic_avl_tree *tree_ = nullptr;
        node_t *node_ptr_ = nullptr;

        ~extract_result_t() noexcept {
            if (node_ptr_) tree_->allocator_.deallocate(node_ptr_, 1);
        }
        extract_result_t(extract_result_t const &) = delete;
        extract_result_t &operator=(extract_result_t const &) = delete;
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

    template <typename comparable_type_>
    bool erase(comparable_type_ &&comparable) noexcept {
        return !!extract(std::forward<comparable_type_>(comparable));
    }

    void clear() noexcept {
        node_t::for_each_bottom_up(root_, [&](node_t *node) noexcept { return allocator_.deallocate(node, 1); });
        root_ = nullptr;
        size_ = 0;
    }

    template <typename callback_type_>
    void for_each(callback_type_ &&callback) noexcept {
        node_t::for_each_bottom_up(root_, [&](node_t *node) noexcept { callback(node->entry); });
    }

    void merge(avl_tree_t &other) noexcept {
        node_t::for_each_bottom_up(other.root_, [&](node_t *node) noexcept {
            auto result = node_t::insert(root_, node);
            root_ = result.root;
            size_ += result.inserted;
        });
        other.root_ = nullptr;
        other.size_ = 0;
    }

    void merge(extract_result_t other) noexcept {
        if (!other.node_ptr_) return;
        auto result = node_t::insert(root_, other.release());
        root_ = result.root;
        size_ += result.inserted;
    }
};

} // namespace ashvardanian::smashtable