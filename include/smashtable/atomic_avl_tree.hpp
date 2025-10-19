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
 *  @brief  Exception-free dynamic array that uses error codes instead of exceptions.
 *          Provides RAII memory management with explicit failure handling.
 */
template <typename element_type_, typename allocator_type_ = std::allocator<element_type_>>
class noexcept_vector {
  public:
    using element_t = element_type_;
    using allocator_t = allocator_type_;

  private:
    element_t *data_ {nullptr};
    std::size_t size_ {0};
    std::size_t capacity_ {0};
    [[no_unique_address]] allocator_t allocator_ {};

  public:
    noexcept_vector() noexcept = default;

    ~noexcept_vector() noexcept {
        if (data_) allocator_.deallocate(data_, capacity_);
    }

    noexcept_vector(noexcept_vector &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
          capacity_(std::exchange(other.capacity_, 0)), allocator_(std::move(other.allocator_)) {}

    noexcept_vector &operator=(noexcept_vector &&other) noexcept {
        if (this != &other) {
            if (data_) allocator_.deallocate(data_, capacity_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
            capacity_ = std::exchange(other.capacity_, 0);
            allocator_ = std::move(other.allocator_);
        }
        return *this;
    }

    noexcept_vector(noexcept_vector const &) = delete;
    noexcept_vector &operator=(noexcept_vector const &) = delete;

    [[nodiscard]] status_t try_reserve(std::size_t new_capacity) noexcept {
        if (new_capacity <= capacity_) return {success_k};

        auto new_data = allocator_.allocate(new_capacity);
        if (!new_data) return {out_of_memory_heap_k};

        // Move existing elements
        for (std::size_t i = 0; i < size_; ++i) new (&new_data[i]) element_t(std::move(data_[i]));

        if (data_) allocator_.deallocate(data_, capacity_);
        data_ = new_data;
        capacity_ = new_capacity;
        return {success_k};
    }

    [[nodiscard]] status_t try_push_back(element_t &&value) noexcept {
        // Auto-grow if needed (2x growth strategy)
        if (size_ >= capacity_) {
            std::size_t new_capacity = capacity_ == 0 ? 4 : capacity_ * 2;
            auto status = try_reserve(new_capacity);
            if (!status) return status;
        }
        new (&data_[size_++]) element_t(std::move(value));
        return {success_k};
    }

    void clear() noexcept { size_ = 0; }
    std::size_t size() const noexcept { return size_; }
    std::size_t capacity() const noexcept { return capacity_; }

    element_t *begin() noexcept { return data_; }
    element_t *end() noexcept { return data_ + size_; }
    element_t const *begin() const noexcept { return data_; }
    element_t const *end() const noexcept { return data_ + size_; }
};

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

/**
 *  @brief  Atomic Binary Search Tree extending AVL-Tree with Versioning.
 *          Provides 2-phase commits for "transactions", with "watching" capabilities for
 *          CAS-like (Compare-And-Swap) operations. Not thread-safe by itself.
 *          Doesn't use any locks or mutexes internally. Doesn't raise any exceptions
 *          unlike STL-based alternatives.
 *
 *  @sa     Wraps entries into `versioned_element`s and puts them into `basic_avl_tree`.
 *
 *  @section Design Goals
 *
 *  First, all operations are atomic. If you are updating many values at once, you don't want
 *  to break in an intermediate state, where only some of the values are updated. With two-phase
 *  commit transactions, you can stage many changes, and then commit them all at once. Or rollback,
 *  if something went wrong in the current transaction or some external condition changed.
 *  A common usecase for this, is synchronizing many updates across many data stores.
 *
 *  The API must be simple, but generalizable, and the implementation must be light-weight.
 *  So this collection @b doesn't provide snapshots or multi-version concurrency control (MVCC).
 *  It means, that if you are starting a transaction, and even "watching" some values through it,
 *  there is no guarantee, that the value received hasn't been updated before transaction "began"
 *  and entry was added to the "watched" list.
 *
 *  Only "Monotonic Atomic View" consistency is guaranteed, including its inferior "Read Committed"
 *  and "Read Uncommitted" levels. In other words, transactions are not allowed to observe writes
 *  from other transactions which do not commit.
 *
 *  @see https://jepsen.io/consistency/models/monotonic-atomic-view
 *  @see https://jepsen.io/consistency/models/read-committed
 *
 *  @section API Overview
 *
 *  - All lookups are heterogeneous, meaning you can provide any type, that is comparable to the
 *    @p element_type_. This allows for greater flexibility in how you interact with the set.
 *  - No iterators are provided, to keep the implementation simple and avoid the complexity of
 *    maintaining persistent iterator validity across transactions and modifications.
 *
 *  @tparam element_type_ Type of the elements stored in the set.
 *  @tparam comparator_type_ Ideally heterogeneous comparator for @c element_type_.
 *  @tparam allocator_type_ Arbitrary "rebindable" allocator for all internal structures.
 */
template < //
    typename element_type_, typename comparator_type_ = std::less<element_type_>,
    typename allocator_type_ = std::allocator<std::uint8_t>>
class atomic_avl_tree {

  public:
    using element_t = element_type_;
    using comparator_t = comparator_type_;
    using allocator_t = allocator_type_;

    using versioning_t = versioned_element<element_t, comparator_t>;
    using identifier_t = typename versioning_t::identifier_t;
    using generation_t = typename versioning_t::generation_t;
    using dated_identifier_t = typename versioning_t::dated_identifier_t;
    using watch_t = typename versioning_t::watch_t;
    using watched_identifier_t = typename versioning_t::watched_identifier_t;
    using entry_t = typename versioning_t::entry_t;
    using entry_comparator_t = typename versioning_t::entry_comparator_t;

  private:
    using entry_node_t = basic_avl_node<entry_t, entry_comparator_t>;
    using entry_allocator_t = typename allocator_t::template rebind<entry_node_t>::other;
    using entry_set_t = basic_avl_tree<entry_t, entry_comparator_t, entry_allocator_t>;
    using entry_iterator_t = entry_node_t *;

    using watches_allocator_t = typename allocator_t::template rebind<watched_identifier_t>::other;
    using watches_vector_t = noexcept_vector<watched_identifier_t, watches_allocator_t>;

    using store_t = atomic_avl_tree;
    using extract_result_t = typename entry_set_t::extract_result_t;

  public:
    class transaction_t {

        friend store_t;
        enum class stage_t {
            created_k,
            staged_k,
            commited_k,
        };

        store_t *store_ {nullptr};
        entry_set_t changes_ {};
        watches_vector_t watches_ {};
        generation_t generation_ {0};
        stage_t stage_ {stage_t::created_k};
        bool is_snapshot_ {false};

        transaction_t(store_t &set) noexcept : store_(&set), generation_(set.new_generation()) {}
        watch_t missing_watch() const noexcept { return watch_t {generation_, true}; }
        store_t &store_ref() noexcept { return *store_; }
        store_t const &store_ref() const noexcept { return *store_; }

      public:
        transaction_t(transaction_t &&) noexcept = default;
        transaction_t &operator=(transaction_t &&) noexcept = default;
        transaction_t(transaction_t const &) = delete;
        transaction_t &operator=(transaction_t const &) = delete;
        generation_t generation() const noexcept { return generation_; }

      public:
        [[nodiscard]] status_t upsert(element_t &&element) noexcept {
            entry_t entry;
            entry.element = std::move(element);
            entry.generation = generation_;
            entry.deleted = false;
            entry.visible = false;
            auto result = changes_.upsert(std::move(entry));
            return result.failed() ? status_t {out_of_memory_heap_k} : status_t {success_k};
        }

        [[nodiscard]] status_t erase(identifier_t const &id) noexcept {
            entry_t entry;
            entry.element = id;
            entry.generation = generation_;
            entry.deleted = true;
            entry.visible = false;
            auto result = changes_.upsert(std::move(entry));
            return result.failed() ? status_t {out_of_memory_heap_k} : status_t {success_k};
        }

        [[nodiscard]] status_t reserve(std::size_t size) noexcept { return watches_.try_reserve(size); }

        [[nodiscard]] status_t watch(identifier_t const &id) noexcept {
            status_t result {success_k};
            auto found = [&](entry_t const &entry) noexcept {
                result =
                    watches_.try_push_back({identifier_t {entry.element}, watch_t {entry.generation, entry.deleted}});
            };
            auto missing = [&]() noexcept { result = watches_.try_push_back({id, missing_watch()}); };
            auto status = store_ref().find(id, found, missing);
            return status ? result : status;
        }

        [[nodiscard]] status_t watch(entry_t const &entry) noexcept {
            return watches_.try_push_back({identifier_t {entry.element}, watch_t {entry.generation, entry.deleted}});
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                    callback_missing_type_ &&callback_missing = {}) const noexcept {
            if (auto iterator = changes_.find(std::forward<comparable_type_>(comparable)); iterator != changes_.end()) {
                !iterator->entry.deleted ? callback_found(iterator->entry) : callback_missing();
                return {success_k};
            }
            else
                return store_ref().find(std::forward<comparable_type_>(comparable),
                                        std::forward<callback_found_type_>(callback_found),
                                        std::forward<callback_missing_type_>(callback_missing));
        }

        template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
                  typename callback_missing_type_ = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                           callback_missing_type_ &&callback_missing = {}) const noexcept {
            auto external_previous_id = identifier_t(comparable);
            auto internal_iterator = changes_.upper_bound(std::forward<comparable_type_>(comparable));
            while (internal_iterator != changes_.end() && internal_iterator->entry.deleted) ++internal_iterator;

            // Once picking the next smallest element from the global store,
            // we might face an entry, that was already deleted from here,
            // so this might become a multi-step process.
            auto faced_deleted_entry = false;
            auto callback_external_found = [&](element_t const &external_element) {
                // The simplest case is when we have an external object.
                if (internal_iterator == changes_.end()) return callback_found(external_element);

                element_t const &internal_element = internal_iterator->entry;
                if (!entry_comparator_t {}(external_element, internal_element)) return callback_found(internal_element);

                // Check if this entry was deleted and we should try again.
                auto external_id = identifier_t(external_element);
                auto external_element_internal_state = changes_.find(external_element);
                if (external_element_internal_state != changes_.end() &&
                    external_element_internal_state->entry.deleted) {
                    faced_deleted_entry = true;
                    external_previous_id = external_id;
                    return;
                }
                else { return callback_found(external_element); }
            };
            auto callback_external_missing = [&] {
                if (internal_iterator == changes_.end()) return callback_missing();
                else {
                    element_t const &internal_element = internal_iterator->entry;
                    return callback_found(internal_element);
                }
            };

            // Iterate until we find the a non-deleted external value
            auto &store = store_ref();
            auto status = status_t {};
            do {
                status = store.upper_bound(external_previous_id, callback_external_found, callback_external_missing);
            } while (faced_deleted_entry && status);
            return status;
        }

        [[nodiscard]] status_t stage() noexcept {
            // First, check if we have any collisions by validating watches.
            auto &store = store_ref();
            auto entry_missing = missing_watch();
            for (auto const &id_and_watch : watches_) {
                auto consistency_violated = false;
                auto status = store.find_latest_for_watch(
                    id_and_watch.id,
                    [&](entry_t const &entry) noexcept { consistency_violated = entry != id_and_watch.watch; },
                    [&]() noexcept { consistency_violated = entry_missing != id_and_watch.watch; });
                if (consistency_violated) return {errc_t::consistency_k};
                if (!status) return status;
            }

            // Now all of our watches will be replaced with "links" to entries
            // we are merging into the main tree.
            watches_.clear();
            auto status = watches_.try_reserve(changes_.size());
            if (!status) return status;

            // No new memory allocations or failures are possible after that.
            // It is all safe.
            changes_.for_each([&](entry_t const &entry) noexcept {
                [[maybe_unused]] auto push_status =
                    watches_.try_push_back({identifier_t {entry.element}, watch_t {generation_, entry.deleted}});
                assert(push_status && "Should never fail after reserve");
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
            auto &store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const &id_and_watch : watches_)
                    store.entries_.erase(dated_identifier_t {id_and_watch.id, id_and_watch.watch.generation});

            watches_.clear();
            changes_.clear();
            stage_ = stage_t::created_k;
            generation_ = store.new_generation();
            return {success_k};
        }

        [[nodiscard]] status_t rollback() noexcept {
            if (stage_ != stage_t::staged_k) return {operation_not_permitted_k};

            // If the transaction was "staged",
            // we must delete all the entries.
            auto &store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const &id_and_watch : watches_)
                    changes_.merge(
                        store.entries_.extract(dated_identifier_t {id_and_watch.id, id_and_watch.watch.generation}));

            watches_.clear();
            stage_ = stage_t::created_k;
            generation_ = store.new_generation();
            return {success_k};
        }

        [[nodiscard]] status_t commit() noexcept {
            if (stage_ != stage_t::staged_k) return {operation_not_permitted_k};

            // Once we make an entry visible,
            // if there are more than one with the same key,
            // the older generation must die.
            auto &store = store_ref();
            for (auto const &id_and_watch : watches_)
                store.unmask_and_compact(id_and_watch.id, id_and_watch.watch.generation);

            stage_ = stage_t::created_k;
            return {success_k};
        }
    };

  private:
    entry_set_t entries_;
    generation_t generation_ {0};
    std::size_t visible_count_ {0};
    std::size_t visible_deleted_count_ {0};

    friend class transaction_t;
    generation_t new_generation() noexcept { return ++generation_; }

    /**
     *  @brief Finds the latest (highest generation) entry for watch validation.
     *         Unlike find(), this checks ALL entries including staged (invisible) ones.
     *         This is critical for detecting write-write conflicts with concurrent transactions.
     */
    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t find_latest_for_watch(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                                 callback_missing_type_ &&callback_missing = {}) const noexcept {

        entry_node_t *latest = nullptr;
        entry_node_t::range(entries_.root(), comparable, comparable, [&](entry_node_t *node) noexcept {
            // Find HIGHEST generation, regardless of visibility
            if (!latest || node->entry.generation > latest->entry.generation) { latest = node; }
        });

        return (latest && !latest->entry.deleted)
                   ? invoke_safely([&] { callback_found(latest->entry); })
                   : invoke_safely(std::forward<callback_missing_type_>(callback_missing));
    }

    void unmask_and_compact(identifier_t const &id, generation_t generation_to_unmask) noexcept {
        // This is similar to the public `erase_range()`, but adds generation-matching conditions.
        auto current = entries_.lower_bound(id);
        if (!current) return; // Null check

        auto less = entry_comparator_t {};
        auto last_visible_entry = std::optional<dated_identifier_t> {};
        while (current && less.same(id, current->entry.element)) {
            auto next = entries_.upper_bound(current->entry);
            auto was_visible = current->entry.visible;
            current->entry.visible |= current->entry.generation == generation_to_unmask;

            // Update counters if visibility changed
            if (!was_visible && current->entry.visible) {
                ++visible_count_;
                visible_deleted_count_ += current->entry.deleted;
            }

            if (!current->entry.visible) {
                current = next;
                continue;
            }

            // Older revisions must die
            if (last_visible_entry) {
                auto to_erase = entries_.find(*last_visible_entry);
                if (to_erase != entries_.end() && to_erase->entry.visible) {
                    --visible_count_;
                    visible_deleted_count_ -= to_erase->entry.deleted;
                }
                entries_.extract(*last_visible_entry);
            }
            last_visible_entry = dated_identifier_t {id, current->entry.generation};
            current = next;
        }
    }

  public:
    atomic_avl_tree() noexcept {}
    atomic_avl_tree(atomic_avl_tree &&other) noexcept
        : entries_(std::move(other.entries_)), generation_(other.generation_), visible_count_(other.visible_count_),
          visible_deleted_count_(other.visible_deleted_count_) {}

    atomic_avl_tree &operator=(atomic_avl_tree &&other) noexcept {
        entries_ = std::move(other.entries_);
        generation_ = other.generation_;
        visible_count_ = other.visible_count_;
        visible_deleted_count_ = other.visible_deleted_count_;
        return *this;
    }

    [[nodiscard]] std::size_t size() const noexcept { return visible_count_ - visible_deleted_count_; }
    [[nodiscard]] static std::optional<store_t> make(allocator_t &&allocator = {}) noexcept { return store_t {}; }
    [[nodiscard]] std::optional<transaction_t> transaction() noexcept { return transaction_t {*this}; }

    [[nodiscard]] status_t upsert(element_t &&element) noexcept {
        auto node = entries_.allocator().allocate(1);
        if (!node) return {out_of_memory_heap_k};

        identifier_t id {element};
        generation_t generation = new_generation();
        auto &entry = node->entry;
        new (&entry.element) element_t(std::move(element));
        entry.generation = generation;
        entry.deleted = false;
        entry.visible = true;
        entries_.merge(extract_result_t {&entries_, node});
        ++visible_count_;
        assert(!entry.deleted && "entry.deleted is always false here, otherwise update visible_deleted_count_");

        return erase_range(id, dated_identifier_t {id, generation});
    }

    template <typename elements_begin_type_, typename elements_end_type_ = elements_begin_type_>
    [[nodiscard]] status_t upsert(elements_begin_type_ begin, elements_end_type_ end) noexcept {

        // To make such batch insertions cheaper and easier until we have fast joins,
        // we can build a linked-list of pre-allocated nodes. Populate them and insert
        // one-by-one with the same generation.
        std::size_t const count = end - begin;
        std::size_t count_remaining = count;
        entry_node_t *last_node = nullptr;
        while (count_remaining) {
            entry_node_t *next_node = entries_.allocator().allocate(1);
            if (!next_node) break;
            // Reset the state
            next_node->right = nullptr;
            // Link for future iteration
            if (last_node) last_node->right = next_node;
            next_node->left = last_node;
            // Update state for next loop cycle
            last_node = next_node;
            count_remaining--;
        }

        // We have failed to allocate all the needed nodes.
        if (count_remaining) {
            while (count_remaining != count) {
                entry_node_t *prev_node = last_node->left;
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
            entry_node_t *prev_node = last_node->left;
            last_node->left = nullptr;
            last_node->right = nullptr;

            auto &entry = last_node->entry;
            new (&entry.element) element_t(*begin);
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            entries_.merge(extract_result_t {&entries_, last_node});
            ++visible_count_;
            assert(!entry.deleted && "entry.deleted is always false here, otherwise update visible_deleted_count_");

            // Remove older revisions
            identifier_t id {entry.element};
            auto status = erase_range(id, dated_identifier_t {id, generation});
            if (!status) return status;

            // Update state for next loop cycle
            last_node = prev_node;
            ++count_remaining;
            ++begin;
        }

        return {success_k};
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t find(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                callback_missing_type_ &&callback_missing = {}) const noexcept {

        entry_node_t *largest_visible = nullptr;
        entry_node_t::range(entries_.root(), comparable, comparable, [&](entry_node_t *node) noexcept {
            if ((node->entry.visible) &&
                (!largest_visible || node->entry.generation > largest_visible->entry.generation))
                largest_visible = node;
        });

        // static_assert(noexcept(callback_found(largest_visible->entry)));
        // static_assert(noexcept(callback_missing()));
        largest_visible ? callback_found(largest_visible->entry) : callback_missing();
        return {success_k};
    }

    template <typename comparable_type_ = identifier_t, typename callback_found_type_ = no_op_t,
              typename callback_missing_type_ = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_type_ &&comparable, callback_found_type_ &&callback_found,
                                       callback_missing_type_ &&callback_missing = {}) const noexcept {

        // Skip all the invisible entries
        entry_node_t *next_visible = entry_node_t::upper_bound(entries_.root(), comparable);
        while (next_visible && !next_visible->entry.visible)
            next_visible = entry_node_t::upper_bound(entries_.root(), next_visible->entry);

        // static_assert(noexcept(callback_found(next_visible->entry)));
        // static_assert(noexcept(callback_missing()));
        next_visible ? callback_found(next_visible->entry) : callback_missing();
        return {success_k};
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) const noexcept {
        entry_node_t::range(entries_.root(), std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                            [&](entry_node_t *node) noexcept {
                                if (node->entry.visible) callback(node->entry.element);
                                static_assert(noexcept(callback(node->entry.element)));
                            });
        return {success_k};
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t range(lower_type_ &&lower, upper_type_ &&upper, callback_type_ &&callback) noexcept {
        generation_t generation = new_generation();
        entry_node_t::range(entries_.root(), std::forward<lower_type_>(lower), std::forward<upper_type_>(upper),
                            [&](entry_node_t *node) noexcept {
                                if (node->entry.visible)
                                    callback(node->entry.element), node->entry.generation = generation;
                                static_assert(noexcept(callback(node->entry.element)));
                            });
        return {success_k};
    }

    template <typename lower_type_ = identifier_t, typename upper_type_ = identifier_t,
              typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                       callback_type_ &&callback = {}) noexcept {
        // Implementing Splits and Joins for AVL can be tricky.
        // Let's start with deleting them one by one.
        // TODO: Implement range-removals.
        auto last = entries_.lower_bound(std::forward<lower_type_>(lower));
        auto less = entry_comparator_t {};
        while (last != entries_.end() && less(last->entry, upper)) {
            auto next = entries_.upper_bound(last->entry);
            if (last->entry.visible) {
                callback(last->entry.element);
                --visible_count_;
                visible_deleted_count_ -= last->entry.deleted;
                entries_.extract(last->entry);
            }
            last = next;
        }
        return {success_k};
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                        callback_type_ &&callback) const noexcept {

        auto node = entry_node_t::sample_range( //
            entries_.root(), lower, upper, std::forward<generator_type_>(generator),
            [](entry_node_t *node) noexcept { return node->entry.visible; });
        if (node) callback(node->entry);
        return {success_k};
    }

    template <typename lower_type_, typename upper_type_, typename generator_type_, typename output_iterator_type_>
    [[nodiscard]] status_t sample_range(lower_type_ &&lower, upper_type_ &&upper, generator_type_ &&generator,
                                        std::size_t &seen, std::size_t reservoir_capacity,
                                        output_iterator_type_ &&reservoir) const noexcept {

        using output_iterator_t = std::remove_reference_t<output_iterator_type_>;
        using output_category_t = typename std::iterator_traits<output_iterator_t>::iterator_category;
        static_assert(std::is_same<std::random_access_iterator_tag, output_category_t>(), "Must be random access!");

        auto sampler = [&](element_t const &element) noexcept {
            if (seen < reservoir_capacity) reservoir[seen] = element;

            else {
                std::uniform_int_distribution<std::size_t> distribution {0, seen};
                auto slot_to_replace = distribution(generator);
                if (slot_to_replace < reservoir_capacity) reservoir[slot_to_replace] = element;
            }

            ++seen;
        };
        return range(std::forward<lower_type_>(lower), std::forward<upper_type_>(upper), sampler);
    }

    [[nodiscard]] status_t clear() noexcept {
        entries_.clear();
        generation_ = 0;
        visible_count_ = 0;
        visible_deleted_count_ = 0;
        return {success_k};
    }

    /**
     *  @brief Debug utility to print tree contents.
     *  @note Requires `#include <ostream>` (not included by default to reduce header weight)
     */
    template <typename dont_instantiate_me_type_>
    void print(dont_instantiate_me_type_ &cout) {
        cout << "Items: " << entries_.size() << "\n";
        cout << "Imbalance: " << entries_.total_imbalance() << "\n";
        entry_node_t::for_each_left_right(entries_.root(), [&](entry_node_t *node) {
            char const *marker = node->entry.visible ? "✓" : "✗";
            cout << identifier_t {node->entry.element} << " @" << node->entry.generation << marker << " ";
        });
        cout << "\n";
    }
};

} // namespace ashvardanian::smashtable