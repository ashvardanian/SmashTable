/**
 *  @file include/smashtable/immutable_splus_tree.hpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief An immutable S+ tree over sorted keys: every key in sorted leaves, separator copies on
 *      the levels above.
 *
 *  @section immutable_splus_tree_layout Layout
 *
 *  The leaves are the sorted keys cut into rows of @c B, the last row padded. Each level above cuts
 *  the one below into groups of @c B+1 nodes, and node @c t holds as its key @c j the first key of
 *  child @c t*(B+1)+j+1, so the first level up copies every @c B-th key. Levels are stored root
 *  first and leaves last. A lookup reads one row per level and lands on the leaf whose position is
 *  the rank, so @c select is an index into the leaves.
 *
 *  @section immutable_splus_tree_mapped Mapped Values
 *
 *  A map form keeps the rows exactly as a set does and puts the mapped values in one array indexed
 *  by rank, which every lookup already computes. The leaves are themselves in rank order, so a walk
 *  reads the rows and the mapped array in step.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::size_t`, `std::ptrdiff_t`
#include <cstdint> // `std::uint64_t`

#include <array>       // `std::array`
#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span`
#include <type_traits> // `std::bool_constant`, `std::conditional_t`
#include <utility>     // `std::exchange`, `std::move`

#include "basic_vector.hpp"
#include "row_search.hpp"

namespace ashvardanian::smashtable {

/**
 *  An immutable S+ tree over @p value_type_, built once from a sorted span, whose nodes are rows of
 *  @p keys_per_row_ keys searched by @p row_kit_type_. A set holds one allocation of rows; a map
 *  holds a second one of mapped values, indexed by rank.
 *
 *  @tparam keys_per_row_ Keys per node, which is @c default_row_bytes_k worth by default.
 */
template <typename value_type_,
          std::size_t keys_per_row_ =
              keys_per_row<typename mapping_key_type_or_itself<value_type_>::type>(default_row_bytes_k),
          row_kit row_kit_type_ = native_row_kit_t, typename allocator_type_ = default_allocator<value_type_>>
class immutable_splus_tree {
  public:
    /** The whole stored element: the key itself for a set, a @c mapping of both halves for a map. */
    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility

    /** The key half the rows are searched on, which is the whole element for a set. */
    using key_t = typename mapping_key_type_or_itself<value_t>::type;
    using key_type = key_t; // ? STL compatibility

    /** The mapped half, held in rank order beside the rows, and @c void for a set. */
    using mapped_t = typename mapped_value_type_or_void<value_t>::type;
    using mapped_type = mapped_t; // ? STL compatibility

    /** What one entry of the mapped array holds, which is an empty stand-in when the tree is a set. */
    using mapped_slot_t = std::conditional_t<is_mapping<value_t>, mapped_t, placeholder_t>;

    using is_associative = std::bool_constant<is_mapping<value_t>>;

    using format_t = row_format<key_t, keys_per_row_>;
    using word_t = typename format_t::word_t;
    using kit_t = row_kit_type_;
    using allocator_t = allocator_type_;
    using allocator_type = allocator_t; // ? STL compatibility

    static constexpr std::size_t keys_per_row_k = keys_per_row_;
    static constexpr std::size_t fanout_k = keys_per_row_ + 1;

    /** More levels than a fanout of three needs for 2^64 keys. */
    static constexpr std::size_t levels_capacity_k = 48;

    static_assert(row_searchable_key<key_t>, "the row kits search the key, whatever rides beside it");
    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "immutable_splus_tree requires allocators that propagate on move assignment");

  private:
    using words_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<word_t>;
    using mapped_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<mapped_slot_t>;
    using level_nodes_t = std::array<std::size_t, levels_capacity_k + 1>;

    basic_vector<word_t, words_allocator_t> words_;
    basic_vector<mapped_slot_t, mapped_allocator_t> mapped_;
    std::size_t size_ {0};
    std::size_t levels_count_ {0};
    level_nodes_t level_firsts_ {}; // ? The first node of each level, root first, then one past the leaves

  public:
    /** A forward walk over the keys in order, yielding each by value. A map walks its keys too, and
     *  @c rank indexes @c mapped_at for the value beside one. */
    class iterator {
        friend class immutable_splus_tree;

        immutable_splus_tree const *tree_ {nullptr};
        std::size_t rank_ {0};

        iterator(immutable_splus_tree const *tree, std::size_t rank) noexcept : tree_(tree), rank_(rank) {}

      public:
        using value_type = key_t;
        using difference_type = std::ptrdiff_t;

        iterator() noexcept = default;

        [[nodiscard]] key_t operator*() const noexcept { return tree_->select(rank_); }

        iterator &operator++() noexcept {
            ++rank_;
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator const previous = *this;
            ++rank_;
            return previous;
        }

        /** The position of the current key in sorted order. */
        [[nodiscard]] std::size_t rank() const noexcept { return rank_; }

        [[nodiscard]] friend bool operator==(iterator const &left, iterator const &right) noexcept {
            return left.rank_ == right.rank_;
        }
    };

    immutable_splus_tree() noexcept = default;

    explicit immutable_splus_tree(allocator_t allocator) noexcept
        : words_(words_allocator_t(allocator)), mapped_(mapped_allocator_t(allocator)) {}

    immutable_splus_tree(immutable_splus_tree &&other) noexcept
        : words_(std::move(other.words_)), mapped_(std::move(other.mapped_)), size_(std::exchange(other.size_, 0)),
          levels_count_(std::exchange(other.levels_count_, 0)), level_firsts_(other.level_firsts_) {}

    immutable_splus_tree &operator=(immutable_splus_tree &&other) noexcept {
        if (this == &other) return *this;
        words_ = std::move(other.words_);
        mapped_ = std::move(other.mapped_);
        size_ = std::exchange(other.size_, 0);
        levels_count_ = std::exchange(other.levels_count_, 0);
        level_firsts_ = other.level_firsts_;
        return *this;
    }

    immutable_splus_tree(immutable_splus_tree const &) = delete;
    immutable_splus_tree &operator=(immutable_splus_tree const &) = delete;

    /**
     *  Builds the tree over @p sorted, which may repeat keys. A map keeps the mapped values in the
     *  order they arrive, so a repeated key reads back the value that came first.
     *
     *  @return The tree, @c invalid_argument_k when @p sorted is out of order, or
     *      @c out_of_memory_heap_k.
     */
    [[nodiscard]] static expected<immutable_splus_tree> make(std::span<value_t const> sorted,
                                                             allocator_t allocator = {}) noexcept {
        for (std::size_t index = 1; index < sorted.size(); ++index)
            if (mapping_key_or_itself(sorted[index]) < mapping_key_or_itself(sorted[index - 1]))
                return invalid_argument_k;

        level_nodes_t leaf_counts {};
        std::size_t const levels_count = count_levels_(sorted.size(), leaf_counts);
        immutable_splus_tree tree(std::move(allocator));
        std::size_t nodes_count = 0;
        for (std::size_t level = 0; level < levels_count; ++level) {
            tree.level_firsts_[level] = nodes_count;
            nodes_count += leaf_counts[levels_count - 1 - level];
        }
        tree.level_firsts_[levels_count] = nodes_count;
        if (nodes_count > size_max_k / fanout_k / format_t::words_per_row_k) return out_of_memory_heap_k;
        status_t const status = tree.words_.resize(nodes_count * format_t::words_per_row_k);
        if (failed(status)) return status;
        tree.size_ = sorted.size();
        tree.levels_count_ = levels_count;
        if (levels_count == 0) return expected<immutable_splus_tree>(std::move(tree), success_k);

        std::size_t const leaves_level = levels_count - 1;
        std::size_t const leaves_count = leaf_counts[0];
        for (std::size_t ordinal = 0; ordinal < leaves_count * keys_per_row_k; ++ordinal)
            format_t::store(tree.row_(leaves_level, ordinal / keys_per_row_k), ordinal % keys_per_row_k,
                            ordinal < sorted.size() ? mapping_key_or_itself(sorted[ordinal]) : format_t::padding_k);

        if constexpr (is_mapping<value_t>) {
            status_t const reserved = tree.mapped_.reserve(sorted.size());
            if (failed(reserved)) return reserved;
            for (value_t const &element : sorted) {
                status_t const appended = tree.mapped_.emplace_back(assume_reserved, element.mapped);
                if (failed(appended)) return appended;
            }
        }

        // A node `levels_up` levels above the leaves starts at leaf `node * fanout^levels_up`.
        std::size_t child_leaf_span = 1;
        for (std::size_t level = leaves_level; level-- > 0;) {
            std::size_t const nodes_on_level = tree.level_firsts_[level + 1] - tree.level_firsts_[level];
            for (std::size_t node = 0; node < nodes_on_level; ++node)
                for (std::size_t index = 0; index < keys_per_row_k; ++index) {
                    std::size_t const leaf = (node * fanout_k + index + 1) * child_leaf_span;
                    key_t const separator =
                        leaf < leaves_count ? format_t::key_at(tree.row_(leaves_level, leaf), 0) : format_t::padding_k;
                    format_t::store(tree.row_(level, node), index, separator);
                }
            child_leaf_span *= fanout_k;
        }
        return expected<immutable_splus_tree>(std::move(tree), success_k);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /** The bytes this tree stores, rows and mapped values together. */
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return words_.size() * sizeof(word_t) + mapped_.size() * sizeof(mapped_slot_t);
    }

    /** The bytes a tree over @p keys_count elements stores: whole rows on every level, and a value per key for a
     *  map. */
    [[nodiscard]] static constexpr std::size_t size_bytes(std::size_t keys_count) noexcept {
        level_nodes_t leaf_counts {};
        std::size_t const levels_count = count_levels_(keys_count, leaf_counts);
        std::size_t nodes_count = 0;
        for (std::size_t level = 0; level < levels_count; ++level) nodes_count += leaf_counts[level];
        std::size_t const mapped_bytes = is_mapping<value_t> ? keys_count * sizeof(mapped_slot_t) : 0;
        return nodes_count * format_t::bytes_per_row_k + mapped_bytes;
    }

    /** The stored words, root level first, for writing the tree out. */
    [[nodiscard]] std::span<word_t const> words() const noexcept { return {words_.data(), words_.size()}; }

    /** How many keys order below @p wanted, which is also the position of its lower bound. */
    [[nodiscard]] std::size_t rank(key_t wanted) const noexcept {
        if (levels_count_ == 0) return 0;
        std::size_t node = 0;
        for (std::size_t level = 0; level + 1 < levels_count_; ++level)
            node = node * fanout_k + count_below_in_row<format_t, kit_t>(row_(level, node), wanted);
        return node * keys_per_row_k + count_below_in_row<format_t, kit_t>(row_(levels_count_ - 1, node), wanted);
    }

    /** The position of @p wanted, or @c key_not_found_k. Where keys repeat, the first of them. */
    [[nodiscard]] expected<std::size_t> find(key_t wanted) const noexcept {
        std::size_t const position = rank(wanted);
        if (position == size_ || !(select(position) == wanted)) return key_not_found_k;
        return expected<std::size_t>(position, success_k);
    }

    /** The key at position @p ordinal in sorted order, which must be below @c size. */
    [[nodiscard]] key_t select(std::size_t ordinal) const noexcept {
        assert(ordinal < size_ && "select takes a position below size");
        return format_t::key_at(row_(levels_count_ - 1, ordinal / keys_per_row_k), ordinal % keys_per_row_k);
    }

    /** The value the key at position @p ordinal carries, which must be below @c size. */
    [[nodiscard]] mapped_slot_t const &mapped_at(std::size_t ordinal) const noexcept
        requires is_mapping<value_t>
    {
        assert(ordinal < size_ && "mapped_at takes a position below size");
        return mapped_[ordinal];
    }

    [[nodiscard]] iterator begin() const noexcept { return iterator(this, 0); }
    [[nodiscard]] iterator end() const noexcept { return iterator(this, size_); }

    /** A walk from position @p ordinal, or @c end when there is no such position. */
    [[nodiscard]] iterator at_rank(std::size_t ordinal) const noexcept {
        return iterator(this, smaller_of(ordinal, size_));
    }

    /** A walk from the first key not below @p wanted. */
    [[nodiscard]] iterator lower_bound(key_t wanted) const noexcept { return iterator(this, rank(wanted)); }

  private:
    [[nodiscard]] word_t const *row_(std::size_t level, std::size_t node) const noexcept {
        return words_.data() + (level_firsts_[level] + node) * format_t::words_per_row_k;
    }
    [[nodiscard]] word_t *row_(std::size_t level, std::size_t node) noexcept {
        return words_.data() + (level_firsts_[level] + node) * format_t::words_per_row_k;
    }

    /** Fills @p leaf_counts with the nodes on each level from the leaves up, and returns how many levels there are. */
    static constexpr std::size_t count_levels_(std::size_t keys_count, level_nodes_t &leaf_counts) noexcept {
        std::size_t nodes = keys_count / keys_per_row_k + (keys_count % keys_per_row_k != 0);
        std::size_t levels_count = 0;
        while (nodes != 0) {
            leaf_counts[levels_count++] = nodes;
            nodes = nodes == 1 ? 0 : nodes / fanout_k + (nodes % fanout_k != 0);
        }
        return levels_count;
    }
};

/** An immutable S+ tree of bare keys. */
template <typename key_type_, std::size_t keys_per_row_ = keys_per_row<key_type_>(default_row_bytes_k),
          row_kit row_kit_type_ = native_row_kit_t, typename allocator_type_ = default_allocator<key_type_>>
using immutable_splus_set = immutable_splus_tree<key_type_, keys_per_row_, row_kit_type_, allocator_type_>;

/** An immutable S+ tree pairing each key with a value the rank indexes. */
template <typename key_type_, typename mapped_type_,
          std::size_t keys_per_row_ = keys_per_row<key_type_>(default_row_bytes_k),
          row_kit row_kit_type_ = native_row_kit_t,
          typename allocator_type_ = default_allocator<mapping<key_type_, mapped_type_>>>
using immutable_splus_map =
    immutable_splus_tree<mapping<key_type_, mapped_type_>, keys_per_row_, row_kit_type_, allocator_type_>;

static_assert(tagged_collection<immutable_splus_set<std::uint64_t>>,
              "an immutable S+ tree names its element, its key and its shape like every other container");
static_assert(!immutable_splus_set<std::uint64_t>::is_associative::value, "the set alias stores bare keys");
static_assert(immutable_splus_map<std::uint64_t, double>::is_associative::value, "the map alias stores a mapping");
static_assert(std::is_same_v<immutable_splus_map<std::uint64_t, double>::key_type, std::uint64_t> &&
                  std::is_same_v<immutable_splus_map<std::uint64_t, double>::mapped_type, double>,
              "a map splits into the two halves the alias was given");

} // namespace ashvardanian::smashtable
