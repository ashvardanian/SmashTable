/**
 *  @file include/smashtable/immutable_b_tree.hpp
 *  @author Ash Vardanian
 *  @date September 15, 2026
 *  @brief An immutable B-tree over sorted keys, stored breadth-first with implicit children and
 *      keys on every level.
 *
 *  @section immutable_btree_layout Layout
 *
 *  Node @c i holds one row of @c B keys, and its @c B+1 children are the nodes @c i*(B+1)+j+1.
 *  Nodes are numbered breadth-first and keys are placed in order, so every level above the deepest
 *  is complete, and the unused slots, all at the end of the order, hold the padding key. A lookup
 *  reads one row per level. Within one level the keys below a wanted key form a prefix, so the rank
 *  adds up as the descent goes, with no subtree sizes stored.
 *
 *  @section immutable_btree_mapped Mapped Values
 *
 *  A map form keeps the rows exactly as a set does and puts the mapped values in one array indexed
 *  by rank, which every lookup already computes. The rows stay a run of bare keys the kits can load
 *  whole, at the price of a second cache line per hit: the rows are ordered breadth-first while the
 *  mapped array is ordered by rank, so a key and its value never share one.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::size_t`, `std::ptrdiff_t`
#include <cstdint> // `std::uint64_t`

#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span`
#include <type_traits> // `std::bool_constant`, `std::conditional_t`
#include <utility>     // `std::exchange`, `std::move`

#include "basic_vector.hpp"
#include "row_search.hpp"

namespace ashvardanian::smashtable {

/**
 *  An immutable B-tree over @p value_type_, built once from a sorted span, whose nodes are rows of
 *  @p keys_per_row_ keys searched by @p row_kit_type_. A set holds one allocation of rows; a map
 *  holds a second one of mapped values, indexed by rank.
 *
 *  @tparam keys_per_row_ Keys per node, which is @c default_row_bytes_k worth by default.
 */
template <typename value_type_,
          std::size_t keys_per_row_ =
              keys_per_row<typename mapping_key_type_or_itself<value_type_>::type>(default_row_bytes_k),
          row_kit row_kit_type_ = native_row_kit_t, typename allocator_type_ = default_allocator<value_type_>>
class immutable_b_tree {
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

    static_assert(row_searchable_key<key_t>, "the row kits search the key, whatever rides beside it");
    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "immutable_b_tree requires allocators that propagate on move assignment");

  private:
    using words_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<word_t>;
    using mapped_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<mapped_slot_t>;

    struct slot_t {
        std::size_t node;
        std::size_t index;
    };

    struct descent_t {
        std::size_t rank;
        slot_t lower_bound;
    };

    basic_vector<word_t, words_allocator_t> words_;
    basic_vector<mapped_slot_t, mapped_allocator_t> mapped_;
    std::size_t size_ {0};
    std::size_t nodes_count_ {0};
    std::size_t deepest_level_ {0};

  public:
    /** A forward walk over the keys in order, yielding each by value. A map walks its keys too, and
     *  @c rank indexes @c mapped_at for the value beside one. */
    class iterator {
        friend class immutable_b_tree;

        immutable_b_tree const *tree_ {nullptr};
        slot_t slot_ {0, 0};
        std::size_t rank_ {0};

        iterator(immutable_b_tree const *tree, slot_t slot, std::size_t rank) noexcept
            : tree_(tree), slot_(slot), rank_(rank) {}

      public:
        using value_type = key_t;
        using difference_type = std::ptrdiff_t;

        iterator() noexcept = default;

        [[nodiscard]] key_t operator*() const noexcept {
            return format_t::key_at(tree_->row_(slot_.node), slot_.index);
        }

        iterator &operator++() noexcept {
            slot_ = tree_->next_slot_(slot_);
            ++rank_;
            return *this;
        }

        iterator operator++(int) noexcept {
            iterator const previous = *this;
            ++*this;
            return previous;
        }

        /** The position of the current key in sorted order. */
        [[nodiscard]] std::size_t rank() const noexcept { return rank_; }

        [[nodiscard]] friend bool operator==(iterator const &left, iterator const &right) noexcept {
            return left.rank_ == right.rank_;
        }
    };

    immutable_b_tree() noexcept = default;

    explicit immutable_b_tree(allocator_t allocator) noexcept
        : words_(words_allocator_t(allocator)), mapped_(mapped_allocator_t(allocator)) {}

    immutable_b_tree(immutable_b_tree &&other) noexcept
        : words_(std::move(other.words_)), mapped_(std::move(other.mapped_)), size_(std::exchange(other.size_, 0)),
          nodes_count_(std::exchange(other.nodes_count_, 0)), deepest_level_(std::exchange(other.deepest_level_, 0)) {}

    immutable_b_tree &operator=(immutable_b_tree &&other) noexcept {
        if (this == &other) return *this;
        words_ = std::move(other.words_);
        mapped_ = std::move(other.mapped_);
        size_ = std::exchange(other.size_, 0);
        nodes_count_ = std::exchange(other.nodes_count_, 0);
        deepest_level_ = std::exchange(other.deepest_level_, 0);
        return *this;
    }

    immutable_b_tree(immutable_b_tree const &) = delete;
    immutable_b_tree &operator=(immutable_b_tree const &) = delete;

    /**
     *  Builds the tree over @p sorted, which may repeat keys. A map keeps the mapped values in the
     *  order they arrive, so a repeated key reads back the value that came first.
     *
     *  @return The tree, @c invalid_argument_k when @p sorted is out of order, or
     *      @c out_of_memory_heap_k.
     */
    [[nodiscard]] static expected<immutable_b_tree> make(std::span<value_t const> sorted,
                                                         allocator_t allocator = {}) noexcept {
        for (std::size_t index = 1; index < sorted.size(); ++index)
            if (mapping_key_or_itself(sorted[index]) < mapping_key_or_itself(sorted[index - 1]))
                return invalid_argument_k;
        std::size_t const nodes_count = sorted.size() / keys_per_row_k + (sorted.size() % keys_per_row_k != 0);
        if (nodes_count > size_max_k / fanout_k / format_t::words_per_row_k) return out_of_memory_heap_k;

        immutable_b_tree tree(std::move(allocator));
        status_t const status = tree.words_.resize(nodes_count * format_t::words_per_row_k);
        if (failed(status)) return status;
        tree.size_ = sorted.size();
        tree.nodes_count_ = nodes_count;
        for (std::size_t first = 0; first * fanout_k + 1 < nodes_count; first = first * fanout_k + 1)
            ++tree.deepest_level_;

        slot_t slot = tree.leftmost_slot_();
        for (std::size_t ordinal = 0; ordinal < nodes_count * keys_per_row_k; ++ordinal) {
            format_t::store(tree.row_(slot.node), slot.index,
                            ordinal < sorted.size() ? mapping_key_or_itself(sorted[ordinal]) : format_t::padding_k);
            slot = tree.next_slot_(slot);
        }

        if constexpr (is_mapping<value_t>) {
            status_t const reserved = tree.mapped_.reserve(sorted.size());
            if (failed(reserved)) return reserved;
            for (value_t const &element : sorted) {
                status_t const appended = tree.mapped_.emplace_back(assume_reserved, element.mapped);
                if (failed(appended)) return appended;
            }
        }
        return expected<immutable_b_tree>(std::move(tree), success_k);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /** The bytes this tree stores, rows and mapped values together. */
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return words_.size() * sizeof(word_t) + mapped_.size() * sizeof(mapped_slot_t);
    }

    /** The bytes a tree over @p keys_count elements stores: whole rows of keys, and a value per key for a map. */
    [[nodiscard]] static constexpr std::size_t size_bytes(std::size_t keys_count) noexcept {
        std::size_t const rows_count = keys_count / keys_per_row_k + (keys_count % keys_per_row_k != 0);
        std::size_t const mapped_bytes = is_mapping<value_t> ? keys_count * sizeof(mapped_slot_t) : 0;
        return rows_count * format_t::bytes_per_row_k + mapped_bytes;
    }

    /** The stored words, node after node, for writing the tree out. */
    [[nodiscard]] std::span<word_t const> words() const noexcept { return {words_.data(), words_.size()}; }

    /** How many keys order below @p wanted, which is also the position of its lower bound. */
    [[nodiscard]] std::size_t rank(key_t wanted) const noexcept { return descend_(wanted).rank; }

    /** The position of @p wanted, or @c key_not_found_k. Where keys repeat, the first of them. */
    [[nodiscard]] expected<std::size_t> find(key_t wanted) const noexcept {
        descent_t const descent = descend_(wanted);
        if (descent.rank == size_ ||
            !(format_t::key_at(row_(descent.lower_bound.node), descent.lower_bound.index) == wanted))
            return key_not_found_k;
        return expected<std::size_t>(descent.rank, success_k);
    }

    /** The key at position @p ordinal in sorted order, which must be below @c size. */
    [[nodiscard]] key_t select(std::size_t ordinal) const noexcept {
        assert(ordinal < size_ && "select takes a position below size");
        slot_t const slot = slot_at_(ordinal);
        return format_t::key_at(row_(slot.node), slot.index);
    }

    /** The value the key at position @p ordinal carries, which must be below @c size. */
    [[nodiscard]] mapped_slot_t const &mapped_at(std::size_t ordinal) const noexcept
        requires is_mapping<value_t>
    {
        assert(ordinal < size_ && "mapped_at takes a position below size");
        return mapped_[ordinal];
    }

    [[nodiscard]] iterator begin() const noexcept { return iterator(this, leftmost_slot_(), 0); }
    [[nodiscard]] iterator end() const noexcept { return iterator(this, slot_t {nodes_count_, 0}, size_); }

    /** A walk from position @p ordinal, or @c end when there is no such position. */
    [[nodiscard]] iterator at_rank(std::size_t ordinal) const noexcept {
        return ordinal < size_ ? iterator(this, slot_at_(ordinal), ordinal) : end();
    }

    /** A walk from the first key not below @p wanted. */
    [[nodiscard]] iterator lower_bound(key_t wanted) const noexcept { return at_rank(rank(wanted)); }

  private:
    [[nodiscard]] word_t const *row_(std::size_t node) const noexcept {
        return words_.data() + node * format_t::words_per_row_k;
    }
    [[nodiscard]] word_t *row_(std::size_t node) noexcept { return words_.data() + node * format_t::words_per_row_k; }

    [[nodiscard]] descent_t descend_(key_t wanted) const noexcept {
        descent_t descent {0, slot_t {0, 0}};
        std::size_t first = 0;
        std::size_t node = 0;
        for (; node < nodes_count_; first = first * fanout_k + 1) {
            std::size_t const below = count_below_in_row<format_t, kit_t>(row_(node), wanted);
            descent.rank += (node - first) * keys_per_row_k + below;
            if (below < keys_per_row_k) descent.lower_bound = slot_t {node, below};
            node = node * fanout_k + below + 1;
        }
        // Past the end of the tree, every node left on the deeper levels orders before the path.
        for (; first < nodes_count_; first = first * fanout_k + 1)
            descent.rank += (nodes_count_ - first) * keys_per_row_k;
        assert(descent.rank <= size_ && "padding orders at or after every key");
        return descent;
    }

    [[nodiscard]] slot_t leftmost_slot_() const noexcept {
        std::size_t node = 0;
        while (node * fanout_k + 1 < nodes_count_) node = node * fanout_k + 1;
        return slot_t {node, 0};
    }

    [[nodiscard]] slot_t next_slot_(slot_t slot) const noexcept {
        std::size_t node = slot.node * fanout_k + slot.index + 2;
        if (node < nodes_count_) {
            while (node * fanout_k + 1 < nodes_count_) node = node * fanout_k + 1;
            return slot_t {node, 0};
        }
        if (slot.index + 1 < keys_per_row_k) return slot_t {slot.node, slot.index + 1};
        for (node = slot.node; node != 0;) {
            std::size_t const child = (node - 1) % fanout_k;
            node = (node - 1) / fanout_k;
            if (child < keys_per_row_k) return slot_t {node, child};
        }
        return slot_t {nodes_count_, 0};
    }

    /** Descends by subtree sizes, which follow from the level of a node since only the deepest level is partial. */
    [[nodiscard]] slot_t slot_at_(std::size_t ordinal) const noexcept {
        std::size_t node = 0;
        for (std::size_t level = 0; level < deepest_level_; ++level) {
            std::size_t const complete_levels = deepest_level_ - level - 1;
            std::size_t deepest_nodes = 1;
            std::size_t deepest_first = node * fanout_k + 1;
            for (std::size_t step = 0; step < complete_levels; ++step) {
                deepest_nodes *= fanout_k;
                deepest_first = deepest_first * fanout_k + 1;
            }
            std::size_t child = 0;
            for (;; ++child) {
                std::size_t const child_node = node * fanout_k + child + 1;
                std::size_t const child_deepest_first = deepest_first + child * deepest_nodes;
                std::size_t const deepest_present = child_deepest_first < nodes_count_
                                                        ? smaller_of(nodes_count_ - child_deepest_first, deepest_nodes)
                                                        : 0;
                std::size_t const child_slots =
                    child_node < nodes_count_ ? deepest_nodes - 1 + deepest_present * keys_per_row_k : 0;
                if (ordinal < child_slots) break;
                ordinal -= child_slots;
                assert(child < keys_per_row_k && "select takes a position below size");
                if (ordinal == 0) return slot_t {node, child};
                --ordinal;
            }
            node = node * fanout_k + child + 1;
        }
        return slot_t {node, ordinal};
    }
};

/** An immutable B-tree of bare keys. */
template <typename key_type_, std::size_t keys_per_row_ = keys_per_row<key_type_>(default_row_bytes_k),
          row_kit row_kit_type_ = native_row_kit_t, typename allocator_type_ = default_allocator<key_type_>>
using immutable_b_set = immutable_b_tree<key_type_, keys_per_row_, row_kit_type_, allocator_type_>;

/** An immutable B-tree pairing each key with a value the rank indexes. */
template <typename key_type_, typename mapped_type_,
          std::size_t keys_per_row_ = keys_per_row<key_type_>(default_row_bytes_k),
          row_kit row_kit_type_ = native_row_kit_t,
          typename allocator_type_ = default_allocator<mapping<key_type_, mapped_type_>>>
using immutable_b_map =
    immutable_b_tree<mapping<key_type_, mapped_type_>, keys_per_row_, row_kit_type_, allocator_type_>;

static_assert(tagged_collection<immutable_b_set<std::uint64_t>>,
              "an immutable B-tree names its element, its key and its shape like every other container");
static_assert(!immutable_b_set<std::uint64_t>::is_associative::value, "the set alias stores bare keys");
static_assert(immutable_b_map<std::uint64_t, double>::is_associative::value, "the map alias stores a mapping");
static_assert(std::is_same_v<immutable_b_map<std::uint64_t, double>::key_type, std::uint64_t> &&
                  std::is_same_v<immutable_b_map<std::uint64_t, double>::mapped_type, double>,
              "a map splits into the two halves the alias was given");

} // namespace ashvardanian::smashtable
