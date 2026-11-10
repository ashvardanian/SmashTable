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
 *
 *  @section immutable_splus_tree_saved Saved Form
 *
 *  @c save writes a @c splus_tree_header_t, the rows as they sit in memory, and then the mapped
 *  array, all in this machine's byte order. @c immutable_splus_tree_view searches that form in
 *  place with the same kits, so a tree written into a file is read from a mapping or from CXL
 *  memory without a copy. The rows are handed to the kits as words, so the view needs its bytes to
 *  start on a word boundary, which any mapped page and any section placed on one already is.
 */
#pragma once
#include <cassert> // `assert`
#include <cstddef> // `std::size_t`, `std::ptrdiff_t`
#include <cstdint> // `std::uint64_t`, `std::uintptr_t`
#include <cstring> // `std::memcpy`, `std::memset`

#include <array>       // `std::array`
#include <memory>      // `std::allocator_traits`
#include <span>        // `std::span`
#include <type_traits> // `std::bool_constant`, `std::conditional_t`, `std::is_trivially_copyable_v`
#include <utility>     // `std::exchange`, `std::move`

#include "smashtable/basic_vector.hpp"
#include "smashtable/row_search.hpp"

namespace ashvardanian::smashtable {

/** What @c save writes ahead of the rows, one 64-bit word per field, so a view refuses a tree of
 *  another shape. */
struct splus_tree_header_t {

    /** How many keys the tree holds. */
    std::uint64_t count {0};

    /** How many keys one row holds. */
    std::uint64_t keys_per_row {0};

    /** How many bytes one key takes. */
    std::uint64_t key_bytes {0};

    /** How many bytes one mapped value takes, zero for a set. */
    std::uint64_t mapped_bytes {0};
};

/**
 *  @brief An immutable S+ tree searched in place, over rows and mapped values it does not own:
 *      either a saved form in borrowed bytes, or the storage of the tree it was taken from.
 *
 *  A few pointers and the level offsets, so it copies freely and stays valid exactly as long as
 *  the bytes it was made over.
 *
 *  @tparam keys_per_row_ Keys per node, which must match the tree that saved the bytes.
 */
template <typename value_type_,
          std::size_t keys_per_row_ =
              keys_per_row<typename mapping_key_type_or_itself<value_type_>::type>(default_row_bytes_k),
          row_kit row_kit_type_ = native_row_kit_t>
class immutable_splus_tree_view {
  public:
    /** Every stored element: the key itself for a set, a @c mapping of both halves for a map. */
    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility

    /** The key half the rows are searched on, which is the whole element for a set. */
    using key_t = typename mapping_key_type_or_itself<value_t>::type;
    using key_type = key_t; // ? STL compatibility

    /** The mapped half, held in rank order beside the rows, and @c void for a set. */
    using mapped_t = typename mapped_value_type_or_void<value_t>::type;
    using mapped_type = mapped_t; // ? STL compatibility

    /** What one entry of the mapped array holds: an empty stand-in when the tree is a set. */
    using mapped_slot_t = std::conditional_t<is_mapping<value_t>, mapped_t, placeholder_t>;

    using is_associative = std::bool_constant<is_mapping<value_t>>;

    using format_t = row_format<key_t, keys_per_row_>;
    using word_t = typename format_t::word_t;
    using kit_t = row_kit_type_;

    static constexpr std::size_t keys_per_row_k = keys_per_row_;
    static constexpr std::size_t fanout_k = keys_per_row_ + 1;

    /** More levels than a fanout of three needs for 2^64 keys. */
    static constexpr std::size_t levels_capacity_k = 48;

    /** Where the rows start in the saved form, right after the header. */
    static constexpr std::size_t saved_rows_offset_k = sizeof(splus_tree_header_t);

    /** The boundary saved bytes start on, so the kits read whole words and mapped values align. */
    static constexpr std::size_t saved_alignment_k = alignof(word_t) > alignof(mapped_slot_t) ? alignof(word_t)
                                                                                              : alignof(mapped_slot_t);

    static_assert(row_searchable_key<key_t>, "the row kits search the key, whatever rides beside it");
    static_assert(saved_rows_offset_k % saved_alignment_k == 0,
                  "the header keeps the rows on the alignment the saved bytes start on");

  private:
    template <typename, std::size_t, row_kit, typename>
    friend class immutable_splus_tree;

    using level_nodes_t = std::array<std::size_t, levels_capacity_k + 1>;

    word_t const *words_ {nullptr};
    mapped_slot_t const *mapped_ {nullptr};
    std::size_t size_ {0};
    std::size_t levels_count_ {0};
    level_nodes_t level_firsts_ {}; // ? The first node of each level, root first, then one past the leaves

  public:
    /** A forward walk over the keys in order, yielding each by value. A map walks its keys too, and
     *  @c rank indexes @c mapped_at for the value beside one. */
    class iterator {
        friend class immutable_splus_tree_view;

        immutable_splus_tree_view const *tree_ {nullptr};
        std::size_t rank_ {0};

        iterator(immutable_splus_tree_view const *tree, std::size_t rank) noexcept : tree_(tree), rank_(rank) {}

      public:
        using value_type = key_t;
        using difference_type = std::ptrdiff_t;

        /** Default-constructs an empty @c iterator. */
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

    /** Default-constructs an empty @c immutable_splus_tree_view. */
    constexpr immutable_splus_tree_view() noexcept = default;

    /**
     *  @brief Reads the saved form at the front of @p bytes, starting on @c saved_alignment_k.
     *  @return The view.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p bytes is misaligned, ends before the rows and mapped values its
     *    header names, or was saved by a tree of another key, row width or mapped value.
     */
    static expected<immutable_splus_tree_view> make(std::span<std::byte const> bytes) noexcept {
        splus_tree_header_t header;
        if (bytes.size() < sizeof(header)) return invalid_argument_k;
        if (reinterpret_cast<std::uintptr_t>(bytes.data()) % saved_alignment_k != 0) return invalid_argument_k;
        std::memcpy(&header, bytes.data(), sizeof(header));
        if (header.keys_per_row != keys_per_row_k || header.key_bytes != sizeof(key_t) ||
            header.mapped_bytes != saved_mapped_bytes_k)
            return invalid_argument_k;
        // Every key sits in a leaf, so a larger count cannot fit; this bounds the arithmetic below.
        if (header.count > bytes.size() / sizeof(key_t)) return invalid_argument_k;

        immutable_splus_tree_view view;
        std::size_t const nodes_count = view.lay_out_(static_cast<std::size_t>(header.count));
        std::size_t const rows_bytes = nodes_count * format_t::bytes_per_row_k;
        if (bytes.size() < view.saved_bytes_after_(rows_bytes)) return invalid_argument_k;
        view.words_ = reinterpret_cast<word_t const *>(bytes.data() + saved_rows_offset_k);
        view.mapped_ = reinterpret_cast<mapped_slot_t const *>(bytes.data() + saved_mapped_offset_(rows_bytes));
        return expected<immutable_splus_tree_view>(view, success_k);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /** The bytes the rows and the mapped values take. */
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes(size_); }

    /** The bytes a tree over @p keys_count elements stores: whole rows on every level, and a value
     *  per key for a map. */
    [[nodiscard]] static constexpr std::size_t size_bytes(std::size_t keys_count) noexcept {
        level_nodes_t leaf_counts {};
        std::size_t const levels_count = count_levels_(keys_count, leaf_counts);
        std::size_t nodes_count = 0;
        for (std::size_t level = 0; level < levels_count; ++level) nodes_count += leaf_counts[level];
        return nodes_count * format_t::bytes_per_row_k + keys_count * saved_mapped_bytes_k;
    }

    /** The bytes the saved form spans, so a reader can step past it to whatever follows. */
    [[nodiscard]] std::size_t saved_bytes() const noexcept {
        return saved_bytes_after_(level_firsts_[levels_count_] * format_t::bytes_per_row_k);
    }

    /** How many keys order below @p wanted, which is also the position of its lower bound. */
    [[nodiscard]] std::size_t rank(key_t wanted) const noexcept {
        if (levels_count_ == 0) return 0;
        std::size_t node = 0;
        for (std::size_t level = 0; level + 1 < levels_count_; ++level)
            node = node * fanout_k + count_below_in_row<format_t, kit_t>(row_(level, node), wanted);
        return node * keys_per_row_k + count_below_in_row<format_t, kit_t>(row_(levels_count_ - 1, node), wanted);
    }

    /** The position of @p wanted, or @c key_not_found_k. Where keys repeat, the first of them. */
    expected<std::size_t> find(key_t wanted) const noexcept {
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
        return iterator(this, min_of(ordinal, size_));
    }

    /** A walk from the first key not below @p wanted. */
    [[nodiscard]] iterator lower_bound(key_t wanted) const noexcept { return iterator(this, rank(wanted)); }

  private:
    /** The bytes one mapped value takes in the saved form, zero for a set. */
    static constexpr std::size_t saved_mapped_bytes_k = is_mapping<value_t> ? sizeof(mapped_slot_t) : 0;

    /** Where the mapped values start after @p rows_bytes of rows, on their own alignment. */
    static constexpr std::size_t saved_mapped_offset_(std::size_t rows_bytes) noexcept {
        std::size_t const after_rows = saved_rows_offset_k + rows_bytes;
        std::size_t const alignment = alignof(mapped_slot_t);
        return (after_rows + alignment - 1) / alignment * alignment;
    }

    /** The bytes the saved form spans with @p rows_bytes of rows. */
    [[nodiscard]] std::size_t saved_bytes_after_(std::size_t rows_bytes) const noexcept {
        return saved_mapped_offset_(rows_bytes) + size_ * saved_mapped_bytes_k;
    }

    [[nodiscard]] word_t const *row_(std::size_t level, std::size_t node) const noexcept {
        return words_ + (level_firsts_[level] + node) * format_t::words_per_row_k;
    }

    /** Sets the size, the levels and where each starts for @p keys_count keys, answering how many
     *  nodes they hold together. */
    std::size_t lay_out_(std::size_t keys_count) noexcept {
        level_nodes_t leaf_counts {};
        std::size_t const levels_count = count_levels_(keys_count, leaf_counts);
        std::size_t nodes_count = 0;
        for (std::size_t level = 0; level < levels_count; ++level) {
            level_firsts_[level] = nodes_count;
            nodes_count += leaf_counts[levels_count - 1 - level];
        }
        level_firsts_[levels_count] = nodes_count;
        size_ = keys_count;
        levels_count_ = levels_count;
        return nodes_count;
    }

    /** Fills @p leaf_counts with the nodes on each level from the leaves up, and returns how many
     *  levels there are. */
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

/**
 *  @brief An immutable S+ tree over @p value_type_, built once from a sorted span, whose nodes are
 *      rows of @p keys_per_row_ keys searched by @p row_kit_type_. A set holds one allocation of
 *      rows; a map holds a second one of mapped values, indexed by rank.
 *
 *  @tparam keys_per_row_ Keys per node, which is @p default_row_bytes_k worth by default.
 */
template <typename value_type_,
          std::size_t keys_per_row_ =
              keys_per_row<typename mapping_key_type_or_itself<value_type_>::type>(default_row_bytes_k),
          row_kit row_kit_type_ = native_row_kit_t, typename allocator_type_ = default_allocator<value_type_>>
class immutable_splus_tree {
  public:
    /** The in-place view every lookup goes through, over this tree's storage or saved bytes. */
    using view_t = immutable_splus_tree_view<value_type_, keys_per_row_, row_kit_type_>;

    /** Every stored element: the key itself for a set, a @c mapping of both halves for a map. */
    using value_t = value_type_;
    using value_type = value_t; // ? STL compatibility

    /** The key half the rows are searched on, which is the whole element for a set. */
    using key_t = typename view_t::key_t;
    using key_type = key_t; // ? STL compatibility

    /** The mapped half, held in rank order beside the rows, and @c void for a set. */
    using mapped_t = typename view_t::mapped_t;
    using mapped_type = mapped_t; // ? STL compatibility

    /** What one entry of the mapped array holds: an empty stand-in when the tree is a set. */
    using mapped_slot_t = typename view_t::mapped_slot_t;

    using is_associative = typename view_t::is_associative;

    using format_t = typename view_t::format_t;
    using word_t = typename view_t::word_t;
    using kit_t = row_kit_type_;
    using allocator_t = allocator_type_;
    using allocator_type = allocator_t; // ? STL compatibility

    /** A forward walk over the keys in order, as the view walks them. */
    using iterator = typename view_t::iterator;

    static constexpr std::size_t keys_per_row_k = keys_per_row_;
    static constexpr std::size_t fanout_k = keys_per_row_ + 1;

    static_assert(std::allocator_traits<allocator_t>::propagate_on_container_move_assignment::value,
                  "immutable_splus_tree requires allocators that propagate on move assignment");

  private:
    using words_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<word_t>;
    using mapped_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<mapped_slot_t>;

    basic_vector<word_t, words_allocator_t> words_;
    basic_vector<mapped_slot_t, mapped_allocator_t> mapped_;
    view_t view_; // ? Over `words_` and `mapped_`, whose buffers a move hands over unchanged

  public:
    immutable_splus_tree() noexcept = default;

    explicit immutable_splus_tree(allocator_t allocator) noexcept
        : words_(words_allocator_t(allocator)), mapped_(mapped_allocator_t(allocator)) {}

    immutable_splus_tree(immutable_splus_tree &&other) noexcept
        : words_(std::move(other.words_)), mapped_(std::move(other.mapped_)),
          view_(std::exchange(other.view_, view_t {})) {}

    immutable_splus_tree &operator=(immutable_splus_tree &&other) noexcept {
        if (this == &other) return *this;
        words_ = std::move(other.words_);
        mapped_ = std::move(other.mapped_);
        view_ = std::exchange(other.view_, view_t {});
        return *this;
    }

    immutable_splus_tree(immutable_splus_tree const &) = delete;
    immutable_splus_tree &operator=(immutable_splus_tree const &) = delete;

    /**
     *  @brief Builds the tree over @p sorted, which may repeat keys. A map keeps the mapped values
     *      in the order they arrive, so a repeated key reads back the value that came first.
     *
     *  @return The tree.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p sorted is out of order.
     */
    static expected<immutable_splus_tree> make(std::span<value_t const> sorted, allocator_t allocator = {}) noexcept {
        for (std::size_t index = 1; index < sorted.size(); ++index)
            if (mapping_key_or_itself(sorted[index]) < mapping_key_or_itself(sorted[index - 1]))
                return invalid_argument_k;

        immutable_splus_tree tree(std::move(allocator));
        view_t &view = tree.view_;
        std::size_t const nodes_count = view.lay_out_(sorted.size());
        if (nodes_count > size_max_k / fanout_k / format_t::words_per_row_k) return out_of_memory_heap_k;
        status_t const status = tree.words_.resize(nodes_count * format_t::words_per_row_k);
        if (failed(status)) return status;
        view.words_ = tree.words_.data();
        std::size_t const levels_count = view.levels_count_;
        if (levels_count == 0) return expected<immutable_splus_tree>(std::move(tree), success_k);

        std::size_t const leaves_level = levels_count - 1;
        std::size_t const leaves_count = nodes_count - view.level_firsts_[leaves_level];
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
            view.mapped_ = tree.mapped_.data();
        }

        // A node `levels_up` levels above the leaves starts at leaf `node * fanout^levels_up`.
        std::size_t child_leaf_span = 1;
        for (std::size_t level = leaves_level; level-- > 0;) {
            std::size_t const nodes_on_level = view.level_firsts_[level + 1] - view.level_firsts_[level];
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

    [[nodiscard]] std::size_t size() const noexcept { return view_.size(); }
    [[nodiscard]] bool empty() const noexcept { return view_.empty(); }

    /** The bytes this tree stores, rows and mapped values together. */
    [[nodiscard]] std::size_t size_bytes() const noexcept {
        return words_.size() * sizeof(word_t) + mapped_.size() * sizeof(mapped_slot_t);
    }

    /** The bytes a tree over @p keys_count elements stores: whole rows on every level, and a value
     *  per key for a map. */
    [[nodiscard]] static constexpr std::size_t size_bytes(std::size_t keys_count) noexcept {
        return view_t::size_bytes(keys_count);
    }

    /** The stored words, root level first, for writing the tree out. */
    [[nodiscard]] std::span<word_t const> words() const noexcept { return {words_.data(), words_.size()}; }

    /** This tree searched in place, valid while the tree lives and stays where it is. */
    [[nodiscard]] view_t const &view() const noexcept { return view_; }

    /** The bytes @c save writes: the header, the rows, and the mapped values on their alignment. */
    [[nodiscard]] std::size_t saved_bytes() const noexcept { return view_.saved_bytes(); }

    /**
     *  @brief Writes the saved form into the front of @p bytes, for @c immutable_splus_tree_view to
     *      search in place.
     *
     *  @par Statuses:
     *  - @c invalid_argument_k @p bytes is shorter than @c saved_bytes, and nothing is written.
     */
    status_t save(std::span<std::byte> bytes) const noexcept
        requires std::is_trivially_copyable_v<mapped_slot_t>
    {
        if (bytes.size() < saved_bytes()) return invalid_argument_k;
        splus_tree_header_t const header {.count = view_.size(),
                                          .keys_per_row = keys_per_row_k,
                                          .key_bytes = sizeof(key_t),
                                          .mapped_bytes = view_t::saved_mapped_bytes_k};
        std::memcpy(bytes.data(), &header, sizeof(header));
        std::size_t const rows_bytes = words_.size() * sizeof(word_t);
        if (rows_bytes) std::memcpy(bytes.data() + view_t::saved_rows_offset_k, words_.data(), rows_bytes);
        std::size_t const mapped_offset = view_t::saved_mapped_offset_(rows_bytes);
        std::memset(bytes.data() + view_t::saved_rows_offset_k + rows_bytes, 0,
                    mapped_offset - view_t::saved_rows_offset_k - rows_bytes);
        if constexpr (is_mapping<value_t>)
            if (!mapped_.empty())
                std::memcpy(bytes.data() + mapped_offset, mapped_.data(), mapped_.size() * sizeof(mapped_slot_t));
        return success_k;
    }

    /** How many keys order below @p wanted, which is also the position of its lower bound. */
    [[nodiscard]] std::size_t rank(key_t wanted) const noexcept { return view_.rank(wanted); }

    /** The position of @p wanted, or @c key_not_found_k. Where keys repeat, the first of them. */
    expected<std::size_t> find(key_t wanted) const noexcept { return view_.find(wanted); }

    /** The key at position @p ordinal in sorted order, which must be below @c size. */
    [[nodiscard]] key_t select(std::size_t ordinal) const noexcept { return view_.select(ordinal); }

    /** The value the key at position @p ordinal carries, which must be below @c size. */
    [[nodiscard]] mapped_slot_t const &mapped_at(std::size_t ordinal) const noexcept
        requires is_mapping<value_t>
    {
        return view_.mapped_at(ordinal);
    }

    [[nodiscard]] iterator begin() const noexcept { return view_.begin(); }
    [[nodiscard]] iterator end() const noexcept { return view_.end(); }

    /** A walk from position @p ordinal, or @c end when there is no such position. */
    [[nodiscard]] iterator at_rank(std::size_t ordinal) const noexcept { return view_.at_rank(ordinal); }

    /** A walk from the first key not below @p wanted. */
    [[nodiscard]] iterator lower_bound(key_t wanted) const noexcept { return view_.lower_bound(wanted); }

  private:
    [[nodiscard]] word_t *row_(std::size_t level, std::size_t node) noexcept {
        return words_.data() + (view_.level_firsts_[level] + node) * format_t::words_per_row_k;
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
static_assert(tagged_collection<immutable_splus_set<std::uint64_t>::view_t>,
              "a view names its element, its key and its shape as the tree it reads does");
static_assert(std::is_same_v<immutable_splus_map<std::uint64_t, double>::key_type, std::uint64_t> &&
                  std::is_same_v<immutable_splus_map<std::uint64_t, double>::mapped_type, double>,
              "a map splits into the two halves the alias was given");

} // namespace ashvardanian::smashtable
