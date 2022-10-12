#pragma once
#include <cstdint>       //
#include <system_error>  // `ENOMEM`
#include <functional>    // `std::less`
#include <set>           // `std::set`
#include <unordered_map> // `std::unordered_map`

namespace av {

enum consistent_set_errc_t {
    success_k = 0,
    unknown_k = -1,

    consistency_k,
    transaction_not_recoverable_k = ENOTRECOVERABLE,
    sequence_number_overflow_k = EOVERFLOW,

    out_of_memory_heap_k = ENOMEM,
    out_of_memory_arena_k = ENOBUFS,
    out_of_memory_disk_k = ENOSPC,

    invalid_argument_k = EINVAL,
    operation_in_progress_k = EINPROGRESS,
    operation_not_permitted_k = EPERM,
    operation_not_supported_k = EOPNOTSUPP,
    operation_would_block_k = EWOULDBLOCK,
    operation_canceled_k = ECANCELED,

    connection_broken_k = EPIPE,
    connection_aborted_k = ECONNABORTED,
    connection_already_in_progress_k = EALREADY,
    connection_refused_k = ECONNREFUSED,
    connection_reset_k = ECONNRESET,

};

/**
 * @brief
 *
 */
struct consistent_set_status_t {
    consistent_set_errc_t errc = consistent_set_errc_t::success_k;
    inline operator bool() const noexcept { return errc == consistent_set_errc_t::success_k; }
};

template <typename callable_at>
consistent_set_status_t invoke_safely(callable_at&& callable) noexcept {
    if constexpr (noexcept(callable())) {
        callable();
        return {};
    }
    else {
        try {
            callable();
            return {};
        }
        catch (std::bad_alloc const&) {
            return {consistent_set_errc_t::out_of_memory_heap_k};
        }
        catch (...) {
            return {consistent_set_errc_t::unknown_k};
        }
    }
}

/**
 * @brief Atomic (in DBMS sense) Transactional Store on top of a Binary Search Tree.
 * It can be a Key-Value store, if you store `std::pair` as entries.
 *
 * @section Design Goals
 * !> Atomicity of batch operations.
 * !> Simplicity and familiarity.
 * For performance, consistency, Multi-Version Concurrency control and others,
 * check out the `set_avl_gt`.
 *
 * @tparam element_at
 * @tparam comparator_at
 * @tparam allocator_at
 */
template < //
    typename element_at,
    typename comparator_at = std::less<element_at>,
    typename allocator_at = std::allocator<std::uint8_t>>
class consistent_set_gt {

  public:
    using element_t = element_at;
    using comparator_t = comparator_at;
    using allocator_t = allocator_at;

    static constexpr bool is_safe_to_move_k =              //
        std::is_nothrow_move_constructible<element_t>() && //
        std::is_nothrow_move_assignable<element_t>();

    static_assert(!std::is_reference<element_t>(), "Only value types are supported.");
    static_assert(std::is_nothrow_default_constructible<element_t>(), "We need an empty state.");
    static_assert(is_safe_to_move_k, "To make all the methods `noexcept`, the moves must be safe too.");

    using identifier_t = typename comparator_t::value_type;
    using generation_t = std::int64_t;
    using status_t = consistent_set_status_t;

    struct dated_identifier_t {
        identifier_t id;
        generation_t generation {0};
    };

    struct watch_t {
        generation_t generation {0};
        bool deleted {false};
    };

    struct entry_t {
        mutable element_t element;
        mutable generation_t generation {0};
        mutable bool deleted {false};
        mutable bool visible {true};

        entry_t() = default;
        entry_t(entry_t&&) noexcept = default;
        entry_t& operator=(entry_t&&) noexcept = default;
        entry_t(entry_t const&) noexcept = default;
        entry_t& operator=(entry_t const&) noexcept = default;
        entry_t(identifier_t id) noexcept : element(id) {}
        entry_t(element_t&& element) noexcept : element(std::move(element)) {}

        operator element_t const&() const& { return element; }
        bool operator==(watch_t const& watch) const noexcept {
            return watch.deleted == deleted && watch.generation == generation;
        }
        bool operator!=(watch_t const& watch) const noexcept {
            return watch.deleted != deleted || watch.generation != generation;
        }
    };

    struct entry_less_t {
        using is_transparent = void;

        dated_identifier_t date(entry_t const& entry) const noexcept { return {entry.element, entry.generation}; }

        bool less(identifier_t const& a, identifier_t const& b) const noexcept { return comparator_t {}(a, b); }
        bool less(entry_t const& a, identifier_t const& b) const noexcept { return comparator_t {}(a.element, b); }
        bool less(identifier_t const& a, entry_t const& b) const noexcept { return comparator_t {}(a, b.element); }

        bool less(entry_t const& a, entry_t const& b) const noexcept { return less(date(a), date(b)); }
        bool less(entry_t const& a, dated_identifier_t const& b) const noexcept { return less(date(a), b); }
        bool less(dated_identifier_t const& a, entry_t const& b) const noexcept { return less(a, date(b)); }
        bool less(dated_identifier_t const& a, dated_identifier_t const& b) const noexcept {
            comparator_t less;
            auto a_less_b = less(a.id, b.id);
            auto b_less_a = less(b.id, a.id);
            return !a_less_b && !b_less_a ? a.generation < b.generation : a_less_b;
        }

        template <typename first_at, typename second_at>
        bool operator()(first_at const& a, second_at const& b) const noexcept {
            return less(a, b);
        }

        template <typename first_at, typename second_at>
        bool same(first_at const& a, second_at const& b) const noexcept {
            return !less(a, b) && !less(b, a);
        }
    };

    using entry_allocator_t = typename allocator_t::template rebind<entry_t>::other;
    using entry_set_t = std::set< //
        entry_t,
        entry_less_t,
        entry_allocator_t>;
    using entry_iterator_t = typename entry_set_t::iterator;

    using watches_allocator_t = typename allocator_t::template rebind<std::pair<identifier_t const, watch_t>>::other;
    using watches_map_t = std::unordered_map< //
        identifier_t,
        watch_t,
        std::hash<identifier_t>,
        std::equal_to<identifier_t>,
        watches_allocator_t>;
    using watch_iterator_t = typename watches_map_t::iterator;

    using this_t = consistent_set_gt;

    class transaction_t {

        friend this_t;
        enum class stage_t {
            created_k,
            staged_k,
            commited_k,
        };

        this_t& set_;
        entry_set_t changes_ {};
        watches_map_t watches_ {};
        generation_t generation_ {0};
        stage_t stage_ {stage_t::created_k};

        transaction_t(this_t& set) noexcept(false) : set_(set) {}
        void date(generation_t generation) noexcept { generation_ = generation; }

      public:
        transaction_t(transaction_t&&) noexcept = default;
        transaction_t& operator=(transaction_t&& other) noexcept {
            std::swap(changes_, other.changes_);
            std::swap(watches_, other.watches_);
            std::swap(generation_, other.generation_);
            std::swap(stage_, other.stage_);
            return *this;
        }

        [[nodiscard]] status_t watch(identifier_t id) noexcept {
            return find(
                id,
                [&](entry_t const& entry) {
                    watches_.insert_or_assign(entry.element, watch_t {entry.generation, entry.deleted});
                },
                [&] {
                    watches_.insert_or_assign(id, watch_t {0, true});
                });
        }

        [[nodiscard]] status_t watch(entry_t const& entry) noexcept {
            return invoke_safely([&] {
                watches_.insert_or_assign(entry.element, watch_t {entry.generation, entry.deleted});
            });
        }

        template <typename callback_found_at, typename callback_missing_at>
        [[nodiscard]] status_t find(identifier_t id,
                                    callback_found_at&& callback_found,
                                    callback_missing_at&& callback_missing) noexcept {
            if (auto iterator = changes_.find(id); iterator != changes_.end())
                return !iterator->deleted ? invoke_safely([&callback_found, &iterator] { callback_found(*iterator); })
                                          : invoke_safely(callback_missing);
            else
                return set_.find(id,
                                 std::forward<callback_found_at>(callback_found),
                                 std::forward<callback_missing_at>(callback_missing));
        }

        template <typename callback_found_at, typename callback_missing_at>
        [[nodiscard]] status_t find_next(identifier_t id,
                                         callback_found_at&& callback_found,
                                         callback_missing_at&& callback_missing) noexcept {

            auto internal_iterator = changes_.upper_bound(id);
            while (internal_iterator != changes_.end() && internal_iterator->deleted)
                ++internal_iterator;

            // Once picking the next smallest element from the global store,
            // we might face an entry, that was already deleted from here,
            // so this might become a multi-step process.
            auto external_previous_id = id;
            auto faced_deleted_entry = false;
            do {
                auto status = set_.find_next(
                    external_previous_id,
                    [&](element_t const& external_element) {
                        // The simplest case is when we have an external object.
                        if (internal_iterator == changes_.end())
                            return callback_found(external_element);

                        entry_less_t less;
                        element_t const& internal_element = internal_iterator->element;
                        auto internal_is_smaller = less(internal_element, external_element);
                        auto external_is_smaller = less(external_element, internal_element);
                        auto same = !internal_is_smaller && //
                                    !external_is_smaller;

                        if (internal_is_smaller || same)
                            return callback_found(internal_element);

                        // Check if this entry was deleted and we should try again.
                        auto external_id = identifier_t(external_element);
                        auto external_element_internal_state = changes_.find(external_element);
                        if (external_element_internal_state != changes_.end() &&
                            external_element_internal_state->deleted) {
                            faced_deleted_entry = true;
                            external_previous_id = external_id;
                            return;
                        }
                        else
                            return callback_found(external_element);
                    },
                    [&] {
                        if (internal_iterator == changes_.end())
                            return callback_missing();
                        else {
                            element_t const& internal_element = internal_iterator->element;
                            return callback_found(internal_element);
                        }
                    });
                if (!status)
                    return status;
            } while (!faced_deleted_entry);
            return {};
        }

        [[nodiscard]] status_t upsert(element_t&& element) noexcept {
            return invoke_safely([&] {
                auto iterator = changes_.lower_bound(element);
                if (iterator == changes_.end() || !entry_less_t {}.same(iterator->element, element))
                    iterator = changes_.emplace_hint(iterator, std::move(element));
                else
                    iterator->element = std::move(element);
                iterator->generation = generation_;
                iterator->deleted = false;
            });
        }

        [[nodiscard]] status_t erase(identifier_t id) noexcept {
            return invoke_safely([&] {
                auto iterator = changes_.lower_bound(id);
                if (iterator == changes_.end() || !entry_less_t {}.same(iterator->element, id))
                    iterator = changes_.emplace_hint(iterator, id);
                iterator->generation = generation_;
                iterator->deleted = true;
            });
        }

        [[nodiscard]] status_t stage() noexcept {
            // First, check if we have any collisions.
            for (auto const& [id, watch] : watches_) {
                auto consistency_violated = false;
                auto status = set_.find(
                    id,
                    [&](entry_t const& entry) noexcept { consistency_violated = entry != watch; },
                    [&]() noexcept { consistency_violated = !watch.deleted; });
                if (consistency_violated)
                    return {consistent_set_errc_t::consistency_k};
                if (!status)
                    return status;
            }

            // Now all of our watches will be replaced with "links" to entries we
            // are merging into the main tree.
            watches_.clear();
            auto status = invoke_safely([&] { watches_.reserve(changes_.size()); });
            if (!status)
                return status;

            // No new memory allocations or failures are possible after that.
            // It is all safe.
            for (auto const& entry : changes_)
                watches_.insert_or_assign(entry.element, entry.generation);

            // Than just merge our current nodes.
            // The visibility will be updated later in the `commit`.
            set_.entries_.merge(changes_);
            stage_ = stage_t::staged_k;
            return {};
        }

        /**
         * @brief Resets the state of the transaction.
         * > All the updates staged in DB will be reverted.
         * > All the updates will in this Transaction will be lost.
         * > All the watches will be lost.
         */
        [[nodiscard]] status_t reset() noexcept {
            // If the transaction was "staged",
            // we must delete all the entries.
            if (stage_ == stage_t::staged_k)
                for (auto const& [id, watch] : watches_)
                    // Heterogeneous `erase` is only coming in C++23.
                    if (auto iterator = set_.entries_.find(dated_identifier_t {id, watch.generation});
                        iterator != set_.entries_.end())
                        set_.entries_.erase(iterator);

            watches_.clear();
            changes_.clear();
            stage_ = stage_t::created_k;
            return {};
        }

        /**
         * @brief Rolls-back a previously "staged" transaction.
         * > All the updates will be reverted in the DB.
         * > All the updates will re-emerge in this Transaction.
         * > All the watches will be lost.
         */
        [[nodiscard]] status_t rollback() noexcept {
            if (stage_ != stage_t::staged_k)
                return {operation_not_permitted_k};

            // If the transaction was "staged",
            // we must delete all the entries.
            if (stage_ == stage_t::staged_k)
                for (auto const& [id, watch] : watches_) {
                    auto source = set_.entries_.find(dated_identifier_t {id, watch.generation});
                    auto node = set_.entries_.extract(source);
                    changes_.insert(std::move(node));
                }

            watches_.clear();
            stage_ = stage_t::created_k;
            return {};
        }

        [[nodiscard]] status_t commit() noexcept {
            if (stage_ != stage_t::staged_k)
                return {operation_not_permitted_k};

            // Once we make an entry visible,
            // if there are more than one with the same key,
            // the older generation must die.
            for (auto const& [id, watch] : watches_) {
                auto range = set_.entries_.equal_range(id);
                set_.compact_outdated_entries(range.first, range.second, watch.generation);
            }

            return {};
        }
    };

  private:
    entry_set_t entries_;
    generation_t generation_ {0};

    consistent_set_gt() noexcept(false) {}

    void compact_visible_entries(entry_iterator_t begin, entry_iterator_t end) noexcept {
        entry_iterator_t& current = begin;
        for (; current != end; ++current)
            if (current->visible)
                entries_.erase(current);
    }

    void compact_outdated_entries(entry_iterator_t begin,
                                  entry_iterator_t end,
                                  generation_t generation_to_keep) noexcept {
        entry_iterator_t& current = begin;
        entry_iterator_t last_visible_entry = end;
        for (; current != end; ++current) {
            auto keep_this = current->generation == generation_to_keep;
            current->visible |= keep_this;
            if (!current->visible)
                continue;

            // Older revisions must die
            if (last_visible_entry != end)
                entries_.erase(last_visible_entry);
            last_visible_entry = current;
        }
    }

  public:
    [[nodiscard]] generation_t new_generation() noexcept { return ++generation_; }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    [[nodiscard]] static std::optional<consistent_set_gt> make() noexcept {
        std::optional<consistent_set_gt> result;
        invoke_safely([&] { result = consistent_set_gt {}; });
        return result;
    }

    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        generation_t generation = new_generation();
        std::optional<transaction_t> result;
        invoke_safely([&] {
            result = transaction_t(*this);
            result->date(generation);
        });
        return result;
    }

    [[nodiscard]] status_t upsert(element_t&& element) noexcept {
        generation_t generation = new_generation();
        return invoke_safely([&] {
            auto range_start = entries_.lower_bound(element);
            auto entry = entry_t {std::move(element)};
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            auto range_end = entries_.insert(std::move(entry)).first;
            compact_visible_entries(range_start, range_end);
        });
    }

    [[nodiscard]] status_t upsert(entry_set_t&& source) noexcept {
        generation_t generation = new_generation();
        for (auto source_it = source.begin(); source_it != source.end();) {
            source_it->generation = generation;
            source_it->visible = true;
            auto source_node = source.extract(source_it++);
            auto range_end = entries_.insert(std::move(source_node)).position;
            auto range_start = entries_.lower_bound(range_end->element);
            compact_visible_entries(range_start, range_end);
        }
        return {};
    }

    template <typename elements_begin_at, typename elements_end_at = elements_begin_at>
    [[nodiscard]] status_t upsert(elements_begin_at begin, elements_end_at end) noexcept {
        std::optional<entry_set_t> batch;
        auto batch_construction_status = invoke_safely([&] {
            batch = entry_set_t {};
            for (; begin != end; ++begin)
                batch->emplace(*begin);
        });
        if (!batch_construction_status)
            return batch_construction_status;

        return upsert(std::move(batch.value()));
    }

    template <typename callback_found_at, typename callback_missing_at>
    [[nodiscard]] status_t find(identifier_t id,
                                callback_found_at&& callback_found,
                                callback_missing_at&& callback_missing) const noexcept {
        auto range = entries_.equal_range(id);
        if (range.first == entries_.end())
            return invoke_safely(std::forward<callback_missing_at>(callback_missing));

        // Skip all the invisible entries
        while (range.first != range.second && !range.first->visible)
            ++range.first;

        // Check if there are no visible entries at all
        if (range.first == range.second)
            return invoke_safely(std::forward<callback_missing_at>(callback_missing));

        else
            return invoke_safely([&] { callback_found(*range.first); });
    }

    template <typename callback_found_at, typename callback_missing_at>
    [[nodiscard]] status_t find_next(identifier_t id,
                                     callback_found_at&& callback_found,
                                     callback_missing_at&& callback_missing) const noexcept {
        auto iterator = entries_.upper_bound(id);
        if (iterator == entries_.end())
            return invoke_safely(std::forward<callback_missing_at>(callback_missing));

        // Skip all the invisible entries
        identifier_t next_id {iterator->element};
        while (true)
            if (iterator == entries_.end() || !entry_less_t {}.same(iterator->element, next_id))
                return invoke_safely(std::forward<callback_missing_at>(callback_missing));
            else if (!iterator->visible)
                ++iterator;
            else
                return invoke_safely([&] { callback_found(*iterator); });
    }

    /**
     * @brief Implements a heterogeneous lookup for all the entries falling into the `equal_range`.
     */
    template <typename comparable_at, typename callback_found_at>
    [[nodiscard]] status_t find_all(comparable_at&& comparable, callback_found_at&& callback) const noexcept {
        auto range = entries_.equal_range(std::forward<comparable_at>(comparable));
        for (; range.first != range.second; ++range.first)
            if (range.first->visible)
                if (auto status = invoke_safely([&] { callback(range.first->element); }); !status)
                    return status;

        return {};
    }

    /**
     * @brief Implements a heterogeneous lookup, allowing in-place modification of the object,
     * for all the entries falling into the `equal_range`.
     */
    template <typename comparable_at, typename callback_found_at>
    [[nodiscard]] status_t find_all(comparable_at&& comparable, callback_found_at&& callback) noexcept {
        generation_t generation = new_generation();
        auto range = entries_.equal_range(std::forward<comparable_at>(comparable));
        for (; range.first != range.second; ++range.first)
            if (range.first->visible)
                if (auto status =
                        invoke_safely([&] { callback(range.first->element), range.first->generation = generation; });
                    !status)
                    return status;

        return {};
    }

    [[nodiscard]] status_t erase(identifier_t id) noexcept {
        return invoke_safely([&] {
            auto range = entries_.equal_range(id);
            compact_visible_entries(range.first, range.second);
        });
    }

    template <typename comparable_at>
    [[nodiscard]] status_t erase_all(comparable_at&& comparable) noexcept {
        return invoke_safely([&] {
            auto range = entries_.equal_range(std::forward<comparable_at>(comparable));
            compact_visible_entries(range.first, range.second);
        });
    }

    [[nodiscard]] status_t clear() noexcept {
        entries_.clear();
        generation_ = 0;
        return {};
    }
};

/**
 * @brief Unlike `std::set<>::merge`, this function overwrites existing values.
 *
 * https://en.cppreference.com/w/cpp/container/set#Member_types
 * https://en.cppreference.com/w/cpp/container/set/insert
 */
template <typename keys_at, typename compare_at, typename allocator_at>
void merge_overwrite(std::set<keys_at, compare_at, allocator_at>& target,
                     std::set<keys_at, compare_at, allocator_at>& source) noexcept {
    for (auto source_it = source.begin(); source_it != source.end();) {
        auto node = source.extract(source_it++);
        auto result = target.insert(std::move(node));
        if (!result.inserted)
            std::swap(*result.position, result.node.value());
    }
}

/**
 * @brief
 * Detects dead-locks and reports `operation_would_block`.
 */
class set_locked_gt {};

} // namespace av