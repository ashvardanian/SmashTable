#pragma once
#include <functional> // `std::less` as default
#include <memory>     // `std::allocator` as default
#include <optional>   // `std::optional` for "expected"
#include <set>        // `std::set` for entries
#include <vector>     // `std::vector` for watches

#include "status.hpp"

namespace av {

template <typename callable_at>
consistent_set_status_t invoke_safely(callable_at&& callable) noexcept {
    if constexpr (noexcept(callable())) {
        callable();
        return {success_k};
    }
    else {
        try {
            callable();
            return {success_k};
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
 * @brief Atomic (in DBMS sense) Transactional Store on top of a Binary Search
 * Tree. It can be a Key-Value store, if you store `std::pair` as entries.
 *
 * @section Design Goals
 * !> Atomicity of batch operations.
 * !> Simplicity and familiarity.
 * For performance, consistency, Multi-Version Concurrency control and others,
 * check out the `set_avl_gt`.
 *
 * @section Heterogeneous Comparisons
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

    using versioning_t = element_versioning_gt<element_t, comparator_t>;
    using identifier_t = typename versioning_t::identifier_t;
    using generation_t = typename versioning_t::generation_t;
    using status_t = typename versioning_t::status_t;
    using dated_identifier_t = typename versioning_t::dated_identifier_t;
    using watch_t = typename versioning_t::watch_t;
    using watched_identifier_t = typename versioning_t::watched_identifier_t;
    using entry_t = typename versioning_t::entry_t;
    using entry_comparator_t = typename versioning_t::entry_comparator_t;

  private:
    using entry_allocator_t = typename allocator_t::template rebind<entry_t>::other;
    using entry_set_t = std::set< //
        entry_t,
        entry_comparator_t,
        entry_allocator_t>;
    using entry_iterator_t = typename entry_set_t::iterator;

    using watches_allocator_t = typename allocator_t::template rebind<watched_identifier_t>::other;
    using watches_array_t = std::vector<watched_identifier_t, watches_allocator_t>;
    using watch_iterator_t = typename watches_array_t::iterator;

    using store_t = consistent_set_gt;

  public:
    class transaction_t {

        friend store_t;
        enum class stage_t {
            created_k,
            staged_k,
            commited_k,
        };

        store_t* store_ {nullptr};
        entry_set_t changes_ {};
        watches_array_t watches_ {};
        generation_t generation_ {0};
        stage_t stage_ {stage_t::created_k};

        transaction_t(store_t& set) noexcept(false) : store_(&set) {}
        void date(generation_t generation) noexcept { generation_ = generation; }
        watch_t missing_watch() const noexcept { return watch_t {generation_, true}; }
        store_t& store_ref() noexcept { return *store_; }
        store_t const& store_ref() const noexcept { return *store_; }

      public:
        transaction_t(transaction_t&&) noexcept = default;
        transaction_t& operator=(transaction_t&&) noexcept = default;
        transaction_t(transaction_t const&) = delete;
        transaction_t& operator=(transaction_t const&) = delete;

        [[nodiscard]] status_t upsert(element_t&& element) noexcept {
            return invoke_safely([&] {
                auto iterator = changes_.lower_bound(element);
                if (iterator == changes_.end() || !entry_comparator_t {}.same(iterator->element, element))
                    iterator = changes_.emplace_hint(iterator, std::move(element));
                else
                    iterator->element = std::move(element);
                iterator->generation = generation_;
                iterator->deleted = false;
                iterator->visible = false;
            });
        }

        [[nodiscard]] status_t erase(identifier_t const& id) noexcept {
            return invoke_safely([&] {
                auto iterator = changes_.lower_bound(id);
                if (iterator == changes_.end() || !entry_comparator_t {}.same(iterator->element, id))
                    iterator = changes_.emplace_hint(iterator, id);
                iterator->generation = generation_;
                iterator->deleted = true;
                iterator->visible = false;
            });
        }

        [[nodiscard]] status_t watch(identifier_t const& id) noexcept {
            return store_ref().find(
                id,
                [&](entry_t const& entry) {
                    watches_.push_back({identifier_t {entry.element}, watch_t {entry.generation, entry.deleted}});
                },
                [&] {
                    watches_.push_back({id, missing_watch()});
                });
        }

        [[nodiscard]] status_t watch(entry_t const& entry) noexcept {
            return invoke_safely([&] {
                watches_.push_back({identifier_t {entry.element}, watch_t {entry.generation, entry.deleted}});
            });
        }

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t find(comparable_at&& comparable,
                                    callback_found_at&& callback_found,
                                    callback_missing_at&& callback_missing = {}) const noexcept {
            if (auto iterator = changes_.find(std::forward<comparable_at>(comparable)); iterator != changes_.end())
                return !iterator->deleted ? invoke_safely([&callback_found, &iterator] { callback_found(*iterator); })
                                          : invoke_safely(callback_missing);
            else
                return store_ref().find(std::forward<comparable_at>(comparable),
                                        std::forward<callback_found_at>(callback_found),
                                        std::forward<callback_missing_at>(callback_missing));
        }

        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t find_next(comparable_at&& comparable,
                                         callback_found_at&& callback_found,
                                         callback_missing_at&& callback_missing = {}) const noexcept {
            auto external_previous_id = identifier_t(comparable);
            auto internal_iterator = changes_.upper_bound(std::forward<comparable_at>(comparable));
            while (internal_iterator != changes_.end() && internal_iterator->deleted)
                ++internal_iterator;

            // Once picking the next smallest element from the global store,
            // we might face an entry, that was already deleted from here,
            // so this might become a multi-step process.
            auto faced_deleted_entry = false;
            auto callback_external_found = [&](element_t const& external_element) {
                // The simplest case is when we have an external object.
                if (internal_iterator == changes_.end())
                    return callback_found(external_element);

                element_t const& internal_element = internal_iterator->element;
                if (!entry_comparator_t {}(external_element, internal_element))
                    return callback_found(internal_element);

                // Check if this entry was deleted and we should try again.
                auto external_id = identifier_t(external_element);
                auto external_element_internal_state = changes_.find(external_element);
                if (external_element_internal_state != changes_.end() && external_element_internal_state->deleted) {
                    faced_deleted_entry = true;
                    external_previous_id = external_id;
                    return;
                }
                else
                    return callback_found(external_element);
            };
            auto callback_external_missing = [&] {
                if (internal_iterator == changes_.end())
                    return callback_missing();
                else {
                    element_t const& internal_element = internal_iterator->element;
                    return callback_found(internal_element);
                }
            };

            // Iterate until we find the a non-deleted external value
            auto& store = store_ref();
            auto status = status_t {};
            do {
                status = store.find_next(external_previous_id, callback_external_found, callback_external_missing);
            } while (faced_deleted_entry && status);
            return status;
        }

        [[nodiscard]] status_t stage() noexcept {
            // First, check if we have any collisions.
            auto& store = store_ref();
            auto entry_missing = missing_watch();
            for (auto const& id_and_watch : watches_) {
                auto consistency_violated = false;
                auto status = store.find(
                    id_and_watch.id,
                    [&](entry_t const& entry) noexcept { consistency_violated = entry != id_and_watch.watch; },
                    [&]() noexcept { consistency_violated = entry_missing != id_and_watch.watch; });
                if (consistency_violated)
                    return {consistent_set_errc_t::consistency_k};
                if (!status)
                    return status;
            }

            // Now all of our watches will be replaced with "links" to entries
            // we are merging into the main tree.
            watches_.clear();
            auto status = invoke_safely([&] { watches_.reserve(changes_.size()); });
            if (!status)
                return status;

            // No new memory allocations or failures are possible after that.
            // It is all safe.
            for (auto const& entry : changes_)
                watches_.push_back({identifier_t {entry.element}, watch_t {generation_, entry.deleted}});

            // Than just merge our current nodes.
            // The visibility will be updated later in the `commit`.
            store.entries_.merge(changes_);
            stage_ = stage_t::staged_k;
            return {success_k};
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
            auto& store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const& id_and_watch : watches_) {
                    // Heterogeneous `erase` is only coming in C++23.
                    dated_identifier_t dated {id_and_watch.id, id_and_watch.watch.generation};
                    if (auto iterator = store.entries_.find(dated); iterator != store.entries_.end())
                        store.entries_.erase(iterator);
                }

            watches_.clear();
            changes_.clear();
            stage_ = stage_t::created_k;
            return {success_k};
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
            auto& store = store_ref();
            if (stage_ == stage_t::staged_k)
                for (auto const& id_and_watch : watches_) {
                    dated_identifier_t dated {id_and_watch.id, id_and_watch.watch.generation};
                    auto source = store.entries_.find(dated);
                    auto node = store.entries_.extract(source);
                    changes_.insert(std::move(node));
                }

            watches_.clear();
            stage_ = stage_t::created_k;
            return {success_k};
        }

        [[nodiscard]] status_t commit() noexcept {
            if (stage_ != stage_t::staged_k)
                return {operation_not_permitted_k};

            // Once we make an entry visible,
            // if there are more than one with the same key,
            // the older generation must die.
            auto& store = store_ref();
            for (auto const& id_and_watch : watches_) {
                auto range = store.entries_.equal_range(id_and_watch.id);
                store.unmask_and_compact(range.first, range.second, id_and_watch.watch.generation);
            }

            stage_ = stage_t::created_k;
            return {success_k};
        }
    };

  private:
    entry_set_t entries_;
    generation_t generation_ {0};

    consistent_set_gt() noexcept(false) {}
    generation_t new_generation() noexcept { return ++generation_; }

    template <typename callback_at = no_op_t>
    void erase_visible(entry_iterator_t begin, entry_iterator_t end, callback_at&& callback = {}) noexcept {
        entry_iterator_t& current = begin;
        while (current != end)
            if (current->visible) {
                callback(*current);
                current = entries_.erase(current);
            }
            else
                ++current;
    }

    void unmask_and_compact(entry_iterator_t begin, entry_iterator_t end, generation_t generation_to_unmask) noexcept {
        entry_iterator_t& current = begin;
        entry_iterator_t last_visible_entry = end;
        for (; current != end; ++current) {
            auto keep_this = current->generation == generation_to_unmask;
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
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    [[nodiscard]] static std::optional<store_t> make() noexcept {
        std::optional<store_t> result;
        invoke_safely([&] { result.emplace(store_t {}); });
        return result;
    }

    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        generation_t generation = new_generation();
        std::optional<transaction_t> result;
        invoke_safely([&] { result.emplace(transaction_t {*this}).date(generation); });
        return result;
    }

    [[nodiscard]] status_t upsert(element_t&& element) noexcept {
        generation_t generation = new_generation();
        return invoke_safely([&] {
            auto entry = entry_t {std::move(element)};
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            auto range_end = entries_.insert(std::move(entry)).first;
            auto range_start = entries_.lower_bound(range_end->element);
            erase_visible(range_start, range_end);
        });
    }

    /**
     * @brief
     *
     * @section Why not take R-Value?
     * As this operation must be consistent, as the rest of the container,
     * we need a place to return all the objects, if the operation fails.
     * With R-Value, the batch would be lost.
     *
     * @param sources
     * @return status_t
     */
    [[nodiscard]] status_t upsert(entry_set_t& sources) noexcept {
        for (auto source = sources.begin(); source != sources.end();) {
            bool should_compact = source->visible;
            auto source_node = sources.extract(source++);
            auto range_end = entries_.insert(std::move(source_node)).position;
            if (should_compact) {
                auto range_start = entries_.lower_bound(range_end->element);
                erase_visible(range_start, range_end);
            }
        }
        return {success_k};
    }

    template <typename elements_begin_at, typename elements_end_at = elements_begin_at>
    [[nodiscard]] status_t upsert(elements_begin_at begin, elements_end_at end) noexcept {
        generation_t generation = new_generation();
        std::optional<entry_set_t> batch;
        auto batch_construction_status = invoke_safely([&] {
            batch = entry_set_t {};
            for (; begin != end; ++begin) {
                auto iterator = batch->emplace(*begin).first;
                iterator->generation = generation;
                iterator->visible = true;
            }
        });
        if (!batch_construction_status)
            return batch_construction_status;

        return upsert(batch.value());
    }

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find(comparable_at&& comparable,
                                callback_found_at&& callback_found,
                                callback_missing_at&& callback_missing = {}) const noexcept {
        auto range = entries_.equal_range(std::forward<comparable_at>(comparable));
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

    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find_next(comparable_at&& comparable,
                                     callback_found_at&& callback_found,
                                     callback_missing_at&& callback_missing = {}) const noexcept {
        auto iterator = entries_.upper_bound(comparable);
        if (iterator == entries_.end())
            return invoke_safely(std::forward<callback_missing_at>(callback_missing));

        // Skip all the invisible entries
        identifier_t next_id {iterator->element};
        while (true)
            if (iterator == entries_.end() || !entry_comparator_t {}.same(iterator->element, next_id))
                return invoke_safely(std::forward<callback_missing_at>(callback_missing));
            else if (!iterator->visible)
                ++iterator;
            else
                return invoke_safely([&] { callback_found(*iterator); });
    }

    /**
     * @brief Implements a heterogeneous lookup for all the entries falling into
     * the `equal_range`.
     */
    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t find_interval(comparable_at&& comparable, callback_at&& callback) const noexcept {
        auto range = entries_.equal_range(std::forward<comparable_at>(comparable));
        for (; range.first != range.second; ++range.first)
            if (range.first->visible)
                if (auto status = invoke_safely([&] { callback(range.first->element); }); !status)
                    return status;

        return {success_k};
    }

    /**
     * @brief Implements a heterogeneous lookup, allowing in-place @b
     * modification of the object, for all the entries falling into the
     * `equal_range`.
     */
    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t find_interval(comparable_at&& comparable, callback_at&& callback) noexcept {
        generation_t generation = new_generation();
        auto range = entries_.equal_range(std::forward<comparable_at>(comparable));
        for (; range.first != range.second; ++range.first)
            if (range.first->visible)
                if (auto status =
                        invoke_safely([&] { callback(range.first->element), range.first->generation = generation; });
                    !status)
                    return status;

        return {success_k};
    }

    /**
     * @brief Removes one or more objects from the collection, that fall into
     * the `equal_range`.
     */
    template <typename comparable_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t erase_interval(comparable_at&& comparable, callback_at&& callback = {}) noexcept {
        return invoke_safely([&] {
            auto range = entries_.equal_range(std::forward<comparable_at>(comparable));
            erase_visible(range.first, range.second, std::forward<callback_at>(callback));
        });
    }

    [[nodiscard]] status_t clear() noexcept {
        entries_.clear();
        generation_ = 0;
        return {success_k};
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

} // namespace av