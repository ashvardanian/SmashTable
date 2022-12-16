#pragma once
#include <functional> // `std::less` as default
#include <memory>     // `std::allocator` as default
#include <optional>   // `std::optional` for "expected"
#include <set>        // `std::set` for entries
#include <vector>     // `std::vector` for watches
#include <random>     // `std::uniform_int_distribution` fir sampling

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
 * @brief Atomic (in DBMS and Set Theory sense) Transactional Store on top of a
 * Standard Templates Library. It can be used as a Key-Value store, if you store
 * @c `std::pair` as entries.
 *
 * @section Design Goals
 *
 * - Atomicity of batch operations.
 * - Simplicity and familiarity.
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
    using entry_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<entry_t>;
    using entry_set_t = std::set< //
        entry_t,
        entry_comparator_t,
        entry_allocator_t>;
    using entry_iterator_t = typename entry_set_t::iterator;

    using watches_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<watched_identifier_t>;
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

        transaction_t(store_t& set) noexcept(false) : store_(&set), generation_(set.new_generation()) {}
        watch_t missing_watch() const noexcept { return watch_t {generation_, true}; }
        store_t& store_ref() noexcept { return *store_; }
        store_t const& store_ref() const noexcept { return *store_; }

      public:
        transaction_t(transaction_t&&) noexcept = default;
        transaction_t& operator=(transaction_t&&) noexcept = default;
        transaction_t(transaction_t const&) = delete;
        transaction_t& operator=(transaction_t const&) = delete;
        generation_t generation() const noexcept { return generation_; }

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

        [[nodiscard]] status_t reserve(std::size_t size) noexcept {
            return invoke_safely([&] { watches_.reserve(size); });
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

        /**
         * @brief Finds a member @b equal to the given @ref `comparable`.
         *        You may want to `watch()` the received object, it's not done by default.
         *        Unlike `consistent_set_gt::find()`, will include the entries added to this
         *        transaction.
         *
         * @ref `comparable`            Object, comparable to @c `element_t` and convertible to @c `identifier_t`.
         * @param callback_found        Callback to receive an `element_t const &`. Ideally, `noexcept.`
         * @param callback_missing      Callback to be triggered, if nothing was found.
         */
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

        /**
         * @brief Finds the first member @b greater than the given @ref `comparable`.
         *        You may want to `watch()` the received object, it's not done by default.
         *        Unlike `consistent_set_gt::find()`, will include the entries added to this
         *        transaction.
         *
         * @ref `comparable`            Object, comparable to @c `element_t` and convertible to @c `identifier_t`.
         * @param callback_found        Callback to receive an `element_t const &`. Ideally, `noexcept.`
         * @param callback_missing      Callback to be triggered, if nothing was found.
         */
        template <typename comparable_at = identifier_t,
                  typename callback_found_at = no_op_t,
                  typename callback_missing_at = no_op_t>
        [[nodiscard]] status_t upper_bound(comparable_at&& comparable,
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
                status = store.upper_bound(external_previous_id, callback_external_found, callback_external_missing);
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
         *
         * In more detail:
         * - All the updates staged in DB will be reverted.
         * - All the updates will in this Transaction will be lost.
         * - All the watches will be lost.
         * - New generation will be assigned.
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
            generation_ = store.new_generation();
            return {success_k};
        }

        /**
         * @brief Rolls-back a previously "staged" transaction.
         *
         * In more detail:
         * - All the updates will be reverted in the DB.
         * - All the updates will re-emerge in this Transaction.
         * - All the watches will be lost.
         * - New generation will be assigned.
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
            generation_ = store.new_generation();
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
    std::size_t visible_count_ {0};

    friend class transaction_t;

    consistent_set_gt() noexcept(false) {}
    generation_t new_generation() noexcept { return ++generation_; }

    template <typename callback_at = no_op_t>
    void erase_visible(entry_iterator_t begin, entry_iterator_t end, callback_at&& callback = {}) noexcept {
        entry_iterator_t& current = begin;
        while (current != end)
            if (current->visible) {
                callback(*current);
                current = entries_.erase(current);
                --visible_count_;
            }
            else
                ++current;
    }

    void unmask_and_compact(entry_iterator_t begin, entry_iterator_t end, generation_t generation_to_unmask) noexcept {
        entry_iterator_t& current = begin;
        entry_iterator_t last_visible_entry = end;
        for (; current != end; ++current) {
            auto keep_this = current->generation == generation_to_unmask;
            if (keep_this) {
                visible_count_ += !current->visible;
                current->visible = true;
            }

            if (!current->visible)
                continue;

            // Older revisions must die
            if (last_visible_entry != end) {
                entries_.erase(last_visible_entry);
                --visible_count_;
            }
            last_visible_entry = current;
        }
    }

    /**
     * @brief Atomically @b updates-or-inserts a collection of entries.
     * Either all entries will be inserted, or all will fail.
     * This operation is identical to creating and committing
     * a transaction with all the same elements put into it.
     *
     * @section Why not take R-Value?
     * We want this operation to be consistent, as the rest of the container,
     * so we need a place to return all the objects, if the operation fails.
     * With R-Value, the batch would be lost.
     *
     * @param[inout] sources   Collection of entries to import.
     * @return status_t        Can fail, if out of memory.
     */
    [[nodiscard]] status_t upsert(entry_set_t& sources) noexcept {
        for (auto source = sources.begin(); source != sources.end();) {
            bool should_compact = source->visible;
            auto source_node = sources.extract(source++);
            auto range_end = entries_.insert(std::move(source_node)).position;
            ++visible_count_;
            if (should_compact) {
                auto range_start = entries_.lower_bound(range_end->element);
                erase_visible(range_start, range_end);
            }
        }
        return {success_k};
    }

  public:
    [[nodiscard]] std::size_t size() const noexcept { return visible_count_; }
    [[nodiscard]] bool empty() const noexcept { return !visible_count_; }

    /**
     * @brief Creates a new collection of this type without throwing exceptions.
     * If fails, an empty @c `std::optional` is returned.
     */
    [[nodiscard]] static std::optional<store_t> make() noexcept {
        std::optional<store_t> result;
        invoke_safely([&] { result.emplace(store_t {}); });
        return result;
    }

    /**
     * @brief Starts a transaction with a new sequence number.
     * If succeeded, that transaction can later be reset to reuse the memory.
     * If fails, an empty @c `std::optional` is returned.
     */
    [[nodiscard]] std::optional<transaction_t> transaction() noexcept {
        std::optional<transaction_t> result;
        invoke_safely([&] { result.emplace(transaction_t {*this}); });
        return result;
    }

    /**
     * @brief Moves a single new @param element into the container.
     * This operation is identical to creating and committing
     * a single upsert transaction.
     *
     * @param[in] element   The element to import.
     * @return status_t     Can fail, if out of memory.
     */
    [[nodiscard]] status_t upsert(element_t&& element) noexcept {
        generation_t generation = new_generation();
        return invoke_safely([&] {
            auto entry = entry_t {std::move(element)};
            entry.generation = generation;
            entry.deleted = false;
            entry.visible = true;
            auto range_end = entries_.insert(std::move(entry)).first;
            auto range_start = entries_.lower_bound(range_end->element);
            ++visible_count_;
            erase_visible(range_start, range_end);
        });
    }

    /**
     * @brief Atomically @b updates-or-inserts a batch of entries.
     * Either all entries will be inserted, or all will fail.
     *
     * The dereferencing operator of the passed @param iterator
     * should return R-Value references of @c `element_t`.
     * @see `std::make_move_iterator()`.
     *
     * @param begin
     * @param end
     */
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

    /**
     * @brief Finds a member @b equal to the given @ref `comparable`.
     *
     * @ref `comparable`            Object, comparable to @c `element_t` and convertible to @c `identifier_t`.
     * @param callback_found        Callback to receive an `element_t const &`. Ideally, `noexcept.`
     * @param callback_missing      Callback to be triggered, if nothing was found.
     */
    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t find(comparable_at&& comparable,
                                callback_found_at&& callback_found,
                                callback_missing_at&& callback_missing = {}) const noexcept {

        auto range = entries_.equal_range(std::forward<comparable_at>(comparable));

        // Skip all the invisible entries
        while (range.first != range.second && !range.first->visible)
            ++range.first;

        // Check if there are no visible entries at all
        return range.first != range.second //
                   ? invoke_safely([&] { callback_found(*range.first); })
                   : invoke_safely(std::forward<callback_missing_at>(callback_missing));
    }

    /**
     * @brief Finds the first member @b greater than the given @ref `comparable`.
     *
     * @ref `comparable`            Object, comparable to @c `element_t` and convertible to @c `identifier_t`.
     * @param callback_found        Callback to receive an `element_t const &`. Ideally, `noexcept.`
     * @param callback_missing      Callback to be triggered, if nothing was found.
     */
    template <typename comparable_at = identifier_t,
              typename callback_found_at = no_op_t,
              typename callback_missing_at = no_op_t>
    [[nodiscard]] status_t upper_bound(comparable_at&& comparable,
                                       callback_found_at&& callback_found,
                                       callback_missing_at&& callback_missing = {}) const noexcept {

        auto iterator = entries_.upper_bound(std::forward<comparable_at>(comparable));

        // Skip all the invisible entries
        while (iterator != entries_.end() && !iterator->visible)
            ++iterator;

        return iterator != entries_.end() //
                   ? invoke_safely([&] { callback_found(*iterator); })
                   : invoke_safely(std::forward<callback_missing_at>(callback_missing));
    }

    /**
     * @brief Implements a heterogeneous lookup for all the entries falling in
     * between the @ref `lower` and the @ref `upper`. Degrades to `equal_range()`,
     * if they are the same.
     */
    template <typename lower_at = identifier_t, typename upper_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t range(lower_at&& lower, upper_at&& upper, callback_at&& callback) const noexcept {
        auto lower_iterator = entries_.lower_bound(std::forward<lower_at>(lower));
        auto const upper_iterator = entries_.lower_bound(std::forward<upper_at>(upper));
        for (; lower_iterator != upper_iterator; ++lower_iterator)
            if (lower_iterator->visible)
                if (auto status = invoke_safely([&] { callback(lower_iterator->element); }); !status)
                    return status;

        return {success_k};
    }

    /**
     * @brief Implements a heterogeneous lookup for all the entries falling in
     * between the @ref `lower` and the @ref `upper`. Degrades to `equal_range()`,
     * if they are the same. Allows in-place @b modification.
     */
    template <typename lower_at = identifier_t, typename upper_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t range(lower_at&& lower, upper_at&& upper, callback_at&& callback) noexcept {
        generation_t generation = new_generation();
        auto lower_iterator = entries_.lower_bound(std::forward<lower_at>(lower));
        auto const upper_iterator = entries_.lower_bound(std::forward<upper_at>(upper));
        for (; lower_iterator != upper_iterator; ++lower_iterator)
            if (lower_iterator->visible)
                if (auto status = invoke_safely(
                        [&] { callback(lower_iterator->element), lower_iterator->generation = generation; });
                    !status)
                    return status;

        return {success_k};
    }

    /**
     * @brief Erases all the entries falling in between the @ref `lower` and the @ref `upper`.
     */
    template <typename lower_at = identifier_t, typename upper_at = identifier_t, typename callback_at = no_op_t>
    [[nodiscard]] status_t erase_range(lower_at&& lower, upper_at&& upper, callback_at&& callback) noexcept {
        auto lower_iterator = entries_.lower_bound(std::forward<lower_at>(lower));
        auto const upper_iterator = entries_.lower_bound(std::forward<upper_at>(upper));
        erase_visible(lower_iterator, upper_iterator, std::forward<callback_at>(callback));
        return {success_k};
    }

    /**
     * @brief Removes all the data from the container.
     */
    [[nodiscard]] status_t clear() noexcept {
        entries_.clear();
        generation_ = 0;
        return {success_k};
    }

    /**
     * @brief Optimization, that informs container to pre-allocate memory in-advance.
     * Doesn't guarantee, that the following "upserts" won't fail with "out of memory".
     */
    [[nodiscard]] status_t reserve(std::size_t) noexcept { return {}; }

    /**
     * @brief Uniformly Random-Samples just one entry from the container.
     * Searches within entries that compare equal to the provided @ref `comparable`.
     *
     * ! This implementation is extremely inefficient and requires a two-pass approach.
     * ! On the first run we estimate the number of entries matching the @ref `comparable`.
     * ! On the second run we choose a random integer below the number of matched entries
     * ! and loop until we advance the STL iterator enough.
     *
     * ! Depends on the `equal_range`. Use the Reservoir Sampling overload with
     * ! temporary memory if you want to sample more than one entry.
     *
     * @param[in] generator     Random generator to be invoked on the internal distribution.
     * @param[in] callback      Callback to receive the sampled @c `element_t` entry.
     */
    template <typename lower_at, typename upper_at, typename generator_at, typename callback_at = no_op_t>
    [[nodiscard]] status_t sample_range(lower_at&& lower,
                                        upper_at&& upper,
                                        generator_at&& generator,
                                        callback_at&& callback) const noexcept {

        std::size_t count = 0;
        auto status = range(lower, upper, [&](element_t const&) noexcept { ++count; });
        if (!status)
            return status;

        if (!count)
            return {};

        std::uniform_int_distribution<std::size_t> distribution {0, count - 1};
        std::size_t matches_to_skip = distribution(generator);
        return range(lower, upper, [&](element_t const& element) noexcept {
            if (matches_to_skip)
                --matches_to_skip;
            else
                callback(element);
        });
    }

    /**
     * @brief Implements Uniform Reservoir Sampling into the provided output buffer.
     * Searches within entries that compare equal to the provided @ref `comparable`.
     *
     * @param[in] generator             Random generator to be invoked on the internal distribution.
     * @param[inout] seen               The number of previously seen entries. Zero, by default.
     * @param[in] reservoir_capacity    The number of entries that can fit in @ref `reservoir`.
     * @param[in] reservoir             Iterator to the beginning of the output reservoir.
     */
    template <typename lower_at, typename upper_at, typename generator_at, typename output_iterator_at>
    [[nodiscard]] status_t sample_range(lower_at&& lower,
                                        upper_at&& upper,
                                        generator_at&& generator,
                                        std::size_t& seen,
                                        std::size_t reservoir_capacity,
                                        output_iterator_at&& reservoir) const noexcept {

        using output_iterator_t = std::remove_reference_t<output_iterator_at>;
        using output_category_t = typename std::iterator_traits<output_iterator_t>::iterator_category;
        static_assert(std::is_same<std::random_access_iterator_tag, output_category_t>(), "Must be random access!");

        auto sampler = [&](element_t const& element) noexcept {
            if (seen < reservoir_capacity)
                reservoir[seen] = element;

            else {
                std::uniform_int_distribution<std::size_t> distribution {0, seen};
                auto slot_to_replace = distribution(generator);
                if (slot_to_replace < reservoir_capacity)
                    reservoir[slot_to_replace] = element;
            }

            ++seen;
        };
        return range(std::forward<lower_at>(lower), std::forward<upper_at>(upper), sampler);
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