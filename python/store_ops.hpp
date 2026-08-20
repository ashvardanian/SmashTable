/**
 *  @brief Turns one store instantiation into a @c store_ops_t table of function pointers.
 *  @author Ash Vardanian
 *  @file python/store_ops.hpp
 *  @date August 18, 2026
 *
 *  Every operation the binding performs on a store passes through here exactly once, at compile time.
 *  The bridge is what lets @c container.cpp name no concrete store: it resolves the callback-shaped,
 *  template-typed C++ surface into plain function pointers over @c void*, and reports what the core
 *  could not supply as a null slot rather than as a runtime refusal.
 *
 *  The ordered surface is gated on the C++ side by trailing @c requires clauses, so naming an absent
 *  member in an unguarded template is ill-formed rather than quietly removed from overload resolution.
 *  Every optional slot is therefore probed with @c requires before it is named.
 */
#pragma once
#include <iterator> // `std::make_move_iterator`

#include "shared.hpp"

#include <smashtable/locked_store.hpp>
#include <smashtable/monotonic_store.hpp>
#include <smashtable/partitioned_store.hpp>
#include <smashtable/snapshot_store.hpp>

namespace ashvardanian::smashtable::py {

#pragma region Bridge

/**
 *  @brief One store type's whole @c store_ops_t table, generated member by member.
 *
 *  @tparam store_type_ A fully wrapped store - a core, inside an isolation level, inside a sharing
 *    strategy - whose transactions the binding drives.
 */
template <typename store_type_>
struct store_bridge {

    using store_t = store_type_;
    using transaction_t = typename store_t::transaction_t;
    using value_t = typename store_t::value_t;

    /** @brief Whether elements carry a mapped value, which decides map-versus-set at every shared site. */
    static constexpr bool associative_k = is_mapping<value_t>;
    /** @brief Whether a partition hash routes the keys, which is what distinguishes the two sharings. */
    static constexpr bool partitioned_k = requires { typename store_t::hash_t; };

    /** @brief Whether the core can hand back every element, which the collector's traversal needs. */
    static constexpr bool enumerable_k = requires(store_t const &store) { store.for_each(no_op_t {}); };

    /** @brief Whether the core orders its keys, which is what the ordered slots rest on. */
    static constexpr bool ordered_k =
        requires(store_t &store, key_variant_t const &key) { store.lower_bound(key, no_op_t {}, no_op_t {}); };

    static store_t &store_of(void *store) noexcept { return *static_cast<store_t *>(store); }
    static transaction_t &transaction_of(void *transaction) noexcept {
        return *static_cast<transaction_t *>(transaction);
    }

    /** @brief Builds the element a write stores, which is a pair for a map and the key alone for a set. */
    static value_t element_of(key_variant_t &&key, value_variant_t *value) noexcept {
        if constexpr (associative_k) {
            assert(value && "a map write needs a value");
            return value_t {std::move(key), std::move(*value)};
        }
        else {
            assert(!value && "a set write carries no value");
            return value_t {std::move(key)};
        }
    }

#pragma region Lifetime

    /**
     *  @brief Builds @p object_type_ in storage CPython owns, so the binding's allocations are accounted
     *    for by the same allocator every Python object uses and appear in @c tracemalloc.
     *
     *  @c PyObject_Malloc guarantees alignment for anything up to @c max_align_t, which the assertion below
     *    pins - an over-aligned store would otherwise be constructed on a boundary it did not ask for.
     */
    template <typename object_type_, typename... arguments_type_>
    [[nodiscard]] static object_type_ *own_in_python_storage(arguments_type_ &&...arguments) noexcept {
        static_assert(alignof(object_type_) <= alignof(std::max_align_t),
                      "an over-aligned type needs storage `PyObject_Malloc` does not promise");
        void *raw = PyObject_Malloc(sizeof(object_type_));
        if (!raw) return nullptr;
        return std::construct_at(static_cast<object_type_ *>(raw), std::forward<arguments_type_>(arguments)...);
    }

    /** @brief Destroys and returns what @c own_in_python_storage handed out, tolerating a null. */
    template <typename object_type_>
    static void release_python_storage(void *owned) noexcept {
        if (!owned) return;
        std::destroy_at(static_cast<object_type_ *>(owned));
        PyObject_Free(owned);
    }

    /**
     *  @brief Builds the core, with whatever its sharing strategy is addressed by.
     *
     *  The one place that strategy shows through: a partitioned store routes keys by hash and needs
     *  one, a locked store forwards what its core takes, and neither default-constructs because
     *  @c key_less_t has no default constructor. An ordered core takes the comparator chosen for this
     *  layout, an unordered one the equality, and builds its own hasher - which is why that hasher is
     *  stateless rather than the function-pointer form.
     */
    static auto built_core(key_ops_t const *ops) noexcept {
        if constexpr (ordered_k) {
            if constexpr (partitioned_k) return store_t::make(key_less_t {ops->less}, key_hash_t {ops->hash});
            else return store_t::make(key_less_t {ops->less}, std::allocator<value_t> {});
        }
        else {
            if constexpr (partitioned_k) return store_t::make(key_variant_equal_t {}, key_hash_t {ops->hash});
            else return store_t::make(key_variant_equal_t {}, std::allocator<value_t> {});
        }
    }

    /**
     *  @brief Puts a built core on the heap, handing back the pointer @c destroy owns.
     *
     *  The probe in @c built_core asks for the member type rather than trying the call, since a store
     *  forwarding its arguments variadically accepts every signature at the declaration and refuses
     *  only inside the body, where a @c requires expression cannot see.
     */
    static expected<void *> make(key_ops_t const *ops) noexcept {
        auto built = built_core(ops);
        if (!built) return expected<void *> {built.status()};

        // The store outlives the frame that built it, so it moves onto the heap rather than into the
        // container object, which no longer has a slot shaped like any one instantiation.
        store_t *owned = own_in_python_storage<store_t>(std::move(*built));
        if (!owned) return expected<void *> {out_of_memory_heap_k};
        return expected<void *> {owned, success_k};
    }

    static void destroy(releases_t &releases, void *store) noexcept {
        shared_lock deferral {releases};
        release_python_storage<store_t>(store);
    }

#pragma endregion Lifetime

#pragma region Point Access

    static std::size_t size(void *store) noexcept { return store_of(store).size(); }

    static status_t clear(releases_t &releases, void *store) noexcept {
        shared_lock deferral {releases};
        return store_of(store).clear();
    }

    static expected<bool> contains(void *store, key_variant_t const &key) noexcept {
        return store_of(store).contains(key);
    }

    static expected<value_variant_t> find(releases_t &releases, void *store, key_variant_t const &key) noexcept {
        shared_lock deferral {releases};
        expected<value_variant_t> answer {key_not_found_k};
        if constexpr (associative_k)
            if (status_t const read = store_of(store).find(
                    key, [&](value_t const &entry) noexcept { answer = entry.mapped.copy(); }, no_op_t {});
                failed(read))
                return expected<value_variant_t> {read};
        return answer;
    }

    static status_t upsert(releases_t &releases, void *store, key_variant_t &&key, value_variant_t *value) noexcept {
        shared_lock deferral {releases};
        return store_of(store).upsert(element_of(std::move(key), value));
    }

    static status_t upsert_entries(releases_t &releases, void *store, entry_t *entries, std::size_t count) noexcept {
        shared_lock deferral {releases};
        // Moved rather than copied, which is the shape the store's batch form takes: it forwards
        // each element into one transaction it opens, stages, and commits.
        if constexpr (associative_k)
            return store_of(store).upsert(std::make_move_iterator(entries), std::make_move_iterator(entries + count));
        else return operation_not_permitted_k;
    }

    static status_t upsert_members(releases_t &releases, void *store, key_variant_t *members,
                                   std::size_t count) noexcept {
        shared_lock deferral {releases};
        if constexpr (!associative_k)
            return store_of(store).upsert(std::make_move_iterator(members), std::make_move_iterator(members + count));
        else return operation_not_permitted_k;
    }

    static expected<value_variant_t> erase(releases_t &releases, void *store, key_variant_t const &key) noexcept {
        shared_lock deferral {releases};
        expected<value_variant_t> removed {value_variant_t {}, success_k};
        status_t const status = store_of(store).erase(
            key,
            [&](value_t const &element) noexcept {
                if constexpr (associative_k) removed = element.mapped.copy();
            },
            no_op_t {});
        if (failed(status)) return expected<value_variant_t> {status};
        return removed;
    }

    static expected<entry_t> pop_smallest(releases_t &releases, void *store) noexcept
        requires ordered_k
    {
        shared_lock deferral {releases};
        expected<entry_t> removed {entry_t {}, success_k};
        status_t const status = store_of(store).pop_smallest(
            [&](value_t const &element) noexcept {
                auto key = mapping_key_or_itself<value_t>(element).copy();
                if (!key) return void(removed = expected<entry_t> {key.status()});
                entry_t taken;
                taken.key = std::move(*key);
                if constexpr (associative_k) {
                    auto held = element.mapped.copy();
                    if (!held) return void(removed = expected<entry_t> {held.status()});
                    taken.mapped = std::move(*held);
                }
                removed = std::move(taken);
            },
            no_op_t {});
        if (failed(status)) return expected<entry_t> {status};
        return removed;
    }

    /** @brief Runs one algebra, dispatching the compile-time parameter from the runtime request. */
    static status_t set_algebra(releases_t &releases, void *first, void *second, void *result,
                                algebra_t algebra) noexcept
        requires(!associative_k)
    {
        shared_lock deferral {releases};
        status_t absorbed = success_k;
        auto keep = [&](value_t const &member) noexcept {
            if (failed(absorbed)) return;
            auto copied = copy_safely(member);
            if (!copied) return void(absorbed = copied.status());
            absorbed = store_of(result).upsert(std::move(*copied));
        };

        status_t walked = success_k;
        switch (algebra) {
        case algebra_t::union_k:
            walked = walk_algebra<algebra_t::union_k>(store_of(first), store_of(second), keep);
            break;
        case algebra_t::intersection_k:
            walked = walk_algebra<algebra_t::intersection_k>(store_of(first), store_of(second), keep);
            break;
        case algebra_t::difference_k:
            walked = walk_algebra<algebra_t::difference_k>(store_of(first), store_of(second), keep);
            break;
        case algebra_t::symmetric_difference_k:
            walked = walk_algebra<algebra_t::symmetric_difference_k>(store_of(first), store_of(second), keep);
            break;
        }
        return first_failure(walked, absorbed);
    }

    static expected<bool> is_subset(void *first, void *second) noexcept
        requires(!associative_k)
    {
        return smashtable::is_subset(store_of(first), store_of(second));
    }

    static expected<bool> is_disjoint(void *first, void *second) noexcept
        requires(!associative_k)
    {
        return smashtable::is_disjoint(store_of(first), store_of(second));
    }

    static expected<value_variant_t> insert_if_missing(releases_t &releases, void *store, key_variant_t const &key,
                                                       value_variant_t &&value) noexcept {
        shared_lock deferral {releases};
        if constexpr (!associative_k) return expected<value_variant_t> {operation_not_permitted_k};
        else {
            auto copied = key.copy();
            if (!copied) return expected<value_variant_t> {copied.status()};
            expected<value_variant_t> winner {value_variant_t {}, success_k};
            // One store call rather than an insert followed by a read: between two calls another
            // thread can erase the key, leaving the read with nothing and the caller with a value
            // nobody stored. Whichever branch the store takes reports the winner from inside it.
            status_t const status = store_of(store).insert_if_missing(
                value_t {std::move(*copied), std::move(value)},
                [&](value_t const &inserted) noexcept { winner = inserted.mapped.copy(); },
                [&](value_t const &existing) noexcept { winner = existing.mapped.copy(); });
            if (failed(status)) return expected<value_variant_t> {status};
            return winner;
        }
    }

#pragma endregion Point Access

#pragma region Ordered Surface

    /**
     *  @brief Erases the half-open window, with either end open.
     *
     *  Four named entry points rather than one call with a bound standing for "no bound": a slice
     *  may leave either end out, and there is no greatest key for the text layouts to close the
     *  upper end with. Which one applies is decided here, where the names live, so the container
     *  never has to synthesize a bound it cannot spell.
     */
    static status_t erase_range(releases_t &releases, void *store, key_variant_t const *lower,
                                key_variant_t const *upper) noexcept
        requires ordered_k
    {
        shared_lock deferral {releases};
        auto &self = store_of(store);
        if (lower && upper) return self.erase_range(*lower, *upper);
        if (lower) return self.erase_from(*lower);
        if (upper) return self.erase_up_to(*upper);
        return self.clear();
    }

#pragma endregion Ordered Surface

#pragma region Transactions

    using cursor_t = typename store_t::ordered_cursor_t;

    static cursor_t &cursor_of(void *cursor) noexcept { return *static_cast<cursor_t *>(cursor); }

    /** @brief Opens the walk the pair of bounds names; a null bound is unbounded on that side. */
    static cursor_t opened_cursor(void *store, key_variant_t const *from, key_variant_t const *upper) noexcept
        requires ordered_k
    {
        if (from && upper) return store_of(store).cursor_range(key_variant_t(*from), key_variant_t(*upper));
        if (from) return store_of(store).cursor_from(key_variant_t(*from));
        if (upper) return store_of(store).cursor_up_to(key_variant_t(*upper));
        return store_of(store).cursor();
    }

    static expected<void *> cursor_make(void *store, key_variant_t const *from, key_variant_t const *upper) noexcept
        requires ordered_k
    {
        cursor_t opened = opened_cursor(store, from, upper);
        cursor_t *owned = own_in_python_storage<cursor_t>(std::move(opened));
        if (!owned) return expected<void *> {out_of_memory_heap_k};
        return expected<void *> {owned, success_k};
    }

    static void cursor_destroy(releases_t &releases, void *cursor) noexcept
        requires ordered_k
    {
        shared_lock deferral {releases};
        release_python_storage<cursor_t>(cursor);
    }

    static bool cursor_next(void *cursor, key_variant_t &key, value_variant_t *value) noexcept
        requires ordered_k
    {
        bool handed = false;
        if (cursor_of(cursor).exhausted()) return false;
        cursor_of(cursor).next([&](value_t const &element) noexcept {
            key = mapping_key_or_itself<value_t>(element);
            if constexpr (associative_k)
                if (value) *value = element.mapped;
            handed = true;
        });
        return handed;
    }

    static expected<void *> transaction_make(void *store) noexcept {
        auto opened = store_of(store).transaction();
        if (!opened) return expected<void *> {opened.status()};
        transaction_t *owned = own_in_python_storage<transaction_t>(std::move(*opened));
        if (!owned) return expected<void *> {out_of_memory_heap_k};
        return expected<void *> {owned, success_k};
    }

    static void transaction_destroy(releases_t &releases, void *transaction) noexcept {
        shared_lock deferral {releases};
        release_python_storage<transaction_t>(transaction);
    }

    static expected<bool> transaction_contains(void *transaction, key_variant_t const &key) noexcept {
        return transaction_of(transaction).contains(key);
    }

    static expected<value_variant_t> transaction_find(releases_t &releases, void *transaction,
                                                      key_variant_t const &key) noexcept {
        shared_lock deferral {releases};
        expected<value_variant_t> answer {key_not_found_k};
        if constexpr (associative_k)
            if (status_t const read =
                    transaction_of(transaction)
                        .find(
                            key, [&](value_t const &entry) noexcept { answer = entry.mapped.copy(); }, no_op_t {});
                failed(read))
                return expected<value_variant_t> {read};
        return answer;
    }

    static status_t transaction_upsert(releases_t &releases, void *transaction, key_variant_t &&key,
                                       value_variant_t *value) noexcept {
        shared_lock deferral {releases};
        return transaction_of(transaction).upsert(element_of(std::move(key), value));
    }

    static status_t transaction_erase(releases_t &releases, void *transaction, key_variant_t const &key) noexcept {
        shared_lock deferral {releases};
        return transaction_of(transaction).erase(key);
    }

    static status_t transaction_watch(void *transaction, key_variant_t const &key) noexcept {
        return transaction_of(transaction).watch(key);
    }

    /** @brief Copies one element into @p into, reporting what the copy could not do. */
    static status_t collect(value_t const &element, basic_vector<entry_t> &into) noexcept {
        auto key = mapping_key_or_itself<value_t>(element).copy();
        if (!key) return key.status();
        entry_t taken;
        taken.key = std::move(*key);
        if constexpr (associative_k) {
            auto held = element.mapped.copy();
            if (!held) return held.status();
            taken.mapped = std::move(*held);
        }
        return into.push_back(std::move(taken));
    }

    /**
     *  @brief Collects the half-open window, which is the read a phantom is detected against.
     *
     *  Which walk answers depends on which ends are named, because each records a different read and
     *  the level is validated against exactly what was recorded: a closed window records that window,
     *  an open end records one running to that end of the keyspace, and a window with no ends at all is
     *  the whole keyspace and says so. Each is one store call, so the window a commit is validated
     *  against is the one the caller asked for rather than one this layer chose.
     */
    static status_t transaction_scan(releases_t &releases, void *transaction, key_variant_t const *lower,
                                     key_variant_t const *upper, std::size_t limit,
                                     basic_vector<entry_t> &collected) noexcept
        requires ordered_k
    {
        shared_lock deferral {releases};
        auto &self = transaction_of(transaction);
        status_t collecting = success_k;
        auto step = [&](value_t const &element) noexcept {
            if (failed(collecting) || collected.size() == limit) return;
            collecting = collect(element, collected);
        };

        if (lower && upper) {
            status_t const walked = self.range(*lower, *upper, step);
            return first_failure(walked, collecting);
        }
        if (lower) {
            status_t const walked = self.range_from(*lower, step);
            return first_failure(walked, collecting);
        }
        if (upper) {
            status_t const walked = self.range_up_to(*upper, step);
            return first_failure(walked, collecting);
        }

        status_t const walked = self.for_each(step);
        return first_failure(walked, collecting);
    }

    /** @brief Stages a tombstone for every member of that same window. */
    static status_t transaction_erase_range(releases_t &releases, void *transaction, key_variant_t const *lower,
                                            key_variant_t const *upper) noexcept
        requires ordered_k
    {
        shared_lock deferral {releases};
        auto &self = transaction_of(transaction);
        if (lower && upper) return self.erase_range(*lower, *upper, no_op_t {});
        if (lower) return self.erase_from(*lower, no_op_t {});
        if (upper) return self.erase_up_to(*upper, no_op_t {});

        // A transaction has no `clear`: erasing through one means staging a tombstone per member, and
        // the members are collected before any of them is staged, because a tombstone moves what the
        // merged walk answers underneath the walk that produced it.
        basic_vector<entry_t> doomed;
        status_t collecting = success_k;
        if (status_t const walked = self.for_each([&](value_t const &element) noexcept {
                if (succeeded(collecting)) collecting = collect(element, doomed);
            });
            failed(walked))
            return walked;
        if (failed(collecting)) return collecting;
        for (std::size_t index = 0; index != doomed.size(); ++index)
            if (status_t const staged = self.erase(doomed[index].key); failed(staged)) return staged;
        return success_k;
    }

    static status_t transaction_stage(releases_t &releases, void *transaction) noexcept {
        shared_lock deferral {releases};
        return transaction_of(transaction).stage();
    }
    static status_t transaction_commit(releases_t &releases, void *transaction) noexcept {
        shared_lock deferral {releases};
        return transaction_of(transaction).commit();
    }
    static status_t transaction_validate(releases_t &releases, void *transaction) noexcept {
        shared_lock deferral {releases};
        return transaction_of(transaction).validate_for_commit();
    }
    static void transaction_publish(releases_t &releases, void *transaction) noexcept {
        shared_lock deferral {releases};
        transaction_of(transaction).publish_under();
    }
    static status_t transaction_rollback(releases_t &releases, void *transaction) noexcept {
        shared_lock deferral {releases};
        return transaction_of(transaction).rollback();
    }
    static status_t transaction_reset(releases_t &releases, void *transaction) noexcept {
        shared_lock deferral {releases};
        return transaction_of(transaction).reset();
    }

    /**
     *  @brief Reports every stored object to the collector, so a cycle through a container is seen.
     *
     *  Runs with the GIL held and the store's lock taken for the walk. That is sound only because
     *  nothing inside allocates a Python object: the visitor merely records, and a reference is
     *  never dropped here - dropping is what @c tp_clear does, through the deferring path.
     */
    static int visit_values(void *store, visitproc visit, void *arg) noexcept
        requires(enumerable_k && associative_k)
    {
        int outcome = 0;
        // The walk's own status is dropped here, and this is the one slot where that is right:
        // `tp_traverse` answers the collector with an `int` that means "keep going" or "stop", and
        // has no room for a reason. A walk that could not complete reports nothing rather than
        // fewer objects, so the collector simply sees what was reachable at that moment.
        [[maybe_unused]] status_t const walked = store_of(store).for_each([&](value_t const &element) noexcept {
            if (outcome != 0) return;
            auto const *held = std::get_if<object_t>(&element.mapped.value);
            if (held && held->held) outcome = visit(held->held, arg);
        });
        return outcome;
    }

#pragma endregion Transactions

    /** @brief The whole table, with every slot a core cannot supply left null. */
    static constexpr store_ops_t table() noexcept {
        store_ops_t built {};
        built.isolation = store_t::isolation_k;
        built.isolation_name = isolation_name_of(store_t::isolation_k);
        built.sharing_name = partitioned_k ? "partitioned" : "locked";
        built.is_associative = associative_k;
        built.is_ordered = ordered_k;

        built.make = &make;
        built.destroy = &destroy;
        built.size = &size;
        built.clear = &clear;
        built.contains = &contains;
        built.find = associative_k ? &find : nullptr;
        built.upsert = &upsert;
        built.upsert_entries = associative_k ? &upsert_entries : nullptr;
        built.upsert_members = associative_k ? nullptr : &upsert_members;
        built.erase = &erase;
        built.insert_if_missing = associative_k ? &insert_if_missing : nullptr;

        if constexpr (!associative_k) {
            built.set_algebra = &set_algebra;
            built.is_subset = &is_subset;
            built.is_disjoint = &is_disjoint;
        }

        if constexpr (ordered_k) {
            built.erase_range = &erase_range;
            built.pop_smallest = &pop_smallest;
            built.cursor_make = &cursor_make;
            built.cursor_destroy = &cursor_destroy;
            built.cursor_next = &cursor_next;
        }

        if constexpr (enumerable_k && associative_k) built.visit_values = &visit_values;

        built.transaction_make = &transaction_make;
        built.transaction_destroy = &transaction_destroy;
        built.transaction_contains = &transaction_contains;
        built.transaction_find = associative_k ? &transaction_find : nullptr;
        built.transaction_upsert = &transaction_upsert;
        built.transaction_erase = &transaction_erase;
        built.transaction_watch = &transaction_watch;
        if constexpr (ordered_k) {
            built.transaction_scan = &transaction_scan;
            built.transaction_erase_range = &transaction_erase_range;
        }
        built.transaction_stage = &transaction_stage;
        built.transaction_commit = &transaction_commit;
        // Null where the engine decides and writes in one call, which is what a partitioned store does;
        // the group reads the null and falls back to committing each participant in turn.
        if constexpr (splits_its_commit<transaction_t>) {
            built.transaction_validate = &transaction_validate;
            built.transaction_publish = &transaction_publish;
        }
        built.transaction_rollback = &transaction_rollback;
        built.transaction_reset = &transaction_reset;
        return built;
    }
};

#pragma endregion Bridge

} // namespace ashvardanian::smashtable::py
