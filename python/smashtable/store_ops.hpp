/**
 *  @brief Turns one store instantiation into a @c store_ops_t table of function pointers.
 *  @author Ash Vardanian
 *  @file python/smashtable/store_ops.hpp
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
    static constexpr bool ordered_k = requires(store_t &store, key_variant_t const &key) {
        store.lower_bound(key, no_op_t {}, no_op_t {});
    };

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

    static expected<void *> make(key_ops_t const *ops) noexcept {
        // Which arguments a store is built from is the one place the sharing strategy shows through:
        // a partitioned store routes keys by hash and so needs one, while a locked store forwards
        // whatever its inner core takes. Neither can be default-constructed, because `key_less_t`
        // deliberately has no default constructor.
        //
        // The probe asks for the member type rather than trying the call, because a store that
        // forwards its arguments variadically accepts every signature at the declaration and only
        // refuses inside the body, where a `requires` expression cannot see.
        // What a core is addressed by: an ordered one takes the comparator chosen for this layout,
        // an unordered one takes the equality and builds its own hasher, which is why that hasher
        // has to be stateless rather than the function-pointer form.
        auto built = [&]() noexcept {
            if constexpr (ordered_k) {
                if constexpr (partitioned_k) return store_t::make(key_less_t {ops->less}, key_hash_t {ops->hash});
                else return store_t::make(key_less_t {ops->less}, std::allocator<value_t> {});
            }
            else {
                if constexpr (partitioned_k) return store_t::make(key_variant_equal_t {}, key_hash_t {ops->hash});
                else return store_t::make(key_variant_equal_t {}, std::allocator<value_t> {});
            }
        }();
        if (!built) return expected<void *> {built.status()};

        // The store outlives the frame that built it, so it moves onto the heap rather than into the
        // container object, which no longer has a slot shaped like any one instantiation.
        store_t *owned = new (std::nothrow) store_t(std::move(*built));
        if (!owned) return expected<void *> {out_of_memory_heap_k};
        return expected<void *> {static_cast<void *>(owned), success_k};
    }

    static void destroy(void *store) noexcept {
        deferring_store_call_t deferral;
        delete static_cast<store_t *>(store);
    }

#pragma endregion Lifetime

#pragma region Point Access

    static std::size_t size(void *store) noexcept { return store_of(store).size(); }

    static status_t clear(void *store) noexcept {
        deferring_store_call_t deferral;
        return store_of(store).clear();
    }

    static bool contains(void *store, key_variant_t const &key) noexcept { return store_of(store).contains(key); }

    static bool find(void *store, key_variant_t const &key, value_variant_t &value) noexcept {
        deferring_store_call_t deferral;
        bool found = false;
        if constexpr (associative_k)
            store_of(store).find(
                key, [&](value_t const &entry) noexcept { value = entry.mapped, found = true; }, no_op_t {});
        return found;
    }

    static status_t upsert(void *store, key_variant_t &&key, value_variant_t *value) noexcept {
        deferring_store_call_t deferral;
        return store_of(store).upsert(element_of(std::move(key), value));
    }

    static status_t erase(void *store, key_variant_t const &key) noexcept {
        deferring_store_call_t deferral;
        return store_of(store).erase(key);
    }

    static status_t insert_if_missing(void *store, key_variant_t const &key, value_variant_t &&value,
                                      value_variant_t &winner) noexcept {
        deferring_store_call_t deferral;
        if constexpr (!associative_k) return operation_not_permitted_k;
        else {
            auto copied = key.copy();
            if (!copied) return copied.status();
            // One store call rather than an insert followed by a read: between two calls another
            // thread can erase the key, leaving the read with nothing and the caller with a value
            // nobody stored. Whichever branch the store takes reports the winner from inside it.
            return store_of(store).insert_if_missing(
                value_t {std::move(*copied), std::move(value)},
                [&](value_t const &inserted) noexcept { winner = inserted.mapped; },
                [&](value_t const &existing) noexcept { winner = existing.mapped; });
        }
    }

#pragma endregion Point Access

#pragma region Ordered Surface

    /** @brief Copies out the element a bound landed on, which is the key alone for a set. */
    static void take(value_t const &element, key_variant_t &key, value_variant_t *value) noexcept {
        key = mapping_key_or_itself<value_t>(element);
        if constexpr (associative_k)
            if (value) *value = element.mapped;
    }

    static bool lower_bound(void *store, key_variant_t const &from, key_variant_t &key,
                            value_variant_t *value) noexcept
        requires ordered_k
    {
        deferring_store_call_t deferral;
        bool found = false;
        store_of(store).lower_bound(
            from, [&](value_t const &element) noexcept { take(element, key, value), found = true; }, no_op_t {});
        return found;
    }

    static bool upper_bound(void *store, key_variant_t const &from, key_variant_t &key,
                            value_variant_t *value) noexcept
        requires ordered_k
    {
        deferring_store_call_t deferral;
        bool found = false;
        store_of(store).upper_bound(
            from, [&](value_t const &element) noexcept { take(element, key, value), found = true; }, no_op_t {});
        return found;
    }

    /**
     *  @brief Erases the half-open window, with either end open.
     *
     *  Four named entry points rather than one call with a bound standing for "no bound": a slice
     *  may leave either end out, and there is no greatest key for the text layouts to close the
     *  upper end with. Which one applies is decided here, where the names live, so the container
     *  never has to synthesize a bound it cannot spell.
     */
    static status_t erase_range(void *store, key_variant_t const *lower, key_variant_t const *upper) noexcept
        requires ordered_k
    {
        deferring_store_call_t deferral;
        auto &self = store_of(store);
        if (lower && upper) return self.erase_range(*lower, *upper);
        if (lower) return self.erase_from(*lower);
        if (upper) return self.erase_up_to(*upper);
        return self.clear();
    }

#pragma endregion Ordered Surface

#pragma region Transactions

    static expected<void *> transaction_make(void *store) noexcept {
        auto opened = store_of(store).transaction();
        if (!opened) return expected<void *> {opened.status()};
        transaction_t *owned = new (std::nothrow) transaction_t(std::move(*opened));
        if (!owned) return expected<void *> {out_of_memory_heap_k};
        return expected<void *> {static_cast<void *>(owned), success_k};
    }

    static void transaction_destroy(void *transaction) noexcept {
        deferring_store_call_t deferral;
        delete static_cast<transaction_t *>(transaction);
    }

    static bool transaction_contains(void *transaction, key_variant_t const &key) noexcept {
        return transaction_of(transaction).contains(key);
    }

    static bool transaction_find(void *transaction, key_variant_t const &key, value_variant_t &value) noexcept {
        deferring_store_call_t deferral;
        bool found = false;
        if constexpr (associative_k)
            transaction_of(transaction)
                .find(
                    key, [&](value_t const &entry) noexcept { value = entry.mapped, found = true; }, no_op_t {});
        return found;
    }

    static status_t transaction_upsert(void *transaction, key_variant_t &&key, value_variant_t *value) noexcept {
        deferring_store_call_t deferral;
        return transaction_of(transaction).upsert(element_of(std::move(key), value));
    }

    static status_t transaction_erase(void *transaction, key_variant_t const &key) noexcept {
        deferring_store_call_t deferral;
        return transaction_of(transaction).erase(key);
    }

    static status_t transaction_watch(void *transaction, key_variant_t const &key) noexcept {
        return transaction_of(transaction).watch(key);
    }

    static status_t transaction_stage(void *transaction) noexcept {
        deferring_store_call_t deferral;
        return transaction_of(transaction).stage();
    }
    static status_t transaction_commit(void *transaction) noexcept {
        deferring_store_call_t deferral;
        return transaction_of(transaction).commit();
    }
    static status_t transaction_rollback(void *transaction) noexcept {
        deferring_store_call_t deferral;
        return transaction_of(transaction).rollback();
    }
    static status_t transaction_reset(void *transaction) noexcept {
        deferring_store_call_t deferral;
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
        store_of(store).for_each([&](value_t const &element) noexcept {
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
        built.erase = &erase;
        built.insert_if_missing = associative_k ? &insert_if_missing : nullptr;

        if constexpr (ordered_k) {
            built.lower_bound = &lower_bound;
            built.upper_bound = &upper_bound;
            built.erase_range = &erase_range;
        }

        if constexpr (enumerable_k && associative_k) built.visit_values = &visit_values;

        built.transaction_make = &transaction_make;
        built.transaction_destroy = &transaction_destroy;
        built.transaction_contains = &transaction_contains;
        built.transaction_find = associative_k ? &transaction_find : nullptr;
        built.transaction_upsert = &transaction_upsert;
        built.transaction_erase = &transaction_erase;
        built.transaction_watch = &transaction_watch;
        built.transaction_stage = &transaction_stage;
        built.transaction_commit = &transaction_commit;
        built.transaction_rollback = &transaction_rollback;
        built.transaction_reset = &transaction_reset;
        return built;
    }
};

#pragma endregion Bridge

} // namespace ashvardanian::smashtable::py
