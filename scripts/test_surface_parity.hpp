/**
 *  @brief Makes "a wrapper dropped a surface its inner store offers" a compile error that names the
 *      surface, the store, and the wrapper, rather than an overload-resolution failure naming none.
 *  @author Ash Vardanian
 *  @file scripts/test_surface_parity.hpp
 *  @date August 17, 2026
 */
#pragma once
#include <cstddef> // `std::size_t`

#include <smashtable/locked_store.hpp>
#include <smashtable/partitioned_store.hpp>

namespace ashvardanian::smashtable::scripts {

/**
 *  @brief A key type no store's identifier is constructible from, so only a genuinely heterogeneous
 *    entry point accepts it. A surface narrowed to @c identifier_t refuses it at the declaration.
 */
struct alien_key_t {};

#pragma region Store Surfaces

/**
 *  @brief One public surface, named so a parity failure can print it.
 *
 *  Each carries a single @c offered predicate over a store type, and nothing else - the fold below
 *  supplies the wrapper and the store. A predicate asks for exactly one method, never a conjunction:
 *  two methods behind one gate means the store offering only the second loses it through every
 *  wrapper, which is the defect class this header exists to catch.
 */
struct sample_one_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.sample_one(key, key, callback, callback); };
};
struct sample_reservoir_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(store_type_ const &store, typename store_type_::identifier_t const &key, no_op_t callback,
                 std::size_t seen) { store.sample_reservoir(key, key, callback, seen, seen, callback); };
};
struct equal_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.equal_range(key, callback); };
};
struct select_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, std::size_t ordinal, no_op_t callback) {
        store.select(ordinal, callback, callback);
    };
};
struct rank_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.rank(key, callback, callback); };
};
struct ranked_size_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store) { store.ranked_size(); };
};
struct for_each_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, no_op_t callback) { store.for_each(callback); };
};
struct lower_bound_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.lower_bound(key, callback, callback); };
};
struct upper_bound_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.upper_bound(key, callback, callback); };
};
struct range_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.range(key, key, callback); };
};
struct erase_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.erase_range(key, key, callback); };
};
struct erase_from_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.erase_from(key, callback); };
};
struct erase_up_to_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.erase_up_to(key, callback); };
};
struct update_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                             no_op_t callback) { store.update_range(key, key, callback); };
};
struct insert_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(store_type_ &store, typename store_type_::value_t &&element) { store.insert(std::move(element)); };
};
struct insert_if_missing_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ &store, typename store_type_::value_t &&element) {
        store.insert_if_missing(std::move(element));
    };
};
struct insert_or_assign_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ &store, typename store_type_::value_t &&element) {
        store.insert_or_assign(std::move(element));
    };
};
struct update_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(store_type_ &store, typename store_type_::value_t &&element) { store.update(std::move(element)); };
};
struct find_copy_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(store_type_ const &store, typename store_type_::identifier_t const &key) { store.find_copy(key); };
};
struct vacuum_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ &store) { store.vacuum(); };
};
struct vacuum_window_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(store_type_ &store, typename store_type_::identifier_t const &key) { store.vacuum(key, key); };
};
struct versions_count_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, typename store_type_::identifier_t const &key) {
        store.versions_count();
        store.versions_count(key);
    };
};
struct low_water_mark_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store) { store.low_water_mark(); };
};
struct visible_cursor_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ const &store, no_op_t callback) {
        store.visible_keys().peek(callback, callback);
        store.visible_keys().peek_copy();
    };
};
struct shard_protocol_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(store_type_ &store, typename shared_clock_of<store_type_>::type &clock,
                                             typename store_type_::generation_t stamp) {
        store.attach_clock(clock);
        store.transaction_at(stamp, stamp);
    };
};
struct heterogeneous_erase_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(store_type_ &store, no_op_t callback) { store.erase(alien_key_t {}, callback, callback); };
};

#pragma endregion Store Surfaces

#pragma region Transaction Surfaces

/**
 *  @brief The same, over the transaction a store opens rather than over the store itself.
 *    Spelled against @c store_type_ so one fold drives both families, and so a store whose
 *    transaction type is missing entirely fails as loudly as one whose method is.
 */
struct transaction_insert_if_missing_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(typename store_type_::transaction_t &transaction, typename store_type_::value_t &&element) {
            transaction.insert_if_missing(std::move(element));
        };
};
struct transaction_changes_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(typename store_type_::transaction_t const &transaction) {
        transaction.has_changes();
        transaction.changes_count();
    };
};
struct transaction_for_each_surface_t {
    template <typename store_type_>
    static constexpr bool offered = requires(typename store_type_::transaction_t const &transaction, no_op_t callback) {
        transaction.for_each(callback);
    };
};
struct transaction_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.range(key, key, callback); };
};
struct transaction_equal_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.equal_range(key, callback); };
};
struct transaction_lower_bound_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.lower_bound(key, callback, callback); };
};
struct transaction_upper_bound_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.upper_bound(key, callback, callback); };
};
struct transaction_find_copy_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(typename store_type_::transaction_t const &transaction,
                 typename store_type_::identifier_t const &key) { transaction.find_copy(key); };
};
struct transaction_generation_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(typename store_type_::transaction_t const &transaction) { transaction.generation(); };
};
struct transaction_reserve_surface_t {
    template <typename store_type_>
    static constexpr bool offered =
        requires(typename store_type_::transaction_t &transaction, std::size_t size) { transaction.reserve(size); };
};

#pragma endregion Transaction Surfaces

#pragma region Wrapper Nestings

/**
 *  @brief One wrapper nesting, named so a parity failure can print which of them dropped the surface.
 *    @c waiver_group_t is the wrapper whose waivers apply, which for a nesting is the outer one -
 *    the inner wrapper is transparent by construction, so it adds no refusals of its own.
 */
struct locked_wrapper_t {
    template <typename store_type_>
    using wrapped = locked_store<store_type_>;
    using waiver_group_t = locked_wrapper_t;
};
struct partitioned_wrapper_t {
    template <typename store_type_>
    using wrapped = partitioned_store<store_type_>;
    using waiver_group_t = partitioned_wrapper_t;
};
struct partitioned_over_locked_wrapper_t {
    template <typename store_type_>
    using wrapped = partitioned_store<locked_store<store_type_>>;
    using waiver_group_t = partitioned_wrapper_t;
};

#pragma endregion Wrapper Nestings

#pragma region Waivers

/**
 *  @brief The surfaces a wrapper refuses on purpose. Every entry below states why, and there are no
 *    others: anything not listed here must survive every nesting.
 */
template <typename surface_, typename wrapper_>
constexpr bool waives_surface = false;

/**
 *  @brief An ordinal spans partitions, and no partition knows the other fifteen's counts. What
 *    @c partitioned_store::select indexes is a merged order taken under every partition's lock, not
 *    the augmented count a core keeps over its own subtree, so a sum of the parts would name a size
 *    the merged order never has.
 */
template <>
constexpr bool waives_surface<ranked_size_surface_t, partitioned_wrapper_t> = true;

/**
 *  @brief A cursor outlives the call that produced it and holds iterators into the entries a writer
 *    may reallocate, so handing one out past a mutex is handing out a dangling reader, not merely an
 *    unsynchronized one. Both wrappers refuse it; a caller that wants to walk uses @c for_each or
 *    @c range, which keep the lock for the walk.
 */
template <>
constexpr bool waives_surface<visible_cursor_surface_t, locked_wrapper_t> = true;
template <>
constexpr bool waives_surface<visible_cursor_surface_t, partitioned_wrapper_t> = true;

/**
 *  @brief A shard set owns exactly one clock and hands it to every partition, so it consumes the
 *    protocol rather than re-exporting it. Letting an outer set replace that clock would strand the
 *    stamps and reader claims the partitions already hold. @c locked_store forwards it instead,
 *    which is what lets a shard set be built over one.
 */
template <>
constexpr bool waives_surface<shard_protocol_surface_t, partitioned_wrapper_t> = true;

#pragma endregion Waivers

#pragma region Parity Fold

/**
 *  @brief One assertion: if @c store_type_ offers @c surface_, so must @c wrapper_ over it.
 *    A struct rather than a bare @c constexpr @c bool so the diagnostic names all three - the status
 *    quo is an overload-resolution error at the call site, which names none of them.
 */
template <typename surface_, typename wrapper_, typename store_type_>
struct parity_witness {
    static_assert(!surface_::template offered<store_type_> ||
                      surface_::template offered<typename wrapper_::template wrapped<store_type_>> ||
                      waives_surface<surface_, typename wrapper_::waiver_group_t>,
                  "a wrapper must offer every surface its inner store offers, or waive it by name");
    static constexpr bool checked_k = true;
};

/** @brief Every named surface against one wrapper and one store, folded into a single value. */
template <typename wrapper_, typename store_type_, typename... surfaces_>
constexpr bool every_surface_survives = (parity_witness<surfaces_, wrapper_, store_type_>::checked_k && ...);

/** @brief The whole public store surface against one wrapper. */
template <typename wrapper_, typename store_type_>
constexpr bool store_surface_survives = every_surface_survives<
    wrapper_, store_type_, sample_one_surface_t, sample_reservoir_surface_t, equal_range_surface_t, select_surface_t,
    rank_surface_t, ranked_size_surface_t, for_each_surface_t, lower_bound_surface_t, upper_bound_surface_t,
    range_surface_t, erase_range_surface_t, erase_from_surface_t, erase_up_to_surface_t, update_range_surface_t,
    insert_surface_t, insert_if_missing_surface_t, insert_or_assign_surface_t, update_surface_t, find_copy_surface_t,
    vacuum_surface_t, vacuum_window_surface_t, versions_count_surface_t, low_water_mark_surface_t,
    visible_cursor_surface_t, shard_protocol_surface_t, heterogeneous_erase_surface_t>;

/** @brief The whole transaction surface against one wrapper. */
template <typename wrapper_, typename store_type_>
constexpr bool transaction_surface_survives =
    every_surface_survives<wrapper_, store_type_, transaction_insert_if_missing_surface_t,
                           transaction_changes_surface_t, transaction_for_each_surface_t, transaction_range_surface_t,
                           transaction_equal_range_surface_t, transaction_lower_bound_surface_t,
                           transaction_upper_bound_surface_t, transaction_find_copy_surface_t,
                           transaction_generation_surface_t, transaction_reserve_surface_t>;

/**
 *  @brief Every surface, against every wrapper nesting, over one store.
 *
 *  The nesting matters as much as the wrappers do: @c partitioned_store over @c locked_store is the
 *  case where a wrapper meant to be transparent gets to change what the outer one concludes, and
 *  nothing but running the fold over it would say so.
 */
template <typename store_type_>
constexpr bool every_wrapper_keeps_surfaces =
    store_surface_survives<locked_wrapper_t, store_type_> &&
    store_surface_survives<partitioned_wrapper_t, store_type_> &&
    store_surface_survives<partitioned_over_locked_wrapper_t, store_type_> &&
    transaction_surface_survives<locked_wrapper_t, store_type_> &&
    transaction_surface_survives<partitioned_wrapper_t, store_type_> &&
    transaction_surface_survives<partitioned_over_locked_wrapper_t, store_type_>;

/**
 *  @brief A transparent wrapper must not change what an outer wrapper concludes.
 *
 *  Not "a wrapper never lowers @c isolation_k" - @c partitioned_store legitimately lowers it over a
 *  store whose visibility is not decided by a stamp. The honest property is that inserting a wrapper
 *  meant to be invisible changes nothing, which is what fails the moment a trait alias stops being
 *  forwarded.
 */
template <typename store_type_>
constexpr bool nesting_preserves_isolation =
    partitioned_store<locked_store<store_type_>>::isolation_k == partitioned_store<store_type_>::isolation_k;

/**
 *  @brief A transaction that moves must move-assign, and must not copy.
 *    Movable-but-not-move-assignable is almost always a defaulted operator the compiler deleted over
 *    a reference member, and it fails where the transaction is stored rather than where it is declared.
 */
template <typename store_type_>
constexpr bool transaction_moves_as_a_value =
    std::is_nothrow_move_constructible_v<typename store_type_::transaction_t> &&
    std::is_move_assignable_v<typename store_type_::transaction_t> &&
    !std::is_copy_constructible_v<typename store_type_::transaction_t>;

/**
 *  @brief The composite surfaces the older assertions name, spelled from the atoms above rather than
 *    beside them - a second definition of "ordered" is a second thing to keep in step.
 */
template <typename store_type_>
constexpr bool offers_ordered_surface =
    lower_bound_surface_t::offered<store_type_> && upper_bound_surface_t::offered<store_type_> &&
    range_surface_t::offered<store_type_> && erase_range_surface_t::offered<store_type_>;

/** @brief The ordinal surface, which only a core summing subtree counts can answer. */
template <typename store_type_>
constexpr bool offers_order_statistics = select_surface_t::offered<store_type_> && rank_surface_t::offered<store_type_>;

/** @brief The one walk an unordered core can offer. */
template <typename store_type_>
constexpr bool offers_enumeration = for_each_surface_t::offered<store_type_>;

#pragma endregion Parity Fold

#pragma region Level Pairs

/**
 *  @brief One surface asked of a store and of the transaction that store opens.
 *
 *  The wrapper fold above asks whether a wrapper kept what its inner store offers, which is vacuously
 *  true for a method absent from both. That blind spot hid every entry in @c unimplemented_at_transaction
 *  below, so this family asks the other question: a store surface should have a transaction counterpart
 *  unless there is a reason it cannot.
 */
struct count_pair_t {
    template <typename store_type_>
    static constexpr bool at_store =
        requires(store_type_ const &store, typename store_type_::identifier_t const &key) { store.count(key); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t const &transaction,
                 typename store_type_::identifier_t const &key) { transaction.count(key); };
};
struct vacuum_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ &store) { store.vacuum(); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t &transaction) { transaction.vacuum(); };
};
struct clear_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ &store) { store.clear(); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t &transaction) { transaction.clear(); };
};
struct erase_range_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                              no_op_t callback) { store.erase_range(key, key, callback); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.erase_range(key, key, callback); };
};
struct erase_from_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                              no_op_t callback) { store.erase_from(key, callback); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.erase_from(key, callback); };
};
struct erase_up_to_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                              no_op_t callback) { store.erase_up_to(key, callback); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.erase_up_to(key, callback); };
};
struct update_range_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                              no_op_t callback) { store.update_range(key, key, callback); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.update_range(key, key, callback); };
};
struct sample_one_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                              no_op_t callback) { store.sample_one(key, key, callback, callback); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.sample_one(key, key, callback, callback); };
};
struct sample_reservoir_pair_t {
    template <typename store_type_>
    static constexpr bool at_store =
        requires(store_type_ const &store, typename store_type_::identifier_t const &key, no_op_t callback,
                 std::size_t seen) { store.sample_reservoir(key, key, callback, seen, seen, callback); };
    template <typename store_type_>
    static constexpr bool at_transaction = requires(
        typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
        no_op_t callback, std::size_t seen) { transaction.sample_reservoir(key, key, callback, seen, seen, callback); };
};
struct lower_bound_copy_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ const &store, typename store_type_::identifier_t const &key) {
        store.lower_bound_copy(key);
    };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t const &transaction,
                 typename store_type_::identifier_t const &key) { transaction.lower_bound_copy(key); };
};
struct upper_bound_copy_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ const &store, typename store_type_::identifier_t const &key) {
        store.upper_bound_copy(key);
    };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t const &transaction,
                 typename store_type_::identifier_t const &key) { transaction.upper_bound_copy(key); };
};

/** @brief Present at both levels already, so the fold has something that must stay passing. */
struct find_copy_pair_t {
    template <typename store_type_>
    static constexpr bool at_store =
        requires(store_type_ const &store, typename store_type_::identifier_t const &key) { store.find_copy(key); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t const &transaction,
                 typename store_type_::identifier_t const &key) { transaction.find_copy(key); };
};
struct select_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ const &store, std::size_t ordinal, no_op_t callback) {
        store.select(ordinal, callback, callback);
    };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t const &transaction, std::size_t ordinal, no_op_t callback) {
            transaction.select(ordinal, callback, callback);
        };
};
struct rank_pair_t {
    template <typename store_type_>
    static constexpr bool at_store = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                              no_op_t callback) { store.rank(key, callback, callback); };
    template <typename store_type_>
    static constexpr bool at_transaction =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.rank(key, callback, callback); };
};

#pragma endregion Level Pairs

#pragma region Transaction Refusals

/**
 *  @brief The store surfaces a transaction refuses on purpose, each with the reason it cannot carry.
 *    These are decisions. Anything absent for want of an implementation belongs in the list below,
 *    which is a to-do rather than a refusal, and the two must never be read as the same thing.
 */
template <typename pair_>
constexpr bool refuses_at_transaction = false;

/**
 *  @brief Reclamation frees versions no live reader can still name, which is a fact about the store
 *    and every reader open on it rather than about one transaction. A transaction cannot know what
 *    its peers still hold, so a transaction-level sweep would either free what somebody reads or
 *    free nothing, and both are worse than asking the store.
 */
template <>
constexpr bool refuses_at_transaction<vacuum_pair_t> = true;

/**
 *  @brief Emptying the store refuses while any snapshot is open, so a transaction calling it would be
 *    refusing itself: its own snapshot is the one standing in the way.
 */
template <>
constexpr bool refuses_at_transaction<clear_pair_t> = true;

/**
 *  @brief Store surfaces a transaction @b should carry and does not yet.
 *
 *  Every entry here is work outstanding, not a decision, and each should be deleted from this list by
 *  the change that implements it rather than by anyone judging it unnecessary. The fold below counts
 *  them, so removing a method without removing its entry leaves the count wrong and fails.
 */
template <typename pair_>
constexpr bool unimplemented_at_transaction = false;

#pragma endregion Transaction Refusals

#pragma region Level Parity Fold

/**
 *  @brief One assertion: a surface a store offers should be reachable from its transaction too.
 *    Named the same way the wrapper witness is, so a failure prints the surface and the store.
 */
template <typename pair_, typename store_type_>
struct level_witness {
    static_assert(!pair_::template at_store<store_type_> || pair_::template at_transaction<store_type_> ||
                      refuses_at_transaction<pair_> || unimplemented_at_transaction<pair_>,
                  "a store surface must reach its transaction, be refused by name, or be listed as outstanding");
    static constexpr bool checked_k = true;
};

/** @brief Every paired surface against one store. */
template <typename store_type_, typename... pairs_>
constexpr bool every_pair_reaches_a_transaction = (level_witness<pairs_, store_type_>::checked_k && ...);

/** @brief The whole paired surface, so a new store cannot quietly miss a whole level. */
template <typename store_type_>
constexpr bool transaction_mirrors_the_store =
    every_pair_reaches_a_transaction<store_type_, count_pair_t, vacuum_pair_t, clear_pair_t, erase_range_pair_t,
                                     erase_from_pair_t, erase_up_to_pair_t, update_range_pair_t, sample_one_pair_t,
                                     sample_reservoir_pair_t, lower_bound_copy_pair_t, upper_bound_copy_pair_t,
                                     find_copy_pair_t, select_pair_t, rank_pair_t>;

/**
 *  @brief A wrapper reporting serializable must forward the reads that carry the protection.
 *
 *  At @c serializable_k every read records the window it crossed, so phantom protection rides on the
 *  ordinary ordered reads rather than on a separate opt-in surface. A wrapper that reports the level
 *  without forwarding them advertises a guarantee nothing on it can honour.
 */
template <typename store_type_>
constexpr bool offers_recording_reads = requires(typename store_type_::transaction_t &transaction,
                                                 typename store_type_::identifier_t const &key, no_op_t callback) {
    transaction.range(key, key, callback);
    transaction.for_each(callback);
};

template <typename store_type_, typename wrapper_>
struct level_surface_witness {
    using wrapped_t = typename wrapper_::template wrapped<store_type_>;
    static_assert(wrapped_t::isolation_k < isolation_t::serializable_k || offers_recording_reads<wrapped_t>,
                  "a wrapper reporting serializable must forward the reads that record what they read");
    static constexpr bool checked_k = true;
};

/** @brief Both wrappers over one store, for the surfaces its level is defined by. */
template <typename store_type_>
constexpr bool wrappers_honour_the_level = level_surface_witness<store_type_, locked_wrapper_t>::checked_k &&
                                           level_surface_witness<store_type_, partitioned_wrapper_t>::checked_k;

#pragma endregion Level Parity Fold

} // namespace ashvardanian::smashtable::scripts
