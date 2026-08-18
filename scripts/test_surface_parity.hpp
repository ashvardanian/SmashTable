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
 *  Each carries a single @c offered_k predicate over a store type, and nothing else - the fold below
 *  supplies the wrapper and the store. A predicate asks for exactly one method, never a conjunction:
 *  two methods behind one gate means the store offering only the second loses it through every
 *  wrapper, which is the defect class this header exists to catch.
 */
struct sample_one_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.sample_one(key, key, callback, callback); };
};
struct sample_reservoir_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(store_type_ const &store, typename store_type_::identifier_t const &key, no_op_t callback,
                 std::size_t seen) { store.sample_reservoir(key, key, callback, seen, seen, callback); };
};
struct equal_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.equal_range(key, callback); };
};
struct select_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store, std::size_t ordinal, no_op_t callback) {
        store.select(ordinal, callback, callback);
    };
};
struct rank_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.rank(key, callback, callback); };
};
struct ranked_size_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store) { store.ranked_size(); };
};
struct for_each_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(store_type_ const &store, no_op_t callback) { store.for_each(callback); };
};
struct lower_bound_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.lower_bound(key, callback, callback); };
};
struct upper_bound_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.upper_bound(key, callback, callback); };
};
struct range_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.range(key, key, callback); };
};
struct erase_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.erase_range(key, key, callback); };
};
struct erase_from_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.erase_from(key, callback); };
};
struct erase_up_to_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.erase_up_to(key, callback); };
};
struct update_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ &store, typename store_type_::identifier_t const &key,
                                               no_op_t callback) { store.update_range(key, key, callback); };
};
struct insert_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(store_type_ &store, typename store_type_::value_t &&element) { store.insert(std::move(element)); };
};
struct insert_if_missing_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ &store, typename store_type_::value_t &&element) {
        store.insert_if_missing(std::move(element));
    };
};
struct insert_or_assign_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ &store, typename store_type_::value_t &&element) {
        store.insert_or_assign(std::move(element));
    };
};
struct update_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(store_type_ &store, typename store_type_::value_t &&element) { store.update(std::move(element)); };
};
struct find_copy_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(store_type_ const &store, typename store_type_::identifier_t const &key) { store.find_copy(key); };
};
struct vacuum_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ &store) { store.vacuum(); };
};
struct vacuum_window_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(store_type_ &store, typename store_type_::identifier_t const &key) { store.vacuum(key, key); };
};
struct versions_count_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(store_type_ const &store, typename store_type_::identifier_t const &key) {
            store.versions_count();
            store.versions_count(key);
        };
};
struct low_water_mark_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store) { store.low_water_mark(); };
};
struct visible_cursor_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ const &store, no_op_t callback) {
        store.visible_keys().peek(callback, callback);
        store.visible_keys().peek_copy();
    };
};
struct shard_protocol_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(store_type_ &store, typename shared_clock_of<store_type_>::type &clock,
                                               typename store_type_::generation_t stamp) {
        store.attach_clock(clock);
        store.transaction_at(stamp, stamp);
    };
};
struct heterogeneous_erase_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
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
    static constexpr bool offered_k =
        requires(typename store_type_::transaction_t &transaction, typename store_type_::value_t &&element) {
            transaction.insert_if_missing(std::move(element));
        };
};
struct transaction_changes_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(typename store_type_::transaction_t const &transaction) {
        transaction.has_changes();
        transaction.changes_count();
    };
};
struct transaction_for_each_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k = requires(typename store_type_::transaction_t const &transaction,
                                               no_op_t callback) { transaction.for_each(callback); };
};
struct transaction_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.range(key, key, callback); };
};
struct transaction_equal_range_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.equal_range(key, callback); };
};
struct transaction_lower_bound_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.lower_bound(key, callback, callback); };
};
struct transaction_upper_bound_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(typename store_type_::transaction_t const &transaction, typename store_type_::identifier_t const &key,
                 no_op_t callback) { transaction.upper_bound(key, callback, callback); };
};
struct transaction_find_copy_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(typename store_type_::transaction_t const &transaction,
                 typename store_type_::identifier_t const &key) { transaction.find_copy(key); };
};
struct transaction_generation_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
        requires(typename store_type_::transaction_t const &transaction) { transaction.generation(); };
};
struct transaction_reserve_surface_t {
    template <typename store_type_>
    static constexpr bool offered_k =
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
constexpr bool waives_surface_k = false;

/**
 *  @brief An ordinal spans partitions, and no partition knows the other fifteen's counts. What
 *    @c partitioned_store::select indexes is a merged order taken under every partition's lock, not
 *    the augmented count a core keeps over its own subtree, so a sum of the parts would name a size
 *    the merged order never has.
 */
template <>
constexpr bool waives_surface_k<ranked_size_surface_t, partitioned_wrapper_t> = true;

/**
 *  @brief A cursor outlives the call that produced it and reads the store through a bare pointer,
 *    so handing one out past a mutex is handing out an unsynchronized reader. Both wrappers refuse
 *    it; a caller that wants to walk uses @c for_each or @c range, which keep the lock for the walk.
 */
template <>
constexpr bool waives_surface_k<visible_cursor_surface_t, locked_wrapper_t> = true;
template <>
constexpr bool waives_surface_k<visible_cursor_surface_t, partitioned_wrapper_t> = true;

/**
 *  @brief A shard set owns exactly one clock and hands it to every partition, so it consumes the
 *    protocol rather than re-exporting it. Letting an outer set replace that clock would strand the
 *    stamps and reader claims the partitions already hold. @c locked_store forwards it instead,
 *    which is what lets a shard set be built over one.
 */
template <>
constexpr bool waives_surface_k<shard_protocol_surface_t, partitioned_wrapper_t> = true;

/**
 *  @brief An exact match lives in one partition, but the key after it can live in any of them, so a
 *    transaction's inclusive bound is a merge of every partition's own bound over staged writes as
 *    well as the snapshot. The store-level @c lower_bound answers by probing then stepping, which is
 *    not atomic and is not what a transaction may promise. Until that merged walk exists the surface
 *    stays absent rather than sometimes right.
 */
template <>
constexpr bool waives_surface_k<transaction_lower_bound_surface_t, partitioned_wrapper_t> = true;

#pragma endregion Waivers

#pragma region Parity Fold

/**
 *  @brief One assertion: if @c store_type_ offers @c surface_, so must @c wrapper_ over it.
 *    A struct rather than a bare @c constexpr @c bool so the diagnostic names all three - the status
 *    quo is an overload-resolution error at the call site, which names none of them.
 */
template <typename surface_, typename wrapper_, typename store_type_>
struct parity_witness_t {
    static_assert(!surface_::template offered_k<store_type_> ||
                      surface_::template offered_k<typename wrapper_::template wrapped<store_type_>> ||
                      waives_surface_k<surface_, typename wrapper_::waiver_group_t>,
                  "a wrapper must offer every surface its inner store offers, or waive it by name");
    static constexpr bool checked_k = true;
};

/** @brief Every named surface against one wrapper and one store, folded into a single value. */
template <typename wrapper_, typename store_type_, typename... surfaces_>
constexpr bool every_surface_survives_k = (parity_witness_t<surfaces_, wrapper_, store_type_>::checked_k && ...);

/** @brief The whole public store surface against one wrapper. */
template <typename wrapper_, typename store_type_>
constexpr bool store_surface_survives_k = every_surface_survives_k<
    wrapper_, store_type_, sample_one_surface_t, sample_reservoir_surface_t, equal_range_surface_t, select_surface_t,
    rank_surface_t, ranked_size_surface_t, for_each_surface_t, lower_bound_surface_t, upper_bound_surface_t,
    range_surface_t, erase_range_surface_t, erase_from_surface_t, erase_up_to_surface_t, update_range_surface_t,
    insert_surface_t, insert_if_missing_surface_t, insert_or_assign_surface_t, update_surface_t, find_copy_surface_t,
    vacuum_surface_t, vacuum_window_surface_t, versions_count_surface_t, low_water_mark_surface_t,
    visible_cursor_surface_t, shard_protocol_surface_t, heterogeneous_erase_surface_t>;

/** @brief The whole transaction surface against one wrapper. */
template <typename wrapper_, typename store_type_>
constexpr bool transaction_surface_survives_k =
    every_surface_survives_k<wrapper_, store_type_, transaction_insert_if_missing_surface_t,
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
constexpr bool every_wrapper_keeps_surfaces_k =
    store_surface_survives_k<locked_wrapper_t, store_type_> &&
    store_surface_survives_k<partitioned_wrapper_t, store_type_> &&
    store_surface_survives_k<partitioned_over_locked_wrapper_t, store_type_> &&
    transaction_surface_survives_k<locked_wrapper_t, store_type_> &&
    transaction_surface_survives_k<partitioned_wrapper_t, store_type_> &&
    transaction_surface_survives_k<partitioned_over_locked_wrapper_t, store_type_>;

/**
 *  @brief A transparent wrapper must not change what an outer wrapper concludes.
 *
 *  Not "a wrapper never lowers @c isolation_k" - @c partitioned_store legitimately lowers it over a
 *  store whose visibility is not decided by a stamp. The honest property is that inserting a wrapper
 *  meant to be invisible changes nothing, which is what fails the moment a trait alias stops being
 *  forwarded.
 */
template <typename store_type_>
constexpr bool nesting_preserves_isolation_k =
    partitioned_store<locked_store<store_type_>>::isolation_k == partitioned_store<store_type_>::isolation_k;

/**
 *  @brief A transaction that moves must move-assign, and must not copy.
 *    Movable-but-not-move-assignable is almost always a defaulted operator the compiler deleted over
 *    a reference member, and it fails where the transaction is stored rather than where it is declared.
 */
template <typename store_type_>
constexpr bool transaction_moves_as_a_value_k =
    std::is_nothrow_move_constructible_v<typename store_type_::transaction_t> &&
    std::is_move_assignable_v<typename store_type_::transaction_t> &&
    !std::is_copy_constructible_v<typename store_type_::transaction_t>;

/**
 *  @brief The composite surfaces the older assertions name, spelled from the atoms above rather than
 *    beside them - a second definition of "ordered" is a second thing to keep in step.
 */
template <typename store_type_>
constexpr bool offers_ordered_surface_k =
    lower_bound_surface_t::offered_k<store_type_> && upper_bound_surface_t::offered_k<store_type_> &&
    range_surface_t::offered_k<store_type_> && erase_range_surface_t::offered_k<store_type_>;

/** @brief The ordinal surface, which only a core summing subtree counts can answer. */
template <typename store_type_>
constexpr bool offers_order_statistics_k =
    select_surface_t::offered_k<store_type_> && rank_surface_t::offered_k<store_type_>;

/** @brief The one walk an unordered core can offer. */
template <typename store_type_>
constexpr bool offers_enumeration_k = for_each_surface_t::offered_k<store_type_>;

#pragma endregion Parity Fold

} // namespace ashvardanian::smashtable::scripts
