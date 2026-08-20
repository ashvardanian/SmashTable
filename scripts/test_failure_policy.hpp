/**
 *  @brief Detectors for two defect classes the suites had no way of naming - a status that reports one
 *      cause as another, and two failure policies hiding under one return type.
 *  @author Ash Vardanian
 *  @file scripts/test_failure_policy.hpp
 *  @date August 18, 2026
 *
 *  @section failure_policy_facade The Refusing Facade
 *
 *  @c refusing_store wraps a real store and refuses one named method on one chosen call, forwarding
 *  everything else untouched. Which call refuses is the point: a partitioned store walks its parts in
 *  ascending order, one call each, so "the fifth call to @c erase_range refuses" is exactly "the fifth
 *  partition refuses", and the tally of calls that still arrived afterwards is the observable footprint
 *  of the bulk policy.
 *
 *  @section failure_policy_causes Causes Against Statuses
 *
 *  @c status_of_cause is the table saying which status each distinct failure cause must be reported
 *  as. It is checked for pairwise distinctness at compile time, and a cause added to
 *  @c failure_cause_t with no entry leaves an incomplete type rather than quietly aliasing onto a
 *  status another cause already owns.
 *
 *  The runtime half is the other property the table cannot state: a status meaning "nothing happened"
 *  must come with nothing having happened. Every driven refusal is bracketed by a footprint taken
 *  before and after, which is what a ghost commit - a refusal reported while its writes were already
 *  published - fails.
 */
#pragma once
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`

#include <array>       // `std::array`
#include <limits>      // `std::numeric_limits`
#include <type_traits> // `std::remove_cvref_t`
#include <utility>     // `std::index_sequence`, `std::move`

#include <smashtable/shared.hpp>

#include "test_basic.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Refusal Plan

/**
 *  @brief The methods @c refusing_store can be told to refuse, and the tallies it keeps per method.
 *
 *  @c none_k is the standing instruction that refuses nothing, and @c methods_count_k is one past the
 *  last method, sizing the tallies rather than naming a method of its own.
 */
enum class refusable_method_t : std::uint8_t {
    none_k,
    clear_k,
    reserve_k,
    erase_range_k,
    erase_from_k,
    erase_up_to_k,
    update_range_k,
    upsert_k,
    methods_count_k,
};

/** @brief How many entries a per-method table needs. */
inline constexpr std::size_t refusable_methods_k = static_cast<std::size_t>(refusable_method_t::methods_count_k);

/** @brief The call index no call ever carries, as opposed to @c 0, which refuses the very first one. */
inline constexpr std::size_t no_refusal_k = std::numeric_limits<std::size_t>::max();

/**
 *  @brief The standing instruction @c refusing_store consults, plus the tally of calls that reached it.
 *
 *  Process-wide for the same reason @c copy_budget_t is: a partitioned store builds its parts through
 *  @c inner_store_t::make(), so a test has no reference to hand each of the sixteen.
 */
struct refusal_plan_t {
    /** @brief Which method refuses, or @c none_k while every call is forwarded. */
    static inline refusable_method_t refused_method {refusable_method_t::none_k};
    /** @brief The zero-based index, among calls to that method, of the one call that refuses. */
    static inline std::size_t refused_call {no_refusal_k};
    /** @brief The status that refusal reports. */
    static inline status_t refused_status {status_t::out_of_memory_heap_k};
    /** @brief How many calls each method has received since the last reset. */
    static inline std::array<std::size_t, refusable_methods_k> calls {};

    /** @brief Forgets the instruction and the tallies, so a fresh measurement starts from zero. */
    static void reset() noexcept {
        refused_method = refusable_method_t::none_k;
        refused_call = no_refusal_k;
        refused_status = status_t::out_of_memory_heap_k;
        calls = {};
    }

    /** @brief Arms the refusal of call @p call_index to @p method, reported as @p status. */
    static void refuse(refusable_method_t method, std::size_t call_index, status_t status) noexcept {
        refused_method = method;
        refused_call = call_index;
        refused_status = status;
    }

    /** @brief How many calls @p method has received since the last reset - the observable footprint. */
    [[nodiscard]] static std::size_t calls_count(refusable_method_t method) noexcept {
        return calls[static_cast<std::size_t>(method)];
    }

    /** @brief Records one call to @p method and reports whether the facade must refuse this one. */
    [[nodiscard]] static budget_outcome_t spend(refusable_method_t method) noexcept {
        std::size_t const index = calls[static_cast<std::size_t>(method)]++;
        if (method != refused_method || index != refused_call) return budget_outcome_t::granted_k;
        return budget_outcome_t::refused_k;
    }
};

#pragma endregion Refusal Plan

#pragma region Refusing Store

/**
 *  @brief A store that refuses one named call and forwards the rest, so a wrapper's own failure
 *    handling can be driven without a single allocation going wrong.
 *
 *  Derives from the store it stands in for rather than forwarding its surface by hand: the wrappers
 *  detect what a part can do through requires-expressions over dozens of members, and a hand-written
 *  facade would answer those questions about itself rather than about the store underneath. Only the
 *  refusable methods are redeclared, each carrying the constraint its base declares so an inner store
 *  that never had the method does not appear to grow one.
 *
 *  @tparam store_type_ The store standing behind the facade, such as @c monotonic_avl_map.
 */
template <typename store_type_>
class refusing_store : public store_type_ {
  public:
    using base_t = store_type_;
    using store_t = refusing_store;
    using self_t = refusing_store;
    using value_t = typename base_t::value_t;
    using comparator_t = typename base_t::comparator_t;
    using allocator_t = typename base_t::allocator_t;

    refusing_store() noexcept = default;
    explicit refusing_store(base_t &&base) noexcept : base_t(std::move(base)) {}

    [[nodiscard]] static expected<refusing_store> make(allocator_t const &allocator = {}) noexcept {
        expected<base_t> inner = base_t::make(allocator);
        if (!inner) return inner.status();
        return refusing_store {*std::move(inner)};
    }

    [[nodiscard]] static expected<refusing_store> make(comparator_t const &comparator,
                                                       allocator_t const &allocator) noexcept {
        expected<base_t> inner = base_t::make(comparator, allocator);
        if (!inner) return inner.status();
        return refusing_store {*std::move(inner)};
    }

    [[nodiscard]] status_t clear() noexcept {
        return refused_or_(refusable_method_t::clear_k, [&]() noexcept { return base_t::clear(); });
    }

    [[nodiscard]] status_t reserve(std::size_t size) noexcept {
        return refused_or_(refusable_method_t::reserve_k, [&]() noexcept { return base_t::reserve(size); });
    }

    using base_t::upsert; // ? Keeps the iterator-pair overload the base declares
    [[nodiscard]] status_t upsert(value_t &&element) noexcept {
        return refused_or_(refusable_method_t::upsert_k, [&]() noexcept { return base_t::upsert(std::move(element)); });
    }

    template <typename lower_type_, typename upper_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_range(lower_type_ &&lower, upper_type_ &&upper,
                                       callback_type_ &&callback = {}) noexcept
        requires requires(base_t &base) { base.erase_range(lower, upper, callback); }
    {
        return refused_or_(refusable_method_t::erase_range_k,
                           [&]() noexcept { return base_t::erase_range(lower, upper, callback); });
    }

    template <typename lower_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_from(lower_type_ &&lower, callback_type_ &&callback = {}) noexcept
        requires requires(base_t &base) { base.erase_from(lower, callback); }
    {
        return refused_or_(refusable_method_t::erase_from_k,
                           [&]() noexcept { return base_t::erase_from(lower, callback); });
    }

    template <typename upper_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t erase_up_to(upper_type_ &&upper, callback_type_ &&callback = {}) noexcept
        requires requires(base_t &base) { base.erase_up_to(upper, callback); }
    {
        return refused_or_(refusable_method_t::erase_up_to_k,
                           [&]() noexcept { return base_t::erase_up_to(upper, callback); });
    }

    /**
     *  @brief Reports a status where the base reports nothing, since a walk that cannot refuse cannot
     *    be made to. The partitioned wrapper accepts either shape and widens to this one.
     */
    template <typename lower_type_, typename upper_type_, typename callback_type_ = no_op_t>
    [[nodiscard]] status_t update_range(lower_type_ &&lower, upper_type_ &&upper,
                                        callback_type_ &&callback = {}) noexcept
        requires requires(base_t &base) { base.update_range(lower, upper, callback); }
    {
        return refused_or_(refusable_method_t::update_range_k,
                           [&]() noexcept { return base_t::update_range(lower, upper, callback); });
    }

  private:
    /** @brief Spends one call of @p method, answering the standing refusal or forwarding to the base. */
    template <typename forwarded_type_>
    [[nodiscard]] status_t refused_or_(refusable_method_t method, forwarded_type_ &&forwarded) noexcept {
        if (refusal_plan_t::spend(method) == budget_outcome_t::refused_k) return refusal_plan_t::refused_status;
        if constexpr (std::is_void_v<decltype(forwarded())>) {
            forwarded();
            return success_k;
        }
        else { return forwarded(); }
    }
};

#pragma endregion Refusing Store

#pragma region Failure Causes

/**
 *  @brief The distinct reasons an operation can be refused, each of which must carry its own status.
 *
 *  The list is closed on purpose: a cause added here with no @c status_of_cause entry leaves an
 *  incomplete type at the first use, which is what turns "a new cause silently reuses someone else's
 *  status" from a review question into a build failure.
 */
enum class failure_cause_t : std::uint8_t {
    /** @brief The allocator answered no room. */
    heap_refused_k,
    /** @brief A bounded probe sequence ran out of slots, which more memory would not fix. */
    probe_exhausted_k,
    /** @brief The key an operation demanded be there was not. */
    key_absent_k,
    /** @brief The key an operation demanded be free was taken. */
    key_present_k,
    /** @brief Something a transaction watched moved under it. */
    watch_conflicted_k,
    /** @brief One past the last cause, sizing the table rather than naming a cause. */
    causes_count_k,
};

/** @brief How many causes the table has to cover. */
inline constexpr std::size_t failure_causes_k = static_cast<std::size_t>(failure_cause_t::causes_count_k);

/** @brief The status one cause must be reported as. Declared with no definition: every cause specializes it. */
template <failure_cause_t cause_>
struct status_of_cause;

template <>
struct status_of_cause<failure_cause_t::heap_refused_k> {
    static constexpr status_t status_k = status_t::out_of_memory_heap_k;
};
template <>
struct status_of_cause<failure_cause_t::probe_exhausted_k> {
    static constexpr status_t status_k = status_t::capacity_exhausted_k;
};
template <>
struct status_of_cause<failure_cause_t::key_absent_k> {
    static constexpr status_t status_k = status_t::key_not_found_k;
};
template <>
struct status_of_cause<failure_cause_t::key_present_k> {
    static constexpr status_t status_k = status_t::key_already_exists_k;
};
template <>
struct status_of_cause<failure_cause_t::watch_conflicted_k> {
    static constexpr status_t status_k = status_t::read_conflict_k;
};

/** @brief The table read as an array, which is what makes it checkable rather than merely readable. */
template <std::size_t... indices_>
[[nodiscard]] constexpr std::array<status_t, sizeof...(indices_)> statuses_of_causes(
    std::index_sequence<indices_...>) noexcept {
    return {status_of_cause<static_cast<failure_cause_t>(indices_)>::status_k...};
}

/** @brief Whether no two causes name the same status, and none of them names success. */
[[nodiscard]] constexpr bool causes_name_distinct_statuses() noexcept {
    auto const statuses = statuses_of_causes(std::make_index_sequence<failure_causes_k> {});
    for (std::size_t first = 0; first != failure_causes_k; ++first) {
        if (succeeded(statuses[first])) return false;
        for (std::size_t second = first + 1; second != failure_causes_k; ++second)
            if (statuses[first] == statuses[second]) return false;
    }
    return true;
}

static_assert(causes_name_distinct_statuses(), "two failure causes share one status, so no caller can tell them apart");

/** @brief Whether a cause was driven against a given store at all, since not every store can produce every one. */
enum class cause_drive_t : bool { skipped_k, driven_k };

/**
 *  @brief What each cause was observed to report, so distinctness is asserted of the run and not only
 *    of the table.
 */
struct observed_causes_t {
    /** @brief The status each cause reported, meaningful only where @c drives says it was driven. */
    std::array<status_t, failure_causes_k> statuses {};
    /** @brief Whether each cause was reachable against the store under test. */
    std::array<cause_drive_t, failure_causes_k> drives {};

    /** @brief Records @p observed for @p cause, checking it against the table on the spot. */
    void note(failure_cause_t cause, status_t observed) noexcept {
        std::size_t const index = static_cast<std::size_t>(cause);
        auto const expected_statuses = statuses_of_causes(std::make_index_sequence<failure_causes_k> {});
        st_verify_eq_(observed, expected_statuses[index], "a cause must be reported as the status it owns");
        statuses[index] = observed;
        drives[index] = cause_drive_t::driven_k;
    }

    /** @brief Aborts unless the causes actually driven reported statuses no two of them shared. */
    void verify_distinct() const noexcept {
        std::size_t driven_count = 0;
        for (std::size_t first = 0; first != failure_causes_k; ++first) {
            if (drives[first] == cause_drive_t::skipped_k) continue;
            ++driven_count;
            for (std::size_t second = first + 1; second != failure_causes_k; ++second) {
                if (drives[second] == cause_drive_t::skipped_k) continue;
                st_verify_ne_(statuses[first], statuses[second],
                              "two distinct causes reported the same status against this store");
            }
        }
        st_verify_ge_(driven_count, 2, "a store must expose at least two causes for distinctness to mean anything");
    }
};

#pragma endregion Failure Causes

#pragma region Observable Footprint

/** @brief What an outside observer can see of a store, so "nothing happened" becomes a comparison. */
struct store_footprint_t {
    /** @brief How many members the store reports. */
    std::size_t size {0};
    /** @brief An order-independent mix of every member, which a changed value moves and a walk does not. */
    std::size_t checksum {0};

    bool operator==(store_footprint_t const &) const noexcept = default;
};

/** @brief Mixes one member into a footprint, folding in the mapped side wherever it is a number. */
template <typename member_type_>
[[nodiscard]] std::size_t checksum_of_member(member_type_ const &member) noexcept {
    auto const &key = mapping_key_or_itself<member_type_>(member);
    std::size_t mixed = hash<std::remove_cvref_t<decltype(key)>> {}(key);
    if constexpr (is_mapping<member_type_>)
        if constexpr (std::is_convertible_v<decltype(member.mapped), std::size_t>)
            mixed ^= static_cast<std::size_t>(member.mapped) * std::size_t {0x9E3779B9};
    return mixed;
}

/** @brief Takes the footprint of @p store, walking its members where it enumerates them. */
template <typename store_type_>
[[nodiscard]] store_footprint_t footprint_of(store_type_ const &store) noexcept {
    store_footprint_t seen;
    seen.size = store.size();
    if constexpr (requires(no_op_t callback) { store.for_each(callback); })
        st_verify_(store.for_each([&](typename store_type_::value_t const &member) noexcept {
            seen.checksum += checksum_of_member(member);
        }));
    return seen;
}

#pragma endregion Observable Footprint

#pragma region Cause Detectors

/**
 *  @brief A refused strict insert must name the occupied key, and must leave that key as it was.
 *  @param[out] observed Collects the status, so the run's distinctness can be asserted afterwards.
 */
template <typename store_type_>
void drive_present_key_cause(store_type_ &store, observed_causes_t &observed, std::size_t size) {

    using member_t = typename store_type_::value_type;
    if constexpr (requires(store_type_ &probe, member_t &&element) { probe.insert(std::move(element)); }) {
        store_footprint_t const before = footprint_of(store);
        auto duplicate = trivial_id_to_member<member_t>(size / 2, size * 7);
        status_t const refused = store.insert(std::move(duplicate));
        store_footprint_t const after = footprint_of(store);
        observed.note(failure_cause_t::key_present_k, refused);
        st_verify_eq_(before, after, "a refused strict insert must leave the occupied key untouched");
    }
}

/** @brief A refused strict update must name the absent key, and must not create it. */
template <typename store_type_>
void drive_absent_key_cause(store_type_ &store, observed_causes_t &observed, std::size_t size) {

    using member_t = typename store_type_::value_type;
    if constexpr (requires(store_type_ &probe, member_t &&element) { probe.update(std::move(element)); }) {
        store_footprint_t const before = footprint_of(store);
        auto missing = trivial_id_to_member<member_t>(size * 4, size * 7);
        status_t const refused = store.update(std::move(missing));
        store_footprint_t const after = footprint_of(store);
        observed.note(failure_cause_t::key_absent_k, refused);
        st_verify_eq_(before, after, "a refused strict update must not create the key it could not find");
    }
}

/**
 *  @brief A commit whose watch drifted must say so, and must publish none of what it staged.
 *
 *  The second half is the ghost commit: a transaction that writes a key nobody else touched, watches a
 *  key somebody does, and is refused. Reporting the conflict while the untouched key's new value is
 *  already visible is a refusal that lies about having done nothing.
 */
template <typename store_type_>
void drive_watch_conflict_cause(store_type_ &store, observed_causes_t &observed, std::size_t size) {

    using member_t = typename store_type_::value_type;
    if constexpr (store_type_::is_transactional::value) {
        auto contested = store.transaction();
        st_verify_((contested.has_value()) && "a store must be able to open a transaction");
        st_verify_(contested->watch(trivial_id_to_key<member_t>(1)));
        st_verify_(contested->upsert(trivial_id_to_member<member_t>(size * 5, size * 5)));

        // An outside writer moves the watched key, which is what the validation has to catch.
        st_verify_(store.upsert(trivial_id_to_member<member_t>(1, size * 9)));

        store_footprint_t const before = footprint_of(store);
        status_t refused = contested->stage();
        if (succeeded(refused)) refused = contested->commit();
        store_footprint_t const after = footprint_of(store);

        observed.note(failure_cause_t::watch_conflicted_k, refused);
        st_verify_eq_(before, after, "a refused commit must publish none of what it staged");
        [[maybe_unused]] status_t const unwound = contested->rollback();
    }
}

/**
 *  @brief An allocator that answers no room must be reported as no room, and must change nothing.
 *
 *  Only stores built around @c stateful_allocator can be driven here; every other store skips the
 *  cause rather than pretending to have produced it.
 */
template <typename store_type_>
void drive_heap_refusal_cause(store_type_ &store, allocation_ledger_t &ledger, observed_causes_t &observed,
                              std::size_t size) {

    using member_t = typename store_type_::value_type;
    store_footprint_t const before = footprint_of(store);
    ledger.refuse_everything();
    auto rejected = trivial_id_to_member<member_t>(size * 6, size * 6);
    status_t const refused = store.upsert(std::move(rejected));
    ledger.allow(unlimited_budget_k);
    store_footprint_t const after = footprint_of(store);

    observed.note(failure_cause_t::heap_refused_k, refused);
    st_verify_eq_(before, after, "a store that could not allocate must not have stored anything");
}

/**
 *  @brief Every failure cause this store can produce reports its own status, and none of them lies
 *    about having left the store alone.
 *
 *  @param[in] size How many members to seed, which also spaces the keys the causes are driven with.
 */
template <typename store_type_>
void test_store_maps_causes_to_distinct_statuses(std::size_t size = 32) {

    using store_t = store_type_;
    using member_t = typename store_t::value_type;

    auto built = store_t::make();
    st_verify_((built.has_value()) && "the store under test must build");
    store_t &store = *built;
    for (std::size_t identifier = 0; identifier != size; ++identifier)
        st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier, identifier)));

    observed_causes_t observed;
    drive_present_key_cause(store, observed, size);
    drive_absent_key_cause(store, observed, size);
    drive_watch_conflict_cause(store, observed, size);
    observed.verify_distinct();
}

/**
 *  @brief The same detector for a store whose allocator can be told to refuse, which adds the heap cause.
 *  @param[in] size How many members to seed before the allocator is closed.
 */
template <typename store_type_>
void test_budgeted_store_maps_causes_to_distinct_statuses(std::size_t size = 32) {

    using store_t = store_type_;
    using member_t = typename store_t::value_type;
    using allocator_t = typename store_t::allocator_t;

    allocation_ledger_t ledger;
    {
        auto built = store_t::make(allocator_t {ledger});
        st_verify_((built.has_value()) && "the budgeted store under test must build");
        store_t &store = *built;
        for (std::size_t identifier = 0; identifier != size; ++identifier)
            st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier, identifier)));

        observed_causes_t observed;
        drive_heap_refusal_cause(store, ledger, observed, size);
        drive_present_key_cause(store, observed, size);
        drive_absent_key_cause(store, observed, size);
        drive_watch_conflict_cause(store, observed, size);
        observed.verify_distinct();

        clear_container(store);
    }
    ledger.verify_balanced();
}

/**
 *  @brief A wrapper must relay the cause its part named, verbatim, and publish nothing while doing it.
 *
 *  Every cause is driven through @c refusing_store rather than through a real failure, which is the only
 *  way to ask a wrapper about a cause its parts happen never to produce - an exhausted probe under a
 *  tree, say. A wrapper that folds several causes into one status fails on the first of them.
 */
template <typename store_type_, std::size_t... indices_>
void drive_relayed_causes(store_type_ &store, std::size_t size, std::index_sequence<indices_...>) {

    using member_t = typename store_type_::value_type;
    auto drive_one = [&](std::size_t offset, status_t named) noexcept {
        refusal_plan_t::reset();
        refusal_plan_t::refuse(refusable_method_t::upsert_k, 0, named);
        store_footprint_t const before = footprint_of(store);
        status_t const reported = store.upsert(trivial_id_to_member<member_t>(size * 3 + offset, offset));
        store_footprint_t const after = footprint_of(store);
        refusal_plan_t::reset();
        st_verify_eq_(reported, named, "a wrapper must relay the cause its part named, not one of its own");
        st_verify_eq_(before, after, "a relayed refusal must leave the store as the caller found it");
    };
    (drive_one(indices_, status_of_cause<static_cast<failure_cause_t>(indices_)>::status_k), ...);
}

/** @brief Runs every cause through @p store_type_, whose parts must be @c refusing_store. */
template <typename store_type_>
void test_wrapper_relays_every_cause(std::size_t size = 64) {

    using member_t = typename store_type_::value_type;

    refusal_plan_t::reset();
    auto built = store_type_::make();
    st_verify_((built.has_value()) && "the wrapper under test must build");
    store_type_ &store = *built;
    for (std::size_t identifier = 0; identifier != size; ++identifier)
        st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier, identifier)));

    drive_relayed_causes(store, size, std::make_index_sequence<failure_causes_k> {});
}

#pragma endregion Cause Detectors

#pragma region Bulk Failure Policy

/**
 *  @brief What a partitioned bulk method does with the partitions after one of them refuses.
 *
 *  @c stop_at_first_k is what @c for_all does - the walk returns on the refusal and the partitions
 *  behind it are never touched. @c attempt_every_k is what the range erasures do - every partition is
 *  attempted and the last refusal is the one reported.
 */
enum class bulk_failure_policy_t : std::uint8_t { stop_at_first_k, attempt_every_k };

/**
 *  @brief The partitioned methods that walk every partition, each of which must declare a policy.
 *    @c methods_count_k is one past the last of them, and sizes the sweep rather than naming a method.
 */
enum class bulk_method_t : std::uint8_t {
    clear_k,
    reserve_k,
    erase_range_k,
    erase_from_k,
    erase_up_to_k,
    update_range_k,
    methods_count_k,
};

/** @brief How many bulk methods the sweep has to cover. */
inline constexpr std::size_t bulk_methods_k = static_cast<std::size_t>(bulk_method_t::methods_count_k);

/**
 *  @brief What one bulk method promises and how to drive it. Declared with no definition on purpose.
 *
 *  A method added to @c bulk_method_t without a specialization here leaves an incomplete type at the
 *  sweep below, so a new all-partition walk cannot join a policy by accident - it has to say which one
 *  it keeps, and hand the detector a way to run it.
 */
template <bulk_method_t method_>
struct bulk_method_traits;

template <>
struct bulk_method_traits<bulk_method_t::clear_k> {
    static constexpr bulk_failure_policy_t policy_k = bulk_failure_policy_t::stop_at_first_k;
    static constexpr refusable_method_t refusable_k = refusable_method_t::clear_k;

    template <typename store_type_>
    [[nodiscard]] static status_t drive(store_type_ &store) noexcept {
        return store.clear();
    }
};

template <>
struct bulk_method_traits<bulk_method_t::reserve_k> {
    static constexpr bulk_failure_policy_t policy_k = bulk_failure_policy_t::stop_at_first_k;
    static constexpr refusable_method_t refusable_k = refusable_method_t::reserve_k;

    template <typename store_type_>
    [[nodiscard]] static status_t drive(store_type_ &store) noexcept {
        return store.reserve(1024);
    }
};

template <>
struct bulk_method_traits<bulk_method_t::erase_range_k> {
    static constexpr bulk_failure_policy_t policy_k = bulk_failure_policy_t::attempt_every_k;
    static constexpr refusable_method_t refusable_k = refusable_method_t::erase_range_k;

    template <typename store_type_>
    [[nodiscard]] static status_t drive(store_type_ &store) noexcept {
        using member_t = typename store_type_::value_type;
        return store.erase_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(1000));
    }
};

template <>
struct bulk_method_traits<bulk_method_t::erase_from_k> {
    static constexpr bulk_failure_policy_t policy_k = bulk_failure_policy_t::attempt_every_k;
    static constexpr refusable_method_t refusable_k = refusable_method_t::erase_from_k;

    template <typename store_type_>
    [[nodiscard]] static status_t drive(store_type_ &store) noexcept {
        using member_t = typename store_type_::value_type;
        return store.erase_from(trivial_id_to_key<member_t>(0));
    }
};

template <>
struct bulk_method_traits<bulk_method_t::erase_up_to_k> {
    static constexpr bulk_failure_policy_t policy_k = bulk_failure_policy_t::attempt_every_k;
    static constexpr refusable_method_t refusable_k = refusable_method_t::erase_up_to_k;

    template <typename store_type_>
    [[nodiscard]] static status_t drive(store_type_ &store) noexcept {
        using member_t = typename store_type_::value_type;
        return store.erase_up_to(trivial_id_to_key<member_t>(1000));
    }
};

template <>
struct bulk_method_traits<bulk_method_t::update_range_k> {
    static constexpr bulk_failure_policy_t policy_k = bulk_failure_policy_t::attempt_every_k;
    static constexpr refusable_method_t refusable_k = refusable_method_t::update_range_k;

    template <typename store_type_>
    [[nodiscard]] static status_t drive(store_type_ &store) noexcept {
        using member_t = typename store_type_::value_type;
        return store.update_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(1000),
                                  [](auto const &, auto &) noexcept {});
    }
};

/**
 *  @brief Whether the store itself declares the policy of @p method_, which the detector then checks
 *    its own table against rather than replacing it.
 *
 *  Until the wrapper carries the constants, the table above is the only statement of the policy and
 *  this answers @c false, which keeps the detector running against the behaviour as it stands.
 */
template <typename store_type_, bulk_method_t method_>
[[nodiscard]] constexpr bool store_declares_bulk_policy() noexcept {
    if constexpr (method_ == bulk_method_t::clear_k) return requires { store_type_::clear_policy_k; };
    else if constexpr (method_ == bulk_method_t::reserve_k) return requires { store_type_::reserve_policy_k; };
    else if constexpr (method_ == bulk_method_t::erase_range_k) return requires { store_type_::erase_range_policy_k; };
    else if constexpr (method_ == bulk_method_t::erase_from_k) return requires { store_type_::erase_from_policy_k; };
    else if constexpr (method_ == bulk_method_t::erase_up_to_k) return requires { store_type_::erase_up_to_policy_k; };
    else return requires { store_type_::update_range_policy_k; };
}

/** @brief The constant the store declares for @p method_, in the store's own enumeration. */
template <typename store_type_, bulk_method_t method_>
[[nodiscard]] constexpr auto library_bulk_policy() noexcept {
    if constexpr (method_ == bulk_method_t::clear_k) return store_type_::clear_policy_k;
    else if constexpr (method_ == bulk_method_t::reserve_k) return store_type_::reserve_policy_k;
    else if constexpr (method_ == bulk_method_t::erase_range_k) return store_type_::erase_range_policy_k;
    else if constexpr (method_ == bulk_method_t::erase_from_k) return store_type_::erase_from_policy_k;
    else if constexpr (method_ == bulk_method_t::erase_up_to_k) return store_type_::erase_up_to_policy_k;
    else return store_type_::update_range_policy_k;
}

/**
 *  @brief The same policy read into this header's enumeration, matched by name rather than by value so
 *    the two enumerations stay free to order their enumerators differently.
 */
template <typename store_type_, bulk_method_t method_>
[[nodiscard]] constexpr bulk_failure_policy_t declared_bulk_policy() noexcept {
    constexpr auto named = library_bulk_policy<store_type_, method_>();
    using library_policy_t = std::remove_cvref_t<decltype(named)>;
    return named == library_policy_t::stop_at_first_k ? bulk_failure_policy_t::stop_at_first_k
                                                      : bulk_failure_policy_t::attempt_every_k;
}

/**
 *  @brief Drives one bulk method through a refusal at partition @p refusing_partition and asserts the
 *    footprint matches the policy the method declares.
 *
 *  The footprint is the tally of calls that reached the partitions: a walk that stops at the first
 *  refusal leaves the tally at the refusing partition, and one that attempts every partition leaves it
 *  at the partition count. Nothing else distinguishes the two, which is why the wrappers could carry
 *  both under one return type unnoticed.
 */
template <typename store_type_, bulk_method_t method_>
void test_one_bulk_method_matches_policy(std::size_t size, std::size_t refusing_partition) {

    using traits_t = bulk_method_traits<method_>;
    using member_t = typename store_type_::value_type;

    // Named before anything is guarded: a method whose traits are missing has to fail the build here,
    // not disappear into a requires-expression that quietly answers no.
    constexpr bulk_failure_policy_t declared_k = traits_t::policy_k;
    constexpr refusable_method_t refusable_k = traits_t::refusable_k;

    if constexpr (requires(store_type_ &store) { traits_t::drive(store); }) {
        // Where the wrapper states the policy itself, the detector's table has to agree with it, or one
        // of the two is describing a method the other does not.
        if constexpr (store_declares_bulk_policy<store_type_, method_>())
            static_assert(declared_bulk_policy<store_type_, method_>() == declared_k,
                          "the wrapper and the detector disagree about a bulk method's failure policy");

        refusal_plan_t::reset();
        auto built = store_type_::make();
        st_verify_((built.has_value()) && "the partitioned store under test must build");
        store_type_ &store = *built;
        for (std::size_t identifier = 0; identifier != size; ++identifier)
            st_verify_(store.upsert(trivial_id_to_member<member_t>(identifier, identifier)));

        refusal_plan_t::reset();
        refusal_plan_t::refuse(refusable_k, refusing_partition, status_t::out_of_memory_heap_k);
        status_t const reported = traits_t::drive(store);
        std::size_t const attempted = refusal_plan_t::calls_count(refusable_k);
        refusal_plan_t::reset();

        st_verify_eq_(reported, status_t::out_of_memory_heap_k, "a bulk walk must report the refusal it met");
        if constexpr (declared_k == bulk_failure_policy_t::stop_at_first_k)
            st_verify_eq_(attempted, refusing_partition + 1,
                          "a stop-at-first walk must leave the partitions behind the refusal untouched");
        else
            st_verify_eq_(attempted, store_type_::partitions_k,
                          "an attempt-every walk must reach every partition despite the refusal");
    }
}

/** @brief Sweeps every enumerator, which is where a method with no declared policy fails to compile. */
template <typename store_type_, std::size_t... indices_>
void sweep_bulk_methods(std::size_t size, std::size_t refusing_partition, std::index_sequence<indices_...>) {
    (test_one_bulk_method_matches_policy<store_type_, static_cast<bulk_method_t>(indices_)>(size, refusing_partition),
     ...);
}

/**
 *  @brief Every all-partition method keeps the failure policy it declares, driven at one chosen partition.
 *  @warning The partitions have to be @c refusing_store, since that is what the refusal is armed on; a
 *    store built over a plain part never refuses and the detector says so rather than passing quietly.
 *  @param[in] size How many members to seed, spread across the partitions by their hashes.
 *  @param[in] refusing_partition Which partition refuses, zero-based - the fifth by default, so a walk
 *    that stopped and one that carried on leave measurably different tallies.
 */
template <typename store_type_>
void test_bulk_methods_match_declared_policy(std::size_t size = 256, std::size_t refusing_partition = 4) {
    st_verify_lt_(refusing_partition + 1, store_type_::partitions_k,
                  "the refusal has to land where partitions remain behind it, or the policies read alike");
    sweep_bulk_methods<store_type_>(size, refusing_partition, std::make_index_sequence<bulk_methods_k> {});
}

#pragma endregion Bulk Failure Policy

} // namespace ashvardanian::smashtable::scripts
