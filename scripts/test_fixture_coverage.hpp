/**
 *  @brief Runs the accounting fixtures through every container family, so a leak or a severed probe run
 *      becomes arithmetic rather than something only a sanitizer might notice.
 *  @author Ash Vardanian
 *  @file scripts/test_fixture_coverage.hpp
 *  @date August 18, 2026
 *
 *  These suites exist because of what the fixtures can see that the containers' own tests cannot. A key
 *  that owns no heap leaks invisibly - the node around it is freed, so a sanitizer reports nothing, and
 *  only a construction/destruction tally catches it. A well-spread hash never builds a probe run, so the
 *  tombstone and chain handling of the open-addressed cores goes unexercised until a key is made to
 *  collide on purpose.
 */
#pragma once
#include <cstddef> // `std::size_t`

#include "test_basic.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Element Accounting

/**
 *  @brief Every key a container constructs, it must destroy.
 *
 *  Drives the paths where a container moves elements around rather than merely holding them - growth,
 *  overwrite, erase, and teardown - and asserts the tally returns to zero. A defect count above zero
 *  means a key was read after destruction or destroyed twice, which the per-object magic word catches.
 */
template <typename container_type_>
void test_container_balances_counted_keys(std::size_t size = 128) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    counted_key_t::reset();
    {
        container_t container;
        for (std::size_t identifier = 0; identifier != size; ++identifier)
            st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

        // Overwriting is where a container can drop the incumbent without destroying it.
        for (std::size_t identifier = 0; identifier < size; identifier += 2)
            st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

        for (std::size_t identifier = 0; identifier < size; identifier += 3) [[maybe_unused]]
            auto const erased = container.erase(trivial_id_to_key<member_t>(identifier));

        st_verify_gt_(counted_key_t::alive(), 0, "the container must be holding keys at this point");
        clear_container(container);
    }

    st_verify_eq_(counted_key_t::defects_count(), std::size_t {0});
    counted_key_t::verify_balanced();
}

/** @brief A transaction that rolls back must destroy the keys it staged, exactly as a commit does. */
template <typename container_type_>
void test_rollback_balances_counted_keys(std::size_t size = 64) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    counted_key_t::reset();
    {
        container_t container;

        auto committed = container.transaction();
        for (std::size_t identifier = 0; identifier != size; ++identifier)
            st_verify_(committed->upsert(trivial_id_to_member<member_t>(identifier)));
        st_verify_(committed->stage());
        st_verify_(committed->commit());

        auto abandoned = container.transaction();
        for (std::size_t identifier = size; identifier != size * 2; ++identifier)
            st_verify_(abandoned->upsert(trivial_id_to_member<member_t>(identifier)));
        st_verify_(abandoned->stage());
        st_verify_(abandoned->rollback());

        clear_container(container);
    }

    st_verify_eq_(counted_key_t::defects_count(), std::size_t {0});
    counted_key_t::verify_balanced();
}

#pragma endregion Element Accounting

#pragma region Probe Runs

/**
 *  @brief Every key stays reachable when a whole group shares one home slot.
 *
 *  The hash of @c colliding_key_t keeps only the group, so consecutive identifiers pile into one probe
 *  run. That is the shape in which a tombstone left by an erase can sever the run behind it and strand
 *  the keys past the gap - a well-spread hash never builds a run long enough to show it.
 */
template <typename container_type_>
void test_container_walks_collision_runs(std::size_t groups = 4) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    std::size_t const size = groups * colliding_key_t::collision_run_length_k;

    container_t container;
    for (std::size_t identifier = 0; identifier != size; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    for (std::size_t identifier = 0; identifier != size; ++identifier) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(identifier)), true,
                      "every key in a shared probe run must stay reachable");
    }

    // Erasing from the middle of each run is what leaves a tombstone the survivors must be found past.
    for (std::size_t identifier = 1; identifier < size; identifier += colliding_key_t::collision_run_length_k)
        [[maybe_unused]]
        auto const erased = container.erase(trivial_id_to_key<member_t>(identifier));

    for (std::size_t identifier = 0; identifier != size; ++identifier) {
        bool const erased = identifier % colliding_key_t::collision_run_length_k == 1;
        st_verify_ne_(container.contains(trivial_id_to_key<member_t>(identifier)), erased,
                      "a tombstone must hide its own key and no other");
    }
}

/** @brief A staged write over a saturated probe run stays invisible until it commits, then is findable. */
template <typename container_type_>
void test_transaction_walks_collision_runs(std::size_t groups = 4) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    std::size_t const size = groups * colliding_key_t::collision_run_length_k;

    container_t container;
    for (std::size_t identifier = 0; identifier != size; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    auto transaction = container.transaction();
    for (std::size_t identifier = size; identifier != size + groups; ++identifier)
        st_verify_(transaction->upsert(trivial_id_to_member<member_t>(identifier)));
    st_verify_(transaction->stage());

    for (std::size_t identifier = size; identifier != size + groups; ++identifier) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(identifier)), false,
                      "a staged key must stay invisible however long its probe run is");
    }

    st_verify_(transaction->commit());

    for (std::size_t identifier = 0; identifier != size + groups; ++identifier) {
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(identifier)), true,
                      "every key must be reachable once the transaction publishes");
    }
}

#pragma endregion Probe Runs

#pragma region Lookup Cost

/**
 *  @brief A hash lookup compares a bounded number of candidates, however many keys are present.
 *
 *  A table's whole claim is that a probe run stays short. The defect that breaks it is one pathological
 *  run at one key, so every key is probed rather than a stride of them, and the worst run the table can
 *  produce at @p size is stated outright - it moves only when the hash, the load factor or @p size does.
 */
template <typename container_type_>
void test_hash_lookup_cost_is_bounded(std::size_t size = 4096) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (std::size_t identifier = 0; identifier != size; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    std::size_t worst_equalities = 0;
    for (std::size_t identifier = 0; identifier != size; ++identifier) {
        counting_call_tally_t::reset();
        st_verify_eq_(container.contains(trivial_id_to_key<member_t>(identifier)), true);
        std::size_t const equalities = counting_call_tally_t::equalities_count();
        worst_equalities = equalities > worst_equalities ? equalities : worst_equalities;
    }

    // The worst run this hash and this load factor produce over @p size keys, measured over all of them.
    st_verify_eq_(worst_equalities, 19, "a lookup must walk its probe run, not the table");
}

#pragma endregion Lookup Cost

#pragma region Hostile Element Types

/** @brief A container over an over-aligned key must honour that alignment in the storage it hands out. */
template <typename container_type_>
void test_container_honours_over_alignment(std::size_t size = 64) {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (std::size_t identifier = 0; identifier != size; ++identifier)
        st_verify_(container.upsert(trivial_id_to_member<member_t>(identifier)));

    for (std::size_t identifier = 0; identifier != size; ++identifier)
        st_verify_(container.find(
            trivial_id_to_key<member_t>(identifier),
            [](member_t const &member) noexcept {
                st_verify_((mapping_key_or_itself(member).is_correctly_aligned()) &&
                           "an over-aligned key must not be handed under-aligned storage");
            },
            []() noexcept { st_verify_(false && "every stored key must be findable"); }));
}

/**
 *  @brief A copy that refuses must surface as a status, not as a silently truncated result.
 *
 *  @c find_copy is the one read that materializes a value, so it is the one read that can fail. A key
 *  whose @c copy() is refused on a schedule is what makes that path reachable without real exhaustion.
 */
template <typename container_type_>
void test_find_copy_reports_a_refused_copy() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    copy_budget_t::reset();
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1)));

    copy_budget_t::allow(0);
    auto const refused = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_((!refused) && "a refused copy must not report a value");
    st_verify_gt_(copy_budget_t::refusals_count, 0, "the budget must have been consulted");

    copy_budget_t::reset();
    auto const granted = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_((granted) && "the same read must succeed once the budget allows it");
}

#pragma endregion Hostile Element Types

} // namespace ashvardanian::smashtable::scripts
