/**
 *  @brief Regressions for the reclamation and reporting defects of the @c std::set-backed oracle.
 *      Templated on the container so the chain-backed store can be held to the same behaviour.
 *  @author Ash Vardanian
 *  @file scripts/test_reference_store_defects.hpp
 *  @date August 17, 2026
 */
#pragma once
#include "test.hpp"
#include "test_basic.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Tombstone Reclamation

/** @brief Commits an erase of @p identifier through a fresh transaction. */
template <typename container_type_>
void commit_erase_of(container_type_ &container, trivial_id_t identifier) {
    using member_t = typename container_type_::value_type;
    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(transaction->erase(trivial_id_to_key<member_t>(identifier)));
    st_verify_(transaction->stage());
    st_verify_(transaction->commit());
}

/**
 *  @brief A committed erase must leave nothing behind that @c vacuum can still find twice.
 *
 *  The tombstone a transactional erase publishes answers every read as absence, so the only way to
 *  observe it is to sweep for it: the first sweep reclaims one entry per erased key, the second finds none.
 */
template <typename container_type_>
void test_committed_tombstones_are_reclaimable(std::size_t size = 32) {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    for (std::size_t identifier = 0; identifier < size; ++identifier)
        st_verify_(container.insert(trivial_id_to_member<member_t>(identifier)));
    for (std::size_t identifier = 0; identifier < size; ++identifier) commit_erase_of(container, identifier);

    st_verify_eq_(container.size(), 0u);
    auto const reclaimed = container.vacuum();
    st_verify_(reclaimed.has_value());
    st_verify_eq_(*reclaimed, size);

    auto const swept_again = container.vacuum();
    st_verify_(swept_again.has_value());
    st_verify_eq_(*swept_again, 0u);
    st_verify_eq_(container.size(), 0u);
}

/** @brief A windowed sweep touches the tombstones inside its bounds and no others. */
template <typename container_type_>
void test_vacuum_respects_its_window(std::size_t size = 32) {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    for (std::size_t identifier = 0; identifier < size; ++identifier)
        st_verify_(container.insert(trivial_id_to_member<member_t>(identifier)));
    for (std::size_t identifier = 0; identifier < size; ++identifier) commit_erase_of(container, identifier);

    std::size_t const midpoint = size / 2;
    auto const first_half = container.vacuum(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(midpoint));
    st_verify_(first_half.has_value());
    st_verify_eq_(*first_half, midpoint);

    auto const rest = container.vacuum();
    st_verify_(rest.has_value());
    st_verify_eq_(*rest, size - midpoint);
}

/** @brief Erasing a key whose tombstone is committed reports absence and reclaims the tombstone. */
template <typename container_type_>
void test_erase_reports_and_reclaims_tombstone() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    st_verify_(container.insert(trivial_id_to_member<member_t>(7)));
    commit_erase_of(container, 7);

    bool reported_missing = false;
    auto const status = container.erase(
        trivial_id_to_key<member_t>(7), [](member_t const &) noexcept {}, [&]() noexcept { reported_missing = true; });
    st_verify_(reported_missing);
    st_verify_eq_(status, status_t::key_not_found_k);

    // Nothing is left for a sweep to find, and an absent key erases the same way.
    auto const reclaimed = container.vacuum();
    st_verify_(reclaimed.has_value());
    st_verify_eq_(*reclaimed, 0u);
    st_verify_eq_(container.erase(trivial_id_to_key<member_t>(9)), status_t::key_not_found_k);
}

/** @brief A range erase reports the elements a reader could see, never a committed tombstone. */
template <typename container_type_>
void test_erase_range_skips_tombstones() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    st_verify_(container.insert(trivial_id_to_member<member_t>(1)));
    st_verify_(container.insert(trivial_id_to_member<member_t>(2)));
    commit_erase_of(container, 1);

    std::size_t reported = 0;
    st_verify_(container.erase_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(10),
                                     [&](member_t const &) noexcept { ++reported; }));
    st_verify_eq_(reported, 1u);
    st_verify_eq_(container.size(), 0u);
}

/** @brief A sweep must not touch what a running transaction has staged. */
template <typename container_type_>
void test_vacuum_leaves_staged_entries() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    st_verify_(container.insert(trivial_id_to_member<member_t>(3)));
    commit_erase_of(container, 3);

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(3)));
    st_verify_(transaction->stage());

    auto const reclaimed = container.vacuum();
    st_verify_(reclaimed.has_value());
    st_verify_eq_(*reclaimed, 1u);

    st_verify_(transaction->commit());
    st_verify_(container.contains(trivial_id_to_key<member_t>(3)));
    st_verify_eq_(container.size(), 1u);
}

#pragma endregion Tombstone Reclamation

#pragma region Commit Reporting

/** @brief A commit whose staged versions were wiped out from under it must say so. */
template <typename container_type_>
void test_commit_reports_lost_versions() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(5)));
    st_verify_(transaction->stage());

    st_verify_(container.clear());
    st_verify_eq_(transaction->commit(), status_t::consistency_k);
    st_verify_eq_(container.size(), 0u);
}

/** @brief A commit that publishes what it staged still reports plain success. */
template <typename container_type_>
void test_commit_reports_success_when_published(std::size_t size = 8) {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    for (std::size_t identifier = 0; identifier < size; ++identifier)
        st_verify_(transaction->upsert(trivial_id_to_member<member_t>(identifier)));

    // Touching one key twice lists it twice, and the second pass must not read as a loss.
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(0)));
    st_verify_(transaction->stage());
    st_verify_eq_(transaction->commit(), status_t::success_k);
    st_verify_eq_(container.size(), size);
}

#pragma endregion Commit Reporting

#pragma region Generation Stamping

/**
 *  @brief Every entry a transaction stages carries that transaction's generation.
 *
 *  The generation is half of the set's key, so a staged entry that kept the default stamp would
 *  order as a different element than the one the commit later looks for - and would collide with the
 *  generation that stands for a key which is not there. Rewriting a key inside one transaction,
 *  rolling back and staging again all have to preserve it.
 */
template <typename container_type_>
void test_staged_entries_keep_one_generation() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_ne_(transaction->generation(), absent_generation_k);

    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(1)));
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(1)));
    st_verify_(transaction->erase(trivial_id_to_key<member_t>(1)));
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(1)));
    st_verify_eq_(transaction->changes_count(), 1u);

    st_verify_(transaction->stage());
    st_verify_(transaction->rollback());
    st_verify_ne_(transaction->generation(), absent_generation_k);
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(2)));
    st_verify_(transaction->stage());
    st_verify_eq_(transaction->commit(), status_t::success_k);
    st_verify_eq_(container.size(), 2u);
}

/**
 *  @brief Clearing the store must not hand a new transaction a stamp an open one already carries.
 *
 *  A generation is half of the entry key, so two live transactions sharing one cannot stage the same
 *  key twice: the second merge finds the slot taken and silently keeps its own version, and the later
 *  commit publishes the earlier transaction's value under the later one's name.
 */
template <typename container_type_>
void test_clear_keeps_generations_running() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    auto early = container.transaction();
    st_verify_(early.has_value());
    st_verify_(container.clear());
    auto late = container.transaction();
    st_verify_(late.has_value());
    st_verify_ne_(early->generation(), late->generation());

    st_verify_(late->upsert(trivial_id_to_member<member_t>(5, 500)));
    st_verify_(late->stage());
    st_verify_eq_(late->commit(), status_t::success_k);

    st_verify_(early->upsert(trivial_id_to_member<member_t>(5, 100)));
    st_verify_(early->stage());
    st_verify_eq_(early->commit(), status_t::success_k);

    auto const stored = container.find_copy(trivial_id_to_key<member_t>(5));
    st_verify_(stored.has_value());
    st_verify_eq_(stored->mapped, 100);
}

#pragma endregion Generation Stamping

#pragma region Insert Reporting

/** @brief A refused insert names the collision, at the store and inside a transaction alike. */
template <typename container_type_>
void test_insert_refuses_with_key_already_exists() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    st_verify_(container.insert(trivial_id_to_member<member_t>(1)));
    st_verify_eq_(container.insert(trivial_id_to_member<member_t>(1)), status_t::key_already_exists_k);

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_eq_(transaction->insert(trivial_id_to_member<member_t>(1)), status_t::key_already_exists_k);
    st_verify_(transaction->insert(trivial_id_to_member<member_t>(2)));
    st_verify_eq_(transaction->insert(trivial_id_to_member<member_t>(2)), status_t::key_already_exists_k);
}

/** @brief Every insert path hands its callbacks the element, so one lambda fits every backend. */
template <typename container_type_>
void test_insert_reports_the_stored_element() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    auto expected_key = trivial_id_to_key<member_t>(3);
    std::size_t stored_reports = 0;
    std::size_t existing_reports = 0;
    auto note_stored = [&](member_t const &stored) noexcept {
        stored_reports += mapping_key_or_itself(stored) == expected_key;
    };
    auto note_existing = [&](member_t const &existing) noexcept {
        existing_reports += mapping_key_or_itself(existing) == expected_key;
    };

    st_verify_(container.insert(trivial_id_to_member<member_t>(3), note_stored, note_existing));
    st_verify_eq_(stored_reports, 1u);
    st_verify_eq_(container.insert(trivial_id_to_member<member_t>(3), note_stored, note_existing),
                  status_t::key_already_exists_k);
    st_verify_eq_(existing_reports, 1u);

    expected_key = trivial_id_to_key<member_t>(4);
    st_verify_(container.insert_if_missing(trivial_id_to_member<member_t>(4), note_stored, note_existing));
    st_verify_eq_(stored_reports, 2u);
    st_verify_(container.insert_if_missing(trivial_id_to_member<member_t>(4), note_stored, note_existing));
    st_verify_eq_(existing_reports, 2u);

    expected_key = trivial_id_to_key<member_t>(5);
    st_verify_(container.insert_or_assign(trivial_id_to_member<member_t>(5), note_stored, note_existing));
    st_verify_eq_(stored_reports, 3u);
    st_verify_(container.insert_or_assign(trivial_id_to_member<member_t>(5), note_stored, note_existing));
    st_verify_eq_(existing_reports, 3u);

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    expected_key = trivial_id_to_key<member_t>(6);
    st_verify_(transaction->insert(trivial_id_to_member<member_t>(6), note_stored, note_existing));
    st_verify_eq_(stored_reports, 4u);
    st_verify_eq_(transaction->insert(trivial_id_to_member<member_t>(6), note_stored, note_existing),
                  status_t::key_already_exists_k);
    st_verify_eq_(existing_reports, 4u);

    expected_key = trivial_id_to_key<member_t>(7);
    st_verify_(
        succeeded(transaction->insert_if_missing(trivial_id_to_member<member_t>(7), note_stored, note_existing)));
    st_verify_eq_(stored_reports, 5u);
    st_verify_(
        succeeded(transaction->insert_if_missing(trivial_id_to_member<member_t>(7), note_stored, note_existing)));
    st_verify_eq_(existing_reports, 5u);
}

/** @brief A key erased inside a transaction is gone for that transaction, so an insert may take it back. */
template <typename container_type_>
void test_insert_after_local_erase_succeeds() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    st_verify_(container.insert(trivial_id_to_member<member_t>(7, 700)));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(transaction->erase(trivial_id_to_key<member_t>(7)));
    st_verify_(transaction->insert(trivial_id_to_member<member_t>(7, 777)));
    st_verify_(transaction->stage());
    st_verify_eq_(transaction->commit(), status_t::success_k);

    auto const stored = container.find_copy(trivial_id_to_key<member_t>(7));
    st_verify_(stored.has_value());
    st_verify_eq_(stored->mapped, 777);
}

#pragma endregion Insert Reporting

#pragma region Staging Discipline

/** @brief Staging is a one-way step, so a second call is refused rather than merged again. */
template <typename container_type_>
void test_stage_refuses_when_already_staged() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(1)));
    st_verify_(transaction->stage());
    st_verify_eq_(transaction->stage(), status_t::operation_not_permitted_k);
    st_verify_eq_(transaction->commit(), status_t::success_k);
    st_verify_eq_(container.size(), 1u);
}

/** @brief A rollback that cannot recover what it staged reports the loss instead of dropping it. */
template <typename container_type_>
void test_rollback_reports_lost_versions() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(5)));
    st_verify_(transaction->stage());

    st_verify_(container.clear());
    st_verify_eq_(transaction->rollback(), status_t::consistency_k);
}

#pragma endregion Staging Discipline

#pragma region Ordered Walks

/**
 *  @brief An ordered walk steps over a store entry the transaction erased, and terminates.
 *
 *  The restart that a locally erased entry forces has to ask for the @b next store entry. Asking for
 *  the same bound again returns the same entry, and the walk never ends.
 */
template <typename container_type_>
void test_transaction_bounds_skip_locally_erased() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    st_verify_(maybe_container.has_value());
    auto &container = *maybe_container;

    st_verify_(container.insert(trivial_id_to_member<member_t>(5, 50)));
    st_verify_(container.insert(trivial_id_to_member<member_t>(9, 90)));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(transaction->erase(trivial_id_to_key<member_t>(5)));
    st_verify_(transaction->upsert(trivial_id_to_member<member_t>(9, 99)));

    std::size_t found = 0;
    auto const ninth = trivial_id_to_key<member_t>(9);
    auto note = [&](member_t const &element) noexcept { found += mapping_key_or_itself(element) == ninth; };
    transaction->lower_bound(trivial_id_to_key<member_t>(0), note, []() noexcept {});
    st_verify_eq_(found, 1u);
    transaction->upper_bound(trivial_id_to_key<member_t>(0), note, []() noexcept {});
    st_verify_eq_(found, 2u);
}

#pragma endregion Ordered Walks

} // namespace ashvardanian::smashtable::scripts
