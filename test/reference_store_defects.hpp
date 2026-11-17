/**
 *  @file test/reference_store_defects.hpp
 *  @author Ash Vardanian
 *  @date August 17, 2026
 *  @brief Regressions for the reclamation and reporting defects of the @c std::set-backed oracle.
 *      Templated on the container so the chain-backed store can be held to the same behaviour.
 */
#pragma once
#include "harness.hpp"
#include "fixtures.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Tombstone Reclamation

/** Commits an erase of @p identifier through a fresh transaction. */
template <typename container_type_>
void commit_erase_of(container_type_ &container, trivial_id_t identifier) {
    using member_t = typename container_type_::value_type;
    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expect_equal(transaction->erase(trivial_id_to_key<member_t>(identifier)), success_k);
    expect_equal(transaction->stage(), success_k);
    expect_equal(transaction->commit(), success_k);
}

/**
 *  @brief A committed erase must leave nothing behind that @c vacuum can still find twice.
 *
 *  The tombstone a transactional erase publishes answers every read as absence, so only a sweep
 *  observes it: the first sweep reclaims one entry per erased key, and the second finds none.
 */
template <typename container_type_>
void check_committed_tombstones_are_reclaimable(std::size_t size = 32) {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    for (std::size_t identifier = 0; identifier < size; ++identifier)
        expect_equal(container.insert(trivial_id_to_member<member_t>(identifier)), success_k);
    for (std::size_t identifier = 0; identifier < size; ++identifier) commit_erase_of(container, identifier);

    expect_equal(container.size(), 0u);
    auto const reclaimed = container.vacuum();
    if (!expect_true(reclaimed.has_value())) return;
    expect_equal(*reclaimed, size);

    auto const swept_again = container.vacuum();
    if (!expect_true(swept_again.has_value())) return;
    expect_equal(*swept_again, 0u);
    expect_equal(container.size(), 0u);
}

/** A windowed sweep touches the tombstones inside its bounds and no others. */
template <typename container_type_>
void check_vacuum_respects_its_window(std::size_t size = 32) {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    for (std::size_t identifier = 0; identifier < size; ++identifier)
        expect_equal(container.insert(trivial_id_to_member<member_t>(identifier)), success_k);
    for (std::size_t identifier = 0; identifier < size; ++identifier) commit_erase_of(container, identifier);

    std::size_t const midpoint = size / 2;
    auto const first_half = container.vacuum(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(midpoint));
    if (!expect_true(first_half.has_value())) return;
    expect_equal(*first_half, midpoint);

    auto const rest = container.vacuum();
    if (!expect_true(rest.has_value())) return;
    expect_equal(*rest, size - midpoint);
}

/** Erasing a key whose tombstone is committed reports absence and reclaims the tombstone. */
template <typename container_type_>
void check_erase_reports_and_reclaims_tombstone() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    expect_equal(container.insert(trivial_id_to_member<member_t>(7)), success_k);
    commit_erase_of(container, 7);

    bool reported_missing = false;
    auto const status = container.erase(
        trivial_id_to_key<member_t>(7), [](member_t const &) noexcept {}, [&]() noexcept { reported_missing = true; });
    expect_true(reported_missing);
    expect_equal(status, status_t::key_not_found_k);

    // Nothing is left for a sweep to find, and an absent key erases the same way.
    auto const reclaimed = container.vacuum();
    if (!expect_true(reclaimed.has_value())) return;
    expect_equal(*reclaimed, 0u);
    expect_equal(container.erase(trivial_id_to_key<member_t>(9)), status_t::key_not_found_k);
}

/** A range erase reports the elements a reader could see, never a committed tombstone. */
template <typename container_type_>
void check_erase_range_skips_tombstones() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    expect_equal(container.insert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(container.insert(trivial_id_to_member<member_t>(2)), success_k);
    commit_erase_of(container, 1);

    std::size_t reported = 0;
    expect_equal(container.erase_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(10),
                                       [&](member_t const &) noexcept { ++reported; }),
                 success_k);
    expect_equal(reported, 1u);
    expect_equal(container.size(), 0u);
}

/** A sweep must not touch what a running transaction has staged. */
template <typename container_type_>
void check_vacuum_leaves_staged_entries() {
    using member_t = typename container_type_::value_type;

    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    expect_equal(container.insert(trivial_id_to_member<member_t>(3)), success_k);
    commit_erase_of(container, 3);

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(3)), success_k);
    expect_equal(transaction->stage(), success_k);

    auto const reclaimed = container.vacuum();
    if (!expect_true(reclaimed.has_value())) return;
    expect_equal(*reclaimed, 1u);

    expect_equal(transaction->commit(), success_k);
    expect_equal(container.contains(trivial_id_to_key<member_t>(3)), true);
    expect_equal(container.size(), 1u);
}

#pragma endregion Tombstone Reclamation

#pragma region Commit Reporting

/**
 *  @brief Staged versions cannot be wiped out from under the transaction about to publish them.
 *
 *  Dropping one would make publishing fallible, and a publish that can refuse is how a commit
 *  spanning several stores ends up reporting a refusal over writes an earlier store already made
 *  visible. Emptying is the only path that could reach a staged version, so it is refused while any
 *  version is staged.
 */
template <typename container_type_>
void check_staged_versions_survive_a_clear() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(5)), success_k);
    expect_equal(transaction->stage(), success_k);

    expect_equal(container.clear(), status_t::operation_not_permitted_k);
    expect_equal(transaction->commit(), success_k);
    expect_equal(container.size(), 1u);
    expect_equal(container.clear(), success_k);
}

/** A commit that publishes what it staged still reports plain success. */
template <typename container_type_>
void check_commit_reports_success_when_published(std::size_t size = 8) {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    for (std::size_t identifier = 0; identifier < size; ++identifier)
        expect_equal(transaction->upsert(trivial_id_to_member<member_t>(identifier)), success_k);

    // Touching one key twice lists it twice, and the second pass must not read as a loss.
    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(0)), success_k);
    expect_equal(transaction->stage(), success_k);
    expect_equal(transaction->commit(), status_t::success_k);
    expect_equal(container.size(), size);
}

#pragma endregion Commit Reporting

#pragma region Generation Stamping

/**
 *  @brief Every entry a transaction stages carries that transaction's generation.
 *
 *  The generation is half of the set's key, so a staged entry that kept the default stamp would
 *  order as a different element than the one the commit later looks for - and would collide with
 *  the generation that stands for a key which is not there. Rewriting a key inside one transaction,
 *  rolling back and staging again all have to preserve it.
 */
template <typename container_type_>
void check_staged_entries_keep_one_generation() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expect_true(transaction->generation() != absent_generation_k);

    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(transaction->erase(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(transaction->changes_count(), 1u);

    expect_equal(transaction->stage(), success_k);
    expect_equal(transaction->rollback(), success_k);
    expect_true(transaction->generation() != absent_generation_k);
    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(2)), success_k);
    expect_equal(transaction->stage(), success_k);
    expect_equal(transaction->commit(), status_t::success_k);
    expect_equal(container.size(), 2u);
}

/**
 *  @brief Clearing the store must not hand a new transaction a stamp an open one already carries.
 *
 *  A generation is half of the entry key, so two live transactions sharing one cannot stage the
 *  same key twice: the second merge finds the slot taken and silently keeps its own version, and
 *  the later commit publishes the earlier transaction's value under the later one's name.
 */
template <typename container_type_>
void check_clear_keeps_generations_running() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    auto early = container.transaction();
    if (!expect_true(early.has_value())) return;
    expect_equal(container.clear(), success_k);
    auto late = container.transaction();
    if (!expect_true(late.has_value())) return;
    expect_true(early->generation() != late->generation());

    expect_equal(late->upsert(trivial_id_to_member<member_t>(5, 500)), success_k);
    expect_equal(late->stage(), success_k);
    expect_equal(late->commit(), status_t::success_k);

    expect_equal(early->upsert(trivial_id_to_member<member_t>(5, 100)), success_k);
    expect_equal(early->stage(), success_k);
    expect_equal(early->commit(), status_t::success_k);

    auto const stored = container.find_copy(trivial_id_to_key<member_t>(5));
    if (!expect_true(stored.has_value())) return;
    expect_equal(stored->mapped, 100);
}

#pragma endregion Generation Stamping

#pragma region Insert Reporting

/** A refused insert names the collision, at the store and inside a transaction alike. */
template <typename container_type_>
void check_insert_refuses_with_key_already_exists() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    expect_equal(container.insert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(container.insert(trivial_id_to_member<member_t>(1)), status_t::key_already_exists_k);

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expect_equal(transaction->insert(trivial_id_to_member<member_t>(1)), status_t::key_already_exists_k);
    expect_equal(transaction->insert(trivial_id_to_member<member_t>(2)), success_k);
    expect_equal(transaction->insert(trivial_id_to_member<member_t>(2)), status_t::key_already_exists_k);
}

/** Every insert path hands its callbacks the element, so one lambda fits every backend. */
template <typename container_type_>
void check_insert_reports_the_stored_element() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
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

    expect_equal(container.insert(trivial_id_to_member<member_t>(3), note_stored, note_existing), success_k);
    expect_equal(stored_reports, 1u);
    expect_equal(container.insert(trivial_id_to_member<member_t>(3), note_stored, note_existing),
                 status_t::key_already_exists_k);
    expect_equal(existing_reports, 1u);

    expected_key = trivial_id_to_key<member_t>(4);
    expect_equal(container.insert_if_missing(trivial_id_to_member<member_t>(4), note_stored, note_existing), success_k);
    expect_equal(stored_reports, 2u);
    expect_equal(container.insert_if_missing(trivial_id_to_member<member_t>(4), note_stored, note_existing), success_k);
    expect_equal(existing_reports, 2u);

    expected_key = trivial_id_to_key<member_t>(5);
    expect_equal(container.insert_or_assign(trivial_id_to_member<member_t>(5), note_stored, note_existing), success_k);
    expect_equal(stored_reports, 3u);
    expect_equal(container.insert_or_assign(trivial_id_to_member<member_t>(5), note_stored, note_existing), success_k);
    expect_equal(existing_reports, 3u);

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expected_key = trivial_id_to_key<member_t>(6);
    expect_equal(transaction->insert(trivial_id_to_member<member_t>(6), note_stored, note_existing), success_k);
    expect_equal(stored_reports, 4u);
    expect_equal(transaction->insert(trivial_id_to_member<member_t>(6), note_stored, note_existing),
                 status_t::key_already_exists_k);
    expect_equal(existing_reports, 4u);

    expected_key = trivial_id_to_key<member_t>(7);
    expect_equal(transaction->insert_if_missing(trivial_id_to_member<member_t>(7), note_stored, note_existing),
                 success_k);
    expect_equal(stored_reports, 5u);
    expect_equal(transaction->insert_if_missing(trivial_id_to_member<member_t>(7), note_stored, note_existing),
                 success_k);
    expect_equal(existing_reports, 5u);
}

/** A key erased in a transaction is gone for that transaction, so an insert can take it back. */
template <typename container_type_>
void check_insert_after_local_erase_succeeds() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    expect_equal(container.insert(trivial_id_to_member<member_t>(7, 700)), success_k);

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expect_equal(transaction->erase(trivial_id_to_key<member_t>(7)), success_k);
    expect_equal(transaction->insert(trivial_id_to_member<member_t>(7, 777)), success_k);
    expect_equal(transaction->stage(), success_k);
    expect_equal(transaction->commit(), status_t::success_k);

    auto const stored = container.find_copy(trivial_id_to_key<member_t>(7));
    if (!expect_true(stored.has_value())) return;
    expect_equal(stored->mapped, 777);
}

#pragma endregion Insert Reporting

#pragma region Staging Discipline

/** Staging is a one-way step, so a second call is refused rather than merged again. */
template <typename container_type_>
void check_stage_refuses_when_already_staged() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(1)), success_k);
    expect_equal(transaction->stage(), success_k);
    expect_equal(transaction->stage(), status_t::operation_not_permitted_k);
    expect_equal(transaction->commit(), status_t::success_k);
    expect_equal(container.size(), 1u);
}

/**
 *  @brief A rollback recovers what it staged, because nothing can take those versions away from it.
 *
 *  @note Emptying the store is refused while anything is staged, so it cannot provoke the loss that
 *      @c rollback reports with @c consistency_k. Nothing else in the suite provokes it either, so
 *      that path is uncovered - worth knowing rather than assuming otherwise.
 */
template <typename container_type_>
void check_rollback_recovers_what_it_staged() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(5)), success_k);
    expect_equal(transaction->stage(), success_k);

    expect_equal(container.clear(), status_t::operation_not_permitted_k);
    expect_equal(transaction->rollback(), success_k);
    expect_equal(container.size(), 0u);
    expect_equal(container.clear(), success_k);
}

#pragma endregion Staging Discipline

#pragma region Ordered Walks

/**
 *  @brief An ordered walk steps over a store entry the transaction erased, and terminates.
 *
 *  The restart that a locally erased entry forces has to ask for the @b next store entry. Asking
 *  for the same bound again returns the same entry, and the walk never ends.
 */
template <typename container_type_>
void check_transaction_bounds_skip_locally_erased() {
    using member_t = typename container_type_::value_type;
    auto maybe_container = container_type_::make();
    if (!expect_true(maybe_container.has_value())) return;
    auto &container = *maybe_container;

    expect_equal(container.insert(trivial_id_to_member<member_t>(5, 50)), success_k);
    expect_equal(container.insert(trivial_id_to_member<member_t>(9, 90)), success_k);

    auto transaction = container.transaction();
    if (!expect_true(transaction.has_value())) return;
    expect_equal(transaction->erase(trivial_id_to_key<member_t>(5)), success_k);
    expect_equal(transaction->upsert(trivial_id_to_member<member_t>(9, 99)), success_k);

    std::size_t found = 0;
    auto const ninth = trivial_id_to_key<member_t>(9);
    auto note = [&](member_t const &element) noexcept { found += mapping_key_or_itself(element) == ninth; };
    expect_equal(transaction->lower_bound(trivial_id_to_key<member_t>(0), note, []() noexcept {}), success_k);
    expect_equal(found, 1u);
    expect_equal(transaction->upper_bound(trivial_id_to_key<member_t>(0), note, []() noexcept {}), success_k);
    expect_equal(found, 2u);
}

#pragma endregion Ordered Walks

} // namespace ashvardanian::smashtable::test
