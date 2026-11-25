/**
 *  @file test/monotonic_store_defects.hpp
 *  @author Ash Vardanian
 *  @date August 17, 2026
 *  @brief Template test functions for the transactional store's staging window and tombstone
 *      lifecycle. Covers what a direct write may do to a staged version, what an ordered read may
 *      show of a committed erase, and what @c vacuum reclaims afterwards.
 */
#pragma once
#include <utility> // `std::move`

#include "fixtures.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Staging Window

/** A direct write lands on the published version and leaves the staged one for its own commit. The
 *  staged version wins at commit, since commit order decides and not generation order. */
template <typename container_type_>
void check_direct_write_spares_staged_version() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(7, 100)), success_k);
    expect_equal(transaction.stage(), success_k);

    // A staged version is invisible, so the store still reports an empty container.
    expect_equal(container.size(), 0);

    expect_equal(container.upsert(trivial_id_to_member<member_t>(7, 200)), success_k);
    expect_equal(container.size(), 1);
    expect_equal(container.contains(trivial_id_to_key<member_t>(7)), true);

    // The staged version survived the direct write and is what the commit publishes.
    expect_equal(transaction.commit(), success_k);
    expect_equal(container.size(), 1);
    auto [committed, committed_status] = container.find_copy(trivial_id_to_key<member_t>(7));
    if (!expect_equal(committed_status, success_k)) return;
    if constexpr (a_mapping<member_t>) expect_equal(committed.mapped, typename member_t::mapped_type(100));
}

/** A direct erase takes the published version only, so the staged one still commits. */
template <typename container_type_>
void check_direct_erase_spares_staged_version() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(3, 1)), success_k);

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(3, 42)), success_k);
    expect_equal(transaction.stage(), success_k);

    expect_equal(container.erase(trivial_id_to_key<member_t>(3)), success_k);
    expect_equal(container.size(), 0);

    expect_equal(transaction.commit(), success_k);
    expect_equal(container.size(), 1);
    auto [committed, committed_status] = container.find_copy(trivial_id_to_key<member_t>(3));
    if (!expect_equal(committed_status, success_k)) return;
    if constexpr (a_mapping<member_t>) expect_equal(committed.mapped, typename member_t::mapped_type(42));
}

/** A ranged erase takes published versions only, so an overlapping staged write still commits. */
template <typename container_type_>
void check_erase_range_spares_staged_versions() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 8; ++index)
        expect_equal(container.upsert(trivial_id_to_member<member_t>(index, index)), success_k);

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(4, 400)), success_k);
    expect_equal(transaction.stage(), success_k);

    expect_equal(container.erase_range(trivial_id_to_key<member_t>(2), trivial_id_to_key<member_t>(6)), success_k);
    expect_equal(container.size(), 4);

    expect_equal(transaction.commit(), success_k);
    expect_equal(container.size(), 5);
    auto [committed, committed_status] = container.find_copy(trivial_id_to_key<member_t>(4));
    if (!expect_equal(committed_status, success_k)) return;
    if constexpr (a_mapping<member_t>) expect_equal(committed.mapped, typename member_t::mapped_type(400));
}

/** Clearing keeps handing out fresh stamps, so a later transaction never reuses a live one. */
template <typename container_type_>
void check_clear_keeps_generations_moving() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);
    auto [before, before_status] = container.transaction();
    if (!expect_equal(before_status, success_k)) return;
    auto const stamp_before = before.generation();

    expect_equal(container.clear(), success_k);

    auto [after, after_status] = container.transaction();
    if (!expect_equal(after_status, success_k)) return;
    expect_true(after.generation() > stamp_before);
}

#pragma endregion Staging Window

#pragma region Tombstone Visibility

/** A committed erase is invisible to every point read, exactly as it is to @c find. */
template <typename container_type_>
void check_committed_erase_hidden_from_point_reads() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);
    expect_equal(container.upsert(trivial_id_to_member<member_t>(2, 2)), success_k);

    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(transaction.erase(trivial_id_to_key<member_t>(2)), success_k);
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.commit(), success_k);

    expect_equal(container.size(), 1);
    expect_equal(container.contains(trivial_id_to_key<member_t>(2)), false);
    expect_equal(container.count(trivial_id_to_key<member_t>(2)), std::size_t {0});
    expect_true(!container.find_copy(trivial_id_to_key<member_t>(2)).has_value());

    std::size_t matches = 0;
    expect_equal(container.equal_range(trivial_id_to_key<member_t>(2), [&](member_t const &) noexcept { ++matches; }),
                 success_k);
    expect_equal(matches, 0);
}

/** A committed erase is invisible to every ordered read, not only to @c find. */
template <typename container_type_>
void check_committed_erase_hidden_from_ordered_reads() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 4; ++index)
        expect_equal(container.upsert(trivial_id_to_member<member_t>(index, index)), success_k);

    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(transaction.erase(trivial_id_to_key<member_t>(2)), success_k);
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.commit(), success_k);
    expect_equal(container.size(), 3);

    std::size_t walked = 0;
    expect_equal(
        container.range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(4),
                        [&](member_t const &member) noexcept {
                            ++walked;
                            expect_true(trivial_id_to_key<member_t>(2) != mapping_key_or_itself<member_t>(member));
                        }),
        success_k);
    expect_equal(walked, 3);

    // The tombstone sits exactly on the sought key, so a bound that ignored it would return it.
    expect_equal(container.lower_bound(
                     trivial_id_to_key<member_t>(2),
                     [&](member_t const &member) noexcept {
                         expect_true(trivial_id_to_key<member_t>(3) == mapping_key_or_itself<member_t>(member));
                     },
                     []() noexcept { expect_true(false, "A live key sits above the tombstone"); }),
                 success_k);
    expect_equal(container.upper_bound(
                     trivial_id_to_key<member_t>(1),
                     [&](member_t const &member) noexcept {
                         expect_true(trivial_id_to_key<member_t>(3) == mapping_key_or_itself<member_t>(member));
                     },
                     []() noexcept { expect_true(false, "A live key sits above the tombstone"); }),
                 success_k);

    // Sampling shares the range surface, so it must not draw the tombstone either.
    split_mix_generator_t generator {case_stream_key()};
    for (std::size_t attempt = 0; attempt != 64; ++attempt)
        expect_equal(container.sample_one(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(4), generator,
                                          [&](member_t const &member) noexcept {
                                              expect_true(trivial_id_to_key<member_t>(2) !=
                                                          mapping_key_or_itself<member_t>(member));
                                          }),
                     success_k);
}

/** A mutating ranged update never hands a caller the payload of a tombstone. */
template <typename container_type_>
void check_committed_erase_hidden_from_update_range() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 4; ++index)
        expect_equal(container.upsert(trivial_id_to_member<member_t>(index, index)), success_k);

    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(transaction.erase(trivial_id_to_key<member_t>(2)), success_k);
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.commit(), success_k);

    std::size_t visited = 0;
    expect_equal(container.update_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(4),
                                        [&](auto const &key, auto &mapped) noexcept {
                                            ++visited;
                                            expect_true(trivial_id_to_key<member_t>(2) != key);
                                            mapped = typename member_t::mapped_type(9);
                                        }),
                 success_k);
    expect_equal(visited, 3);
    expect_equal(container.contains(trivial_id_to_key<member_t>(2)), false);
}

#pragma endregion Tombstone Visibility

#pragma region Vacuuming

/** A committed tombstone is reclaimable, and reclaiming it changes nothing observable. */
template <typename container_type_>
void check_vacuum_reclaims_committed_tombstones() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 6; ++index)
        expect_equal(container.upsert(trivial_id_to_member<member_t>(index, index)), success_k);

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;
    expect_equal(transaction.erase(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(transaction.erase(trivial_id_to_key<member_t>(4)), success_k);
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.commit(), success_k);
    expect_equal(container.size(), 4);

    auto const [reclaimed, reclaimed_status] = container.vacuum();
    if (!expect_equal(reclaimed_status, success_k)) return;
    expect_equal(reclaimed, 2);
    expect_equal(container.size(), 4);
    expect_equal(container.contains(trivial_id_to_key<member_t>(1)), false);
    expect_equal(container.contains(trivial_id_to_key<member_t>(0)), true);

    // Nothing is left to reclaim, and a live key is never mistaken for a tombstone.
    auto const [second_pass, second_pass_status] = container.vacuum();
    if (!expect_equal(second_pass_status, success_k)) return;
    expect_equal(second_pass, 0);
    expect_equal(container.size(), 4);
}

/** A staged version behind a tombstone keeps its entry, since a commit may still publish it. */
template <typename container_type_>
void check_vacuum_spares_staged_versions() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(5, 5)), success_k);

    auto [erasing, erasing_status] = container.transaction();
    if (!expect_equal(erasing_status, success_k)) return;
    expect_equal(erasing.erase(trivial_id_to_key<member_t>(5)), success_k);
    expect_equal(erasing.stage(), success_k);
    expect_equal(erasing.commit(), success_k);

    auto [rewriting, rewriting_status] = container.transaction();
    if (!expect_equal(rewriting_status, success_k)) return;
    expect_equal(rewriting.upsert(trivial_id_to_member<member_t>(5, 55)), success_k);
    expect_equal(rewriting.stage(), success_k);

    auto const [reclaimed, reclaimed_status] = container.vacuum();
    if (!expect_equal(reclaimed_status, success_k)) return;
    expect_equal(reclaimed, 0);

    expect_equal(rewriting.commit(), success_k);
    expect_equal(container.size(), 1);
    expect_equal(container.contains(trivial_id_to_key<member_t>(5)), true);
}

/** The windowed overload reclaims only its own slice, so a caller can step through the keyspace. */
template <typename container_type_>
void check_windowed_vacuum_reclaims_one_slice() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 8; ++index)
        expect_equal(container.upsert(trivial_id_to_member<member_t>(index, index)), success_k);

    auto [transaction, transaction_status] = container.transaction();
    if (!expect_equal(transaction_status, success_k)) return;
    expect_equal(transaction.erase(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(transaction.erase(trivial_id_to_key<member_t>(6)), success_k);
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.commit(), success_k);

    auto const [lower_slice, lower_slice_status] =
        container.vacuum(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(4));
    if (!expect_equal(lower_slice_status, success_k)) return;
    expect_equal(lower_slice, 1);

    auto const [upper_slice, upper_slice_status] =
        container.vacuum(trivial_id_to_key<member_t>(4), trivial_id_to_key<member_t>(8));
    if (!expect_equal(upper_slice_status, success_k)) return;
    expect_equal(upper_slice, 1);

    auto const [exhausted, exhausted_status] = container.vacuum();
    if (!expect_equal(exhausted_status, success_k)) return;
    expect_equal(exhausted, 0);
    expect_equal(container.size(), 6);
}

#pragma endregion Vacuuming

#pragma region Transaction Reads

/** A transaction's own range is one sorted sequence, however its staged keys interleave with the
 *  committed ones, and a staged tombstone hides the committed key underneath it. */
template <typename container_type_>
void check_transaction_range_interleaves_staged_and_committed() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index : {0u, 2u, 4u, 6u})
        expect_equal(container.upsert(trivial_id_to_member<member_t>(index, index)), success_k);

    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    for (std::size_t index : {1u, 3u, 5u})
        expect_equal(transaction.upsert(trivial_id_to_member<member_t>(index, index)), success_k);
    expect_equal(transaction.erase(trivial_id_to_key<member_t>(4)), success_k);

    std::size_t const expected_ids[] = {0, 1, 2, 3, 5, 6};
    std::size_t walked = 0;
    expect_equal(transaction.range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(8),
                                   [&](member_t const &member) noexcept {
                                       expect_true(walked < 6);
                                       expect_true(trivial_id_to_key<member_t>(expected_ids[walked]) ==
                                                   mapping_key_or_itself<member_t>(member));
                                       ++walked;
                                   }),
                 success_k);
    expect_equal(walked, 6);

    // A half-open window sees the same order, and the bound still excludes its upper key.
    walked = 0;
    expect_equal(transaction.range(trivial_id_to_key<member_t>(1), trivial_id_to_key<member_t>(5),
                                   [&](member_t const &member) noexcept {
                                       expect_true(walked < 3);
                                       expect_true(trivial_id_to_key<member_t>(expected_ids[walked + 1]) ==
                                                   mapping_key_or_itself<member_t>(member));
                                       ++walked;
                                   }),
                 success_k);
    expect_equal(walked, 3);
}

/** A transaction's @c equal_range answers from its own staged writes as well as the store. */
template <typename container_type_>
void check_transaction_equal_range_sees_staged_writes() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);

    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(2, 22)), success_k);
    expect_equal(transaction.erase(trivial_id_to_key<member_t>(1)), success_k);

    std::size_t staged_matches = 0;
    expect_equal(transaction.equal_range(trivial_id_to_key<member_t>(2),
                                         [&](member_t const &member) noexcept {
                                             ++staged_matches;
                                             if constexpr (a_mapping<member_t>)
                                                 expect_equal(member.mapped, typename member_t::mapped_type(22));
                                         }),
                 success_k);
    expect_equal(staged_matches, 1);

    std::size_t erased_matches = 0;
    expect_equal(
        transaction.equal_range(trivial_id_to_key<member_t>(1), [&](member_t const &) noexcept { ++erased_matches; }),
        success_k);
    expect_equal(erased_matches, 0);

    std::size_t absent_matches = 0;
    expect_equal(
        transaction.equal_range(trivial_id_to_key<member_t>(9), [&](member_t const &) noexcept { ++absent_matches; }),
        success_k);
    expect_equal(absent_matches, 0);
}

#pragma endregion Transaction Reads

#pragma region Status Reporting

/** A key collision reads the same at both levels, and from the bulk entry point too. */
template <typename container_type_>
void check_insert_reports_key_already_exists() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);

    expect_equal(container.insert(trivial_id_to_member<member_t>(1, 9)), status_t::key_already_exists_k);

    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(transaction.insert(trivial_id_to_member<member_t>(1, 9)), status_t::key_already_exists_k);

    if constexpr (std::is_copy_constructible_v<member_t>) {
        member_t const batch[] = {trivial_id_to_member<member_t>(1, 9), trivial_id_to_member<member_t>(2, 2)};
        expect_equal(container.insert(batch, batch + 2), status_t::key_already_exists_k);
    }
}

/** A lookup that found nothing says so, rather than reporting a reason it never established. */
template <typename container_type_>
void check_find_copy_reports_key_not_found() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);

    expect_equal(container.find_copy(trivial_id_to_key<member_t>(2)).status(), status_t::key_not_found_k);

    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(transaction.find_copy(trivial_id_to_key<member_t>(2)).status(), status_t::key_not_found_k);
}

/** Staging twice is refused, so a second pass cannot file the same keys again. */
template <typename container_type_>
void check_second_stage_is_rejected() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto [transaction, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(transaction.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);
    expect_equal(transaction.stage(), success_k);
    expect_equal(transaction.stage(), status_t::operation_not_permitted_k);

    expect_equal(transaction.commit(), success_k);
    expect_equal(container.size(), 1);
    expect_equal(container.contains(trivial_id_to_key<member_t>(1)), true);
}

/**
 *  @brief A refused validation writes nothing, so a caller spanning several stores can turn back.
 *
 *  The half a commit can refuse and the half that applies are separate calls, because a wrapper
 *  committing across partitions has to ask every one of them before any of them writes. Asking
 *  after the first has written is how a refusal ends up reported over writes readers already see.
 */
template <typename container_type_>
void test_validate_refuses_before_publishing() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    expect_equal(container.upsert(trivial_id_to_member<member_t>(1, 100)), success_k);

    auto [writer, writer_status] = container.transaction();
    if (!expect_equal(writer_status, success_k)) return;
    expect_equal(writer.watch(trivial_id_to_key<member_t>(1)), success_k);
    expect_equal(writer.upsert(trivial_id_to_member<member_t>(2, 200)), success_k);
    expect_equal(writer.stage(), success_k);

    // Somebody moves the watched key while this transaction sits staged.
    auto [outsider, outsider_status] = container.transaction();
    if (!expect_equal(outsider_status, success_k)) return;
    expect_equal(outsider.upsert(trivial_id_to_member<member_t>(1, 999)), success_k);
    expect_equal(outsider.stage(), success_k);
    expect_equal(outsider.commit(), success_k);

    expect_equal(writer.validate_for_commit(), status_t::read_conflict_k);
    expect_equal(container.contains(trivial_id_to_key<member_t>(2)), false);
}

/**
 *  @brief Publishing after a validation that passed shows every staged key, and cannot refuse.
 *
 *  Every path that removes an entry leaves a staged version alone - an erase retires only the
 *  published one, reclamation passes over a chain still carrying one, and another transaction's
 *  unmasking walks only visible versions. This drives all three under a staged transaction and then
 *  publishes, so the second phase is exercised against the cases that could have emptied it.
 */
template <typename container_type_>
void test_publish_cannot_refuse() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    for (trivial_id_t identifier = 1; identifier != 5; ++identifier)
        expect_equal(container.upsert(trivial_id_to_member<member_t>(identifier, identifier)), success_k);

    auto [writer, writer_status] = container.transaction();
    if (!expect_equal(writer_status, success_k)) return;
    for (trivial_id_t identifier = 1; identifier != 5; ++identifier)
        expect_equal(writer.upsert(trivial_id_to_member<member_t>(identifier, identifier * 10)), success_k);
    expect_equal(writer.stage(), success_k);

    // Another transaction publishes over two of the same keys.
    auto [peer, peer_status] = container.transaction();
    if (!expect_equal(peer_status, success_k)) return;
    expect_equal(peer.upsert(trivial_id_to_member<member_t>(1, 7)), success_k);
    expect_equal(peer.upsert(trivial_id_to_member<member_t>(2, 7)), success_k);
    expect_equal(peer.stage(), success_k);
    expect_equal(peer.commit(), success_k);

    // A direct erase retires the published version of a third, and reclamation sweeps behind it.
    [[maybe_unused]] status_t const erased = container.erase(trivial_id_to_key<member_t>(3));
    [[maybe_unused]] status_t const reclaimed = container.vacuum().status();

    expect_equal(writer.validate_for_commit(), success_k);
    writer.publish_under();

    for (trivial_id_t identifier = 1; identifier != 5; ++identifier)
        expect_equal(container.contains(trivial_id_to_key<member_t>(identifier)), true);
}

/** Emptying the store is refused while a transaction still has something staged to publish. */
template <typename container_type_>
void test_clear_refuses_while_staged() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    container_t container;
    auto [writer, status] = container.transaction();
    if (!expect_equal(status, success_k)) return;
    expect_equal(writer.upsert(trivial_id_to_member<member_t>(5, 500)), success_k);
    expect_equal(writer.stage(), success_k);

    expect_equal(container.clear(), status_t::operation_not_permitted_k);

    expect_equal(writer.rollback(), success_k);
    expect_equal(container.clear(), success_k);
}

#pragma endregion Status Reporting

} // namespace ashvardanian::smashtable::test
