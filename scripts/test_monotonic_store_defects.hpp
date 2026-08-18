/**
 *  @brief Template test functions for the transactional store's staging window and tombstone lifecycle.
 *      Covers what a direct write may do to a staged version, what an ordered read may show of a
 *      committed erase, and what @c vacuum reclaims afterwards.
 *  @author Ash Vardanian
 *  @file scripts/test_monotonic_store_defects.hpp
 *  @date August 17, 2026
 */
#pragma once
#include <random>  // `std::mt19937`
#include <utility> // `std::move`

#include "test_basic.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Staging Window

/**
 *  @brief A direct write lands on the published version and leaves the staged one for its own commit.
 *    The staged version wins at commit, since commit order decides and not generation order.
 */
template <typename container_type_>
void test_direct_write_spares_staged_version() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->upsert(trivial_id_to_member<member_t>(7, 100))));
    st_verify_(succeeded(transaction->stage()));

    // A staged version is invisible, so the store still reports an empty container.
    st_verify_eq_(container.size(), 0);

    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(7, 200))));
    st_verify_eq_(container.size(), 1);
    st_verify_(container.contains(trivial_id_to_key<member_t>(7)));

    // The staged version survived the direct write and is what the commit publishes.
    st_verify_(succeeded(transaction->commit()));
    st_verify_eq_(container.size(), 1);
    auto committed = container.find_copy(trivial_id_to_key<member_t>(7));
    st_verify_(committed.has_value());
    if constexpr (is_mapping<member_t>) st_verify_eq_(committed->mapped, typename member_t::mapped_type(100));
}

/** @brief A direct erase takes the published version only, so the staged one still commits. */
template <typename container_type_>
void test_direct_erase_spares_staged_version() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(3, 1))));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->upsert(trivial_id_to_member<member_t>(3, 42))));
    st_verify_(succeeded(transaction->stage()));

    st_verify_(succeeded(container.erase(trivial_id_to_key<member_t>(3))));
    st_verify_eq_(container.size(), 0);

    st_verify_(succeeded(transaction->commit()));
    st_verify_eq_(container.size(), 1);
    auto committed = container.find_copy(trivial_id_to_key<member_t>(3));
    st_verify_(committed.has_value());
    if constexpr (is_mapping<member_t>) st_verify_eq_(committed->mapped, typename member_t::mapped_type(42));
}

/** @brief A ranged erase takes published versions only, so an overlapping staged write still commits. */
template <typename container_type_>
void test_erase_range_spares_staged_versions() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 8; ++index)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(index, index))));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->upsert(trivial_id_to_member<member_t>(4, 400))));
    st_verify_(succeeded(transaction->stage()));

    st_verify_(succeeded(container.erase_range(trivial_id_to_key<member_t>(2), trivial_id_to_key<member_t>(6))));
    st_verify_eq_(container.size(), 4);

    st_verify_(succeeded(transaction->commit()));
    st_verify_eq_(container.size(), 5);
    auto committed = container.find_copy(trivial_id_to_key<member_t>(4));
    st_verify_(committed.has_value());
    if constexpr (is_mapping<member_t>) st_verify_eq_(committed->mapped, typename member_t::mapped_type(400));
}

/** @brief Clearing keeps handing out fresh stamps, so a later transaction never reuses a live one. */
template <typename container_type_>
void test_clear_keeps_generations_moving() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(1, 1))));
    auto before = container.transaction();
    st_verify_(before.has_value());
    auto const stamp_before = before->generation();

    st_verify_(succeeded(container.clear()));

    auto after = container.transaction();
    st_verify_(after.has_value());
    st_verify_(after->generation() > stamp_before);
}

#pragma endregion Staging Window

#pragma region Tombstone Visibility

/** @brief A committed erase is invisible to every point read, exactly as it is to @c find. */
template <typename container_type_>
void test_committed_erase_hidden_from_point_reads() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(1, 1))));
    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(2, 2))));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->erase(trivial_id_to_key<member_t>(2))));
    st_verify_(succeeded(transaction->stage()));
    st_verify_(succeeded(transaction->commit()));

    st_verify_eq_(container.size(), 1);
    st_verify_(!container.contains(trivial_id_to_key<member_t>(2)));
    st_verify_eq_(container.count(trivial_id_to_key<member_t>(2)), 0);
    st_verify_(!container.find_copy(trivial_id_to_key<member_t>(2)).has_value());

    std::size_t matches = 0;
    container.equal_range(trivial_id_to_key<member_t>(2), [&](member_t const &) noexcept { ++matches; });
    st_verify_eq_(matches, 0);
}

/** @brief A committed erase is invisible to every ordered read, not only to @c find. */
template <typename container_type_>
void test_committed_erase_hidden_from_ordered_reads() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 4; ++index)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(index, index))));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->erase(trivial_id_to_key<member_t>(2))));
    st_verify_(succeeded(transaction->stage()));
    st_verify_(succeeded(transaction->commit()));
    st_verify_eq_(container.size(), 3);

    std::size_t walked = 0;
    container.range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(4),
                    [&](member_t const &member) noexcept {
                        ++walked;
                        st_verify_(trivial_id_to_key<member_t>(2) != mapping_key_or_itself<member_t>(member));
                    });
    st_verify_eq_(walked, 3);

    // The tombstone sits exactly on the sought key, so a bound that ignored it would return it.
    container.lower_bound(
        trivial_id_to_key<member_t>(2),
        [&](member_t const &member) noexcept {
            st_verify_(trivial_id_to_key<member_t>(3) == mapping_key_or_itself<member_t>(member));
        },
        []() noexcept { st_verify_(false && "A live key sits above the tombstone"); });
    container.upper_bound(
        trivial_id_to_key<member_t>(1),
        [&](member_t const &member) noexcept {
            st_verify_(trivial_id_to_key<member_t>(3) == mapping_key_or_itself<member_t>(member));
        },
        []() noexcept { st_verify_(false && "A live key sits above the tombstone"); });

    // Sampling shares the range surface, so it must not draw the tombstone either.
    std::mt19937 generator(42);
    for (std::size_t attempt = 0; attempt != 64; ++attempt)
        container.sample_one(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(4), generator,
                             [&](member_t const &member) noexcept {
                                 st_verify_(trivial_id_to_key<member_t>(2) != mapping_key_or_itself<member_t>(member));
                             });
}

/** @brief A mutating ranged update never hands a caller the payload of a tombstone. */
template <typename container_type_>
void test_committed_erase_hidden_from_update_range() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 4; ++index)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(index, index))));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->erase(trivial_id_to_key<member_t>(2))));
    st_verify_(succeeded(transaction->stage()));
    st_verify_(succeeded(transaction->commit()));

    std::size_t visited = 0;
    container.update_range(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(4),
                           [&](auto const &key, auto &mapped) noexcept {
                               ++visited;
                               st_verify_(trivial_id_to_key<member_t>(2) != key);
                               mapped = typename member_t::mapped_type(9);
                           });
    st_verify_eq_(visited, 3);
    st_verify_(!container.contains(trivial_id_to_key<member_t>(2)));
}

#pragma endregion Tombstone Visibility

#pragma region Vacuuming

/** @brief A committed tombstone is reclaimable, and reclaiming it changes nothing observable. */
template <typename container_type_>
void test_vacuum_reclaims_committed_tombstones() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 6; ++index)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(index, index))));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->erase(trivial_id_to_key<member_t>(1))));
    st_verify_(succeeded(transaction->erase(trivial_id_to_key<member_t>(4))));
    st_verify_(succeeded(transaction->stage()));
    st_verify_(succeeded(transaction->commit()));
    st_verify_eq_(container.size(), 4);

    auto reclaimed = container.vacuum();
    st_verify_(reclaimed.has_value());
    st_verify_eq_(*reclaimed, 2);
    st_verify_eq_(container.size(), 4);
    st_verify_(!container.contains(trivial_id_to_key<member_t>(1)));
    st_verify_(container.contains(trivial_id_to_key<member_t>(0)));

    // Nothing is left to reclaim, and a live key is never mistaken for a tombstone.
    auto second_pass = container.vacuum();
    st_verify_(second_pass.has_value());
    st_verify_eq_(*second_pass, 0);
    st_verify_eq_(container.size(), 4);
}

/** @brief A staged version behind a tombstone keeps its entry, since a commit may still publish it. */
template <typename container_type_>
void test_vacuum_spares_staged_versions() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(5, 5))));

    auto erasing = container.transaction();
    st_verify_(erasing.has_value());
    st_verify_(succeeded(erasing->erase(trivial_id_to_key<member_t>(5))));
    st_verify_(succeeded(erasing->stage()));
    st_verify_(succeeded(erasing->commit()));

    auto rewriting = container.transaction();
    st_verify_(rewriting.has_value());
    st_verify_(succeeded(rewriting->upsert(trivial_id_to_member<member_t>(5, 55))));
    st_verify_(succeeded(rewriting->stage()));

    auto reclaimed = container.vacuum();
    st_verify_(reclaimed.has_value());
    st_verify_eq_(*reclaimed, 0);

    st_verify_(succeeded(rewriting->commit()));
    st_verify_eq_(container.size(), 1);
    st_verify_(container.contains(trivial_id_to_key<member_t>(5)));
}

/** @brief The windowed overload reclaims only its own slice, so a caller can walk the keyspace in steps. */
template <typename container_type_>
void test_windowed_vacuum_reclaims_one_slice() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index = 0; index != 8; ++index)
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(index, index))));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->erase(trivial_id_to_key<member_t>(1))));
    st_verify_(succeeded(transaction->erase(trivial_id_to_key<member_t>(6))));
    st_verify_(succeeded(transaction->stage()));
    st_verify_(succeeded(transaction->commit()));

    auto lower_slice = container.vacuum(trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(4));
    st_verify_(lower_slice.has_value());
    st_verify_eq_(*lower_slice, 1);

    auto upper_slice = container.vacuum(trivial_id_to_key<member_t>(4), trivial_id_to_key<member_t>(8));
    st_verify_(upper_slice.has_value());
    st_verify_eq_(*upper_slice, 1);

    auto exhausted = container.vacuum();
    st_verify_(exhausted.has_value());
    st_verify_eq_(*exhausted, 0);
    st_verify_eq_(container.size(), 6);
}

#pragma endregion Vacuuming

#pragma region Transaction Reads

/**
 *  @brief A transaction's own range is one sorted sequence, however its staged keys interleave with
 *    the committed ones, and a staged tombstone hides the committed key underneath it.
 */
template <typename container_type_>
void test_transaction_range_interleaves_staged_and_committed() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    for (std::size_t index : {0u, 2u, 4u, 6u})
        st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(index, index))));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    for (std::size_t index : {1u, 3u, 5u})
        st_verify_(succeeded(transaction->upsert(trivial_id_to_member<member_t>(index, index))));
    st_verify_(succeeded(transaction->erase(trivial_id_to_key<member_t>(4))));

    std::size_t const expected_ids[] = {0, 1, 2, 3, 5, 6};
    std::size_t walked = 0;
    transaction->range(
        trivial_id_to_key<member_t>(0), trivial_id_to_key<member_t>(8), [&](member_t const &member) noexcept {
            st_verify_(walked < 6);
            st_verify_(trivial_id_to_key<member_t>(expected_ids[walked]) == mapping_key_or_itself<member_t>(member));
            ++walked;
        });
    st_verify_eq_(walked, 6);

    // A half-open window sees the same order, and the bound still excludes its upper key.
    walked = 0;
    transaction->range(trivial_id_to_key<member_t>(1), trivial_id_to_key<member_t>(5),
                       [&](member_t const &member) noexcept {
                           st_verify_(walked < 3);
                           st_verify_(trivial_id_to_key<member_t>(expected_ids[walked + 1]) ==
                                      mapping_key_or_itself<member_t>(member));
                           ++walked;
                       });
    st_verify_eq_(walked, 3);
}

/** @brief A transaction's @c equal_range answers from its own staged writes as well as the store. */
template <typename container_type_>
void test_transaction_equal_range_sees_staged_writes() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(1, 1))));

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->upsert(trivial_id_to_member<member_t>(2, 22))));
    st_verify_(succeeded(transaction->erase(trivial_id_to_key<member_t>(1))));

    std::size_t staged_matches = 0;
    transaction->equal_range(trivial_id_to_key<member_t>(2), [&](member_t const &member) noexcept {
        ++staged_matches;
        if constexpr (is_mapping<member_t>) st_verify_eq_(member.mapped, typename member_t::mapped_type(22));
    });
    st_verify_eq_(staged_matches, 1);

    std::size_t erased_matches = 0;
    transaction->equal_range(trivial_id_to_key<member_t>(1), [&](member_t const &) noexcept { ++erased_matches; });
    st_verify_eq_(erased_matches, 0);

    std::size_t absent_matches = 0;
    transaction->equal_range(trivial_id_to_key<member_t>(9), [&](member_t const &) noexcept { ++absent_matches; });
    st_verify_eq_(absent_matches, 0);
}

#pragma endregion Transaction Reads

#pragma region Status Reporting

/** @brief A key collision reads the same at both levels, and from the bulk entry point too. */
template <typename container_type_>
void test_insert_reports_key_already_exists() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(1, 1))));

    st_verify_eq_(container.insert(trivial_id_to_member<member_t>(1, 9)), status_t::key_already_exists_k);

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_eq_(transaction->insert(trivial_id_to_member<member_t>(1, 9)), status_t::key_already_exists_k);

    if constexpr (std::is_copy_constructible_v<member_t>) {
        member_t const batch[] = {trivial_id_to_member<member_t>(1, 9), trivial_id_to_member<member_t>(2, 2)};
        st_verify_eq_(container.insert(batch, batch + 2), status_t::key_already_exists_k);
    }
}

/** @brief A lookup that found nothing says so, rather than reporting a reason it never established. */
template <typename container_type_>
void test_find_copy_reports_key_not_found() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(succeeded(container.upsert(trivial_id_to_member<member_t>(1, 1))));

    auto missing = container.find_copy(trivial_id_to_key<member_t>(2));
    st_verify_(!missing.has_value());
    st_verify_eq_(missing.status(), status_t::key_not_found_k);

    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    auto missing_in_transaction = transaction->find_copy(trivial_id_to_key<member_t>(2));
    st_verify_(!missing_in_transaction.has_value());
    st_verify_eq_(missing_in_transaction.status(), status_t::key_not_found_k);
}

/** @brief Staging twice is refused, so a second pass cannot file the same keys again. */
template <typename container_type_>
void test_second_stage_is_rejected() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    auto transaction = container.transaction();
    st_verify_(transaction.has_value());
    st_verify_(succeeded(transaction->upsert(trivial_id_to_member<member_t>(1, 1))));
    st_verify_(succeeded(transaction->stage()));
    st_verify_eq_(transaction->stage(), status_t::operation_not_permitted_k);

    st_verify_(succeeded(transaction->commit()));
    st_verify_eq_(container.size(), 1);
    st_verify_(container.contains(trivial_id_to_key<member_t>(1)));
}

#pragma endregion Status Reporting

} // namespace ashvardanian::smashtable::scripts
