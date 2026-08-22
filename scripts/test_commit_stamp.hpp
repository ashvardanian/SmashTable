/**
 *  @brief Template tests for the rule that a commit stamp - and nothing else - decides which version of a key
 *      is current. Covers what that buys, which is a transaction validating through someone else's staging
 *      window, and what it must still refuse, which is a second writer publishing over a stale base.
 *  @author Ash Vardanian
 *  @file scripts/test_commit_stamp.hpp
 *  @date August 17, 2026
 */
#pragma once
#include "test_basic.hpp"

namespace ashvardanian::smashtable::scripts {

/**
 *  @brief A staged write that is later rolled back must abort nobody.
 *
 *  The writer never publishes, so nothing was ever current but the value both transactions read.
 *  Refusing the peer would be a false abort - the retry it forces recomputes the same answer from
 *  the same base, and under contention a queue of writers that all roll back starves every reader.
 */
template <typename container_type_>
void test_rolled_back_stage_does_not_abort_a_peer() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 100)));

    // The writer stages a change it will never publish.
    auto writer = container.transaction();
    st_verify_(writer->upsert(trivial_id_to_member<member_t>(1, 999)));
    st_verify_(writer->stage());

    // The peer reads the key, records what it saw, and stages its own change on top of that read.
    auto peer = container.transaction();
    int observed = 0;
    st_verify_(peer->find_and_watch(
        trivial_id_to_key<member_t>(1), [&](member_t const &member) noexcept { observed = int(member.mapped); },
        []() noexcept {}));
    st_verify_eq_(observed, 100);
    st_verify_(peer->upsert(trivial_id_to_member<member_t>(1, 200)));
    st_verify_(succeeded(peer->stage()) && "a write nobody has committed must not refuse a reader");

    // The writer gives up, so the peer's read was never invalidated by anything.
    st_verify_(writer->rollback());
    st_verify_(succeeded(peer->commit()) && "a rolled-back write must not refuse the commit either");

    auto maybe_final = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_(maybe_final.has_value());
    st_verify_eq_(int(maybe_final->mapped), 200);
}

/**
 *  @brief Two writers computing from one base: the second one must be refused, not silently merged.
 *
 *  Both read 100 and both intend 101, so letting both land would leave a counter one short of the
 *  two increments it was asked for. The commit stamp is what catches it: the loser's watch names a
 *  version that is not the committed one by the time it publishes.
 */
template <typename container_type_>
void test_lost_update_is_refused() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;
    st_verify_(container.upsert(trivial_id_to_member<member_t>(1, 100)));

    auto first = container.transaction();
    auto second = container.transaction();

    // Whether a watch fired is a separate fact from what it read, so two callbacks that never ran
    // cannot agree their way past this by leaving both seeds untouched.
    int read_by_first = 0, read_by_second = 0;
    std::size_t watches_fired = 0;
    st_verify_(first->find_and_watch(
        trivial_id_to_key<member_t>(1),
        [&](member_t const &member) noexcept {
            read_by_first = int(member.mapped);
            ++watches_fired;
        },
        []() noexcept {}));
    st_verify_(second->find_and_watch(
        trivial_id_to_key<member_t>(1),
        [&](member_t const &member) noexcept {
            read_by_second = int(member.mapped);
            ++watches_fired;
        },
        []() noexcept {}));
    st_verify_eq_(watches_fired, std::size_t {2}, "both watches must have read the key they were pointed at");
    st_verify_eq_(read_by_first, read_by_second);

    // Each stages the increment it computed. Neither is committed, so neither refuses the other.
    st_verify_(first->upsert(trivial_id_to_member<member_t>(1, read_by_first + 1)));
    st_verify_(second->upsert(trivial_id_to_member<member_t>(1, read_by_second + 1)));
    st_verify_(first->stage());
    st_verify_(second->stage());

    // The first to publish wins the key, and the second is turned away rather than losing the
    // increment underneath it.
    st_verify_(first->commit());
    auto const refused = second->commit();
    st_verify_eq_(refused, status_t::read_conflict_k);

    auto maybe_final = container.find_copy(trivial_id_to_key<member_t>(1));
    st_verify_(maybe_final.has_value());
    st_verify_eq_(int(maybe_final->mapped), 101);
}

/**
 *  @brief A read recorded through @c find_and_watch on a key that is not there is still a read.
 *    Whoever creates that key afterwards invalidates it, which is what stops two transactions from
 *    both believing they are the one inserting it.
 */
template <typename container_type_>
void test_find_and_watch_records_absence() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;

    static_assert(container_t::is_associative::value, "Container must be key-value");
    static_assert(container_t::is_transactional::value, "Container must be transactional");

    container_t container;

    auto reader = container.transaction();
    bool saw_absence = false;
    st_verify_(reader->find_and_watch(
        trivial_id_to_key<member_t>(7), [](member_t const &) noexcept {}, [&]() noexcept { saw_absence = true; }));
    st_verify_(saw_absence && "the key must be reported missing");

    // Somebody else creates the key the reader was promised was absent.
    st_verify_(container.upsert(trivial_id_to_member<member_t>(7, 42)));

    st_verify_(reader->upsert(trivial_id_to_member<member_t>(7, 43)));
    auto const refused = reader->stage();
    st_verify_eq_(refused, status_t::read_conflict_k);
}

} // namespace ashvardanian::smashtable::scripts
