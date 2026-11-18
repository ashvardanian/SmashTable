/**
 *  @file test/fuzz.hpp
 *  @author Ash Vardanian
 *  @date August 20, 2026
 *  @brief Randomized differential suites: one engine against the oracle, groups against tearing.
 *
 *  Included by @c test/snapshot_store.cpp and no other binary, so a suite added here runs once.
 *
 *  @section fuzz_oracle The Oracle
 *
 *  One Engine Against the Oracle steps a @c reference_store through the same operation sequence as
 *  the engine under test and compares the two after every step rather than at the end - so a
 *  divergence names the operation that caused it instead of the walk that noticed. Groups Against
 *  Tearing compares each participant against what its round intended instead: the snapshot taken
 *  before a refused commit, and a @c reference_store advanced by the same write after an accepted
 *  one. Never one participant against the other, which two stores publishing nothing would pass.
 *
 *  @section fuzz_seed Reproducing a Failure
 *
 *  Every sequence is drawn from @c case_stream_key, which mixes the run's seed with the case's
 *  label, so @c SMASHTABLE_SEED or otherwise 42 picks it. A failing run is reproduced by the rerun
 *  line it printed, a sweep loops over the variable instead of editing source, and
 *  @c SMASHTABLE_SCALE stretches every suite's rounds.
 */
#pragma once
#include <compare> // `std::compare_three_way`
#include <map>     // `std::map`
#include <vector>  // `std::vector`

#include <smashtable/reference_store.hpp>

#include "harness.hpp"
#include "fixtures.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Shared Fixtures

/** How many keys a fuzz round draws from, kept small so collisions and reuse are common. */
inline constexpr trivial_id_t fuzz_keyspace_k = 24;

/** The oracle every engine here is checked against, over the same key and mapped types. */
using fuzz_oracle_t =
    reference_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/** What one engine and the oracle both hold, so a divergence is one comparison, not a walk. */
struct fuzz_snapshot_t {
    std::size_t size {0};
    std::size_t checksum {0};

    friend bool operator==(fuzz_snapshot_t const &, fuzz_snapshot_t const &) noexcept = default;
};

/**
 *  @brief Folds a store's visible contents into a size and an order-independent checksum.
 *
 *  Order-independent so an unordered core answers the same as an ordered one, which is what lets a
 *  single comparison stand in for a zipped walk the hash engines could not take.
 */
template <typename store_type_>
fuzz_snapshot_t fuzz_snapshot_of(store_type_ &store) noexcept {
    fuzz_snapshot_t taken;
    [[maybe_unused]] status_t const walked = store.for_each([&](auto const &member) noexcept {
        taken.checksum += std::size_t {member.key.unique_id} * 1000003u + std::size_t(member.mapped);
        ++taken.size;
    });
    return taken;
}

#pragma endregion Shared Fixtures

#pragma region One Engine Against the Oracle

/**
 *  @brief Drives a random write sequence through an engine and the oracle, comparing at each step.
 *
 *  The strict writers are the point rather than @c upsert: @c insert, @c insert_if_missing and
 *  @c update each answer a question about what the caller can already see, and an engine consulting
 *  the store where its transaction has staged a tombstone answers a different one. Every status is
 *  compared, not only the resulting state, since a write that reports success and stages nothing
 *  leaves the state agreeing for the wrong reason.
 */
template <typename store_type_>
void test_random_writes_match_the_oracle() {
    std::size_t const rounds = scaled_iterations(400);

    using store_t = store_type_;
    using member_t = typename store_t::value_type;

    static_assert(store_t::is_associative::value, "Container must be key-value");
    static_assert(store_t::is_transactional::value, "Container must be transactional");

    split_mix_generator_t generator {case_stream_key()};
    store_t engine;
    fuzz_oracle_t oracle;

    for (std::size_t round = 0; round != rounds; ++round) {
        trivial_id_t const identifier = trivial_id_t(generator() % fuzz_keyspace_k);
        int const value = int(generator() % 1000);

        auto engine_transaction = engine.transaction();
        if (!expect_true(engine_transaction.has_value())) return;
        auto oracle_transaction = oracle.transaction();
        if (!expect_true(oracle_transaction.has_value())) return;

        // A tombstone staged before the write is what separates the merged view from the store, so
        // roughly a third of the rounds erase first and then ask a strict writer about the same key.
        bool const erasing_first = (generator() % 3) == 0;
        if (erasing_first) {
            [[maybe_unused]] status_t const from_engine = engine_transaction->erase(trivial_key_t {identifier});
            [[maybe_unused]] status_t const from_oracle = oracle_transaction->erase(trivial_key_t {identifier});
        }

        status_t from_engine = success_k;
        status_t from_oracle = success_k;
        switch (generator() % 4) {
        case 0:
            from_engine = engine_transaction->upsert(trivial_id_to_member<member_t>(identifier, value));
            from_oracle = oracle_transaction->upsert(trivial_id_to_member<member_t>(identifier, value));
            break;
        case 1:
            from_engine = engine_transaction->insert(trivial_id_to_member<member_t>(identifier, value));
            from_oracle = oracle_transaction->insert(trivial_id_to_member<member_t>(identifier, value));
            break;
        case 2:
            from_engine = engine_transaction->insert_if_missing(trivial_id_to_member<member_t>(identifier, value));
            from_oracle = oracle_transaction->insert_if_missing(trivial_id_to_member<member_t>(identifier, value));
            break;
        default:
            from_engine = engine_transaction->update(trivial_id_to_member<member_t>(identifier, value));
            from_oracle = oracle_transaction->update(trivial_id_to_member<member_t>(identifier, value));
            break;
        }
        expect_equal(from_engine, from_oracle, "an engine and the oracle answered a strict write differently");

        expected<bool> const engine_sees = engine_transaction->contains(trivial_key_t {identifier});
        expected<bool> const oracle_sees = oracle_transaction->contains(trivial_key_t {identifier});
        if (!expect_true(engine_sees.has_value())) return;
        if (!expect_true(oracle_sees.has_value())) return;
        expect_equal((*engine_sees), (*oracle_sees), "a transaction disagreed with the oracle about its own view");

        expect_equal(engine_transaction->stage(), success_k);
        expect_equal(oracle_transaction->stage(), success_k);
        expect_equal(engine_transaction->commit(), success_k);
        expect_equal(oracle_transaction->commit(), success_k);

        expect_equal(engine.size(), oracle.size(), "a committed round left the two holding different counts");
        expect_equal(fuzz_snapshot_of(engine), fuzz_snapshot_of(oracle),
                     "a committed round left the two holding different contents");
    }
}

/**
 *  @brief Keeps one transaction across every round and compares the engine against a @c std::map.
 *
 *  A committed transaction returns to pending and is reused, so every round restages keys the same
 *  transaction already published. A round ends in a commit, in a reset of what it staged, or in a
 *  rollback followed by a commit. A reader opened at the round's start is read while the writer
 *  sits staged, since a version restaged over its own published one vanishes from everyone else.
 */
template <typename store_type_>
void test_a_reused_transaction_matches_a_map() {
    std::size_t const rounds = scaled_iterations(300);

    using store_t = store_type_;
    using member_t = typename store_t::value_type;

    split_mix_generator_t generator {case_stream_key()};
    store_t engine;
    std::map<trivial_id_t, int> model;

    auto const expect_reads = [](auto const &readable, std::map<trivial_id_t, int> const &wanted, char const *when) {
        for (trivial_id_t identifier = 0; identifier != fuzz_keyspace_k; ++identifier) {
            expected<int> held {status_t::key_not_found_k};
            expect_equal(readable.find(
                             trivial_key_t {identifier}, [&](member_t const &member) noexcept { held = member.mapped; },
                             no_op_t {}),
                         success_k);
            auto const modelled = wanted.find(identifier);
            if (modelled == wanted.end()) expect_equal(held.status(), status_t::key_not_found_k, "{}", when);
            else if (expect_equal(held.status(), success_k, "{}", when))
                expect_equal(*held, modelled->second, "{}", when);
        }
    };

    auto writer = engine.transaction();
    if (!expect_true(writer.has_value())) return;

    for (std::size_t round = 0; round != rounds; ++round) {
        std::map<trivial_id_t, int> intended = model;
        {
            auto pinned = engine.transaction();
            if (!expect_true(pinned.has_value())) return;

            for (std::size_t written = 0, writes = 1 + generator() % 3; written != writes; ++written) {
                trivial_id_t const identifier = trivial_id_t(generator() % fuzz_keyspace_k);
                if ((generator() % 4) == 0) {
                    [[maybe_unused]] status_t const erased = writer->erase(trivial_key_t {identifier});
                    intended.erase(identifier);
                }
                else {
                    int const value = int(generator() % 1000);
                    expect_equal(writer->upsert(trivial_id_to_member<member_t>(identifier, value)), success_k);
                    intended[identifier] = value;
                }
            }

            expect_equal(writer->stage(), success_k);
            expect_reads(*pinned, model, "a reader pinned before the round, while the writer is staged");
            expect_reads(engine, model, "the store, while the writer is staged");

            switch (generator() % 3) {
            case 0:
                expect_equal(writer->commit(), success_k);
                model = intended;
                break;
            case 1: expect_equal(writer->reset(), success_k); break;
            default:
                expect_equal(writer->rollback(), success_k);
                expect_reads(engine, model, "the store, after the writer rolled back");
                expect_equal(writer->stage(), success_k);
                expect_equal(writer->commit(), success_k);
                model = intended;
                break;
            }
        }

        expect_reads(engine, model, "the store, after the round");
        expect_equal(engine.size(), model.size(), "a reused transaction left the store miscounting its members");
        if constexpr (requires { engine.versions_count(); }) {
            expect_true(engine.vacuum().has_value());
            expect_equal(engine.versions_count(), engine.size(),
                         "a vacuum with no reader open keeps one version a key");
        }
    }
}

/**
 *  @brief Drives whole-window mutators, which reach every partition of a partitioned store.
 *
 *  A sharded store answers these by walking every partition rather than the one a key hashes to, so
 *  a window mutator that never records which partitions it reached stages tombstones the commit
 *  walks past. The oracle has no partitions and so cannot lose them.
 */
template <typename store_type_>
void test_random_windows_match_the_oracle() {
    std::size_t const rounds = scaled_iterations(120);

    using store_t = store_type_;
    using member_t = typename store_t::value_type;

    split_mix_generator_t generator {case_stream_key()};
    store_t engine;
    fuzz_oracle_t oracle;

    for (std::size_t round = 0; round != rounds; ++round) {
        auto engine_transaction = engine.transaction();
        if (!expect_true(engine_transaction.has_value())) return;
        auto oracle_transaction = oracle.transaction();
        if (!expect_true(oracle_transaction.has_value())) return;

        for (std::size_t written = 0; written != 4; ++written) {
            trivial_id_t const identifier = trivial_id_t(generator() % fuzz_keyspace_k);
            int const value = int(generator() % 1000);
            expect_equal(engine_transaction->upsert(trivial_id_to_member<member_t>(identifier, value)), success_k);
            expect_equal(oracle_transaction->upsert(trivial_id_to_member<member_t>(identifier, value)), success_k);
        }

        // A window drawn from the same keyspace, so it usually holds something rather than testing
        // that an empty range erases nothing.
        if ((generator() % 2) == 0) {
            trivial_id_t const first = trivial_id_t(generator() % fuzz_keyspace_k);
            trivial_id_t const second = trivial_id_t(generator() % fuzz_keyspace_k);
            trivial_key_t const lower {first < second ? first : second};
            trivial_key_t const upper {first < second ? second : first};
            expect_equal(engine_transaction->erase_range(lower, upper, no_op_t {}), success_k);
            expect_equal(oracle_transaction->erase_range(lower, upper, no_op_t {}), success_k);
        }

        expect_equal(engine_transaction->stage(), success_k);
        expect_equal(oracle_transaction->stage(), success_k);
        expect_equal(engine_transaction->commit(), success_k);
        expect_equal(oracle_transaction->commit(), success_k);

        expect_equal(engine.size(), oracle.size(), "a window mutator left the two holding different counts");
        expect_equal(fuzz_snapshot_of(engine), fuzz_snapshot_of(oracle),
                     "a window mutator left the two holding different contents");
    }
}

#pragma endregion One Engine Against the Oracle

#pragma region Groups Against Tearing

/**
 *  @brief Refuses a random group commit and checks that no participant moved.
 *
 *  A group that publishes one participant and is then turned away by the next leaves the two
 *  disagreeing about whether the update happened, and no retry can reconcile them. So the refusal
 *  is forced rather than raced - a watched key is published over while the group sits staged - and
 *  both participants are compared against the snapshot taken before the commit was attempted.
 */
template <typename first_store_type_, typename second_store_type_>
void test_a_refused_group_publishes_nothing() {
    std::size_t const rounds = scaled_iterations(60);

    using first_t = first_store_type_;
    using second_t = second_store_type_;
    using first_member_t = typename first_t::value_type;
    using second_member_t = typename second_t::value_type;

    split_mix_generator_t generator {case_stream_key()};
    // One commit across two stores means one order both were built into; two stores that each made
    // their own are two visibility domains, and `make_transaction_group` refuses them by design.
    using order_t = typename first_t::order_t;
    order_t order;
    first_t alpha {order};
    second_t beta {order};

    for (trivial_id_t identifier = 0; identifier != fuzz_keyspace_k; ++identifier) {
        expect_equal(alpha.upsert(trivial_id_to_member<first_member_t>(identifier, 0)), success_k);
        expect_equal(beta.upsert(trivial_id_to_member<second_member_t>(identifier, 0)), success_k);
    }

    // A group visits its participants in ascending store address, so the refusal has to land on the
    // one visited last or the earlier participant never reaches the point of publishing and a torn
    // commit cannot happen. Which of the two that is depends on where the compiler put them, so it
    // is asked rather than assumed - the same total order the group itself is built on.
    bool const beta_is_last =
        std::compare_three_way {}(static_cast<void const *>(&beta), static_cast<void const *>(&alpha)) > 0;

    for (std::size_t round = 0; round != rounds; ++round) {
        trivial_id_t const watched = trivial_id_t(generator() % fuzz_keyspace_k);
        trivial_id_t const written = trivial_id_t(generator() % fuzz_keyspace_k);

        auto group = make_transaction_group(alpha, beta);
        if (!expect_true(group.has_value())) return;
        auto &into_alpha = group->template participant<0>();
        auto &into_beta = group->template participant<1>();

        // The participant carrying the watch writes the watched key too, so the refusal is always the
        // write set catching it rather than the read set - one code to assert instead of two.
        if (beta_is_last) {
            expect_equal(into_alpha.upsert(trivial_id_to_member<first_member_t>(written, int(round) + 1)), success_k);
            expect_equal(into_beta.upsert(trivial_id_to_member<second_member_t>(watched, int(round) + 1)), success_k);
            expect_equal(into_beta.watch(trivial_key_t {watched}), success_k);
        }
        else {
            expect_equal(into_alpha.upsert(trivial_id_to_member<first_member_t>(watched, int(round) + 1)), success_k);
            expect_equal(into_beta.upsert(trivial_id_to_member<second_member_t>(written, int(round) + 1)), success_k);
            expect_equal(into_alpha.watch(trivial_key_t {watched}), success_k);
        }
        expect_equal(group->stage(), success_k);

        // Published over while the group is staged, which is the one thing a validated commit still
        // has to catch and the only refusal a staged group can meet.
        if (beta_is_last) expect_equal(beta.upsert(trivial_id_to_member<second_member_t>(watched, 9999)), success_k);
        else expect_equal(alpha.upsert(trivial_id_to_member<first_member_t>(watched, 9999)), success_k);

        fuzz_snapshot_t const alpha_before = fuzz_snapshot_of(alpha);
        fuzz_snapshot_t const beta_before = fuzz_snapshot_of(beta);
        status_t const refused = group->commit();
        expect_equal(refused, status_t::write_conflict_k, "a commit over a published watch must be refused");
        expect_equal(fuzz_snapshot_of(alpha), alpha_before, "a refused group published its first participant");
        expect_equal(fuzz_snapshot_of(beta), beta_before, "a refused group published its second participant");
        expect_equal(group->staging(), staging_t::staged_k, "a refused group must stay staged to be unwindable");
        expect_equal(group->rollback(), success_k);
    }
}

/**
 *  @brief Commits random groups that nothing refuses, and checks all participants advanced as one.
 *
 *  The mirror of the refusal case: a group that publishes must publish all of itself, so the pair
 *  is compared against a model advanced only when the commit answered success.
 */
template <typename first_store_type_, typename second_store_type_>
void test_an_accepted_group_publishes_everything() {
    std::size_t const rounds = scaled_iterations(120);

    using first_t = first_store_type_;
    using second_t = second_store_type_;
    using first_member_t = typename first_t::value_type;
    using second_member_t = typename second_t::value_type;
    using model_member_t = typename fuzz_oracle_t::value_type;

    split_mix_generator_t generator {case_stream_key()};
    // One commit across two stores means one order both were built into; two stores that each made
    // their own are two visibility domains, and `make_transaction_group` refuses them by design.
    using order_t = typename first_t::order_t;
    order_t order;
    first_t alpha {order};
    second_t beta {order};
    fuzz_oracle_t model;
    std::size_t published_rounds = 0;

    // One group across every round, so each commit restages keys the same participants published.
    auto group = make_transaction_group(alpha, beta);
    if (!expect_true(group.has_value())) return;
    std::vector<model_member_t> pending;

    for (std::size_t round = 0; round != rounds; ++round) {
        trivial_id_t const identifier = trivial_id_t(generator() % fuzz_keyspace_k);
        int const value = int(round) + 1;
        bool const unwinding = (generator() % 4) == 0;

        expect_equal(group->template participant<0>().upsert(trivial_id_to_member<first_member_t>(identifier, value)),
                     success_k);
        expect_equal(group->template participant<1>().upsert(trivial_id_to_member<second_member_t>(identifier, value)),
                     success_k);
        pending.push_back(trivial_id_to_member<model_member_t>(identifier, value));
        expect_equal(group->stage(), success_k);

        // A rollback keeps the round's writes pending, so the next commit publishes them as well.
        if (unwinding) { expect_equal(group->rollback(), success_k); }
        else {
            expect_equal(group->commit(), success_k);
            for (model_member_t const &member : pending) expect_equal(model.upsert(model_member_t(member)), success_k);
            pending.clear();
            ++published_rounds;
        }

        // Each participant answers to the model rather than to the other, so a round that published
        // nothing anywhere cannot pass by leaving the two equally empty.
        fuzz_snapshot_t const intended = fuzz_snapshot_of(model);
        expect_equal(fuzz_snapshot_of(alpha), intended, "the first participant lost what its round published");
        expect_equal(fuzz_snapshot_of(beta), intended, "the second participant lost what its round published");
        expect_equal(alpha.size(), model.size(), "the first participant miscounts what the group published");
        expect_equal(beta.size(), model.size(), "the second participant miscounts what the group published");
    }

    expect_true(published_rounds != 0, "no round published, so this proves nothing");
}

#pragma endregion Groups Against Tearing

} // namespace ashvardanian::smashtable::test
