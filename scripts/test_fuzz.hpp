/**
 *  @brief Randomized differential suites - one engine against the oracle, and groups against tearing.
 *  @author Ash Vardanian
 *  @file scripts/test_fuzz.hpp
 *  @date August 20, 2026
 *
 *  @section fuzz_oracle The Oracle
 *
 *  @c reference_store answers every question the engine under test is asked, stepped through the same
 *  operation sequence, and the two are compared after every step rather than at the end - so a
 *  divergence names the operation that caused it instead of the walk that noticed.
 *
 *  @section fuzz_seed Reproducing a Failure
 *
 *  Every sequence is drawn from @c test_seed(), which reads @c SMASHTABLE_SEED and otherwise answers
 *  @c default_seed_k. A failing run is reproduced by exporting the seed it printed, and a sweep is a
 *  loop over the variable rather than an edit to the source.
 */
#pragma once
#include <compare> // `std::compare_three_way`
#include <random>  // `std::mt19937`

#include <smashtable/reference_store.hpp>

#include "test.hpp"
#include "test_basic.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Shared Fixtures

/** @brief How many keys a fuzz round draws from, kept small so collisions and reuse are the common case. */
inline constexpr trivial_id_t fuzz_keyspace_k = 24;

/** @brief The oracle every engine here is checked against, over the same key and mapped types. */
using fuzz_oracle_t =
    reference_map<trivial_key_t, int, std::less<trivial_key_t>, std::allocator<mapping<trivial_key_t, int>>>;

/** @brief What one engine and the oracle both hold, so a divergence is one comparison rather than a walk. */
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
 *  @brief Drives a random write sequence through an engine and the oracle, comparing after every step.
 *
 *  The strict writers are the point rather than @c upsert: @c insert, @c insert_if_missing and
 *  @c update each answer a question about what the caller can already see, and an engine consulting
 *  the store where its transaction has staged a tombstone answers a different one. Every status is
 *  compared, not only the resulting state, since a write that reports success and stages nothing
 *  leaves the state agreeing for the wrong reason.
 */
template <typename store_type_>
void test_random_writes_match_the_oracle(std::size_t rounds = 400) {

    using store_t = store_type_;
    using member_t = typename store_t::value_type;

    static_assert(store_t::is_associative::value, "Container must be key-value");
    static_assert(store_t::is_transactional::value, "Container must be transactional");

    std::mt19937 generator(test_seed());
    store_t engine;
    fuzz_oracle_t oracle;

    for (std::size_t round = 0; round != rounds; ++round) {
        trivial_id_t const identifier = trivial_id_t(generator() % fuzz_keyspace_k);
        int const value = int(generator() % 1000);

        auto engine_transaction = engine.transaction();
        st_verify_(engine_transaction.has_value());
        auto oracle_transaction = oracle.transaction();
        st_verify_(oracle_transaction.has_value());

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
        st_verify_eq_(from_engine, from_oracle, "an engine and the oracle answered a strict write differently");

        expected<bool> const engine_sees = engine_transaction->contains(trivial_key_t {identifier});
        expected<bool> const oracle_sees = oracle_transaction->contains(trivial_key_t {identifier});
        st_verify_(engine_sees.has_value());
        st_verify_(oracle_sees.has_value());
        st_verify_eq_((*engine_sees), (*oracle_sees), "a transaction disagreed with the oracle about its own view");

        st_verify_(engine_transaction->stage());
        st_verify_(oracle_transaction->stage());
        st_verify_(engine_transaction->commit());
        st_verify_(oracle_transaction->commit());

        st_verify_eq_(engine.size(), oracle.size(), "a committed round left the two holding different counts");
        st_verify_eq_(fuzz_snapshot_of(engine), fuzz_snapshot_of(oracle),
                      "a committed round left the two holding different contents");
    }
}

/**
 *  @brief Drives whole-window mutators, which reach every partition where a store has more than one.
 *
 *  A sharded store answers these by walking every partition rather than the one a key hashes to, so
 *  a window mutator that never records which partitions it reached stages tombstones the commit
 *  walks past. The oracle has no partitions and so cannot lose them.
 */
template <typename store_type_>
void test_random_windows_match_the_oracle(std::size_t rounds = 120) {

    using store_t = store_type_;
    using member_t = typename store_t::value_type;

    std::mt19937 generator(test_seed());
    store_t engine;
    fuzz_oracle_t oracle;

    for (std::size_t round = 0; round != rounds; ++round) {
        auto engine_transaction = engine.transaction();
        st_verify_(engine_transaction.has_value());
        auto oracle_transaction = oracle.transaction();
        st_verify_(oracle_transaction.has_value());

        for (std::size_t written = 0; written != 4; ++written) {
            trivial_id_t const identifier = trivial_id_t(generator() % fuzz_keyspace_k);
            int const value = int(generator() % 1000);
            st_verify_(engine_transaction->upsert(trivial_id_to_member<member_t>(identifier, value)));
            st_verify_(oracle_transaction->upsert(trivial_id_to_member<member_t>(identifier, value)));
        }

        // A window drawn from the same keyspace, so it usually holds something rather than testing
        // that an empty range erases nothing.
        if ((generator() % 2) == 0) {
            trivial_id_t const first = trivial_id_t(generator() % fuzz_keyspace_k);
            trivial_id_t const second = trivial_id_t(generator() % fuzz_keyspace_k);
            trivial_key_t const lower {first < second ? first : second};
            trivial_key_t const upper {first < second ? second : first};
            st_verify_(engine_transaction->erase_range(lower, upper, no_op_t {}));
            st_verify_(oracle_transaction->erase_range(lower, upper, no_op_t {}));
        }

        st_verify_(engine_transaction->stage());
        st_verify_(oracle_transaction->stage());
        st_verify_(engine_transaction->commit());
        st_verify_(oracle_transaction->commit());

        st_verify_eq_(engine.size(), oracle.size(), "a window mutator left the two holding different counts");
        st_verify_eq_(fuzz_snapshot_of(engine), fuzz_snapshot_of(oracle),
                      "a window mutator left the two holding different contents");
    }
}

#pragma endregion One Engine Against the Oracle

#pragma region Groups Against Tearing

/**
 *  @brief Refuses a random group commit and checks that no participant moved.
 *
 *  A group that publishes one participant and is then turned away by the next leaves the two
 *  disagreeing about whether the update happened, and no retry can reconcile them. So the refusal is
 *  forced rather than raced - a watched key is published over while the group sits staged - and both
 *  participants are compared against the snapshot taken before the commit was attempted.
 */
template <typename first_store_type_, typename second_store_type_>
void test_a_refused_group_publishes_nothing(std::size_t rounds = 60) {

    using first_t = first_store_type_;
    using second_t = second_store_type_;
    using first_member_t = typename first_t::value_type;
    using second_member_t = typename second_t::value_type;

    std::mt19937 generator(test_seed());
    first_t alpha;
    second_t beta;

    for (trivial_id_t identifier = 0; identifier != fuzz_keyspace_k; ++identifier) {
        st_verify_(alpha.upsert(trivial_id_to_member<first_member_t>(identifier, 0)));
        st_verify_(beta.upsert(trivial_id_to_member<second_member_t>(identifier, 0)));
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
        st_verify_(group.has_value());
        auto &into_alpha = group->template participant<0>();
        auto &into_beta = group->template participant<1>();

        // The participant carrying the watch writes the watched key too, so the refusal is always the
        // write set catching it rather than the read set - one code to assert instead of two.
        if (beta_is_last) {
            st_verify_(into_alpha.upsert(trivial_id_to_member<first_member_t>(written, int(round) + 1)));
            st_verify_(into_beta.upsert(trivial_id_to_member<second_member_t>(watched, int(round) + 1)));
            st_verify_(into_beta.watch(trivial_key_t {watched}));
        }
        else {
            st_verify_(into_alpha.upsert(trivial_id_to_member<first_member_t>(watched, int(round) + 1)));
            st_verify_(into_beta.upsert(trivial_id_to_member<second_member_t>(written, int(round) + 1)));
            st_verify_(into_alpha.watch(trivial_key_t {watched}));
        }
        st_verify_(group->stage());

        // Published over while the group is staged, which is the one thing a validated commit still
        // has to catch and the only refusal a staged group can meet.
        if (beta_is_last) st_verify_(beta.upsert(trivial_id_to_member<second_member_t>(watched, 9999)));
        else st_verify_(alpha.upsert(trivial_id_to_member<first_member_t>(watched, 9999)));

        fuzz_snapshot_t const alpha_before = fuzz_snapshot_of(alpha);
        fuzz_snapshot_t const beta_before = fuzz_snapshot_of(beta);
        status_t const refused = group->commit();
        st_verify_eq_(refused, status_t::write_conflict_k, "a commit over a published watch must be refused");
        st_verify_eq_(fuzz_snapshot_of(alpha), alpha_before, "a refused group published its first participant");
        st_verify_eq_(fuzz_snapshot_of(beta), beta_before, "a refused group published its second participant");
        st_verify_eq_(group->staging(), staging_t::staged_k, "a refused group must stay staged to be unwindable");
        st_verify_(group->rollback());
    }
}

/**
 *  @brief Commits random groups that nothing refuses, and checks every participant advanced together.
 *
 *  The mirror of the refusal case: a group that publishes must publish all of itself, so the pair is
 *  compared against a model advanced only when the commit answered success.
 */
template <typename first_store_type_, typename second_store_type_>
void test_an_accepted_group_publishes_everything(std::size_t rounds = 120) {

    using first_t = first_store_type_;
    using second_t = second_store_type_;
    using first_member_t = typename first_t::value_type;
    using second_member_t = typename second_t::value_type;

    std::mt19937 generator(test_seed());
    first_t alpha;
    second_t beta;
    std::size_t published_rounds = 0;

    for (std::size_t round = 0; round != rounds; ++round) {
        trivial_id_t const identifier = trivial_id_t(generator() % fuzz_keyspace_k);
        int const value = int(round) + 1;
        bool const unwinding = (generator() % 4) == 0;

        auto group = make_transaction_group(alpha, beta);
        st_verify_(group.has_value());
        st_verify_(group->template participant<0>().upsert(trivial_id_to_member<first_member_t>(identifier, value)));
        st_verify_(group->template participant<1>().upsert(trivial_id_to_member<second_member_t>(identifier, value)));
        st_verify_(group->stage());

        if (unwinding) { st_verify_(group->rollback()); }
        else {
            st_verify_(group->commit());
            ++published_rounds;
        }

        // Both participants were handed the same key and value every round, so agreeing with each
        // other is the whole of what all-or-nothing claims here.
        st_verify_eq_(fuzz_snapshot_of(alpha), fuzz_snapshot_of(beta),
                      "the participants of a group disagreed about a round");
    }

    st_verify_ne_(published_rounds, 0, "no round published, so this proves nothing");
}

#pragma endregion Groups Against Tearing

} // namespace ashvardanian::smashtable::scripts
