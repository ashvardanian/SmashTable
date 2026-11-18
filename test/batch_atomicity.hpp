/**
 *  @file test/batch_atomicity.hpp
 *  @author Ash Vardanian
 *  @date September 17, 2026
 *  @brief The suite every container with a range modifier answers: a batch refused part-way leaves
 *      nothing of itself behind, whichever element refused it.
 *
 *  @section batch_atomicity_sweep Sweeping The Refusal
 *
 *  Failing at the first element, at the last, and in the middle are three different code paths, so
 *  nothing here picks an index. A probe run with nothing armed records how many times the batch
 *  asks the allocator, and the sweep refuses each of those requests in turn, which covers all three
 *  ends by construction rather than by choosing them. Sweeping requests rather than elements is
 *  deliberate: how many allocations an element costs is the container's business, and a sweep over
 *  elements would either miss chances or arm indices that never arrive.
 *
 *  Every verb is swept, since each absorbs the staged range its own way, and an absorb that asked
 *  the allocator again once elements had started moving would leave a prefix behind at the request
 *  that refused. An element refusing its own copy is the other cause, swept the same way over the
 *  copies a batch makes; both land while the range is still staged beside the container.
 *
 *  @section batch_atomicity_contracts The Three Contracts
 *
 *  A range modifier carries the contract its name promises, and the three differ only in what they
 *  do with a key the container already holds: @c insert refuses the whole batch over one,
 *  @c insert_if_missing leaves it alone and takes the rest, and @c upsert overwrites it. A refusing
 *  verb checks the keys before the absorb, so the batch it refuses never reached the container.
 */
#pragma once
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::uint8_t`

#include <map>    // `std::map`
#include <vector> // `std::vector`

#include "harness.hpp"
#include "fixtures.hpp"

namespace ashvardanian::smashtable::test {

#pragma region Batch Atomicity Test Templates

/** The members a batch offers, drawn well clear of whatever the container was seeded with. */
template <typename member_type_>
[[nodiscard]] std::vector<member_type_> batch_of(std::size_t count) {
    std::vector<member_type_> members;
    for (std::size_t index = 0; index != count; ++index)
        members.push_back(member_type_(static_cast<trivial_id_t>(100 + index * 10)));
    return members;
}

/** Seeds @p container with a couple of members the batch must never disturb. */
template <typename container_type_>
void seed_for_batch(container_type_ &container, std::size_t count) {
    using member_t = typename container_type_::value_type;
    for (std::size_t index = 0; index != count; ++index)
        expect_true(succeeded(container.upsert(member_t(static_cast<trivial_id_t>(index + 1)))));
}

/** The three range verbs every batching container spells, each absorbing its staged range its own
 *  way. */
enum class batch_verb_t : std::uint8_t { insert_if_missing_k, insert_k, upsert_k };

/** Runs @p verb_ over [ @p first, @p last ) into @p container. */
template <batch_verb_t verb_, typename container_type_, typename iterator_type_>
[[nodiscard]] status_t apply_batch(container_type_ &container, iterator_type_ first, iterator_type_ last) noexcept {
    if constexpr (verb_ == batch_verb_t::insert_k) return container.insert(first, last);
    else if constexpr (verb_ == batch_verb_t::upsert_k) return container.upsert(first, last);
    else return container.insert_if_missing(first, last);
}

/**
 *  @brief How many times an unrefused batch asks the allocator, which is the width of the sweep.
 *  @param[in] make_container Hands back a container allocating through the ledger it is given.
 */
template <typename container_type_, batch_verb_t verb_, typename factory_type_>
[[nodiscard]] std::size_t batch_allocation_chances(factory_type_ &&make_container, std::size_t seeded,
                                                   std::size_t batch_size) {
    using member_t = typename container_type_::value_type;
    allocation_ledger_t ledger;
    std::size_t chances = 0;
    {
        container_type_ container = make_container(ledger);
        seed_for_batch(container, seeded);
        std::vector<member_t> const offered = batch_of<member_t>(batch_size);
        std::size_t const paid = ledger.allocations;
        expect_equal(apply_batch<verb_>(container, offered.begin(), offered.end()), success_k);
        chances = ledger.allocations - paid;
    }
    ledger.check_balanced();
    return chances;
}

/**
 *  @brief One point of the sweep: the batch is refused its @p refusal_index -th request for memory.
 *
 *  A point of a sweep rather than a test of its own, so the index that failed prints itself beneath
 *  the failed check instead of hiding inside a loop the oracle cannot see.
 */
template <typename container_type_, batch_verb_t verb_, typename factory_type_>
void check_batch_refuses_whole(factory_type_ &&make_container, std::size_t seeded, std::size_t batch_size,
                               std::size_t refusal_index) {

    using member_t = typename container_type_::value_type;
    allocation_ledger_t ledger;
    {
        container_type_ container = make_container(ledger);
        seed_for_batch(container, seeded);
        std::vector<member_t> const offered = batch_of<member_t>(batch_size);
        std::size_t const held = container.size();
        std::size_t const refused_before = ledger.refusals;

        // Arming and disarming bracket the one call, so nothing the oracle itself needs is refused.
        ledger.allow(refusal_index);
        status_t const reported = apply_batch<verb_>(container, offered.begin(), offered.end());
        ledger.allow(unlimited_budget_k);

        expect_equal(reported, status_t::out_of_memory_heap_k,
                     "a batch the allocator turned away reports the heap, never a key");
        expect_true(ledger.refusals > refused_before,
                    "an index past what the batch asks for would pass without refusing anything");
        expect_equal(container.size(), held, "a refused batch leaves the container exactly as it was");
        for (member_t const &member : offered)
            expect_equal(container.contains(member), false, "no member of a refused batch may be found afterwards");
        for (std::size_t index = 0; index != seeded; ++index)
            expect_equal(container.contains(member_t(static_cast<trivial_id_t>(index + 1))), true,
                         "and the members it was seeded with are untouched");
    }
    ledger.check_balanced(); // ? Whatever the partial build took must all have come back
}

/**
 *  @brief Refuses the allocator at every point @p verb_ asks: its first, its last and the rest.
 *
 *  A batch that fits the room the container already holds asks for nothing, and a container growing
 *  in large steps holds a lot of it - so the size is grown until the batch reaches the allocator
 *  rather than assumed to. Refusing nothing would otherwise read as a container that cannot fail.
 */
template <typename container_type_, batch_verb_t verb_, typename factory_type_>
void sweep_batch_allocations(factory_type_ &&make_container, std::size_t seeded, std::size_t batch_size) {

    std::size_t chances = 0;
    while ((chances = batch_allocation_chances<container_type_, verb_>(make_container, seeded, batch_size)) == 0 &&
           batch_size <= 4096)
        batch_size *= 4;

    expect_true(chances > 0u, "no batch this container will take ever reaches the allocator");
    expect_true(chances <= 8u * batch_size, "a batch paying more than eight allocations an element has a cost defect");
    for (std::size_t refusal_index = 0; refusal_index != chances; ++refusal_index)
        check_batch_refuses_whole<container_type_, verb_>(make_container, seeded, batch_size, refusal_index);
}

/** Sweeps the allocator under every verb, since each absorbs the staged range its own way. */
template <typename container_type_, typename factory_type_>
void check_batch_is_all_or_nothing(factory_type_ &&make_container, std::size_t seeded = 2, std::size_t batch_size = 8) {
    sweep_batch_allocations<container_type_, batch_verb_t::insert_if_missing_k>(make_container, seeded, batch_size);
    sweep_batch_allocations<container_type_, batch_verb_t::insert_k>(make_container, seeded, batch_size);
    sweep_batch_allocations<container_type_, batch_verb_t::upsert_k>(make_container, seeded, batch_size);
}

/**
 *  @brief Refuses every element copy a batch makes, in turn, and finds the container as it was.
 *  @tparam container_type_ A container whose members duplicate through @c copy_budget_t.
 *
 *  The second cause a batch meets: the range hands over lvalues, so each element is duplicated as
 *  it is staged, and a copy that refuses stops the batch while the range is still staged beside
 *  the container, under whichever verb, since all three stage alike.
 */
template <typename container_type_>
void check_batch_refuses_copies_whole(std::size_t seeded = 2, std::size_t batch_size = 4) {

    using member_t = typename container_type_::value_type;
    auto const make_member = [](trivial_id_t identifier) noexcept {
        expected<member_t> made = member_t::make(identifier);
        expect_true(bool(made));
        return *std::move(made);
    };
    std::vector<member_t> offered;
    for (std::size_t index = 0; index != batch_size; ++index)
        offered.push_back(make_member(static_cast<trivial_id_t>(100 + index * 10)));

    // How many copies an unrefused batch makes, which is the width of the sweep
    std::size_t chances = 0;
    {
        container_type_ counting;
        copy_budget_t::allow(unlimited_budget_k - 1);
        expect_equal(counting.insert_if_missing(offered.begin(), offered.end()), success_k);
        chances = unlimited_budget_k - 1 - copy_budget_t::copies_till_fail;
        copy_budget_t::reset();
    }
    expect_true(chances > 0u, "a batch over lvalues duplicates the elements it takes");

    for (std::size_t refusal_index = 0; refusal_index != chances; ++refusal_index) {
        container_type_ container;
        for (std::size_t index = 0; index != seeded; ++index)
            expect_true(succeeded(container.upsert(make_member(static_cast<trivial_id_t>(index + 1)))));
        std::size_t const held = container.size();

        copy_budget_t::allow(refusal_index);
        status_t const reported = container.insert_if_missing(offered.begin(), offered.end());
        std::size_t const refusals = copy_budget_t::refusals_count;
        copy_budget_t::reset();

        expect_equal(reported, status_t::out_of_memory_heap_k, "a refused copy reports its own cause");
        expect_true(refusals > 0u, "an index past what the batch copies would pass without refusing anything");
        expect_equal(container.size(), held, "a batch refused a copy leaves the container exactly as it was");
        for (member_t const &member : offered)
            expect_equal(container.contains(member), false, "no member of a refused batch may be found afterwards");
    }
}

/** Tests that the range modifiers keep the contracts their names promise. */
template <typename container_type_>
void check_batch_contracts() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    std::vector<member_t> const offered = batch_of<member_t>(4);
    member_t const shared = member_t(static_cast<trivial_id_t>(110)); // ? The second member of the batch

    container_t lenient, strict, overwriting, updating;
    for (container_t *container : {&lenient, &strict, &overwriting, &updating})
        expect_true(succeeded(container->upsert(member_t(shared))));

    expect_equal(lenient.insert_if_missing(offered.begin(), offered.end()), success_k);
    expect_equal(lenient.size(), offered.size(), "a key already here is skipped and the rest still land");

    expect_equal(strict.insert(offered.begin(), offered.end()), status_t::key_already_exists_k,
                 "a key already here refuses the whole batch");
    expect_equal(strict.size(), 1u, "and none of that batch is stored");
    for (member_t const &member : offered)
        if (!(member == shared))
            expect_equal(strict.contains(member), false, "not one member of a refused batch is findable");

    expect_equal(overwriting.upsert(offered.begin(), offered.end()), success_k);
    expect_equal(overwriting.size(), offered.size(), "an upsert takes every member, incumbent or not");

    // A range update checks every key before the absorb, which would write the missing ones too
    if constexpr (requires(container_t &probe) { probe.update(offered.begin(), offered.end()); }) {
        expect_equal(updating.update(offered.begin(), offered.end()), status_t::key_not_found_k,
                     "a key missing here refuses the whole update");
        expect_equal(updating.size(), 1u, "and none of that batch is stored");
    }
}

/** The verbs of a reused transaction that may ask the allocator, each refused in turn below. */
enum class refused_verb_t : std::uint8_t { write_k, stage_k, rollback_k };

/** Whether @p store_type_ stages by relinking its transactions' own nodes, so that staging and
 *  rolling back ask the allocator for nothing and only a write can be refused. */
template <typename store_type_>
constexpr bool relinks_staged_nodes = requires { requires store_type_::relinks_staged_nodes_k; };

/**
 *  @brief Refuses every request @p verb_ makes on a transaction reused after its own commit.
 *  @tparam store_type_ A transactional map allocating through an @c allocation_ledger_t.
 *
 *  The transaction has already published key 1 and now writes it again beside a new key 3. A
 *  refused write of key 3 leaves the transaction as it was, so writing it again lands. A refused
 *  @c stage leaves the store as the first commit left it and the writes still pending, so the
 *  retry publishes both. A refused @c rollback has lost a write, so the transaction comes back
 *  empty, and a commit retried after it publishes nothing, least of all key 3 without its value.
 *  A store relinking staged nodes has nothing to refuse in the last two.
 */
template <typename store_type_, refused_verb_t verb_>
void sweep_reused_transaction_refusals() {

    using member_t = typename store_type_::value_type;
    using model_t = std::map<trivial_id_t, int>;
    model_t const first_commit {{1, 20}, {2, 1}};
    model_t const both_commits {{1, 30}, {2, 1}, {3, 30}};
    auto const expect_holds = [](store_type_ &store, model_t const &wanted, char const *when) {
        expect_equal(store.size(), wanted.size(), "{}", when);
        for (auto const &[identifier, mapped] : wanted) {
            auto const held = store.find_copy(trivial_id_to_key<member_t>(identifier));
            if (expect_true(held.has_value(), "{}", when)) expect_equal(held->mapped, mapped, "{}", when);
        }
    };

    std::size_t refused_requests = 0;
    for (std::size_t refusal_index = 0;; ++refusal_index) {
        allocation_ledger_t ledger;
        std::size_t refusals = 0;
        {
            store_type_ store {typename store_type_::allocator_t(ledger)};
            expect_equal(store.upsert(trivial_id_to_member<member_t>(1, 1)), success_k);
            expect_equal(store.upsert(trivial_id_to_member<member_t>(2, 1)), success_k);

            auto transaction = store.transaction();
            if (!expect_equal(transaction.status(), success_k)) return;
            expect_equal(transaction->upsert(trivial_id_to_member<member_t>(1, 20)), success_k);
            expect_equal(transaction->stage(), success_k);
            expect_equal(transaction->commit(), success_k);

            expect_equal(transaction->upsert(trivial_id_to_member<member_t>(1, 30)), success_k);
            if constexpr (verb_ != refused_verb_t::write_k)
                expect_equal(transaction->upsert(trivial_id_to_member<member_t>(3, 30)), success_k);
            if constexpr (verb_ == refused_verb_t::rollback_k) expect_equal(transaction->stage(), success_k);

            // Armed around the one call, so nothing the oracle itself needs is refused.
            ledger.allow(refusal_index);
            status_t const reported = verb_ == refused_verb_t::write_k
                                          ? transaction->upsert(trivial_id_to_member<member_t>(3, 30))
                                      : verb_ == refused_verb_t::stage_k ? transaction->stage()
                                                                         : transaction->rollback();
            ledger.allow(unlimited_budget_k);
            refusals = ledger.refusals;

            if (failed(reported)) expect_holds(store, first_commit, "after the refused verb");
            if (verb_ == refused_verb_t::write_k && failed(reported))
                expect_equal(transaction->upsert(trivial_id_to_member<member_t>(3, 30)), success_k);
            bool const emptied = verb_ == refused_verb_t::rollback_k && failed(reported);
            if (verb_ != refused_verb_t::stage_k || failed(reported)) expect_equal(transaction->stage(), success_k);
            expect_equal(transaction->commit(), success_k);
            expect_holds(store, emptied ? first_commit : both_commits, "after the retried commit");
        }
        ledger.check_balanced();
        if (refusals == 0) break; // ? An index past what the verb asks for refuses nothing
        refused_requests += refusals;
    }
    if (relinks_staged_nodes<store_type_> && verb_ != refused_verb_t::write_k)
        expect_equal(refused_requests, 0u, "relinking a staged node, either way, asks the allocator for nothing");
    else expect_true(refused_requests > 0u, "the verb never reached the allocator, so nothing was refused");
}

/** Sweeps the allocator under every verb of a reused transaction that can lose memory. */
template <typename store_type_>
void check_refusals_spare_a_reused_transaction() {
    sweep_reused_transaction_refusals<store_type_, refused_verb_t::write_k>();
    sweep_reused_transaction_refusals<store_type_, refused_verb_t::stage_k>();
    sweep_reused_transaction_refusals<store_type_, refused_verb_t::rollback_k>();
}

/**
 *  @brief Commits and aborts through one reused transaction and finds the allocator untouched.
 *  @tparam store_type_ A transactional map allocating through an @c allocation_ledger_t.
 *
 *  Eight commits over the same keys warm both sides: each prune empties the versions the commit
 *  masked, and the transaction borrows those nodes for its next writes. From then on neither an
 *  overwriting commit nor a staged write that is reset asks the allocator for anything.
 */
template <typename store_type_>
void check_warm_commits_ask_the_allocator_for_nothing() {

    using member_t = typename store_type_::value_type;
    constexpr trivial_id_t written_keys_k = 4;
    constexpr int warm_rounds_k = 8, measured_rounds_k = 64;
    allocation_ledger_t ledger;
    {
        store_type_ store {typename store_type_::allocator_t(ledger)};
        auto transaction = store.transaction();
        if (!expect_equal(transaction.status(), success_k)) return;
        auto const stage_overwrites = [&](int round) {
            for (trivial_id_t identifier = 1; identifier <= written_keys_k; ++identifier)
                expect_equal(transaction->upsert(trivial_id_to_member<member_t>(identifier, round)), success_k);
            expect_equal(transaction->stage(), success_k);
        };

        int round = 0;
        for (; round != warm_rounds_k; ++round) {
            stage_overwrites(round);
            expect_equal(transaction->commit(), success_k);
        }
        std::size_t const allocated = ledger.allocations, freed = ledger.deallocations;

        for (; round != warm_rounds_k + measured_rounds_k; ++round) {
            stage_overwrites(round);
            expect_equal(transaction->commit(), success_k);
        }
        expect_equal(ledger.allocations, allocated, "a warm commit builds in the nodes its last prune emptied");
        expect_equal(ledger.deallocations, freed, "and keeps the ones this prune empties");

        for (int aborted = 0; aborted != measured_rounds_k; ++aborted) {
            stage_overwrites(round + aborted);
            expect_equal(transaction->reset(), success_k);
        }
        expect_equal(ledger.allocations, allocated, "a staged write that is reset keeps its node for the next");
        expect_equal(ledger.deallocations, freed, "and returns none");

        expect_equal(store.size(), written_keys_k);
        for (trivial_id_t identifier = 1; identifier <= written_keys_k; ++identifier) {
            auto const held = store.find_copy(trivial_id_to_key<member_t>(identifier));
            if (expect_true(held.has_value())) expect_equal(held->mapped, round - 1, "the last commit still reads");
        }
    }
    ledger.check_balanced();
}

#pragma endregion Batch Atomicity Test Templates

} // namespace ashvardanian::smashtable::test
