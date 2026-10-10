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
        st_verify_(container.upsert(member_t(static_cast<trivial_id_t>(index + 1))));
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
        std::size_t const paid = ledger.granted_count;
        st_verify_(apply_batch<verb_>(container, offered.begin(), offered.end()));
        chances = ledger.granted_count - paid;
    }
    ledger.verify_balanced();
    return chances;
}

/**
 *  @brief One point of the sweep: the batch is refused its @p refusal_index -th request for memory.
 *
 *  A point of a sweep rather than a test of its own, so the index that failed prints itself in the
 *  abort message instead of hiding inside a loop the oracle cannot see.
 */
template <typename container_type_, batch_verb_t verb_, typename factory_type_>
void verify_batch_refuses_whole(factory_type_ &&make_container, std::size_t seeded, std::size_t batch_size,
                                std::size_t refusal_index) {

    using member_t = typename container_type_::value_type;
    allocation_ledger_t ledger;
    {
        container_type_ container = make_container(ledger);
        seed_for_batch(container, seeded);
        std::vector<member_t> const offered = batch_of<member_t>(batch_size);
        std::size_t const held = container.size();
        std::size_t const refused_before = ledger.refused_count;

        // Arming and disarming bracket the one call, so nothing the oracle itself needs is refused.
        ledger.allow(refusal_index);
        status_t const reported = apply_batch<verb_>(container, offered.begin(), offered.end());
        ledger.allow(unlimited_budget_k);

        st_verify_eq_(reported, status_t::out_of_memory_heap_k,
                      "a batch the allocator turned away reports the heap, never a key");
        st_verify_gt_(ledger.refused_count, refused_before,
                      "an index past what the batch asks for would pass without refusing anything");
        st_verify_eq_(container.size(), held, "a refused batch leaves the container exactly as it was");
        for (member_t const &member : offered)
            st_verify_eq_(container.contains(member), false, "no member of a refused batch may be found afterwards");
        for (std::size_t index = 0; index != seeded; ++index)
            st_verify_eq_(container.contains(member_t(static_cast<trivial_id_t>(index + 1))), true,
                          "and the members it was seeded with are untouched");
    }
    ledger.verify_balanced(); // ? Whatever the partial build took must all have come back
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

    st_verify_gt_(chances, 0u, "no batch this container will take ever reaches the allocator");
    st_verify_le_(chances, 8u * batch_size, "a batch paying more than eight allocations an element has a cost defect");
    for (std::size_t refusal_index = 0; refusal_index != chances; ++refusal_index)
        verify_batch_refuses_whole<container_type_, verb_>(make_container, seeded, batch_size, refusal_index);
}

/** Sweeps the allocator under every verb, since each absorbs the staged range its own way. */
template <typename container_type_, typename factory_type_>
void test_batch_is_all_or_nothing(factory_type_ &&make_container, std::size_t seeded = 2, std::size_t batch_size = 8) {
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
void test_batch_refuses_copies_whole(std::size_t seeded = 2, std::size_t batch_size = 4) {

    using member_t = typename container_type_::value_type;
    auto const make_member = [](trivial_id_t identifier) noexcept {
        expected<member_t> made = member_t::make(identifier);
        st_verify_(bool(made));
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
        st_verify_(counting.insert_if_missing(offered.begin(), offered.end()));
        chances = unlimited_budget_k - 1 - copy_budget_t::copies_till_fail;
        copy_budget_t::reset();
    }
    st_verify_gt_(chances, 0u, "a batch over lvalues duplicates the elements it takes");

    for (std::size_t refusal_index = 0; refusal_index != chances; ++refusal_index) {
        container_type_ container;
        for (std::size_t index = 0; index != seeded; ++index)
            st_verify_(container.upsert(make_member(static_cast<trivial_id_t>(index + 1))));
        std::size_t const held = container.size();

        copy_budget_t::allow(refusal_index);
        status_t const reported = container.insert_if_missing(offered.begin(), offered.end());
        std::size_t const refusals = copy_budget_t::refusals_count;
        copy_budget_t::reset();

        st_verify_eq_(reported, status_t::out_of_memory_heap_k, "a refused copy reports its own cause");
        st_verify_gt_(refusals, 0u, "an index past what the batch copies would pass without refusing anything");
        st_verify_eq_(container.size(), held, "a batch refused a copy leaves the container exactly as it was");
        for (member_t const &member : offered)
            st_verify_eq_(container.contains(member), false, "no member of a refused batch may be found afterwards");
    }
}

/** Tests that the range modifiers keep the contracts their names promise. */
template <typename container_type_>
void test_batch_contracts() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    std::vector<member_t> const offered = batch_of<member_t>(4);
    member_t const shared = member_t(static_cast<trivial_id_t>(110)); // ? The second member of the batch

    container_t lenient, strict, overwriting, updating;
    for (container_t *container : {&lenient, &strict, &overwriting, &updating})
        st_verify_(container->upsert(member_t(shared)));

    st_verify_(lenient.insert_if_missing(offered.begin(), offered.end()));
    st_verify_eq_(lenient.size(), offered.size(), "a key already here is skipped and the rest still land");

    st_verify_eq_(strict.insert(offered.begin(), offered.end()), status_t::key_already_exists_k,
                  "a key already here refuses the whole batch");
    st_verify_eq_(strict.size(), 1u, "and none of that batch is stored");
    for (member_t const &member : offered)
        if (!(member == shared))
            st_verify_eq_(strict.contains(member), false, "not one member of a refused batch is findable");

    st_verify_(overwriting.upsert(offered.begin(), offered.end()));
    st_verify_eq_(overwriting.size(), offered.size(), "an upsert takes every member, incumbent or not");

    // A range update checks every key before the absorb, which would write the missing ones too
    if constexpr (requires(container_t &probe) { probe.update(offered.begin(), offered.end()); }) {
        st_verify_eq_(updating.update(offered.begin(), offered.end()), status_t::key_not_found_k,
                      "a key missing here refuses the whole update");
        st_verify_eq_(updating.size(), 1u, "and none of that batch is stored");
    }
}

#pragma endregion Batch Atomicity Test Templates

} // namespace ashvardanian::smashtable::test
