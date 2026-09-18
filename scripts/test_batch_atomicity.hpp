/**
 *  @file scripts/test_batch_atomicity.hpp
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
 *  @section batch_atomicity_contracts The Three Contracts
 *
 *  A range modifier carries the contract its name promises, and the three differ only in what they
 *  do with a key the container already holds: @c insert refuses the whole batch over one,
 *  @c insert_if_missing leaves it alone and takes the rest, and @c upsert overwrites it.
 */
#pragma once
#include <cstddef> // `std::size_t`

#include <vector> // `std::vector`

#include "test.hpp"
#include "test_basic.hpp"

namespace ashvardanian::smashtable::scripts {

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

/**
 *  @brief How many times an unrefused batch asks the allocator, which is the width of the sweep.
 *  @param[in] make_container Hands back a container allocating through the ledger it is given.
 */
template <typename container_type_, typename factory_type_>
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
        st_verify_(container.insert_if_missing(offered.begin(), offered.end()));
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
template <typename container_type_, typename factory_type_>
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
        status_t const reported = container.insert_if_missing(offered.begin(), offered.end());
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
 *  @brief Refuses the allocator at every point the batch asks, covering its first, its last and the rest.
 *
 *  A batch that fits the room the container already holds asks for nothing, and a container growing
 *  in large steps holds a lot of it - so the size is grown until the batch reaches the allocator
 *  rather than assumed to. Refusing nothing would otherwise read as a container that cannot fail.
 */
template <typename container_type_, typename factory_type_>
void test_batch_is_all_or_nothing(factory_type_ &&make_container, std::size_t seeded = 2, std::size_t batch_size = 8) {

    std::size_t chances = 0;
    while ((chances = batch_allocation_chances<container_type_>(make_container, seeded, batch_size)) == 0 &&
           batch_size <= 4096)
        batch_size *= 4;

    st_verify_gt_(chances, 0u, "no batch this container will take ever reaches the allocator");
    st_verify_le_(chances, 8u * batch_size, "a batch paying more than eight allocations an element has a cost defect");
    for (std::size_t refusal_index = 0; refusal_index != chances; ++refusal_index)
        verify_batch_refuses_whole<container_type_>(make_container, seeded, batch_size, refusal_index);
}

/** Tests that the three range modifiers keep the three contracts their names promise. */
template <typename container_type_>
void test_batch_contracts() {

    using container_t = container_type_;
    using member_t = typename container_t::value_type;
    std::vector<member_t> const offered = batch_of<member_t>(4);
    member_t const shared = member_t(static_cast<trivial_id_t>(110)); // ? The second member of the batch

    container_t lenient, strict, overwriting;
    for (container_t *container : {&lenient, &strict, &overwriting}) st_verify_(container->upsert(member_t(shared)));

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
}

#pragma endregion Batch Atomicity Test Templates

} // namespace ashvardanian::smashtable::scripts
