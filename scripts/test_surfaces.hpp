/**
 *  @file scripts/test_surfaces.hpp
 *  @author Ash Vardanian
 *  @date September 18, 2026
 *  @brief One entry point per container, running every suite whose surface that container offers.
 *
 *  @section test_surfaces_dispatch Dispatch Rather Than Recollection
 *
 *  A suite used to reach a container because somebody typed its name into that container's file, so
 *  a container added later got whatever its author remembered and a suite added later reached only
 *  the files somebody updated. The concepts already say what a container offers, so they decide
 *  instead: a container satisfying a surface runs the suite that proves it, and one that does not is
 *  not asked to.
 *
 *  @section test_surfaces_waivers What Is Deliberately Not Covered
 *
 *  Dispatching on a concept cannot tell a surface that never applied from one that quietly vanished,
 *  so a surface offered without a suite behind it is a mistake rather than a gap. @c suite_waived_for
 *  names the exceptions, defaulting to none, the way the wrapper parity fold already names its own.
 */
#pragma once
#include <cstddef> // `std::size_t`

#include <smashtable/shared.hpp>

#include "test.hpp"
#include "test_basic.hpp"
#include "test_batch_atomicity.hpp"

namespace ashvardanian::smashtable::scripts {

#pragma region Surface Dispatch

/** Names a suite, so a waiver can be written against it by name rather than by position. */
enum class suite_t : std::uint8_t {

    /** The sweep that refuses the allocator at every point a batch asks for memory. */
    batch_atomicity_k,

    /** The three range verbs against a key already here and a key repeated inside the range. */
    batch_contracts_k,
};

/**
 *  @brief Whether @p suite_ is deliberately not run over @p container_type_, despite it offering the
 *      surface that suite proves.
 *
 *  Specialize with a docblock saying why. An unspecialized pair is covered, which is what makes a
 *  missing suite a compile-time question rather than a silence in the output.
 */
template <suite_t suite_, typename container_type_>
struct suite_waived_for : std::false_type {};

/**
 *  @brief Runs every suite whose surface @p container_type_ offers, and none that it does not.
 *  @param[in] make_container Hands back a container allocating through the ledger it is given.
 */
template <typename container_type_, typename factory_type_>
void test_every_offered_surface(factory_type_ &&make_container) {

    static_assert(tagged_collection<container_type_>,
                  "every container names its element, its key and its shape before any suite asks it anything");

    if constexpr (batches_atomically<container_type_>) {
        if constexpr (!suite_waived_for<suite_t::batch_atomicity_k, container_type_>::value)
            test_batch_is_all_or_nothing<container_type_>(make_container);
        if constexpr (!suite_waived_for<suite_t::batch_contracts_k, container_type_>::value)
            test_batch_contracts<container_type_>();
    }
}

#pragma endregion Surface Dispatch

} // namespace ashvardanian::smashtable::scripts
