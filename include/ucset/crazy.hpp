#pragma once
#include "status.hpp"

namespace unum::ucset {

template <typename collection_at>
class crazy_gt {};

template <typename collection_at>
using stl_compatible_gt = crazy_gt<collection_at>;

} // namespace unum::ucset