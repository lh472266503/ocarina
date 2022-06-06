//
// Created by Zero on 15/05/2022.
//

#pragma once
#include "core/basic_traits.h"
#include "dsl/struct.h"

namespace nano {

struct alignas(16) Hit {
    uint inst_id{};
    uint prim_id{};
    float2 bary;
};

NN_STRUCT(nano::Hit, inst_id, prim_id, bary)

}// namespace nano