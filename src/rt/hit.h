//
// Created by Zero on 15/05/2022.
//

#pragma once
#include "core/basic_traits.h"
#include "dsl/struct.h"

namespace ocarina {

struct alignas(16) Hit {
    uint inst_id{};
    uint prim_id{};
    float2 bary;
};

OC_STRUCT(ocarina::Hit, inst_id, prim_id, bary) {
    void print() {
        inst_id = 1.7f;
    }
};


}// namespace ocarina