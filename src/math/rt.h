//
// Created by Zero on 27/07/2022.
//

#pragma once

#include "core/basic_traits.h"
#include "dsl/operators.h"
#include "dsl/struct.h"

namespace ocarina {

struct alignas(16) Hit {
    uint inst_id{uint(-1)};
    uint prim_id{uint(-1)};
    float2 bary;
};

}// namespace ocarina

OC_STRUCT(ocarina::Hit, inst_id, prim_id, bary){
    void init(){
        inst_id = uint(-1);
    }

    [[nodiscard]] auto is_miss() noexcept {
        return make_expr(inst_id == uint(-1));
    }
};

namespace ocarina {
struct Triangle {
public:
    uint i, j, k;
    Triangle(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triangle() = default;
};
}// namespace ocarina
OC_STRUCT(ocarina::Triangle, i, j, k){};


