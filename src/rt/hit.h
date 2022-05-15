//
// Created by Zero on 15/05/2022.
//

#pragma once
#include "core/basic_traits.h"
#include "ast/type_registry.h"
#include "dsl/struct.h"

namespace katana {

struct alignas(16) Hit {
    uint inst_id{};
    uint prim_id{};
    float2 bary;
};

KTN_STRUCT(katana::Hit, inst_id, prim_id, bary)

}// namespace katana