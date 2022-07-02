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


//OC_MAKE_COMPUTABLE_BODY(ocarina::Hit, inst_id, prim_id, bary)


OC_STRUCT(ocarina::Hit, inst_id, prim_id, bary)

//template<>
//struct Var<ocarina::Hit> : public detail::Computable<ocarina::Hit>{
//    using this_type = ocarina::Hit;
////    Var<std::remove_cvref_t<decltype(this_type::inst_id)> inst_id{};
////    Var < std::remove_cvref_t<decltype(this_type::prim_id)> prim_id{};
////    Var < std::remove_cvref_t<decltype(this_type::bary)> bary{};
//};

}// namespace ocarina