//
// Created by Zero on 27/07/2022.
//

#pragma once

#include "core/stl.h"
#include "dsl/struct.h"
#include "core/basic_traits.h"
#include "dsl/operators.h"

namespace ocarina {

template<typename T>
void coordinate_system(const T &v1, T &v2, T &v3) noexcept {
    v2 = select(abs(v1.x) > abs(v1.y),
                make_float3(-v1.z, 0.f, v1.x) / sqrt(v1.x * v1.x + v1.z * v1.z),
                make_float3(0.f, v1.z, -v1.y) / sqrt(v1.y * v1.y + v1.z * v1.z));
    v3 = cross(v1, v2);
}

struct Frame {
    float3 x, y, z;
};
}// namespace ocarina

OC_STRUCT(ocarina::Frame, x, y, z){

};
