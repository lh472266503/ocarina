//
// Created by Zero on 27/07/2022.
//

#pragma once

#include "core/stl.h"
#include "dsl/struct.h"
#include "core/basic_traits.h"
#include "dsl/operators.h"

namespace ocarina {
struct Frame {
    float3 x, y, z;
};
}// namespace ocarina

OC_STRUCT(ocarina::Frame, x, y, z){

};
OC_MAKE_STRUCT_VAR(ocarina::Frame, x, y, z)