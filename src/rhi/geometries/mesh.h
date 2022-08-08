//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "rhi/resources/resource.h"
#include "rhi/command.h"
#include "dsl/struct.h"


namespace ocarina {

struct Triangle {
    uint32_t i, j, k;
};

class Mesh : public RHIResource {
public:
    enum UsageTag : uint8_t {
        FAST_BUILD,
        FAST_TRACE
    };

public:
};

}// namespace ocarina

OC_STRUCT(ocarina::Triangle, i, j, k){};
