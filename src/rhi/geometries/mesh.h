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
}// namespace ocarina
OC_STRUCT(ocarina::Triangle, i, j, k){};

namespace ocarina {

class Mesh : public RHIResource {
public:
    enum UsageTag : uint8_t {
        FAST_BUILD,
        FAST_TRACE
    };

    class Impl {
    protected:
        uint tri_count{};
        uint vertex_stride{};
        handle_ty vertices_handle{};
        handle_ty tri_handle{};
        friend class Mesh;
    };

public:
    Mesh(Device::Impl *device, handle_ty v_handle,
         handle_ty t_handle, uint t_count, uint v_stride)
        : RHIResource(device, Tag::MESH, 0) {}


};

}// namespace ocarina

