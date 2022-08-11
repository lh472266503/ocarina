//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "core/stl.h"
#include "rhi/rtx/accel.h"
#include "util.h"

namespace ocarina {
class OptixAccel : public Accel::Impl {
private:

public:
    void add_mesh(Mesh::Impl *mesh, float4x4 mat) noexcept {

    }
};
}// namespace ocarina