//
// Created by Zero on 09/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/geometries/mesh.h"
#include "util.h"


namespace ocarina {
class CUDAMesh : public Mesh::Impl {
private:
    CUdeviceptr _v_handle{};
    CUdeviceptr _t_handle{};

public:
    CUDAMesh(handle_ty v_handle, handle_ty t_handle,
             uint v_stride, uint tri_num,AccelUsageTag usage_tag)
        : Mesh::Impl(tri_num, v_stride,usage_tag),
          _v_handle(v_handle), _t_handle(t_handle) {}
};
}// namespace ocarina