//
// Created by Zero on 09/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/geometries/mesh.h"
#include "util.h"

namespace ocarina {
class CUDADevice;
class CUDAMesh : public Mesh::Impl {
private:
    CUDADevice *_device{};
    CUdeviceptr _v_handle{};
    CUdeviceptr _t_handle{};
    OptixTraversableHandle _BLAS_handle{};
    OptixDeviceContext _optix_device_context{};
    OptixBuildInput _build_input{};
    OptixPipeline _optix_pipeline{};

public:
    CUDAMesh(CUDADevice *device, handle_ty v_handle, handle_ty t_handle, handle_ty vert_num,
             uint v_stride, uint tri_num, AccelUsageTag usage_tag)
        : Mesh::Impl(vert_num, tri_num, v_stride, usage_tag), _device(device),
          _v_handle(v_handle), _t_handle(t_handle) {
        init_build_input();
    }
    void init_build_input() noexcept;
    void build_bvh(const MeshBuildCommand *cmd) noexcept;
};
}// namespace ocarina