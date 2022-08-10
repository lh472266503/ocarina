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
    CUDADevice *_device;
    MeshParams _params;
    OptixTraversableHandle _BLAS_handle{};
    OptixDeviceContext _optix_device_context{};
    OptixBuildInput _build_input{};
    OptixPipeline _optix_pipeline{};

public:
    CUDAMesh(CUDADevice *device, const MeshParams &params)
        :  _device(device), _params(params) {
        init_build_input();
    }
    void init_build_input() noexcept;
    void build_bvh(const MeshBuildCommand *cmd) noexcept;
};
}// namespace ocarina