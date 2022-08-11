//
// Created by Zero on 09/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/rtx/mesh.h"
#include "util.h"

namespace ocarina {
class CUDADevice;
class CUDAMesh : public Mesh::Impl {
private:
    CUDADevice *_device;
    MeshParams _params;
    OptixTraversableHandle _blas_handle{};
    ocarina::unique_ptr<Buffer<std::byte>> _blas_buffer{};
    OptixBuildInput _build_input{};

public:
    CUDAMesh(CUDADevice *device, const MeshParams &params)
        :  _device(device), _params(params) {
        init_build_input();
    }
    ~CUDAMesh();
    void init_build_input() noexcept;
    void build_bvh(const MeshBuildCommand *cmd) noexcept;
};
}// namespace ocarina