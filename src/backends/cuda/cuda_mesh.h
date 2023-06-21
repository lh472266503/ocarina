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
    Buffer<std::byte> _blas_buffer{};
    OptixBuildInput _build_input{};

public:
    CUDAMesh(CUDADevice *device, const MeshParams &params)
        : _device(device), _params(params) {
        init_build_input();
    }
    ~CUDAMesh();
    void init_build_input() noexcept;
    [[nodiscard]] handle_ty blas_handle() const noexcept override { return _blas_handle; }
    [[nodiscard]] uint vertex_num() const noexcept override { return _params.vert_num; }
    [[nodiscard]] uint triangle_num() const noexcept override { return _params.tri_num; }
    void build_bvh(const BLASBuildCommand *cmd) noexcept;
};
}// namespace ocarina