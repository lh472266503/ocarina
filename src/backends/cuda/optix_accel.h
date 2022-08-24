//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "core/stl.h"
#include "rhi/rtx/accel.h"
#include "util.h"

namespace ocarina {
class CUDADevice;
class CUDACommandVisitor;
class OptixAccel : public Accel::Impl {
private:
    Buffer<std::byte> _tlas_buffer;
    OptixTraversableHandle _tlas_handle{};
    ocarina::vector<const Mesh::Impl *> _meshes;
    ocarina::vector<float4x4> _transforms;
    CUDADevice *_device;
    Buffer<OptixInstance> _instances{};
    
public:
    explicit OptixAccel(CUDADevice *device) : _device(device) {}

    void add_mesh(const Mesh::Impl *mesh, float4x4 mat) noexcept override {
        _meshes.push_back(mesh);
        _transforms.push_back(mat);
    }

    void build_bvh(CUDACommandVisitor *visitor) noexcept;
    [[nodiscard]] size_t data_size() const noexcept override;
    [[nodiscard]] size_t data_alignment() const noexcept override;
    [[nodiscard]] handle_ty handle() const noexcept override { return _tlas_handle; }
    [[nodiscard]] const void *handle_ptr() const noexcept override { return &_tlas_handle; }
};
}// namespace ocarina