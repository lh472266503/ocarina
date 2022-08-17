//
// Created by Zero on 2022/8/8.
//

#pragma once

#include "core/stl.h"
#include "rhi/rtx/accel.h"
#include "util.h"

namespace ocarina {
class CUDADevice;
class OptixAccel : public Accel::Impl {
private:
    ocarina::unique_ptr<Buffer<std::byte>> _tlas_buffer;
    OptixTraversableHandle _tlas_handle;
    CUDADevice *_device;
    
public:
    explicit OptixAccel(CUDADevice *device) : _device(device) {}

    void add_mesh(Mesh::Impl *mesh, float4x4 mat) noexcept override {

    }
};
}// namespace ocarina