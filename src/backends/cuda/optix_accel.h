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
    Buffer<std::byte> tlas_buffer_;
    OptixTraversableHandle tlas_handle_{};
    CUDADevice *device_;
    Buffer<OptixInstance> instances_{};
    
public:
    explicit OptixAccel(CUDADevice *device) : device_(device) {}
    void build_bvh(CUDACommandVisitor *visitor) noexcept;
    void update_bvh(CUDACommandVisitor *visitor) noexcept;
    [[nodiscard]] vector<OptixTraversableHandle> blas_handles() noexcept;
    [[nodiscard]] OptixBuildInput construct_build_input(uint instance_num) noexcept;
    [[nodiscard]] size_t data_size() const noexcept override;
    [[nodiscard]] size_t data_alignment() const noexcept override;
    void clear() noexcept override;
    [[nodiscard]] handle_ty handle() const noexcept override { return tlas_handle_; }
    [[nodiscard]] const void *handle_ptr() const noexcept override { return &tlas_handle_; }
};
}// namespace ocarina