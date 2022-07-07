//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "runtime/device.h"
#include <cuda.h>
#include "util.h"

namespace ocarina {
class CUDADevice : public Device::Impl {
private:
    CUdevice _cu_device{};
    CUstream _cu_stream{};
    CUcontext _cu_ctx{};

public:
    explicit CUDADevice(Context *context);
    void destroy_buffer(handle_ty handle) noexcept override;
    void destroy_texture(handle_ty handle) noexcept override;
    void destroy_stream(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_buffer(size_t size) noexcept override;
    void compile(const Function &function) noexcept override;
};
}// namespace ocarina

