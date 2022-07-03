//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "runtime/device.h"
#include "cuda.h"

namespace ocarina {
class CUDADevice : public Device {
private:
    CUdeviceptr _handle{};
    CUstream _stream{};
public:
    explicit CUDADevice(Context *context)
        : Device(context) {}
    [[nodiscard]] uint64_t create_buffer(size_t bytes) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    void compile(const Function &function) noexcept override;
};
}// namespace ocarina

