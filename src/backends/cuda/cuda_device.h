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
class CUDADevice : public Device {
private:
    CUdeviceptr _handle{};
    CUstream _stream{};

public:
    explicit CUDADevice(Context *context);
    [[nodiscard]] handle_ty create_buffer(size_t bytes) noexcept override;
    void destroy_buffer(handle_ty handle) noexcept override;
    void compile(const Function &function) noexcept override;
};
}// namespace ocarina

