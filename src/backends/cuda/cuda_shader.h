//
// Created by zero on 2022/7/18.
//

#pragma once

#include "core/stl.h"
#include "runtime/shader.h"
#include <cuda.h>

namespace ocarina {

class CUDAShader : public Shader<>::Impl {
private:
    CUmodule _module{};
    CUfunction _function{};
    ocarina::string _entry{};

public:
    CUDAShader(Device::Impl *device,
               const ocarina::string &ptx,
               ocarina::string_view entry);

    void launch(handle_ty stream) noexcept override;
};

}// namespace ocarina