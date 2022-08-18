//
// Created by zero on 2022/7/18.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/shader.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDAShader : public Shader<>::Impl {
private:
    CUmodule _module{};
    CUfunction _func_handle{};
    const Function &_function;
    CUDADevice *_device{};

public:
    CUDAShader(Device::Impl *device,
               const ocarina::string &ptx,
               const Function &f);
    static CUDAShader *create(Device::Impl *device,
                              const string &ptx,
                              const Function &f);
    ~CUDAShader();
    void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept override;
    void compute_fit_size() noexcept override;
};

}// namespace ocarina