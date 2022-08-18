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
protected:
    const Function &_function;
    CUDADevice *_device{};

public:
    CUDAShader(Device::Impl *device,
               const ocarina::string &ptx,
               const Function &f);
    virtual ~CUDAShader() {}
    static CUDAShader *create(Device::Impl *device,
                              const string &ptx,
                              const Function &f);
};

}// namespace ocarina