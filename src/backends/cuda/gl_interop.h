//
// Created by Zero on 2024/3/22.
//

#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "rhi/resources/gl_interop.h"

namespace ocarina {

class CUDAGLInterop : public GLInterop {
private:
    cudaGraphicsResource *_cuda_graphics_resource{nullptr};

public:
    explicit CUDAGLInterop(Texture *texture) : GLInterop(texture) {}
    void register_() noexcept override {
    }
    void mapping() noexcept override {
    }
    void unmapping() noexcept override {
    }
    void unregister_() noexcept override {
    }
    GLuint get_pbo() noexcept override {
        return _pbo;
    }
};

}// namespace ocarina