//
// Created by Zero on 2024/3/22.
//

#pragma once

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "rhi/resources/gl_interop.h"
#include "rhi/resources/texture.h"
#include "util.h"

namespace ocarina {

class CUDAGLInterop : public GLInterop {
private:
    cudaGraphicsResource *_cuda_graphics_resource{nullptr};

public:
    explicit CUDAGLInterop(Texture *texture) : GLInterop(texture) {}
    void register_() noexcept override {
        CHECK_GL(glGenBuffers(1, &_pbo));
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, _pbo));

        uint2 res = _texture->resolution().xy();
        uint stride = _texture->pixel_size();
        CHECK_GL(glBufferData(GL_ARRAY_BUFFER, stride * res.x * res.y,
                              nullptr, GL_STREAM_DRAW));
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0u));
        OC_CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
            &_cuda_graphics_resource,
            _pbo,
            cudaGraphicsMapFlagsWriteDiscard));
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