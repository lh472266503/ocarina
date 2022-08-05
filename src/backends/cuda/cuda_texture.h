//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/texture.h"
#include <cuda.h>

namespace ocarina {
class CUDATexture : public RHITexture::Impl {
public:
    CUDATexture(uint2 res, PixelStorage pixel_storage)
        : RHITexture::Impl(res, pixel_storage) {}
};
}// namespace ocarina