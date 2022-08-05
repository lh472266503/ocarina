//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/texture.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDATexture : public RHITexture::Impl {
private:
    CUDADevice *_device{};
    uint2 _res{};
    PixelStorage _pixel_storage{};
    CUtexObject _tex_handle{};
    CUarray _array_handle{};
    CUsurfObject _surface_handle{};

public:
    CUDATexture(CUDADevice *device, uint2 res, PixelStorage pixel_storage);
    void init();
    [[nodiscard]] uint2 resolution() const noexcept override { return _res; }
    [[nodiscard]] PixelStorage pixel_storage() const noexcept override { return _pixel_storage; }
};
}// namespace ocarina