//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/texture.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDATexture : public RHITexture<float>::Impl {
private:
    CUDADevice *_device{};
    uint2 _res{};
    PixelStorage _pixel_storage{};
    CUtexObject _tex_handle{};
    CUarray _array_handle{};
    CUsurfObject _surface_handle{};

public:
    CUDATexture(CUDADevice *device, uint2 res, PixelStorage pixel_storage);
    ~CUDATexture();
    void init();
    [[nodiscard]] uint2 resolution() const noexcept override { return _res; }
    [[nodiscard]] CUarray array_handle() const noexcept { return _array_handle; }
    [[nodiscard]] CUsurfObject surface_handle() const noexcept { return _surface_handle; }
    [[nodiscard]] CUtexObject tex_handle() const noexcept { return _tex_handle; }
    [[nodiscard]] handle_ty handle() const noexcept override { return reinterpret_cast<handle_ty>(array_handle());}
    [[nodiscard]] PixelStorage pixel_storage() const noexcept override { return _pixel_storage; }
};
}// namespace ocarina