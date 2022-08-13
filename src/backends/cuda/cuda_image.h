//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/image.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDAImage : public Image<float>::Impl {
private:
    struct ImageData {
        CUtexObject texture{};
        CUsurfObject surface{};
        PixelStorage pixel_storage{};
    };
    ImageData _oc_texture;
    CUDADevice *_device{};
    uint2 _res{};
    CUarray _array_handle{};

public:
    CUDAImage(CUDADevice *device, uint2 res, PixelStorage pixel_storage);
    ~CUDAImage();
    void init();
    [[nodiscard]] uint2 resolution() const noexcept override { return _res; }
    [[nodiscard]] handle_ty array_handle() const noexcept override { return reinterpret_cast<handle_ty>(_array_handle); }
    [[nodiscard]] handle_ty tex_handle() const noexcept override {
        return _oc_texture.texture;
    }
    [[nodiscard]] const handle_ty *tex_handle_address() const noexcept override {
        return &_oc_texture.texture;
    }
    [[nodiscard]] PixelStorage pixel_storage() const noexcept override { return _oc_texture.pixel_storage; }
};
}// namespace ocarina