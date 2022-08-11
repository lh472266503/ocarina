//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/texture.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDATexture : public Texture<float>::Impl {
private:
    struct OCTexture {
        CUtexObject texture{};
        CUsurfObject surface{};
        PixelStorage pixel_storage{};
    };
    OCTexture _oc_texture;
    CUDADevice *_device{};
    uint2 _res{};
    CUarray _array_handle{};

public:
    CUDATexture(CUDADevice *device, uint2 res, PixelStorage pixel_storage);
    ~CUDATexture();
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