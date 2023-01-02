//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/texture.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDATexture : public RHITexture::Impl {
private:
    OCTexture _image_data;
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
        return _image_data.texture;
    }
    [[nodiscard]] const void *handle_ptr() const noexcept override {
        return &_image_data;
    }
    [[nodiscard]] size_t data_size() const noexcept override;
    [[nodiscard]] size_t data_alignment() const noexcept override;
    [[nodiscard]] size_t max_member_size() const noexcept override;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept override { return _image_data.pixel_storage; }
};
}// namespace ocarina