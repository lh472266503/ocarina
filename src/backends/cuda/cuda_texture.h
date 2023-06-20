//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/texture.h"
#include <cuda.h>

namespace ocarina {
class CUDADevice;
class CUDATexture : public Texture::Impl {
private:
    OCTexture _data;
    CUDADevice *_device{};
    uint3 _res{};
    CUarray _array_handle{};
    uint level_num{1u};

public:
    CUDATexture(CUDADevice *device, uint3 res, PixelStorage pixel_storage, uint level_num);
    ~CUDATexture();
    void init();
    [[nodiscard]] uint3 resolution() const noexcept override { return _res; }
    [[nodiscard]] handle_ty array_handle() const noexcept override { return reinterpret_cast<handle_ty>(_array_handle); }
    [[nodiscard]] handle_ty tex_handle() const noexcept override {
        return _data.texture;
    }
    [[nodiscard]] const void *handle_ptr() const noexcept override {
        return &_data;
    }
    [[nodiscard]] size_t data_size() const noexcept override;
    [[nodiscard]] size_t data_alignment() const noexcept override;
    [[nodiscard]] size_t max_member_size() const noexcept override;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept override { return _data.pixel_storage; }
};
}// namespace ocarina