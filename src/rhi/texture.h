//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "resource.h"

namespace ocarina {

class RHITexture : public RHIResource {
public:
    class Impl {
    public:
        uint2 res{};
        PixelStorage pixel_storage{};

    public:
        Impl(uint2 res, PixelStorage pixel_storage)
            : res(res), pixel_storage(pixel_storage) {}
    };

public:
    explicit RHITexture(Device::Impl *device, uint2 res,
                        PixelStorage pixel_storage);
    [[nodiscard]] Impl *impl() noexcept { return reinterpret_cast<Impl *>(_handle); }
    [[nodiscard]] const Impl *impl() const noexcept { return reinterpret_cast<const Impl *>(_handle); }
    [[nodiscard]] uint2 resolution() const noexcept { return impl()->res; }
    [[nodiscard]] PixelStorage pixel_storage() const noexcept { return impl()->pixel_storage; }
};

}// namespace ocarina