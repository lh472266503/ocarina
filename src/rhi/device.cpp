//
// Created by Zero on 06/06/2022.
//

#include "device.h"
#include "resources/texture.h"
#include "resources/stream.h"
#include "rtx/accel.h"
#include "resources/resource_array.h"

namespace ocarina {

Stream Device::create_stream() noexcept {
    return _create<Stream>();
}

Accel Device::create_accel() noexcept {
    return _create<Accel>();
}

ResourceArray Device::create_resource_array() noexcept {
    return _create<ResourceArray>();
}

RHITexture Device::create_texture(uint3 res, PixelStorage storage) noexcept {
    return _create<RHITexture>(res, storage);
}

RHITexture Device::create_texture(uint2 res, PixelStorage storage) noexcept{
    return create_texture(make_uint3(res, 1u), storage);
}

}// namespace ocarina