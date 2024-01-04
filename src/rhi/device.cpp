//
// Created by Zero on 06/06/2022.
//

#include "device.h"
#include "resources/texture.h"
#include "resources/stream.h"
#include "rtx/accel.h"
#include "resources/bindless_array.h"

namespace ocarina {

Stream Device::create_stream() noexcept {
    return _create<Stream>();
}

Accel Device::create_accel() noexcept {
    return _create<Accel>();
}

BindlessArray Device::create_resource_array() noexcept {
    return _create<BindlessArray>();
}

Texture Device::create_texture(uint3 res, PixelStorage storage, const string &desc) noexcept {
    return _create<Texture>(res, storage);
}

Texture Device::create_texture(uint2 res, PixelStorage storage, const string &desc) noexcept {
    return create_texture(make_uint3(res, 1u), storage, desc);
}

}// namespace ocarina