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

BindlessArray Device::create_bindless_array() noexcept {
    return _create<BindlessArray>();
}

RHITexture Device::create_texture(uint2 res, PixelStorage storage) noexcept {
    return _create<RHITexture>(res, storage);
}

}// namespace ocarina