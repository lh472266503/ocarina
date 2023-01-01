//
// Created by Zero on 06/06/2022.
//

#include "device.h"
#include "resources/texture.h"
#include "resources/stream.h"
#include "rtx/accel.h"

namespace ocarina {

Stream Device::create_stream() noexcept {
    return _create<Stream>();
}

Accel Device::create_accel() noexcept {
    return _create<Accel>();
}
RHITexture Device::create_image(uint2 res, PixelStorage storage) noexcept {
    return _create<RHITexture>(res, storage);
}

}// namespace ocarina