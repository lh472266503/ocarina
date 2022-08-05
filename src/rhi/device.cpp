//
// Created by Zero on 06/06/2022.
//

#include "device.h"
#include "rhi/texture.h"
#include "rhi/stream.h"

namespace ocarina {

Stream Device::create_stream() noexcept {
    return _create<Stream>();
}
RHITexture Device::create_texture(uint2 res, PixelStorage pixel_storage) noexcept {
    return _create<RHITexture>(res, pixel_storage);
}
}// namespace ocarina