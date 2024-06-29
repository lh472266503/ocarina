//
// Created by Zero on 06/06/2022.
//

#include "device.h"
#include "resources/texture.h"
#include "resources/stream.h"
#include "rtx/accel.h"
#include "resources/bindless_array.h"
#include "resources/byte_buffer.h"

namespace ocarina {

ByteBuffer Device::create_byte_buffer(size_t size, const std::string &name) const noexcept {
    return ByteBuffer(impl_.get(), size, name);
}

Stream Device::create_stream() noexcept {
    return _create<Stream>();
}

Accel Device::create_accel() const noexcept {
    return _create<Accel>();
}

BindlessArray Device::create_bindless_array() const noexcept {
    return _create<BindlessArray>();
}

Texture Device::create_texture(uint3 res, PixelStorage storage, const string &desc) const noexcept {
    return _create<Texture>(res, storage, 1, desc);
}

Texture Device::create_texture(uint2 res, PixelStorage storage, const string &desc) const noexcept {
    return create_texture(make_uint3(res, 1u), storage, desc);
}

}// namespace ocarina