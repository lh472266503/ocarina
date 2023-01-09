//
// Created by Zero on 12/01/2023.
//

#include "bindless_array.h"
#include "texture.h"

namespace ocarina {

BindlessArray::BindlessArray(Device::Impl *device)
    : RHIResource(device, Tag::BINDLESS_ARRAY,
                  device->create_bindless_array()) {}

size_t BindlessArray::emplace(const RHITexture &texture) noexcept {
    return impl()->emplace_texture(texture.tex_handle());
}

}// namespace ocarina