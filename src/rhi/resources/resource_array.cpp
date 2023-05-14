//
// Created by Zero on 12/01/2023.
//

#include "resource_array.h"
#include "texture.h"
#include "buffer.h"

namespace ocarina {

ResourceArray::ResourceArray(Device::Impl *device)
    : RHIResource(device, Tag::RESOURCE_ARRAY,
                  device->create_resource_array()) {}

size_t ResourceArray::emplace(const Texture &texture) noexcept {
    return impl()->emplace_texture(texture.tex_handle());
}

}// namespace ocarina