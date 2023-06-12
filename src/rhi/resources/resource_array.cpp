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

size_t ResourceArray::buffer_num() const noexcept {
    return impl()->buffer_num();
}

size_t ResourceArray::texture_num() const noexcept {
    return impl()->texture_num();
}

}// namespace ocarina