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

void BindlessArray::remove_buffer(handle_ty index) noexcept {

}

void BindlessArray::remove_texture(handle_ty index) noexcept {

}

BufferUploadCommand *BindlessArray::upload_buffer_handles() noexcept {
    return impl()->upload_buffer_handles();
}

BufferUploadCommand *BindlessArray::upload_texture_handles() noexcept {
    return impl()->upload_texture_handles();
}

}// namespace ocarina