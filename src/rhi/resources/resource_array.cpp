//
// Created by Zero on 12/01/2023.
//

#include "resource_array.h"
#include "texture.h"

namespace ocarina {

ResourceArray::ResourceArray(Device::Impl *device)
    : RHIResource(device, Tag::RESOURCE_ARRAY,
                  device->create_resource_array()) {}

size_t ResourceArray::emplace(const RHITexture &texture) noexcept {
    return impl()->emplace_texture(texture.tex_handle());
}

void ResourceArray::remove_buffer(handle_ty index) noexcept {
    impl()->remove_buffer(index);
}

void ResourceArray::remove_texture(handle_ty index) noexcept {
    impl()->remove_texture(index);
}

BufferUploadCommand *ResourceArray::upload_buffer_handles() noexcept {
    return impl()->upload_buffer_handles();
}

BufferUploadCommand *ResourceArray::upload_texture_handles() noexcept {
    return impl()->upload_texture_handles();
}

BufferUploadCommand *ResourceArray::upload_mix_buffer_handles() noexcept {
    return impl()->upload_mix_buffer_handles();
}

BufferUploadCommand *ResourceArray::upload_buffer_handles_sync() noexcept {
    return impl()->upload_buffer_handles_sync();
}

BufferUploadCommand *ResourceArray::upload_texture_handles_sync() noexcept {
    return impl()->upload_texture_handles_sync();
}

BufferUploadCommand *ResourceArray::upload_mix_buffer_handles_sync() noexcept {
    return impl()->upload_mix_buffer_handles_sync();
}

}// namespace ocarina