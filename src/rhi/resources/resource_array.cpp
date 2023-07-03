//
// Created by Zero on 12/01/2023.
//

#include "resource_array.h"
#include "texture.h"
#include "buffer.h"
#include "managed.h"

namespace ocarina {

ResourceArray::ResourceArray(Device::Impl *device)
    : RHIResource(device, Tag::RESOURCE_ARRAY,
                  device->create_resource_array()) {}

size_t ResourceArray::emplace(const Texture &texture) noexcept {
    return impl()->emplace_texture(texture.tex_handle());
}

void ResourceArray::set_texture(ocarina::handle_ty index, const ocarina::Texture &texture) noexcept {
    impl()->set_texture(index, texture.tex_handle());
}

CommandList ResourceArray::upload_handles_sync() noexcept {
    CommandList ret;
    ret.push_back(impl()->upload_buffer_handles_sync());
    ret.push_back(impl()->upload_texture_handles_sync());
    return ret;
}

CommandList ResourceArray::upload_handles() noexcept {
    CommandList ret;
    ret.push_back(impl()->upload_buffer_handles());
    ret.push_back(impl()->upload_texture_handles());
    return ret;
}

uint ResourceArray::buffer_num() const noexcept {
    return impl()->buffer_num();
}

uint ResourceArray::texture_num() const noexcept {
    return impl()->texture_num();
}

}// namespace ocarina