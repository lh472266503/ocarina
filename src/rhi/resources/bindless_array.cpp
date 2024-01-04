//
// Created by Zero on 12/01/2023.
//

#include "bindless_array.h"
#include "texture.h"
#include "buffer.h"
#include "managed.h"

namespace ocarina {

BindlessArray::BindlessArray(Device::Impl *device)
    : RHIResource(device, Tag::RESOURCE_ARRAY,
                  device->create_bindless_array()) {}

size_t BindlessArray::emplace(const Texture &texture) noexcept {
    return impl()->emplace_texture(texture.tex_handle());
}

void BindlessArray::set_texture(ocarina::handle_ty index, const ocarina::Texture &texture) noexcept {
    impl()->set_texture(index, texture.tex_handle());
}

CommandList BindlessArray::upload_handles_sync() noexcept {
    CommandList ret;
    ret.push_back(impl()->upload_buffer_handles_sync());
    ret.push_back(impl()->upload_texture_handles_sync());
    return ret;
}

CommandList BindlessArray::upload_handles() noexcept {
    CommandList ret;
    ret.push_back(impl()->upload_buffer_handles());
    ret.push_back(impl()->upload_texture_handles());
    return ret;
}

uint BindlessArray::buffer_num() const noexcept {
    return impl()->buffer_num();
}

uint BindlessArray::texture_num() const noexcept {
    return impl()->texture_num();
}

}// namespace ocarina