//
// Created by Zero on 12/01/2023.
//

#include "cuda_bindless_array.h"
#include "cuda_device.h"

namespace ocarina {

CUDABindlessArray::CUDABindlessArray(CUDADevice *device)
    : _device(device) {}

size_t CUDABindlessArray::emplace_buffer(handle_ty handle) noexcept {
    auto ret = _buffers.host().size();
    _buffers.push_back(handle);
    return ret;
}

size_t CUDABindlessArray::emplace_texture(handle_ty handle) noexcept {
    auto ret = _textures.host().size();
    _textures.push_back(handle);
    return ret;
}

void CUDABindlessArray::remove_buffer(handle_ty index) noexcept {

}

void CUDABindlessArray::remove_texture(handle_ty index) noexcept {

}

BufferUploadCommand *CUDABindlessArray::upload_texture_handles() const noexcept {
    return _textures.upload();
}

BufferUploadCommand *CUDABindlessArray::upload_buffer_handles() const noexcept {
    return _buffers.upload();
}

}// namespace ocarina