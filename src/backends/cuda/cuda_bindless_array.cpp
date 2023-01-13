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

void CUDABindlessArray::prepare_slotSOA() noexcept {
    _slot_soa.buffer_slot = _buffers.head();
    _slot_soa.tex_slot = _textures.head();
}

namespace detail {

template<typename T>
void remove_by_index(vector<T> &v, handle_ty index) noexcept {
    auto iter = v.begin();
    for (int i = 0; i < v.size(); ++i, ++iter) {
        if (i == index) {
            v.erase(iter);
            return;
        }
    }
}

}// namespace detail

void CUDABindlessArray::remove_buffer(handle_ty index) noexcept {
    detail::remove_by_index(_buffers.host(), index);
}

void CUDABindlessArray::remove_texture(handle_ty index) noexcept {
    detail::remove_by_index(_textures.host(), index);
}

BufferUploadCommand *CUDABindlessArray::upload_texture_handles() const noexcept {
    return _textures.upload();
}

BufferUploadCommand *CUDABindlessArray::upload_buffer_handles() const noexcept {
    return _buffers.upload();
}

}// namespace ocarina