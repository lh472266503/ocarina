//
// Created by Zero on 12/01/2023.
//

#include "cuda_resource_array.h"
#include "cuda_device.h"

namespace ocarina {

CUDAResourceArray::CUDAResourceArray(CUDADevice *device)
    : _device(device) {}

size_t CUDAResourceArray::emplace_buffer(handle_ty handle) noexcept {
    auto ret = _buffers.host().size();
    _buffers.push_back(handle);
    return ret;
}

size_t CUDAResourceArray::emplace_texture(handle_ty handle) noexcept {
    auto ret = _textures.host().size();
    _textures.push_back(handle);
    return ret;
}

size_t CUDAResourceArray::emplace_mix_buffer(handle_ty handle) noexcept {
    auto ret = _mix_buffers.host().size();
    _mix_buffers.push_back(handle);
    return ret;
}

void CUDAResourceArray::prepare_slotSOA(Device &device) noexcept {
    _buffers.reset_device_buffer(device);
    _textures.reset_device_buffer(device);
    _mix_buffers.reset_device_buffer(device);
    _slot_soa.buffer_slot = _buffers.head();
    _slot_soa.tex_slot = _textures.head();
    _slot_soa.mix_buffer_slot = _mix_buffers.head();
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

void CUDAResourceArray::remove_buffer(handle_ty index) noexcept {
    detail::remove_by_index(_buffers.host(), index);
}

void CUDAResourceArray::remove_texture(handle_ty index) noexcept {
    detail::remove_by_index(_textures.host(), index);
}

void CUDAResourceArray::remove_mix_buffer(handle_ty index) noexcept {
    detail::remove_by_index(_mix_buffers.host(), index);
}

BufferUploadCommand *CUDAResourceArray::upload_texture_handles() const noexcept {
    return _textures.upload();
}

BufferUploadCommand *CUDAResourceArray::upload_buffer_handles() const noexcept {
    return _buffers.upload();
}

BufferUploadCommand *CUDAResourceArray::upload_mix_buffer_handles() const noexcept {
    return _mix_buffers.upload();
}

BufferUploadCommand *CUDAResourceArray::upload_buffer_handles_sync() const noexcept {
    return _buffers.upload_sync();
}

BufferUploadCommand *CUDAResourceArray::upload_texture_handles_sync() const noexcept {
    return _textures.upload_sync();
}

BufferUploadCommand *CUDAResourceArray::upload_mix_buffer_handles_sync() const noexcept {
    return _mix_buffers.upload_sync();
}

}// namespace ocarina