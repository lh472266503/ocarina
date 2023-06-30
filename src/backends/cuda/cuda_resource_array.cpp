//
// Created by Zero on 12/01/2023.
//

#include "cuda_resource_array.h"
#include "cuda_device.h"

namespace ocarina {

CUDAResourceArray::CUDAResourceArray(CUDADevice *device)
    : _device(device) {}

size_t CUDAResourceArray::emplace_buffer(handle_ty handle,size_t size_in_byte) noexcept {
    auto ret = _buffers.host_buffer().size();
    _buffers.emplace_back(handle, size_in_byte);
    return ret;
}

size_t CUDAResourceArray::emplace_texture(handle_ty handle) noexcept {
    auto ret = _textures.host_buffer().size();
    _textures.push_back(handle);
    return ret;
}

void CUDAResourceArray::prepare_slotSOA(Device &device) noexcept {
    _buffers.reset_device_buffer_immediately(device);
    _textures.reset_device_buffer_immediately(device);
    _slot_soa.buffer_slot = reinterpret_cast<void*>(_buffers.head());
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

void CUDAResourceArray::remove_buffer(handle_ty index) noexcept {
    detail::remove_by_index(_buffers.host_buffer(), index);
}

void CUDAResourceArray::remove_texture(handle_ty index) noexcept {
    detail::remove_by_index(_textures.host_buffer(), index);
}

void CUDAResourceArray::set_buffer(ocarina::handle_ty index, ocarina::handle_ty handle, size_t size_in_byte) noexcept {
    OC_ASSERT(index < _buffers.host_buffer().size());
    _buffers.at(index) = {handle, size_in_byte};
}

void CUDAResourceArray::set_texture(ocarina::handle_ty index, ocarina::handle_ty handle) noexcept {
    OC_ASSERT(index < _textures.host_buffer().size());
    _textures.at(index) = handle;
}

size_t CUDAResourceArray::buffer_num() const noexcept {
    return _buffers.host_buffer().size();
}

size_t CUDAResourceArray::texture_num() const noexcept {
    return _textures.host_buffer().size();
}

BufferUploadCommand *CUDAResourceArray::upload_texture_handles() const noexcept {
    return _textures.upload();
}

BufferUploadCommand *CUDAResourceArray::upload_buffer_handles() const noexcept {
    return _buffers.upload();
}

BufferUploadCommand *CUDAResourceArray::upload_buffer_handles_sync() const noexcept {
    return _buffers.upload_sync();
}

BufferUploadCommand *CUDAResourceArray::upload_texture_handles_sync() const noexcept {
    return _textures.upload_sync();
}
}// namespace ocarina