//
// Created by Zero on 12/01/2023.
//

#include "cuda_bindless_array.h"
#include "cuda_device.h"

namespace ocarina {

CUDABindlessArray::CUDABindlessArray(CUDADevice *device)
    : _device(device), _buffers(device, slot_size, "CUDABindlessArray::_buffers"),
      _textures(device, slot_size, "CUDABindlessArray::_textures") {
    _slot_soa.buffer_slot = _buffers.head();
    _slot_soa.tex_slot = _textures.head();
}

size_t CUDABindlessArray::emplace_buffer(handle_ty handle, size_t size_in_byte) noexcept {
    auto ret = _buffers.host_buffer().size();
    _buffers.emplace_back(handle, size_in_byte);
    return ret;
}

size_t CUDABindlessArray::emplace_texture(handle_ty handle) noexcept {
    auto ret = _textures.host_buffer().size();
    _textures.push_back(handle);
    return ret;
}

void CUDABindlessArray::prepare_slotSOA(Device &device) noexcept {
    _buffers.reset_device_buffer_immediately(device);
    _textures.reset_device_buffer_immediately(device);
    _slot_soa.buffer_slot = _buffers.head();
    _slot_soa.tex_slot = _textures.head();
}

CommandList CUDABindlessArray::update_slotSOA(bool async) noexcept {
    CommandList ret;
    append(ret, _buffers.device_buffer().reallocate(_buffers.host_buffer().size(), async));
    append(ret, _textures.device_buffer().reallocate(_textures.host_buffer().size(), async));
    ret.push_back(HostFunctionCommand::create([&]() {
        _slot_soa.buffer_slot = _buffers.head();
        _slot_soa.tex_slot = _textures.head();
    },
                                              async));
    return ret;
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
    detail::remove_by_index(_buffers.host_buffer(), index);
}

void CUDABindlessArray::remove_texture(handle_ty index) noexcept {
    detail::remove_by_index(_textures.host_buffer(), index);
}

void CUDABindlessArray::set_buffer(ocarina::handle_ty index, ocarina::handle_ty handle, size_t size_in_byte) noexcept {
    OC_ASSERT(index < _buffers.host_buffer().size());
    _buffers.at(index) = {handle, size_in_byte};
}

BufferDesc CUDABindlessArray::buffer_view(ocarina::uint index) const noexcept {
    return _buffers.at(index);
}

void CUDABindlessArray::set_texture(ocarina::handle_ty index, ocarina::handle_ty handle) noexcept {
    OC_ASSERT(index < _textures.host_buffer().size());
    _textures.at(index) = handle;
}

size_t CUDABindlessArray::buffer_num() const noexcept {
    return _buffers.host_buffer().size();
}

size_t CUDABindlessArray::texture_num() const noexcept {
    return _textures.host_buffer().size();
}

BufferUploadCommand *CUDABindlessArray::upload_texture_handles(bool async) const noexcept {
    return _textures.upload(async);
}

BufferUploadCommand *CUDABindlessArray::upload_buffer_handles(bool async) const noexcept {
    return _buffers.upload(async);
}

}// namespace ocarina