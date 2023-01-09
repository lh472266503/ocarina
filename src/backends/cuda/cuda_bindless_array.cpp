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

}// namespace ocarina