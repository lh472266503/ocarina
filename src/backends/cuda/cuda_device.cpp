//
// Created by Zero on 06/06/2022.
//

#include "cuda_device.h"

namespace ocarina {

void CUDADevice::compile(Function function) noexcept {

}
uint64_t CUDADevice::create_buffer(size_t bytes) noexcept {
    return 0;
}
void CUDADevice::destroy_buffer(uint64_t handle) noexcept {
}
}// namespace ocarina

OC_EXPORT_API ocarina::Device *create(ocarina::Context *context) {
    return ocarina::new_with_allocator<ocarina::CUDADevice>(context);
}

OC_EXPORT_API void destroy(ocarina::Device *device) {
    ocarina::delete_with_allocator(device);
}