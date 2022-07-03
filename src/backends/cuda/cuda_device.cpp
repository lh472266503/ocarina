//
// Created by Zero on 06/06/2022.
//

#include "cuda_device.h"

namespace ocarina {

CUDADevice::CUDADevice(Context *context)
    : Device(context) {
    OC_CU_CHECK(cuInit(0));
    OC_CU_CHECK(cuDeviceGet(&_cu_device, 0));
    OC_CU_CHECK(cuCtxCreate(&_cu_ctx, 0, _cu_device));
    OC_CU_CHECK(cuCtxSetCurrent(_cu_ctx));
}

void CUDADevice::compile(const Function &function) noexcept {
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