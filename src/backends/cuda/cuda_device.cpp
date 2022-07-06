//
// Created by Zero on 06/06/2022.
//

#include "cuda_device.h"
#include "core/logging.h"

namespace ocarina {

CUDADevice::CUDADevice(Context *context)
    : Device::Impl(context) {
    OC_CU_CHECK(cuInit(0));
    OC_CU_CHECK(cuDeviceGet(&_cu_device, 0));
    OC_CU_CHECK(cuCtxCreate(&_cu_ctx, 0, _cu_device));
    OC_CU_CHECK(cuCtxSetCurrent(_cu_ctx));
}
handle_ty CUDADevice::create_buffer(size_t size) noexcept {
    return 0;
}
void CUDADevice::destroy_buffer(handle_ty handle) noexcept {
}

void CUDADevice::destroy_texture(handle_ty handle) noexcept {

}

void CUDADevice::compile(const Function &function) noexcept {
}
}// namespace ocarina

OC_EXPORT_API ocarina::Device::Impl *create(ocarina::Context *context) {
    return ocarina::new_with_allocator<ocarina::CUDADevice>(context);
}

OC_EXPORT_API void destroy(ocarina::Device *device) {
    ocarina::delete_with_allocator(device);
}