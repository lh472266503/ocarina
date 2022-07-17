//
// Created by Zero on 06/06/2022.
//

#include "cuda_device.h"
#include "core/logging.h"
#include "cuda_stream.h"

namespace ocarina {

CUDADevice::CUDADevice(Context *context)
    : Device::Impl(context) {
    OC_CU_CHECK(cuInit(0));
    OC_CU_CHECK(cuDeviceGet(&_cu_device, 0));
    OC_CU_CHECK(cuDevicePrimaryCtxRetain(&_cu_ctx, _cu_device));
}

handle_ty CUDADevice::create_buffer(size_t size) noexcept {
    return bind_handle([&] {
        handle_ty handle{};
        OC_CU_CHECK(cuMemAlloc(&handle, size));
        return handle;
    });
}

handle_ty CUDADevice::create_stream() noexcept {
    return bind_handle([&] {
        CUDAStream *stream = ocarina::new_with_allocator<CUDAStream>(this);
        return reinterpret_cast<handle_ty>(stream);
    });
}

handle_ty CUDADevice::create_shader(ocarina::string_view str) noexcept {
    return {};
}

void CUDADevice::destroy_buffer(handle_ty handle) noexcept {
}

void CUDADevice::destroy_texture(handle_ty handle) noexcept {
}

void CUDADevice::destroy_stream(handle_ty handle) noexcept {
}

}// namespace ocarina

OC_EXPORT_API ocarina::CUDADevice *create(ocarina::Context *context) {
    return ocarina::new_with_allocator<ocarina::CUDADevice>(context);
}

OC_EXPORT_API void destroy(ocarina::CUDADevice *device) {
    ocarina::delete_with_allocator(device);
}