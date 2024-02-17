//
// Created by Zero on 06/06/2022.
//

#include "cuda_device.h"
#include "cuda_stream.h"
#include "cuda_texture.h"
#include "cuda_shader.h"
#include "cuda_mesh.h"
#include "cuda_bindless_array.h"
#include "rhi/context.h"
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <nvrtc.h>
#include "embed/cuda_device_builtin_embed.h"
#include "embed/cuda_device_math_embed.h"
#include "embed/cuda_device_resource_embed.h"
#include "cuda_compiler.h"
#include "optix_accel.h"
#include "cuda_command_visitor.h"

namespace ocarina {

CUDADevice::CUDADevice(FileManager *context)
    : Device::Impl(context) {
    OC_CU_CHECK(cuInit(0));
    OC_CU_CHECK(cuDeviceGet(&_cu_device, 0));
    OC_CU_CHECK(cuDevicePrimaryCtxRetain(&_cu_ctx, _cu_device));
    _cmd_visitor = std::make_unique<CUDACommandVisitor>(this);
    init_hardware_info();
}

void CUDADevice::init_hardware_info() {
    auto compute_cap_major = 0;
    auto compute_cap_minor = 0;
    OC_CU_CHECK(cuDeviceGetAttribute(&compute_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _cu_device));
    OC_CU_CHECK(cuDeviceGetAttribute(&compute_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _cu_device));
    OC_INFO_FORMAT(
        "Created CUDA device : (capability = {}.{}).",
        compute_cap_major, compute_cap_minor);
    _compute_capability = 10u * compute_cap_major + compute_cap_minor;
}

handle_ty CUDADevice::create_buffer(size_t size, const string &desc) noexcept {
    OC_ASSERT(size > 0);
    return use_context([&] {
        handle_ty handle{};
        OC_CU_CHECK(cuMemAlloc(&handle, size));
        MemoryStats::instance().on_buffer_allocate(handle, size, desc);
        return handle;
    });
}

namespace detail {
void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}
}// namespace detail

void CUDADevice::init_optix_context() noexcept {
    if (_optix_device_context) {
        return;
    }
    use_context([&] {
        OC_CU_CHECK(cuMemFree(0));
        OC_OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions ctx_options = {};
#ifndef NDEBUG
        ctx_options.logCallbackLevel = 4;// status/progress
#else
        ctx_options.logCallbackLevel = 2;// error
#endif
        ctx_options.logCallbackFunction = detail::context_log_cb;
#if (OPTIX_VERSION >= 70200)
        ctx_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif
        CUcontext cu_context = nullptr;
        OC_OPTIX_CHECK(optixDeviceContextCreate(cu_context, &ctx_options, &_optix_device_context));
    });
}

handle_ty CUDADevice::create_stream() noexcept {
    return use_context([&] {
        CUDAStream *stream = ocarina::new_with_allocator<CUDAStream>(this);
        return reinterpret_cast<handle_ty>(stream);
    });
}

handle_ty CUDADevice::create_texture(uint3 res, PixelStorage pixel_storage,
                                     uint level_num,
                                     const string &desc) noexcept {
    return use_context([&] {
        auto texture = ocarina::new_with_allocator<CUDATexture>(this, res, pixel_storage, level_num);
        return reinterpret_cast<handle_ty>(texture);
    });
}

handle_ty CUDADevice::create_shader(const Function &function) noexcept {
    CUDACompiler compiler(this);
    ocarina::string ptx = compiler.compile(function, _compute_capability);

    auto ptr = use_context([&] {
        auto shader = CUDAShader::create(this, ptx, function);
        return reinterpret_cast<handle_ty>(shader);
    });

    return ptr;
}

handle_ty CUDADevice::create_mesh(const MeshParams &params) noexcept {
    auto ret = new_with_allocator<CUDAMesh>(this, params);
    return reinterpret_cast<handle_ty>(ret);
}

void CUDADevice::destroy_mesh(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDAMesh *>(handle));
}

handle_ty CUDADevice::create_bindless_array() noexcept {
    auto ret = new_with_allocator<CUDABindlessArray>(this);
    return reinterpret_cast<handle_ty>(ret);
}

void CUDADevice::destroy_bindless_array(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDABindlessArray *>(handle));
}

void CUDADevice::destroy_buffer(handle_ty handle) noexcept {
    if (handle != 0) {
        MemoryStats::instance().on_buffer_free(handle);
        OC_CU_CHECK(cuMemFree(handle));
    }
}

void CUDADevice::destroy_shader(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDAShader *>(handle));
}

void CUDADevice::destroy_texture(handle_ty handle) noexcept {
    use_context([&] {
        ocarina::delete_with_allocator(reinterpret_cast<CUDATexture *>(handle));
    });
}

void CUDADevice::destroy_stream(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDAStream *>(handle));
}
handle_ty CUDADevice::create_accel() noexcept {
    return use_context([&] {
        auto accel = new_with_allocator<OptixAccel>(this);
        return reinterpret_cast<handle_ty>(accel);
    });
}
void CUDADevice::destroy_accel(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<OptixAccel *>(handle));
}
CommandVisitor *CUDADevice::command_visitor() noexcept {
    return _cmd_visitor.get();
}

}// namespace ocarina

OC_EXPORT_API ocarina::CUDADevice *create(ocarina::FileManager *context) {
    return ocarina::new_with_allocator<ocarina::CUDADevice>(context);
}

OC_EXPORT_API void destroy(ocarina::CUDADevice *device) {
    ocarina::delete_with_allocator(device);
}