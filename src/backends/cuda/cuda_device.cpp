//
// Created by Zero on 06/06/2022.
//

#include "cuda_device.h"
#include "cuda_stream.h"
#include "cuda_texture.h"
#include "cuda_codegen.h"
#include "cuda_shader.h"
#include "cuda_mesh.h"
#include "rhi/context.h"
#include <nvrtc.h>
#include "embed/cuda_device_builtin_embed.h"
#include "embed/cuda_device_math_embed.h"
#include "embed/cuda_device_resource_embed.h"

namespace ocarina {

#define CUDA_NVRTC_OPTIONS                 \
    "-std=c++17",                          \
        "-arch",                           \
        "compute_50",                      \
        "-use_fast_math",                  \
        "-lineinfo",                       \
        "-default-device",                 \
        "-include=cuda_device_builtin.h",  \
        "-include=cuda_device_math.h",     \
        "-include=cuda_device_resource.h", \
        "-rdc",                            \
        "true",                            \
        "-D__x86_64",

namespace detail {
[[nodiscard]] string get_ptx(const string &cu) noexcept {
    nvrtcProgram program{};
    ocarina::vector<const char *> compile_option = {CUDA_NVRTC_OPTIONS};
    std::array header_names{"cuda_device_builtin.h", "cuda_device_math.h", "cuda_device_resource.h"};
    std::array header_sources{cuda_device_builtin, cuda_device_math, cuda_device_resource};

    OC_NVRTC_CHECK(nvrtcCreateProgram(&program, cu.c_str(), "cuda_kernel.cu",
                                      header_names.size(), header_sources.data(),
                                      header_names.data()));
    const nvrtcResult compile_res = nvrtcCompileProgram(program, compile_option.size(), compile_option.data());
    size_t log_size = 0;
    OC_NVRTC_CHECK(nvrtcGetProgramLogSize(program, &log_size));
    string log;
    log.resize(log_size);
    if (log_size > 1) {
        OC_NVRTC_CHECK(nvrtcGetProgramLog(program, log.data()));
    }
    if (compile_res != NVRTC_SUCCESS) {
        cout << log << endl;
        std::abort();
    }
    size_t ptx_size = 0;
    ocarina::string ptx;
    OC_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
    ptx.resize(ptx_size);
    OC_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
    OC_NVRTC_CHECK(nvrtcDestroyProgram(&program));
    return ptx;
}
}// namespace detail
#undef CUDA_NVRTC_OPTIONS

CUDADevice::CUDADevice(Context *context)
    : Device::Impl(context) {
    OC_CU_CHECK(cuInit(0));
    OC_CU_CHECK(cuDeviceGet(&_cu_device, 0));
    OC_CU_CHECK(cuDevicePrimaryCtxRetain(&_cu_ctx, _cu_device));
}

handle_ty CUDADevice::create_buffer(size_t size) noexcept {
    return use_context([&] {
        handle_ty handle{};
        OC_CU_CHECK(cuMemAlloc(&handle, size));
        return handle;
    });
}

handle_ty CUDADevice::create_stream() noexcept {
    return use_context([&] {
        CUDAStream *stream = ocarina::new_with_allocator<CUDAStream>(this);
        return reinterpret_cast<handle_ty>(stream);
    });
}

ocarina::string CUDADevice::get_ptx(const Function &function) const noexcept {
    ocarina::string ptx_fn = function.func_name() + ".ptx";
    string cu_fn = function.func_name() + ".cu";
    ocarina::string ptx;
    if (!_context->is_exist_cache(ptx_fn)) {
        if (!_context->is_exist_cache(cu_fn)) {
            CUDACodegen codegen;
            codegen.emit(function);
            const ocarina::string &cu = codegen.scratch().c_str();
            cout << cu << endl;
            _context->write_cache(cu_fn, cu);
            ptx = detail::get_ptx(cu);
            _context->write_cache(ptx_fn, ptx);
        } else {
            const ocarina::string &cu = _context->read_cache(cu_fn);
            cout << cu << endl;
            ptx = detail::get_ptx(cu);
            //            _context->write_cache(ptx_fn, ptx);
        }
    } else {
        ptx = _context->read_cache(ptx_fn);
    }
    return ptx;
}

handle_ty CUDADevice::create_texture(uint2 res, PixelStorage pixel_storage) noexcept {
    return use_context([&] {
        auto texture = ocarina::new_with_allocator<CUDATexture>(this, res, pixel_storage);
        return reinterpret_cast<handle_ty>(texture);
    });
}

handle_ty CUDADevice::create_shader(const Function &function) noexcept {
    ocarina::string ptx = get_ptx(function);

    auto ptr = use_context([&] {
        auto shader = ocarina::new_with_allocator<CUDAShader>(this, ptx, function);
        return reinterpret_cast<handle_ty>(shader);
    });

    return ptr;
}

handle_ty CUDADevice::create_mesh(handle_ty v_handle, handle_ty t_handle,
                                  uint v_stride, uint tri_num,AccelUsageTag usage_tag) noexcept {
    auto ptr = use_context([&]{
        return ocarina::new_with_allocator<CUDAMesh>(v_handle, t_handle, v_stride, tri_num, usage_tag);
    });
    return reinterpret_cast<handle_ty>(ptr);
}

void CUDADevice::destroy_mesh(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDAMesh *>(handle));
}

void CUDADevice::destroy_buffer(handle_ty handle) noexcept {
    OC_CU_CHECK(cuMemFree(handle));
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
    return 0;
}
void CUDADevice::destroy_accel(handle_ty handle) noexcept {
}

}// namespace ocarina

OC_EXPORT_API ocarina::CUDADevice *create(ocarina::Context *context) {
    return ocarina::new_with_allocator<ocarina::CUDADevice>(context);
}

OC_EXPORT_API void destroy(ocarina::CUDADevice *device) {
    ocarina::delete_with_allocator(device);
}