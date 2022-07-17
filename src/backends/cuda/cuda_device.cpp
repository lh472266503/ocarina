//
// Created by Zero on 06/06/2022.
//

#include "cuda_device.h"
#include "cuda_stream.h"
#include "cuda_codegen.h"
#include <nvrtc.h>

namespace ocarina {

#define CUDA_NVRTC_OPTIONS \
    "-std=c++11",          \
        "-arch",           \
        "compute_50",      \
        "-use_fast_math",  \
        "-lineinfo",       \
        "-default-device", \
        "-rdc",            \
        "true",            \
        "-D__x86_64",

namespace detail {
[[nodiscard]] string get_ptx(const string &cu) noexcept {
    nvrtcProgram program{};
    cout << cu << endl;
    ocarina::vector<const char *> compile_option = {CUDA_NVRTC_OPTIONS};
    OC_NVRTC_CHECK(nvrtcCreateProgram(&program, cu.c_str(), "test", 0, nullptr, nullptr));
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

handle_ty CUDADevice::create_shader(const Function &function) noexcept {
    CUDACodegen codegen;
    codegen.emit(function);
    const ocarina::string &cu = codegen.scratch().c_str();

    ocarina::string ptx = detail::get_ptx(cu);

    auto ret = bind_handle([&] {
        CUmodule module{};
        OC_CU_CHECK(cuModuleLoadData(&module, ptx.c_str()));
        return reinterpret_cast<handle_ty>(module);
    });

    return ret;
}

void CUDADevice::destroy_buffer(handle_ty handle) noexcept {
    OC_CU_CHECK(cuMemFree(handle));
}

void CUDADevice::destroy_texture(handle_ty handle) noexcept {
}

void CUDADevice::destroy_stream(handle_ty handle) noexcept {
    ocarina::delete_with_allocator(reinterpret_cast<CUDAStream *>(handle));
}

}// namespace ocarina

OC_EXPORT_API ocarina::CUDADevice *create(ocarina::Context *context) {
    return ocarina::new_with_allocator<ocarina::CUDADevice>(context);
}

OC_EXPORT_API void destroy(ocarina::CUDADevice *device) {
    ocarina::delete_with_allocator(device);
}