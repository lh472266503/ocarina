//
// Created by Zero on 10/08/2022.
//

#include "cuda_compiler.h"
#include "cuda_device.h"
#include "ast/function.h"
#include "embed/cuda_device_builtin_embed.h"
#include "embed/cuda_device_math_embed.h"
#include "embed/cuda_device_resource_embed.h"
#include "cuda_codegen.h"
#include "rhi/context.h"
#include "core/util.h"

namespace ocarina {
#define CUDA_NVRTC_OPTIONS                 \
    "-std=c++17",                          \
        "-arch",                           \
        "compute_50",                      \
        "-use_fast_math",                  \
        "-restrict",                       \
        "-default-device",                 \
        "-include=cuda_device_builtin.h",  \
        "-include=cuda_device_math.h",     \
        "-include=cuda_device_resource.h", \


CUDACompiler::CUDACompiler(CUDADevice *device, const Function &f)
    : _device(device), _function(f) {}

ocarina::string CUDACompiler::compile(const string &cu, const string &fn, int sm) const noexcept {
    TIMER_TAG(compile, "compile " + fn);
    nvrtcProgram program{};
    ocarina::vector<const char *> compile_option = {CUDA_NVRTC_OPTIONS};
    std::array header_names{"cuda_device_builtin.h", "cuda_device_math.h", "cuda_device_resource.h"};
    std::array header_sources{cuda_device_builtin, cuda_device_math, cuda_device_resource};

    OC_NVRTC_CHECK(nvrtcCreateProgram(&program, cu.c_str(), fn.c_str(),
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

ocarina::string CUDACompiler::obtain_ptx() const noexcept {
    ocarina::string ptx_fn = _function.func_name() + ".ptx";
    string cu_fn = _function.func_name() + ".cu";
    ocarina::string ptx;
    Context *context = _device->context();
    if (!context->is_exist_cache(ptx_fn)) {
        if (!context->is_exist_cache(cu_fn)) {
            CUDACodegen codegen;
            codegen.emit(_function);
            const ocarina::string &cu = codegen.scratch().c_str();
            cout << cu << endl;
            context->write_cache(cu_fn, cu);
            ptx = compile(cu, cu_fn);
            context->write_cache(ptx_fn, ptx);
        } else {
            const ocarina::string &cu = context->read_cache(cu_fn);
            cout << cu << endl;
            ptx = compile(cu, cu_fn);
            context->write_cache(ptx_fn, ptx);
        }
    } else {
        ptx = context->read_cache(ptx_fn);
    }
    return ptx;
}

}// namespace ocarina