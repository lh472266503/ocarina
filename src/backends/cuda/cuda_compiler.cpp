//
// Created by Zero on 10/08/2022.
//

#include "cuda_compiler.h"
#include "cuda_device.h"
#include "ast/function.h"
#include "embed/cuda_device_builtin_embed.h"
#include "embed/cuda_device_math_embed.h"
#include "embed/cuda_device_resource_embed.h"
#include "embed/optix_device_header_embed.h"
#include "cuda_codegen.h"
#include "rhi/context.h"
#include "core/util.h"
#include "dsl/dsl.h"

namespace ocarina {

static constexpr auto optix_include = OC_STRINGIFY(OPTIX_INCLUDE);

CUDACompiler::CUDACompiler(CUDADevice *device)
    : _device(device) {}

ocarina::string CUDACompiler::compile(const Function &function, int sm) const noexcept {

    function.set_raytracing(true);

    int ver_major = 0;
    int ver_minor = 0;
    OC_NVRTC_CHECK(nvrtcVersion(&ver_major, &ver_minor));
    int nvrtc_version = ver_major * 10000 + ver_minor * 100;
    auto nvrtc_option = fmt::format("-DLC_NVRTC_VERSION={}", nvrtc_version);
    std::vector header_names{"cuda_device_builtin.h", "cuda_device_math.h", "cuda_device_resource.h"};
    std::vector header_sources{cuda_device_builtin, cuda_device_math, cuda_device_resource};
    auto compute_sm = ocarina::format("-arch=compute_{}", sm);
    auto rt_option = fmt::format("-DLC_OPTIX_VERSION={}", 70300);
    auto const_option = fmt::format("-Dlc_constant={}", nvrtc_version <= 110200 ? "const" : "constexpr");
    ocarina::vector<const char *> compile_option = {
        "--std=c++17",
        compute_sm.c_str(),
        const_option.c_str(),
        rt_option.c_str(),
        nvrtc_option.c_str(),
        "-default-device",
        "--use_fast_math",
        "-restrict",
#ifndef NDEBUG
        "-lineinfo",
#endif
        "-extra-device-vectorization",
        "-dw",
        "-w"};
    ocarina::vector<string> includes;
    if (function.is_raytracing()) {
        includes.push_back(ocarina::format("-I {}", optix_include));
        compile_option.push_back(includes.back().c_str());
        header_names.push_back("optix_device_header.h");
        header_sources.push_back(optix_device_header);
    }
    for (const auto &header_name : header_names) {
        includes.push_back(ocarina::format("-include={}", header_name));
        compile_option.push_back(includes.back().c_str());
    }

    uint64_t ext_hash = hash64(hash64_list(compile_option), hash64_list(header_sources));

    auto compile = [&](const string &cu, const string &fn, int sm) -> string {
        TIMER_TAG(compile, "compile " + fn);
        nvrtcProgram program{};
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
    };

    string fn = function.func_name(ext_hash, function.description());

    ocarina::string ptx_fn = fn + ".ptx";
    string cu_fn = fn + ".cu";
    ocarina::string ptx;
    FileManager *context = _device->context();
    if (!context->is_exist_cache(ptx_fn)) {
        if (!context->is_exist_cache(cu_fn)) {
            CUDACodegen codegen{Env::code_obfuscation()};
            codegen.emit(function);
            const ocarina::string &cu = codegen.scratch().c_str();
            //            cout << cu << endl;
            context->write_global_cache(cu_fn, cu);
            ptx = compile(cu, cu_fn, 75);
            context->write_global_cache(ptx_fn, ptx);
        } else {
            const ocarina::string &cu = context->read_global_cache(cu_fn);
            //            cout << cu << endl;
            ptx = compile(cu, cu_fn, sm);
            context->write_global_cache(ptx_fn, ptx);
        }
    } else {
        OC_INFO_FORMAT("find ptx file {}", ptx_fn);
        ptx = context->read_global_cache(ptx_fn);
    }
    return ptx;
}

}// namespace ocarina