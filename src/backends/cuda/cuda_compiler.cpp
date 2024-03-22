//
// Created by Zero on 10/08/2022.
//

#include "cuda_compiler.h"
#include "cuda_device.h"
#include "ast/function.h"
#include "cuda_codegen.h"
#include "util/file_manager.h"
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
    std::vector header_names{"cuda_device_type.h","cuda_device_builtin.h", "cuda_device_math.h", "cuda_device_resource.h"};
    std::vector<string> header_sources;
    std::vector<const char *> header_sources_ptr;

    for (auto fn : header_names) {
        string source = FileManager::read_file(string("cuda/") + fn);
        header_sources_ptr.push_back(source.c_str());
        header_sources.push_back(ocarina::move(source));
    }

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
        string source = FileManager::read_file(string("cuda/optix_device_header.h"));
        header_sources_ptr.push_back(source.c_str());
        header_sources.push_back(ocarina::move(source));
    }
    for (const auto &header_name : header_names) {
        includes.push_back(ocarina::format("-include={}", header_name));
        compile_option.push_back(includes.back().c_str());
    }

    uint64_t ext_hash = hash64(hash64_list(compile_option), hash64_list(header_sources_ptr));

    auto compile = [&](const string &cu, const string &fn, int sm) -> string {
        TIMER_TAG(compile, "compile " + fn);
        nvrtcProgram program{};
        OC_NVRTC_CHECK(nvrtcCreateProgram(&program, cu.c_str(), fn.c_str(),
                                          header_names.size(), header_sources_ptr.data(),
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
    FileManager *file_manager = _device->file_manager();
    if (!file_manager->is_exist_cache(ptx_fn)) {
        OC_INFO_FORMAT("miss ptx file {}", ptx_fn);
        if (!file_manager->is_exist_cache(cu_fn)) {
            CUDACodegen codegen{Env::code_obfuscation()};
            codegen.emit(function);
            const ocarina::string &cu = codegen.scratch().c_str();
            file_manager->write_global_cache(cu_fn, cu);
            ptx = compile(cu, cu_fn, 75);
            file_manager->write_global_cache(ptx_fn, ptx);
        } else {
            const ocarina::string &cu = file_manager->read_global_cache(cu_fn);
            ptx = compile(cu, cu_fn, sm);
            file_manager->write_global_cache(ptx_fn, ptx);
        }
    } else {
        OC_INFO_FORMAT("find ptx file {}", ptx_fn);
        ptx = file_manager->read_global_cache(ptx_fn);
    }
    return ptx;
}

}// namespace ocarina