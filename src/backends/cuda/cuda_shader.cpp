//
// Created by zero on 2022/7/18.
//

#include "cuda_shader.h"
#include "util.h"
#include "cuda_device.h"
#include <optix_stack_size.h>
#include <optix_stubs.h>

namespace ocarina {

CUDAShader::CUDAShader(Device::Impl *device,
                       const Function &func)
    : _device(dynamic_cast<CUDADevice *>(device)),
      _function(func) {}

class CUDASimpleShader : public CUDAShader {
private:
    CUmodule _module{};
    CUfunction _func_handle{};

public:
    CUDASimpleShader(Device::Impl *device,
                     const ocarina::string &ptx,
                     const Function &f) : CUDAShader(device, f) {
        OC_CU_CHECK(cuModuleLoadData(&_module, ptx.c_str()));
        OC_CU_CHECK(cuModuleGetFunction(&_func_handle, _module, _function.func_name().c_str()));
#if CUDA_ARGUMENT_PUSH == 0
        _function.for_each_uniform_var([&](const UniformBinding &uniform) {
            const string &var_name = uniform.expression()->variable().name();
            CUdeviceptr ptr{};
            size_t size{};
            OC_CU_CHECK(cuModuleGetGlobal(&ptr, &size, _module, var_name.c_str()));
            OC_CU_CHECK(cuMemcpyHtoD(ptr, uniform.handle_ptr(), size));
        });
#endif
    }
    ~CUDASimpleShader() override {
        OC_CU_CHECK(cuModuleUnload(_module));
    }
    void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept override {
        uint3 grid_dim = make_uint3(1);
        uint3 block_dim = make_uint3(1);
        if (_function.has_configure()) {
            grid_dim = _function.grid_dim();
            block_dim = _function.block_dim();
        } else {
            grid_dim = (cmd->dispatch_dim() + block_dim - 1u) / block_dim;
        }
        OC_CU_CHECK(cuLaunchKernel(_func_handle, grid_dim.x, grid_dim.y, grid_dim.z,
                                   block_dim.x, block_dim.y, block_dim.z,
                                   0, reinterpret_cast<CUstream>(stream), cmd->args().data(), nullptr));
    }
    void compute_fit_size() noexcept {
        _device->use_context([&] {
            int min_grid_size;
            int auto_block_size;
            OC_CU_CHECK(cuOccupancyMaxPotentialBlockSize(&min_grid_size, &auto_block_size,
                                                         _func_handle, 0, 0, 0));

            _function.set_grid_dim(min_grid_size);
            _function.set_block_dim(auto_block_size);
        });
    }
};

class OptixShader : public CUDAShader {
private:
    OptixModule _optix_module{};
    OptixPipeline _optix_pipeline{};
    OptixPipelineCompileOptions _pipeline_compile_options = {};

public:
    void init_module(const string_view &ptx_code) {
        OptixModuleCompileOptions module_compile_options = {};
        // TODO: REVIEW THIS
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#else
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif
        _pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        _pipeline_compile_options.usesMotionBlur = false;
        _pipeline_compile_options.numPayloadValues = 2u;
        _pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        _pipeline_compile_options.numAttributeValues = 2;

#ifndef NDEBUG
        _pipeline_compile_options.exceptionFlags = (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                                                    OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                                    OPTIX_EXCEPTION_FLAG_DEBUG);
#else
        _pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        _pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        char log[2048];
        size_t log_size = sizeof(log);
        OC_OPTIX_CHECK_WITH_LOG(optixModuleCreateFromPTX(
                                    _device->optix_device_context(),
                                    &module_compile_options,
                                    &_pipeline_compile_options,
                                    ptx_code.data(), ptx_code.size(),
                                    log, &log_size, &_optix_module),
                                log);
    }

    OptixShader(Device::Impl *device,
                const ocarina::string &ptx,
                const Function &f) : CUDAShader(device, f) {
        init_module(ptx);
    }
    void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept override {
    }
    ~OptixShader() override {
    }
};

CUDAShader *CUDAShader::create(Device::Impl *device, const string &ptx, const Function &f) {
    if (f.is_raytracing()) {
        return ocarina::new_with_allocator<OptixShader>(device, ptx, f);
    } else {
        return ocarina::new_with_allocator<CUDASimpleShader>(device, ptx, f);
    }
}

}// namespace ocarina