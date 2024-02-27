//
// Created by zero on 2022/7/18.
//

#include "cuda_shader.h"
#include "util.h"
#include "cuda_device.h"
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include "cuda_command_visitor.h"

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
    void compute_fit_size() noexcept override {
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

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTRecord {
    std::byte data[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct ProgramName {
    const char *raygen{};
    const char *closesthit_closest{};
    const char *closesthit_any{};
};

struct ProgramGroupTable {
    OptixProgramGroup raygen_group{nullptr};
    OptixProgramGroup miss_closest_group{nullptr};
    OptixProgramGroup hit_closest_group{nullptr};
    OptixProgramGroup hit_any_group{nullptr};

    ProgramGroupTable() = default;

    static constexpr auto size() {
        return sizeof(ProgramGroupTable) / sizeof(OptixProgramGroup);
    }

    void clear() const {
        OC_OPTIX_CHECK(optixProgramGroupDestroy(raygen_group));
        OC_OPTIX_CHECK(optixProgramGroupDestroy(hit_closest_group));
        OC_OPTIX_CHECK(optixProgramGroupDestroy(hit_any_group));
        OC_OPTIX_CHECK(optixProgramGroupDestroy(miss_closest_group));
    }
};

class OptixShader : public CUDAShader {
private:
    OptixModule _optix_module{};
    OptixPipeline _optix_pipeline{};
    OptixPipelineCompileOptions _pipeline_compile_options = {};
    ProgramGroupTable _program_group_table;
    Buffer<SBTRecord> _sbt_records{};
    OptixShaderBindingTable _sbt{};
    Buffer<std::byte> _params;

public:
    void init_module(const string_view &ptx_code) {
        OptixModuleCompileOptions module_compile_options = {};
        // TODO: REVIEW THIS
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        //#ifndef NDEBUG
        //        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
        //#else
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        //        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        //#endif
        _pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        _pipeline_compile_options.usesMotionBlur = false;
        _pipeline_compile_options.numPayloadValues = 4;
        _pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        //        _pipeline_compile_options.numAttributeValues = 2;

        //#ifndef NDEBUG
        //        _pipeline_compile_options.exceptionFlags = (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
        //                                                    OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        //                                                    OPTIX_EXCEPTION_FLAG_DEBUG);
        //#else
        _pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        //#endif
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

    void build_pipeline(OptixDeviceContext optix_device_context) noexcept {

        constexpr int max_trace_depth = 1;

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        char log[2048];
        size_t sizeof_log = sizeof(log);

        OC_OPTIX_CHECK_WITH_LOG(optixPipelineCreate(
                                    optix_device_context,
                                    &_pipeline_compile_options,
                                    &pipeline_link_options,
                                    (OptixProgramGroup *)&_program_group_table,
                                    _program_group_table.size(),
                                    log, &sizeof_log,
                                    &_optix_pipeline),
                                log);

        // Set shaders stack sizes.
        OptixStackSizes stack_sizes = {};
        OC_OPTIX_CHECK(optixUtilAccumulateStackSizes(_program_group_table.raygen_group, &stack_sizes));
        OC_OPTIX_CHECK(optixUtilAccumulateStackSizes(_program_group_table.miss_closest_group, &stack_sizes));
        OC_OPTIX_CHECK(optixUtilAccumulateStackSizes(_program_group_table.hit_closest_group, &stack_sizes));
        OC_OPTIX_CHECK(optixUtilAccumulateStackSizes(_program_group_table.hit_any_group, &stack_sizes));

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OC_OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                                  max_trace_depth,
                                                  0,// maxCCDepth
                                                  0,// maxDCDEpth
                                                  &direct_callable_stack_size_from_traversal,
                                                  &direct_callable_stack_size_from_state,
                                                  &continuation_stack_size));
        OC_OPTIX_CHECK(optixPipelineSetStackSize(_optix_pipeline,
                                                 direct_callable_stack_size_from_traversal,
                                                 direct_callable_stack_size_from_state,
                                                 continuation_stack_size,
                                                 2// maxTraversableDepth
                                                 ));
    }

    void build_sbt(ProgramGroupTable program_group_table) {
        _sbt_records = Buffer<SBTRecord>(_device, 4, "OptixShader::_sbt_records");
        SBTRecord sbt[4] = {};
        OC_OPTIX_CHECK(optixSbtRecordPackHeader(program_group_table.raygen_group, &sbt[0]));
        OC_OPTIX_CHECK(optixSbtRecordPackHeader(program_group_table.hit_closest_group, &sbt[1]));
        OC_OPTIX_CHECK(optixSbtRecordPackHeader(program_group_table.hit_any_group, &sbt[2]));
        OC_OPTIX_CHECK(optixSbtRecordPackHeader(program_group_table.miss_closest_group, &sbt[3]));
        _sbt_records.upload_immediately(sbt);

        _sbt.raygenRecord = _sbt_records.ptr<CUdeviceptr>();
        _sbt.hitgroupRecordBase = _sbt_records.address<CUdeviceptr>(1);
        _sbt.hitgroupRecordStrideInBytes = sizeof(SBTRecord);
        _sbt.hitgroupRecordCount = 2;
        _sbt.missRecordBase = _sbt_records.address<CUdeviceptr>(3);
        _sbt.missRecordStrideInBytes = sizeof(SBTRecord);
        _sbt.missRecordCount = 1;
    }

    ProgramGroupTable create_program_groups(OptixDeviceContext optix_device_context,
                                            const ProgramName &program_name) {
        OptixProgramGroupOptions program_group_options = {};
        char log[2048];
        size_t sizeof_log = sizeof(log);
        ProgramGroupTable program_group_table;
        {
            OptixProgramGroupDesc raygen_prog_group_desc = {};
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = _optix_module;
            raygen_prog_group_desc.raygen.entryFunctionName = program_name.raygen;
            OC_OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                                        optix_device_context,
                                        &raygen_prog_group_desc,
                                        1,// num program groups
                                        &program_group_options,
                                        log,
                                        &sizeof_log,
                                        &(program_group_table.raygen_group)),
                                    log);
        }
        {
            OptixProgramGroupDesc hit_prog_group_desc = {};
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH = _optix_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = program_name.closesthit_closest;
            sizeof_log = sizeof(log);

            OC_OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                                        optix_device_context,
                                        &hit_prog_group_desc,
                                        1,// num program groups
                                        &program_group_options,
                                        log,
                                        &sizeof_log,
                                        &(program_group_table.hit_closest_group)),
                                    log);

            memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
            hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hit_prog_group_desc.hitgroup.moduleCH = _optix_module;
            hit_prog_group_desc.hitgroup.entryFunctionNameCH = program_name.closesthit_any;
            sizeof_log = sizeof(log);

            OC_OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                                        optix_device_context,
                                        &hit_prog_group_desc,
                                        1,// num program groups
                                        &program_group_options,
                                        log,
                                        &sizeof_log,
                                        &(program_group_table.hit_any_group)),
                                    log);
        }
        {
            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            sizeof_log = sizeof(log);
            OC_OPTIX_CHECK_WITH_LOG(optixProgramGroupCreate(
                                        optix_device_context,
                                        &miss_prog_group_desc,
                                        1,// num program groups
                                        &program_group_options,
                                        log,
                                        &sizeof_log,
                                        &(program_group_table.miss_closest_group)),
                                    log);
        }

        return program_group_table;
    }

    OptixShader(Device::Impl *device,
                const ocarina::string &ptx,
                const Function &f) : CUDAShader(device, f) {
        _device->init_optix_context();
        init_module(ptx);
        string raygen_entry = _function.func_name();
        ProgramName entries{
            raygen_entry.c_str(),
            "__closesthit__closest",
            "__closesthit__any"};
        _program_group_table = create_program_groups(_device->optix_device_context(), entries);
        build_sbt(_program_group_table);
        build_pipeline(_device->optix_device_context());
    }

    void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept override {
        auto dim = cmd->dispatch_dim();
        uint x = dim.x;
        uint y = dim.y;
        uint z = dim.z;
        auto cu_stream = reinterpret_cast<CUstream>(stream);
        size_t total_size = cmd->params_size();
        if (!_params.valid() || _params.size() < total_size) {

            OC_CU_CHECK(cuMemFreeAsync(_params.handle(), cu_stream));
            OC_CU_CHECK(cuMemAllocAsync(reinterpret_cast<CUdeviceptr *>(_params.handle_ptr()), total_size, cu_stream));

            std::function<void()> func = [this, total_size]() {
                _params.set_size(total_size);
                _params.set_device(_device);
            };
            auto ptr = new_with_allocator<std::function<void()>>(ocarina::move(func));
            OC_CU_CHECK(cuLaunchHostFunc(
                cu_stream, [](void *ptr) {
                    auto func = reinterpret_cast<std::function<void()> *>(ptr);
                    (*func)();
                    delete_with_allocator(func);
                },
                ptr));
            OC_CU_CHECK(cuStreamSynchronize(cu_stream));
        }
        auto arguments = cmd->argument_data();
        OC_CU_CHECK(cuMemcpyHtoDAsync(_params.handle(), arguments.data(),
                                      arguments.size(), cu_stream));

        OC_OPTIX_CHECK(optixLaunch(_optix_pipeline,
                                   cu_stream,
                                   _params.handle(),
                                   arguments.size(),
                                   &_sbt,
                                   x, y, z));
    }
    ~OptixShader() override {
        _program_group_table.clear();
        optixModuleDestroy(_optix_module);
        optixPipelineDestroy(_optix_pipeline);
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