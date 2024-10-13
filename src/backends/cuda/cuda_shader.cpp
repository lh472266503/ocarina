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
    : device_(dynamic_cast<CUDADevice *>(device)),
      function_(func) {}

class CUDASimpleShader : public CUDAShader {
private:
    CUmodule module_{};
    CUfunction func_handle_{};

public:
    CUDASimpleShader(Device::Impl *device,
                     const ocarina::string &ptx,
                     const Function &f) : CUDAShader(device, f) {
        OC_CU_CHECK(cuModuleLoadData(&module_, ptx.c_str()));
        OC_CU_CHECK(cuModuleGetFunction(&func_handle_, module_, function_.func_name().c_str()));
    }
    ~CUDASimpleShader() override {
        OC_CU_CHECK(cuModuleUnload(module_));
    }
    void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept override {
        uint3 grid_dim = make_uint3(1);
        uint3 block_dim = make_uint3(1);
        if (function_.has_configure()) {
            grid_dim = function_.grid_dim();
            block_dim = function_.block_dim();
        } else {
            grid_dim = (cmd->dispatch_dim() + block_dim - 1u) / block_dim;
        }
        OC_CU_CHECK(cuLaunchKernel(func_handle_, grid_dim.x, grid_dim.y, grid_dim.z,
                                   block_dim.x, block_dim.y, block_dim.z,
                                   0, reinterpret_cast<CUstream>(stream), cmd->args().data(), nullptr));
    }
    void compute_fit_size() noexcept override {
        device_->use_context([&] {
            int min_grid_size;
            int auto_block_size;
            OC_CU_CHECK(cuOccupancyMaxPotentialBlockSize(&min_grid_size, &auto_block_size,
                                                         func_handle_, 0, 0, 0));

            function_.set_grid_dim(min_grid_size);
            function_.set_block_dim(auto_block_size);
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
    OptixModule optix_module_{};
    OptixPipeline optix_pipeline_{};
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    ProgramGroupTable program_group_table_;
    Buffer<SBTRecord> sbt_records_{};
    OptixShaderBindingTable sbt_{};
    Buffer<std::byte> params_;

public:
    void init_module(const string_view &ptx_code) {
        OptixModuleCompileOptions module_compile_options = {};
        static constexpr std::array<uint, 1> ray_trace_payload_semantics{
            OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ,
        };

        static constexpr std::array<uint, 2> ray_query_payload_semantics{
            OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ | OPTIX_PAYLOAD_SEMANTICS_AH_READ,
            OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE | OPTIX_PAYLOAD_SEMANTICS_IS_READ | OPTIX_PAYLOAD_SEMANTICS_AH_READ,
        };

        std::array<OptixPayloadType, 1> payload_types{};
        payload_types[0].numPayloadValues = ray_trace_payload_semantics.size();
        payload_types[0].payloadSemantics = ray_trace_payload_semantics.data();
        // payload_types[1].numPayloadValues = ray_query_payload_semantics.size();
        // payload_types[1].payloadSemantics = ray_query_payload_semantics.data();

        // TODO: REVIEW THIS
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        //#ifndef NDEBUG
        //        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
//        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        //#else
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        // module_compile_options.numPayloadTypes = payload_types.size();
        // module_compile_options.payloadTypes = payload_types.data();

        //        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        //#endif
        pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options_.usesMotionBlur = false;
        pipeline_compile_options_.numPayloadValues = 4;
        pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        //        pipeline_compile_options_.numAttributeValues = 2;

        //#ifndef NDEBUG
        //        pipeline_compile_options_.exceptionFlags = (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
        //                                                    OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        //                                                    OPTIX_EXCEPTION_FLAG_DEBUG);
        //#else
        pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        //#endif
        pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
        char log[2048];
        size_t log_size = sizeof(log);
        OC_OPTIX_CHECK_WITH_LOG(optixModuleCreate(
                                    device_->optix_device_context(),
                                    &module_compile_options,
                                    &pipeline_compile_options_,
                                    ptx_code.data(), ptx_code.size(),
                                    log, &log_size, &optix_module_),
                                log);
    }

    void build_pipeline(OptixDeviceContext optix_device_context) noexcept {

        constexpr int max_trace_depth = 1;

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
//        pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        char log[2048];
        size_t sizeof_log = sizeof(log);

        OC_OPTIX_CHECK_WITH_LOG(optixPipelineCreate(
                                    optix_device_context,
                                    &pipeline_compile_options_,
                                    &pipeline_link_options,
                                    (OptixProgramGroup *)&program_group_table_,
                                    program_group_table_.size(),
                                    log, &sizeof_log,
                                    &optix_pipeline_),
                                log);

        // Set shaders stack sizes.
        OptixStackSizes stack_sizes = {};
        OC_OPTIX_CHECK(optixUtilAccumulateStackSizes(program_group_table_.raygen_group, &stack_sizes,optix_pipeline_));
        OC_OPTIX_CHECK(optixUtilAccumulateStackSizes(program_group_table_.miss_closest_group, &stack_sizes,optix_pipeline_));
        OC_OPTIX_CHECK(optixUtilAccumulateStackSizes(program_group_table_.hit_closest_group, &stack_sizes,optix_pipeline_));
        OC_OPTIX_CHECK(optixUtilAccumulateStackSizes(program_group_table_.hit_any_group, &stack_sizes,optix_pipeline_));

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
        OC_OPTIX_CHECK(optixPipelineSetStackSize(optix_pipeline_,
                                                 direct_callable_stack_size_from_traversal,
                                                 direct_callable_stack_size_from_state,
                                                 continuation_stack_size,
                                                 2// maxTraversableDepth
                                                 ));
    }

    void build_sbt(ProgramGroupTable program_group_table) {
        sbt_records_ = Buffer<SBTRecord>(device_, 4, "OptixShader::sbt_records_");
        SBTRecord sbt[ProgramGroupTable::size()] = {};
        OC_OPTIX_CHECK(optixSbtRecordPackHeader(program_group_table.raygen_group, &sbt[0]));
        OC_OPTIX_CHECK(optixSbtRecordPackHeader(program_group_table.hit_closest_group, &sbt[1]));
        OC_OPTIX_CHECK(optixSbtRecordPackHeader(program_group_table.hit_any_group, &sbt[2]));
        OC_OPTIX_CHECK(optixSbtRecordPackHeader(program_group_table.miss_closest_group, &sbt[3]));
        sbt_records_.upload_immediately(sbt);

        sbt_.raygenRecord = sbt_records_.ptr<CUdeviceptr>();
        sbt_.hitgroupRecordBase = sbt_records_.address<CUdeviceptr>(1);
        sbt_.hitgroupRecordStrideInBytes = sizeof(SBTRecord);
        sbt_.hitgroupRecordCount = 2;
        sbt_.missRecordBase = sbt_records_.address<CUdeviceptr>(3);
        sbt_.missRecordStrideInBytes = sizeof(SBTRecord);
        sbt_.missRecordCount = 1;
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
            raygen_prog_group_desc.raygen.module = optix_module_;
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
            hit_prog_group_desc.hitgroup.moduleCH = optix_module_;
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
            hit_prog_group_desc.hitgroup.moduleCH = optix_module_;
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
        device_->init_optix_context();
        init_module(ptx);
        string raygen_entry = function_.func_name();
        ProgramName entries{
            raygen_entry.c_str(),
            "__closesthit__closest",
            "__closesthit__any"};
        program_group_table_ = create_program_groups(device_->optix_device_context(), entries);
        build_sbt(program_group_table_);
        build_pipeline(device_->optix_device_context());
    }

    void launch(handle_ty stream, ShaderDispatchCommand *cmd) noexcept override {
        auto dim = cmd->dispatch_dim();
        uint x = dim.x;
        uint y = dim.y;
        uint z = dim.z;
        auto cu_stream = reinterpret_cast<CUstream>(stream);
        size_t total_size = cmd->params_size();
        if (!params_.valid() || params_.size() < total_size) {

            OC_CU_CHECK(cuMemFreeAsync(params_.handle(), cu_stream));
            OC_CU_CHECK(cuMemAllocAsync(reinterpret_cast<CUdeviceptr *>(params_.handle_ptr()), total_size, cu_stream));

            std::function<void()> func = [this, total_size]() {
                params_.set_size(total_size);
                params_.set_device(device_);
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
        OC_CU_CHECK(cuMemcpyHtoDAsync(params_.handle(), arguments.data(),
                                      arguments.size(), cu_stream));

        OC_OPTIX_CHECK(optixLaunch(optix_pipeline_,
                                   cu_stream,
                                   params_.handle(),
                                   arguments.size(),
                                   &sbt_,
                                   x, y, z));
    }
    ~OptixShader() override {
        program_group_table_.clear();
        optixModuleDestroy(optix_module_);
        optixPipelineDestroy(optix_pipeline_);
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