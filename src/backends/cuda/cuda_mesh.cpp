//
// Created by Zero on 09/08/2022.
//

#include "cuda_mesh.h"
#include "cuda_device.h"
#include <optix_stack_size.h>
#include <optix.h>
#include <optix_stubs.h>

namespace ocarina {

CUDAMesh::~CUDAMesh() {
}

void CUDAMesh::build_bvh(const MeshBuildCommand *cmd) noexcept {
    _device->use_context([&] {
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
        accel_options.motionOptions.numKeys = 1;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gas_buffer_sizes;
        OC_OPTIX_CHECK(optixAccelComputeMemoryUsage(
            _device->optix_device_context(),
            &accel_options,
            &_build_input,
            1,// num_build_inputs
            &gas_buffer_sizes));

        OC_INFO_FORMAT("outputSizeInBytes is {} byte, tempSizeInBytes is {} byte",
                       gas_buffer_sizes.outputSizeInBytes,
                       gas_buffer_sizes.tempSizeInBytes);

        _blas_buffer = std::make_unique<Buffer<std::byte>>(_device, gas_buffer_sizes.outputSizeInBytes);

        Buffer temp_buffer = Buffer(_device, gas_buffer_sizes.tempSizeInBytes);
        Buffer compact_size_buffer = Buffer<uint64_t>(_device, 1);
        OptixAccelEmitDesc emit_desc;
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compact_size_buffer.handle();

        OC_OPTIX_CHECK(optixAccelBuild(_device->optix_device_context(), nullptr,
                                       &accel_options, &_build_input, 1,
                                       temp_buffer.handle(), gas_buffer_sizes.tempSizeInBytes,
                                       _blas_buffer->handle(), gas_buffer_sizes.outputSizeInBytes,
                                       &_blas_handle, &emit_desc, 1));

        auto compacted_gas_size = _device->download<size_t>(emit_desc.result);

        OC_INFO_FORMAT("compacted_gas_size is {} byte", compacted_gas_size);

        if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
            _blas_buffer = std::make_unique<Buffer<std::byte>>(_device, compacted_gas_size);
            OC_OPTIX_CHECK(optixAccelCompact(_device->optix_device_context(), nullptr,
                                             _blas_handle,
                                             _blas_buffer->handle(),
                                             compacted_gas_size,
                                             &_blas_handle));
            OC_INFO("optixAccelCompact was executed");
        }
        OC_CU_CHECK(cuCtxSynchronize());
    });
}

void CUDAMesh::init_build_input() noexcept {
    _build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    {
        _build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        _build_input.triangleArray.vertexStrideInBytes = _params.vert_stride;
        _build_input.triangleArray.numVertices = _params.vert_num;
        _build_input.triangleArray.vertexBuffers = reinterpret_cast<const CUdeviceptr *>(&_params.vert_handle);
    }
    {
        _build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        _build_input.triangleArray.indexStrideInBytes = _params.tri_stride;
        _build_input.triangleArray.numIndexTriplets = _params.tri_num;
        _build_input.triangleArray.indexBuffer = _params.tri_handle;
    }
    {
        static constexpr uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
        _build_input.triangleArray.flags = &geom_flags;
        _build_input.triangleArray.numSbtRecords = 1;
        _build_input.triangleArray.sbtIndexOffsetBuffer = reinterpret_cast<CUdeviceptr>(nullptr);
        _build_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        _build_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);
    }
}
}// namespace ocarina