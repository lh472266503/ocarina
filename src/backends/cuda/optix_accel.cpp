//
// Created by Zero on 2022/8/8.
//

#include "optix_accel.h"
#include "cuda_device.h"
#include <optix_stubs.h>
#include "cuda_command_visitor.h"

namespace ocarina {

namespace detail {
void mat4x4_to_array12(float4x4 mat, float *output) {

    output[0] = mat[0][0];
    output[1] = mat[1][0];
    output[2] = mat[2][0];
    output[3] = mat[3][0];

    output[4] = mat[0][1];
    output[5] = mat[1][1];
    output[6] = mat[2][1];
    output[7] = mat[3][1];

    output[8] = mat[0][2];
    output[9] = mat[1][2];
    output[10] = mat[2][2];
    output[11] = mat[3][2];
}
}// namespace detail

void OptixAccel::add_mesh(const RHIMesh::Impl *mesh, ocarina::float4x4 mat) noexcept {
    _meshes.push_back(mesh);
    _transforms.push_back(mat);
}

void OptixAccel::clear() noexcept {
    _meshes.clear();
    _transforms.clear();
    _tlas_buffer.destroy();
    _instances.destroy();
    if (_tlas_handle != 0) {
        OC_CU_CHECK(cuMemFree(_tlas_handle));
        _tlas_handle = 0;
    }
}

void OptixAccel::build_bvh(CUDACommandVisitor *visitor) noexcept {
    _device->use_context([&] {
        vector<OptixTraversableHandle> traversable_handles;
        traversable_handles.reserve(_meshes.size());
        for (const RHIMesh::Impl *mesh : _meshes) {
            traversable_handles.push_back(mesh->blas_handle());
        }
        size_t instance_num = _meshes.size();
        OptixBuildInput instance_input = {};
        _instances = Buffer<OptixInstance>(_device, instance_num);
        instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        instance_input.instanceArray.numInstances = instance_num;
        instance_input.instanceArray.instances = _instances.ptr<CUdeviceptr>();

        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes ias_buffer_sizes;
        OC_OPTIX_CHECK(optixAccelComputeMemoryUsage(_device->optix_device_context(),
                                                    &accel_options, &instance_input,
                                                    1,// num build inputs
                                                    &ias_buffer_sizes));
        OC_INFO_FORMAT("tlas: outputSizeInBytes is {} byte, tempSizeInBytes is {} byte",
                       ias_buffer_sizes.outputSizeInBytes,
                       ias_buffer_sizes.tempSizeInBytes);

        auto ias_buffer = Buffer<std::byte>(_device, ias_buffer_sizes.outputSizeInBytes);
        auto temp_buffer = Buffer<std::byte>(_device, ias_buffer_sizes.tempSizeInBytes);
        Buffer compact_size_buffer = Buffer<uint64_t>(_device, 1);
        OptixAccelEmitDesc emit_desc;
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = compact_size_buffer.handle();

        vector<OptixInstance> optix_instances;
        optix_instances.reserve(instance_num);

        for (int i = 0; i < instance_num; ++i) {
            float4x4 transform = _transforms[i];
            OptixInstance optix_instance;
            optix_instance.traversableHandle = traversable_handles[i];
            optix_instance.flags = OPTIX_INSTANCE_FLAG_NONE;
            optix_instance.instanceId = i;
            optix_instance.visibilityMask = 1;
            optix_instance.sbtOffset = 0;
            detail::mat4x4_to_array12(transform, optix_instance.transform);
            optix_instances.push_back(optix_instance);
        }

        _instances.upload_immediately(optix_instances.data());
        OC_OPTIX_CHECK(optixAccelBuild(_device->optix_device_context(),
                                       nullptr, &accel_options,
                                       &instance_input, 1,
                                       temp_buffer.ptr<CUdeviceptr>(),
                                       ias_buffer_sizes.tempSizeInBytes,
                                       ias_buffer.ptr<CUdeviceptr>(),
                                       ias_buffer_sizes.outputSizeInBytes,
                                       &_tlas_handle, &emit_desc, 1));

        auto compacted_gas_size = _device->download<size_t>(emit_desc.result);
        OC_INFO_FORMAT("tlas : compacted_gas_size is {} byte", compacted_gas_size);

        if (compacted_gas_size < ias_buffer_sizes.outputSizeInBytes) {
            _tlas_buffer = Buffer<std::byte>(_device, compacted_gas_size);
            OC_OPTIX_CHECK(optixAccelCompact(_device->optix_device_context(), nullptr,
                                             _tlas_handle,
                                             _tlas_buffer.ptr<CUdeviceptr>(),
                                             compacted_gas_size,
                                             &_tlas_handle));
            OC_INFO("tlas : optixAccelCompact was executed");
        }
        OC_INFO("tlas handle is ", _tlas_handle);
        OC_CU_CHECK(cuCtxSynchronize());
    });
}

size_t OptixAccel::data_size() const noexcept {
    return CUDADevice::size(Type::Tag::ACCEL);
}

size_t OptixAccel::data_alignment() const noexcept {
    return CUDADevice::alignment(Type::Tag::ACCEL);
}

}// namespace ocarina