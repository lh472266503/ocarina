//
// Created by Zero on 2022/8/8.
//

#include "optix_accel.h"
#include "cuda_device.h"

namespace ocarina {

void OptixAccel::build_bvh() noexcept {
    vector<OptixTraversableHandle> traversable_handles;
    for (const Mesh::Impl *mesh : _meshes) {
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
}

}// namespace ocarina