//
// Created by Zero on 09/08/2022.
//

#include "cuda_mesh.h"
#include "cuda_device.h"

namespace ocarina {

void CUDAMesh::init_build_input() noexcept {
    _build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    {
        _build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        _build_input.triangleArray.vertexStrideInBytes = _params.vert_stride;
        _build_input.triangleArray.numVertices = _params.vert_num;
        _build_input.triangleArray.vertexBuffers = reinterpret_cast<const CUdeviceptr*>(_params.vert_handle_address);
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