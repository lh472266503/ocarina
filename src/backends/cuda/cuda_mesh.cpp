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
        _build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
        _build_input.triangleArray.numVertices = vert_num;
        _build_input.triangleArray.vertexBuffers = &_v_handle;
    }
    {
        _build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        _build_input.triangleArray.indexStrideInBytes = sizeof(Triangle);
        _build_input.triangleArray.numIndexTriplets = tri_num;
        _build_input.triangleArray.indexBuffer = _t_handle;
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