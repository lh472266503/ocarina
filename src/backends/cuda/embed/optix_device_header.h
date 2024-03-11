//
// Created by Zero on 2022/8/17.
//

#include "optix.h"

template<typename... Args>
__device__ inline void trace(OptixTraversableHandle handle,
                             Ray ray,
                             oc_uint flags,
                             oc_uint SBToffset,
                             oc_uint SBTstride,
                             oc_uint missSBTIndex,
                             Args &&...payload) {
    auto origin = ::make_float3(ray.m0.x, ray.m0.y, ray.m0.z);
    auto direction = ::make_float3(ray.m1.x, ray.m1.y, ray.m1.z);

    optixTrace(
        handle,
        origin,
        direction,
        ray.m0.w,
        ray.m1.w,
        0.0f,// rayTime
        OptixVisibilityMask(1),
        flags,
        SBToffset,   // SBT offset
        SBTstride,   // SBT stride
        missSBTIndex,// missSBTIndex
        std::forward<Args>(payload)...);
}

__device__ oc_uint3 getLaunchIndex() {
    auto idx = optixGetLaunchIndex();
    return oc_make_uint3(idx.x, idx.y, idx.z);
}

__device__ oc_uint3 getLaunchDim() {
    auto idx = optixGetLaunchDimensions();
    return oc_make_uint3(idx.x, idx.y, idx.z);
}

__device__ inline void *unpack_pointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void *ptr = reinterpret_cast<void *>(uptr);
    return ptr;
}

__device__ inline void pack_pointer(void *ptr, unsigned int &i0, unsigned int &i1) {
    const auto uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

__device__ inline void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>(occluded));
}

__device__ inline oc_float2 getTriangleBarycentric() {
    auto barycentric = optixGetTriangleBarycentrics();
    return oc_make_float2(1 - barycentric.y - barycentric.x, barycentric.x);
}

__device__ inline Hit oc_trace_closest(OptixTraversableHandle handle,
                                          Ray ray) {
    unsigned int u0, u1;
    Hit hit;
    pack_pointer(&hit, u0, u1);
    trace(handle, ray, OPTIX_RAY_FLAG_DISABLE_ANYHIT,
          0,        // SBT offset
          0,           // SBT stride
          0,        // missSBTIndex
          u0, u1);
    return hit;
}

__device__ inline bool oc_trace_any(OptixTraversableHandle handle, Ray ray) {
    unsigned int occluded = 0u;
    trace(handle, ray, OPTIX_RAY_FLAG_DISABLE_ANYHIT
                       | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
          1,        // SBT offset
          0,           // SBT stride
          0,        // missSBTIndex
          occluded);
    return bool(occluded);
}

__device__ inline Hit getClosestHit() {
    Hit ret;
    ret.m0 = optixGetInstanceId();
    ret.m1 = optixGetPrimitiveIndex();
    ret.m2 = getTriangleBarycentric();
    return ret;
}

template<typename T = Hit>
__device__ inline T *getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpack_pointer(u0, u1));
}

extern "C" __global__ void __closesthit__closest() {
    Hit *hit = getPRD<Hit>();
    *hit = getClosestHit();
}

extern "C" __global__ void __closesthit__any() {
    setPayloadOcclusion(true);
}


