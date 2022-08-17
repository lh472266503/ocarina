//
// Created by Zero on 2022/8/17.
//

/// optix builtin start

static __forceinline__ __device__ oc_uint3 optixGetLaunchIndex() {
    unsigned int u0, u1, u2;
    asm( "call (%0), _optix_get_launch_index_x, ();" : "=r"( u0 ) : );
    asm( "call (%0), _optix_get_launch_index_y, ();" : "=r"( u1 ) : );
    asm( "call (%0), _optix_get_launch_index_z, ();" : "=r"( u2 ) : );
    return oc_make_uint3( u0, u1, u2 );
}

static __forceinline__ __device__ void optixSetPayload_0( unsigned int p ) {
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 0 ), "r"( p ) : );
}

static __forceinline__ __device__ unsigned int optixGetPayload_0() {
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 0 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int optixGetPayload_1() {
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 1 ) : );
    return result;
}

static __forceinline__ __device__ float3 optixGetWorldRayOrigin()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_world_ray_origin_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_world_ray_origin_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_world_ray_origin_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float3 optixGetWorldRayDirection()
{
    float f0, f1, f2;
    asm( "call (%0), _optix_get_world_ray_direction_x, ();" : "=f"( f0 ) : );
    asm( "call (%0), _optix_get_world_ray_direction_y, ();" : "=f"( f1 ) : );
    asm( "call (%0), _optix_get_world_ray_direction_z, ();" : "=f"( f2 ) : );
    return make_float3( f0, f1, f2 );
}

static __forceinline__ __device__ float optixGetRayTmin()
{
    float f0;
    asm( "call (%0), _optix_get_ray_tmin, ();" : "=f"( f0 ) : );
    return f0;
}

static __forceinline__ __device__ float optixGetRayTmax()
{
    float f0;
    asm( "call (%0), _optix_get_ray_tmax, ();" : "=f"( f0 ) : );
    return f0;
}

static __forceinline__ __device__ unsigned int optixGetInstanceId()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_instance_id, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int optixGetPrimitiveIndex()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_primitive_idx, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ float2 optixGetTriangleBarycentrics()
{
    float f0, f1;
    asm( "call (%0, %1), _optix_get_triangle_barycentrics, ();" : "=f"( f0 ), "=f"( f1 ) : );
    return make_float2( f0, f1 );
}


/// optix builtin end

inline void *unpackPointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void *ptr = reinterpret_cast<void *>( uptr );
    return ptr;
}

inline void packPointer(void *ptr, unsigned int &i0, unsigned int &i1) {
    const auto uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

inline void setPayloadOcclusion(bool occluded) {
    optixSetPayload_0(static_cast<unsigned int>( occluded ));
}

struct alignas(16) OCHit {
    oc_uint inst_id{};
    oc_uint prim_id{};
    oc_float2 bary;
};

template<typename T = OCHit>
inline T *getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
}


extern "C" __global__ void __closesthit__closest() {

}

extern "C" __global__ void __closesthit__any() {

}