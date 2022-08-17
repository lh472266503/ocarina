//
// Created by Zero on 2022/8/17.
//

/// optix builtin start

static __forceinline__ __device__ oc_uint3 oc_optixGetLaunchIndex() {
    unsigned int u0, u1, u2;
    asm( "call (%0), _optix_get_launch_index_x, ();" : "=r"( u0 ) : );
    asm( "call (%0), _optix_get_launch_index_y, ();" : "=r"( u1 ) : );
    asm( "call (%0), _optix_get_launch_index_z, ();" : "=r"( u2 ) : );
    return oc_make_uint3( u0, u1, u2 );
}

static __forceinline__ __device__ void oc_optixSetPayload_0( unsigned int p ) {
    asm volatile( "call _optix_set_payload, (%0, %1);" : : "r"( 0 ), "r"( p ) : );
}

static __forceinline__ __device__ unsigned int oc_optixGetPayload_0() {
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 0 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int oc_optixGetPayload_1() {
    unsigned int result;
    asm volatile( "call (%0), _optix_get_payload, (%1);" : "=r"( result ) : "r"( 1 ) : );
    return result;
}

static __forceinline__ __device__ unsigned int oc_optixGetInstanceId()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_instance_id, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ unsigned int oc_optixGetPrimitiveIndex()
{
    unsigned int u0;
    asm( "call (%0), _optix_read_primitive_idx, ();" : "=r"( u0 ) : );
    return u0;
}

static __forceinline__ __device__ oc_float2 oc_optixGetTriangleBarycentrics()
{
    float f0, f1;
    asm( "call (%0, %1), _optix_get_triangle_barycentrics, ();" : "=f"( f0 ), "=f"( f1 ) : );
    return oc_make_float2( f0, f1 );
}

template <typename... Payload>
static __forceinline__ __device__ void oc_optixTrace( OptixPayloadTypeID     type,
                                                   OptixTraversableHandle handle,
                                                   oc_float3              rayOrigin,
                                                   oc_float3              rayDirection,
                                                   float                  tmin,
                                                   float                  tmax,
                                                   float                  rayTime,
                                                   OptixVisibilityMask    visibilityMask,
                                                   unsigned int           rayFlags,
                                                   unsigned int           SBToffset,
                                                   unsigned int           SBTstride,
                                                   unsigned int           missSBTIndex,
                                                   Payload&...            payload )
{
    // std::is_same compares each type in the two TypePacks to make sure that all types are unsigned int.
    // TypePack 1    unsigned int    T0      T1      T2   ...   Tn-1        Tn
    // TypePack 2      T0            T1      T2      T3   ...   Tn        unsigned int
    static_assert( sizeof...( Payload ) <= 32, "Only up to 32 payload values are allowed." );
    static_assert( std::is_same<optix_internal::TypePack<unsigned int, Payload...>, optix_internal::TypePack<Payload..., unsigned int>>::value,
                   "All payload parameters need to be unsigned int." );

    float        ox = rayOrigin.x, oy = rayOrigin.y, oz = rayOrigin.z;
    float        dx = rayDirection.x, dy = rayDirection.y, dz = rayDirection.z;
    unsigned int p[33]       = { 0, payload... };
    int          payloadSize = (int)sizeof...( Payload );

    asm volatile(
        "call"
        "(%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%"
        "29,%30,%31),"
        "_optix_trace_typed_32,"
        "(%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%"
        "59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80);"
        : "=r"( p[1] ), "=r"( p[2] ), "=r"( p[3] ), "=r"( p[4] ), "=r"( p[5] ), "=r"( p[6] ), "=r"( p[7] ),
          "=r"( p[8] ), "=r"( p[9] ), "=r"( p[10] ), "=r"( p[11] ), "=r"( p[12] ), "=r"( p[13] ), "=r"( p[14] ),
          "=r"( p[15] ), "=r"( p[16] ), "=r"( p[17] ), "=r"( p[18] ), "=r"( p[19] ), "=r"( p[20] ), "=r"( p[21] ),
          "=r"( p[22] ), "=r"( p[23] ), "=r"( p[24] ), "=r"( p[25] ), "=r"( p[26] ), "=r"( p[27] ), "=r"( p[28] ),
          "=r"( p[29] ), "=r"( p[30] ), "=r"( p[31] ), "=r"( p[32] )
        : "r"( type ), "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ),
          "f"( tmax ), "f"( rayTime ), "r"( visibilityMask ), "r"( rayFlags ), "r"( SBToffset ), "r"( SBTstride ),
          "r"( missSBTIndex ), "r"( payloadSize ), "r"( p[1] ), "r"( p[2] ), "r"( p[3] ), "r"( p[4] ), "r"( p[5] ),
          "r"( p[6] ), "r"( p[7] ), "r"( p[8] ), "r"( p[9] ), "r"( p[10] ), "r"( p[11] ), "r"( p[12] ), "r"( p[13] ),
          "r"( p[14] ), "r"( p[15] ), "r"( p[16] ), "r"( p[17] ), "r"( p[18] ), "r"( p[19] ), "r"( p[20] ),
          "r"( p[21] ), "r"( p[22] ), "r"( p[23] ), "r"( p[24] ), "r"( p[25] ), "r"( p[26] ), "r"( p[27] ),
          "r"( p[28] ), "r"( p[29] ), "r"( p[30] ), "r"( p[31] ), "r"( p[32] )
        : );
    unsigned int index = 1;
    (void)std::initializer_list<unsigned int>{ index, ( payload = p[index++] )... };
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