//
// Created by Zero on 2022/8/17.
//

constexpr float ray_t_max = 1e16f;

struct alignas(16) OCHit {
    oc_uint inst_id{};
    oc_uint prim_id{};
    oc_float2 bary;
};

struct alignas(16) OCRay {
public:
    float org_x{0.f};
    float org_y{0.f};
    float org_z{0.f};
    float dir_x{0.f};
    float dir_y{0.f};
    float dir_z{0.f};
    float t_max{0.f};

public:

    __device__ OCRay(const oc_float3 origin, const oc_float3 direction,
               float t_max = ray_t_max) noexcept : t_max(t_max) {
        update_origin(origin);
        update_direction(direction);
    }

    __device__ oc_float3 at(float t) const {
        return origin() + direction() * t;
    }

    __device__ void update_origin(oc_float3 origin) noexcept {
        org_x = origin.x;
        org_y = origin.y;
        org_z = origin.z;
    }

    __device__ void update_direction(oc_float3 direction) noexcept {
        dir_x = direction.x;
        dir_y = direction.y;
        dir_z = direction.z;
    }

    __device__ oc_float3 origin() const noexcept {
        return oc_make_float3(org_x, org_y, org_z);
    }

    __device__ oc_float3 direction() const noexcept {
        return oc_make_float3(dir_x, dir_y, dir_z);
    }
};

/// optix builtin start

typedef enum OptixPayloadTypeID {
    OPTIX_PAYLOAD_TYPE_DEFAULT = 0,
    OPTIX_PAYLOAD_TYPE_ID_0 = (1 << 0u),
    OPTIX_PAYLOAD_TYPE_ID_1 = (1 << 1u),
    OPTIX_PAYLOAD_TYPE_ID_2 = (1 << 2u),
    OPTIX_PAYLOAD_TYPE_ID_3 = (1 << 3u),
    OPTIX_PAYLOAD_TYPE_ID_4 = (1 << 4u),
    OPTIX_PAYLOAD_TYPE_ID_5 = (1 << 5u),
    OPTIX_PAYLOAD_TYPE_ID_6 = (1 << 6u),
    OPTIX_PAYLOAD_TYPE_ID_7 = (1 << 7u)
} OptixPayloadTypeID;

/// Traversable handle
typedef unsigned long long OptixTraversableHandle;

/// Visibility mask
typedef unsigned int OptixVisibilityMask;

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

static __forceinline__ __device__ unsigned int oc_oc_optixGetPayload_0() {
    unsigned int resuoc_lt;
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
static __forceinline__ __device__ void oc_optixTrace( OptixTraversableHandle handle,
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
    static_assert( sizeof...( Payload ) <= 32, "Only up to 32 payload values are allowed." );
    // std::is_same compares each type in the two TypePacks to make sure that all types are unsigned int.
    // TypePack 1    unsigned int    T0      T1      T2   ...   Tn-1        Tn
    // TypePack 2      T0            T1      T2      T3   ...   Tn        unsigned int

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
        : "r"( 0 ), "l"( handle ), "f"( ox ), "f"( oy ), "f"( oz ), "f"( dx ), "f"( dy ), "f"( dz ), "f"( tmin ),
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

template<typename... Args>
static __device__ void trace(OptixTraversableHandle handle,
                                OCRay ray,
                                uint32_t flags,
                                uint32_t SBToffset,
                                uint32_t SBTstride,
                                uint32_t missSBTIndex,
                                Args &&... payload) {
    auto origin = oc_make_float3(ray.org_x, ray.org_y, ray.org_z);
    auto direction = oc_make_float3(ray.dir_x, ray.dir_y, ray.dir_z);

    oc_optixTrace(
            handle,
            origin,
            direction,
            0,
            ray.t_max,
            0.0f,                // rayTime
            OptixVisibilityMask(1),
            flags,
            SBToffset,        // SBT offset
            SBTstride,           // SBT stride
            missSBTIndex,        // missSBTIndex
            std::forward<Args>(payload)...);
}


__device__ inline void *unpack_pointer(unsigned int i0, unsigned int i1) {
    const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
    void *ptr = reinterpret_cast<void *>( uptr );
    return ptr;
}

__device__ inline void pack_pointer(void *ptr, unsigned int &i0, unsigned int &i1) {
    const auto uptr = reinterpret_cast<unsigned long long>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

__device__ inline void setPayloadOcclusion(bool occluded) {
    oc_optixSetPayload_0(static_cast<unsigned int>( occluded ));
}

static LM_GPU_INLINE oc_float2 getTriangleBarycentric() {
    auto barycentric = oc_optixGetTriangleBarycentrics();
    return oc_make_float2(1 - barycentric.y - barycentric.x, barycentric.x);
}


__device__ inline OCHit getClosestHit() {
    OCHit ret;
    ret.instance_id = oc_optixGetInstanceId();
    ret.prim_id = oc_optixGetPrimitiveIndex();
    ret.bary = getTriangleBarycentric();
    return ret;
}

template<typename T = OCHit>
__device__ inline T *getPRD() {
    const unsigned int u0 = oc_optixGetPayload_0();
    const unsigned int u1 = oc_optixGetPayload_1();
    return reinterpret_cast<T *>(unpack_pointer(u0, u1));
}


extern "C" __global__ void __closesthit__closest() {
    OCHit *hit = getPRD<OCHit>();
}

extern "C" __global__ void __closesthit__any() {
    setPayloadOcclusion(true);
}

