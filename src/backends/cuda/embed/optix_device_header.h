//
// Created by Zero on 2022/8/17.
//

constexpr float ray_t_max = 1e16f;

struct alignas(16) OCHit {
    oc_uint inst_id{oc_uint(-1)};
    oc_uint prim_id{oc_uint(-1)};
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
    float t_min{0.f};

public:
    explicit __device__ Ray(float t_max = ray_t_max) noexcept
        : t_max(t_max) {}

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

template<typename... Args>
__device__ inline void trace(OptixTraversableHandle handle,
                             OCRay ray,
                             uint32_t flags,
                             uint32_t SBToffset,
                             uint32_t SBTstride,
                             uint32_t missSBTIndex,
                             Args &&...payload) {
    auto origin = ::make_float3(ray.org_x, ray.org_y, ray.org_z);
    auto direction = ::make_float3(ray.dir_x, ray.dir_y, ray.dir_z);

    optixTrace(
        handle,
        origin,
        direction,
        0,
        ray.t_max,
        0.0f,// rayTime
        OptixVisibilityMask(1),
        flags,
        SBToffset,   // SBT offset
        SBTstride,   // SBT stride
        missSBTIndex,// missSBTIndex
        std::forward<Args>(payload)...);
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

__device__ inline OCHit traceClosestHit(OptixTraversableHandle handle,
                                          OCRay ray) {
    unsigned int u0, u1;
    OCHit hit;
    pack_pointer(hit, u0, u1);
    trace(handle, ray, OPTIX_RAY_FLAG_DISABLE_ANYHIT,
          0,        // SBT offset
          0,           // SBT stride
          0,        // missSBTIndex
          u0, u1);
    return hit;
}

__device__ inline bool traceAnyHit(OptixTraversableHandle handle, OCRay ray) {
    unsigned int occluded = 0u;
    trace(handle, ray, OPTIX_RAY_FLAG_DISABLE_ANYHIT
                       | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
          1,        // SBT offset
          0,           // SBT stride
          0,        // missSBTIndex
          occluded);
    return bool(occluded);
}

__device__ inline OCHit getClosestHit() {
    OCHit ret;
    ret.instance_id = optixGetInstanceId();
    ret.prim_id = optixGetPrimitiveIndex();
    ret.bary = getTriangleBarycentric();
    return ret;
}

template<typename T = OCHit>
__device__ inline T *getPRD() {
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpack_pointer(u0, u1));
}

extern "C" __global__ void __closesthit__closest() {
    OCHit *hit = getPRD<OCHit>();
    *hit = getClosestHit();
}

extern "C" __global__ void __closesthit__any() {
    setPayloadOcclusion(true);
}
