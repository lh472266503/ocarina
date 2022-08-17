//
// Created by Zero on 2022/8/17.
//

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