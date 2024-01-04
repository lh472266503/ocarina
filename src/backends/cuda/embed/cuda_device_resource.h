

template<class To, class From>
[[nodiscard]] __device__ To bit_cast(const From &src) noexcept {
    static_assert(sizeof(To) == sizeof(From));
    union {
        From from;
        To to;
    } u;
    u.from = src;
    return u.to;
}

struct alignas(16) OCHit {
    oc_uint m0{oc_uint(-1)};
    oc_uint m1{oc_uint(-1)};
    oc_float2 m2;
    OCHit() = default;
    OCHit(oc_uint inst_id, oc_uint prim_id, oc_float2 bary)
        : m0(inst_id), m1(prim_id), m2(bary) {}
};

struct alignas(16) OCRay {
public:
    oc_float4 m0;
    oc_float4 m1;
    OCRay() = default;
    OCRay(oc_float4 org, oc_float4 dir)
        : m0(org), m1(dir) {}
};

constexpr float ray_t_max = 1e16f;

__device__ oc_float3 oc_offset_ray_origin(const oc_float3 &p_in, const oc_float3 &n_in) noexcept {
    constexpr auto origin = 1.0f / 32.0f;
    constexpr auto float_scale = 1.0f / 65536.0f;
    constexpr auto int_scale = 256.0f;

    oc_float3 n = n_in;
    auto of_i = oc_make_int3(static_cast<int>(int_scale * n.x),
                             static_cast<int>(int_scale * n.y),
                             static_cast<int>(int_scale * n.z));

    oc_float3 p = p_in;
    oc_float3 p_i = oc_make_float3(
        bit_cast<float>(bit_cast<int>(p.x) + oc_select(p.x < 0, -of_i.x, of_i.x)),
        bit_cast<float>(bit_cast<int>(p.y) + oc_select(p.y < 0, -of_i.y, of_i.y)),
        bit_cast<float>(bit_cast<int>(p.z) + oc_select(p.z < 0, -of_i.z, of_i.z)));

    return oc_select(oc_abs(p) < origin, p + float_scale * n, p_i);
}

__device__ inline OCRay oc_make_ray(oc_float3 org, oc_float3 dir, oc_float t_max = ray_t_max) noexcept {
    OCRay ret;

    ret.m0 = oc_make_float4(org, 0.f);
    ret.m1 = oc_make_float4(dir, t_max);

    return ret;
}

enum struct OCPixelStorage : oc_uint {
    BYTE1,
    BYTE2,
    BYTE4,

    UINT1,
    UINT2,
    UINT4,

    FLOAT1,
    FLOAT2,
    FLOAT4,

    UNKNOWN
};

template<typename T>
inline T oc_atomicExch(T &a, T v) noexcept {
    return atomicExch(&a, v);
}

template<typename T>
inline T oc_atomicAdd(T &a, T v) noexcept {
    return atomicAdd(&a, v);
}

template<typename T>
inline T oc_atomicSub(T &a, T v) noexcept {
    return atomicAdd(&a, -v);
}

struct OCTexture {
    cudaTextureObject_t texture{};
    cudaSurfaceObject_t surface{};
    OCPixelStorage pixel_storage{};
};

[[nodiscard]] inline bool oc_is_null_buffer(void *buffer) noexcept {
    return buffer == nullptr;
}

[[nodiscard]] inline bool oc_is_null_texture(OCTexture obj) noexcept {
    return obj.texture == 0;
}

struct BufferDesc {
    void *head{};
    size_t size_in_byte{};
};

struct OCResourceArray {
    BufferDesc *buffer_slot;
    cudaTextureObject_t *tex_slot{};
};

template<typename A, typename B>
struct oc_is_same {
    static constexpr bool value = false;
};

template<typename A>
struct oc_is_same<A, A> {
    static constexpr bool value = true;
};

template<typename A, typename B>
static constexpr bool oc_is_same_v = oc_is_same<A, B>::value;

using uchar = unsigned char;

__device__ auto oc_tex_sample_float1(OCTexture obj, oc_float u, oc_float v, oc_float w = 0.f) noexcept {
    auto ret = tex3D<float>(obj.texture, u, v, w);
    return ret;
}
__device__ auto oc_tex_sample_float2(OCTexture obj, oc_float u, oc_float v, oc_float w = 0.f) noexcept {
    auto ret = tex3D<float2>(obj.texture, u, v, w);
    return oc_make_float2(ret.x, ret.y);
}
__device__ auto oc_tex_sample_float4(OCTexture obj, oc_float u, oc_float v, oc_float w = 0.f) noexcept {
    auto ret = tex3D<float4>(obj.texture, u, v, w);
    return oc_make_float4(ret.x, ret.y, ret.z, ret.w);
}

template<oc_uint N>
__device__ oc_array<float, N> _oc_tex_sample_float(cudaTextureObject_t texture, oc_float u, oc_float v, oc_float w = 0.f) noexcept {
    if constexpr (N == 1) {
        auto ret = tex3D<float>(texture, u, v, w);
        return {ret};
    } else if constexpr (N == 2) {
        auto ret = tex3D<float2>(texture, u, v, w);
        return {ret.x, ret.y};
    } else if constexpr (N == 3) {
        auto ret = tex3D<float4>(texture, u, v, w);
        return {ret.x, ret.y, ret.z};
    } else if constexpr (N == 4) {
        auto ret = tex3D<float4>(texture, u, v, w);
        return {ret.x, ret.y, ret.z, ret.w};
    }
    return {};
}

template<oc_uint N>
__device__ oc_array<float, N> oc_tex_sample_float(OCTexture obj, oc_float u, oc_float v, oc_float w = 0.f) noexcept {
    return _oc_tex_sample_float<N>(obj.texture, u, v, w);
}

template<typename T>
__device__ T oc_bindless_array_buffer_read(OCResourceArray bindless_array, oc_uint buffer_index, oc_uint index) noexcept {
    const T *buffer = reinterpret_cast<T *>(bindless_array.buffer_slot[buffer_index].head);
    return buffer[index];
}

__device__ oc_uint oc_bindless_array_buffer_size(OCResourceArray bindless_array, oc_uint buffer_index) noexcept {
    return bindless_array.buffer_slot[buffer_index].size_in_byte;
}

template<typename T>
__device__ T oc_bindless_array_byte_buffer_read(OCResourceArray bindless_array, oc_uint buffer_index, oc_uint offset) noexcept {
    const char *buffer = reinterpret_cast<char *>(bindless_array.buffer_slot[buffer_index].head);
    return *reinterpret_cast<const T *>(&buffer[offset]);
}

template<typename T>
__device__ void oc_bindless_array_buffer_write(OCResourceArray bindless_array, oc_uint buffer_index,
                                               oc_uint index, const T &val) noexcept {
    T *buffer = reinterpret_cast<T *>(bindless_array.buffer_slot[buffer_index].head);
    buffer[index] = val;
}

template<typename T>
__device__ T oc_bindless_array_tex_sample(OCResourceArray bindless_array, oc_uint tex_index,
                                          oc_float u, oc_float v, oc_float w = 0.f) noexcept {
    cudaTextureObject_t texture = bindless_array.tex_slot[tex_index];
    if constexpr (oc_is_same_v<T, oc_float>) {
        float ret = tex3D<float>(texture, u, v, w);
        return ret;
    } else if constexpr (oc_is_same_v<T, oc_float2>) {
        float2 ret = tex3D<float2>(texture, u, v, w);
        return oc_make_float2(ret.x, ret.y);
    } else if constexpr (oc_is_same_v<T, oc_float4>) {
        float4 ret = tex3D<float4>(texture, u, v, w);
        return oc_make_float4(ret.x, ret.y, ret.z, ret.w);
    }
    return T{};
}

template<oc_uint N>
__device__ oc_array<float, N> oc_bindless_array_tex_sample(OCResourceArray bindless_array, oc_uint tex_index,
                                                           oc_float u, oc_float v, oc_float w = 0.f) noexcept {
    cudaTextureObject_t texture = bindless_array.tex_slot[tex_index];
    return _oc_tex_sample_float<N>(texture, u, v, w);
}

template<typename T>
struct oc_type {
    static constexpr auto dimension = 1;
    using element_type = T;
};

template<typename T, int N>
struct oc_vec {
    using type = T;
};

#define OC_MAKE_VEC_TYPE(T, N) \
    template<>                 \
    struct oc_vec<T, N> {      \
        using type = T##N;     \
    };

#define OC_MAKE_VEC_TYPE_N(T) \
    OC_MAKE_VEC_TYPE(T, 2)    \
    OC_MAKE_VEC_TYPE(T, 3)    \
    OC_MAKE_VEC_TYPE(T, 4)

OC_MAKE_VEC_TYPE_N(oc_float)
OC_MAKE_VEC_TYPE_N(oc_int)
OC_MAKE_VEC_TYPE_N(oc_uchar)
OC_MAKE_VEC_TYPE_N(oc_uint)

template<typename T, int N>
using oc_vec_t = typename oc_vec<T, N>::type;

#undef OC_MAKE_VEC_TYPE
#undef OC_MAKE_VEC_TYPE_N

#define OC_MAKE_TYPE_DIM(type, dim)            \
    template<>                                 \
    struct oc_type<type##dim> {                \
        static constexpr auto dimension = dim; \
        using element_type = type;             \
    };

#define OC_MAKE_TYPE(type)    \
    OC_MAKE_TYPE_DIM(type, 2) \
    OC_MAKE_TYPE_DIM(type, 3) \
    OC_MAKE_TYPE_DIM(type, 4)

OC_MAKE_TYPE(oc_uint)
OC_MAKE_TYPE(oc_int)
OC_MAKE_TYPE(oc_float)
OC_MAKE_TYPE(oc_uchar)
OC_MAKE_TYPE(oc_bool)

template<typename T>
static constexpr auto oc_type_dim = oc_type<T>::dimension;

template<typename T>
using oc_type_element_t = typename oc_type<T>::element_type;

#undef OC_MAKE_TYPE
#undef OC_MAKE_TYPE_DIM

template<typename Dst, typename Src>
__device__ auto oc_convert_scalar(const Src &src) noexcept {
    static_assert(oc_type_dim<Src> == 1 && oc_type_dim<Dst> == 1);
    if constexpr (oc_is_same_v<Dst, Src>) {
        return src;
    } else if constexpr (oc_is_same_v<Dst, float>) {
        // Src is uchar
        return static_cast<float>(src / 255.f);
    } else if constexpr (oc_is_same_v<Dst, uchar>) {
        // Src is float
        return static_cast<uchar>(src * 255);
    }
    __builtin_unreachable();
}

template<typename Dst, typename Src>
__device__ auto oc_convert_vector(const Src &val) noexcept {
    static_assert(oc_type_dim<Dst> == oc_type_dim<Src>);
    using element_type = oc_type_element_t<Dst>;
    if constexpr (oc_type_dim<Dst> == 1) {
        return oc_convert_scalar<Dst>(val);
    } else if constexpr (oc_type_dim<Dst> == 2) {
        return Dst(oc_convert_scalar<element_type>(val.x),
                   oc_convert_scalar<element_type>(val.y));
    } else if constexpr (oc_type_dim<Dst> == 4) {
        return Dst(oc_convert_scalar<element_type>(val.x),
                   oc_convert_scalar<element_type>(val.y),
                   oc_convert_scalar<element_type>(val.z),
                   oc_convert_scalar<element_type>(val.w));
    }
}

template<int Dim, typename Src>
__device__ auto oc_fit(const Src &src) noexcept {
    using element_type = oc_type_element_t<Src>;
    static constexpr auto dim = oc_type_dim<Src>;
    using ret_type = typename oc_vec_t<element_type, Dim>;
    if constexpr (dim == 2) {
    }
}

template<typename T>
__device__ T oc_texture_read(OCTexture obj, oc_uint x, oc_uint y, oc_uint z = 0) noexcept {
    if constexpr (oc_is_same_v<T, uchar> || oc_is_same_v<T, float>) {
        switch (obj.pixel_storage) {
            case OCPixelStorage::BYTE1: {
                auto v = surf3Dread<uchar>(obj.surface, x * sizeof(oc_uchar), y, z, cudaBoundaryModeZero);
                return oc_convert_scalar<T>(v);
            }
            case OCPixelStorage::FLOAT1: {
                auto v = surf3Dread<float>(obj.surface, x * sizeof(float), y, z, cudaBoundaryModeZero);
                return oc_convert_scalar<T>(v);
            }
        }
    } else if constexpr (oc_is_same_v<T, oc_uchar2> || oc_is_same_v<T, oc_float2>) {
        switch (obj.pixel_storage) {
            case OCPixelStorage::BYTE2: {
                auto v = surf3Dread<uchar2>(obj.surface, x * sizeof(uchar2), y, z, cudaBoundaryModeZero);
                return oc_convert_vector<T>(oc_make_uchar2(v.x, v.y));
            }
            case OCPixelStorage::FLOAT2: {
                auto v = surf3Dread<float2>(obj.surface, x * sizeof(float2), y, z, cudaBoundaryModeZero);
                return oc_convert_vector<T>(oc_make_float2(v.x, v.y));
            }
        }
    } else if constexpr (oc_is_same_v<T, oc_uchar4> || oc_is_same_v<T, oc_float4>) {
        switch (obj.pixel_storage) {
            case OCPixelStorage::BYTE4: {
                auto v = surf3Dread<uchar4>(obj.surface, x * sizeof(uchar4), y, z, cudaBoundaryModeZero);
                return oc_convert_vector<T>(oc_make_uchar4(v.x, v.y, v.z, v.w));
            }
            case OCPixelStorage::FLOAT4: {
                auto v = surf3Dread<float4>(obj.surface, x * sizeof(float4), y, z, cudaBoundaryModeZero);
                return oc_convert_vector<T>(oc_make_float4(v.x, v.y, v.z, v.w));
            }
        }
    } else if constexpr (oc_is_same_v<T, oc_uint>) {
        auto v = surf3Dread<unsigned int>(obj.surface, x * sizeof(unsigned int), y, z, cudaBoundaryModeZero);
        return v;
    } else if constexpr (oc_is_same_v<T, oc_uint2>) {
        auto v = surf3Dread<uint2>(obj.surface, x * sizeof(uint2), y, z, cudaBoundaryModeZero);
        return oc_make_uint2(v.x, v.y);
    } else if constexpr (oc_is_same_v<T, oc_uint4>) {
        auto v = surf3Dread<uint4>(obj.surface, x * sizeof(uint4), y, z, cudaBoundaryModeZero);
        return oc_make_uint4(v.x, v.y, v.z, v.w);
    }
    assert(0);
    __builtin_unreachable();
}

template<typename T>
__device__ void oc_texture_write(OCTexture obj, T val, oc_uint x, oc_uint y, oc_uint z = 0) noexcept {
    if constexpr (oc_is_same_v<T, uchar> || oc_is_same_v<T, float>) {
        switch (obj.pixel_storage) {
            case OCPixelStorage::BYTE1: {
                uchar v = oc_convert_scalar<uchar>(val);
                surf3Dwrite(v, obj.surface, x * sizeof(uchar), y, z, cudaBoundaryModeZero);
                return;
            }
            case OCPixelStorage::FLOAT1: {
                oc_float v = oc_convert_vector<float>(val);
                surf3Dwrite(v, obj.surface, x * sizeof(float), y, z, cudaBoundaryModeZero);
                return;
            }
        }
    } else if constexpr (oc_is_same_v<T, oc_uchar2> || oc_is_same_v<T, oc_float2>) {
        switch (obj.pixel_storage) {
            case OCPixelStorage::BYTE2: {
                oc_uchar2 v = oc_convert_vector<oc_uchar2>(val);
                surf3Dwrite(make_uchar2(v.x, v.y), obj.surface, x * sizeof(uchar2), y, z, cudaBoundaryModeZero);
                return;
            }
            case OCPixelStorage::FLOAT2: {
                oc_float2 v = oc_convert_vector<oc_float2>(val);
                surf3Dwrite(make_float2(v.x, v.y), obj.surface, x * sizeof(float2), y, z, cudaBoundaryModeZero);
                return;
            }
        }
    } else if constexpr (oc_is_same_v<T, oc_uchar4> || oc_is_same_v<T, oc_float4>) {
        switch (obj.pixel_storage) {
            case OCPixelStorage::BYTE4: {
                oc_uchar4 v = oc_convert_vector<oc_uchar4>(val);
                surf3Dwrite(make_uchar4(v.x, v.y, v.z, v.w), obj.surface, x * sizeof(uchar4), y, z, cudaBoundaryModeZero);
                return;
            }
            case OCPixelStorage::FLOAT4: {
                oc_float4 v = oc_convert_vector<oc_float4>(val);
                surf3Dwrite(make_float4(v.x, v.y, v.z, v.w), obj.surface, x * sizeof(float4), y, z, cudaBoundaryModeZero);
                return;
            }
        }
    } else if constexpr (oc_is_same_v<T, oc_uint>) {
        surf3Dwrite(val, obj.surface, x * sizeof(oc_uint), y, z, cudaBoundaryModeZero);
        return;
    } else if constexpr (oc_is_same_v<T, oc_uint2>) {
        surf3Dwrite(make_uint2(val.x, val.y), obj.surface, x * sizeof(uint2), y, z, cudaBoundaryModeZero);
        return;
    } else if constexpr (oc_is_same_v<T, oc_uint4>) {
        surf3Dwrite(make_uint4(val.x, val.y, val.z, val.w), obj.surface, x * sizeof(uint4), y, z, cudaBoundaryModeZero);
        return;
    }
    assert(0);
    __builtin_unreachable();
}
