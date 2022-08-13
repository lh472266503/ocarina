
enum struct OCPixelStorage : oc_uint {
    BYTE1,
    BYTE2,
    BYTE4,
    FLOAT1,
    FLOAT2,
    FLOAT4,
    UNKNOWN
};

struct ImageData {
    cudaTextureObject_t texture;
    cudaSurfaceObject_t surface;
    OCPixelStorage pixel_storage;
};

using uchar = unsigned char;

__device__ auto oc_tex_sample_float1(ImageData obj, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float>(obj.texture, u, 1 - v);
    return ret;
}
__device__ auto oc_tex_sample_float2(ImageData obj, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float2>(obj.texture, u, 1 - v);
    return oc_make_float2(ret.x, ret.y);
}
__device__ auto oc_tex_sample_float4(ImageData obj, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float4>(obj.texture, u, 1 - v);
    return oc_make_float4(ret.x, ret.y, ret.z, ret.w);
}

__device__ auto oc_tex_read_uchar1(ImageData obj, oc_uint x, oc_uint y) noexcept {
    auto ret = surf2Dread<uchar>(obj.surface, x * sizeof(uchar), y, cudaBoundaryModeZero);
    return ret;
}

__device__ auto oc_tex_read_uchar2(ImageData obj, oc_uint x, oc_uint y) noexcept {
    auto ret = surf2Dread<uchar2>(obj.surface, x * sizeof(uchar2), y, cudaBoundaryModeZero);
    return oc_make_uchar2(ret.x, ret.y);
}

__device__ auto oc_tex_read_uchar4(ImageData obj, oc_uint x, oc_uint y) noexcept {
    auto ret = surf2Dread<uchar4>(obj.surface, x * sizeof(uchar4), y, cudaBoundaryModeZero);
    return oc_make_uchar4(ret.x, ret.y, ret.x, ret.y);
}

__device__ auto oc_tex_read_float1(ImageData obj, oc_uint x, oc_uint y) noexcept {
    auto ret = surf2Dread<float>(obj.surface, x * sizeof(float), y, cudaBoundaryModeZero);
    return ret;
}

__device__ auto oc_tex_read_float2(ImageData obj, oc_uint x, oc_uint y) noexcept {
    auto ret = surf2Dread<float2>(obj.surface, x * sizeof(float2), y, cudaBoundaryModeZero);
    return oc_make_float2(ret.x, ret.y);
}

__device__ auto oc_tex_read_float4(ImageData obj, oc_uint x, oc_uint y) noexcept {
    auto ret = surf2Dread<float4>(obj.surface, x * sizeof(float4), y, cudaBoundaryModeZero);
    return oc_make_float4(ret.x, ret.y, ret.x, ret.y);
}

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

template<typename T>
struct oc_type {
    static constexpr auto dimension = 1;
    using element_type = T;
};

template<typename T, size_t N>
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

template<size_t Dim, typename Src>
__device__ auto oc_fit(const Src &src) noexcept {
    using element_type = oc_type_element_t<Src>;
    static constexpr auto dim = oc_type_dim<Src>;
    return;
}

template<typename T>
__device__ auto oc_image_read(ImageData obj, oc_uint x, oc_uint y) noexcept {
    switch (obj.pixel_storage) {
        case OCPixelStorage::BYTE1:
            auto v = surf2Dread<uchar>(obj.surface, x * sizeof(oc_uchar), y, cudaBoundaryModeZero);

            break;
        case OCPixelStorage::BYTE2:
            auto v = surf2Dread<uchar2>(obj.surface, x * sizeof(uchar2), y, cudaBoundaryModeZero);

            break;
        case OCPixelStorage::BYTE4:
            auto v = surf2Dread<uchar4>(obj.surface, x * sizeof(uchar4), y, cudaBoundaryModeZero);
            break;
        case OCPixelStorage::FLOAT1:
            auto v = surf2Dread<float>(obj.surface, x * sizeof(float), y, cudaBoundaryModeZero);
            break;
        case OCPixelStorage::FLOAT2:
            auto v = surf2Dread<float2>(obj.surface, x * sizeof(float2), y, cudaBoundaryModeZero);
            break;
        case OCPixelStorage::FLOAT4:
            auto v = surf2Dread<float2>(obj.surface, x * sizeof(float2), y, cudaBoundaryModeZero);
            break;
        case OCPixelStorage::UNKNOWN:
            break;
    }
    return T{};
}
