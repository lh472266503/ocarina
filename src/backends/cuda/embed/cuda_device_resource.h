
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

// __device__ void oc_tex_write_uchar1(ImageData obj, oc_uint x, oc_uint y, oc_uchar var) noexcept {
//     surf2Dwrite(v, obj.surface, x * sizeof(uchar), y, cudaBoundaryModeZero);
// }