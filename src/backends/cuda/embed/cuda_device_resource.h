
enum struct OCPixelStorage : oc_uint {
    BYTE1,
    BYTE2,
    BYTE4,
    FLOAT1,
    FLOAT2,
    FLOAT4,
    UNKNOWN
};

__device__ auto oc_tex_sample_float1(cudaTextureObject_t handle, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float>(handle, u, 1 - v);
    return ret;
}
__device__ auto oc_tex_sample_float2(cudaTextureObject_t handle, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float2>(handle, u, 1 - v);
    return oc_make_float2(ret.x, ret.y);
}
__device__ auto oc_tex_sample_float4(cudaTextureObject_t handle, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float4>(handle, u, 1 - v);
    return oc_make_float4(ret.x, ret.y, ret.z, ret.w);
}

struct OCTexture {
    cudaTextureObject_t texture;
    cudaSurfaceObject_t surface;
    OCPixelStorage pixel_storage;
};

__device__ auto oc_tex_sample_float1(OCTexture obj, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float>(obj.texture, u, 1 - v);
    return ret;
}
__device__ auto oc_tex_sample_float2(OCTexture obj, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float2>(obj.texture, u, 1 - v);
    return oc_make_float2(ret.x, ret.y);
}
__device__ auto oc_tex_sample_float4(OCTexture obj, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float4>(obj.texture, u, 1 - v);
    return oc_make_float4(ret.x, ret.y, ret.z, ret.w);
}

__device__ auto oc_tex_read_char1(OCTexture obj, oc_uint x, oc_uint y) noexcept {
    auto ret = surf2Dread<char>(obj.surface, x * sizeof(char), y, cudaBoundaryModeZero);
    return ret;
}


