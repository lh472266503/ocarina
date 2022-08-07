
__device__ auto tex_sample_float1(cudaTextureObject_t handle, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float>(handle, u, 1 - v);
    return ret;
}
__device__ auto tex_sample_float2(cudaTextureObject_t handle, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float2>(handle, u, 1 - v);
    return oc_make_float2(ret.x, ret.y);
}
__device__ auto tex_sample_float4(cudaTextureObject_t handle, oc_float u, oc_float v) noexcept {
    auto ret = tex2D<float4>(handle, u, 1 - v);
    return oc_make_float4(ret.x, ret.y, ret.z, ret.w);
}

