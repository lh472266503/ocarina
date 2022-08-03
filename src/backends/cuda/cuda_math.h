__device__ inline auto oc_inverse(oc_float3x3 m) noexcept {// from GLM
    oc_float one_over_determinant = 1.0f / (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
                                     m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +
                                     m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
                                     return 0;
    // return oc_float3x3(
    //     (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
    //     (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
    //     (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
    //     (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
    //     (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
    //     (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
    //     (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
    //     (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
    //     (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);
}
 