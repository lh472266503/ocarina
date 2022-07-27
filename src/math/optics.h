//
// Created by Zero on 26/07/2022.
//

#pragma once

#include <cstdlib>
#include "core/basic_types.h"
#include "base.h"

namespace ocarina {

enum ColorSpace {
    LINEAR,
    SRGB
};

enum EToneMap {
    Gamma,
    Filmic,
    Reinhard,
    Linear
};

[[nodiscard]] inline uint32_t make_8bit(const float f) {
    return fmin(255, fmax(0, int(f * 256.f)));
}

[[nodiscard]] inline uint32_t make_rgba(const float3 color) {
    return (make_8bit(color.x) << 0) +
           (make_8bit(color.y) << 8) +
           (make_8bit(color.z) << 16) +
           (0xffU << 24);
}

[[nodiscard]] inline uint32_t make_rgba(const float4 color) {
    return (make_8bit(color.x) << 0) +
           (make_8bit(color.y) << 8) +
           (make_8bit(color.z) << 16) +
           (make_8bit(color.w) << 24);
}

template<typename T>
requires ocarina::is_vector3_v<expr_value_t<T>>
[[nodiscard]] T reflect(const T &wo, const T &n) {
    return -wo + 2 * dot(wo, n) * n;
}

template<typename T>
[[nodiscard]] T schlick_weight(const T &cos_theta) {
    T m = clamp(1.f - cos_theta, 0.f, 1.f);
    return Pow<5>(m);
}

template<typename T, typename U>
[[nodiscard]] auto fresnel_schlick(const T &R0, const U &cos_theta) {
    return lerp(schlick_weight(cos_theta), R0, T{1.f});
}

template<typename T>
[[nodiscard]] T srgb_to_linear(T S) {
    return select((S < T(0.04045f)),
                  (S / T(12.92f)),
                  (pow((S + 0.055f) * 1.f / 1.055f, 2.4f)));
}

template<typename T>
[[nodiscard]] T linear_to_srgb(T L) {
    return select((L < T(0.0031308f)),
                  (L * T(12.92f)),
                  (T(1.055f) * pow(L, T(1.0f / 2.4f)) - T(0.055f)));
}

}// namespace ocarina