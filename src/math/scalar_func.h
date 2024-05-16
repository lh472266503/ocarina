//
// Created by Zero on 2024/5/16.
//

#pragma once

#include "basic_types.h"
#include "dsl/type_trait.h"
#include "core/concepts.h"
#include <numeric>

namespace ocarina {

template<typename T, typename F>
[[nodiscard]] constexpr auto select(bool pred, T &&t, F &&f) noexcept {
    return pred ? t : f;
}

template<typename T>
requires std::is_unsigned_v<T> && (sizeof(T) == 4u || sizeof(T) == 8u)
[[nodiscard]] constexpr auto next_pow2(T v) noexcept {
    v--;
    v |= v >> 1u;
    v |= v >> 2u;
    v |= v >> 4u;
    v |= v >> 8u;
    v |= v >> 16u;
    if constexpr (sizeof(T) == 8u) { v |= v >> 32u; }
    return v + 1u;
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr T sign(T val) {
    return select(val >= 0, T(1), T(-1));
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr auto
radians(const T &deg) noexcept {
    return deg * (constants::Pi / 180.0f);
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr auto
degrees(T rad) noexcept {
    return rad * (constants::InvPi * 180.0f);
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr auto
rcp(const T &v) {
    return 1.f / v;
}

template<typename T, typename U>
requires ocarina::is_all_scalar_expr_v<T, U>
auto divide(T &&t, U &&u) noexcept {
    return OC_FORWARD(t) * rcp(OC_FORWARD(u));
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] constexpr auto
rsqrt(const T &v) {
    return 1.f / sqrt(v);
}

template<typename T>
requires is_floating_point_v<T>
[[nodiscard]] constexpr auto
fract(const T &v) {
    return v - floorf(v);
}

[[nodiscard]] inline float mod(float x, float y) {
    return x - y * floor(x / y);
}

template<typename T, typename U>
requires is_all_floating_point_expr_v<T, U>
[[nodiscard]] condition_t<bool, T, U> is_close(T t, U u, float epsilon = 0.00001f) {
    return abs(t - u) < epsilon;
}

template<typename T>
requires is_scalar_v<T>
[[nodiscard]] auto saturate(const T &f) { return min(1.f, max(0.f, f)); }

template<typename T>
requires is_scalar_v<T>
OC_NODISCARD constexpr auto sqr(const T &v) {
    return v * v;
}

inline void oc_memcpy(void *dst, const void *src, size_t size) {
#ifdef _MSC_VER
    std::memcpy(dst, src, size);
#else
    std::wmemcpy(reinterpret_cast<wchar_t *>(dst), reinterpret_cast<const wchar_t *>(src), size);
#endif
}

[[nodiscard]] inline bool isnan(float x) noexcept {
    auto u = 0u;
    ::memcpy(&u, &x, sizeof(float));
    return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) != 0u;
}

[[nodiscard]] inline bool isinf(float x) noexcept {
    auto u = 0u;
    ::memcpy(&u, &x, sizeof(float));
    return (u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu) == 0u;
}

// math
using std::abs;
using std::acos;
using std::acosh;
using std::asin;
using std::asinh;
using std::atan;
using std::atan2;
using std::atanh;
using std::ceil;
using std::copysign;
using std::cos;
using std::cosh;
using std::exp;
using std::exp2;
using std::floor;
using std::fma;
using std::fmod;
using std::log;
using std::log10;
using std::log2;
using std::max;
using std::min;
using std::pow;
using std::round;
using std::roundf;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

}// namespace ocarina
