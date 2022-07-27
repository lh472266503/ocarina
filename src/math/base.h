//
// Created by Zero on 26/07/2022.
//

#pragma once

#include "core/basic_types.h"
#include "core/concepts.h"
#include "core/constants.h"
#include "dsl/operators.h"


namespace ocarina {

template<typename T>
requires oc_multiply_check(T, T)
[[nodiscard]] constexpr auto sqr(T v) {
    return v * v;
}

template<int n, typename T>
requires oc_multiply_check(T, T)
[[nodiscard]] constexpr T Pow(T v) {
    if constexpr (n < 0) {
        return 1.f / Pow<-n>(v);
    } else if constexpr (n == 1) {
        return v;
    } else if constexpr (n == 0) {
        return 1;
    }
    auto n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}

template<typename T, typename F>
[[nodiscard]] constexpr auto select(bool pred, T t, F f) noexcept {
    return pred ? t : f;
}

template<typename T, typename U, typename V>
[[nodiscard]] constexpr T clamp(T val, U low, V high) noexcept {
    if (val < low) {
        return low;
    } else if (val > high) {
        return high;
    } else {
        return val;
    }
}

template<typename F, typename A, typename B>
requires ocarina::is_floating_point_v<expr_value_t<F>>
[[nodiscard]] constexpr auto lerp(F t, A a, B b) noexcept {
    return a + t * (b - a);
}

template<typename T>
[[nodiscard]] constexpr auto radians(T deg) noexcept {
    return deg * constants::Pi / 180.0f;
}

template<typename T>
[[nodiscard]] constexpr auto degrees(T rad) noexcept {
    return rad * constants::InvPi * 180.0f;
}

template<typename T>
[[nodiscard]] T sign(T val) {
    return select(val >= 0, T(1), T(-1));
}

template<typename T, typename F2>
[[nodiscard]] T triangle_lerp(F2 barycentric, T v0, T v1, T v2) noexcept {
    auto u = barycentric.x;
    auto v = barycentric.y;
    auto w = 1 - barycentric.x - barycentric.y;
    return u * v0 + v * v1 + w * v2;
}

}// namespace ocarina