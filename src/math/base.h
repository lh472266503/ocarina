//
// Created by Zero on 26/07/2022.
//

#pragma once

#include "core/basic_types.h"
#include "core/concepts.h"
#include "core/constants.h"
#include "dsl/operators.h"

namespace ocarina {

using std::abs;
using std::sqrt;

template<typename T, typename F>
[[nodiscard]] constexpr auto select(bool pred, T t, F f) noexcept {
    return pred ? t : f;
}

template<typename T>
[[nodiscard]] T sign(T val) {
    return select(val >= 0, T(1), T(-1));
}

template<typename T>
[[nodiscard]] constexpr auto radians(const T &deg) noexcept {
    return deg * (constants::Pi / 180.0f);
}

template<typename T>
[[nodiscard]] constexpr auto degrees(T rad) noexcept {
    return rad * constants::InvPi * 180.0f;
}

template<typename T>
[[nodiscard]] auto rcp(const T &v) {
    return 1.f / v;
}

template<typename T>
requires oc_multiply_check(T, T)
[[nodiscard]] constexpr auto sqr(T v) {
    return v * v;
}

#define MAKE_VECTOR_UNARY_FUNC(func)                                         \
    template<typename T, uint N>                                             \
    [[nodiscard]] constexpr auto func(Vector<T, N> v) noexcept {             \
        static_assert(N == 2 || N == 3 || N == 4);                           \
        if constexpr (N == 2) {                                              \
            return Vector<T, 2>{func(v.x), func(v.y)};                       \
        } else if constexpr (N == 3) {                                       \
            return Vector<T, 3>(func(v.x), func(v.y), func(v.z));            \
        } else {                                                             \
            return Vector<T, 4>(func(v.x), func(v.y), func(v.z), func(v.w)); \
        }                                                                    \
    }

MAKE_VECTOR_UNARY_FUNC(rcp)
MAKE_VECTOR_UNARY_FUNC(abs)
MAKE_VECTOR_UNARY_FUNC(sqrt)
MAKE_VECTOR_UNARY_FUNC(sqr)
MAKE_VECTOR_UNARY_FUNC(sign)
MAKE_VECTOR_UNARY_FUNC(degrees)
//MAKE_VECTOR_UNARY_FUNC(radians)

#undef MAKE_VECTOR_UNARY_FUNC

#define MAKE_VECTOR_BINARY_FUNC(func)                                                            \
    template<typename T, uint N>                                                                 \
    [[nodiscard]] constexpr auto func(Vector<T, N> v, Vector<T, N> u) noexcept {                 \
        static_assert(N == 2 || N == 3 || N == 4);                                               \
        if constexpr (N == 2) {                                                                  \
            return Vector<T, 2>{func(v.x, u.x), func(v.y, u.y)};                                 \
        } else if constexpr (N == 3) {                                                           \
            return Vector<T, 3>(func(v.x, u.x), func(v.y, u.y), func(v.z, u.z));                 \
        } else {                                                                                 \
            return Vector<T, 4>(func(v.x, u.x), func(v.y, u.y), func(v.z, u.z), func(v.w, u.w)); \
        }                                                                                        \
    }                                                                                            \
    template<typename T, uint N>                                                                 \
    [[nodiscard]] constexpr auto func(T v, Vector<T, N> u) noexcept {                            \
        static_assert(N == 2 || N == 3 || N == 4);                                               \
        if constexpr (N == 2) {                                                                  \
            return Vector<T, 2>{func(v, u.x), func(v, u.y)};                                     \
        } else if constexpr (N == 3) {                                                           \
            return Vector<T, 3>(func(v, u.x), func(v, u.y), func(v, u.z));                       \
        } else {                                                                                 \
            return Vector<T, 4>(func(v, u.x), func(v, u.y), func(v, u.z), func(v, u.w));         \
        }                                                                                        \
    }                                                                                            \
    template<typename T, uint N>                                                                 \
    [[nodiscard]] constexpr auto func(Vector<T, N> v, T u) noexcept {                            \
        static_assert(N == 2 || N == 3 || N == 4);                                               \
        if constexpr (N == 2) {                                                                  \
            return Vector<T, 2>{func(v.x, u), func(v.y, u)};                                     \
        } else if constexpr (N == 3) {                                                           \
            return Vector<T, 3>(func(v.x, u), func(v.y, u), func(v.z, u));                       \
        } else {                                                                                 \
            return Vector<T, 4>(func(v.x, u), func(v.y, u), func(v.z, u), func(v.w, u));         \
        }                                                                                        \
    }

using std::atan2;
using std::max;
using std::min;
using std::pow;
MAKE_VECTOR_BINARY_FUNC(pow)
MAKE_VECTOR_BINARY_FUNC(min)
MAKE_VECTOR_BINARY_FUNC(max)
MAKE_VECTOR_BINARY_FUNC(atan2)

#undef MAKE_VECTOR_BINARY_FUNC

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


template<typename T, typename F2>
[[nodiscard]] T triangle_lerp(F2 barycentric, T v0, T v1, T v2) noexcept {
    auto u = barycentric.x;
    auto v = barycentric.y;
    auto w = 1 - barycentric.x - barycentric.y;
    return u * v0 + v * v1 + w * v2;
}

}// namespace ocarina