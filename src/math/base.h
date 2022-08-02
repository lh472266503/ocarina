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
using std::atan2;
using std::max;
using std::min;
using std::pow;
using std::sqrt;

template<typename T>
[[nodiscard]] constexpr auto select(bool pred, T &&t, T &&f) noexcept {
    return pred ? t : f;
}

template<typename T, uint N>
requires ocarina::is_scalar_v<T>
[[nodiscard]] constexpr auto select(Vector<bool, N> pred, Vector<T, N> t, Vector<T, N> f) noexcept {
    static_assert(N == 2 || N == 3 || N == 4);
    if constexpr (N == 2) {
        return Vector<T, N>{select(pred.x, t.x, f.x), select(pred.y, t.y, f.y)};
    } else if constexpr (N == 3) {
        return Vector<T, N>{select(pred.x, t.x, f.x), select(pred.y, t.y, f.y), select(pred.z, t.z, f.z)};
    } else {
        return Vector<T, N>{select(pred.x, t.x, f.x), select(pred.y, t.y, f.y), select(pred.z, t.z, f.z),
                            select(pred.w, t.w, f.w)};
    }
}

template<typename T>
requires is_scalar_v<expr_value_t<T>>
    OC_NODISCARD constexpr T sign(T val) {
    return select(val >= 0, T(1), T(-1));
}

template<typename T>
requires is_scalar_v<expr_value_t<T>>
    OC_NODISCARD constexpr auto
    radians(const T &deg) noexcept {
    return deg * (constants::Pi / 180.0f);
}

template<typename T>
requires is_scalar_v<expr_value_t<T>>
    OC_NODISCARD constexpr auto
    degrees(T rad) noexcept {
    return rad * (constants::InvPi * 180.0f);
}

template<typename T>
requires is_scalar_v<expr_value_t<T>>
    OC_NODISCARD constexpr auto
    rcp(const T &v) {
    return 1.f / v;
}

template<typename T>
[[nodiscard]] auto saturate(const T &f) { return min(1.f, max(0.f, f)); }

template<typename T>
requires OC_MULTIPLY_CHECK(T, T)
OC_NODISCARD constexpr auto sqr(T v) {
    return v * v;
}

#define MAKE_VECTOR_UNARY_FUNC(func)                                   \
    template<typename T>                                               \
    requires is_vector_v<expr_value_t<T>>                              \
        OC_NODISCARD auto                                                  \
    func(const T &v) noexcept {                                        \
        static constexpr auto N = vector_dimension_v<expr_value_t<T>>; \
        static_assert(N == 2 || N == 3 || N == 4);                     \
        if constexpr (N == 2) {                                        \
            return T{func(v.x), func(v.y)};                            \
        } else if constexpr (N == 3) {                                 \
            return T(func(v.x), func(v.y), func(v.z));                 \
        } else {                                                       \
            return T(func(v.x), func(v.y), func(v.z), func(v.w));      \
        }                                                              \
    }

MAKE_VECTOR_UNARY_FUNC(rcp)
MAKE_VECTOR_UNARY_FUNC(abs)
MAKE_VECTOR_UNARY_FUNC(sqrt)
MAKE_VECTOR_UNARY_FUNC(sqr)
MAKE_VECTOR_UNARY_FUNC(sign)
MAKE_VECTOR_UNARY_FUNC(degrees)
MAKE_VECTOR_UNARY_FUNC(radians)

#undef MAKE_VECTOR_UNARY_FUNC

#define MAKE_VECTOR_BINARY_FUNC(func)                                                 \
    template<typename T>                                                              \
    requires is_vector_v<expr_value_t<T>>                                             \
        OC_NODISCARD auto                                                             \
        func(const T &v, const T &u) noexcept {                                       \
        static constexpr auto N = vector_dimension_v<expr_value_t<T>>;                \
        static_assert(N == 2 || N == 3 || N == 4);                                    \
        if constexpr (N == 2) {                                                       \
            return T{func(v.x, u.x), func(v.y, u.y)};                                 \
        } else if constexpr (N == 3) {                                                \
            return T(func(v.x, u.x), func(v.y, u.y), func(v.z, u.z));                 \
        } else {                                                                      \
            return T(func(v.x, u.x), func(v.y, u.y), func(v.z, u.z), func(v.w, u.w)); \
        }                                                                             \
    }                                                                                 \
    template<typename T, typename U>                                                  \
    requires is_scalar_v<expr_value_t<T>> && is_vector_v<expr_value_t<U>>             \
        OC_NODISCARD auto                                                             \
        func(const T &t, const U &u) noexcept {                                       \
        static constexpr auto N = vector_dimension_v<expr_value_t<U>>;                \
        static_assert(N == 2 || N == 3 || N == 4);                                    \
        if constexpr (N == 2) {                                                       \
            return U{func(t, u.x), func(t, u.y)};                                     \
        } else if constexpr (N == 3) {                                                \
            return U(func(t, u.x), func(t, u.y), func(t, u.z));                       \
        } else {                                                                      \
            return U(func(t, u.x), func(t, u.y), func(t, u.z), func(t, u.w));         \
        }                                                                             \
    }                                                                                 \
    template<typename T, typename U>                                                  \
    requires is_vector_v<expr_value_t<T>> && is_scalar_v<expr_value_t<U>>             \
        OC_NODISCARD auto func(const T &v, const U &u) noexcept {                     \
        static constexpr auto N = vector_dimension_v<expr_value_t<T>>;                \
        static_assert(N == 2 || N == 3 || N == 4);                                    \
        if constexpr (N == 2) {                                                       \
            return T{func(v.x, u), func(v.y, u)};                                     \
        } else if constexpr (N == 3) {                                                \
            return T(func(v.x, u), func(v.y, u), func(v.z, u));                       \
        } else {                                                                      \
            return T(func(v.x, u), func(v.y, u), func(v.z, u), func(v.w, u));         \
        }                                                                             \
    }

MAKE_VECTOR_BINARY_FUNC(pow)
MAKE_VECTOR_BINARY_FUNC(min)
MAKE_VECTOR_BINARY_FUNC(max)
MAKE_VECTOR_BINARY_FUNC(atan2)

#undef MAKE_VECTOR_BINARY_FUNC

template<int n, typename T>
requires OC_MULTIPLY_CHECK(T, T)
OC_NODISCARD constexpr T Pow(T v) {
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
OC_NODISCARD constexpr auto
lerp(F t, A a, B b) noexcept {
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