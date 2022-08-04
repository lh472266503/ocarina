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

template<typename T, size_t N>
requires ocarina::is_scalar_v<T> [
    [nodiscard]] constexpr auto
select(Vector<bool, N> pred, Vector<T, N> t, Vector<T, N> f) noexcept {
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
requires is_scalar_v<T>
OC_NODISCARD constexpr auto sqr(const T &v) {
    return v * v;
}

#define MAKE_VECTOR_UNARY_FUNC(func)                              \
    template<typename T>                                          \
    requires is_vector_v<T>                                       \
    OC_NODISCARD auto                                             \
    func(const T &v) noexcept {                                   \
        static constexpr auto N = vector_dimension_v<T>;          \
        static_assert(N == 2 || N == 3 || N == 4);                \
        if constexpr (N == 2) {                                   \
            return T{func(v.x), func(v.y)};                       \
        } else if constexpr (N == 3) {                            \
            return T(func(v.x), func(v.y), func(v.z));            \
        } else {                                                  \
            return T(func(v.x), func(v.y), func(v.z), func(v.w)); \
        }                                                         \
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
    OC_NODISCARD auto                                                                 \
    func(const T &v, const T &u) noexcept {                                           \
        static constexpr auto N = vector_dimension_v<T>;                              \
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
    requires is_scalar_v<T> && is_vector_v<U>                                         \
    OC_NODISCARD auto                                                                 \
    func(const T &t, const U &u) noexcept {                                           \
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
    requires is_vector_v<T> && is_scalar_v<U>                                         \
    OC_NODISCARD auto func(const T &v, const U &u) noexcept {                         \
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
requires none_dsl_v<F, A, B>
OC_NODISCARD constexpr auto
lerp(F t, A a, B b) noexcept {
    return a + t * (b - a);
}

template<typename T, size_t N>
[[nodiscard]] constexpr auto volume(Vector<T, N> v) noexcept {
    static_assert(N == 2 || N == 3 || N == 4);
    if constexpr (N == 2) {
        return v.x * v.y;
    } else if constexpr (N == 3) {
        return v.x * v.y * v.z;
    } else {
        return v.x * v.y * v.z * v.w;
    }
}

template<typename T, size_t N>
[[nodiscard]] auto dot(const Vector<T, N> &u, const Vector<T, N> &v) noexcept {
    static_assert(N == 2 || N == 3 || N == 4);
    if constexpr (N == 2) {
        return u.x * v.x + u.y * v.y;
    } else if constexpr (N == 3) {
        return u.x * v.x + u.y * v.y + u.z * v.z;
    } else {
        return u.x * v.x + u.y * v.y + u.z * v.z + u.w * v.w;
    }
}

template<typename T, size_t N>
[[nodiscard]] constexpr auto abs_dot(Vector<T, N> u, Vector<T, N> v) noexcept {
    return abs(dot(u, v));
}

template<typename T, size_t N>
[[nodiscard]] constexpr auto length(Vector<T, N> u) noexcept {
    return sqrt(dot(u, u));
}

template<typename T, size_t N>
[[nodiscard]] constexpr auto length_squared(Vector<T, N> u) noexcept {
    return dot(u, u);
}

template<typename T, size_t N>
[[nodiscard]] constexpr auto normalize(Vector<T, N> u) noexcept {
    return u * (1.0f / length(u));
}

template<typename T, size_t N>
[[nodiscard]] constexpr auto distance(Vector<T, N> u, Vector<T, N> v) noexcept {
    return length(u - v);
}

template<typename T, size_t N>
[[nodiscard]] constexpr auto distance_squared(Vector<T, N> u, Vector<T, N> v) noexcept {
    return length_squared(u - v);
}

template<typename T>
[[nodiscard]] constexpr auto cross(Vector<T, 3> u, Vector<T, 3> v) noexcept {
    return Vector<T, 3>(u.y * v.z - v.y * u.z,
                        u.z * v.x - v.z * u.x,
                        u.x * v.y - v.x * u.y);
}

[[nodiscard]] inline float3 face_forward(float3 v1, float3 v2) noexcept {
    return dot(v1, v2) > 0 ? v1 : -v1;
}

[[nodiscard]] inline auto transpose(const float2x2 m) noexcept {
    return make_float2x2(m[0].x, m[1].x, m[0].y, m[1].y);
}

[[nodiscard]] inline auto transpose(const float3x3 m) noexcept {
    return make_float3x3(m[0].x, m[1].x, m[2].x, m[0].y, m[1].y, m[2].y, m[0].z, m[1].z, m[2].z);
}

[[nodiscard]] inline auto transpose(const float4x4 m) noexcept {
    return make_float4x4(m[0].x, m[1].x, m[2].x, m[3].x, m[0].y, m[1].y, m[2].y, m[3].y, m[0].z, m[1].z, m[2].z, m[3].z, m[0].w, m[1].w, m[2].w, m[3].w);
}

[[nodiscard]] inline auto det(const float2x2 m) noexcept {
    return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

[[nodiscard]] inline auto det(const float3x3 m) noexcept {// from GLM
    return m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) - m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) + m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z);
}

[[nodiscard]] inline auto det(const float4x4 m) noexcept {// from GLM
    const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
    const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
    const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
    const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
    const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
    const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
    const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
    const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
    const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
    const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
    const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
    const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
    const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
    const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
    const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
    const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
    const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
    const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
    const auto fac0 = float4(coef00, coef00, coef02, coef03);
    const auto fac1 = float4(coef04, coef04, coef06, coef07);
    const auto fac2 = float4(coef08, coef08, coef10, coef11);
    const auto fac3 = float4(coef12, coef12, coef14, coef15);
    const auto fac4 = float4(coef16, coef16, coef18, coef19);
    const auto fac5 = float4(coef20, coef20, coef22, coef23);
    const auto Vec0 = float4(m[1].x, m[0].x, m[0].x, m[0].x);
    const auto Vec1 = float4(m[1].y, m[0].y, m[0].y, m[0].y);
    const auto Vec2 = float4(m[1].z, m[0].z, m[0].z, m[0].z);
    const auto Vec3 = float4(m[1].w, m[0].w, m[0].w, m[0].w);
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    const auto sign_a = float4(+1.0f, -1.0f, +1.0f, -1.0f);
    const auto sign_b = float4(-1.0f, +1.0f, -1.0f, +1.0f);
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m[0] * float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
    return dot0.x + dot0.y + dot0.z + dot0.w;
}

inline auto inverse(const float2x2 m) noexcept {
    const auto one_over_determinant = 1.0f / (m[0][0] * m[1][1] - m[1][0] * m[0][1]);
    return make_float2x2(m[1][1] * one_over_determinant,
                            -m[0][1] * one_over_determinant,
                            -m[1][0] * one_over_determinant,
                            +m[0][0] * one_over_determinant);
}

inline auto inverse(float3x3 m) noexcept {// from GLM
    float one_over_determinant = 1.0f / (m[0].x * (m[1].y * m[2].z - m[2].y * m[1].z) -
                                            m[1].x * (m[0].y * m[2].z - m[2].y * m[0].z) +
                                            m[2].x * (m[0].y * m[1].z - m[1].y * m[0].z));
    return make_float3x3(
        (m[1].y * m[2].z - m[2].y * m[1].z) * one_over_determinant,
        (m[2].y * m[0].z - m[0].y * m[2].z) * one_over_determinant,
        (m[0].y * m[1].z - m[1].y * m[0].z) * one_over_determinant,
        (m[2].x * m[1].z - m[1].x * m[2].z) * one_over_determinant,
        (m[0].x * m[2].z - m[2].x * m[0].z) * one_over_determinant,
        (m[1].x * m[0].z - m[0].x * m[1].z) * one_over_determinant,
        (m[1].x * m[2].y - m[2].x * m[1].y) * one_over_determinant,
        (m[2].x * m[0].y - m[0].x * m[2].y) * one_over_determinant,
        (m[0].x * m[1].y - m[1].x * m[0].y) * one_over_determinant);
}

[[nodiscard]] inline auto inverse(const float4x4 m) noexcept {// from GLM
    const auto coef00 = m[2].z * m[3].w - m[3].z * m[2].w;
    const auto coef02 = m[1].z * m[3].w - m[3].z * m[1].w;
    const auto coef03 = m[1].z * m[2].w - m[2].z * m[1].w;
    const auto coef04 = m[2].y * m[3].w - m[3].y * m[2].w;
    const auto coef06 = m[1].y * m[3].w - m[3].y * m[1].w;
    const auto coef07 = m[1].y * m[2].w - m[2].y * m[1].w;
    const auto coef08 = m[2].y * m[3].z - m[3].y * m[2].z;
    const auto coef10 = m[1].y * m[3].z - m[3].y * m[1].z;
    const auto coef11 = m[1].y * m[2].z - m[2].y * m[1].z;
    const auto coef12 = m[2].x * m[3].w - m[3].x * m[2].w;
    const auto coef14 = m[1].x * m[3].w - m[3].x * m[1].w;
    const auto coef15 = m[1].x * m[2].w - m[2].x * m[1].w;
    const auto coef16 = m[2].x * m[3].z - m[3].x * m[2].z;
    const auto coef18 = m[1].x * m[3].z - m[3].x * m[1].z;
    const auto coef19 = m[1].x * m[2].z - m[2].x * m[1].z;
    const auto coef20 = m[2].x * m[3].y - m[3].x * m[2].y;
    const auto coef22 = m[1].x * m[3].y - m[3].x * m[1].y;
    const auto coef23 = m[1].x * m[2].y - m[2].x * m[1].y;
    const auto fac0 = float4(coef00, coef00, coef02, coef03);
    const auto fac1 = float4(coef04, coef04, coef06, coef07);
    const auto fac2 = float4(coef08, coef08, coef10, coef11);
    const auto fac3 = float4(coef12, coef12, coef14, coef15);
    const auto fac4 = float4(coef16, coef16, coef18, coef19);
    const auto fac5 = float4(coef20, coef20, coef22, coef23);
    const auto Vec0 = float4(m[1].x, m[0].x, m[0].x, m[0].x);
    const auto Vec1 = float4(m[1].y, m[0].y, m[0].y, m[0].y);
    const auto Vec2 = float4(m[1].z, m[0].z, m[0].z, m[0].z);
    const auto Vec3 = float4(m[1].w, m[0].w, m[0].w, m[0].w);
    const auto inv0 = Vec1 * fac0 - Vec2 * fac1 + Vec3 * fac2;
    const auto inv1 = Vec0 * fac0 - Vec2 * fac3 + Vec3 * fac4;
    const auto inv2 = Vec0 * fac1 - Vec1 * fac3 + Vec3 * fac5;
    const auto inv3 = Vec0 * fac2 - Vec1 * fac4 + Vec2 * fac5;
    const auto sign_a = float4(+1.0f, -1.0f, +1.0f, -1.0f);
    const auto sign_b = float4(-1.0f, +1.0f, -1.0f, +1.0f);
    const auto inv_0 = inv0 * sign_a;
    const auto inv_1 = inv1 * sign_b;
    const auto inv_2 = inv2 * sign_a;
    const auto inv_3 = inv3 * sign_b;
    const auto dot0 = m[0] * float4(inv_0.x, inv_1.x, inv_2.x, inv_3.x);
    const auto dot1 = dot0.x + dot0.y + dot0.z + dot0.w;
    const auto one_over_determinant = 1.0f / dot1;
    return make_float4x4(inv_0 * one_over_determinant,
                            inv_1 * one_over_determinant,
                            inv_2 * one_over_determinant,
                            inv_3 * one_over_determinant);
}


template<typename T, size_t N>
[[nodiscard]] auto triangle_area(Vector<T, N> p0, Vector<T, N> p1, Vector<T, N> p2) noexcept {
    static_assert(N == 3 || N == 2, "N must be greater than 1!");
    if constexpr (N == 2) {
        Vector<T, 3> pp0 = Vector<T, 3>{p0.x, p0.y, 0};
        Vector<T, 3> pp1 = Vector<T, 3>{p1.x, p1.y, 0};
        Vector<T, 3> pp2 = Vector<T, 3>{p2.x, p2.y, 0};
        return 0.5 * length(cross(pp1 - pp0, pp2 - pp0));
    } else {
        return 0.5 * length(cross(p1 - p0, p2 - p0));
    }
}

template<typename T, typename F2>
[[nodiscard]] T triangle_lerp(F2 barycentric, T v0, T v1, T v2) noexcept {
    auto u = barycentric.x;
    auto v = barycentric.y;
    auto w = 1 - barycentric.x - barycentric.y;
    return u * v0 + v * v1 + w * v2;
}

}// namespace ocarina