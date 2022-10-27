//
// Created by Zero on 26/07/2022.
//

#pragma once

#include "core/basic_types.h"
#include "core/concepts.h"
#include "core/constants.h"
#include "dsl/operators.h"
#include "dsl/type_trait.h"

namespace ocarina {

template<typename T, typename F>
[[nodiscard]] constexpr auto select(bool pred, T &&t, F &&f) noexcept {
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
requires std::is_unsigned_v<T> &&(sizeof(T) == 4u || sizeof(T) == 8u)
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
    OC_NODISCARD constexpr T sign(T val) {
    return select(val >= 0, T(1), T(-1));
}

template<typename T>
requires is_scalar_v<T>
    OC_NODISCARD constexpr auto
    radians(const T &deg) noexcept {
    return deg * (constants::Pi / 180.0f);
}

template<typename T>
requires is_scalar_v<T>
OC_NODISCARD constexpr auto
degrees(T rad) noexcept {
    return rad * (constants::InvPi * 180.0f);
}

template<typename T>
requires is_scalar_v<T>
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
MAKE_VECTOR_UNARY_FUNC(cos)
MAKE_VECTOR_UNARY_FUNC(sin)
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

template<typename U, typename V>
[[nodiscard]] constexpr auto abs_dot(const U &u, const V &v) noexcept {
    return abs(dot(u, v));
}

template<typename T>
[[nodiscard]] constexpr T safe_sqrt(const T& t) noexcept {
    return sqrt(max(0.f, t));
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

[[nodiscard]] inline uint32_t make_8bit(const float f) {
    return fmin(255, fmax(0, int(f * 256.f)));
}

template<typename V>
requires ocarina::is_vector3_v<expr_value_t<V>>
[[nodiscard]] inline uint32_t make_rgba(const V &color) {
    return (make_8bit(color.x) << 0) +
           (make_8bit(color.y) << 8) +
           (make_8bit(color.z) << 16) +
           (0xffU << 24);
}

template<typename V>
requires ocarina::is_vector4_v<expr_value_t<V>>
[[nodiscard]] inline uint32_t make_rgba(const V &color) {
    return (make_8bit(color.x) << 0) +
           (make_8bit(color.y) << 8) +
           (make_8bit(color.z) << 16) +
           (make_8bit(color.w) << 24);
}

#include "common_lib.h"

}// namespace ocarina