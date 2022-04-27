//
// Created by Zero on 25/04/2022.
//

#pragma once

#include "basic_traits.h"

namespace sycamore {
inline namespace size_literals {

SCM_NODISCARD constexpr auto operator""_kb(unsigned long long bytes) noexcept {
    return static_cast<size_t>(bytes * 1024u);
}

SCM_NODISCARD constexpr auto operator""_mb(unsigned long long bytes) noexcept {
    return static_cast<size_t>(bytes * 1024u * 1024u);
}

SCM_NODISCARD constexpr auto operator""_gb(unsigned long long bytes) noexcept {
    return static_cast<size_t>(bytes * 1024u * 1024u * 1024u);
}

}// namespace size_literals

namespace detail {
template<typename T, size_t N>
struct VectorStorage {
    static_assert(always_false_v<T>, "Invalid vector storage");
};

template<typename T>
struct alignas(sizeof(T) * 2) VectorStorage<T, 2> {
    T x, y;
    explicit constexpr VectorStorage(T s = {}) noexcept : x{s}, y{s} {}
    constexpr VectorStorage(T x, T y) noexcept : x{x}, y{y} {}
#include "swizzle_2.inl.h"
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 3> {
    T x, y, z;
    explicit constexpr VectorStorage(T s = {}) noexcept : x{s}, y{s}, z{s} {}
    constexpr VectorStorage(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
#include "swizzle_3.inl.h"
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 4> {
    T x, y, z, w;
    explicit constexpr VectorStorage(T s = {}) noexcept : x{s}, y{s}, z{s}, w{s} {}
    constexpr VectorStorage(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
#include "swizzle_4.inl.h"
};

}// namespace detail

template<typename T, size_t N>
struct Vector : public detail::VectorStorage<T, N> {
    static constexpr auto dimension = N;
    using value_type = T;
    using Storage = detail::VectorStorage<T, N>;
    static_assert(std::disjunction_v<
                      std::is_same<T, bool>,
                      std::is_same<T, float>,
                      std::is_same<T, int>,
                      std::is_same<T, uint>>,
                  "Invalid vector type");
    static_assert(N == 2 || N == 3 || N == 4, "Invalid vector dimension");
    using Storage::VectorStorage;
    SCM_NODISCARD constexpr T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    SCM_NODISCARD constexpr const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
};

#define SYCAMORE_MAKE_VECTOR_TYPES(T) \
    using T##2 = Vector<T, 2>;        \
    using T##3 = Vector<T, 3>;        \
    using T##4 = Vector<T, 4>;

SYCAMORE_MAKE_VECTOR_TYPES(bool)
SYCAMORE_MAKE_VECTOR_TYPES(float)
SYCAMORE_MAKE_VECTOR_TYPES(int)
SYCAMORE_MAKE_VECTOR_TYPES(uint)

#undef SYCAMORE_MAKE_VECTOR_TYPES

template<size_t N>
struct Matrix {
    static_assert(always_false_v<std::integral_constant<size_t, N>>, "Invalid matrix type");
};

template<>
struct Matrix<2> {

    float2 cols[2];

    constexpr Matrix() noexcept
        : cols{float2{1.0f, 0.0f}, float2{0.0f, 1.0f}} {}

    constexpr Matrix(const float2 c0, const float2 c1) noexcept
        : cols{c0, c1} {}

    SCM_NODISCARD constexpr float2 &operator[](size_t i) noexcept { return cols[i]; }
    SCM_NODISCARD constexpr const float2 &operator[](size_t i) const noexcept { return cols[i]; }
};

template<>
struct Matrix<3> {

    float3 cols[3];

    constexpr Matrix() noexcept
        : cols{float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f}} {}

    constexpr Matrix(const float3 c0, const float3 c1, const float3 c2) noexcept
        : cols{c0, c1, c2} {}

    SCM_NODISCARD constexpr float3 &operator[](size_t i) noexcept { return cols[i]; }
    SCM_NODISCARD constexpr const float3 &operator[](size_t i) const noexcept { return cols[i]; }
};

template<>
struct Matrix<4> {

    float4 cols[4];

    constexpr Matrix() noexcept
        : cols{float4{1.0f, 0.0f, 0.0f, 0.0f},
               float4{0.0f, 1.0f, 0.0f, 0.0f},
               float4{0.0f, 0.0f, 1.0f, 0.0f},
               float4{0.0f, 0.0f, 0.0f, 1.0f}} {}

    constexpr Matrix(const float4 c0, const float4 c1, const float4 c2, const float4 c3) noexcept
        : cols{c0, c1, c2, c3} {}

    SCM_NODISCARD constexpr float4 &operator[](size_t i) noexcept { return cols[i]; }
    SCM_NODISCARD constexpr const float4 &operator[](size_t i) const noexcept { return cols[i]; }
};

using float2x2 = Matrix<2>;
using float3x3 = Matrix<3>;
using float4x4 = Matrix<4>;

using basic_types = std::tuple<
    bool, float, int, uint,
    bool2, float2, int2, uint2,
    bool3, float3, int3, uint3,
    bool4, float4, int4, uint4,
    float2x2, float3x3, float4x4>;

SCM_NODISCARD constexpr bool any(const bool2 v) noexcept { return v.x || v.y; }
SCM_NODISCARD constexpr bool any(const bool3 v) noexcept { return v.x || v.y || v.z; }
SCM_NODISCARD constexpr bool any(const bool4 v) noexcept { return v.x || v.y || v.z || v.w; }

SCM_NODISCARD constexpr bool all(const bool2 v) noexcept { return v.x && v.y; }
SCM_NODISCARD constexpr bool all(const bool3 v) noexcept { return v.x && v.y && v.z; }
SCM_NODISCARD constexpr bool all(const bool4 v) noexcept { return v.x && v.y && v.z && v.w; }

SCM_NODISCARD constexpr bool none(const bool2 v) noexcept { return !any(v); }
SCM_NODISCARD constexpr bool none(const bool3 v) noexcept { return !any(v); }
SCM_NODISCARD constexpr bool none(const bool4 v) noexcept { return !any(v); }

}// namespace sycamore

template<typename T, size_t N>
requires sycamore::is_number_v<T>
    SCM_NODISCARD constexpr auto operator+(const sycamore::Vector<T, N> v) noexcept {
    return v;
}

template<typename T, size_t N>
requires sycamore::is_number_v<T>
    SCM_NODISCARD constexpr auto operator-(const sycamore::Vector<T, N> v) noexcept {
    using R = sycamore::Vector<T, N>;
    if constexpr (N == 2) {
        return R{-v.x, -v.y};
    } else if constexpr (N == 3) {
        return R{-v.x, -v.y, -v.z};
    } else {
        return R{-v.x, -v.y, -v.z, -v.w};
    }
}

template<typename T, size_t N>
SCM_NODISCARD constexpr auto operator!(const sycamore::Vector<T, N> v) noexcept {
    if constexpr (N == 2u) {
        return sycamore::bool2{!v.x, !v.y};
    } else if constexpr (N == 3u) {
        return sycamore::bool3{!v.x, !v.y, !v.z};
    } else {
        return sycamore::bool3{!v.x, !v.y, !v.z, !v.w};
    }
}

template<typename T, size_t N>
requires sycamore::is_integral_v<T>
    SCM_NODISCARD constexpr auto operator~(const sycamore::Vector<T, N> v) noexcept {
    using R = sycamore::Vector<T, N>;
    if constexpr (N == 2) {
        return R{~v.x, ~v.y};
    } else if constexpr (N == 3) {
        return R{~v.x, ~v.y, ~v.z};
    } else {
        return R{~v.x, ~v.y, ~v.z, ~v.w};
    }
}

#define SCM_MAKE_VECTOR_BINARY_OPERATOR(op, ...)                               \
    template<typename T, size_t N>                                             \
    requires __VA_ARGS__                                                       \
        SCM_NODISCARD constexpr auto                                           \
        operator op(                                                           \
            sycamore::Vector<T, N> lhs, sycamore::Vector<T, N> rhs) noexcept { \
        if constexpr (N == 2) {                                                \
            return sycamore::Vector<T, 2>{                                     \
                lhs.x op rhs.x,                                                \
                lhs.y op rhs.y};                                               \
        } else if constexpr (N == 3) {                                         \
            return sycamore::Vector<T, 3>{                                     \
                lhs.x op rhs.x,                                                \
                lhs.y op rhs.y,                                                \
                lhs.z op rhs.z};                                               \
        } else {                                                               \
            return sycamore::Vector<T, 4>{                                     \
                lhs.x op rhs.x,                                                \
                lhs.y op rhs.y,                                                \
                lhs.z op rhs.z,                                                \
                lhs.w op rhs.w};                                               \
        }                                                                      \
    }                                                                          \
    template<typename T, size_t N>                                             \
    requires __VA_ARGS__                                                       \
        SCM_NODISCARD constexpr auto                                           \
        operator op(sycamore::Vector<T, N> lhs, T rhs) noexcept {              \
        return lhs op sycamore::Vector<T, N>{rhs};                             \
    }                                                                          \
    template<typename T, size_t N>                                             \
    requires __VA_ARGS__                                                       \
        SCM_NODISCARD constexpr auto                                           \
        operator op(T lhs, sycamore::Vector<T, N> rhs) noexcept {              \
        return sycamore::Vector<T, N>{lhs} op rhs;                             \
    }
SCM_MAKE_VECTOR_BINARY_OPERATOR(+, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_BINARY_OPERATOR(-, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_BINARY_OPERATOR(*, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_BINARY_OPERATOR(/, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_BINARY_OPERATOR(%, sycamore::is_integral_v<T>)
SCM_MAKE_VECTOR_BINARY_OPERATOR(>>, sycamore::is_integral_v<T>)
SCM_MAKE_VECTOR_BINARY_OPERATOR(<<, sycamore::is_integral_v<T>)
SCM_MAKE_VECTOR_BINARY_OPERATOR(|, std::negation_v<sycamore::is_floating_point<T>>)
SCM_MAKE_VECTOR_BINARY_OPERATOR(&, std::negation_v<sycamore::is_floating_point<T>>)
SCM_MAKE_VECTOR_BINARY_OPERATOR(^, std::negation_v<sycamore::is_floating_point<T>>)

#undef SCM_MAKE_VECTOR_BINARY_OPERATOR

#define SCM_MAKE_VECTOR_ASSIGN_OPERATOR(op, ...)                            \
    template<typename T, size_t N>                                          \
    requires __VA_ARGS__ constexpr decltype(auto) operator op(              \
        sycamore::Vector<T, N> &lhs, sycamore::Vector<T, N> rhs) noexcept { \
        lhs.x op rhs.x;                                                     \
        lhs.y op rhs.y;                                                     \
        if constexpr (N >= 3) { lhs.z op rhs.z; }                           \
        if constexpr (N == 4) { lhs.w op rhs.w; }                           \
        return (lhs);                                                       \
    }                                                                       \
    template<typename T, size_t N>                                          \
    requires __VA_ARGS__ constexpr decltype(auto) operator op(              \
        sycamore::Vector<T, N> &lhs, T rhs) noexcept {                      \
        return (lhs op sycamore::Vector<T, N>{rhs});                        \
    }
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(+=, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(-=, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(*=, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(/=, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(%=, sycamore::is_integral_v<T>)
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(<<=, sycamore::is_integral_v<T>)
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(>>=, sycamore::is_integral_v<T>)
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(|=, std::negation_v<sycamore::is_floating_point<T>>)
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(&=, std::negation_v<sycamore::is_floating_point<T>>)
SCM_MAKE_VECTOR_ASSIGN_OPERATOR(^=, std::negation_v<sycamore::is_floating_point<T>>)

#undef SCM_MAKE_VECTOR_ASSIGN_OPERATOR

#define SCM_MAKE_VECTOR_LOGIC_OPERATOR(op, ...)                                \
    template<typename T, size_t N>                                             \
    requires __VA_ARGS__                                                       \
        SCM_NODISCARD constexpr auto                                           \
        operator op(                                                           \
            sycamore::Vector<T, N> lhs, sycamore::Vector<T, N> rhs) noexcept { \
        if constexpr (N == 2) {                                                \
            return sycamore::bool2{                                            \
                lhs.x op rhs.x,                                                \
                lhs.y op rhs.y};                                               \
        } else if constexpr (N == 3) {                                         \
            return sycamore::bool3{                                            \
                lhs.x op rhs.x,                                                \
                lhs.y op rhs.y,                                                \
                lhs.z op rhs.z};                                               \
        } else {                                                               \
            return sycamore::bool4{                                            \
                lhs.x op rhs.x,                                                \
                lhs.y op rhs.y,                                                \
                lhs.z op rhs.z,                                                \
                lhs.w op rhs.w};                                               \
        }                                                                      \
    }                                                                          \
    template<typename T, size_t N>                                             \
    requires __VA_ARGS__                                                       \
        SCM_NODISCARD constexpr auto                                           \
        operator op(sycamore::Vector<T, N> lhs, T rhs) noexcept {              \
        return lhs op sycamore::Vector<T, N>{rhs};                             \
    }                                                                          \
    template<typename T, size_t N>                                             \
    requires __VA_ARGS__                                                       \
        SCM_NODISCARD constexpr auto                                           \
        operator op(T lhs, sycamore::Vector<T, N> rhs) noexcept {              \
        return sycamore::Vector<T, N>{lhs} op rhs;                             \
    }
SCM_MAKE_VECTOR_LOGIC_OPERATOR(||, sycamore::is_boolean_v<T>)
SCM_MAKE_VECTOR_LOGIC_OPERATOR(&&, sycamore::is_boolean_v<T>)
SCM_MAKE_VECTOR_LOGIC_OPERATOR(==, true)
SCM_MAKE_VECTOR_LOGIC_OPERATOR(!=, true)
SCM_MAKE_VECTOR_LOGIC_OPERATOR(<, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_LOGIC_OPERATOR(>, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_LOGIC_OPERATOR(<=, sycamore::is_number_v<T>)
SCM_MAKE_VECTOR_LOGIC_OPERATOR(>=, sycamore::is_number_v<T>)

#undef SCM_MAKE_VECTOR_LOGIC_OPERATOR

SCM_NODISCARD constexpr auto operator*(const sycamore::float2x2 m, float s) noexcept {
    return sycamore::float2x2{m[0] * s, m[1] * s};
}

SCM_NODISCARD constexpr auto operator*(float s, const sycamore::float2x2 m) noexcept {
    return m * s;
}

SCM_NODISCARD constexpr auto operator/(const sycamore::float2x2 m, float s) noexcept {
    return m * (1.0f / s);
}

SCM_NODISCARD constexpr auto operator*(const sycamore::float2x2 m, const sycamore::float2 v) noexcept {
    return v.x * m[0] + v.y * m[1];
}

SCM_NODISCARD constexpr auto operator*(const sycamore::float2x2 lhs, const sycamore::float2x2 rhs) noexcept {
    return sycamore::float2x2{lhs * rhs[0], lhs * rhs[1]};
}

SCM_NODISCARD constexpr auto operator+(const sycamore::float2x2 lhs, const sycamore::float2x2 rhs) noexcept {
    return sycamore::float2x2{lhs[0] + rhs[0], lhs[1] + rhs[1]};
}

SCM_NODISCARD constexpr auto operator-(const sycamore::float2x2 lhs, const sycamore::float2x2 rhs) noexcept {
    return sycamore::float2x2{lhs[0] - rhs[0], lhs[1] - rhs[1]};
}

SCM_NODISCARD constexpr auto operator*(const sycamore::float3x3 m, float s) noexcept {
    return sycamore::float3x3{m[0] * s, m[1] * s, m[2] * s};
}

SCM_NODISCARD constexpr auto operator*(float s, const sycamore::float3x3 m) noexcept {
    return m * s;
}

SCM_NODISCARD constexpr auto operator/(const sycamore::float3x3 m, float s) noexcept {
    return m * (1.0f / s);
}

SCM_NODISCARD constexpr auto operator*(const sycamore::float3x3 m, const sycamore::float3 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2];
}

SCM_NODISCARD constexpr auto operator*(const sycamore::float3x3 lhs, const sycamore::float3x3 rhs) noexcept {
    return sycamore::float3x3{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
}

SCM_NODISCARD constexpr auto operator+(const sycamore::float3x3 lhs, const sycamore::float3x3 rhs) noexcept {
    return sycamore::float3x3{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

SCM_NODISCARD constexpr auto operator-(const sycamore::float3x3 lhs, const sycamore::float3x3 rhs) noexcept {
    return sycamore::float3x3{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

SCM_NODISCARD constexpr auto operator*(const sycamore::float4x4 m, float s) noexcept {
    return sycamore::float4x4{m[0] * s, m[1] * s, m[2] * s, m[3] * s};
}

SCM_NODISCARD constexpr auto operator*(float s, const sycamore::float4x4 m) noexcept {
    return m * s;
}

SCM_NODISCARD constexpr auto operator/(const sycamore::float4x4 m, float s) noexcept {
    return m * (1.0f / s);
}

SCM_NODISCARD constexpr auto operator*(const sycamore::float4x4 m, const sycamore::float4 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2] + v.w * m[3];
}

SCM_NODISCARD constexpr auto operator*(const sycamore::float4x4 lhs, const sycamore::float4x4 rhs) noexcept {
    return sycamore::float4x4{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]};
}

SCM_NODISCARD constexpr auto operator+(const sycamore::float4x4 lhs, const sycamore::float4x4 rhs) noexcept {
    return sycamore::float4x4{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]};
}

SCM_NODISCARD constexpr auto operator-(const sycamore::float4x4 lhs, const sycamore::float4x4 rhs) noexcept {
    return sycamore::float4x4{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]};
}

namespace sycamore {
#define SCM_MAKE_TYPE_N(type)                                                                                                \
    SCM_NODISCARD constexpr auto make_##type##2(type s = {}) noexcept { return type##2(s); }                                 \
    SCM_NODISCARD constexpr auto make_##type##2(type x, type y) noexcept { return type##2(x, y); }                           \
    template<typename T>                                                                                                     \
    SCM_NODISCARD constexpr auto make_##type##2(Vector<T, 2> v) noexcept {                                                   \
        return type##2(                                                                                                      \
            static_cast<type>(v.x),                                                                                          \
            static_cast<type>(v.y));                                                                                         \
    }                                                                                                                        \
    SCM_NODISCARD constexpr auto make_##type##2(type##3 v) noexcept { return type##2(v.x, v.y); }                            \
    SCM_NODISCARD constexpr auto make_##type##2(type##4 v) noexcept { return type##2(v.x, v.y); }                            \
                                                                                                                             \
    SCM_NODISCARD constexpr auto make_##type##3(type s = {}) noexcept { return type##3(s); }                                 \
    SCM_NODISCARD constexpr auto make_##type##3(type x, type y, type z) noexcept { return type##3(x, y, z); }                \
    template<typename T>                                                                                                     \
    SCM_NODISCARD constexpr auto make_##type##3(Vector<T, 3> v) noexcept {                                                   \
        return type##3(                                                                                                      \
            static_cast<type>(v.x),                                                                                          \
            static_cast<type>(v.y),                                                                                          \
            static_cast<type>(v.z));                                                                                         \
    }                                                                                                                        \
    SCM_NODISCARD constexpr auto make_##type##3(type##2 v, type z) noexcept { return type##3(v.x, v.y, z); }                 \
    SCM_NODISCARD constexpr auto make_##type##3(type x, type##2 v) noexcept { return type##3(x, v.x, v.y); }                 \
    SCM_NODISCARD constexpr auto make_##type##3(type##4 v) noexcept { return type##3(v.x, v.y, v.z); }                       \
                                                                                                                             \
    SCM_NODISCARD constexpr auto make_##type##4(type s = {}) noexcept { return type##4(s); }                                 \
    SCM_NODISCARD constexpr auto make_##type##4(type x, type y, type z, type w) noexcept { return type##4(x, y, z, w); }     \
    template<typename T>                                                                                                     \
    SCM_NODISCARD constexpr auto make_##type##4(Vector<T, 4> v) noexcept {                                                   \
        return type##4(                                                                                                      \
            static_cast<type>(v.x),                                                                                          \
            static_cast<type>(v.y),                                                                                          \
            static_cast<type>(v.z),                                                                                          \
            static_cast<type>(v.w));                                                                                         \
    }                                                                                                                        \
    SCM_NODISCARD constexpr auto make_##type##4(type##2 v, type z, type w) noexcept { return type##4(v.x, v.y, z, w); }      \
    SCM_NODISCARD constexpr auto make_##type##4(type x, type##2 v, type w) noexcept { return type##4(x, v.x, v.y, w); }      \
    SCM_NODISCARD constexpr auto make_##type##4(type x, type y, type##2 v) noexcept { return type##4(x, y, v.x, v.y); }      \
    SCM_NODISCARD constexpr auto make_##type##4(type##2 xy, type##2 zw) noexcept { return type##4(xy.x, xy.y, zw.x, zw.y); } \
    SCM_NODISCARD constexpr auto make_##type##4(type##3 v, type w) noexcept { return type##4(v.x, v.y, v.z, w); }            \
    SCM_NODISCARD constexpr auto make_##type##4(type x, type##3 v) noexcept { return type##4(x, v.x, v.y, v.z); }

SCM_MAKE_TYPE_N(bool)
SCM_MAKE_TYPE_N(float)
SCM_MAKE_TYPE_N(int)
SCM_MAKE_TYPE_N(uint)
#undef SCM_MAKE_TYPE_N

SCM_NODISCARD constexpr auto make_float2x2(float s = 1.0f) noexcept {
    return float2x2{float2{s, 0.0f},
                    float2{0.0f, s}};
}

SCM_NODISCARD constexpr auto make_float2x2(
    float m00, float m01,
    float m10, float m11) noexcept {
    return float2x2{float2{m00, m01},
                    float2{m10, m11}};
}

SCM_NODISCARD constexpr auto make_float2x2(float2 c0, float2 c1) noexcept {
    return float2x2{c0, c1};
}

SCM_NODISCARD constexpr auto make_float2x2(float2x2 m) noexcept {
    return m;
}

SCM_NODISCARD constexpr auto make_float2x2(float3x3 m) noexcept {
    return float2x2{float2{m[0].x, m[0].y},
                    float2{m[1].x, m[1].y}};
}

SCM_NODISCARD constexpr auto make_float2x2(float4x4 m) noexcept {
    return float2x2{float2{m[0].x, m[0].y},
                    float2{m[1].x, m[1].y}};
}

SCM_NODISCARD constexpr auto make_float3x3(float s = 1.0f) noexcept {
    return float3x3{float3{s, 0.0f, 0.0f},
                    float3{0.0f, s, 0.0f},
                    float3{0.0f, 0.0f, s}};
}

SCM_NODISCARD constexpr auto make_float3x3(float3 c0, float3 c1, float3 c2) noexcept {
    return float3x3{c0, c1, c2};
}

SCM_NODISCARD constexpr auto make_float3x3(
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22) noexcept {
    return float3x3{float3{m00, m01, m02},
                    float3{m10, m11, m12},
                    float3{m20, m21, m22}};
}

SCM_NODISCARD constexpr auto make_float3x3(float2x2 m) noexcept {
    return float3x3{make_float3(m[0], 0.0f),
                    make_float3(m[1], 0.0f),
                    make_float3(0.f, 0.f, 1.0f)};
}

SCM_NODISCARD constexpr auto make_float3x3(float3x3 m) noexcept {
    return m;
}

SCM_NODISCARD constexpr auto make_float3x3(float4x4 m) noexcept {
    return float3x3{make_float3(m[0]),
                    make_float3(m[1]),
                    make_float3(m[2])};
}

SCM_NODISCARD constexpr auto make_float4x4(float s = 1.0f) noexcept {
    return float4x4{float4{s, 0.0f, 0.0f, 0.0f},
                    float4{0.0f, s, 0.0f, 0.0f},
                    float4{0.0f, 0.0f, s, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, s}};
}

SCM_NODISCARD constexpr auto make_float4x4(float4 c0, float4 c1, float4 c2, float4 c3) noexcept {
    return float4x4{c0, c1, c2, c3};
}

SCM_NODISCARD constexpr auto make_float4x4(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33) noexcept {
    return float4x4{float4{m00, m01, m02, m03},
                    float4{m10, m11, m12, m13},
                    float4{m20, m21, m22, m23},
                    float4{m30, m31, m32, m33}};
}

SCM_NODISCARD constexpr auto make_float4x4(float2x2 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f, 0.0f),
                    make_float4(m[1], 0.0f, 0.0f),
                    float4{0.0f, 0.0f, 1.0f, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

SCM_NODISCARD constexpr auto make_float4x4(float3x3 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f),
                    make_float4(m[1], 0.0f),
                    make_float4(m[2], 0.0f),
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

SCM_NODISCARD constexpr auto make_float4x4(float4x4 m) noexcept {
    return m;
}

}// namespace sycamore