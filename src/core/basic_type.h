//
// Created by Zero on 25/04/2022.
//

#pragma once

#include "basic_trait.h"

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