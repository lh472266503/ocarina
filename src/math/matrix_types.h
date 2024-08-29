//
// Created by Zero on 2024/5/20.
//

#pragma once

#include "vector_types.h"

namespace ocarina {

template<size_t N, size_t M>
struct Matrix {
public:
    static constexpr auto RowNum = M;
    static constexpr auto ColNum = N;
    static constexpr auto ElementNum = M * N;
    using scalar_type = float;
    using vector_type = Vector<scalar_type, M>;
    using array_t = array<vector_type, N>;

private:
    array_t cols_{};

public:
    Matrix() = default;

    template<typename... Args>
    requires(sizeof...(Args) == N && all_is_v<vector_type, Args...>)
    explicit constexpr Matrix(Args &&...args) noexcept
        : cols_(array_t{OC_FORWARD(args)...}) {}

    template<typename... Args>
    requires(sizeof...(Args) == ElementNum && all_is_v<scalar_type, Args...>)
    explicit constexpr Matrix(Args &&...args) noexcept
        : cols_([&]<size_t... i>(std::index_sequence<i...>,
                                 const array<scalar_type, ElementNum> &arr) {
              return array_t{vector_type{addressof(arr.data()[i * M])}...};
          }(std::make_index_sequence<N>(), array<scalar_type, ElementNum>{OC_FORWARD(args)...})) {
    }

    [[nodiscard]] constexpr vector_type &operator[](size_t i) noexcept { return cols_[i]; }
    [[nodiscard]] constexpr const vector_type &operator[](size_t i) const noexcept { return cols_[i]; }
};

#define OC_MAKE_MATRIX_(N, M) \
    using float##N##x##M = Matrix<N, M>;

OC_MAKE_MATRIX_(2, 2)
OC_MAKE_MATRIX_(2, 3)
OC_MAKE_MATRIX_(2, 4)
OC_MAKE_MATRIX_(3, 2)
OC_MAKE_MATRIX_(3, 3)
OC_MAKE_MATRIX_(3, 4)
OC_MAKE_MATRIX_(4, 2)
OC_MAKE_MATRIX_(4, 3)
OC_MAKE_MATRIX_(4, 4)

}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator-(ocarina::Matrix<N, M> m) {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ocarina::Matrix<N, M>((-m[i])...);
    }(std::make_index_sequence<N>());
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator*(ocarina::Matrix<N, M> m, float s) {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ocarina::Matrix<N, M>((m[i] * s)...);
    }(std::make_index_sequence<N>());
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator*(float s, ocarina::Matrix<N, M> m) {
    return m * s;
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator/(ocarina::Matrix<N, M> m, float s) {
    return m * (1.0f / s);
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator*(ocarina::Matrix<N, M> m, ocarina::Vector<float, N> v) noexcept {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ((v[i] * m[i]) + ...);
    }(std::make_index_sequence<N>());
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator*(ocarina::Matrix<N, M> lhs, ocarina::Matrix<M, N> rhs) noexcept {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ocarina::Matrix<M, M>(lhs * rhs[i]...);
    }(std::make_index_sequence<M>());
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator+(ocarina::Matrix<N, M> lhs, ocarina::Matrix<N, M> rhs) noexcept {
    return [&]<size_t... i>(std::index_sequence<i...>) {
        return ocarina::Matrix<N, M>(lhs[i] + rhs[i]...);
    }(std::make_index_sequence<N>());
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator-(ocarina::Matrix<N, M> lhs, ocarina::Matrix<N, M> rhs) noexcept {
    return lhs + (-rhs);
}

namespace ocarina {

[[nodiscard]] constexpr auto make_float2x2(float s = 1.0f) noexcept {
    return float2x2{float2{s, 0.0f},
                    float2{0.0f, s}};
}

[[nodiscard]] constexpr auto make_float2x2(
    float m00, float m01,
    float m10, float m11) noexcept {
    return float2x2{float2{m00, m01},
                    float2{m10, m11}};
}

[[nodiscard]] constexpr auto make_float2x2(float2 c0, float2 c1) noexcept {
    return float2x2{c0, c1};
}

[[nodiscard]] constexpr auto make_float2x2(float2x2 m) noexcept {
    return m;
}

[[nodiscard]] constexpr auto make_float2x2(float3x3 m) noexcept {
    return float2x2{float2{m[0].x, m[0].y},
                    float2{m[1].x, m[1].y}};
}

[[nodiscard]] constexpr auto make_float2x2(float4x4 m) noexcept {
    return float2x2{float2{m[0].x, m[0].y},
                    float2{m[1].x, m[1].y}};
}

[[nodiscard]] constexpr auto make_float3x3(float s = 1.0f) noexcept {
    return float3x3{float3{s, 0.0f, 0.0f},
                    float3{0.0f, s, 0.0f},
                    float3{0.0f, 0.0f, s}};
}

[[nodiscard]] constexpr auto make_float3x3(float3 c0, float3 c1, float3 c2) noexcept {
    return float3x3{c0, c1, c2};
}

[[nodiscard]] constexpr auto make_float3x3(
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22) noexcept {
    return float3x3{float3{m00, m01, m02},
                    float3{m10, m11, m12},
                    float3{m20, m21, m22}};
}

[[nodiscard]] constexpr auto make_float3x3(float2x2 m) noexcept {
    return float3x3{make_float3(m[0], 0.0f),
                    make_float3(m[1], 0.0f),
                    make_float3(0.f, 0.f, 1.0f)};
}

[[nodiscard]] constexpr auto make_float3x3(float3x3 m) noexcept {
    return m;
}

[[nodiscard]] constexpr auto make_float3x3(float4x4 m) noexcept {
    return float3x3{make_float3(m[0]),
                    make_float3(m[1]),
                    make_float3(m[2])};
}

[[nodiscard]] constexpr auto make_float4x4(float s = 1.0f) noexcept {
    return float4x4{float4{s, 0.0f, 0.0f, 0.0f},
                    float4{0.0f, s, 0.0f, 0.0f},
                    float4{0.0f, 0.0f, s, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, s}};
}

[[nodiscard]] constexpr auto make_float4x4(float4 c0, float4 c1, float4 c2, float4 c3) noexcept {
    return float4x4{c0, c1, c2, c3};
}

[[nodiscard]] constexpr auto make_float4x4(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33) noexcept {
    return float4x4{float4{m00, m01, m02, m03},
                    float4{m10, m11, m12, m13},
                    float4{m20, m21, m22, m23},
                    float4{m30, m31, m32, m33}};
}

[[nodiscard]] constexpr auto make_float4x4(float2x2 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f, 0.0f),
                    make_float4(m[1], 0.0f, 0.0f),
                    float4{0.0f, 0.0f, 1.0f, 0.0f},
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

[[nodiscard]] constexpr auto make_float4x4(float3x3 m) noexcept {
    return float4x4{make_float4(m[0], 0.0f),
                    make_float4(m[1], 0.0f),
                    make_float4(m[2], 0.0f),
                    float4{0.0f, 0.0f, 0.0f, 1.0f}};
}

[[nodiscard]] constexpr auto make_float4x4(float4x4 m) noexcept {
    return m;
}
}// namespace ocarina