//
// Created by Zero on 2024/5/20.
//

#pragma once

#include "vector_types.h"

namespace ocarina {
template<size_t N>
struct Matrix {
    static_assert(always_false_v<std::integral_constant<size_t, N>>, "Invalid matrix type");
};

template<size_t N, size_t M>
struct Mat {
public:
    static constexpr auto RowNum = M;
    static constexpr auto ColNum = N;
    using scalar_type = float;
    using vector_type = Vector<scalar_type, M>;

private:
    array<vector_type, N> cols_{};

public:
    Mat() = default;
    template<typename... Args>
    requires(sizeof...(Args) == N && all_is_v<vector_type, Args...>)
    constexpr Mat(Args &&...args) noexcept : cols_(array<vector_type, N>{OC_FORWARD(args)...}) {}
    [[nodiscard]] constexpr vector_type &operator[](size_t i) noexcept { return cols_[i]; }
    [[nodiscard]] constexpr const vector_type &operator[](size_t i) const noexcept { return cols_[i]; }
};

template<>
struct Matrix<2> {
private:
    array<float2, 2> _cols{};

public:
    constexpr Matrix() noexcept
        : _cols{float2{1.0f, 0.0f}, float2{0.0f, 1.0f}} {}

    constexpr Matrix(const float2 c0, const float2 c1) noexcept
        : _cols{c0, c1} {}

    [[nodiscard]] constexpr float2 &operator[](size_t i) noexcept { return _cols[i]; }
    [[nodiscard]] constexpr const float2 &operator[](size_t i) const noexcept { return _cols[i]; }
};

template<>
struct Matrix<3> {
private:
    array<float3, 3> _cols{};

public:
    constexpr Matrix() noexcept
        : _cols{float3{1.0f, 0.0f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f}} {}

    constexpr Matrix(const float3 c0, const float3 c1, const float3 c2) noexcept
        : _cols{c0, c1, c2} {}

    [[nodiscard]] constexpr float3 &operator[](size_t i) noexcept { return _cols[i]; }
    [[nodiscard]] constexpr const float3 &operator[](size_t i) const noexcept { return _cols[i]; }
};

template<>
struct Matrix<4> {
private:
    array<float4, 4> _cols{};

public:
    constexpr Matrix() noexcept
        : _cols{float4{1.0f, 0.0f, 0.0f, 0.0f},
                float4{0.0f, 1.0f, 0.0f, 0.0f},
                float4{0.0f, 0.0f, 1.0f, 0.0f},
                float4{0.0f, 0.0f, 0.0f, 1.0f}} {}

    constexpr Matrix(const float4 c0, const float4 c1, const float4 c2, const float4 c3) noexcept
        : _cols{c0, c1, c2, c3} {}

    [[nodiscard]] constexpr float4 &operator[](size_t i) noexcept { return _cols[i]; }
    [[nodiscard]] constexpr const float4 &operator[](size_t i) const noexcept { return _cols[i]; }
};

using float2x2 = Matrix<2>;
using float3x3 = Matrix<3>;
using float4x4 = Matrix<4>;

}// namespace ocarina


template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator*(ocarina::Mat<N, M> m, float s) {
    return [&]<size_t ...i>(std::index_sequence<i...>) {
        return ocarina::Mat<N, M>((m[i] * s)...);
    }(std::make_index_sequence<N>());
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator*(float s,ocarina::Mat<N, M> m) {
    return m * s;
}

template<size_t N, size_t M>
[[nodiscard]] constexpr auto operator/(ocarina::Mat<N, M> m, float s) {
    return m * (1.0f / s);
}

[[nodiscard]] constexpr auto operator*(ocarina::float2x2 m, float s) noexcept {
    return ocarina::float2x2{m[0] * s, m[1] * s};
}

[[nodiscard]] constexpr auto operator*(float s, ocarina::float2x2 m) noexcept {
    return m * s;
}

[[nodiscard]] constexpr auto operator/(ocarina::float2x2 m, float s) noexcept {
    return m * (1.0f / s);
}

[[nodiscard]] constexpr auto operator*(ocarina::float2x2 m, ocarina::float2 v) noexcept {
    return v.x * m[0] + v.y * m[1];
}

[[nodiscard]] constexpr auto operator*(ocarina::float2x2 lhs, ocarina::float2x2 rhs) noexcept {
    return ocarina::float2x2{lhs * rhs[0], lhs * rhs[1]};
}

[[nodiscard]] constexpr auto operator+(ocarina::float2x2 lhs, ocarina::float2x2 rhs) noexcept {
    return ocarina::float2x2{lhs[0] + rhs[0], lhs[1] + rhs[1]};
}

[[nodiscard]] constexpr auto operator-(ocarina::float2x2 lhs, ocarina::float2x2 rhs) noexcept {
    return ocarina::float2x2{lhs[0] - rhs[0], lhs[1] - rhs[1]};
}

[[nodiscard]] constexpr auto operator*(ocarina::float3x3 m, float s) noexcept {
    return ocarina::float3x3{m[0] * s, m[1] * s, m[2] * s};
}

[[nodiscard]] constexpr auto operator*(float s, ocarina::float3x3 m) noexcept {
    return m * s;
}

[[nodiscard]] constexpr auto operator/(ocarina::float3x3 m, float s) noexcept {
    return m * (1.0f / s);
}

[[nodiscard]] constexpr auto operator*(ocarina::float3x3 m, ocarina::float3 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2];
}

[[nodiscard]] constexpr auto operator*(ocarina::float3x3 lhs, ocarina::float3x3 rhs) noexcept {
    return ocarina::float3x3{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2]};
}

[[nodiscard]] constexpr auto operator+(ocarina::float3x3 lhs, ocarina::float3x3 rhs) noexcept {
    return ocarina::float3x3{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

[[nodiscard]] constexpr auto operator-(ocarina::float3x3 lhs, ocarina::float3x3 rhs) noexcept {
    return ocarina::float3x3{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

[[nodiscard]] constexpr auto operator*(ocarina::float4x4 m, float s) noexcept {
    return ocarina::float4x4{m[0] * s, m[1] * s, m[2] * s, m[3] * s};
}

[[nodiscard]] constexpr auto operator*(float s, ocarina::float4x4 m) noexcept {
    return m * s;
}

[[nodiscard]] constexpr auto operator/(ocarina::float4x4 m, float s) noexcept {
    return m * (1.0f / s);
}

[[nodiscard]] constexpr auto operator*(ocarina::float4x4 m, ocarina::float4 v) noexcept {
    return v.x * m[0] + v.y * m[1] + v.z * m[2] + v.w * m[3];
}

[[nodiscard]] constexpr auto operator*(ocarina::float4x4 lhs, ocarina::float4x4 rhs) noexcept {
    return ocarina::float4x4{lhs * rhs[0], lhs * rhs[1], lhs * rhs[2], lhs * rhs[3]};
}

[[nodiscard]] constexpr auto operator+(ocarina::float4x4 lhs, ocarina::float4x4 rhs) noexcept {
    return ocarina::float4x4{lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2], lhs[3] + rhs[3]};
}

[[nodiscard]] constexpr auto operator-(ocarina::float4x4 lhs, ocarina::float4x4 rhs) noexcept {
    return ocarina::float4x4{lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2], lhs[3] - rhs[3]};
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