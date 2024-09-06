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
    template<typename... Args>
    requires(sizeof...(Args) == N)
    explicit constexpr Matrix(Args &&...args) noexcept
        : cols_(array_t{static_cast<vector_type>(OC_FORWARD(args))...}) {}

    template<typename... Args>
    requires(sizeof...(Args) == ElementNum && is_all_scalar_v<Args...>)
    explicit constexpr Matrix(Args &&...args) noexcept
        : cols_([&]<size_t... i>(std::index_sequence<i...>,
                                 const array<scalar_type, ElementNum> &arr) {
              return array_t{vector_type{addressof(arr.data()[i * M])}...};
          }(std::make_index_sequence<N>(), array<scalar_type, ElementNum>{static_cast<scalar_type>(OC_FORWARD(args))...})) {
    }

    template<size_t NN, size_t MM>
    requires(NN >= N && MM >= M)
    explicit constexpr Matrix(Matrix<NN, MM> mat) noexcept
        : cols_{[&]<size_t... i>(std::index_sequence<i...>) {
              return std::array<Vector<float, M>, N>{Vector<float, M>{mat[i]}...};
          }(std::make_index_sequence<N>())} {}

    constexpr Matrix(scalar_type s = 1) noexcept
        : cols_([&]<size_t... i>(std::index_sequence<i...>) {
              array_t ret{};
              if constexpr (M == N) {
                  ((ret[i][i] = s), ...);
              }
              return ret;
          }(std::make_index_sequence<N>())) {
    }

    [[nodiscard]] constexpr vector_type &operator[](size_t i) noexcept { return cols_[i]; }
    [[nodiscard]] constexpr const vector_type &operator[](size_t i) const noexcept { return cols_[i]; }
};

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

template<size_t N, size_t M, typename... Args>
requires is_all_basic_v<Args...>
[[nodiscard]] constexpr Matrix<N, M> make_float(Args &&...args) noexcept {
    return Matrix<N, M>(OC_FORWARD(args)...);
}

#define OC_MAKE_MATRIX_(N, M)                                                        \
    using float##N##x##M = Matrix<N, M>;                                             \
    template<typename... Args>                                                       \
    requires is_all_basic_v<Args...>                                                 \
    [[nodiscard]] constexpr float##N##x##M make_float##N##x##M(Args &&...args) {     \
        return make_float<N, M>(OC_FORWARD(args)...);                                \
    }                                                                                \
    template<size_t NN, size_t MM>                                                   \
    [[nodiscard]] constexpr float##N##x##M make_float##N##x##M(Matrix<NN, MM> mat) { \
        return float##N##x##M(mat);                                                  \
    }

OC_MAKE_MATRIX_(2, 2)
OC_MAKE_MATRIX_(2, 3)
OC_MAKE_MATRIX_(2, 4)
OC_MAKE_MATRIX_(3, 2)
OC_MAKE_MATRIX_(3, 3)
OC_MAKE_MATRIX_(3, 4)
OC_MAKE_MATRIX_(4, 2)
OC_MAKE_MATRIX_(4, 3)
OC_MAKE_MATRIX_(4, 4)

template<size_t N, size_t M>
[[nodiscard]] constexpr Matrix<M, N> transpose(const Matrix<N, M> &mat) noexcept {
    Matrix<M, N> ret = make_float<M, N>();
    auto func_n = [&]<size_t ...n>(size_t i, std::index_sequence<n...>) {
        return Vector<float, N>((mat[n][i])...);
    };
    auto func_m = [&]<size_t ...m>(std::index_sequence<m...>) {
        return Matrix<M, N>(func_n(m,std::make_index_sequence<N>())...);
    };
    return func_m(std::make_index_sequence<M>());
}

[[nodiscard]] constexpr auto make_float3x3(float2x2 m) noexcept {
    return float3x3{make_float3(m[0], 0.0f),
                    make_float3(m[1], 0.0f),
                    make_float3(0.f, 0.f, 1.0f)};
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

}// namespace ocarina