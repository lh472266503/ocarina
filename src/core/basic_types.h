//
// Created by Zero on 25/04/2022.
//

#pragma once

#include "basic_traits.h"
#include "constants.h"

namespace ocarina {

namespace detail {

template<typename T>
struct valid_vector_impl : public std::disjunction<
                               std::is_same<T, bool>,
                               std::is_same<T, float>,
                               std::is_same<T, int>,
                               std::is_same<T, char>,
                               std::is_same<T, short>,
                               std::is_same<T, ushort>,
                               std::is_same<T, uint64t>,
                               std::is_same<T, uchar>,
                               std::is_same<T, uint>> {};

};// namespace detail

template<typename T>
static constexpr auto valid_vector_v = detail::valid_vector_impl<std::remove_cvref_t<T>>::value;

template<typename T, size_t N>
struct Vector {
    static_assert(always_false_v<T>, "Invalid vector storage");
};

}// namespace ocarina

namespace ocarina {

template<typename T>
class Var;

namespace detail {

template<typename T, size_t N, size_t... Indices>
struct swizzle_impl;

template<typename T>
struct is_swizzle : std::false_type {};

template<typename T, size_t N, size_t... Indices>
struct is_swizzle<swizzle_impl<T, N, Indices...>> : std::true_type {};

OC_DEFINE_TEMPLATE_VALUE(is_swizzle)

template<typename T, size_t N, size_t... Indices>
struct swizzle_impl {
    static constexpr uint num_component = sizeof...(Indices);
    static_assert(num_component <= 4 && std::max({Indices...}) < N);
    using vec_type = ocarina::Vector<T, num_component>;

private:
    template<size_t... index>
    void assign_to(vec_type &vec, std::index_sequence<index...>) const noexcept {
        ((vec[index] = data_[Indices]), ...);
    }

    template<typename U, size_t... index>
    void assign_from(const ocarina::Vector<U, num_component> &vec, std::index_sequence<index...>) noexcept {
        ((data_[Indices] = vec[index]), ...);
    }

public:
    ocarina::array<T, N> data_{};

public:
    template<typename U>
    swizzle_impl &operator=(const ocarina::Vector<U, num_component> &vec) noexcept {
        assign_from(vec, std::make_index_sequence<num_component>());
        return *this;
    }

    template<typename U, size_t... OtherIndices>
    swizzle_impl &operator=(const swizzle_impl<U, N, OtherIndices...> &other) {
        ((data_[Indices] = other.data_[OtherIndices]), ...);
        return *this;
    }

    [[nodiscard]] vec_type to_vec() const noexcept {
        vec_type ret;
        assign_to(ret, std::make_index_sequence<num_component>());
        return ret;
    }

    operator vec_type() const noexcept {
        return to_vec();
    }

#define OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(op)            \
                                                        \
    template<typename Arg>                              \
    vec_type operator op(Arg &&val) const noexcept {    \
        if constexpr (is_swizzle_v<Arg>) {              \
            return to_vec() op val.to_vec();            \
        } else {                                        \
            return to_vec() op OC_FORWARD(val);         \
        }                                               \
    }                                                   \
                                                        \
    template<typename Arg>                              \
    swizzle_impl &operator op##=(Arg && arg) noexcept { \
        auto tmp = *this;                               \
        *this = tmp op OC_FORWARD(arg);                 \
        return *this;                                   \
    }

    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(+)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(-)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(*)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(/)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(%)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(>>)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(<<)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(|)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(&)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(^)
#undef OC_MAKE_SWIZZLE_MEMBER_BINARY_OP
};
}// namespace detail
}// namespace ocarina

namespace ocarina {

template<typename T>
struct alignas(sizeof(T) * 2) Vector<T, 2> {
    static_assert(valid_vector_v<T>, "Invalid vector type");

public:
    template<size_t... index>
    using swizzle_type = detail::swizzle_impl<T, 2, index...>;

public:
    union {
        struct {
            T x, y;
        };
#include "swizzle_inl/swizzle2.inl.h"
    };
    Vector() : x{}, y{} {}
    explicit constexpr Vector(T s) noexcept : x{s}, y{s} {}
    template<typename U>
    explicit constexpr Vector(U s) noexcept : Vector(static_cast<T>(s)) {}
    constexpr Vector(T x, T y) noexcept : x{x}, y{y} {}
    [[nodiscard]] constexpr T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    [[nodiscard]] constexpr const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
#include "swizzle_inl/swizzle_2.inl.h"
};

template<typename T>
struct alignas(sizeof(T) * 4) Vector<T, 3> {
    static_assert(valid_vector_v<T>, "Invalid vector type");

public:
    template<size_t... index>
    using swizzle_type = detail::swizzle_impl<T, 3, index...>;

public:
    union {
        struct {
            T x, y, z;
        };
#include "swizzle_inl/swizzle3.inl.h"
    };
    Vector() : x{}, y{}, z{} {}
    explicit constexpr Vector(T s) noexcept : x{s}, y{s}, z{s} {}
    template<typename U>
    explicit constexpr Vector(U s) noexcept : Vector(static_cast<T>(s)) {}
    constexpr Vector(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
    [[nodiscard]] constexpr T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    [[nodiscard]] constexpr const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
#include "swizzle_inl/swizzle_3.inl.h"
};

template<typename T>
struct alignas(sizeof(T) * 4) Vector<T, 4> {
    static_assert(valid_vector_v<T>, "Invalid vector type");

public:
    template<size_t... index>
    using swizzle_type = detail::swizzle_impl<T, 4, index...>;

public:
    union {
        struct {
            T x, y, z, w;
        };
#include "swizzle_inl/swizzle4.inl.h"
    };
    Vector() : x{}, y{}, z{}, w{} {}
    explicit constexpr Vector(T s) noexcept : x{s}, y{s}, z{s}, w{s} {}
    template<typename U>
    explicit constexpr Vector(U s) noexcept : Vector(static_cast<T>(s)) {}
    constexpr Vector(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
    [[nodiscard]] constexpr T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    [[nodiscard]] constexpr const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
#include "swizzle_inl/swizzle_4.inl.h"
};

#define OC_MAKE_VECTOR_TYPES(T) \
    using T##2 = Vector<T, 2>;  \
    using T##3 = Vector<T, 3>;  \
    using T##4 = Vector<T, 4>;

OC_MAKE_VECTOR_TYPES(bool)
OC_MAKE_VECTOR_TYPES(float)
OC_MAKE_VECTOR_TYPES(int)
OC_MAKE_VECTOR_TYPES(char)
OC_MAKE_VECTOR_TYPES(short)
OC_MAKE_VECTOR_TYPES(ushort)
OC_MAKE_VECTOR_TYPES(uchar)
OC_MAKE_VECTOR_TYPES(uint)
OC_MAKE_VECTOR_TYPES(uint64t)

#undef OC_MAKE_VECTOR_TYPES

template<size_t N>
struct Matrix {
    static_assert(always_false_v<std::integral_constant<size_t, N>>, "Invalid matrix type");
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

using basic_types = ocarina::tuple<
    bool, float, int, uint, uint64t,
    bool2, float2, int2, uint2, uint64t2,
    bool3, float3, int3, uint3, uint64t3,
    bool4, float4, int4, uint4, uint64t4,
    float2x2, float3x3, float4x4>;

namespace detail {
template<typename T>
struct tuple_to_variant_impl {
    static_assert(always_false_v<T>);
};

template<typename... Ts>
struct tuple_to_variant_impl<ocarina::tuple<Ts...>> {
    using type = ocarina::variant<Ts...>;
};
}// namespace detail

template<typename T>
using tuple_to_variant_t = typename detail::tuple_to_variant_impl<T>::type;

using basic_variant_t = tuple_to_variant_t<basic_types>;

namespace detail {
using texture_elements = ocarina::tuple<uchar, uchar2, uchar4, float, float2, float4>;
template<typename T, typename... Ts>
[[nodiscard]] constexpr bool is_contain(const ocarina::tuple<Ts...> *tp) noexcept {
    return std::disjunction_v<std::is_same<T, Ts>...>;
}

template<typename T>
[[nodiscard]] constexpr bool is_valid_texture_element_impl() noexcept {
    return is_contain<T>(static_cast<texture_elements *>(nullptr));
}

}// namespace detail

template<typename T>
[[nodiscard]] constexpr bool is_valid_texture_element() noexcept {
    return detail::is_valid_texture_element_impl<std::remove_cvref_t<T>>();
}

namespace detail {
template<typename T>
requires(is_valid_texture_element<T>())
struct texture_sample_impl {
    using type = float;
};

template<>
struct texture_sample_impl<float2> {
    using type = float2;
};

template<>
struct texture_sample_impl<uchar2> : public texture_sample_impl<float2> {};

template<>
struct texture_sample_impl<float4> {
    using type = float4;
};

template<>
struct texture_sample_impl<uchar4> : public texture_sample_impl<float4> {};

};// namespace detail

template<typename element_type>
using texture_sample_t = typename detail::texture_sample_impl<std::remove_cvref_t<element_type>>::type;

[[nodiscard]] constexpr bool any(const bool2 v) noexcept { return v.x || v.y; }
[[nodiscard]] constexpr bool any(const bool3 v) noexcept { return v.x || v.y || v.z; }
[[nodiscard]] constexpr bool any(const bool4 v) noexcept { return v.x || v.y || v.z || v.w; }

[[nodiscard]] constexpr bool all(const bool2 v) noexcept { return v.x && v.y; }
[[nodiscard]] constexpr bool all(const bool3 v) noexcept { return v.x && v.y && v.z; }
[[nodiscard]] constexpr bool all(const bool4 v) noexcept { return v.x && v.y && v.z && v.w; }

[[nodiscard]] constexpr bool none(const bool2 v) noexcept { return !any(v); }
[[nodiscard]] constexpr bool none(const bool3 v) noexcept { return !any(v); }
[[nodiscard]] constexpr bool none(const bool4 v) noexcept { return !any(v); }

}// namespace ocarina

template<typename T, size_t N>
requires ocarina::is_number_v<T>
[[nodiscard]] constexpr auto
operator+(const ocarina::Vector<T, N> v) noexcept {
    return v;
}

template<typename T, size_t N>
requires ocarina::is_number_v<T>
[[nodiscard]] constexpr auto
operator-(const ocarina::Vector<T, N> v) noexcept {
    using R = ocarina::Vector<T, N>;
    if constexpr (N == 2) {
        return R{-v.x, -v.y};
    } else if constexpr (N == 3) {
        return R{-v.x, -v.y, -v.z};
    } else {
        return R{-v.x, -v.y, -v.z, -v.w};
    }
}

template<typename T, size_t N>
[[nodiscard]] constexpr auto operator!(const ocarina::Vector<T, N> v) noexcept {
    if constexpr (N == 2u) {
        return ocarina::bool2{!v.x, !v.y};
    } else if constexpr (N == 3u) {
        return ocarina::bool3{!v.x, !v.y, !v.z};
    } else {
        return ocarina::bool3{!v.x, !v.y, !v.z, !v.w};
    }
}

template<typename T, size_t N>
requires ocarina::is_integral_v<T>
[[nodiscard]] constexpr auto
operator~(const ocarina::Vector<T, N> v) noexcept {
    using R = ocarina::Vector<T, N>;
    if constexpr (N == 2) {
        return R{~v.x, ~v.y};
    } else if constexpr (N == 3) {
        return R{~v.x, ~v.y, ~v.z};
    } else {
        return R{~v.x, ~v.y, ~v.z, ~v.w};
    }
}

#define OC_MAKE_VECTOR_BINARY_OPERATOR(op, ...)                          \
    template<typename T, typename U, size_t N>                           \
    requires requires { T{} op U{}; } && __VA_ARGS__                     \
    [[nodiscard]] constexpr auto                                         \
    operator op(                                                         \
        ocarina::Vector<T, N> lhs, ocarina::Vector<U, N> rhs) noexcept { \
        using ret_type = decltype(T{} + U{});                            \
        if constexpr (N == 2) {                                          \
            return ocarina::Vector<ret_type, 2>{                         \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y};                                         \
        } else if constexpr (N == 3) {                                   \
            return ocarina::Vector<ret_type, 3>{                         \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z};                                         \
        } else {                                                         \
            return ocarina::Vector<ret_type, 4>{                         \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z,                                          \
                lhs.w op rhs.w};                                         \
        }                                                                \
    }                                                                    \
    template<typename T, typename U, size_t N>                           \
    requires requires { T{} op U{}; } && __VA_ARGS__                     \
    [[nodiscard]] constexpr auto                                         \
    operator op(ocarina::Vector<T, N> lhs, U rhs) noexcept {             \
        return lhs op ocarina::Vector<U, N>{rhs};                        \
    }                                                                    \
    template<typename T, typename U, size_t N>                           \
    requires requires { T{} op U{}; } && __VA_ARGS__                     \
    [[nodiscard]] constexpr auto                                         \
    operator op(T lhs, ocarina::Vector<U, N> rhs) noexcept {             \
        return ocarina::Vector<T, N>{lhs} op rhs;                        \
    }
OC_MAKE_VECTOR_BINARY_OPERATOR(+, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(-, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(*, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(/, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(%, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(>>, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(<<, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(|, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(&, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(^, ocarina::is_all_integral_v<T, U>)

#undef OC_MAKE_VECTOR_BINARY_OPERATOR

#define OC_MAKE_VECTOR_ASSIGN_OPERATOR(op, ...)                           \
    template<typename T, typename U, size_t N>                            \
    requires __VA_ARGS__                                                  \
    constexpr decltype(auto) operator op(                                 \
        ocarina::Vector<T, N> &lhs, ocarina::Vector<U, N> rhs) noexcept { \
        lhs.x op rhs.x;                                                   \
        lhs.y op rhs.y;                                                   \
        if constexpr (N >= 3) { lhs.z op rhs.z; }                         \
        if constexpr (N == 4) { lhs.w op rhs.w; }                         \
        return (lhs);                                                     \
    }                                                                     \
    template<typename T, typename U, size_t N>                            \
    requires __VA_ARGS__                                                  \
    constexpr decltype(auto) operator op(                                 \
        ocarina::Vector<T, N> &lhs, U rhs) noexcept {                     \
        return (lhs op ocarina::Vector<U, N>{rhs});                       \
    }

OC_MAKE_VECTOR_ASSIGN_OPERATOR(+=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(-=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(*=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(/=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(%=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(<<=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(>>=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(|=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(&=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(^=, ocarina::is_all_integral_v<T, U>)

#undef OC_MAKE_VECTOR_ASSIGN_OPERATOR

#define OC_MAKE_VECTOR_LOGIC_OPERATOR(op, ...)                           \
    template<typename T, typename U, size_t N>                           \
    requires requires { T{} op U{}; } && __VA_ARGS__                     \
    [[nodiscard]] constexpr auto                                         \
    operator op(                                                         \
        ocarina::Vector<T, N> lhs, ocarina::Vector<U, N> rhs) noexcept { \
        if constexpr (N == 2) {                                          \
            return ocarina::bool2{                                       \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y};                                         \
        } else if constexpr (N == 3) {                                   \
            return ocarina::bool3{                                       \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z};                                         \
        } else {                                                         \
            return ocarina::bool4{                                       \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z,                                          \
                lhs.w op rhs.w};                                         \
        }                                                                \
    }                                                                    \
    template<typename T, typename U, size_t N>                           \
    requires requires { T{} op U{}; } && __VA_ARGS__                     \
    [[nodiscard]] constexpr auto                                         \
    operator op(ocarina::Vector<T, N> lhs, U rhs) noexcept {             \
        return lhs op ocarina::Vector<U, N>{rhs};                        \
    }                                                                    \
    template<typename T, typename U, size_t N>                           \
    requires requires { T{} op U{}; } && __VA_ARGS__                     \
    [[nodiscard]] constexpr auto                                         \
    operator op(T lhs, ocarina::Vector<U, N> rhs) noexcept {             \
        return ocarina::Vector<T, N>{lhs} op rhs;                        \
    }
OC_MAKE_VECTOR_LOGIC_OPERATOR(||, ocarina::is_all_boolean_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(&&, ocarina::is_all_boolean_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(==, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(!=, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(<, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(>, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(<=, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(>=, ocarina::is_all_number_v<T>)

#undef OC_MAKE_VECTOR_LOGIC_OPERATOR

#define OC_MAKE_SWIZZLE_BINARY_OP(op)                                                           \
    template<typename Lhs, typename U, size_t N, size_t... Indices>                             \
    auto operator op(Lhs &&lhs, ocarina::detail::swizzle_impl<U, N, Indices...> rhs) noexcept { \
        return OC_FORWARD(lhs) op rhs.to_vec();                                                 \
    }                                                                                           \
                                                                                                \
    template<typename Lhs, typename U, size_t N, size_t... Indices>                             \
    auto operator op##=(Lhs &lhs,                                                               \
                        ocarina::detail::swizzle_impl<U, N, Indices...> rhs) noexcept {         \
        lhs op## = rhs.to_vec();                                                                \
        return lhs;                                                                             \
    }

OC_MAKE_SWIZZLE_BINARY_OP(+)
OC_MAKE_SWIZZLE_BINARY_OP(-)
OC_MAKE_SWIZZLE_BINARY_OP(*)
OC_MAKE_SWIZZLE_BINARY_OP(/)
OC_MAKE_SWIZZLE_BINARY_OP(%)
OC_MAKE_SWIZZLE_BINARY_OP(>>)
OC_MAKE_SWIZZLE_BINARY_OP(<<)
OC_MAKE_SWIZZLE_BINARY_OP(|)
OC_MAKE_SWIZZLE_BINARY_OP(&)
OC_MAKE_SWIZZLE_BINARY_OP(^)

#undef OC_MAKE_SWIZZLE_BINARY_OP

#define OC_MAKE_SWIZZLE_UNARY_OP(op)                                                 \
    template<typename T, size_t N, size_t... Indices>                                \
    auto operator op(ocarina::detail::swizzle_impl<T, N, Indices...> val) noexcept { \
        return op val.to_vec();                                                      \
    }
OC_MAKE_SWIZZLE_UNARY_OP(+)
OC_MAKE_SWIZZLE_UNARY_OP(-)
OC_MAKE_SWIZZLE_UNARY_OP(!)
OC_MAKE_SWIZZLE_UNARY_OP(~)

#undef OC_MAKE_SWIZZLE_UNARY_OP

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
#define OC_MAKE_TYPE_N(type)                                                                                                 \
    [[nodiscard]] constexpr auto make_##type##2(type s = {}) noexcept { return type##2(s); }                                 \
    [[nodiscard]] constexpr auto make_##type##2(type x, type y) noexcept { return type##2(x, y); }                           \
    template<typename T, size_t N>                                                                                           \
    requires(N >= 2)                                                                                                         \
    [[nodiscard]] constexpr auto make_##type##2(Vector<T, N> v) noexcept {                                                   \
        return type##2(                                                                                                      \
            static_cast<type>(v.x),                                                                                          \
            static_cast<type>(v.y));                                                                                         \
    }                                                                                                                        \
    template<typename T, size_t N>                                                                                           \
    requires(N >= 2)                                                                                                         \
    [[nodiscard]] constexpr auto make_##type##2(ocarina::array<T, N> v) noexcept {                                           \
        return type##2(                                                                                                      \
            static_cast<type>(v[0]),                                                                                         \
            static_cast<type>(v[1]));                                                                                        \
    }                                                                                                                        \
    [[nodiscard]] constexpr auto make_##type##2(type##3 v) noexcept { return type##2(v.x, v.y); }                            \
    [[nodiscard]] constexpr auto make_##type##2(type##4 v) noexcept { return type##2(v.x, v.y); }                            \
                                                                                                                             \
    [[nodiscard]] constexpr auto make_##type##3(type s = {}) noexcept { return type##3(s); }                                 \
    [[nodiscard]] constexpr auto make_##type##3(type x, type y, type z) noexcept { return type##3(x, y, z); }                \
    template<typename T, size_t N>                                                                                           \
    requires(N >= 3)                                                                                                         \
    [[nodiscard]] constexpr auto make_##type##3(Vector<T, N> v) noexcept {                                                   \
        return type##3(                                                                                                      \
            static_cast<type>(v.x),                                                                                          \
            static_cast<type>(v.y),                                                                                          \
            static_cast<type>(v.z));                                                                                         \
    }                                                                                                                        \
    template<typename T, size_t N>                                                                                           \
    requires(N >= 3)                                                                                                         \
    [[nodiscard]] constexpr auto make_##type##3(ocarina::array<T, N> v) noexcept {                                           \
        return type##3(                                                                                                      \
            static_cast<type>(v[0]),                                                                                         \
            static_cast<type>(v[1]),                                                                                         \
            static_cast<type>(v[2]));                                                                                        \
    }                                                                                                                        \
    [[nodiscard]] constexpr auto make_##type##3(type##2 v, type z) noexcept { return type##3(v.x, v.y, z); }                 \
    [[nodiscard]] constexpr auto make_##type##3(type x, type##2 v) noexcept { return type##3(x, v.x, v.y); }                 \
    [[nodiscard]] constexpr auto make_##type##3(type##4 v) noexcept { return type##3(v.x, v.y, v.z); }                       \
                                                                                                                             \
    [[nodiscard]] constexpr auto make_##type##4(type s = {}) noexcept { return type##4(s); }                                 \
    [[nodiscard]] constexpr auto make_##type##4(type x, type y, type z, type w) noexcept { return type##4(x, y, z, w); }     \
    template<typename T, size_t N>                                                                                           \
    requires(N == 4)                                                                                                         \
    [[nodiscard]] constexpr auto make_##type##4(Vector<T, N> v) noexcept {                                                   \
        return type##4(                                                                                                      \
            static_cast<type>(v.x),                                                                                          \
            static_cast<type>(v.y),                                                                                          \
            static_cast<type>(v.z),                                                                                          \
            static_cast<type>(v.w));                                                                                         \
    }                                                                                                                        \
    template<typename T, size_t N>                                                                                           \
    requires(N == 4)                                                                                                         \
    [[nodiscard]] constexpr auto make_##type##4(ocarina::array<T, N> v) noexcept {                                           \
        return type##4(                                                                                                      \
            static_cast<type>(v[0]),                                                                                         \
            static_cast<type>(v[1]),                                                                                         \
            static_cast<type>(v[2]),                                                                                         \
            static_cast<type>(v[3]));                                                                                        \
    }                                                                                                                        \
    [[nodiscard]] constexpr auto make_##type##4(type##2 v, type z, type w) noexcept { return type##4(v.x, v.y, z, w); }      \
    [[nodiscard]] constexpr auto make_##type##4(type x, type##2 v, type w) noexcept { return type##4(x, v.x, v.y, w); }      \
    [[nodiscard]] constexpr auto make_##type##4(type x, type y, type##2 v) noexcept { return type##4(x, y, v.x, v.y); }      \
    [[nodiscard]] constexpr auto make_##type##4(type##2 xy, type##2 zw) noexcept { return type##4(xy.x, xy.y, zw.x, zw.y); } \
    [[nodiscard]] constexpr auto make_##type##4(type##3 v, type w) noexcept { return type##4(v.x, v.y, v.z, w); }            \
    [[nodiscard]] constexpr auto make_##type##4(type x, type##3 v) noexcept { return type##4(x, v.x, v.y, v.z); }

OC_MAKE_TYPE_N(bool)
OC_MAKE_TYPE_N(float)
OC_MAKE_TYPE_N(int)
OC_MAKE_TYPE_N(uint)
OC_MAKE_TYPE_N(uchar)
OC_MAKE_TYPE_N(char)
#undef OC_MAKE_TYPE_N

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

namespace detail {
template<typename T>
struct literal_value {
    static_assert(always_false_v<T>);
};

template<typename... T>
struct literal_value<ocarina::tuple<T...>> {
    using type = ocarina::variant<T...>;
};
}// namespace detail

template<typename T>
using literal_value_t = typename detail::literal_value<T>::type;

using basic_literal_t = literal_value_t<basic_types>;

}// namespace ocarina