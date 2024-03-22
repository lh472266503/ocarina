
#pragma once

namespace ocarina {
template<typename... Ts>
struct always_false {
    static constexpr bool value = false;
};

template<typename... Ts>
static constexpr bool always_false_v = always_false<Ts...>::value;

template<typename T, size_t N>
struct Vector {
    static_assert(always_false_v<T>, "Invalid vector storage");
};

template<typename T>
struct alignas(sizeof(T) * 2) Vector<T, 2> {
    T x{}, y{};
    __device__ Vector(T s = T{}) noexcept : x{s}, y{s} {}
    __device__ Vector(T x, T y) noexcept : x{x}, y{y} {}
    __device__ T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    __device__ const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
};

template<typename T>
struct alignas(sizeof(T) * 4) Vector<T, 3> {
    T x{}, y{}, z{};
    __device__ Vector(T s = T{}) noexcept : x{s}, y{s}, z{s} {}
    __device__ Vector(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
    __device__ T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    __device__ const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
};

template<typename T>
struct alignas(sizeof(T) * 4) Vector<T, 4> {
    T x{}, y{}, z{}, w{};
    __device__ Vector(T s = T{}) noexcept : x{s}, y{s}, z{s}, w{s} {}
    __device__ Vector(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
    __device__ T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    __device__ const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
};
}// namespace ocarina

using oc_int = int;
using oc_uint = unsigned int;
using oc_float = float;
using oc_bool = bool;
using oc_uchar = unsigned char;
using oc_ushort = unsigned short;
using oc_uint64t = unsigned long long;

#define OC_MAKE_VECTOR_N(type, dim) using type##dim = ocarina::Vector<type, dim>;

#define OC_MAKE_VECTOR(type)  \
    OC_MAKE_VECTOR_N(type, 2) \
    OC_MAKE_VECTOR_N(type, 3) \
    OC_MAKE_VECTOR_N(type, 4)

OC_MAKE_VECTOR(oc_int)
OC_MAKE_VECTOR(oc_uint)
OC_MAKE_VECTOR(oc_float)
OC_MAKE_VECTOR(oc_bool)
OC_MAKE_VECTOR(oc_uchar)
OC_MAKE_VECTOR(oc_ushort)
OC_MAKE_VECTOR(oc_uint64t)

#define OC_MAKE_VECTOR_BINARY_OPERATOR(op, ...)                          \
    template<typename T, typename U, size_t N>                           \
    [[nodiscard]] __device__ constexpr auto                              \
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
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(ocarina::Vector<T, N> lhs, U rhs) noexcept {             \
        return lhs op ocarina::Vector<U, N>{rhs};                        \
    }                                                                    \
    template<typename T, typename U, size_t N>                           \
    [[nodiscard]] __device__ constexpr auto                              \
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

#define OC_MAKE_VECTOR_ASSIGN_OPERATOR(op, ...)                           \
    template<typename T, typename U, size_t N>                            \
    __device__ constexpr decltype(auto) operator op(                      \
        ocarina::Vector<T, N> &lhs, ocarina::Vector<U, N> rhs) noexcept { \
        lhs.x op rhs.x;                                                   \
        lhs.y op rhs.y;                                                   \
        if constexpr (N >= 3) { lhs.z op rhs.z; }                         \
        if constexpr (N == 4) { lhs.w op rhs.w; }                         \
        return (lhs);                                                     \
    }                                                                     \
    template<typename T, typename U, size_t N>                            \
    __device__ constexpr decltype(auto) operator op(                      \
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