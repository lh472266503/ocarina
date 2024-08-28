
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

using uint = unsigned int;
using uint64t = unsigned long long;
using uchar = unsigned char;
using ushort = unsigned short;

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

}// namespace ocarina

using oc_int = int;
using oc_uint = unsigned int;
using oc_float = float;
using oc_bool = bool;
using oc_uchar = unsigned char;
using oc_ushort = unsigned short;
using oc_uint64t = unsigned long long;

namespace ocarina {

template<size_t... Ints>
struct index_sequence {};

template<size_t N, size_t... Ints>
struct make_index_sequence_helper : make_index_sequence_helper<N - 1, N - 1, Ints...> {
};

template<size_t... Ints>
struct make_index_sequence_helper<0, Ints...> {
    using type = index_sequence<Ints...>;
};

template<size_t N>
using make_index_sequence = typename make_index_sequence_helper<N>::type;

template<bool B, typename T = void>
struct enable_if {};

template<typename T>
struct enable_if<true, T> {
    using type = T;
};

template<bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

}// namespace ocarina

#define OC_MAKE_VECTOR_N(type, dim) using type##dim = ocarina::Vector<type, dim>;

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto
operator+(const ocarina::Vector<T, N> v) noexcept {
    return v;
}

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto
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
[[nodiscard]] __device__ constexpr auto operator!(const ocarina::Vector<T, N> v) noexcept {
    if constexpr (N == 2u) {
        return ocarina::Vector<bool, 2>{!v.x, !v.y};
    } else if constexpr (N == 3u) {
        return ocarina::Vector<bool, 3>{!v.x, !v.y, !v.z};
    } else {
        return ocarina::Vector<bool, 3>{!v.x, !v.y, !v.z, !v.w};
    }
}

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto
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

#define OC_MAKE_VECTOR_LOGIC_OPERATOR(op, ...)                           \
    template<typename T, size_t N>                                       \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(                                                         \
        ocarina::Vector<T, N> lhs, ocarina::Vector<T, N> rhs) noexcept { \
        if constexpr (N == 2) {                                          \
            return ocarina::Vector<bool, 2>{                             \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y};                                         \
        } else if constexpr (N == 3) {                                   \
            return ocarina::Vector<bool, 3>{                             \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z};                                         \
        } else {                                                         \
            return ocarina::Vector<bool, 4>{                             \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z,                                          \
                lhs.w op rhs.w};                                         \
        }                                                                \
    }                                                                    \
    template<typename T, size_t N>                                       \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(ocarina::Vector<T, N> lhs, T rhs) noexcept {             \
        return lhs op ocarina::Vector<T, N>{rhs};                        \
    }                                                                    \
    template<typename T, size_t N>                                       \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(T lhs, ocarina::Vector<T, N> rhs) noexcept {             \
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

template<typename T, oc_uint N>
class oc_array {
private:
    T _data[N];

public:
    template<typename... Elem>
    __device__ constexpr oc_array(Elem... elem) noexcept : _data{elem...} {}
    __device__ constexpr oc_array(oc_array &&) noexcept = default;
    __device__ constexpr oc_array(const oc_array &) noexcept = default;
    __device__ constexpr oc_array &operator=(oc_array &&) noexcept = default;
    __device__ constexpr oc_array &operator=(const oc_array &) noexcept = default;
    [[nodiscard]] __device__ T &operator[](size_t i) noexcept { return _data[i]; }
    [[nodiscard]] __device__ T operator[](size_t i) const noexcept { return _data[i]; }
};

namespace ocarina {

template<size_t N, size_t M = N>
struct Matrix {
public:
    static constexpr auto RowNum = M;
    static constexpr auto ColNum = N;
    using scalar_type = float;
    using vector_type = Vector<scalar_type, M>;

private:
    oc_array<vector_type, N> cols_{};

public:
    Matrix() = default;
    template<typename... Args, typename = enable_if_t<sizeof...(Args) == N>>
    constexpr Matrix(Args... args) noexcept : cols_(oc_array<vector_type, N>{args...}) {}
    [[nodiscard]] constexpr vector_type &operator[](size_t i) noexcept { return cols_[i]; }
    [[nodiscard]] constexpr const vector_type &operator[](size_t i) const noexcept { return cols_[i]; }
};

using float2x2 = Matrix<2>;
using float3x3 = Matrix<3>;
using float4x4 = Matrix<4>;

}// namespace ocarina
