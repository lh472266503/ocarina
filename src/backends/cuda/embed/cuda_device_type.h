
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
    __device__ explicit Vector(T s = T{}) noexcept : x{s}, y{s} {}
    __device__ Vector(T x, T y) noexcept : x{x}, y{y} {}
    __device__ T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    __device__ const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
};

template<typename T>
struct alignas(sizeof(T) * 4) Vector<T, 3> {
    T x{}, y{}, z{};
    __device__ explicit Vector(T s = T{}) noexcept : x{s}, y{s}, z{s} {}
    __device__ Vector(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
    __device__ T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    __device__ const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
};

template<typename T>
struct alignas(sizeof(T) * 4) Vector<T, 4> {
    T x{}, y{}, z{}, w{};
    __device__ explicit Vector(T s = T{}) noexcept : x{s}, y{s}, z{s}, w{s} {}
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

#define OC_MAKE_VECTOR_N(type, dim) using oc_##type##dim = ocarina::Vector<type, dim>;

#define OC_MAKE_VECTOR(type)  \
    OC_MAKE_VECTOR_N(type, 2) \
    OC_MAKE_VECTOR_N(type, 3) \
    OC_MAKE_VECTOR_N(type, 4)

// OC_MAKE_VECTOR(oc_int)
// OC_MAKE_VECTOR(oc_uint)
// OC_MAKE_VECTOR(oc_float)
// OC_MAKE_VECTOR(oc_bool)
// OC_MAKE_VECTOR(oc_uchar)
// OC_MAKE_VECTOR(oc_ushort)
// OC_MAKE_VECTOR(oc_uint64t)