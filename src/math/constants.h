//
// Created by Zero on 03/07/2022.
//

#pragma once

#include "core/header.h"
#include "basic_traits.h"

namespace ocarina {
inline namespace constants {
/// pi
constexpr float Pi = 3.14159265358979323846264338327950288f;
/// 2 * pi
constexpr float _2Pi = 2 * Pi;
/// pi/2
constexpr float PiOver2 = 1.57079632679489661923132169163975144f;
/// pi/4
constexpr float PiOver4 = 0.785398163397448309615660845819875721f;
/// 1/pi
constexpr float InvPi = 1.f / Pi;
/// 2/pi
constexpr float _2OverPi = 2.f / Pi;
/// 1 / (4 * pi)
constexpr float Inv4Pi = 1 / (4 * Pi);
/// 1 / (2 * pi)
constexpr float Inv2Pi = 1 / (2 * Pi);
/// sqrt(2)
constexpr float Sqrt2 = 1.41421356237309504880168872420969808f;
/// 1/sqrt(2)
constexpr float InvSqrt2 = 0.707106781186547524400844362104849039f;
/// 1-epsilon
constexpr float OneMinusEpsilon = 0x1.fffffep-1f;
/// tmax for ray
constexpr float RayTMax = 1e16f;

constexpr float ShadowEpsilon = 0.0001f;

constexpr uint32_t InvalidUI32 = uint32_t(-1);

constexpr uint64_t InvalidUI64 = uint64_t(-1);

constexpr uchar InvalidUI8 = uchar(-1);
}// namespace constants

inline namespace constants {

constexpr float shadow_epsilon = 0.0001f;

struct NegInfTy {
    operator double() const { return -std::numeric_limits<double>::infinity(); }
    operator float() const { return -std::numeric_limits<float>::infinity(); }
    operator long long() const { return std::numeric_limits<long long>::min(); }
    operator unsigned long long() const { return std::numeric_limits<unsigned long long>::min(); }
    operator long() const { return std::numeric_limits<long>::min(); }
    operator unsigned long() const { return std::numeric_limits<unsigned long>::min(); }
    operator int() const { return std::numeric_limits<int>::min(); }
    operator unsigned int() const { return std::numeric_limits<unsigned int>::min(); }
    operator short() const { return std::numeric_limits<short>::min(); }
    operator unsigned short() const { return std::numeric_limits<unsigned short>::min(); }
    operator char() const { return std::numeric_limits<char>::min(); }
    operator unsigned char() const { return std::numeric_limits<unsigned char>::min(); }
};

struct PosInfTy {
    operator double() const { return std::numeric_limits<double>::infinity(); }
    operator float() const { return std::numeric_limits<float>::infinity(); }
    operator long long() const { return std::numeric_limits<long long>::max(); }
    operator unsigned long long() const { return std::numeric_limits<unsigned long long>::max(); }
    operator long() const { return std::numeric_limits<long>::max(); }
    operator unsigned long() const { return std::numeric_limits<unsigned long>::max(); }
    operator int() const { return std::numeric_limits<int>::max(); }
    operator unsigned int() const { return std::numeric_limits<unsigned int>::max(); }
    operator short() const { return std::numeric_limits<short>::max(); }
    operator unsigned short() const { return std::numeric_limits<unsigned short>::max(); }
    operator char() const { return std::numeric_limits<char>::max(); }
    operator unsigned char() const { return std::numeric_limits<unsigned char>::max(); }
};

struct NaNTy {
    operator double() const { return std::numeric_limits<double>::quiet_NaN(); }
    operator float() const { return std::numeric_limits<float>::quiet_NaN(); }
};

struct UlpTy {
    operator double() const { return std::numeric_limits<double>::epsilon(); }
    operator float() const { return std::numeric_limits<float>::epsilon(); }
};

template<typename T>
[[nodiscard]] T empty_range_lower() {
    return (T)NegInfTy();
}

template<typename T>
[[nodiscard]] T empty_range_upper() {
    return (T)PosInfTy();
}

}// namespace constants

}// namespace ocarina