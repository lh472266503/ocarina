//
// Created by Zero on 03/07/2022.
//

#pragma once

#include "core/header.h"

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
}
}// namespace ocarina::constants