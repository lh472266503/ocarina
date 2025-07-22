//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "math/basic_types.h"

namespace ocarina {

enum struct UnaryOp : uint32_t {
    POSITIVE,
    NEGATIVE,
    NOT,
    BIT_NOT
};

enum struct CastOp : uint32_t {
    STATIC,
    BITWISE
};

enum struct BinaryOp : uint32_t {
    // arithmetic
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    SHL,
    SHR,
    AND,
    OR,

    // relational
    LESS,
    GREATER,
    LESS_EQUAL,
    GREATER_EQUAL,
    EQUAL,
    NOT_EQUAL
};

enum struct CallOp : uint32_t {
    CUSTOM,

    ALL,
    ANY,
    NONE,

    SELECT,
    CLAMP,
    LERP,
    INVERSE_LERP,

    ABS,
    MIN,
    MAX,

    IS_INF,
    IS_NAN,

    ACOS,
    ASIN,
    ATAN,
    ACOSH,
    ASINH,
    ATANH,
    ATAN2,
    COPYSIGN,

    COS,
    SIN,
    TAN,
    COSH,
    SINH,
    TANH,

    EXP,
    EXP2,
    EXP10,
    LOG,
    LOG2,
    LOG10,
    POW,
    FMOD,
    MOD,
    FRACT,

    SQRT,
    RSQRT,
    SQR,
    RCP,
    SIGN,

    CEIL,
    FLOOR,
    ROUND,

    DEGREES,
    RADIANS,
    SATURATE,

    FMA,

    CROSS,
    DOT,
    DISTANCE,
    DISTANCE_SQUARED,
    LENGTH,
    LENGTH_SQUARED,
    NORMALIZE,
    FACE_FORWARD,
    COORDINATE_SYSTEM,
    MAKE_NORMAL_TANGENT,

    DETERMINANT,
    TRANSPOSE,
    INVERSE,

    IS_NULL_BUFFER,
    IS_NULL_TEXTURE,

    BUFFER_SIZE,

    TEX_SAMPLE,
    TEX_READ,
    TEX_WRITE,

    BYTE_BUFFER_READ,
    BYTE_BUFFER_WRITE,
    BYTE_BUFFER_SIZE,

    BINDLESS_ARRAY_BUFFER_READ,
    BINDLESS_ARRAY_BUFFER_WRITE,
    BINDLESS_ARRAY_BUFFER_SIZE,
    BINDLESS_ARRAY_BYTE_BUFFER_READ,
    BINDLESS_ARRAY_BYTE_BUFFER_WRITE,
    BINDLESS_ARRAY_TEX_SAMPLE,

    SYNCHRONIZE_BLOCK,

    MAKE_BOOL2,
    MAKE_BOOL3,
    MAKE_BOOL4,
    MAKE_INT2,
    MAKE_INT3,
    MAKE_INT4,
    MAKE_UINT2,
    MAKE_UINT3,
    MAKE_UINT4,
    MAKE_UCHAR2,
    MAKE_UCHAR3,
    MAKE_UCHAR4,
    MAKE_FLOAT2,
    MAKE_FLOAT3,
    MAKE_FLOAT4,

    MAKE_FLOAT2X2,
    MAKE_FLOAT2X3,
    MAKE_FLOAT2X4,

    MAKE_FLOAT3X2,
    MAKE_FLOAT3X3,
    MAKE_FLOAT3X4,

    MAKE_FLOAT4X2,
    MAKE_FLOAT4X3,
    MAKE_FLOAT4X4,

    UNREACHABLE,

    ATOMIC_EXCH,
    ATOMIC_ADD,
    ATOMIC_SUB,
    ATOMIC_CAS,

    // ray tracing
    MAKE_RAY,
    RAY_OFFSET_ORIGIN,
    TRACE_CLOSEST,
    TRACE_OCCLUSION,

    COUNT
};

}// namespace ocarina