//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/basic_types.h"

namespace nano {

enum struct UnaryOp : uint32_t {
    POSITIVE,
    NEGATIVE,
    NOT,
    BIT_NOT
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

    SELECT,
    CLAMP,
    LERP,

    ABS,
    MIN,
    MAX,

    IS_INF,
    IS_NAN,

    ACOS,
    ACOSH,
    ASIN,
    ASINH,
    ATAN,
    ATAN2,
    ATANH,

    COS,
    COSH,
    SIN,
    SINH,
    TAN,
    TANH,

    EXP,
    EXP2,
    EXP10,
    LOG,
    LOG2,
    LOG10,
    POW,

    SQRT,
    RSQRT,

    CEIL,
    FLOOR,
    ROUND,

    FMA,

    CROSS,
    DOT,
    LENGTH,
    LENGTH_SQUARED,
    NORMALIZE,
    FACE_FORWARD,

    DETERMINANT,
    TRANSPOSE,
    INVERSE,

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
    MAKE_FLOAT2,
    MAKE_FLOAT3,
    MAKE_FLOAT4,

    MAKE_FLOAT2X2,
    MAKE_FLOAT3X3,
    MAKE_FLOAT4X4,

    COUNT
};

}// namespace nano