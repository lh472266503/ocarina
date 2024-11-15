//
// Created by Zero on 2024/11/8.
//

#include "pyexporter/ocapi.h"
#include "math/base.h"

namespace py = pybind11;
using namespace ocarina;

void export_op_enum(PythonExporter &exporter) {
    using ExpressionTag = Expression::Tag;
    OC_EXPORT_ENUM(exporter.module, ExpressionTag, UNARY,
                   BINARY, MEMBER, SUBSCRIPT,
                   LITERAL, REF, CONSTANT, CALL,
                   CAST, CONDITIONAL)

    OC_EXPORT_ENUM(exporter.module, UnaryOp, POSITIVE,
                   NEGATIVE, NOT, BIT_NOT)

    OC_EXPORT_ENUM(exporter.module, CastOp, STATIC, BITWISE)

    OC_EXPORT_ENUM(exporter.module, BinaryOp, ADD,
                   SUB, MUL, DIV, MOD, BIT_AND,
                   BIT_OR, BIT_XOR, SHL, SHR, AND,
                   OR, LESS, GREATER, LESS_EQUAL,
                   GREATER_EQUAL, EQUAL, NOT_EQUAL)

    OC_EXPORT_ENUM(exporter.module, CallOp, CUSTOM,ALL,ANY,NONE,SELECT,CLAMP,LERP,
                   ABS,MIN,MAX,IS_INF,IS_NAN,ACOS,ASIN,ATAN,ACOSH,ASINH,ATANH,ATAN2,
                   COPYSIGN,COS,SIN,TAN,COSH,SINH,TANH,EXP,EXP2,EXP10,LOG,LOG2,LOG10,
                   POW,FMOD,MOD,FRACT,SQRT,RSQRT,SQR,RCP,SIGN,CEIL,FLOOR,ROUND,
                   DEGREES,RADIANS,SATURATE,FMA,CROSS,DOT,DISTANCE,DISTANCE_SQUARED,LENGTH,
                   LENGTH_SQUARED,NORMALIZE,FACE_FORWARD,COORDINATE_SYSTEM,DETERMINANT,
                   TRANSPOSE,INVERSE,IS_NULL_BUFFER,IS_NULL_TEXTURE,BUFFER_SIZE,
                   TEX_SAMPLE,TEX_READ,TEX_WRITE,BYTE_BUFFER_READ,BYTE_BUFFER_WRITE,BYTE_BUFFER_SIZE,

                   BINDLESS_ARRAY_BUFFER_READ,
                   BINDLESS_ARRAY_BUFFER_WRITE,
                   BINDLESS_ARRAY_BUFFER_SIZE,
                   BINDLESS_ARRAY_BYTE_BUFFER_READ,
                   BINDLESS_ARRAY_BYTE_BUFFER_WRITE,
                   BINDLESS_ARRAY_TEX_SAMPLE,

                   SYNCHRONIZE_BLOCK,

                   MAKE_BOOL2,MAKE_BOOL3,MAKE_BOOL4,MAKE_INT2,MAKE_INT3,MAKE_INT4,
                   MAKE_UINT2,MAKE_UINT3,MAKE_UINT4,MAKE_UCHAR2,MAKE_UCHAR3,MAKE_UCHAR4,
                   MAKE_FLOAT2,MAKE_FLOAT3,MAKE_FLOAT4,

                   MAKE_FLOAT2X2,MAKE_FLOAT2X3,MAKE_FLOAT2X4,
                   MAKE_FLOAT3X2,MAKE_FLOAT3X3,MAKE_FLOAT3X4,
                   MAKE_FLOAT4X2,MAKE_FLOAT4X3,MAKE_FLOAT4X4,

                   UNREACHABLE,

                   ATOMIC_EXCH,ATOMIC_ADD,ATOMIC_SUB,

                   // ray tracing
                   MAKE_RAY,RAY_OFFSET_ORIGIN,TRACE_CLOSEST,TRACE_OCCLUSION,

                   COUNT)
}

void export_expressions(PythonExporter &exporter) {
    export_op_enum(exporter);
    py::class_<ASTNode>(exporter.module, "ASTNode")
        .def("check_context", &ASTNode::check_context);
    py::class_<Expression, ASTNode, concepts::Noncopyable, Hashable>(exporter.module, "Expression")
        .def("hash", &Expression::hash)
        .def("tag", &Expression::tag)
        .def("type", &Expression::type, ret_policy::reference);

    py::class_<UnaryExpr, Expression>(exporter.module, "UnaryExpr")
        .def("op", &UnaryExpr::op);
}