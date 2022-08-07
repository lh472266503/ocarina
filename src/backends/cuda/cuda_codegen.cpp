//
// Created by Zero on 2022/7/15.
//

#include "cuda_codegen.h"
#include "ast/expression.h"

namespace ocarina {

void CUDACodegen::visit(const CallExpr *expr) noexcept {
#define OC_GEN_FUNC_NAME(func_name) current_scratch() << TYPE_PREFIX #func_name
    switch (expr->call_op()) {
        case CallOp::CUSTOM: _emit_func_name(*expr->function()); break;
        case CallOp::ALL: OC_GEN_FUNC_NAME(all); break;
        case CallOp::ANY: OC_GEN_FUNC_NAME(any); break;
        case CallOp::NONE: OC_GEN_FUNC_NAME(none); break;
        case CallOp::SELECT: OC_GEN_FUNC_NAME(select); break;
        case CallOp::CLAMP: OC_GEN_FUNC_NAME(clamp); break;
        case CallOp::LERP: OC_GEN_FUNC_NAME(lerp); break;
        case CallOp::ABS: OC_GEN_FUNC_NAME(abs); break;
        case CallOp::MIN: OC_GEN_FUNC_NAME(min); break;
        case CallOp::MAX: OC_GEN_FUNC_NAME(max); break;
        case CallOp::IS_INF: OC_GEN_FUNC_NAME(is_inf); break;
        case CallOp::IS_NAN: OC_GEN_FUNC_NAME(is_nan); break;
        case CallOp::ACOS: OC_GEN_FUNC_NAME(acos); break;
        case CallOp::ASIN: OC_GEN_FUNC_NAME(asin); break;
        case CallOp::ATAN: OC_GEN_FUNC_NAME(atan); break;
        case CallOp::ATAN2: OC_GEN_FUNC_NAME(atan2); break;
        case CallOp::COS: OC_GEN_FUNC_NAME(cos); break;
        case CallOp::SIN: OC_GEN_FUNC_NAME(sin); break;
        case CallOp::TAN: OC_GEN_FUNC_NAME(tan); break;
        case CallOp::EXP: OC_GEN_FUNC_NAME(exp); break;
        case CallOp::EXP2: OC_GEN_FUNC_NAME(exp2); break;
        case CallOp::EXP10: OC_GEN_FUNC_NAME(exp10); break;
        case CallOp::LOG: OC_GEN_FUNC_NAME(log); break;
        case CallOp::LOG2: OC_GEN_FUNC_NAME(log2); break;
        case CallOp::LOG10: OC_GEN_FUNC_NAME(log10); break;
        case CallOp::POW: OC_GEN_FUNC_NAME(pow); break;
        case CallOp::SQRT: OC_GEN_FUNC_NAME(sqrt); break;
        case CallOp::RSQRT: OC_GEN_FUNC_NAME(rsqrt); break;
        case CallOp::CEIL: OC_GEN_FUNC_NAME(ceil); break;
        case CallOp::FLOOR: OC_GEN_FUNC_NAME(floor); break;
        case CallOp::ROUND: OC_GEN_FUNC_NAME(round); break;
        case CallOp::FMA: OC_GEN_FUNC_NAME(fma); break;
        case CallOp::CROSS: OC_GEN_FUNC_NAME(cross); break;
        case CallOp::DOT: OC_GEN_FUNC_NAME(dot); break;
        case CallOp::LENGTH: OC_GEN_FUNC_NAME(length); break;
        case CallOp::LENGTH_SQUARED: OC_GEN_FUNC_NAME(length_squared); break;
        case CallOp::DISTANCE: OC_GEN_FUNC_NAME(distance); break;
        case CallOp::DISTANCE_SQUARED: OC_GEN_FUNC_NAME(distance_squared); break;
        case CallOp::NORMALIZE: OC_GEN_FUNC_NAME(normalize); break;
        case CallOp::FACE_FORWARD: OC_GEN_FUNC_NAME(face_forward); break;
        case CallOp::DETERMINANT: OC_GEN_FUNC_NAME(det); break;
        case CallOp::TRANSPOSE: OC_GEN_FUNC_NAME(transpose); break;
        case CallOp::INVERSE: OC_GEN_FUNC_NAME(inverse); break;
        case CallOp::SQR: OC_GEN_FUNC_NAME(sqr); break;
        case CallOp::RCP: OC_GEN_FUNC_NAME(rcp); break;
        case CallOp::DEGREES: OC_GEN_FUNC_NAME(degrees); break;
        case CallOp::RADIANS: OC_GEN_FUNC_NAME(radians); break;
        case CallOp::SATURATE: OC_GEN_FUNC_NAME(saturate); break;
        case CallOp::SYNCHRONIZE_BLOCK: break;
        case CallOp::MAKE_BOOL2: OC_GEN_FUNC_NAME(make_bool2); break;
        case CallOp::MAKE_BOOL3: OC_GEN_FUNC_NAME(make_bool3); break;
        case CallOp::MAKE_BOOL4: OC_GEN_FUNC_NAME(make_bool4); break;
        case CallOp::MAKE_INT2: OC_GEN_FUNC_NAME(make_int2); break;
        case CallOp::MAKE_INT3: OC_GEN_FUNC_NAME(make_int3); break;
        case CallOp::MAKE_INT4: OC_GEN_FUNC_NAME(make_int4); break;
        case CallOp::MAKE_UINT2: OC_GEN_FUNC_NAME(make_uint2); break;
        case CallOp::MAKE_UINT3: OC_GEN_FUNC_NAME(make_uint3); break;
        case CallOp::MAKE_UINT4: OC_GEN_FUNC_NAME(make_uint4); break;
        case CallOp::MAKE_FLOAT2: OC_GEN_FUNC_NAME(make_float2); break;
        case CallOp::MAKE_FLOAT3: OC_GEN_FUNC_NAME(make_float3); break;
        case CallOp::MAKE_FLOAT4: OC_GEN_FUNC_NAME(make_float4); break;
        case CallOp::MAKE_FLOAT2X2: OC_GEN_FUNC_NAME(make_float2x2); break;
        case CallOp::MAKE_FLOAT3X3: OC_GEN_FUNC_NAME(make_float3x3); break;
        case CallOp::MAKE_FLOAT4X4: OC_GEN_FUNC_NAME(make_float4x4); break;
        case CallOp::TEX_SAMPLE:
            current_scratch() << "tex_sample_float" << expr->type()->dimension();
            break;
        case CallOp::COUNT: break;
        default: OC_ASSERT(0); break;
    }
#undef OC_GEN_FUNC_NAME
    current_scratch() << "(";
    for (const auto &arg : expr->arguments()) {
        arg->accept(*this);
        current_scratch() << ",";
    }
    if (!expr->arguments().empty()) {
        current_scratch().pop_back();
    }
    current_scratch() << ")";
}

void CUDACodegen::visit(const MemberExpr *expr) noexcept {
    if (expr->is_swizzle()) {
        static constexpr std::string_view xyzw[] = {"x", "y", "z", "w"};
        int swizzle_size = expr->swizzle_size();
        if (swizzle_size == 1) {
            expr->parent()->accept(*this);
            current_scratch() << ".";
            current_scratch() << xyzw[expr->swizzle_index(0)];
        } else {
            _emit_type_name(expr->type());
            current_scratch() << swizzle_size;
            current_scratch() << "(";
            for (int i = 0; i < swizzle_size; ++i) {
                expr->parent()->accept(*this);
                current_scratch() << ".";
                current_scratch() << xyzw[expr->swizzle_index(i)] << ",";
            }
            current_scratch().pop_back();
            current_scratch() << ")";
        }
    } else {
        expr->parent()->accept(*this);
        current_scratch() << ".";
        _emit_member_name(expr->member_index());
    }
}

void CUDACodegen::_emit_function(const Function &f) noexcept {
    if (has_generated(&f)) {
        return;
    }
    switch (f.tag()) {
        case Function::Tag::KERNEL: current_scratch() << "extern \"C\" __global__ "; break;
        case Function::Tag::CALLABLE: current_scratch() << "__device__ "; break;
    }
    CppCodegen::_emit_function(f);
}

void CUDACodegen::_emit_builtin_vars_define(const Function &f) noexcept {
    CppCodegen::_emit_builtin_vars_define(f);
    if (f.dispatch_dim_valid()) {
        // skip the kernel
    }
}

void CUDACodegen::_emit_builtin_var(Variable v) noexcept {
    _emit_type_name(v.type());
    _emit_space();
    _emit_variable_name(v);
    using Tag = Variable::Tag;
    current_scratch() << " = ";
    switch (v.tag()) {
        case Tag::BLOCK_IDX:
            current_scratch() << "oc_uint3(blockIdx.x, blockIdx.y, blockIdx.z)";
            break;
        case Tag::THREAD_IDX:
            current_scratch() << "oc_uint3(threadIdx.x, threadIdx.y, threadIdx.z)";
            break;
        case Tag::THREAD_ID:
            current_scratch() << "(blockIdx.x + blockIdx.y * gridDim.x"
                                 "+ gridDim.x * gridDim.y * blockIdx.z) *"
                                 "(blockDim.x * blockDim.y * blockDim.z)"
                                 " + (threadIdx.z * (blockDim.x * blockDim.y))"
                                 " + (threadIdx.y * blockDim.x) + threadIdx.x";
            break;
        case Tag::DISPATCH_IDX:
            break;
        case Tag::DISPATCH_ID: break;
        case Tag::DISPATCH_DIM: {
            uint3 dim = current_function().dispatch_dim();
            current_scratch() << fmt::format("oc_uint3({}, {}, {})", dim.x, dim.y, dim.z);
            break;
        }
        default:
            OC_ASSERT(0);
            break;
    }
}

void CUDACodegen::_emit_uniform_var(const UniformBinding &uniform) noexcept {
    current_scratch() << "extern \"C\" __constant__ ";
    _emit_type_name(uniform.type());
    _emit_space();
    uniform.expression()->accept(*this);
    current_scratch() << ";";
    _emit_newline();
}

void CUDACodegen::_emit_type_name(const Type *type) noexcept {
    if (type == nullptr) {
        current_scratch() << "void";
    } else {
        switch (type->tag()) {
            case Type::Tag::BOOL: current_scratch() << TYPE_PREFIX "bool"; break;
            case Type::Tag::FLOAT: current_scratch() << TYPE_PREFIX "float"; break;
            case Type::Tag::INT: current_scratch() << TYPE_PREFIX "int"; break;
            case Type::Tag::UINT: current_scratch() << TYPE_PREFIX "uint"; break;
            case Type::Tag::VECTOR:
                _emit_type_name(type->element());
                current_scratch() << type->dimension();
                break;
            case Type::Tag::ARRAY:
                _emit_type_name(type->element());
                current_scratch() << "[";
                current_scratch() << type->dimension();
                current_scratch() << "]";
                break;
            case Type::Tag::MATRIX: {
                auto d = type->dimension();
                current_scratch() << TYPE_PREFIX "float" << d << "x" << d;
                break;
            }
            case Type::Tag::STRUCTURE:
                _emit_struct_name(type->hash());
                break;
            case Type::Tag::BUFFER:
                _emit_type_name(type->element());
                current_scratch() << "*";
                break;
            case Type::Tag::TEXTURE:
                current_scratch() << "cudaTextureObject_t";
                break;
            case Type::Tag::BINDLESS_ARRAY: break;
            case Type::Tag::ACCEL: break;
            case Type::Tag::NONE: break;
        }
    }
}

}// namespace ocarina