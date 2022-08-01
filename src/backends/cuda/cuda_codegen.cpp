//
// Created by Zero on 2022/7/15.
//

#include "cuda_codegen.h"
#include "ast/expression.h"



namespace ocarina {

void CUDACodegen::visit(const CallExpr *expr) noexcept {
    switch (expr->call_op()) {
        case CallOp::CUSTOM: {
            _emit_func_name(*expr->function());
            current_scratch() << "(";
            for (const auto &arg : expr->arguments()) {
                arg->accept(*this);
                current_scratch() << ",";
            }
            if (!expr->arguments().empty()) {
                current_scratch().pop_back();
            }
            current_scratch() << ")";
            break;
        }
        case CallOp::ALL: break;
        case CallOp::ANY: break;
        case CallOp::NONE: break;
        case CallOp::SELECT: {
            current_scratch() << TYPE_PREFIX"select";
            current_scratch() << "(";
            for (const auto &arg : expr->arguments()) {
                arg->accept(*this);
                current_scratch() << ",";
            }
            if (!expr->arguments().empty()) {
                current_scratch().pop_back();
            }
            current_scratch() << ")";
            break;
        }
        case CallOp::CLAMP: break;
        case CallOp::LERP: break;
        case CallOp::ABS: break;
        case CallOp::MIN: break;
        case CallOp::MAX: break;
        case CallOp::IS_INF: break;
        case CallOp::IS_NAN: break;
        case CallOp::ACOS: break;
        case CallOp::ACOSH: break;
        case CallOp::ASIN: break;
        case CallOp::ASINH: break;
        case CallOp::ATAN: break;
        case CallOp::ATAN2: break;
        case CallOp::ATANH: break;
        case CallOp::COS: break;
        case CallOp::COSH: break;
        case CallOp::SIN: break;
        case CallOp::SINH: break;
        case CallOp::TAN: break;
        case CallOp::TANH: break;
        case CallOp::EXP: break;
        case CallOp::EXP2: break;
        case CallOp::EXP10: break;
        case CallOp::LOG: break;
        case CallOp::LOG2: break;
        case CallOp::LOG10: break;
        case CallOp::POW: break;
        case CallOp::SQRT: break;
        case CallOp::RSQRT: break;
        case CallOp::CEIL: break;
        case CallOp::FLOOR: break;
        case CallOp::ROUND: break;
        case CallOp::FMA: break;
        case CallOp::CROSS: break;
        case CallOp::DOT: break;
        case CallOp::LENGTH: break;
        case CallOp::LENGTH_SQUARED: break;
        case CallOp::NORMALIZE: break;
        case CallOp::FACE_FORWARD: break;
        case CallOp::DETERMINANT: break;
        case CallOp::TRANSPOSE: break;
        case CallOp::INVERSE: break;
        case CallOp::SYNCHRONIZE_BLOCK: break;
        case CallOp::MAKE_BOOL2: break;
        case CallOp::MAKE_BOOL3: break;
        case CallOp::MAKE_BOOL4: break;
        case CallOp::MAKE_INT2: break;
        case CallOp::MAKE_INT3: break;
        case CallOp::MAKE_INT4: break;
        case CallOp::MAKE_UINT2: break;
        case CallOp::MAKE_UINT3: break;
        case CallOp::MAKE_UINT4: break;
        case CallOp::MAKE_FLOAT2: break;
        case CallOp::MAKE_FLOAT3: break;
        case CallOp::MAKE_FLOAT4: break;
        case CallOp::MAKE_FLOAT2X2: break;
        case CallOp::MAKE_FLOAT3X3: break;
        case CallOp::MAKE_FLOAT4X4: break;
        case CallOp::COUNT: break;
    }
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
    if (f.has_defined()) {
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
            case Type::Tag::BOOL: current_scratch() << TYPE_PREFIX"bool"; break;
            case Type::Tag::FLOAT: current_scratch() << TYPE_PREFIX"float"; break;
            case Type::Tag::INT: current_scratch() << TYPE_PREFIX"int"; break;
            case Type::Tag::UINT: current_scratch() << TYPE_PREFIX"uint"; break;
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
                current_scratch() << TYPE_PREFIX"float" << d << "x" << d;
                break;
            }
            case Type::Tag::STRUCTURE:
                _emit_struct_name(type->hash());
                break;
            case Type::Tag::BUFFER:
                _emit_type_name(type->element());
                current_scratch() << "*";
                break;
            case Type::Tag::TEXTURE: break;
            case Type::Tag::BINDLESS_ARRAY: break;
            case Type::Tag::ACCEL: break;
            case Type::Tag::NONE: break;
        }
    }
}
#undef TYPE_PREFIX
}// namespace ocarina