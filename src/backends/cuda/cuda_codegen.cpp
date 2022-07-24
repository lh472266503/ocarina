//
// Created by Zero on 2022/7/15.
//

#include "cuda_codegen.h"
#include "ast/expression.h"

#define TYPE_PREFIX "oc_"

namespace ocarina {

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