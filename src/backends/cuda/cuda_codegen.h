//
// Created by zero on 2022/7/15.
//

#pragma once

#include "generator/cpp_codegen.h"

namespace ocarina {
class CUDACodegen final : public CppCodegen {
protected:
    void visit(const MemberExpr *expr) noexcept override;
    void visit(const CallExpr *expr) noexcept override;
    void _emit_function(const Function &f) noexcept override;
    void _emit_type_name(const Type *type) noexcept override;
    void _emit_arguments(const Function &f) noexcept override;
    void _emit_builtin_var(Variable v) noexcept override;
    void _emit_uniform_var(const UniformBinding &uniform) noexcept override;
    void _emit_struct_name(const Type *type) noexcept override;
    void _emit_builtin_vars_define(const Function &f) noexcept override;
};
}// namespace ocarina