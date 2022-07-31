//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "dsl/var.h"
#include "dsl/expr.h"
#include "ast/expression.h"

namespace ocarina {

[[nodiscard]] inline Expr<uint3> dispatch_idx() noexcept {
    return make_expr<uint3>(Function::current()->dispatch_idx());
}

[[nodiscard]] inline Expr<uint3> block_idx() noexcept {
    return make_expr<uint3>(Function::current()->block_idx());
}

[[nodiscard]] inline Expr<uint> thread_id() noexcept {
    return make_expr<uint>(Function::current()->thread_id());
}

[[nodiscard]] inline Expr<uint> dispatch_id() noexcept {
    return make_expr<uint>(Function::current()->dispatch_id());
}

[[nodiscard]] inline Expr<uint3> thread_idx() noexcept {
    return make_expr<uint3>(Function::current()->thread_idx());
}

[[nodiscard]] inline Expr<uint3> dispatch_dim() noexcept {
    return make_expr<uint3>(Function::current()->dispatch_dim());
}

template<typename T>
[[nodiscard]] auto select(Var<bool> pred, T &&t, T &&f) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(), CallOp::SELECT, {OC_EXPR(pred), OC_EXPR(t), OC_EXPR(f)});
    return make_expr<T>(expr);
}

}// namespace ocarina