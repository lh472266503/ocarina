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

template<typename U, typename T, typename F>
requires(is_dsl_v<U> &&
                 vector_dimension_v<expr_value_t<U>> == vector_dimension_v<expr_value_t<T>> &&
         vector_dimension_v<expr_value_t<U>> == vector_dimension_v<expr_value_t<F>>)
    [[nodiscard]] auto select(U &&pred, T &&t, F &&f) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::SELECT,
                                                  {OC_EXPR(pred), OC_EXPR(t), OC_EXPR(f)});
    return make_expr<T>(expr);
}

}// namespace ocarina