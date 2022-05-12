//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "dsl/var.h"
#include "dsl/expr.h"
#include "ast/expression.h"

namespace katana::dsl {
template<typename T>
KTN_NODISCARD inline Var<expr_value_t<T>> def(T &&x) noexcept {
    return Var{Expr{std::forward<T>(x)}};
}

template<typename T>
KTN_NODISCARD inline Var<expr_value_t<T>> def(const ast::Expression *expr) noexcept {
    return Var{Expr<expr_value_t<T>>{expr}};
}
}// namespace katana::dsl