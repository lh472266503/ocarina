//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "dsl/var.h"
#include "dsl/expr.h"
#include "ast/expression.h"

namespace ocarina {
template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(T &&x) noexcept {
    return Var(std::forward<T>(x));
}

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(const Expression *expr) noexcept {
    using RawType = expr_value_t<T>;
    return Var<RawType>(Expr<RawType>(expr));
}

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> def_expr(T &&x) noexcept {
    if constexpr (is_expr_v<std::remove_cvref_t<T>>) {
        return def_expr<T>(x.expression());
    } else {
        return Expr<expr_value_t<T>>(std::forward<T>(x));
    }
}

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> def_expr(const Expression *expr) noexcept {
    using RawType = expr_value_t<T>;
    return Expr<RawType>(expr);
}

}// namespace ocarina