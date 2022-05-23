//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "dsl/var.h"
#include "dsl/expr.h"
#include "ast/expression.h"

namespace katana {
template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(T &&x) noexcept {
    return Var(std::forward<T>(x));
}

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(const Expression *expr) noexcept {
    return Var<expr_value_t<T>>(expr);
}
}// namespace katana