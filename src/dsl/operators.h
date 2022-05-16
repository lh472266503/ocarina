//
// Created by Zero on 16/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "dsl/expr.h"

#define KTN_MAKE_DSL_UNARY_OPERATOR(op, tag)                                                   \
    template<typename T>                                                                       \
    requires katana::is_dsl_v<T>                                                               \
    [[nodiscard]] inline auto operator op(T &&expr) noexcept {                                 \
        using Ret = std::remove_cvref_t<decltype(op std::declval<katana::expr_value_t<T>>())>; \
        return katana::def<Ret>(                                                               \
            katana::FunctionBuilder::current()->unary(                                         \
                katana::Type::of<Ret>(),                                                       \
                katana::UnaryOp::tag,                                                          \
                katana::detail::extract_expression(std::forward<T>(expr))));                   \
    }

KTN_MAKE_DSL_UNARY_OPERATOR(+, POSITIVE)
KTN_MAKE_DSL_UNARY_OPERATOR(-, NEGATIVE)
KTN_MAKE_DSL_UNARY_OPERATOR(!, NOT)
KTN_MAKE_DSL_UNARY_OPERATOR(~, BIT_NOT)

namespace katana {

template<typename Lhs, typename Rhs>
requires any_dsl_v<Lhs, Rhs> && is_basic_v<expr_value_t<Lhs>> && is_basic_v<expr_value_t<Rhs>>
[[nodiscard]] inline auto operator+(Lhs &&lhs, Rhs &&rhs) noexcept {
}

}// namespace katana