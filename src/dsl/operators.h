//
// Created by Zero on 16/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "dsl/expr.h"

namespace katana {

}// namespace katana

#define KTN_MAKE_DSL_UNARY_OPERATOR(op, tag)                                                   \
    template<typename T>                                                                       \
    requires katana::is_dsl_v<T>                                                               \
        KTN_NODISCARD inline auto operator op(T &&expr) noexcept {                             \
        using Ret = std::remove_cvref_t<decltype(op std::declval<katana::expr_value_t<T>>())>; \
        return katana::def<Ret>(katana::FunctionBuilder::current()->unary(                     \
            katana::Type::of<Ret>(),                                                           \
            katana::UnaryOp::tag,                                                              \
            katana::detail::extract_expression(std::forward<T>(expr))));                       \
    }

KTN_MAKE_DSL_UNARY_OPERATOR(+, POSITIVE)
KTN_MAKE_DSL_UNARY_OPERATOR(-, NEGATIVE)
KTN_MAKE_DSL_UNARY_OPERATOR(!, NOT)
KTN_MAKE_DSL_UNARY_OPERATOR(~, BIT_NOT)
