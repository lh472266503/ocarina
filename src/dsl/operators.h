//
// Created by Zero on 16/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "dsl/expr.h"

namespace katana {

template<typename T>
requires katana::is_dsl_v<T>
    KTN_NODISCARD inline auto operator-(T &&expr) noexcept {
    using Ret = std::remove_cvref_t<decltype(-std::declval<katana::expr_value_t<T>>())>;
    return katana::def<Ret>(katana::FunctionBuilder::current()->unary(
        Type::of<Ret>(),
        UnaryOp::NEGATIVE,
        katana::detail::extract_expression(std::forward<T>(expr))));
}

}// namespace katana