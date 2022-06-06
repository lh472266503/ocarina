//
// Created by Zero on 21/05/2022.
//

#pragma once

#include "var.h"
#include "operators.h"

namespace nano {

template<typename Lhs, typename Rhs>
inline void assign(Lhs &&lhs, Rhs &&rhs) noexcept {
    static_assert(tuple_size_v<linear_layout_t<Lhs>> == tuple_size_v<linear_layout_t<Rhs>>);
    if constexpr (concepts::assign_able<expr_value_t<Lhs>, expr_value_t<Rhs>>) {
        FunctionBuilder::current()->assign(
            detail::extract_expression(std::forward<Lhs>(lhs)),
            detail::extract_expression(std::forward<Rhs>(rhs)));
    } else {

    }
}

}// namespace nano