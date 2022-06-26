//
// Created by Zero on 06/05/2022.
//

#pragma once

#include "core/basic_types.h"
#include "dsl/expr_traits.h"
#include "dsl/computable.h"
#include "ast/function.h"

namespace ocarina {

struct Expression;

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(T &&x) noexcept;// implement in builtin.h

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(const Expression *expr) noexcept;// implement in builtin.h

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> def_expr(T &&x) noexcept;

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> def_expr(const Expression *expr) noexcept;

template<typename T>
class Expr : public detail::Computable<T> {
public:
    explicit Expr(const Expression *expression) noexcept
        : detail::Computable<T>(expression) {}

    template<typename Arg>
    requires concepts::non_pointer<std::remove_cvref_t<Arg>> && concepts::different<Expr<T>, std::remove_cvref_t<Arg>>
    explicit Expr(Arg &&arg) : Expr(arg.expression()) {}
    Expr(const Expr &) = delete;
    Expr &operator=(const Expr &) = delete;
    Expr &operator=(Expr &&) = delete;
};

namespace detail {

template<typename T>
[[nodiscard]] decltype(auto) extract_expression(T &&v) noexcept {
    if constexpr (is_dsl_v<T>) {
        return std::forward<T>(v).expression();
    } else {
        return ocarina::Function::current()->literal(Type::of<T>(), std::forward<T>(v));
    }
}

}// namespace detail

}// namespace ocarina