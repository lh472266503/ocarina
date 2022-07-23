//
// Created by Zero on 06/05/2022.
//

#pragma once

#include "core/basic_types.h"
#include "dsl/type_trait.h"
#include "dsl/computable.h"
#include "ast/function.h"

namespace ocarina {

struct Expression;

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

template<typename T>
class Expr : public detail::Computable<T> {
public:
    explicit Expr(const Expression *expression) noexcept
        : detail::Computable<T>(expression) {}

    Expr() = default;

    template<typename Arg>
    requires concepts::non_pointer<std::remove_cvref_t<Arg>> && concepts::different<Expr<T>, std::remove_cvref_t<Arg>>
    explicit Expr(Arg &&arg) : Expr(detail::extract_expression(std::forward<Arg>(arg))) {}

    auto operator->() noexcept { return reinterpret_cast<Proxy<T> *>(this); }
    auto operator->() const noexcept { return reinterpret_cast<const Proxy<T> *>(this); }

    Expr(const Expr &) = delete;
    Expr &operator=(const Expr &) = delete;
    Expr &operator=(Expr &&) = delete;
};

}// namespace ocarina