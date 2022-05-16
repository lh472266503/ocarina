//
// Created by Zero on 06/05/2022.
//

#pragma once

#include "core/basic_types.h"
#include "dsl/expr_traits.h"
#include "ast/function_builder.h"

namespace katana {

struct Expression;

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(T &&x) noexcept;// implement in builtin.h

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(const Expression *expr) noexcept;// implement in builtin.h

namespace detail {
template<typename T>
struct ExprEnableStaticCast {
    template<typename Dest>
    requires concepts::static_convertible<expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using ExprDest = expr_value_t<Dest>;
        return def(katana::FunctionBuilder::current()->cast(Type::of<ExprDest>(), CastOp::STATIC, src));
    }
};

template<typename T>
struct ExprEnableBitwiseCast {
    template<class Dest>
    requires concepts::bitwise_convertible<expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto bit_cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using ExprDest = expr_value_t<Dest>;
        return def(katana::FunctionBuilder::current()->cast(Type::of<ExprDest>(), CastOp::BITWISE, src));
    }
};

template<typename T>
[[nodiscard]] decltype(auto) extract_expression(T &&v) noexcept {
    if constexpr (is_dsl_v<T>) {
        return std::forward<T>(v).expression();
    } else {
        return katana::FunctionBuilder::current()->literal(Type::of<T>(), std::forward<T>(v));
    }
}

}// namespace detail

template<typename T>
class Expr {
};
}// namespace katana