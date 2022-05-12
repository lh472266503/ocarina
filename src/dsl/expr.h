//
// Created by Zero on 06/05/2022.
//

#pragma once

#include "core/basic_types.h"
#include "ast/expression.h"
#include "ast/function_builder.h"

namespace katana::dsl {

template<typename T>
KTN_NODISCARD inline Var<expr_value_t<T>> def(T &&x) noexcept;

template<typename T>
KTN_NODISCARD inline Var<expr_value_t<T>> def(const ast::Expression *expr) noexcept;

namespace detail {
template<typename T>
struct ExprEnableStaticCast {
    template<typename Dest>
    requires concepts::static_convertible<expr_value_t<T>, expr_value_t<Dest>>
        KTN_NODISCARD auto cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using ExprDest = expr_value_t<Dest>;
        return def(ast::FunctionBuilder::current()->cast(Type::of<ExprDest>(), ast::CastOp::STATIC, src));
    }
};

template<typename T>
struct ExprEnableBitwiseCast {
    template<class Dest>
    requires concepts::bitwise_convertible<expr_value_t<T>, expr_value_t<Dest>>
    KTN_NODISCARD auto bit_cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using ExprDest = expr_value_t<Dest>;
        return def(ast::FunctionBuilder::current()->cast(Type::of<ExprDest>(), ast::CastOp::BITWISE, src));
    }
};
}// namespace detail

template<typename T>
class Expr {
};
}// namespace katana::dsl