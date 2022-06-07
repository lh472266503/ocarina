//
// Created by Zero on 16/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "dsl/expr.h"
#include "ast/op.h"

#define NN_MAKE_DSL_UNARY_OPERATOR(op, tag)                                                  \
    template<typename T>                                                                     \
    requires nano::is_dsl_v<T>                                                               \
    [[nodiscard]] inline auto operator op(T &&expr) noexcept {                               \
        using Ret = std::remove_cvref_t<decltype(op std::declval<nano::expr_value_t<T>>())>; \
        return nano::def<Ret>(                                                               \
            nano::FunctionBuilder::current()->unary(                                         \
                nano::Type::of<Ret>(),                                                       \
                nano::UnaryOp::tag,                                                          \
                nano::detail::extract_expression(std::forward<T>(expr))));                   \
    }

NN_MAKE_DSL_UNARY_OPERATOR(+, POSITIVE)
NN_MAKE_DSL_UNARY_OPERATOR(-, NEGATIVE)
NN_MAKE_DSL_UNARY_OPERATOR(!, NOT)
NN_MAKE_DSL_UNARY_OPERATOR(~, BIT_NOT)

#undef NN_MAKE_DSL_UNARY_OPERATOR

#define NN_MAKE_DSL_BINARY_OPERATOR(op, tag)                                            \
    template<typename Lhs, typename Rhs>                                                \
    requires nano::any_dsl_v<Lhs, Rhs> &&                                               \
        nano::is_basic_v<nano::expr_value_t<Lhs>> &&                                    \
        nano::is_basic_v<nano::expr_value_t<Rhs>>                                       \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {              \
        using namespace std::string_view_literals;                                      \
        static constexpr bool is_logic_op = #op == "||"sv || #op == "&&"sv;             \
        static constexpr bool is_bit_op = #op == "|"sv || #op == "&"sv || #op == "^"sv; \
        static constexpr bool is_bool_lhs = nano::is_boolean_expr_v<Lhs>;               \
        static constexpr bool is_bool_rhs = nano::is_boolean_expr_v<Rhs>;               \
        using NormalRet = std::remove_cvref_t<                                          \
            decltype(std::declval<nano::expr_value_t<Lhs>>() op                         \
                         std::declval<nano::expr_value_t<Rhs>>())>;                     \
        using Ret = std::conditional_t<is_bool_lhs && is_logic_op, bool, NormalRet>;    \
        return def<Ret>(nano::FunctionBuilder::current()->binary(                       \
            nano::Type::of<Ret>(),                                                      \
            nano::detail::extract_expression(std::forward<Lhs>(lhs)),                   \
            nano::detail::extract_expression(std::forward<Rhs>(rhs)),                   \
            nano::BinaryOp::tag));                                                      \
    }

NN_MAKE_DSL_BINARY_OPERATOR(+, ADD)
NN_MAKE_DSL_BINARY_OPERATOR(-, SUB)
NN_MAKE_DSL_BINARY_OPERATOR(*, MUL)
NN_MAKE_DSL_BINARY_OPERATOR(/, DIV)
NN_MAKE_DSL_BINARY_OPERATOR(%, MOD)
NN_MAKE_DSL_BINARY_OPERATOR(&, BIT_AND)
NN_MAKE_DSL_BINARY_OPERATOR(|, BIT_OR)
NN_MAKE_DSL_BINARY_OPERATOR(^, BIT_XOR)
NN_MAKE_DSL_BINARY_OPERATOR(<<, SHL)
NN_MAKE_DSL_BINARY_OPERATOR(>>, SHR)
NN_MAKE_DSL_BINARY_OPERATOR(&&, AND)
NN_MAKE_DSL_BINARY_OPERATOR(||, OR)
NN_MAKE_DSL_BINARY_OPERATOR(==, EQUAL)
NN_MAKE_DSL_BINARY_OPERATOR(!=, NOT_EQUAL)
NN_MAKE_DSL_BINARY_OPERATOR(<, LESS)
NN_MAKE_DSL_BINARY_OPERATOR(<=, LESS_EQUAL)
NN_MAKE_DSL_BINARY_OPERATOR(>, GREATER)
NN_MAKE_DSL_BINARY_OPERATOR(>=, GREATER_EQUAL)

#undef NN_MAKE_DSL_BINARY_OPERATOR

#define NN_MAKE_DSL_ASSIGN_OP(op)                                                   \
    template<typename Lhs, typename Rhs>                                            \
    requires requires {                                                             \
        std::declval<Lhs &>() op## = std::declval<nano::expr_value_t<Rhs>>();       \
    }                                                                               \
    void operator+=(nano::Var<Lhs> lhs, Rhs &&rhs) {                                \
        auto x = lhs op std::forward<Rhs>(rhs);                                     \
        nano::FunctionBuilder::current()->assign(lhs.expression(), x.expression()); \
    }

NN_MAKE_DSL_ASSIGN_OP(+)
NN_MAKE_DSL_ASSIGN_OP(-)
NN_MAKE_DSL_ASSIGN_OP(*)
NN_MAKE_DSL_ASSIGN_OP(/)
NN_MAKE_DSL_ASSIGN_OP(|)
NN_MAKE_DSL_ASSIGN_OP(%)
NN_MAKE_DSL_ASSIGN_OP(&)
NN_MAKE_DSL_ASSIGN_OP(>>)
NN_MAKE_DSL_ASSIGN_OP(<<)
NN_MAKE_DSL_ASSIGN_OP(^)

#undef NN_MAKE_DSL_ASSIGN_OP