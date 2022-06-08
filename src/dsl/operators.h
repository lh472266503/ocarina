//
// Created by Zero on 16/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "dsl/expr.h"
#include "ast/op.h"

#define NN_MAKE_DSL_UNARY_OPERATOR(op, tag)                                                     \
    template<typename T>                                                                        \
    requires ocarina::is_dsl_v<T>                                                               \
    [[nodiscard]] inline auto operator op(T &&expr) noexcept {                                  \
        using Ret = std::remove_cvref_t<decltype(op std::declval<ocarina::expr_value_t<T>>())>; \
        return ocarina::def<Ret>(                                                               \
            ocarina::FunctionBuilder::current()->unary(                                         \
                ocarina::Type::of<Ret>(),                                                       \
                ocarina::UnaryOp::tag,                                                          \
                ocarina::detail::extract_expression(std::forward<T>(expr))));                   \
    }

NN_MAKE_DSL_UNARY_OPERATOR(+, POSITIVE)
NN_MAKE_DSL_UNARY_OPERATOR(-, NEGATIVE)
NN_MAKE_DSL_UNARY_OPERATOR(!, NOT)
NN_MAKE_DSL_UNARY_OPERATOR(~, BIT_NOT)

#undef NN_MAKE_DSL_UNARY_OPERATOR

#define NN_MAKE_DSL_BINARY_OPERATOR(op, tag)                                            \
    template<typename Lhs, typename Rhs>                                                \
    requires ocarina::any_dsl_v<Lhs, Rhs> &&                                            \
        ocarina::is_basic_v<ocarina::expr_value_t<Lhs>> &&                              \
        ocarina::is_basic_v<ocarina::expr_value_t<Rhs>>                                 \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {              \
        using namespace std::string_view_literals;                                      \
        static constexpr bool is_logic_op = #op == "||"sv || #op == "&&"sv;             \
        static constexpr bool is_bit_op = #op == "|"sv || #op == "&"sv || #op == "^"sv; \
        static constexpr bool is_bool_lhs = ocarina::is_boolean_expr_v<Lhs>;            \
        static constexpr bool is_bool_rhs = ocarina::is_boolean_expr_v<Rhs>;            \
        using NormalRet = std::remove_cvref_t<                                          \
            decltype(std::declval<ocarina::expr_value_t<Lhs>>() op                      \
                         std::declval<ocarina::expr_value_t<Rhs>>())>;                  \
        using Ret = std::conditional_t<is_bool_lhs && is_logic_op, bool, NormalRet>;    \
        return def<Ret>(ocarina::FunctionBuilder::current()->binary(                    \
            ocarina::Type::of<Ret>(),                                                   \
            ocarina::detail::extract_expression(std::forward<Lhs>(lhs)),                \
            ocarina::detail::extract_expression(std::forward<Rhs>(rhs)),                \
            ocarina::BinaryOp::tag));                                                   \
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

#define NN_MAKE_DSL_ASSIGN_OP(op)                                                      \
    template<typename Lhs, typename Rhs>                                               \
    requires requires {                                                                \
        std::declval<Lhs &>() op## = std::declval<ocarina::expr_value_t<Rhs>>();       \
    }                                                                                  \
    void operator+=(ocarina::Var<Lhs> lhs, Rhs &&rhs) {                                \
        auto x = lhs op std::forward<Rhs>(rhs);                                        \
        ocarina::FunctionBuilder::current()->assign(lhs.expression(), x.expression()); \
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