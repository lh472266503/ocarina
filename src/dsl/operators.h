//
// Created by Zero on 16/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "dsl/expr.h"
#include "ast/op.h"

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

#undef KTN_MAKE_DSL_UNARY_OPERATOR

#define KTN_MAKE_DSL_BINARY_OPERATOR(op, tag)                                            \
    template<typename Lhs, typename Rhs>                                                 \
    requires katana::any_dsl_v<Lhs, Rhs> &&                                              \
        katana::is_basic_v<katana::expr_value_t<Lhs>> &&                                 \
        katana::is_basic_v<katana::expr_value_t<Rhs>>                                    \
    [[nodiscard]] inline auto operator op(Lhs &&lhs, Rhs &&rhs) noexcept {               \
        using namespace std::string_view_literals;                                       \
        static constexpr bool is_logic_op = #op == "||"sv || #op == "&&"sv;              \
        static constexpr bool is_bit_op = #op == "|"sv || #op == "&"sv || #op == "^"sv;  \
        static constexpr bool is_bool_lhs = katana::is_boolean_expr_v<Lhs>;              \
        static constexpr bool is_bool_rhs = katana::is_boolean_expr_v<Rhs>;              \
        using NormalRet = std::remove_cvref_t<                                           \
            decltype(std::declval<katana::expr_value_t<Lhs>>() op                        \
                         std::declval<katana::expr_value_t<Rhs>>())>;                    \
        using Ret = std::conditional_t<is_bool_lhs && is_logic_op, bool, NormalRet>;     \
        return def<Ret>(katana::FunctionBuilder::current()->binary(                      \
                            katana::Type::of<Ret>(),                                     \
                            katana::detail::extract_expression(std::forward<Lhs>(lhs)),  \
                            katana::detail::extract_expression(std::forward<Rhs>(rhs))), \
                        katana::BinaryOp::tag);                                          \
    }

KTN_MAKE_DSL_BINARY_OPERATOR(+, ADD)
KTN_MAKE_DSL_BINARY_OPERATOR(-, SUB)
KTN_MAKE_DSL_BINARY_OPERATOR(*, MUL)
KTN_MAKE_DSL_BINARY_OPERATOR(/, DIV)
KTN_MAKE_DSL_BINARY_OPERATOR(%, MOD)
KTN_MAKE_DSL_BINARY_OPERATOR(&, BIT_AND)
KTN_MAKE_DSL_BINARY_OPERATOR(|, BIT_OR)
KTN_MAKE_DSL_BINARY_OPERATOR(^, BIT_XOR)
KTN_MAKE_DSL_BINARY_OPERATOR(<<, SHL)
KTN_MAKE_DSL_BINARY_OPERATOR(>>, SHR)
KTN_MAKE_DSL_BINARY_OPERATOR(&&, AND)
KTN_MAKE_DSL_BINARY_OPERATOR(||, OR)
KTN_MAKE_DSL_BINARY_OPERATOR(==, EQUAL)
KTN_MAKE_DSL_BINARY_OPERATOR(!=, NOT_EQUAL)
KTN_MAKE_DSL_BINARY_OPERATOR(<, LESS)
KTN_MAKE_DSL_BINARY_OPERATOR(<=, LESS_EQUAL)
KTN_MAKE_DSL_BINARY_OPERATOR(>, GREATER)
KTN_MAKE_DSL_BINARY_OPERATOR(>=, GREATER_EQUAL)

#undef KTN_MAKE_DSL_BINARY_OPERATOR

#define KTN_MAKE_DSL_ASSIGN_OP(op)                                                    \
    template<typename Lhs, typename Rhs>                                              \
    requires requires {                                                               \
        std::declval<Lhs &>() op## = std::declval<katana::expr_value_t<Rhs>>();       \
    }                                                                                 \
    void operator+=(katana::Var<Lhs> lhs, Rhs &&rhs) {                                \
        auto x = lhs op std::forward<Rhs>(rhs);                                       \
        katana::FunctionBuilder::current()->assign(lhs.expression(), x.expression()); \
    }

KTN_MAKE_DSL_ASSIGN_OP(+)
KTN_MAKE_DSL_ASSIGN_OP(-)
KTN_MAKE_DSL_ASSIGN_OP(*)
KTN_MAKE_DSL_ASSIGN_OP(/)
KTN_MAKE_DSL_ASSIGN_OP(|)
KTN_MAKE_DSL_ASSIGN_OP(%)
KTN_MAKE_DSL_ASSIGN_OP(&)
KTN_MAKE_DSL_ASSIGN_OP(>>)
KTN_MAKE_DSL_ASSIGN_OP(<<)
KTN_MAKE_DSL_ASSIGN_OP(^)

#undef KTN_MAKE_DSL_ASSIGN_OP