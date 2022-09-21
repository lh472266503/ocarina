//
// Created by Zero on 16/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "dsl/expr.h"
#include "ast/function.h"
#include "ast/op.h"

#define OC_MAKE_DSL_UNARY_OPERATOR(op, tag)                                                     \
    template<typename T>                                                                        \
    requires ocarina::is_dsl_v<T> [                                                             \
        [nodiscard]] inline auto                                                                \
    operator op(T &&expr) noexcept {                                                            \
        using Ret = std::remove_cvref_t<decltype(op std::declval<ocarina::expr_value_t<T>>())>; \
        return ocarina::make_expr<Ret>(                                                         \
            ocarina::Function::current()->unary(                                                \
                ocarina::Type::of<Ret>(),                                                       \
                ocarina::UnaryOp::tag,                                                          \
                ocarina::detail::extract_expression(std::forward<T>(expr))));                   \
    }

OC_MAKE_DSL_UNARY_OPERATOR(+, POSITIVE)
OC_MAKE_DSL_UNARY_OPERATOR(-, NEGATIVE)
OC_MAKE_DSL_UNARY_OPERATOR(!, NOT)
OC_MAKE_DSL_UNARY_OPERATOR(~, BIT_NOT)

#undef OC_MAKE_DSL_UNARY_OPERATOR

#define OC_MAKE_DSL_BINARY_OPERATOR(op, tag, trait)                                                           \
    template<typename Lhs, typename Rhs>                                                                      \
    requires ocarina::any_dsl_v<Lhs, Rhs> &&                                                                  \
        ocarina::is_basic_v<ocarina::expr_value_t<Lhs>> &&                                                    \
        ocarina::is_basic_v<ocarina::expr_value_t<Rhs>>                                                       \
    [[nodiscard]] inline auto                                                                                 \
    operator op(Lhs &&lhs, Rhs &&rhs) noexcept {                                                              \
        using namespace std::string_view_literals;                                                            \
        static constexpr bool is_logic_op = #op == "||"sv || #op == "&&"sv;                                   \
        static constexpr bool is_bit_op = #op == "|"sv || #op == "&"sv || #op == "^"sv;                       \
        static constexpr bool is_bool_lhs = ocarina::is_boolean_expr_v<Lhs>;                                  \
        static constexpr bool is_bool_rhs = ocarina::is_boolean_expr_v<Rhs>;                                  \
        using NormalRet = std::remove_cvref_t<                                                                \
            decltype(std::declval<ocarina::expr_value_t<Lhs>>() op                                            \
                         std::declval<ocarina::expr_value_t<Rhs>>())>;                                        \
        using Ret = std::conditional_t<is_bool_lhs && is_logic_op, bool, NormalRet>;                          \
        return ocarina::make_expr<Ret>(ocarina::Function::current()->binary(                                  \
            ocarina::Type::of<Ret>(),                                                                         \
            ocarina::BinaryOp::tag,                                                                           \
            ocarina::detail::extract_expression(std::forward<Lhs>(lhs)),                                      \
            ocarina::detail::extract_expression(std::forward<Rhs>(rhs))));                                    \
    }                                                                                                         \
    namespace ocarina {                                                                                       \
    namespace detail {                                                                                        \
    template<typename Lhs, typename Rhs>                                                                      \
    requires none_dsl_v<Lhs, Rhs>                                                                             \
    decltype(std::declval<Lhs>() op std::declval<Rhs>()) trait##_func();                                      \
    template<typename Lhs, typename Rhs>                                                                      \
    requires any_dsl_v<Lhs, Rhs>                                                                              \
        Var<decltype(std::declval<expr_value_t<Lhs>>() op std::declval<expr_value_t<Rhs>>())> trait##_func(); \
    template<typename Lhs, typename Rhs>                                                                      \
    struct trait {                                                                                            \
        using type = decltype(detail::trait##_func<Lhs, Rhs>());                                              \
    };                                                                                                        \
    }                                                                                                         \
    template<typename... T>                                                                                   \
    using trait##_t = typename detail::trait<T...>::type;                                                     \
    };// namespace ocarina

OC_MAKE_DSL_BINARY_OPERATOR(+, ADD, add)
OC_MAKE_DSL_BINARY_OPERATOR(-, SUB, sub)
OC_MAKE_DSL_BINARY_OPERATOR(*, MUL, mul)
OC_MAKE_DSL_BINARY_OPERATOR(/, DIV, div)
OC_MAKE_DSL_BINARY_OPERATOR(%, MOD, mod)
OC_MAKE_DSL_BINARY_OPERATOR(&, BIT_AND, bit_and)
OC_MAKE_DSL_BINARY_OPERATOR(|, BIT_OR, bit_or)
OC_MAKE_DSL_BINARY_OPERATOR(^, BIT_XOR, bit_xor)
OC_MAKE_DSL_BINARY_OPERATOR(<<, SHL, shl)
OC_MAKE_DSL_BINARY_OPERATOR(>>, SHR, shr)
OC_MAKE_DSL_BINARY_OPERATOR(&&, AND, and_op)
OC_MAKE_DSL_BINARY_OPERATOR(||, OR, or_op)
OC_MAKE_DSL_BINARY_OPERATOR(==, EQUAL, equal)
OC_MAKE_DSL_BINARY_OPERATOR(!=, NOT_EQUAL, not_equal)
OC_MAKE_DSL_BINARY_OPERATOR(<, LESS, less)
OC_MAKE_DSL_BINARY_OPERATOR(<=, LESS_EQUAL, less_eq)
OC_MAKE_DSL_BINARY_OPERATOR(>, GREATER, greater)
OC_MAKE_DSL_BINARY_OPERATOR(>=, GREATER_EQUAL, greater_equal)

#undef OC_MAKE_DSL_BINARY_OPERATOR

#define OC_MAKE_DSL_ASSIGN_OP(op)                                                \
    template<typename Lhs, typename Rhs>                                         \
    requires requires {                                                          \
        std::declval<Lhs &>() op## = std::declval<ocarina::expr_value_t<Rhs>>(); \
    }                                                                            \
    void operator op##=(const ocarina::Var<Lhs> &lhs, Rhs &&rhs) {               \
        auto x = lhs op std::forward<Rhs>(rhs);                                  \
        ocarina::Function::current()->assign(lhs.expression(), x.expression());  \
    }

OC_MAKE_DSL_ASSIGN_OP(+)
OC_MAKE_DSL_ASSIGN_OP(-)
OC_MAKE_DSL_ASSIGN_OP(*)
OC_MAKE_DSL_ASSIGN_OP(/)
OC_MAKE_DSL_ASSIGN_OP(|)
OC_MAKE_DSL_ASSIGN_OP(%)
OC_MAKE_DSL_ASSIGN_OP(&)
OC_MAKE_DSL_ASSIGN_OP(>>)
OC_MAKE_DSL_ASSIGN_OP(<<)
OC_MAKE_DSL_ASSIGN_OP(^)

#undef OC_MAKE_DSL_ASSIGN_OP
