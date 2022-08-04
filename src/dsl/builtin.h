//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "dsl/var.h"
#include "dsl/expr.h"
#include "ast/expression.h"

namespace ocarina {

[[nodiscard]] inline Expr<uint3> dispatch_idx() noexcept {
    return make_expr<uint3>(Function::current()->dispatch_idx());
}

[[nodiscard]] inline Expr<uint3> block_idx() noexcept {
    return make_expr<uint3>(Function::current()->block_idx());
}

[[nodiscard]] inline Expr<uint> thread_id() noexcept {
    return make_expr<uint>(Function::current()->thread_id());
}

[[nodiscard]] inline Expr<uint> dispatch_id() noexcept {
    return make_expr<uint>(Function::current()->dispatch_id());
}

[[nodiscard]] inline Expr<uint3> thread_idx() noexcept {
    return make_expr<uint3>(Function::current()->thread_idx());
}

[[nodiscard]] inline Expr<uint3> dispatch_dim() noexcept {
    return make_expr<uint3>(Function::current()->dispatch_dim());
}

template<typename T>
requires is_bool_vector_expr_v<T>
OC_NODISCARD auto
all(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<bool>(),
                                                  CallOp::ALL, {OC_EXPR(t)});
    return make_expr<bool>(expr);
}

template<typename T>
requires is_bool_vector_expr_v<T>
OC_NODISCARD auto
any(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<bool>(),
                                                  CallOp::ANY, {OC_EXPR(t)});
    return make_expr<bool>(expr);
}

template<typename T>
requires is_bool_vector_expr_v<T>
OC_NODISCARD auto
none(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<bool>(),
                                                  CallOp::NONE, {OC_EXPR(t)});
    return make_expr<bool>(expr);
}

template<typename U, typename T, typename F>
requires(any_dsl_v<U, T, F> &&
         vector_dimension_v<expr_value_t<U>> == vector_dimension_v<expr_value_t<T>> &&
         vector_dimension_v<expr_value_t<U>> == vector_dimension_v<expr_value_t<F>>)
OC_NODISCARD auto select(U &&pred, T &&t, F &&f) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::SELECT,
                                                  {OC_EXPR(pred), OC_EXPR(t), OC_EXPR(f)});
    return make_expr<T>(expr);
}

template<typename T, typename A, typename B>
requires any_dsl_v<T, A, B>
OC_NODISCARD auto
clamp(const T &t, const A &a, const B &b) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::CLAMP,
                                                  {OC_EXPR(t), OC_EXPR(a), OC_EXPR(b)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T, typename A, typename B>
requires any_dsl_v<T, A, B>
OC_NODISCARD auto
lerp(const T &t, const A &a, const B &b) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::LERP,
                                                  {OC_EXPR(t), OC_EXPR(a), OC_EXPR(b)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T, typename A, typename B>
requires any_dsl_v<T, A, B>
OC_NODISCARD auto
fma(const T &t, const A &a, const B &b) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::FMA,
                                                  {OC_EXPR(t), OC_EXPR(a), OC_EXPR(b)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T> && is_signed_element_v<expr_value_t<T>>)
OC_NODISCARD auto abs(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::ABS, {OC_EXPR(t)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T>)
OC_NODISCARD auto rcp(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::RCP, {OC_EXPR(t)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T>)
OC_NODISCARD auto sqr(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::SQR, {OC_EXPR(t)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T> && is_vector_v<expr_value_t<T>>)
OC_NODISCARD auto length(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::LENGTH, {OC_EXPR(t)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T> && is_vector_v<expr_value_t<T>>)
OC_NODISCARD auto normalize(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::NORMALIZE, {OC_EXPR(t)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T> && is_vector_v<expr_value_t<T>>)
OC_NODISCARD auto length_squared(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::LENGTH_SQUARED, {OC_EXPR(t)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T> && is_matrix_v<expr_value_t<T>>)
OC_NODISCARD auto determinant(const T &m) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::DETERMINANT, {OC_EXPR(m)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T> && is_matrix_v<expr_value_t<T>>)
OC_NODISCARD auto transpose(const T &m) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::TRANSPOSE, {OC_EXPR(m)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T> && is_matrix_v<expr_value_t<T>>)
OC_NODISCARD auto inverse(const T &m) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::INVERSE, {OC_EXPR(m)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T, typename U>
requires(any_dsl_v<T, U> && is_vector3_v<expr_value_t<T>> && is_vector3_v<expr_value_t<U>>)
OC_NODISCARD auto cross(const T &t, const U &u) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::CROSS, {OC_EXPR(t), OC_EXPR(u)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T, typename U>
requires(any_dsl_v<T, U> && is_vector_same_dimension_v<expr_value_t<U>, expr_value_t<T>>)
OC_NODISCARD auto dot(const T &t, const U &u) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::DOT, {OC_EXPR(t), OC_EXPR(u)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T, typename U>
requires(any_dsl_v<T, U> && is_vector_same_dimension_v<expr_value_t<U>, expr_value_t<T>>)
OC_NODISCARD auto distance(const T &t, const U &u) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::DISTANCE, {OC_EXPR(t), OC_EXPR(u)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename T, typename U>
requires(any_dsl_v<T, U> && is_vector_same_dimension_v<expr_value_t<U>, expr_value_t<T>>)
OC_NODISCARD auto distance_squared(const T &t, const U &u) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::DISTANCE_SQUARED, {OC_EXPR(t), OC_EXPR(u)});
    return make_expr<expr_value_t<T>>(expr);
}

template<typename A, typename B, typename C>
requires(any_dsl_v<A, B, C> &&
         is_all_float_element_v<expr_value_t<A>, expr_value_t<B>, expr_value_t<C>> &&
         is_vector_same_dimension_v<expr_value_t<A>, expr_value_t<B>, expr_value_t<C>>)
OC_NODISCARD auto face_forward(const A &a, const B &b, const C &c) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<A>>(),
                                                  CallOp::FACE_FORWARD, {OC_EXPR(a), OC_EXPR(b), OC_EXPR(c)});
    return make_expr<expr_value_t<A>>(expr);
}

#define OC_MAKE_FLOATING_BUILTIN_FUNC(func, tag)                                   \
    template<typename T>                                                           \
    requires(is_dsl_v<T> && is_float_element_v<expr_value_t<T>>)                   \
    OC_NODISCARD auto func(const T &t) noexcept {                                  \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(), \
                                                      CallOp::tag, {OC_EXPR(t)});  \
        return make_expr<expr_value_t<T>>(expr);                                   \
    }

OC_MAKE_FLOATING_BUILTIN_FUNC(exp, EXP)
OC_MAKE_FLOATING_BUILTIN_FUNC(exp2, EXP2)
OC_MAKE_FLOATING_BUILTIN_FUNC(exp10, EXP10)
OC_MAKE_FLOATING_BUILTIN_FUNC(log, LOG)
OC_MAKE_FLOATING_BUILTIN_FUNC(log2, LOG2)
OC_MAKE_FLOATING_BUILTIN_FUNC(log10, LOG10)
OC_MAKE_FLOATING_BUILTIN_FUNC(cos, COS)
OC_MAKE_FLOATING_BUILTIN_FUNC(sin, SIN)
OC_MAKE_FLOATING_BUILTIN_FUNC(tan, TAN)
OC_MAKE_FLOATING_BUILTIN_FUNC(acos, ACOS)
OC_MAKE_FLOATING_BUILTIN_FUNC(asin, ASIN)
OC_MAKE_FLOATING_BUILTIN_FUNC(atan, ATAN)
OC_MAKE_FLOATING_BUILTIN_FUNC(degrees, DEGREES)
OC_MAKE_FLOATING_BUILTIN_FUNC(radians, RADIANS)
OC_MAKE_FLOATING_BUILTIN_FUNC(ceil, CEIL)
OC_MAKE_FLOATING_BUILTIN_FUNC(round, ROUND)
OC_MAKE_FLOATING_BUILTIN_FUNC(floor, FLOOR)
OC_MAKE_FLOATING_BUILTIN_FUNC(sqrt, SQRT)
OC_MAKE_FLOATING_BUILTIN_FUNC(rsqrt, RSQRT)
OC_MAKE_FLOATING_BUILTIN_FUNC(saturate, SATURATE)

#undef OC_MAKE_FLOATING_BUILTIN_FUNC

}// namespace ocarina