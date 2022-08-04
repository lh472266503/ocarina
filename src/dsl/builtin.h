//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "dsl/var.h"
#include "dsl/expr.h"
#include "ast/expression.h"

namespace ocarina {

#define OC_MAKE_BUILTIN_FUNC(func, type)                     \
    [[nodiscard]] inline Expr<type> func() noexcept {        \
        return make_expr<type>(Function::current()->func()); \
    }
OC_MAKE_BUILTIN_FUNC(dispatch_idx, uint3)
OC_MAKE_BUILTIN_FUNC(block_idx, uint3)
OC_MAKE_BUILTIN_FUNC(thread_id, uint)
OC_MAKE_BUILTIN_FUNC(dispatch_id, uint)
OC_MAKE_BUILTIN_FUNC(thread_idx, uint3)
OC_MAKE_BUILTIN_FUNC(dispatch_dim, uint3)

#undef OC_MAKE_BUILTIN_FUNC

#define OC_MAKE_LOGIC_FUNC(func, tag)                                             \
    template<typename T>                                                          \
    requires is_bool_vector_expr_v<T>                                             \
    OC_NODISCARD auto                                                             \
    func(const T &t) noexcept {                                                   \
        auto expr = Function::current()->call_builtin(Type::of<bool>(),           \
                                                      CallOp::tag, {OC_EXPR(t)}); \
        return make_expr<bool>(expr);                                             \
    }

OC_MAKE_LOGIC_FUNC(all, ALL)
OC_MAKE_LOGIC_FUNC(any, ANY)
OC_MAKE_LOGIC_FUNC(none, NONE)

#undef OC_MAKE_LOGIC_FUNC

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

#define OC_MAKE_TRIPLE_FUNC(func, tag)                                                       \
    template<typename T, typename A, typename B>                                             \
    requires(any_dsl_v<T, A, B> && ocarina::is_same_expr_v<T, A, B>)                         \
    OC_NODISCARD auto func(const T &t, const A &a, const B &b) noexcept {                    \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),           \
                                                      CallOp::tag,                           \
                                                      {OC_EXPR(t), OC_EXPR(a), OC_EXPR(b)}); \
        return make_expr<expr_value_t<T>>(expr);                                             \
    }

OC_MAKE_TRIPLE_FUNC(fma, FMA)
OC_MAKE_TRIPLE_FUNC(lerp, LERP)
OC_MAKE_TRIPLE_FUNC(clamp, CLAMP)

#undef OC_MAKE_TRIPLE_FUNC

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

#define OC_MAKE_UNARY_VECTOR_FUNC(func, tag)                                       \
    template<typename T>                                                           \
    requires(is_dsl_v<T> && is_vector_v<expr_value_t<T>>)                          \
    OC_NODISCARD auto func(const T &t) noexcept {                                  \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(), \
                                                      CallOp::tag, {OC_EXPR(t)});  \
        return make_expr<expr_value_t<T>>(expr);                                   \
    }

OC_MAKE_UNARY_VECTOR_FUNC(normalize, NORMALIZE)
OC_MAKE_UNARY_VECTOR_FUNC(length, LENGTH)
OC_MAKE_UNARY_VECTOR_FUNC(length_squared, LENGTH_SQUARED)

#undef OC_MAKE_UNARY_VECTOR_FUNC

#define OC_MAKE_MATRIX_FUNC(func, tag)                                             \
    template<typename T>                                                           \
    requires(is_dsl_v<T> && is_matrix_v<expr_value_t<T>>)                          \
    OC_NODISCARD auto func(const T &m) noexcept {                                  \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(), \
                                                      CallOp::tag, {OC_EXPR(m)});  \
        return make_expr<expr_value_t<T>>(expr);                                   \
    }

OC_MAKE_MATRIX_FUNC(determinant, DETERMINANT)
OC_MAKE_MATRIX_FUNC(transpose, TRANSPOSE)
OC_MAKE_MATRIX_FUNC(inverse, INVERSE)

#undef OC_MAKE_MATRIX_FUNC

template<typename T, typename U>
requires(any_dsl_v<T, U> && is_vector3_v<expr_value_t<T>> && is_vector3_v<expr_value_t<U>>)
OC_NODISCARD auto cross(const T &t, const U &u) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::CROSS, {OC_EXPR(t), OC_EXPR(u)});
    return make_expr<expr_value_t<T>>(expr);
}

#define OC_MAKE_BINARY_VECTOR_FUNC(func, tag)                                                 \
    template<typename T, typename U>                                                          \
    requires(any_dsl_v<T, U> && is_vector_same_dimension_v<expr_value_t<U>, expr_value_t<T>>) \
    OC_NODISCARD auto func(const T &t, const U &u) noexcept {                                 \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),            \
                                                      CallOp::tag, {OC_EXPR(t), OC_EXPR(u)}); \
        return make_expr<expr_value_t<T>>(expr);                                              \
    }

OC_MAKE_BINARY_VECTOR_FUNC(dot, DOT)
OC_MAKE_BINARY_VECTOR_FUNC(distance, DISTANCE)
OC_MAKE_BINARY_VECTOR_FUNC(distance_squared, DISTANCE_SQUARED)

#undef OC_MAKE_BINARY_VECTOR_FUNC

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

#define OC_MAKE_VEC2_MAKER(type, tag)                                                        \
    template<typename T>                                                                     \
    requires(is_dsl_v<T> && (is_scalar_expr_v<T> || is_vector_expr_v<T>))                    \
    OC_NODISCARD auto make_##type##2(const T &t) noexcept {                                  \
        auto expr = Function::current()->call_builtin(Type::of<type##2>(),                   \
                                                      CallOp::MAKE_##tag##2,                 \
                                                      {OC_EXPR(t)});                         \
        return make_expr<type##2>(expr);                                                     \
    }                                                                                        \
    template<typename A, typename B>                                                         \
    requires(any_dsl_v<A, B> && is_all_##type##_element_v<expr_value_t<A>, expr_value_t<B>>) \
    OC_NODISCARD auto make_##type##2(const A &a, const B &b) noexcept {                      \
        auto expr = Function::current()->call_builtin(Type::of<type##2>(),                   \
                                                      CallOp::MAKE_##tag##2,                 \
                                                      {OC_EXPR(a), OC_EXPR(b)});             \
        return make_expr<type##2>(expr);                                                     \
    }

OC_MAKE_VEC2_MAKER(int, INT)
OC_MAKE_VEC2_MAKER(float, FLOAT)
OC_MAKE_VEC2_MAKER(uint, UINT)
OC_MAKE_VEC2_MAKER(bool, BOOL)

#undef OC_MAKE_VEC2_MAKER

#define OC_MAKE_VEC3_MAKER(type, tag)                                                                            \
    template<typename T>                                                                                         \
    requires(is_dsl_v<T> && (is_scalar_expr_v<T> || is_vector_expr_v<T>))                                        \
    OC_NODISCARD auto make_##type##3(const T &t) noexcept {                                                      \
        auto expr = Function::current()->call_builtin(Type::of<type##3>(),                                       \
                                                      CallOp::MAKE_##tag##3,                                     \
                                                      {OC_EXPR(t)});                                             \
        return make_expr<type##3>(expr);                                                                         \
    }                                                                                                            \
    template<typename A, typename B>                                                                             \
    requires(any_dsl_v<A, B> &&                                                                                  \
             is_all_##type##_element_v<expr_value_t<A>, expr_value_t<B>> &&                                      \
             ((is_vector2_expr_v<A> && is_scalar_expr_v<B>) || (is_vector2_expr_v<B> && is_scalar_expr_v<A>)))   \
        OC_NODISCARD auto make_##type##3(const A &a, const B &b) noexcept {                                      \
        auto expr = Function::current()->call_builtin(Type::of<type##3>(),                                       \
                                                      CallOp::MAKE_##tag##3,                                     \
                                                      {OC_EXPR(a), OC_EXPR(b)});                                 \
        return make_expr<type##3>(expr);                                                                         \
    }                                                                                                            \
    template<typename A, typename B, typename C>                                                                 \
    requires(any_dsl_v<A, B, C> && is_all_##type##_element_v<expr_value_t<A>, expr_value_t<B>, expr_value_t<C>>) \
        OC_NODISCARD auto make_##type##3(const A &a, const B &b, const C &c) noexcept {                          \
        auto expr = Function::current()->call_builtin(Type::of<type##3>(),                                       \
                                                      CallOp::MAKE_##tag##3,                                     \
                                                      {OC_EXPR(a), OC_EXPR(b), OC_EXPR(c)});                     \
        return make_expr<type##3>(expr);                                                                         \
    }

OC_MAKE_VEC3_MAKER(int, INT)
OC_MAKE_VEC3_MAKER(float, FLOAT)
OC_MAKE_VEC3_MAKER(uint, UINT)
OC_MAKE_VEC3_MAKER(bool, BOOL)

#undef OC_MAKE_VEC3_MAKER

template<typename T>
requires(is_dsl_v<T> && (is_scalar_expr_v<T> || is_vector_expr_v<T>))
    OC_NODISCARD auto make_int4(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<int4>(),
                                                  CallOp::MAKE_INT4, {OC_EXPR(t)});
    return make_expr<int4>(expr);
}

template<typename T, typename U>
requires(any_dsl_v<T, U> &&
             is_all_int_element_v<expr_value_t<T>, expr_value_t<U>> &&
         ((is_vector3_expr_v<T> && is_scalar_expr_v<U>) ||
          (is_scalar_expr_v<T> && is_vector3_expr_v<U>) ||
          (is_vector2_expr_v<T> && is_vector2_expr_v<U>)))
    OC_NODISCARD auto make_int4(const T &t, const U &u) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<int4>(),
                                                  CallOp::MAKE_INT4,
                                                  {OC_EXPR(t), OC_EXPR(u)});
    return make_expr<int4>(expr);
}

template<typename A, typename B, typename C>
requires(any_dsl_v<A, B, C> &&
             is_all_int_element_v<expr_value_t<A>, expr_value_t<B>, expr_value_t<C>> &&
         ((is_vector2_expr_v<A> && is_scalar_expr_v<B> && is_scalar_expr_v<C>) ||
          (is_scalar_expr_v<A> && is_vector2_expr_v<B> && is_scalar_expr_v<C>) ||
          (is_scalar_expr_v<A> && is_scalar_expr_v<B> && is_vector2_expr_v<C>)))
    OC_NODISCARD auto make_int4(const A &a, const B &b, const C &c) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<int4>(),
                                                  CallOp::MAKE_INT4,
                                                  {OC_EXPR(a), OC_EXPR(b), OC_EXPR(c)});
    return make_expr<int4>(expr);
}

template<typename A, typename B, typename C, typename D>
requires(any_dsl_v<A, B, C, D> &&is_all_int_element_expr_v<A, B, C, D>)
    OC_NODISCARD auto make_int4(const A &a, const B &b, const C &c, const D &d) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<int4>(),
                                                  CallOp::MAKE_INT4,
                                                  {OC_EXPR(a), OC_EXPR(b),
                                                   OC_EXPR(c), OC_EXPR(d)});
    return make_expr<int4>(expr);
}

}// namespace ocarina