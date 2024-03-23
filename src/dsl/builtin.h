//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "var.h"
#include "expr.h"
#include "operators.h"
#include "math/base.h"
#include "ast/expression.h"

namespace ocarina {

#define OC_MAKE_BUILTIN_FUNC(func, type)                \
    [[nodiscard]] inline auto func() noexcept {         \
        return eval<type>(Function::current()->func()); \
    }
OC_MAKE_BUILTIN_FUNC(dispatch_idx, uint3)
OC_MAKE_BUILTIN_FUNC(block_idx, uint3)
OC_MAKE_BUILTIN_FUNC(thread_id, uint)
OC_MAKE_BUILTIN_FUNC(dispatch_id, uint)
OC_MAKE_BUILTIN_FUNC(thread_idx, uint3)
OC_MAKE_BUILTIN_FUNC(dispatch_dim, uint3)

template<typename DispatchIdx>
requires concepts::all_integral<vector_element_t<expr_value_t<DispatchIdx>>>
[[nodiscard]] auto dispatch_id(DispatchIdx &&idx) {
    if constexpr (is_vector2_expr_v<DispatchIdx>) {
        Uint3 dim = dispatch_dim();
        return idx.y * dim.x + idx.x;
    } else if constexpr (is_vector3_expr_v<DispatchIdx>) {
        Uint3 dim = dispatch_dim();
        return (dim.x * dim.y) * idx.z + dim.x * idx.y + idx.x;
    } else {
        static_assert(always_false_v<DispatchIdx>);
    }
}

template<typename DispatchId>
requires ocarina::is_integral_expr_v<DispatchId>
[[nodiscard]] auto dispatch_idx(DispatchId &&id) {
    Uint2 dim = dispatch_dim().xy();
    return make_uint2(id % dim.x, id / dim.x);
}

#undef OC_MAKE_BUILTIN_FUNC

#define OC_MAKE_LOGIC_FUNC(func, tag)                                             \
    template<typename T>                                                          \
    requires is_bool_vector_expr_v<T> || is_dynamic_array_v<T>                    \
    OC_NODISCARD auto                                                             \
    func(const T &t) noexcept {                                                   \
        auto expr = Function::current()->call_builtin(Type::of<bool>(),           \
                                                      CallOp::tag, {OC_EXPR(t)}); \
        return eval<bool>(expr);                                                  \
    }

OC_MAKE_LOGIC_FUNC(all, ALL)
OC_MAKE_LOGIC_FUNC(any, ANY)
OC_MAKE_LOGIC_FUNC(none, NONE)

#undef OC_MAKE_LOGIC_FUNC

/// used for dsl scalar vector or matrix
template<typename U, typename T, typename F>
requires(any_dsl_v<U, T, F> && std::is_same_v<expr_value_t<T>, expr_value_t<F>> &&
         vector_dimension_v<expr_value_t<T>> == vector_dimension_v<expr_value_t<F>> &&
         is_all_basic_expr_v<U, T, F>)
OC_NODISCARD auto select(U &&pred, T &&t, F &&f) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::SELECT,
                                                  {OC_EXPR(pred), OC_EXPR(t), OC_EXPR(f)});
    return eval<T>(expr);
}

/// used for dsl structure
template<typename U, typename T, typename F>
requires(std::is_same_v<expr_value_t<U>, bool> &&
         is_dsl_v<U> && any_dsl_v<T, F> && !is_basic_v<expr_value_t<F>> && !is_basic_v<expr_value_t<T>> &&
         std::is_same_v<expr_value_t<T>, expr_value_t<F>>) &&
        none_dynamic_array_v<U, T, F>
OC_NODISCARD auto select(U &&pred, T &&t, F &&f) noexcept {
    auto expr = Function::current()->conditional(Type::of<expr_value_t<T>>(), OC_EXPR(pred), OC_EXPR(t), OC_EXPR(f));
    return eval<T>(expr);
}

/// used for dynamic array
template<typename P, typename T>
requires is_all_scalar_v<P, T>
[[nodiscard]] DynamicArray<T> select(const DynamicArray<P> &pred, const DynamicArray<T> &t, const DynamicArray<T> &f) noexcept {
    OC_ASSERT(t.size() == f.size() && t.size() == pred.size());
    auto expr = Function::current()->call_builtin(DynamicArray<T>::type(pred.size()),
                                                  CallOp::SELECT, {OC_EXPR(pred), OC_EXPR(t), OC_EXPR(f)});
    return eval_dynamic_array(DynamicArray<T>(pred.size(), expr));
}

/// used for dynamic array
template<typename P, typename T>
requires is_all_scalar_v<T>
[[nodiscard]] DynamicArray<T> select(const Var<P> &pred, const DynamicArray<T> &t, const DynamicArray<T> &f) noexcept {
    OC_ASSERT(t.size() == f.size());
    DynamicArray<P> pred_arr{t.size()};
    pred_arr = pred;
    return select(pred_arr, t, f);
}

/// used for dynamic array
template<typename P, typename T>
requires is_all_scalar_v<T>
[[nodiscard]] DynamicArray<T> select(const Var<P> &pred, const Var<T> &t, const DynamicArray<T> &f) noexcept {
    DynamicArray<T> arr(f.size());
    arr = t;
    return select(pred, arr, f);
}

/// used for dynamic array
template<typename P, typename T>
requires is_all_scalar_v<T>
[[nodiscard]] DynamicArray<T> select(const Var<P> &pred, const T &t, const DynamicArray<T> &f) noexcept {
    DynamicArray<T> arr(f.size());
    arr = t;
    return select(pred, arr, f);
}

/// used for dynamic array
template<typename P, typename T>
requires is_all_scalar_v<T>
[[nodiscard]] DynamicArray<T> select(const Var<P> &pred, const DynamicArray<T> &t, const Var<T> &f) noexcept {
    DynamicArray<T> arr(t.size());
    arr = f;
    return select(pred, t, arr);
}

/// used for dynamic array
template<typename P, typename T>
requires is_all_scalar_v<T>
[[nodiscard]] DynamicArray<T> select(const Var<P> &pred, const DynamicArray<T> &t, const T &f) noexcept {
    DynamicArray<T> arr(t.size());
    arr = f;
    return select(pred, t, arr);
}

#define OC_MAKE_TRIPLE_FUNC(func, tag)                                                                \
    template<typename T, typename A, typename B>                                                      \
    requires(any_dsl_v<T, A, B> && ocarina::is_same_expr_v<T, A, B> && none_dynamic_array_v<T, A, B>) \
    OC_NODISCARD auto                                                                                 \
    func(const T &t, const A &a, const B &b) noexcept {                                               \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),                    \
                                                      CallOp::tag,                                    \
                                                      {OC_EXPR(t), OC_EXPR(a), OC_EXPR(b)});          \
        return eval<expr_value_t<T>>(expr);                                                           \
    }

inline namespace dsl {
OC_MAKE_TRIPLE_FUNC(clamp, CLAMP)
OC_MAKE_TRIPLE_FUNC(lerp, LERP)
OC_MAKE_TRIPLE_FUNC(fma, FMA)
}// namespace dsl

#undef OC_MAKE_TRIPLE_FUNC

template<typename T>
requires(is_dsl_v<T> && is_signed_element_v<expr_value_t<T>>)
OC_NODISCARD auto abs(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::ABS, {OC_EXPR(t)});
    return eval<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T> && is_signed_element_v<expr_value_t<T>>)
OC_NODISCARD auto sign(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::SIGN, {OC_EXPR(t)});
    return eval<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T>)
OC_NODISCARD auto rcp(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::RCP, {OC_EXPR(t)});
    return eval<expr_value_t<T>>(expr);
}

template<typename T>
requires(is_dsl_v<T>)
OC_NODISCARD auto sqr(const T &t) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),
                                                  CallOp::SQR, {OC_EXPR(t)});
    return eval<expr_value_t<T>>(expr);
}

#define OC_MAKE_ARRAY_UNARY_FUNC(func, tag)                                                                        \
    template<typename T>                                                                                           \
    requires is_basic_v<T>                                                                                         \
    [[nodiscard]] DynamicArray<T> func(const DynamicArray<T> &t) noexcept {                                        \
        auto expr = Function::current()->call_builtin(DynamicArray<T>::type(t.size()), CallOp::tag, {OC_EXPR(t)}); \
        return eval_dynamic_array(DynamicArray<T>(t.size(), expr));                                                \
    }

OC_MAKE_ARRAY_UNARY_FUNC(abs, ABS)
OC_MAKE_ARRAY_UNARY_FUNC(rcp, RCP)
OC_MAKE_ARRAY_UNARY_FUNC(sign, SIGN)
OC_MAKE_ARRAY_UNARY_FUNC(sqr, SQR)

#undef OC_MAKE_ARRAY_UNARY_FUNC

#define OC_MAKE_UNARY_VECTOR_FUNC(func, tag)                                       \
    template<typename T>                                                           \
    requires(is_dsl_v<T> && is_vector_v<expr_value_t<T>>)                          \
    OC_NODISCARD auto                                                              \
    func(const T &t) noexcept {                                                    \
        using ret_type = decltype(func(std::declval<expr_value_t<T>>()));          \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(), \
                                                      CallOp::tag, {OC_EXPR(t)});  \
        return eval<ret_type>(expr);                                               \
    }

OC_MAKE_UNARY_VECTOR_FUNC(normalize, NORMALIZE)
OC_MAKE_UNARY_VECTOR_FUNC(length, LENGTH)
OC_MAKE_UNARY_VECTOR_FUNC(length_squared, LENGTH_SQUARED)

#undef OC_MAKE_UNARY_VECTOR_FUNC

#define OC_MAKE_MATRIX_FUNC(func, tag)                                                    \
    template<typename T>                                                                  \
    requires(is_dsl_v<T> && is_matrix_v<expr_value_t<T>>)                                 \
    OC_NODISCARD auto                                                                     \
    func(const T &m) noexcept {                                                           \
        using ret_type = decltype(func(std::declval<expr_value_t<T>>()));                 \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<ret_type>>(), \
                                                      CallOp::tag, {OC_EXPR(m)});         \
        return eval<expr_value_t<ret_type>>(expr);                                        \
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
    return eval<expr_value_t<T>>(expr);
}

#define OC_MAKE_BINARY_VECTOR_FUNC(func, tag)                                                              \
    template<typename T, typename U>                                                                       \
    requires(any_dsl_v<T, U> && is_vector_same_dimension_v<expr_value_t<U>, expr_value_t<T>>)              \
    OC_NODISCARD auto                                                                                      \
    func(const T &t, const U &u) noexcept {                                                                \
        using ret_type = decltype(func(std::declval<expr_value_t<T>>(), std::declval<expr_value_t<U>>())); \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<ret_type>>(),                  \
                                                      CallOp::tag, {OC_EXPR(t), OC_EXPR(u)});              \
        return eval<expr_value_t<ret_type>>(expr);                                                         \
    }
OC_MAKE_BINARY_VECTOR_FUNC(dot, DOT)
OC_MAKE_BINARY_VECTOR_FUNC(distance, DISTANCE)
OC_MAKE_BINARY_VECTOR_FUNC(distance_squared, DISTANCE_SQUARED)

#undef OC_MAKE_BINARY_VECTOR_FUNC

template<typename... Args>
requires(any_dsl_v<Args...> &&
         is_all_float_element_expr_v<Args...> &&
         is_vector_same_dimension_v<expr_value_t<Args>...>)
OC_NODISCARD auto face_forward(Args &&...args) noexcept {
    using ret_ty = decltype(face_forward(std::declval<expr_value_t<Args>>()...));
    auto expr = Function::current()->call_builtin(Type::of<ret_ty>(),
                                                  CallOp::FACE_FORWARD, {OC_EXPR(args)...});
    return eval<ret_ty>(expr);
}

template<typename A>
requires(is_all_float_element_expr_v<A> &&
         is_all_float_vector3_v<expr_value_t<A>>)
void coordinate_system(const A &a, Var<float3> &b, Var<float3> &c) noexcept {
    auto expr = Function::current()->call_builtin(Type::of<expr_value_t<A>>(),
                                                  CallOp::COORDINATE_SYSTEM, {OC_EXPR(a), OC_EXPR(b), OC_EXPR(c)});
    Function::current()->expr_statement(expr);
}

#define OC_MAKE_FLOATING_BUILTIN_FUNC(func, tag)                                                                   \
    template<typename T>                                                                                           \
    requires(is_dsl_v<T> && is_float_element_expr_v<T>)                                                            \
    OC_NODISCARD auto                                                                                              \
    func(const T &t) noexcept {                                                                                    \
        using ret_type = decltype(func(std::declval<expr_value_t<T>>()));                                          \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),                                 \
                                                      CallOp::tag, {OC_EXPR(t)});                                  \
        return eval<expr_value_t<ret_type>>(expr);                                                                 \
    }                                                                                                              \
    template<typename T>                                                                                           \
    requires is_basic_v<T>                                                                                         \
    OC_NODISCARD DynamicArray<T> func(const DynamicArray<T> &t) noexcept {                                         \
        auto expr = Function::current()->call_builtin(DynamicArray<T>::type(t.size()), CallOp::tag, {OC_EXPR(t)}); \
        return eval_dynamic_array(DynamicArray<T>(t.size(), expr));                                                \
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
OC_MAKE_FLOATING_BUILTIN_FUNC(cosh, COSH)
OC_MAKE_FLOATING_BUILTIN_FUNC(sinh, SINH)
OC_MAKE_FLOATING_BUILTIN_FUNC(tanh, TANH)
OC_MAKE_FLOATING_BUILTIN_FUNC(acos, ACOS)
OC_MAKE_FLOATING_BUILTIN_FUNC(asin, ASIN)
OC_MAKE_FLOATING_BUILTIN_FUNC(atan, ATAN)
OC_MAKE_FLOATING_BUILTIN_FUNC(asinh, ASINH)
OC_MAKE_FLOATING_BUILTIN_FUNC(acosh, ACOSH)
OC_MAKE_FLOATING_BUILTIN_FUNC(atanh, ATANH)
OC_MAKE_FLOATING_BUILTIN_FUNC(degrees, DEGREES)
OC_MAKE_FLOATING_BUILTIN_FUNC(radians, RADIANS)
OC_MAKE_FLOATING_BUILTIN_FUNC(ceil, CEIL)
OC_MAKE_FLOATING_BUILTIN_FUNC(round, ROUND)
OC_MAKE_FLOATING_BUILTIN_FUNC(floor, FLOOR)
OC_MAKE_FLOATING_BUILTIN_FUNC(sqrt, SQRT)
OC_MAKE_FLOATING_BUILTIN_FUNC(rsqrt, RSQRT)
OC_MAKE_FLOATING_BUILTIN_FUNC(isinf, IS_INF)
OC_MAKE_FLOATING_BUILTIN_FUNC(isnan, IS_NAN)
OC_MAKE_FLOATING_BUILTIN_FUNC(fract, FRACT)
OC_MAKE_FLOATING_BUILTIN_FUNC(saturate, SATURATE)

#undef OC_MAKE_FLOATING_BUILTIN_FUNC

#define OC_MAKE_BINARY_BUILTIN_FUNC(func, tag)                                                             \
    template<typename A, typename B>                                                                       \
    requires(any_dsl_v<A, B> &&                                                                            \
             is_basic_v<expr_value_t<A>> &&                                                                \
             is_basic_v<expr_value_t<B>> &&                                                                \
             is_same_expr_v<A, B>)                                                                         \
    OC_NODISCARD auto                                                                                      \
    func(const A &a, const B &b) noexcept {                                                                \
        using ret_type = decltype(func(std::declval<expr_value_t<A>>(), std::declval<expr_value_t<B>>())); \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<A>>(),                         \
                                                      CallOp::tag, {OC_EXPR(a), OC_EXPR(b)});              \
        return eval<expr_value_t<ret_type>>(expr);                                                         \
    }                                                                                                      \
    template<typename T>                                                                                   \
    requires(is_dsl_v<T>)                                                                                  \
    OC_NODISCARD auto                                                                                      \
    func(const T &a, const T &b) noexcept {                                                                \
        using ret_type = decltype(func(std::declval<expr_value_t<T>>(), std::declval<expr_value_t<T>>())); \
        auto expr = Function::current()->call_builtin(Type::of<expr_value_t<T>>(),                         \
                                                      CallOp::tag, {OC_EXPR(a), OC_EXPR(b)});              \
        return eval<expr_value_t<ret_type>>(expr);                                                         \
    }

OC_MAKE_BINARY_BUILTIN_FUNC(max, MAX)
OC_MAKE_BINARY_BUILTIN_FUNC(min, MIN)
OC_MAKE_BINARY_BUILTIN_FUNC(pow, POW)
OC_MAKE_BINARY_BUILTIN_FUNC(fmod, FMOD)
OC_MAKE_BINARY_BUILTIN_FUNC(mod, MOD)
OC_MAKE_BINARY_BUILTIN_FUNC(copysign, COPYSIGN)
OC_MAKE_BINARY_BUILTIN_FUNC(atan2, ATAN2)

#undef OC_MAKE_BINARY_BUILTIN_FUNC

#define OC_MAKE_VEC_MAKER_DIM(type, tag, dim)                                  \
    template<typename... Args>                                                 \
    requires(any_dsl_v<Args...> && requires {                                  \
        make_##type##dim(expr_value_t<Args>{}...);                             \
    })                                                                         \
    OC_NODISCARD auto make_##type##dim(const Args &...args) noexcept {         \
        auto expr = Function::current()->call_builtin(Type::of<type##dim>(),   \
                                                      CallOp::MAKE_##tag##dim, \
                                                      {OC_EXPR(args)...});     \
        return eval<type##dim>(expr);                                          \
    }

#define OC_MAKE_VEC_MAKER(type, tag)    \
    OC_MAKE_VEC_MAKER_DIM(type, tag, 2) \
    OC_MAKE_VEC_MAKER_DIM(type, tag, 3) \
    OC_MAKE_VEC_MAKER_DIM(type, tag, 4)

OC_MAKE_VEC_MAKER(int, INT)
OC_MAKE_VEC_MAKER(uint, UINT)
OC_MAKE_VEC_MAKER(float, FLOAT)
OC_MAKE_VEC_MAKER(bool, BOOL)
OC_MAKE_VEC_MAKER(uchar, UCHAR)

#undef OC_MAKE_VEC_MAKER_DIM
#undef OC_MAKE_VEC_MAKER

#define OC_MAKE_MATRIX(dim)                                                                                 \
    template<typename... Args>                                                                              \
    requires(any_dsl_v<Args...> && requires {                                                               \
        make_float##dim##x##dim(expr_value_t<Args>{}...);                                                   \
    })                                                                                                      \
    OC_NODISCARD auto make_float##dim##x##dim(const Args &...args) {                                        \
        auto expr = Function::current()->call_builtin(Type::of<float##dim##x##dim>(),                       \
                                                      CallOp::MAKE_FLOAT##dim##X##dim, {OC_EXPR(args)...}); \
        return eval<float##dim##x##dim>(expr);                                                              \
    }

OC_MAKE_MATRIX(2)
OC_MAKE_MATRIX(3)
OC_MAKE_MATRIX(4)

#undef OC_MAKE_MATRIX

template<typename Ret = void, typename... Args>
auto call(string_view func_name, Args &&...args) noexcept {
    if constexpr (std::is_void_v<Ret>) {
        const CallExpr *expr = Function::current()->call(nullptr, func_name, {OC_EXPR(args)...});
        Function::current()->expr_statement(expr);
    } else {
        const CallExpr *expr = Function::current()->call(Type::of<Ret>(), func_name, {OC_EXPR(args)...});
        return eval<Ret>(expr);
    }
}

template<typename T>
requires is_vector_v<expr_value_t<T>> || is_scalar_v<expr_value_t<T>>
[[nodiscard]] T zero_if_nan(T t) noexcept {
    return ocarina::select(ocarina::isnan(t), T{}, t);
}

template<typename T>
requires is_vector_v<expr_value_t<T>> || is_scalar_v<expr_value_t<T>>
[[nodiscard]] T zero_if_nan_inf(T t) noexcept {
    return ocarina::select(ocarina::isnan(t) || ocarina::isinf(t), T{}, t);
}

inline void unreachable() noexcept {
    Function::current()->expr_statement(Function::current()->call_builtin(nullptr, CallOp::UNREACHABLE, {}));
}

}// namespace ocarina