//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "expr.h"
#include "operators.h"
#include "core/concepts.h"
#include "math/base.h"
#include "ast/expression.h"
#include "var.h"

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

namespace detail {

template<typename T>
struct match_dsl_unary_func_impl : std::false_type {};

template<typename T>
struct match_dsl_unary_func_impl<Computable<T>> : std::true_type {};

template<typename T>
struct match_dsl_unary_func_impl<Var<T>> : std::true_type {};

template<typename T>
struct match_dsl_unary_func_impl<Expr<T>> : std::true_type {};

template<typename T, size_t N, size_t... Indices>
struct match_dsl_unary_func_impl<Swizzle<T, N, Indices...>> : std::true_type {};

}// namespace detail

template<typename T>
using match_dsl_unary_func = detail::match_dsl_unary_func_impl<std::remove_cvref_t<T>>;
OC_DEFINE_TEMPLATE_VALUE(match_dsl_unary_func)

namespace detail {
template<typename T>
struct deduce_var_impl {};

template<typename T>
struct deduce_var_impl<Computable<T>> {
    using type = Var<T>;
};

template<typename T>
struct deduce_var_impl<Expr<T>> {
    using type = Var<T>;
};

template<typename T>
struct deduce_var_impl<Var<T>> {
    using type = Var<T>;
};

template<typename T, size_t N, size_t... Indices>
struct deduce_var_impl<Swizzle<Var<T>, N, Indices...>> {
    using type = typename Swizzle<Var<T>, N, Indices...>::vec_type;
};

}// namespace detail

template<typename T>
using deduce_var = detail::deduce_var_impl<std::remove_cvref_t<T>>;
OC_DEFINE_TEMPLATE_TYPE(deduce_var)

#define OC_MAKE_DSL_UNARY_FUNC(func, tag)                                              \
    template<typename T>                                                               \
    requires match_dsl_unary_func_v<T>                                                 \
    OC_NODISCARD auto func(const T &arg) noexcept {                                    \
        return deduce_var_t<T>::call_##func(arg);                                      \
    }                                                                                  \
    template<typename T>                                                               \
    requires is_basic_v<T>                                                             \
    OC_NODISCARD DynamicArray<T> func(const DynamicArray<T> &t) noexcept {             \
        auto expr = Function::current()->call_builtin(DynamicArray<T>::type(t.size()), \
                                                      CallOp::tag, {OC_EXPR(t)});      \
        return eval_dynamic_array(DynamicArray<T>(t.size(), expr));                    \
    }

OC_MAKE_DSL_UNARY_FUNC(all, ALL)
OC_MAKE_DSL_UNARY_FUNC(any, ANY)
OC_MAKE_DSL_UNARY_FUNC(none, NONE)
OC_MAKE_DSL_UNARY_FUNC(rcp, RCP)
OC_MAKE_DSL_UNARY_FUNC(abs, ABS)
OC_MAKE_DSL_UNARY_FUNC(sign, SIGN)
OC_MAKE_DSL_UNARY_FUNC(sqr, SQR)
OC_MAKE_DSL_UNARY_FUNC(normalize, NORMALIZE)
OC_MAKE_DSL_UNARY_FUNC(length, LENGTH)
OC_MAKE_DSL_UNARY_FUNC(length_squared, LENGTH_SQUARED)

OC_MAKE_DSL_UNARY_FUNC(exp, EXP)
OC_MAKE_DSL_UNARY_FUNC(exp2, EXP2)
OC_MAKE_DSL_UNARY_FUNC(exp10, EXP10)
OC_MAKE_DSL_UNARY_FUNC(log, LOG)
OC_MAKE_DSL_UNARY_FUNC(log2, LOG2)
OC_MAKE_DSL_UNARY_FUNC(log10, LOG10)
OC_MAKE_DSL_UNARY_FUNC(cos, COS)
OC_MAKE_DSL_UNARY_FUNC(sin, SIN)
OC_MAKE_DSL_UNARY_FUNC(tan, TAN)
OC_MAKE_DSL_UNARY_FUNC(cosh, COSH)
OC_MAKE_DSL_UNARY_FUNC(sinh, SINH)
OC_MAKE_DSL_UNARY_FUNC(tanh, TANH)
OC_MAKE_DSL_UNARY_FUNC(acos, ACOS)
OC_MAKE_DSL_UNARY_FUNC(asin, ASIN)
OC_MAKE_DSL_UNARY_FUNC(atan, ATAN)
OC_MAKE_DSL_UNARY_FUNC(asinh, ASINH)
OC_MAKE_DSL_UNARY_FUNC(acosh, ACOSH)
OC_MAKE_DSL_UNARY_FUNC(atanh, ATANH)
OC_MAKE_DSL_UNARY_FUNC(degrees, DEGREES)
OC_MAKE_DSL_UNARY_FUNC(radians, RADIANS)
OC_MAKE_DSL_UNARY_FUNC(ceil, CEIL)
OC_MAKE_DSL_UNARY_FUNC(round, ROUND)
OC_MAKE_DSL_UNARY_FUNC(floor, FLOOR)
OC_MAKE_DSL_UNARY_FUNC(sqrt, SQRT)
OC_MAKE_DSL_UNARY_FUNC(rsqrt, RSQRT)
OC_MAKE_DSL_UNARY_FUNC(isinf, IS_INF)
OC_MAKE_DSL_UNARY_FUNC(isnan, IS_NAN)
OC_MAKE_DSL_UNARY_FUNC(fract, FRACT)
OC_MAKE_DSL_UNARY_FUNC(saturate, SATURATE)

OC_MAKE_DSL_UNARY_FUNC(determinant, DETERMINANT)
OC_MAKE_DSL_UNARY_FUNC(transpose, TRANSPOSE)
OC_MAKE_DSL_UNARY_FUNC(inverse, INVERSE)

#undef OC_MAKE_DSL_UNARY_FUNC

template<typename... Ts>
using match_dsl_basic_func = std::conjunction<any_device_type<Ts...>, match_basic_func<remove_device_t<Ts>...>>;
OC_DEFINE_TEMPLATE_VALUE_MULTI(match_dsl_basic_func)

#define OC_MAKE_DSL_BINARY_FUNC(func, tag)                                        \
    template<typename Lhs, typename Rhs>                                          \
    requires match_dsl_basic_func_v<Lhs, Rhs>                                     \
    OC_NODISCARD auto func(const Lhs &lhs, const Rhs &rhs) noexcept {             \
        static constexpr auto dimension = type_dimension_v<remove_device_t<Lhs>>; \
        using scalar_type = type_element_t<remove_device_t<Lhs>>;                 \
        using var_type = Var<general_vector_t<scalar_type, dimension>>;           \
        return var_type::call_##func(static_cast<swizzle_decay_t<Lhs>>(lhs),      \
                                     static_cast<swizzle_decay_t<Rhs>>(rhs));     \
    }

OC_MAKE_DSL_BINARY_FUNC(max, MAX)
OC_MAKE_DSL_BINARY_FUNC(min, MIN)
OC_MAKE_DSL_BINARY_FUNC(pow, POW)
OC_MAKE_DSL_BINARY_FUNC(fmod, FMOD)
OC_MAKE_DSL_BINARY_FUNC(mod, MOD)
OC_MAKE_DSL_BINARY_FUNC(copysign, COPYSIGN)
OC_MAKE_DSL_BINARY_FUNC(atan2, ATAN2)

OC_MAKE_DSL_BINARY_FUNC(cross, CROSS)
OC_MAKE_DSL_BINARY_FUNC(dot, DOT)
OC_MAKE_DSL_BINARY_FUNC(distance, DISTANCE)
OC_MAKE_DSL_BINARY_FUNC(distance_squared, DISTANCE_SQUARED)

#undef OC_MAKE_DSL_BINARY_FUNC

#define OC_MAKE_DSL_TRIPLE_FUNC(func, tag)                                      \
    template<typename A, typename B, typename C>                                \
    requires match_dsl_basic_func_v<A, B, C>                                    \
    [[nodiscard]] auto func(const A &a, const B &b, const C &c) noexcept {      \
        static constexpr auto dimension = type_dimension_v<remove_device_t<A>>; \
        using scalar_type = type_element_t<remove_device_t<A>>;                 \
        using var_type = Var<general_vector_t<scalar_type, dimension>>;         \
        return var_type::call_##func(static_cast<swizzle_decay_t<A>>(a),        \
                                     static_cast<swizzle_decay_t<B>>(b),        \
                                     static_cast<swizzle_decay_t<C>>(c));       \
    }

OC_MAKE_DSL_TRIPLE_FUNC(clamp, CLAMP)
OC_MAKE_DSL_TRIPLE_FUNC(lerp, LERP)
OC_MAKE_DSL_TRIPLE_FUNC(fma, FMA)

#undef OC_MAKE_TRIPLE_FUNC

template<typename U, typename T, typename F>
requires (match_basic_func_v<remove_device_t<T>, remove_device_t<F>> &&
         any_device_type_v<U,T,F>)
[[nodiscard]] auto select(const U &u, const T &t, const F &f) noexcept {
    static constexpr auto dimension = type_dimension_v<remove_device_t<T>>;
    using scalar_type = type_element_t<remove_device_t<T>>;
    using var_type = Var<general_vector_t<scalar_type, dimension>>;
    return var_type::call_select(static_cast<swizzle_decay_t<U>>(u),
                                 static_cast<swizzle_decay_t<T>>(t),
                                 static_cast<swizzle_decay_t<F>>(f));
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

template<typename A, typename B>
requires concepts::plus_able<expr_value_t<A>, expr_value_t<B>>
auto atomic_add(A &&a, B &&b) noexcept {
    const Expression *expr = Function::current()->call_builtin(Type::of<expr_value_t<A>>(),
                                                               CallOp::ATOMIC_ADD,
                                                               {OC_EXPR(a), OC_EXPR(b)});
    return eval<expr_value_t<A>>(expr);
}

template<typename A, typename B>
requires concepts::minus_able<expr_value_t<A>, expr_value_t<B>>
auto atomic_sub(A &&a, B &&b) noexcept {
    const Expression *expr = Function::current()->call_builtin(Type::of<expr_value_t<A>>(),
                                                               CallOp::ATOMIC_SUB,
                                                               {OC_EXPR(a), OC_EXPR(b)});
    return eval<expr_value_t<A>>(expr);
}

template<typename A, typename B>
requires concepts::assign_able<expr_value_t<A>, expr_value_t<B>>
auto atomic_exch(A &&a, B &&b) noexcept {
    const Expression *expr = Function::current()->call_builtin(Type::of<expr_value_t<A>>(),
                                                               CallOp::ATOMIC_EXCH,
                                                               {OC_EXPR(a), OC_EXPR(b)});
    return eval<expr_value_t<A>>(expr);
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