//
// Created by Zero on 11/05/2022.
//

#pragma once

#include "core/stl.h"
#include <type_traits>
#include "core/concepts.h"
#include "ast/type.h"
#include "core/basic_traits.h"

namespace ocarina {

template<typename T>
struct Var;

template<typename T>
struct Expr;

namespace detail {

template<typename T>
struct Computable;

/// var
template<typename T>
struct var_to_prototype {
    static_assert(always_false_v<T>, "Invalid type in function definition.");
};

template<typename T>
struct var_to_prototype<Var<T>> {
    using type = T;
};

template<typename T>
struct var_to_prototype<const Var<T> &> {
    using type = T;
};

template<typename T>
struct var_to_prototype<Var<T> &> {
    using type = T &;
};

template<typename T>
struct prototype_to_var {
    using type = Var<T>;
};

template<>
struct prototype_to_var<void> {
    using type = void;
};

template<typename T>
struct prototype_to_var<T &> {
    using type = Var<T> &;
};

template<typename T>
struct prototype_to_var<const T &> {
    using type = const Var<T> &;
};

template<typename T>
using prototype_to_var_t = typename prototype_to_var<T>::type;

template<typename T>
struct expr_value_impl {
    using type = T;
};

template<>
struct expr_value_impl<void> {
    using type = void;
};

template<typename T>
struct expr_value_impl<Expr<T>> {
    using type = T;
};

template<typename T>
struct expr_value_impl<Computable<T>> {
    using type = T;
};

template<typename T>
struct expr_value_impl<Var<T>> {
    using type = T;
};

}// namespace detail

template<typename T>
using expr_value = detail::expr_value_impl<std::remove_cvref_t<T>>;

template<typename T>
using expr_value_t = typename expr_value<T>::type;

template<typename T>
using vector_expr_element = vector_element<expr_value_t<T>>;

template<typename T>
using vector_expr_element_t = typename vector_expr_element<T>::type;

template<typename T>
using vector_expr_dimension = vector_dimension<expr_value_t<T>>;

template<typename T>
constexpr auto vector_expr_dimension_v = vector_expr_dimension<T>::value;

template<typename... T>
using is_vector_expr_same_dimension = is_vector_same_dimension<expr_value_t<T>...>;

template<typename... T>
constexpr auto is_vector_expr_same_dimension_v = is_vector_expr_same_dimension<T...>::value;

template<typename... T>
using is_vector_expr_same_element = concepts::is_same<vector_expr_element_t<T>...>;

template<typename... T>
constexpr auto is_vector_expr_same_element_v = is_vector_expr_same_element<T...>::value;

namespace detail {

template<typename T>
struct is_dsl_impl : std::false_type {};

template<typename T>
struct is_dsl_impl<Expr<T>> : std::true_type {};

template<typename T>
struct is_dsl_impl<Computable<T>> : std::true_type {};

template<typename T>
struct is_dsl_impl<Var<T>> : std::true_type {};

}// namespace detail

template<typename T>
struct is_var : std::false_type {};

template<typename T>
struct is_var<Var<T>> : std::true_type {};

template<typename T>
using is_var_v = typename is_var<T>::value;

template<typename T>
struct is_expr : std::false_type {};

template<typename T>
struct is_expr<Expr<T>> : std::true_type {};

template<typename T>
constexpr auto is_expr_v = is_expr<T>::value;

template<typename T>
using is_dsl = typename detail::is_dsl_impl<std::remove_cvref_t<T>>::type;

template<typename T>
constexpr auto is_dsl_v = is_dsl<T>::value;

template<typename... T>
using any_dsl = std::disjunction<is_dsl<T>...>;

template<typename... T>
constexpr auto any_dsl_v = any_dsl<T...>::value;

template<typename ...T>
constexpr auto none_dsl_v = !any_dsl_v<T...>;

template<typename... T>
using all_dsl = std::conjunction<is_dsl<T>...>;

template<typename... T>
constexpr auto all_dsl_v = all_dsl<T...>::value;

template<typename... T>
using is_same_expr = concepts::is_same<expr_value_t<T>...>;

template<typename... T>
constexpr auto is_same_expr_v = is_same_expr<T...>::value;

#define EXPR_TYPE_TRAITS(type)             \
    template<typename T>                   \
    using is_##type##_expr = is_##type<T>; \
    template<typename T>                   \
    constexpr auto is_##type##_expr_v = is_##type##_expr<T>::value;

EXPR_TYPE_TRAITS(integral)
EXPR_TYPE_TRAITS(boolean)
EXPR_TYPE_TRAITS(floating_point)
EXPR_TYPE_TRAITS(scalar)

#undef EXPR_TYPE_TRAITS

#define EXPR_DIMENSION_TRAITS(cls, dim)                          \
    template<typename T>                                         \
    using is_##cls##dim##_expr = is_##cls##dim<expr_value_t<T>>; \
    template<typename T>                                         \
    constexpr auto is_##cls##dim##_expr_v = is_##cls##dim##_expr<T>::value;

EXPR_DIMENSION_TRAITS(vector, )
EXPR_DIMENSION_TRAITS(vector, 2)
EXPR_DIMENSION_TRAITS(vector, 3)
EXPR_DIMENSION_TRAITS(vector, 4)

EXPR_DIMENSION_TRAITS(matrix, )
EXPR_DIMENSION_TRAITS(matrix, 2)
EXPR_DIMENSION_TRAITS(matrix, 3)
EXPR_DIMENSION_TRAITS(matrix, 4)

#undef EXPR_DIMENSION_TRAITS

#define EXPR_VECTOR_TYPE_TRAITS(type)                                    \
    template<typename T>                                                 \
    using is_##type##_vector_expr = is_##type##_vector<expr_value_t<T>>; \
    template<typename T>                                                 \
    constexpr auto is_##type##_vector_expr_v = is_##type##_vector_expr<T>::value;

EXPR_VECTOR_TYPE_TRAITS(bool)
EXPR_VECTOR_TYPE_TRAITS(float)
EXPR_VECTOR_TYPE_TRAITS(int)
EXPR_VECTOR_TYPE_TRAITS(uint)

#undef EXPR_VECTOR_TYPE_TRAITS

template<typename T>
class Buffer;

template<typename T>
class BufferView;

namespace detail {

template<typename T>
struct is_buffer_impl : std::false_type {};

template<typename T>
struct is_buffer_impl<Buffer<T>> : std::true_type {};

template<typename T>
struct is_buffer_view_impl : std::false_type {};

template<typename T>
struct is_buffer_view_impl<BufferView<T>> : std::true_type {};

template<typename T>
struct buffer_element_impl {
    using type = T;
};

template<typename T>
struct buffer_element_impl<Buffer<T>> {
    using type = T;
};

template<typename T>
struct buffer_element_impl<BufferView<T>> {
    using type = T;
};

}// namespace detail

template<typename T>
using is_buffer = detail::is_buffer_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_view = detail::is_buffer_view_impl<std::remove_cvref_t<T>>;

template<typename T>
using is_buffer_or_view = std::disjunction<is_buffer<T>, is_buffer_view<T>>;

template<typename T>
constexpr auto is_buffer_v = is_buffer<T>::value;

template<typename T>
constexpr auto is_buffer_view_v = is_buffer_view<T>::value;

template<typename T>
constexpr auto is_buffer_or_view_v = is_buffer_or_view<T>::value;

}// namespace ocarina
