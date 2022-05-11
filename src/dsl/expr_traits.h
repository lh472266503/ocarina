//
// Created by Zero on 11/05/2022.
//

#pragma once

#include "core/stl.h"
#include <type_traits>
#include "core/concepts.h"
#include "ast/type.h"
#include "core/basic_traits.h"

namespace katana::dsl {

template<typename T>
struct Expr;

template<typename T>
struct Var;

namespace detail {

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

template<typename T>
struct prototype_to_var<T &> {
    using type = Var<T> &;
};

template<typename T>
struct prototype_to_var<const T &> {
    using type = const Var<T> &;
};

template<typename T>
struct prototype_to_callable_invocation {
    using type = Expr<T>;
};

template<typename T>
struct prototype_to_callable_invocation<const T &> {
    using type = Expr<T>;
};

template<typename T>
struct Ref;

template<typename T>
struct expr_value_impl {
    using type = T;
};

template<typename T>
struct expr_value_impl<Expr<T>> {
    using type = T;
};

template<typename T>
struct expr_value_impl<Ref<T>> {
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
struct is_dsl_impl<Ref<T>> : std::true_type {};

template<typename T>
struct is_dsl_impl<Expr<T>> : std::true_type {};

template<typename T>
struct is_dsl_impl<Var<T>> : std::true_type {};

}// namespace detail

template<typename T>
using is_dsl = typename detail::is_dsl_impl<std::remove_cvref_t<T>>::type;

template<typename T>
constexpr auto is_dsl_v = is_dsl<T>::value;

template<typename... T>
using any_dsl = std::disjunction<is_dsl<T>...>;

template<typename T>
constexpr auto any_dsl_v = any_dsl<T>::value;

template<typename... T>
using all_dsl = std::conjunction<is_dsl<T>...>;

template<typename... T>
constexpr auto all_dsl_v = all_dsl<T...>::value;

template<typename... T>
using is_same_expr = concepts::is_same<expr_value_t<T>...>;

template<typename... T>
constexpr auto is_same_expr_v = is_same_expr<T...>::value;

template<typename T>
using is_integral_expr = is_integral<T>;

template<typename T>
constexpr auto is_integral_expr_v = is_integral_expr<T>::value;

template<typename T>
using is_boolean_expr = is_boolean<T>;

template<typename T>
constexpr auto is_boolean_expr_v = is_boolean_expr<T>::value;

template<typename T>
using is_floating_point_expr = is_floating_point<expr_value_t<T>>;

template<typename T>
constexpr auto is_floating_point_expr_v = is_floating_point_expr<T>::value;

template<typename T>
using is_scalar_expr = is_scalar<expr_value_t<T>>;

template<typename T>
constexpr auto is_scalar_expr_v = is_scalar_expr<T>::value;

template<typename T>
using is_vector_expr = is_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector_expr_v = is_vector_expr<T>::value;

template<typename T>
using is_vector2_expr = is_vector2<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector2_expr_v = is_vector2_expr<T>::value;

template<typename T>
using is_vector3_expr = is_vector3<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector3_expr_v = is_vector3_expr<T>::value;

template<typename T>
using is_vector4_expr = is_vector4<expr_value_t<T>>;

template<typename T>
constexpr auto is_vector4_expr_v = is_vector4_expr<T>::value;

template<typename T>
using is_bool_vector_expr = is_bool_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_bool_vector_expr_v = is_bool_vector_expr<T>::value;

template<typename T>
using is_float_vector_expr = is_float_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_float_vector_expr_v = is_float_vector_expr<T>::value;

template<typename T>
using is_int_vector_expr = is_int_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_int_vector_expr_v = is_int_vector_expr<T>::value;

template<typename T>
using is_uint_vector_expr = is_uint_vector<expr_value_t<T>>;

template<typename T>
constexpr auto is_uint_vector_expr_v = is_uint_vector_expr<T>::value;

}// namespace katana::dsl
