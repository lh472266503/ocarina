//
// Created by Zero on 25/04/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"

namespace ocarina {

#define OC_DEFINE_TEMPLATE_VALUE(template_name) \
    template<typename T>                        \
    constexpr auto template_name##_v = template_name<T>::value;

#define OC_DEFINE_TEMPLATE_VALUE_MULTI(template_name) \
    template<typename... T>                           \
    constexpr auto template_name##_v = template_name<T...>::value;

#define OC_DEFINE_TEMPLATE_TYPE(template_name) \
    template<typename T>                       \
    using template_name##_t = typename template_name<T>::type;

#define OC_DEFINE_TEMPLATE_TYPE_MULTI(template_name) \
    template<typename... T>                          \
    using template_name##_t = typename template_name<T...>::type;

template<typename... T>
struct always_false : std::false_type {};

template<typename... T>
constexpr auto always_false_v = always_false<T...>::value;

template<typename... T>
struct always_true : std::true_type {};

template<typename... T>
constexpr auto always_true_v = always_true<T...>::value;

template<typename T>
requires std::is_enum_v<T> [
    [nodiscard]] constexpr auto
to_underlying(T e) noexcept {
    return static_cast<std::underlying_type_t<T>>(e);
}

using uint = uint32_t;
using uchar = unsigned char;
using ushort = unsigned short;

template<typename T>
using is_integral = std::disjunction<
    std::is_same<std::remove_cvref_t<T>, int>,
    std::is_same<std::remove_cvref_t<T>, uint>,
    std::is_same<std::remove_cvref_t<T>, uchar>,
    std::is_same<std::remove_cvref_t<T>, char>,
    std::is_same<std::remove_cvref_t<T>, short>,
    std::is_same<std::remove_cvref_t<T>, ushort>,
    std::is_same<std::remove_cvref_t<T>, size_t>>;

template<typename T>
constexpr bool is_integral_v = is_integral<T>::value;

template<typename T>
using is_boolean = std::is_same<std::remove_cvref_t<T>, bool>;

template<typename T>
constexpr auto is_boolean_v = is_boolean<T>::value;

template<typename T>
using is_floating_point = std::is_same<std::remove_cvref_t<T>, float>;

template<typename T>
constexpr auto is_floating_point_v = is_floating_point<T>::value;

template<typename T>
using is_char = std::is_same<std::remove_cvref_t<T>, char>;

template<typename T>
using is_uchar = std::is_same<std::remove_cvref_t<T>, uchar>;

template<typename T>
using is_signed = std::disjunction<
    is_floating_point<T>,
    std::is_same<std::remove_cvref_t<T>, int>>;

template<typename T>
constexpr auto is_signed_v = is_signed<T>::value;

template<typename T>
using is_unsigned = std::is_same<std::remove_cvref_t<T>, uint>;

template<typename T>
constexpr auto is_unsigned_v = is_unsigned<T>::value;

template<typename T>
using is_scalar = std::disjunction<is_integral<T>,
                                   is_boolean<T>,
                                   is_char<T>,
                                   is_uchar<T>,
                                   ocarina::is_floating_point<T>>;

template<typename T>
constexpr auto is_scalar_v = is_scalar<T>::value;

template<typename T>
using is_number = std::disjunction<is_integral<T>,
                                   ocarina::is_floating_point<T>>;

template<typename T>
constexpr auto is_number_v = is_number<T>::value;

#define MAKE_ALL_TYPE_TRAITS(type)                           \
    template<typename... T>                                  \
    using is_all_##type = std::disjunction<is_##type<T>...>; \
    OC_DEFINE_TEMPLATE_VALUE_MULTI(is_all_##type)

MAKE_ALL_TYPE_TRAITS(scalar)
MAKE_ALL_TYPE_TRAITS(number)
MAKE_ALL_TYPE_TRAITS(integral)
MAKE_ALL_TYPE_TRAITS(floating_point)
MAKE_ALL_TYPE_TRAITS(boolean)

#undef MAKE_ALL_TYPE_TRAITS

template<typename T, size_t N>
struct Vector;

template<size_t N>
struct Matrix;

namespace detail {

template<typename T, size_t N = 0u>
struct is_vector_impl : std::false_type {};

template<typename T, size_t N>
struct is_vector_impl<Vector<T, N>, N> : std::true_type {};

template<typename T, size_t N>
struct is_vector_impl<Vector<T, N>, 0u> : std::true_type {};

template<typename T, size_t N = 0u>
struct is_matrix_impl : std::false_type {};

template<size_t N>
struct is_matrix_impl<Matrix<N>, N> : std::true_type {};

template<size_t N>
struct is_matrix_impl<Matrix<N>, 0u> : std::true_type {};

template<typename T>
struct vector_element_impl {
    using type = T;
};

template<typename T, size_t N>
struct vector_element_impl<Vector<T, N>> {
    using type = T;
};

template<typename T>
struct vector_dimension_impl {
    static constexpr auto value = static_cast<size_t>(1u);
};

template<typename T, size_t N>
struct vector_dimension_impl<Vector<T, N>> {
    static constexpr auto value = N;
};

template<typename T>
struct matrix_dimension_impl {
    static constexpr auto value = static_cast<size_t>(1u);
};

template<size_t N>
struct matrix_dimension_impl<Matrix<N>> {
    static constexpr auto value = N;
};

template<typename U, typename V>
struct is_vector_same_dimension_impl : std::false_type {};

template<typename U, typename V, size_t N>
struct is_vector_same_dimension_impl<Vector<U, N>, Vector<V, N>> : std::true_type {};

template<typename... T>
struct is_vector_all_same_dimension_impl : std::true_type {};

template<typename First, typename... Other>
struct is_vector_all_same_dimension_impl<First, Other...> : std::conjunction<is_vector_same_dimension_impl<First, Other>...> {};

}// namespace detail

template<typename... T>
using is_vector_same_dimension = detail::is_vector_all_same_dimension_impl<std::remove_cvref_t<T>...>;

template<typename... T>
constexpr auto is_vector_same_dimension_v = is_vector_same_dimension<T...>::value;

template<typename T>
using vector_dimension = detail::vector_dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto vector_dimension_v = vector_dimension<T>::value;

template<typename T>
using matrix_dimension = detail::matrix_dimension_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto matrix_dimension_v = matrix_dimension<T>::value;

namespace detail {
template<typename T>
struct type_dimension_impl {
    static constexpr size_t value = 1;
};

template<typename T, size_t N>
struct type_dimension_impl<Vector<T, N>> {
    static constexpr size_t value = N;
};

template<size_t N>
struct type_dimension_impl<Matrix<N>> {
    static constexpr size_t value = N;
};

}// namespace detail

template<typename T>
using type_dimension = detail::type_dimension_impl<std::remove_cvref_t<T>>;
OC_DEFINE_TEMPLATE_VALUE(type_dimension)

template<typename T>
using vector_element = detail::vector_element_impl<std::remove_cvref_t<T>>;
OC_DEFINE_TEMPLATE_TYPE(vector_element)

//todo clear up the trait
template<typename T, size_t N = 0u>
using is_vector = detail::is_vector_impl<std::remove_cvref_t<T>, N>;

template<typename T>
using is_vector2 = is_vector<T, 2u>;

template<typename T>
using is_vector3 = is_vector<T, 3u>;

template<typename T>
using is_vector4 = is_vector<T, 4u>;

template<typename T, size_t N = 0u>
constexpr auto is_vector_v = is_vector<T, N>::value;

template<typename T>
constexpr auto is_vector2_v = is_vector2<T>::value;

template<typename T>
constexpr auto is_vector3_v = is_vector3<T>::value;

template<typename T>
constexpr auto is_vector4_v = is_vector4<T>::value;

#define OC_MAKE_IS_TYPE_VECTOR_DIM(type, dim)                                                                     \
    template<typename T>                                                                                          \
    using is_##type##_vector##dim = std::conjunction<is_vector##dim<T>, std::is_same<vector_element_t<T>, type>>; \
    OC_DEFINE_TEMPLATE_VALUE(is_##type##_vector##dim)                                                             \
    template<typename... T>                                                                                       \
    using is_all_##type##_vector##dim = std::conjunction<is_##type##_vector##dim<T>...>;                          \
    OC_DEFINE_TEMPLATE_VALUE_MULTI(is_all_##type##_vector##dim)

#define OC_MAKE_IS_TYPE_VECTOR(type)    \
    OC_MAKE_IS_TYPE_VECTOR_DIM(type, )  \
    OC_MAKE_IS_TYPE_VECTOR_DIM(type, 2) \
    OC_MAKE_IS_TYPE_VECTOR_DIM(type, 3) \
    OC_MAKE_IS_TYPE_VECTOR_DIM(type, 4)

OC_MAKE_IS_TYPE_VECTOR(bool)
OC_MAKE_IS_TYPE_VECTOR(int)
OC_MAKE_IS_TYPE_VECTOR(uint)
OC_MAKE_IS_TYPE_VECTOR(float)
OC_MAKE_IS_TYPE_VECTOR(uchar)
OC_MAKE_IS_TYPE_VECTOR(char)

#undef OC_MAKE_IS_TYPE_VECTOR

#undef OC_MAKE_IS_TYPE_VECTOR_DIM

template<typename T, size_t N = 0u>
using is_matrix = detail::is_matrix_impl<std::remove_cvref_t<T>, N>;

template<typename T>
using is_matrix2 = is_matrix<T, 2u>;

template<typename T>
using is_matrix3 = is_matrix<T, 3u>;

template<typename T>
using is_matrix4 = is_matrix<T, 4u>;

template<typename T, size_t N = 0u>
constexpr auto is_matrix_v = is_matrix<T, N>::value;

template<typename T>
constexpr auto is_matrix2_v = is_matrix2<T>::value;

template<typename T>
constexpr auto is_matrix3_v = is_matrix3<T>::value;

template<typename T>
constexpr auto is_matrix4_v = is_matrix4<T>::value;

template<typename T>
using is_basic = std::disjunction<is_scalar<T>, is_vector<T>, is_matrix<T>>;

template<typename T>
constexpr auto is_basic_v = is_basic<T>::value;

template<typename T>
using is_valid_buffer_element = std::conjunction<
    std::is_same<T, std::remove_cvref_t<T>>,
    std::is_trivially_copyable<T>,
    std::is_trivially_destructible<T>>;

template<typename T>
constexpr auto is_valid_buffer_element_v = is_valid_buffer_element<T>::value;

}// namespace ocarina