//
// Created by Zero on 25/04/2022.
//

#pragma once

#include "core/header.h"
#include <cstdint>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace sycamore {

template<typename... T>
struct always_false : std::false_type {};

template<typename... T>
constexpr auto always_false_v = always_false<T...>::value;

template<typename... T>
struct always_true : std::true_type {};

template<typename... T>
constexpr auto always_true_v = always_true<T...>::value;

template<typename T>
requires std::is_enum_v<T> SCM_NODISCARD constexpr auto to_underlying(T e) noexcept {
    return static_cast<std::underlying_type_t<T>>(e);
}

using uint = uint32_t;

template<typename T>
using is_integral = std::disjunction<
    std::is_same<std::remove_cvref_t<T>, int>,
    std::is_same<std::remove_cvref_t<T>, uint>>;

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
                                   std::is_floating_point<T>>;

template<typename T>
constexpr auto is_scalar_v = is_scalar<T>::value;

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

}// namespace sycamore