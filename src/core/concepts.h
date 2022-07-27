//
// Created by Zero on 26/04/2022.
//

#pragma once

#include "basic_types.h"
#include "stl.h"

namespace ocarina::concepts {

struct Noncopyable {
    Noncopyable() noexcept = default;
    Noncopyable(const Noncopyable &) noexcept = delete;
    Noncopyable &operator=(const Noncopyable &) noexcept = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};

class Definable {
private:
    mutable bool _defined{false};

protected:
    Definable() = default;

public:
    [[nodiscard]] bool has_defined() const noexcept { return _defined; }
    void define() const noexcept { _defined = true; }
    void undef() const noexcept { _defined = false; }
};

template<typename T>
concept iterable = requires(T v) {
    v.begin();
    v.end();
};

template<typename T>
concept string_viewable = requires(T v) {
    ocarina::string_view{v};
};

template<typename T>
concept span_convertible = requires(T v) {
    ocarina::span{v};
};

template<typename T, typename... Args>
concept constructible = requires(Args... args) {
    T{args...};
};

template<typename Src, typename Dest>
concept static_convertible = requires(Src s) {
    static_cast<Dest>(s);
};

template<typename Src, typename Dest>
concept bitwise_convertible = sizeof(Src) >= sizeof(Dest);

template<typename Src, typename Dest>
concept reinterpret_convertible = requires(Src s) {
    reinterpret_cast<Dest *>(&s);
};

template<typename F, typename... Args>
concept invocable = std::is_invocable_v<F, Args...>;

template<typename Ret, typename F, typename... Args>
concept invocable_with_return = std::is_invocable_r_v<Ret, F, Args...>;

template<typename T>
concept pointer = std::is_pointer_v<T>;

template<typename T>
concept non_pointer = !std::is_pointer_v<T>;

template<typename T>
concept container = requires(T a) {
    a.begin();
    a.size();
};

template<typename T>
concept integral = is_integral_v<T>;

template<typename T>
concept scalar = is_scalar_v<T>;

template<typename T>
concept vector = is_vector_v<T>;

template<typename T>
concept vector2 = is_vector2_v<T>;

template<typename T>
concept vector3 = is_vector3_v<T>;

template<typename T>
concept vector4 = is_vector4_v<T>;

template<typename T>
concept bool_vector = is_bool_vector_v<T>;

template<typename T>
concept float_vector = is_float_vector_v<T>;

template<typename T>
concept int_vector = is_int_vector_v<T>;

template<typename T>
concept uint_vector = is_uint_vector_v<T>;

template<typename T>
concept matrix = is_matrix_v<T>;

template<typename T>
concept matrix2 = is_matrix2_v<T>;

template<typename T>
concept matrix3 = is_matrix3_v<T>;

template<typename T>
concept matrix4 = is_matrix4_v<T>;

template<typename T>
concept basic = is_basic_v<T>;

namespace detail {
    template<typename... T>
    struct all_same_impl : std::true_type {};

    template<typename First, typename... Other>
    struct all_same_impl<First, Other...> : std::conjunction<std::is_same<First, Other>...> {};
}// namespace detail

template<typename... T>
using is_same = detail::all_same_impl<T...>;

template<typename... T>
constexpr auto is_same_v = is_same<T...>::value;

template<typename... T>
concept same = is_same_v<T...>;

template<typename... T>
concept all_integral = (integral<T> && ...);

template<typename A, typename B>
concept different = !same<A, B>;

template<typename Lhs, typename Rhs>
concept access_able = requires(Lhs lhs, Rhs rhs) { lhs[rhs]; };

template<typename T>
concept bool_able = requires(T t) { bool(t); };

template<typename T>
concept switch_able = std::is_enum_v<T> || ocarina::is_integral_v<T>;

#define OC_UNARY_CHECK(T) \
    requires(T t) { op t; };

#define oc_positive_check(T) OC_UNARY_CHECK(+)
#define oc_negative_check(T) OC_UNARY_CHECK(-)
#define oc_not_check(T) OC_UNARY_CHECK(!)
#define oc_bit_not_check(T) OC_UNARY_CHECK(~)

#define OC_BINARY_CHECK(A, B, op) \
    requires(A a, B b) { a op b; }

#define oc_plus_check(A, B) OC_BINARY_CHECK(A, B, +)
#define oc_minus_check(A, B) OC_BINARY_CHECK(A, B, -)
#define OC_MULTIPLY_CHECK(A, B) OC_BINARY_CHECK(A, B, *)
#define oc_divide_check(A, B) OC_BINARY_CHECK(A, B, /)
#define oc_mode_check(A, B) OC_BINARY_CHECK(A, B, %)
#define oc_bit_and_check(A, B) OC_BINARY_CHECK(A, B, &)
#define oc_bit_or_check(A, B) OC_BINARY_CHECK(A, B, |)
#define oc_bit_xor_check(A, B) OC_BINARY_CHECK(A, B, ^)
#define oc_shift_left_check(A, B) OC_BINARY_CHECK(A, B, <<)
#define oc_shift_right_check(A, B) OC_BINARY_CHECK(A, B, >>)
#define oc_and_check(A, B) OC_BINARY_CHECK(A, B, &&)
#define oc_or_check(A, B) OC_BINARY_CHECK(A, B, ||)
#define oc_equal_check(A, B) OC_BINARY_CHECK(A, B, ==)
#define oc_ne_check(A, B) OC_BINARY_CHECK(A, B, !=)
#define oc_lt_check(A, B) OC_BINARY_CHECK(A, B, <)
#define oc_gt_check(A, B) OC_BINARY_CHECK(A, B, >)
#define oc_le_check(A, B) OC_BINARY_CHECK(A, B, <=)
#define oc_ge_check(A, B) OC_BINARY_CHECK(A, B, >=)
#define OC_ASSIGN_CHECK(A, B) OC_BINARY_CHECK(A, B, =)

#define oc_plus_assign_check(A, B) OC_BINARY_CHECK(A, B, +=)
#define oc_minus_assign_check(A, B) OC_BINARY_CHECK(A, B, -=)
#define oc_multiply_assign_check(A, B) OC_BINARY_CHECK(A, B, *=)
#define oc_divide_assign_check(A, B) OC_BINARY_CHECK(A, B, /=)
#define oc_mod_assign_check(A, B) OC_BINARY_CHECK(A, B, %=)
#define oc_bit_and_assign_check(A, B) OC_BINARY_CHECK(A, B, &=)
#define oc_bit_or_assign_check(A, B) OC_BINARY_CHECK(A, B, |=)
#define oc_bit_xor_assign_check(A, B) OC_BINARY_CHECK(A, B, ^=)
#define oc_shift_left_assign_check(A, B) OC_BINARY_CHECK(A, B, <<=)
#define oc_shift_right_assign_check(A, B) OC_BINARY_CHECK(A, B, >>=)

}// namespace ocarina::concepts