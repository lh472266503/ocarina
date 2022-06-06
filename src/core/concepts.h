//
// Created by Zero on 26/04/2022.
//

#pragma once

#include "basic_types.h"
#include "stl.h"

namespace nano::concepts {

struct Noncopyable {
    Noncopyable() noexcept = default;
    Noncopyable(const Noncopyable &) noexcept = delete;
    Noncopyable &operator=(const Noncopyable &) noexcept = delete;
    Noncopyable(Noncopyable &&) noexcept = default;
    Noncopyable &operator=(Noncopyable &&) noexcept = default;
};

template<typename T>
concept iterable = requires(T v) {
    v.begin();
    v.end();
};

template<typename T>
concept string_viewable = requires(T v) {
    nano::string_view{v};
};

template<typename T>
concept span_convertible = requires(T v) {
    nano::span{v};
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

template<typename A, typename B>
concept different = !same<A, B>;

#define NN_UNARY_OP_CONCEPT(op_name, op) \
    template<typename T>                  \
    concept op_name = requires(T t) {     \
        op t;                             \
    };
NN_UNARY_OP_CONCEPT(positive_able, +)
NN_UNARY_OP_CONCEPT(negative_able, -)
NN_UNARY_OP_CONCEPT(not_able, !)
NN_UNARY_OP_CONCEPT(bit_not_able, ~)
#undef NN_UNARY_OP_CONCEPT

#define NN_BINARY_OP_CONCEPT(op_name, op) \
    template<typename A, typename B>       \
    concept op_name = requires(A a, B b) { \
        a op b;                            \
    };

NN_BINARY_OP_CONCEPT(plus_able, +)
NN_BINARY_OP_CONCEPT(minus_able, -)
NN_BINARY_OP_CONCEPT(multiply_able, *)
NN_BINARY_OP_CONCEPT(divide_able, /)
NN_BINARY_OP_CONCEPT(mod_able, %)
NN_BINARY_OP_CONCEPT(bit_and_able, &)
NN_BINARY_OP_CONCEPT(bit_or_able, |)
NN_BINARY_OP_CONCEPT(bit_xor_able, ^)
NN_BINARY_OP_CONCEPT(shift_left_able, <<)
NN_BINARY_OP_CONCEPT(shift_right_able, >>)
NN_BINARY_OP_CONCEPT(and_able, &&)
NN_BINARY_OP_CONCEPT(or_able, ||)
NN_BINARY_OP_CONCEPT(equal_able, ==)
NN_BINARY_OP_CONCEPT(ne_able, !=)
NN_BINARY_OP_CONCEPT(lt_able, <)
NN_BINARY_OP_CONCEPT(gt_able, >)
NN_BINARY_OP_CONCEPT(ge_able, >=)
NN_BINARY_OP_CONCEPT(le_able, <=)

NN_BINARY_OP_CONCEPT(assign_able, =)
NN_BINARY_OP_CONCEPT(plus_assign_able, +=)
NN_BINARY_OP_CONCEPT(minus_assign_able, -=)
NN_BINARY_OP_CONCEPT(mult_assign_able, *=)
NN_BINARY_OP_CONCEPT(div_assign_able, /=)
NN_BINARY_OP_CONCEPT(mod_assign_able, %=)
NN_BINARY_OP_CONCEPT(bit_and_assign_able, &=)
NN_BINARY_OP_CONCEPT(bit_or_assign_able, |=)
NN_BINARY_OP_CONCEPT(bit_xor_assign_able, ^=)
NN_BINARY_OP_CONCEPT(shift_left_assign_able, <<=)
NN_BINARY_OP_CONCEPT(shift_right_assign_able, >>=)

#undef NN_BINARY_OP_CONCEPT

template<typename Lhs, typename Rhs>
concept access_able = requires(Lhs lhs, Rhs rhs) { lhs[rhs]; };

}// namespace nano::concepts