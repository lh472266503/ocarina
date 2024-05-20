//
// Created by Zero on 2024/5/20.
//

#pragma once

#include "basic_traits.h"
#include "scalar_func.h"
#include "math/constants.h"

namespace ocarina {
namespace detail {

template<typename T>
struct valid_vector_impl : public std::disjunction<
                               std::is_same<T, bool>,
                               std::is_same<T, float>,
                               std::is_same<T, int>,
                               std::is_same<T, char>,
                               std::is_same<T, short>,
                               std::is_same<T, ushort>,
                               std::is_same<T, uint64t>,
                               std::is_same<T, uchar>,
                               std::is_same<T, uint>> {};

};// namespace detail

template<typename T>
static constexpr auto valid_vector_v = detail::valid_vector_impl<std::remove_cvref_t<T>>::value;

}// namespace ocarina

namespace ocarina {

template<typename T>
struct Var;

template<typename T, size_t N, size_t... Indices>
struct Swizzle {
    static constexpr uint num_component = sizeof...(Indices);
    static_assert(num_component <= 4 && std::max({Indices...}) < N);

    template<typename Scalar>
    struct vec {
        using type = Vector<Scalar, num_component>;
    };

    template<typename Scalar>
    struct vec<ocarina::Var<Scalar>> {
        using type = ocarina::Var<Vector<Scalar, num_component>>;
    };

    using vec_type = typename vec<T>::type;
    using scalar_type = T;

private:
    template<size_t... index>
    void assign_to(vec_type &vec, std::index_sequence<index...>) const noexcept {
        ((vec[index] = data_[Indices]), ...);
    }

    template<typename U, size_t... index>
    void assign_from(const U &vec, std::index_sequence<index...>) noexcept {
        ((data_[Indices] = vec[index]), ...);
    }

public:
    ocarina::array<T, N> data_{};

public:
    Swizzle &operator=(const Vector<T, num_component> &vec) noexcept {
        assign_from(vec, std::make_index_sequence<num_component>());
        return *this;
    }

    template<typename U, size_t M, size_t... OtherIndices>
    Swizzle &operator=(const Swizzle<U, M, OtherIndices...> &other) {
        *this = other.to_vec();
        return *this;
    }

    [[nodiscard]] vec_type to_vec() const noexcept {
        vec_type ret;
        assign_to(ret, std::make_index_sequence<num_component>());
        return ret;
    }

    operator vec_type() const noexcept {
        return to_vec();
    }

    [[nodiscard]] vec_type to_vec() noexcept {
        vec_type ret;
        assign_to(ret, std::make_index_sequence<num_component>());
        return ret;
    }

    operator vec_type() noexcept {
        return to_vec();
    }

#define OC_MAKE_SWIZZLE_UNARY_OP(op)    \
    auto operator op() const noexcept { \
        return op to_vec();             \
    }
    OC_MAKE_SWIZZLE_UNARY_OP(+)
    OC_MAKE_SWIZZLE_UNARY_OP(-)
    OC_MAKE_SWIZZLE_UNARY_OP(!)
    OC_MAKE_SWIZZLE_UNARY_OP(~)

#undef OC_MAKE_SWIZZLE_UNARY_OP

#define OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(op)                                  \
                                                                              \
    template<typename U, size_t M, size_t... OtherIndices>                    \
    vec_type operator op(Swizzle<U, M, OtherIndices...> rhs) const noexcept { \
        return to_vec() op rhs.to_vec();                                      \
    }                                                                         \
                                                                              \
    template<typename Arg>                                                    \
    Swizzle &operator op##=(Arg && arg) noexcept {                            \
        auto tmp = *this;                                                     \
        *this = tmp op OC_FORWARD(arg);                                       \
        return *this;                                                         \
    }

    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(+)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(-)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(*)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(/)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(%)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(>>)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(<<)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(|)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(&)
    OC_MAKE_SWIZZLE_MEMBER_BINARY_OP(^)

#undef OC_MAKE_SWIZZLE_MEMBER_BINARY_OP

#define OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP(op, ...)                                  \
    template<typename U, size_t M, size_t... OtherIndices>                        \
    requires __VA_ARGS__                                                          \
    [[nodiscard]] auto operator op(Swizzle<U, M, OtherIndices...> rhs) noexcept { \
        return to_vec() op rhs.to_vec();                                          \
    }

    OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP(||, ocarina::is_all_boolean_v<T, U>)
    OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP(&&, ocarina::is_all_boolean_v<T, U>)
    OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP(==, ocarina::is_all_number_v<T, U>)
    OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP(!=, ocarina::is_all_number_v<T, U>)
    OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP(<, ocarina::is_all_number_v<T, U>)
    OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP(>, ocarina::is_all_number_v<T, U>)
    OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP(<=, ocarina::is_all_number_v<T, U>)
    OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP(>=, ocarina::is_all_number_v<T, U>)

#undef OC_MAKE_SWIZZLE_MEMBER_LOGIC_OP
};

}// namespace ocarina

namespace ocarina {

namespace detail {
template<typename T, size_t N>
struct VectorStorage {
    static_assert(always_false_v<T>, "Invalid vector storage");
};

template<typename T>
struct alignas(sizeof(T) * 2) VectorStorage<T, 2> {
    static_assert(valid_vector_v<T>, "Invalid vector type");

public:
    template<size_t... index>
    using swizzle_type = Swizzle<T, 2, index...>;

public:
    T x{}, y{};
    VectorStorage() : x{}, y{} {}
    explicit constexpr VectorStorage(T s) noexcept : x{s}, y{s} {}
    constexpr VectorStorage(T x, T y) noexcept : x{x}, y{y} {}
#include "swizzle_inl/swizzle2.inl.h"
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 3> {
    static_assert(valid_vector_v<T>, "Invalid vector type");

public:
    template<size_t... index>
    using swizzle_type = Swizzle<T, 3, index...>;

public:
    T x{}, y{}, z{};
    VectorStorage() : x{}, y{}, z{} {}
    explicit constexpr VectorStorage(T s) noexcept : x{s}, y{s}, z{s} {}
    constexpr VectorStorage(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
#include "swizzle_inl/swizzle3.inl.h"
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 4> {
    static_assert(valid_vector_v<T>, "Invalid vector type");

public:
    template<size_t... index>
    using swizzle_type = Swizzle<T, 4, index...>;

public:
    T x{}, y{}, z{}, w{};
    VectorStorage() : x{}, y{}, z{}, w{} {}
    explicit constexpr VectorStorage(T s) noexcept : x{s}, y{s}, z{s}, w{s} {}
    constexpr VectorStorage(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
#include "swizzle_inl/swizzle4.inl.h"
};
}// namespace detail

template<typename T, size_t N>
struct Vector_ : public detail::VectorStorage<T, N> {
    using detail::VectorStorage<T, N>::VectorStorage;
    using this_type = Vector_<T, N>;
    using vec_type = this_type;
    using scalar_type = T;
    static constexpr size_t dimension = N;
    template<typename U>
    explicit constexpr Vector_(U s) noexcept : Vector_(static_cast<T>(s)) {}
    [[nodiscard]] constexpr T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    [[nodiscard]] constexpr const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
    [[nodiscard]] constexpr T &at(size_t index) noexcept { return (&(this->x))[index]; }
    [[nodiscard]] constexpr const T &at(size_t index) const noexcept { return (&(this->x))[index]; }

#define OC_MAKE_UNARY_OP(op)                                                  \
    [[nodiscard]] friend constexpr auto operator op(this_type val) noexcept { \
        using R = Vector_<decltype(op T{}), N>;                               \
        return [&]<size_t... index>(std::index_sequence<index...>) {          \
            return R{op val.at(index)...};                                    \
        }(std::make_index_sequence<N>());                                     \
    }
    OC_MAKE_UNARY_OP(+)
    OC_MAKE_UNARY_OP(-)
    OC_MAKE_UNARY_OP(~)
    OC_MAKE_UNARY_OP(!)
#undef OC_MAKE_UNARY_OP

#define OC_MAKE_VECTOR_BINARY_OPERATOR(op, ...)                                               \
    template<typename U>                                                                      \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(this_type lhs,                            \
                                                    Vector_<U, N> rhs) noexcept {             \
        using ret_type = decltype(T {} op U{});                                               \
        return [&]<size_t... index>(std::index_sequence<index...>) {                          \
            return Vector_<ret_type, N>{(lhs[index] op rhs[index])...};                       \
        }(std::make_index_sequence<N>());                                                     \
    }                                                                                         \
    template<typename U, size_t M, size_t... Indices>                                         \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(Swizzle<U, M, Indices...> lhs,            \
                                                    this_type rhs) noexcept {                 \
        return lhs.to_vec() op rhs;                                                           \
    }                                                                                         \
    template<typename U, size_t M, size_t... Indices>                                         \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(this_type lhs,                            \
                                                    Swizzle<U, M, Indices...> rhs) noexcept { \
        return lhs op rhs.to_vec();                                                           \
    }                                                                                         \
    template<typename U>                                                                      \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(this_type lhs, U rhs) noexcept {          \
        return lhs op Vector_<U, N>{rhs};                                                     \
    }                                                                                         \
    template<typename U>                                                                      \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(T lhs, Vector_<U, N> rhs) noexcept {      \
        return this_type{lhs} op rhs;                                                         \
    }                                                                                         \
    template<typename U>                                                                      \
    requires __VA_ARGS__                                                                      \
    constexpr friend auto &operator op##=(this_type & lhs, Vector_<U, N> rhs) noexcept {      \
        lhs = lhs op rhs;                                                                     \
        return lhs;                                                                           \
    }                                                                                         \
    template<typename U>                                                                      \
    requires __VA_ARGS__                                                                      \
    constexpr friend decltype(auto) operator op##=(this_type &lhs, U rhs) noexcept {          \
        return (lhs op## = Vector_<U, N>{rhs});                                               \
    }

    OC_MAKE_VECTOR_BINARY_OPERATOR(+, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_BINARY_OPERATOR(-, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_BINARY_OPERATOR(*, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_BINARY_OPERATOR(/, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_BINARY_OPERATOR(%, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_BINARY_OPERATOR(>>, ocarina::is_all_integral_v<T, U>)
    OC_MAKE_VECTOR_BINARY_OPERATOR(<<, ocarina::is_all_integral_v<T, U>)
    OC_MAKE_VECTOR_BINARY_OPERATOR(|, ocarina::is_all_integral_v<T, U>)
    OC_MAKE_VECTOR_BINARY_OPERATOR(&, ocarina::is_all_integral_v<T, U>)
    OC_MAKE_VECTOR_BINARY_OPERATOR(^, ocarina::is_all_integral_v<T, U>)

#undef OC_MAKE_VECTOR_BINARY_OPERATOR

#define OC_MAKE_VECTOR_LOGIC_OPERATOR(op, ...)                                                \
    template<typename U>                                                                      \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(this_type lhs,                            \
                                                    Vector_<U, N> rhs) noexcept {             \
        return [&]<size_t... index>(std::index_sequence<index...>) {                          \
            return Vector_<bool, N>{lhs[index] op rhs[index]...};                             \
        }(std::make_index_sequence<N>());                                                     \
    }                                                                                         \
    template<typename U, size_t M, size_t... Indices>                                         \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(Swizzle<U, M, Indices...> lhs,            \
                                                    this_type rhs) noexcept {                 \
        return lhs.to_vec() op rhs;                                                           \
    }                                                                                         \
    template<typename U, size_t M, size_t... Indices>                                         \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(this_type lhs,                            \
                                                    Swizzle<U, M, Indices...> rhs) noexcept { \
        return lhs op rhs.to_vec();                                                           \
    }                                                                                         \
    template<typename U>                                                                      \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(this_type lhs, U rhs) noexcept {          \
        return lhs op Vector_<U, N>{rhs};                                                     \
    }                                                                                         \
    template<typename U>                                                                      \
    requires __VA_ARGS__                                                                      \
    [[nodiscard]] friend constexpr auto operator op(T lhs, Vector_<U, N> rhs) noexcept {      \
        return this_type{lhs} op rhs;                                                         \
    }

    OC_MAKE_VECTOR_LOGIC_OPERATOR(||, ocarina::is_all_boolean_v<T, U>)
    OC_MAKE_VECTOR_LOGIC_OPERATOR(&&, ocarina::is_all_boolean_v<T, U>)
    OC_MAKE_VECTOR_LOGIC_OPERATOR(==, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_LOGIC_OPERATOR(!=, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_LOGIC_OPERATOR(<, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_LOGIC_OPERATOR(>, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_LOGIC_OPERATOR(<=, ocarina::is_all_number_v<T, U>)
    OC_MAKE_VECTOR_LOGIC_OPERATOR(>=, ocarina::is_all_number_v<T, U>)

#undef OC_MAKE_VECTOR_LOGIC_OPERATOR

#define OC_MAKE_VECTOR_UNARY_FUNC(func)                                                      \
    [[nodiscard]] static constexpr decltype(auto) call_##func(const this_type &v) noexcept { \
        using ret_type = Vector_<decltype(func(v.x)), this_type::dimension>;                 \
        return [&]<size_t... index>(std::index_sequence<index...>) {                         \
            return ret_type{func(v.at(index))...};                                           \
        }(std::make_index_sequence<N>());                                                    \
    }

    OC_MAKE_VECTOR_UNARY_FUNC(rcp)
    OC_MAKE_VECTOR_UNARY_FUNC(abs)
    OC_MAKE_VECTOR_UNARY_FUNC(sqrt)
    OC_MAKE_VECTOR_UNARY_FUNC(sqr)
    OC_MAKE_VECTOR_UNARY_FUNC(sign)
    OC_MAKE_VECTOR_UNARY_FUNC(cos)
    OC_MAKE_VECTOR_UNARY_FUNC(sin)
    OC_MAKE_VECTOR_UNARY_FUNC(tan)
    OC_MAKE_VECTOR_UNARY_FUNC(cosh)
    OC_MAKE_VECTOR_UNARY_FUNC(sinh)
    OC_MAKE_VECTOR_UNARY_FUNC(tanh)
    OC_MAKE_VECTOR_UNARY_FUNC(log)
    OC_MAKE_VECTOR_UNARY_FUNC(log2)
    OC_MAKE_VECTOR_UNARY_FUNC(log10)
    OC_MAKE_VECTOR_UNARY_FUNC(exp)
    OC_MAKE_VECTOR_UNARY_FUNC(exp2)
    OC_MAKE_VECTOR_UNARY_FUNC(asin)
    OC_MAKE_VECTOR_UNARY_FUNC(acos)
    OC_MAKE_VECTOR_UNARY_FUNC(atan)
    OC_MAKE_VECTOR_UNARY_FUNC(asinh)
    OC_MAKE_VECTOR_UNARY_FUNC(acosh)
    OC_MAKE_VECTOR_UNARY_FUNC(atanh)
    OC_MAKE_VECTOR_UNARY_FUNC(floor)
    OC_MAKE_VECTOR_UNARY_FUNC(ceil)
    OC_MAKE_VECTOR_UNARY_FUNC(degrees)
    OC_MAKE_VECTOR_UNARY_FUNC(radians)
    OC_MAKE_VECTOR_UNARY_FUNC(round)
    OC_MAKE_VECTOR_UNARY_FUNC(isnan)
    OC_MAKE_VECTOR_UNARY_FUNC(isinf)
    OC_MAKE_VECTOR_UNARY_FUNC(fract)
    OC_MAKE_VECTOR_UNARY_FUNC(copysign)

#undef OC_MAKE_VECTOR_UNARY_FUNC

#define OC_MAKE_VECTOR_BINARY_FUNC(func)                                                         \
    OC_NODISCARD static constexpr decltype(auto) call_##func(const this_type &v,                 \
                                                             const this_type &u) noexcept {      \
        using ret_type = Vector_<std::remove_cvref_t<decltype(func(v.x, u.x))>,                  \
                                 this_type::dimension>;                                          \
        return [&]<size_t... index>(std::index_sequence<index...>) {                             \
            return ret_type{func(v[index], u[index])...};                                        \
        }(std::make_index_sequence<N>());                                                        \
    }                                                                                            \
    OC_NODISCARD static constexpr decltype(auto) call_##func(const this_type &v, T u) noexcept { \
        return call_##func(v, this_type{u});                                                     \
    }                                                                                            \
    OC_NODISCARD static constexpr decltype(auto) call_##func(const T &v, this_type u) noexcept { \
        return call_##func(this_type{v}, u);                                                     \
    }
    OC_MAKE_VECTOR_BINARY_FUNC(max)
    OC_MAKE_VECTOR_BINARY_FUNC(min)
    OC_MAKE_VECTOR_BINARY_FUNC(pow)
    OC_MAKE_VECTOR_BINARY_FUNC(atan2)
#undef OC_MAKE_VECTOR_BINARY_FUNC

    [[nodiscard]] static constexpr auto call_volume(this_type v) noexcept {
        return [&]<size_t... index>(std::index_sequence<index...>) {
            return ((v[index] * ...));
        }(std::make_index_sequence<N>());
    }

    [[nodiscard]] static constexpr auto call_dot(this_type u, this_type v) noexcept {
        return [&]<size_t... index>(std::index_sequence<index...>) {
            return ((v[index] * u[index]) + ...);
        }(std::make_index_sequence<N>());
    }

    [[nodiscard]] this_type call_fma(const this_type &t, const this_type &u, const this_type &v) noexcept {
        return [&]<size_t... index>(std::index_sequence<index...>) {
            return this_type{fma(t[index], u[index], v[index])...};
        }(std::make_index_sequence<N>());
    }
};

#define OC_MAKE_VECTOR_UNARY_FUNC(func)                      \
    template<typename T>                                     \
    requires is_vector_or_swizzle_v<T>                       \
    [[nodiscard]] decltype(auto) func(const T &v) noexcept { \
        return deduce_vec_t<T>::call_##func(v);              \
    }

OC_MAKE_VECTOR_UNARY_FUNC(rcp)
OC_MAKE_VECTOR_UNARY_FUNC(abs)
OC_MAKE_VECTOR_UNARY_FUNC(sqrt)
OC_MAKE_VECTOR_UNARY_FUNC(sqr)
OC_MAKE_VECTOR_UNARY_FUNC(sign)
OC_MAKE_VECTOR_UNARY_FUNC(cos)
OC_MAKE_VECTOR_UNARY_FUNC(sin)
OC_MAKE_VECTOR_UNARY_FUNC(tan)
OC_MAKE_VECTOR_UNARY_FUNC(cosh)
OC_MAKE_VECTOR_UNARY_FUNC(sinh)
OC_MAKE_VECTOR_UNARY_FUNC(tanh)
OC_MAKE_VECTOR_UNARY_FUNC(log)
OC_MAKE_VECTOR_UNARY_FUNC(log2)
OC_MAKE_VECTOR_UNARY_FUNC(log10)
OC_MAKE_VECTOR_UNARY_FUNC(exp)
OC_MAKE_VECTOR_UNARY_FUNC(exp2)
OC_MAKE_VECTOR_UNARY_FUNC(asin)
OC_MAKE_VECTOR_UNARY_FUNC(acos)
OC_MAKE_VECTOR_UNARY_FUNC(atan)
OC_MAKE_VECTOR_UNARY_FUNC(asinh)
OC_MAKE_VECTOR_UNARY_FUNC(acosh)
OC_MAKE_VECTOR_UNARY_FUNC(atanh)
OC_MAKE_VECTOR_UNARY_FUNC(floor)
OC_MAKE_VECTOR_UNARY_FUNC(ceil)
OC_MAKE_VECTOR_UNARY_FUNC(degrees)
OC_MAKE_VECTOR_UNARY_FUNC(radians)
OC_MAKE_VECTOR_UNARY_FUNC(round)
OC_MAKE_VECTOR_UNARY_FUNC(isnan)
OC_MAKE_VECTOR_UNARY_FUNC(isinf)
OC_MAKE_VECTOR_UNARY_FUNC(fract)
OC_MAKE_VECTOR_UNARY_FUNC(copysign)

#undef OC_MAKE_VECTOR_UNARY_FUNC

#define OC_MAKE_VECTOR_BINARY_FUNC(func)                                \
    template<typename T, typename U>                                    \
    OC_NODISCARD decltype(auto) func(const T &t, const U &u) noexcept { \
        using vec_type = op_vector_t<T, U>;                             \
        return vec_type::call_##func(t, u);                             \
    }

OC_MAKE_VECTOR_BINARY_FUNC(pow)
OC_MAKE_VECTOR_BINARY_FUNC(min)
OC_MAKE_VECTOR_BINARY_FUNC(max)
OC_MAKE_VECTOR_BINARY_FUNC(atan2)

#undef OC_MAKE_VECTOR_BINARY_FUNC

template<typename T>
requires is_vector_or_swizzle_v<T>
[[nodiscard]] constexpr auto volume(T val) noexcept {
    return deduce_vec_t<T>::call_volume(val);
}

template<typename T, typename U>
requires is_vector_or_swizzle_v<T>
[[nodiscard]] constexpr auto dot(T t, U u) noexcept {
    return deduce_vec_t<T>::call_dot(t, u);
}

template<typename T, typename U, typename V>
requires is_all_vector_or_swizzle_v<T, U, V>
[[nodiscard]] auto fma(T t, U u, V v) noexcept {
    return deduce_vec_t<T>::call_fma(t, u, v);
}

#define OC_MAKE_VECTOR_TYPES(T) \
    using T##2 = Vector<T, 2>;  \
    using T##3 = Vector<T, 3>;  \
    using T##4 = Vector<T, 4>;

OC_MAKE_VECTOR_TYPES(bool)
OC_MAKE_VECTOR_TYPES(float)
OC_MAKE_VECTOR_TYPES(int)
OC_MAKE_VECTOR_TYPES(char)
OC_MAKE_VECTOR_TYPES(short)
OC_MAKE_VECTOR_TYPES(ushort)
OC_MAKE_VECTOR_TYPES(uchar)
OC_MAKE_VECTOR_TYPES(uint)
OC_MAKE_VECTOR_TYPES(uint64t)

#undef OC_MAKE_VECTOR_TYPES

template<size_t N>
[[nodiscard]] constexpr bool any(Vector<bool, N> v) noexcept {
    return [&]<size_t... index>(std::index_sequence<index...>) {
        return (v[index] || ...);
    }(std::make_index_sequence<N>());
}
template<size_t N>
[[nodiscard]] constexpr bool all(Vector<bool, N> v) noexcept {
    return [&]<size_t... index>(std::index_sequence<index...>) {
        return (v[index] && ...);
    }(std::make_index_sequence<N>());
}
template<size_t N>
[[nodiscard]] constexpr bool none(Vector<bool, N> v) noexcept { return !any(v); }

#define OC_MAKE_SWIZZLE_LOGIC_FUNC(func)                                         \
    template<size_t N, size_t... Indices>                                        \
    [[nodiscard]] constexpr bool func(Swizzle<bool, N, Indices...> v) noexcept { \
        return func(v.to_vec());                                                 \
    }
OC_MAKE_SWIZZLE_LOGIC_FUNC(any)
OC_MAKE_SWIZZLE_LOGIC_FUNC(all)
OC_MAKE_SWIZZLE_LOGIC_FUNC(none)

#undef OC_MAKE_SWIZZLE_LOGIC_FUNC
}// namespace ocarina