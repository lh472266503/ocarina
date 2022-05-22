//
// Created by Zero on 19/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "core/basic_traits.h"
#include "expr_traits.h"
#include "ast/function_builder.h"

namespace katana::detail {

template<typename T>
struct EnableSubscriptAccess {
    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] auto operator[](Index &&index) const noexcept {
        auto self = def<T>(static_cast<const T *>(this)->expression());
        using Element = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
        return def<Element>(katana::FunctionBuilder::current()->access(
            Type::of<Element>(), self.expression(),
            extract_expression(std::forward<Index>(index))));
    }

    //todo
    //    template<typename Index>
    //    requires is_integral_expr_v<Index>
    //    [[nodiscard]] auto operator[](Index &&index) &noexcept {
    //    }
};

template<typename T>
struct EnableGetMemberByIndex {
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        static_assert(i < dimension_v<expr_value_t<T>>);
        auto self = const_cast<T *>(static_cast<const T *>(this));
        return (*self)[static_cast<uint>(i)];
    }
};

template<typename T>
struct EnableStaticCast {
    template<typename Dest>
    requires concepts::static_convertible<expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using ExprDest = expr_value_t<Dest>;
        return def(katana::FunctionBuilder::current()->cast(Type::of<ExprDest>(), CastOp::STATIC, src));
    }
};

template<typename T>
struct EnableBitwiseCast {
    template<class Dest>
    requires concepts::bitwise_convertible<expr_value_t<T>, expr_value_t<Dest>>
    [[nodiscard]] auto bit_cast() const noexcept {
        auto src = def(*static_cast<const T *>(this));
        using ExprDest = expr_value_t<Dest>;
        return def(katana::FunctionBuilder::current()->cast(Type::of<ExprDest>(), CastOp::BITWISE, src));
    }
};

#define KTN_COMPUTABLE_COMMON(...)                                         \
private:                                                                   \
    const Expression *_expression{nullptr};                                \
                                                                           \
public:                                                                    \
    explicit Computable(const Expression *e) noexcept : _expression{e} {}  \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
    Computable(Computable &&) noexcept = default;                          \
    Computable(const Computable &) noexcept = default;

template<typename T>
struct Computable
    : detail::EnableBitwiseCast<T>,
      detail::EnableStaticCast<T> {
    static_assert(is_scalar_v<T>);
    KTN_COMPUTABLE_COMMON(T)
};

template<typename T>
struct Computable<Vector<T, 2>>
    : EnableStaticCast<Computable<Vector<T, 2>>>,
      EnableBitwiseCast<Computable<Vector<T, 2>>>,
      EnableGetMemberByIndex<Computable<Vector<T, 2>>>,
      EnableSubscriptAccess<Computable<Vector<T, 2>>> {
    KTN_COMPUTABLE_COMMON(Vector<T, 2>)
};

template<typename T>
struct Computable<Vector<T, 3>>
    : EnableStaticCast<Computable<Vector<T, 3>>>,
      EnableBitwiseCast<Computable<Vector<T, 3>>>,
      EnableGetMemberByIndex<Computable<Vector<T, 3>>>,
      EnableSubscriptAccess<Computable<Vector<T, 3>>> {
    KTN_COMPUTABLE_COMMON(Vector<T, 3>)
};

template<typename T>
struct Computable<Vector<T, 4>>
    : EnableStaticCast<Computable<Vector<T, 4>>>,
      EnableBitwiseCast<Computable<Vector<T, 4>>>,
      EnableGetMemberByIndex<Computable<Vector<T, 4>>>,
      EnableSubscriptAccess<Computable<Vector<T, 4>>> {
    KTN_COMPUTABLE_COMMON(Vector<T, 4>)
};

template<typename T, size_t N>
struct Computable<std::array<T, N>>
    : EnableSubscriptAccess<std::array<T, N>>,
      EnableGetMemberByIndex<std::array<T, N>> {
    KTN_COMPUTABLE_COMMON(std::array<T, N>)
};

template<size_t N>
struct Computable<Matrix<N>>
    : EnableGetMemberByIndex<Matrix<N>>,
      EnableSubscriptAccess<Matrix<N>> {
    KTN_COMPUTABLE_COMMON(Matrix<N>)
};

template<typename... T>
struct Computable<std::tuple<T...>> {
    using Tuple = std::tuple<T...>;
    KTN_COMPUTABLE_COMMON(std::tuple<T...>)
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        using Elm = std::tuple_element_t<i, Tuple>;
        return Computable<Elm>(katana::FunctionBuilder::current(Type::of<Elm>(), expression(), i));
    }
};

#undef KTN_COMPUTABLE_COMMON

}// namespace katana::detail