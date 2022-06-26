//
// Created by Zero on 19/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "core/basic_traits.h"
#include "expr_traits.h"
#include "ast/function.h"

namespace ocarina {

template<typename Lhs, typename Rhs>
inline void assign(Lhs &&lhs, Rhs &&rhs) noexcept;// implement in stmt.h

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(T &&x) noexcept;// implement in builtin.h

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> def(const Expression *expr) noexcept;// implement in builtin.h

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> def_expr(T &&x) noexcept;

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> def_expr(const Expression *expr) noexcept;

class Expression;
namespace detail {

template<typename T>
struct EnableSubscriptAccess {
};

template<typename T>
struct EnableGetMemberByIndex {

};

template<typename T>
struct EnableStaticCast {
    template<class Dest>
    [[nodiscard]] Var<Dest> cast() const noexcept {
        Var<Dest> ret;
        auto expr = Function::current()->cast(Type::of<Dest>(), CastOp::STATIC, static_cast<const T *>(this)->expression());
        assign(ret, expr);
        return ret;
    }
};

template<typename T>
struct EnableBitwiseCast {

};

#define OC_COMPUTABLE_COMMON(...)                                          \
private:                                                                   \
    const Expression *_expression{nullptr};                                \
                                                                           \
public:                                                                    \
    [[nodiscard]] auto expression() const noexcept { return _expression; } \
                                                                           \
protected:                                                                 \
    explicit Computable(const Expression *e) noexcept : _expression{e} {}  \
    Computable(Computable &&) noexcept = default;                          \
    Computable(const Computable &) noexcept = default;

template<typename T>
struct Computable
    : detail::EnableBitwiseCast<Computable<T>>,
      detail::EnableStaticCast<Computable<T>> {
    static_assert(is_scalar_v<T>);
    OC_COMPUTABLE_COMMON(T)
};

template<typename T>
struct Computable<Vector<T, 2>>
    : detail::EnableStaticCast<Computable<Vector<T, 2>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 2>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 2>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 2>>> {
    OC_COMPUTABLE_COMMON(Vector<T, 2>)
};

template<typename T>
struct Computable<Vector<T, 3>>
    : detail::EnableStaticCast<Computable<Vector<T, 3>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 3>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 3>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 3>>> {
    OC_COMPUTABLE_COMMON(Vector<T, 3>)
};

template<typename T>
struct Computable<Vector<T, 4>>
    : detail::EnableStaticCast<Computable<Vector<T, 4>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 4>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 4>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 4>>> {
    OC_COMPUTABLE_COMMON(Vector<T, 4>)
};

template<typename T, size_t N>
struct Computable<std::array<T, N>>
    : detail::EnableSubscriptAccess<std::array<T, N>>,
      detail::EnableGetMemberByIndex<std::array<T, N>> {
    OC_COMPUTABLE_COMMON(std::array<T, N>)
};

template<size_t N>
struct Computable<Matrix<N>>
    : detail::EnableGetMemberByIndex<Matrix<N>>,
      detail::EnableSubscriptAccess<Matrix<N>> {
    OC_COMPUTABLE_COMMON(Matrix<N>)
};

template<typename... T>
struct Computable<ocarina::tuple<T...>> {
    using Tuple = ocarina::tuple<T...>;
    OC_COMPUTABLE_COMMON(ocarina::tuple<T...>)
    //    template<size_t i>
    //    [[nodiscard]] auto get() const noexcept {
    //        using Elm = ocarina::tuple_element_t<i, Tuple>;
    //        return Computable<Elm>(ocarina::FunctionBuilder::current(Type::of<Elm>(), expression(), i));
    //    }
};
#undef OC_COMPUTABLE_COMMON

}// namespace detail

}// namespace ocarina