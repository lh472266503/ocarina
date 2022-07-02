//
// Created by Zero on 19/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "core/basic_traits.h"
#include "expr_traits.h"
#include "ast/function.h"
#include <utility>

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
[[nodiscard]] decltype(auto) extract_expression(T &&v) noexcept;

#define OC_EXPR(arg) ocarina::detail::extract_expression(OC_FORWARD(arg))

template<typename T>
struct EnableSubscriptAccess {

    using element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto operator[](Index &&index) const noexcept {
        const AccessExpr *expr = Function::current()->access(Type::of<element_type>(),
                                                             static_cast<const T *>(this)->expression(),
                                                             OC_EXPR(index));
        return Var<element_type>(expr);
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto &operator[](Index &&index) noexcept {
        auto f = Function::current();
        const AccessExpr *expr = f->access(Type::of<element_type>(),
                                           static_cast<const T *>(this)->expression(),
                                           OC_EXPR(index));
        Var<element_type> *ret = f->template create_temp_obj<Var<element_type>>(expr);
        return *ret;
    }
};

template<typename T>
struct EnableGetMemberByIndex {
    using element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        return def<element_type>(Function::current()->access(Type::of<element_type>(),
                                                    OC_EXPR(*static_cast<const T *>(this)),
                                                    OC_EXPR(i)));
    }
    template<size_t i>
    auto &get() noexcept {
        auto f = Function::current();
        const AccessExpr *expr = f->access(Type::of<element_type>(),
                                           OC_EXPR(*static_cast<const T *>(this)),
                                           OC_EXPR(int(i)));
        Var<element_type> *ret = f->template create_temp_obj<Var<element_type>>(expr);
        return *ret;
    }
};

template<typename T>
struct EnableStaticCast {
    template<class Dest>
    [[nodiscard]] Expr<Dest> cast() const noexcept {
        const CastExpr *expr = Function::current()->cast(Type::of<Dest>(), CastOp::STATIC,
                                                         static_cast<const T *>(this)->expression());
        return def_expr<Dest>(expr);
    }
};

template<typename T>
struct EnableBitwiseCast {
    template<class Dest>
    [[nodiscard]] Expr<Dest> as() const noexcept {
        const CastExpr *expr = Function::current()->cast(Type::of<Dest>(), CastOp::BITWISE,
                                                         static_cast<const T *>(this)->expression());
        return def_expr<Dest>(expr);
    }
};

#define OC_COMPUTABLE_COMMON(...)                                                       \
private:                                                                                \
    const Expression *_expression{nullptr};                                             \
                                                                                        \
public:                                                                                 \
    [[nodiscard]] const Expression *expression() const noexcept { return _expression; } \
                                                                                        \
protected:                                                                              \
    explicit Computable(const Expression *e) noexcept : _expression{e} {}               \
    Computable(Computable &&) noexcept = default;                                       \
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
public:
    Var<T> x{Function::current()->swizzle(Type::of<T>(), expression(), 0, 1)};
    Var<T> y{Function::current()->swizzle(Type::of<T>(), expression(), 1, 1)};
#include "swizzle_2.inl.h"
};

template<typename T>
struct Computable<Vector<T, 3>>
    : detail::EnableStaticCast<Computable<Vector<T, 3>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 3>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 3>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 3>>> {
    OC_COMPUTABLE_COMMON(Vector<T, 3>)
public:
    Var<T> x{Function::current()->swizzle(Type::of<T>(), expression(), 0, 1)};
    Var<T> y{Function::current()->swizzle(Type::of<T>(), expression(), 1, 1)};
    Var<T> z{Function::current()->swizzle(Type::of<T>(), expression(), 2, 1)};
#include "swizzle_3.inl.h"
};

template<typename T>
struct Computable<Vector<T, 4>>
    : detail::EnableStaticCast<Computable<Vector<T, 4>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 4>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 4>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 4>>> {
    OC_COMPUTABLE_COMMON(Vector<T, 4>)
public:
    Var<T> x{Function::current()->swizzle(Type::of<T>(), expression(), 0, 1)};
    Var<T> y{Function::current()->swizzle(Type::of<T>(), expression(), 1, 1)};
    Var<T> z{Function::current()->swizzle(Type::of<T>(), expression(), 2, 1)};
    Var<T> w{Function::current()->swizzle(Type::of<T>(), expression(), 3, 1)};
#include "swizzle_4.inl.h"
};

template<typename T, size_t N>
struct Computable<std::array<T, N>>
    : detail::EnableSubscriptAccess<Computable<std::array<T, N>>>,
      detail::EnableGetMemberByIndex<Computable<std::array<T, N>>> {
    OC_COMPUTABLE_COMMON(std::array<T, N>)
};

template<typename T, size_t N>
struct Computable<T[N]>
    : detail::EnableSubscriptAccess<Computable<T[N]>>,
      detail::EnableGetMemberByIndex<Computable<T[N]>> {
    OC_COMPUTABLE_COMMON(T[N])
};

template<size_t N>
struct Computable<Matrix<N>>
    : detail::EnableGetMemberByIndex<Computable<Matrix<N>>>,
      detail::EnableSubscriptAccess<Computable<Matrix<N>>> {
    OC_COMPUTABLE_COMMON(Matrix<N>)
};

template<typename... T>
struct Computable<ocarina::tuple<T...>> {
    using Tuple = ocarina::tuple<T...>;
    OC_COMPUTABLE_COMMON(ocarina::tuple<T...>)
public:
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        using Elm = ocarina::tuple_element_t<i, Tuple>;
        return def<Elm>(Function::current()->member(Type::of<Elm>(), expression(), i));
    }
};

#define OC_MAKE_STRUCT_MEMBER(m)                                                                                        \
    Var<std::remove_cvref_t<decltype(this_type::m)>>(m){Function::current()->member(Type::of<decltype(this_type::m)>(), \
                                                                                    expression(),                       \
                                                                                    ocarina::struct_member_tuple<Hit>::member_index(#m))};

#define OC_MAKE_COMPUTABLE_BODY(S, ...)           \
    namespace detail {                            \
    template<>                                    \
    struct Computable<S> {                        \
        using this_type = S;                      \
        OC_COMPUTABLE_COMMON(S)                   \
    public:                                       \
        MAP(OC_MAKE_STRUCT_MEMBER, ##__VA_ARGS__) \
    };                                            \
    }
}// namespace detail

}// namespace ocarina