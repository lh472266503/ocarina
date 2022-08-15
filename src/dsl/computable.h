//
// Created by Zero on 19/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "core/basic_traits.h"
#include "type_trait.h"
#include "ast/function.h"
#include <utility>

namespace ocarina {

namespace detail {
template<typename Lhs, typename Rhs>
inline void assign(Lhs &&lhs, Rhs &&rhs) noexcept;// implement in syntax.h
}

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> eval(T &&x) noexcept;// implement in syntax.h

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> eval(const Expression *expr) noexcept;// implement in syntax.h

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> make_expr(T &&x) noexcept;// implement in syntax.h

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> make_expr(const Expression *expr) noexcept;// implement in syntax.h

namespace detail {

template<typename T>
[[nodiscard]] decltype(auto) extract_expression(T &&v) noexcept;

#define OC_EXPR(arg) ocarina::detail::extract_expression(OC_FORWARD(arg))

template<typename T,
         typename element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>>
struct EnableSubscriptAccess {

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto operator[](Index &&index) const noexcept {
        const AccessExpr *expr = Function::current()->access(Type::of<element_type>(),
                                                             static_cast<const T *>(this)->expression(),
                                                             OC_EXPR(index));
        return eval<element_type>(expr);
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
struct EnableReadAndWrite {
    using element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto read(Index &&index) const noexcept {
        const AccessExpr *expr = Function::current()->access(Type::of<element_type>(),
                                                             static_cast<const T *>(this)->expression(),
                                                             OC_EXPR(index));
        return eval<element_type>(expr);
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<element_type, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        const AccessExpr *expr = Function::current()->access(Type::of<element_type>(),
                                                             static_cast<const T *>(this)->expression(),
                                                             OC_EXPR(index));
        assign(expr, OC_FORWARD(elm));
    }
};

template<typename T>
struct EnableTextureReadAndWrite {

    using texture_type = expr_value_t<T>;
    using element_type = texture_element_t<texture_type>;

    template<typename Target = element_type, typename X, typename Y>
    requires(is_all_integral_expr_v<X, Y>)
        OC_NODISCARD auto read(const X &x, const Y &y) const noexcept {
        const T *texture = static_cast<const T *>(this);
        const CallExpr *expr = Function::current()->call_builtin(Type::of<Target>(), CallOp::IMAGE_READ,
                                                                 {texture->expression(),OC_EXPR(x), OC_EXPR(y)},
                                                                 {Type::of<Target>(), Type::of<element_type>()});
        return eval<Target>(expr);
    }

    template<typename Target = element_type, typename XY>
    requires(is_int_vector2_v<expr_value_t<XY>> || is_uint_vector2_v<expr_value_t<XY>>)
    OC_NODISCARD auto read(const XY &xy) const noexcept {
        return read<Target>(xy.x, xy.y);
    }

    template<typename X, typename Y, typename Val>
    requires(is_all_integral_expr_v<X, Y> && (is_uchar_element_expr_v<Val> || is_float_element_expr_v<Val>))
    void write(const X &x, const Y &y, const Val &elm) noexcept {
        const T *texture = static_cast<const T *>(this);
        const CallExpr *expr = Function::current()->call_builtin(Type::of<element_type>(), CallOp::IMAGE_WRITE,
                                                                   {texture->expression(),
                                                                    OC_EXPR(x), OC_EXPR(y), OC_EXPR(elm)},
                                                                    {Type::of<Val>(), Type::of<element_type>()});
        Function::current()->expr_statement(expr);
    }

    template<typename XY, typename Val>
    requires(is_uint_vector2_v<expr_value_t<XY>>)
    void write(const XY &xy, const Val &elm) noexcept {
        write(xy.x, xy.y, elm);
    }
};

template<typename T>
struct EnableTextureSample {

    template<typename U, typename V>
    requires(is_all_floating_point_expr_v<U, V>)
    OC_NODISCARD auto sample(const U &u, const V &v) const noexcept {
        using texture_type = expr_value_t<T>;
        using element_type = texture_element_t<texture_type>;
        const T *texture = static_cast<const T *>(this);
        using sample_type = texture_sample_t<element_type>;
        const CallExpr *expr = Function::current()->call_builtin(Type::of<sample_type>(),
                                                                 CallOp::TEX_SAMPLE,
                                                                 {texture->expression(),
                                                                  OC_EXPR(u),
                                                                  OC_EXPR(v)});
        return make_expr<sample_type>(expr);
    }

    template<typename UV>
    requires(is_float_vector2_v<expr_value_t<UV>>)
    OC_NODISCARD auto sample(const UV &uv) const noexcept {
        return sample(uv.x, uv.y);
    }
};

template<typename T>
struct EnableGetMemberByIndex {
    using element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
    template<int i>
    [[nodiscard]] auto get() const noexcept {
        return eval<element_type>(Function::current()->access(Type::of<element_type>(),
                                                              OC_EXPR(*static_cast<const T *>(this)),
                                                              OC_EXPR(i)));
    }
    template<int i>
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
    requires concepts::static_convertible<Dest, expr_value_t<T>>
    OC_NODISCARD Expr<Dest>
    cast() const noexcept {
        const CastExpr *expr = Function::current()->cast(Type::of<Dest>(), CastOp::STATIC,
                                                         static_cast<const T *>(this)->expression());
        return make_expr<Dest>(expr);
    }
};

template<typename T>
struct EnableBitwiseCast {
    template<class Dest>
    requires concepts::bitwise_convertible<Dest, expr_value_t<T>>
    OC_NODISCARD Expr<Dest>
    as() const noexcept {
        const CastExpr *expr = Function::current()->cast(Type::of<Dest>(), CastOp::BITWISE,
                                                         static_cast<const T *>(this)->expression());
        return make_expr<Dest>(expr);
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
    OC_COMPUTABLE_COMMON(Computable<T>)
};

template<typename T>
struct Computable<Vector<T, 2>>
    : detail::EnableStaticCast<Computable<Vector<T, 2>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 2>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 2>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 2>>> {
    OC_COMPUTABLE_COMMON(Computable<Vector<T, 2>>)
public:
    Var<T> x{Function::current()->swizzle(Type::of<T>(), expression(), 0, 1)};
    Var<T> y{Function::current()->swizzle(Type::of<T>(), expression(), 1, 1)};
#include "swizzle_inl/swizzle_2.inl.h"
};

template<typename T>
struct Computable<Vector<T, 3>>
    : detail::EnableStaticCast<Computable<Vector<T, 3>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 3>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 3>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 3>>> {
    OC_COMPUTABLE_COMMON(Computable<Vector<T, 3>>)
public:
    Var<T> x{Function::current()->swizzle(Type::of<T>(), expression(), 0, 1)};
    Var<T> y{Function::current()->swizzle(Type::of<T>(), expression(), 1, 1)};
    Var<T> z{Function::current()->swizzle(Type::of<T>(), expression(), 2, 1)};
#include "swizzle_inl/swizzle_3.inl.h"
};

template<typename T>
struct Computable<Vector<T, 4>>
    : detail::EnableStaticCast<Computable<Vector<T, 4>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 4>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 4>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 4>>> {
    OC_COMPUTABLE_COMMON(Computable<Vector<T, 4>>)
public:
    Var<T> x{Function::current()->swizzle(Type::of<T>(), expression(), 0, 1)};
    Var<T> y{Function::current()->swizzle(Type::of<T>(), expression(), 1, 1)};
    Var<T> z{Function::current()->swizzle(Type::of<T>(), expression(), 2, 1)};
    Var<T> w{Function::current()->swizzle(Type::of<T>(), expression(), 3, 1)};
#include "swizzle_inl/swizzle_4.inl.h"
};

template<typename T, size_t N>
struct Computable<std::array<T, N>>
    : detail::EnableSubscriptAccess<Computable<std::array<T, N>>>,
      detail::EnableGetMemberByIndex<Computable<std::array<T, N>>> {
    OC_COMPUTABLE_COMMON(Computable<std::array<T, N>>)
};

template<typename T, size_t N>
struct Computable<T[N]>
    : detail::EnableSubscriptAccess<Computable<T[N]>>,
      detail::EnableGetMemberByIndex<Computable<T[N]>> {
    OC_COMPUTABLE_COMMON(Computable<T[N]>)
};

template<typename T>
struct Computable<Buffer<T>>
    : detail::EnableReadAndWrite<Computable<Buffer<T>>>,
      detail::EnableSubscriptAccess<Computable<Buffer<T>>> {
    OC_COMPUTABLE_COMMON(Computable<Buffer<T>>)
};

template<typename T>
struct Computable<Image<T>>
    : detail::EnableTextureSample<Computable<Image<T>>>,
      detail::EnableTextureReadAndWrite<Computable<Image<T>>> {
    OC_COMPUTABLE_COMMON(Computable<Image>)
};

template<size_t N>
struct Computable<Matrix<N>>
    : detail::EnableGetMemberByIndex<Computable<Matrix<N>>>,
      detail::EnableSubscriptAccess<Computable<Matrix<N>>> {
    OC_COMPUTABLE_COMMON(Computable<Matrix<N>>)
};

template<typename... T>
struct Computable<ocarina::tuple<T...>> {
    using Tuple = ocarina::tuple<T...>;
    OC_COMPUTABLE_COMMON(Computable<ocarina::tuple<T...>>)
public:
    template<size_t i>
    [[nodiscard]] auto get() const noexcept {
        using Elm = ocarina::tuple_element_t<i, Tuple>;
        return eval<Elm>(Function::current()->member(Type::of<Elm>(), expression(), i));
    }
};

#define OC_MAKE_STRUCT_MEMBER(m)                                                                                        \
    Var<std::remove_cvref_t<decltype(this_type::m)>>(m){Function::current()->member(Type::of<decltype(this_type::m)>(), \
                                                                                    expression(),                       \
                                                                                    ocarina::struct_member_tuple<this_type>::member_index(#m))};

#define OC_MAKE_COMPUTABLE_BODY(S, ...)           \
    namespace ocarina {                           \
    namespace detail {                            \
    template<>                                    \
    struct Computable<S> {                        \
        using this_type = S;                      \
        OC_COMPUTABLE_COMMON(S)                   \
    public:                                       \
        MAP(OC_MAKE_STRUCT_MEMBER, ##__VA_ARGS__) \
    };                                            \
    }                                             \
    }
}// namespace detail

template<typename T>
struct Proxy : public ocarina::detail::Computable<T> {
    static_assert(ocarina::always_false_v<T>, "proxy is invalid !");
};

#define OC_MAKE_GET_PROXY                                                                      \
    auto operator->() noexcept { return reinterpret_cast<ocarina::Proxy<this_type> *>(this); } \
    auto operator->() const noexcept { return reinterpret_cast<const ocarina::Proxy<this_type> *>(this); }

#define OC_MAKE_PROXY(S) \
    template<>           \
    struct ocarina::Proxy<S> : public ocarina::detail::Computable<S>

}// namespace ocarina