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

template<typename T>
class Array;

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

class Hit;
class Ray;

namespace detail {

template<typename T>
[[nodiscard]] decltype(auto) extract_expression(T &&v) noexcept;// implement in dsl/expr.h

#define OC_EXPR(arg) ocarina::detail::extract_expression(OC_FORWARD(arg))

template<typename T,
         typename element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>>
struct EnableSubscriptAccess {

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto operator[](Index &&index) const noexcept {
        const SubscriptExpr *expr = Function::current()->subscript(Type::of<element_type>(),
                                                                   static_cast<const T *>(this)->expression(),
                                                                   OC_EXPR(index));
        return eval<element_type>(expr);
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto &operator[](Index &&index) noexcept {
        auto f = Function::current();
        const SubscriptExpr *expr = f->subscript(Type::of<element_type>(),
                                                 static_cast<const T *>(this)->expression(),
                                                 OC_EXPR(index));
        Var<element_type> *ret = f->template create_temp_obj<Var<element_type>>(expr);
        return *ret;
    }
};

template<typename T>
struct EnableReadAndWrite {
    using element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
    template<typename... Index>
    requires concepts::all_integral<expr_value_t<Index>...>
    auto read(Index &&...index) const noexcept {
        const SubscriptExpr *expr = Function::current()->subscript(Type::of<element_type>(),
                                                                   static_cast<const T *>(this)->expression(),
                                                                   {OC_EXPR(index)...});
        return eval<element_type>(expr);
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<element_type, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        const SubscriptExpr *expr = Function::current()->subscript(Type::of<element_type>(),
                                                                   static_cast<const T *>(this)->expression(),
                                                                   OC_EXPR(index));
        assign(expr, OC_FORWARD(elm));
    }
};

template<typename T>
struct EnableTextureReadAndWrite {

    template<typename Output, typename X, typename Y>
    requires(is_all_integral_expr_v<X, Y>)
    OC_NODISCARD auto read(const X &x, const Y &y) const noexcept {
        const T *texture = static_cast<const T *>(this);
        const CallExpr *expr = Function::current()->call_builtin(Type::of<Output>(), CallOp::TEX_READ,
                                                                 {texture->expression(), OC_EXPR(x), OC_EXPR(y)},
                                                                 {Type::of<Output>()});
        return eval<Output>(expr);
    }

    template<typename Output, typename X, typename Y, typename Z>
    requires(is_all_integral_expr_v<X, Y, Z>)
    OC_NODISCARD auto read(const X &x, const Y &y, const Z &z) const noexcept {
        const T *texture = static_cast<const T *>(this);
        const CallExpr *expr = Function::current()->call_builtin(Type::of<Output>(), CallOp::TEX_READ,
                                                                 {texture->expression(), OC_EXPR(x), OC_EXPR(y), OC_EXPR(z)},
                                                                 {Type::of<Output>()});
        return eval<Output>(expr);
    }

    template<typename Target, typename XY>
    requires(is_int_vector2_v<expr_value_t<XY>> ||
             is_uint_vector2_v<expr_value_t<XY>> &&
                 (is_uchar_element_expr_v<Target> || is_float_element_expr_v<Target>))
    OC_NODISCARD auto read(const XY &xy) const noexcept {
        return read<Target>(xy.x, xy.y);
    }

    template<typename Target, typename XYZ>
    requires(is_int_vector3_v<expr_value_t<XYZ>> ||
             is_uint_vector3_v<expr_value_t<XYZ>> &&
                 (is_uchar_element_expr_v<Target> || is_float_element_expr_v<Target>))
    OC_NODISCARD auto read(const XYZ &xyz) const noexcept {
        return read<Target>(xyz.x, xyz.y, xyz.z);
    }

    template<typename X, typename Y, typename Val>
    requires(is_all_integral_expr_v<X, Y> &&
             (is_uchar_element_expr_v<Val> ||
              is_float_element_expr_v<Val>))
    void write(const X &x, const Y &y, const Val &elm) noexcept {
        const T *texture = static_cast<const T *>(this);
        const CallExpr *expr = Function::current()->call_builtin(Type::of<uchar4>(), CallOp::TEX_WRITE,
                                                                 {texture->expression(),
                                                                  OC_EXPR(elm), OC_EXPR(x), OC_EXPR(y)});
        Function::current()->expr_statement(expr);
    }

    template<typename X, typename Y, typename Z, typename Val>
    requires(is_all_integral_expr_v<X, Y, Z> &&
             (is_uchar_element_expr_v<Val> ||
              is_float_element_expr_v<Val>))
    void write(const X &x, const Y &y, const Z &z, const Val &elm) noexcept {
        const T *texture = static_cast<const T *>(this);
        const CallExpr *expr = Function::current()->call_builtin(Type::of<uchar4>(), CallOp::TEX_WRITE,
                                                                 {texture->expression(),
                                                                  OC_EXPR(elm), OC_EXPR(x), OC_EXPR(y), OC_EXPR(z)});
        Function::current()->expr_statement(expr);
    }

    template<typename XYZ, typename Val>
    requires(is_uint_vector3_v<expr_value_t<XYZ>>)
    void write(const XYZ &xyz, const Val &elm) noexcept {
        write(xyz.x, xyz.y, xyz.z, elm);
    }

    template<typename XY, typename Val>
    requires(is_uint_vector2_v<expr_value_t<XY>>)
    void write(const XY &xy, const Val &elm) noexcept {
        write(xy.x, xy.y, elm);
    }
};

template<typename T>
struct AtomicRef {
private:
    const SubscriptExpr *_expression{};

public:
    explicit AtomicRef(const SubscriptExpr *expression)
        : _expression(expression) {}
    AtomicRef(AtomicRef &&) noexcept = delete;
    AtomicRef(const AtomicRef &) noexcept = delete;
    AtomicRef &operator=(AtomicRef &&) noexcept = delete;
    AtomicRef &operator=(const AtomicRef &) noexcept = delete;

    Var<T> exchange(Var<T> value) noexcept {
        const Expression *expr = Function::current()->call_builtin(Type::of<T>(),
                                                                   CallOp::ATOMIC_EXCH,
                                                                   {_expression, OC_EXPR(value)});
        return eval<T>(expr);
    }

    Var<T> fetch_add(Var<T> value) noexcept {
        const Expression *expr = Function::current()->call_builtin(Type::of<T>(),
                                                                   CallOp::ATOMIC_ADD,
                                                                   {_expression, OC_EXPR(value)});
        return eval<T>(expr);
    }

    Var<T> fetch_sub(Var<T> value) noexcept {
        const Expression *expr = Function::current()->call_builtin(Type::of<T>(),
                                                                   CallOp::ATOMIC_SUB,
                                                                   {_expression, OC_EXPR(value)});
        return eval<T>(expr);
    }
};

template<typename T>
struct BufferAsAtomicAddress {
    template<typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] AtomicRef<T> atomic(Index &&index) noexcept {
        static_assert(is_scalar_expr_v<T>);
        return AtomicRef<T>(Function::current()->subscript(Type::of<T>(), static_cast<Computable<Buffer<T>> *>(this)->expression(), OC_EXPR(index)));
    }
};

template<typename T>
struct EnableTextureSample {

    template<typename U, typename V>
    requires(is_all_floating_point_expr_v<U, V>)
    OC_NODISCARD Array<float> sample(uint channel_num, const U &u, const V &v)
        const noexcept;// implement in dsl/array.h

    template<typename U, typename V, typename W>
    requires(is_all_floating_point_expr_v<U, V, W>)
    OC_NODISCARD Array<float> sample(uint channel_num, const U &u, const V &v, const W &w)
        const noexcept;// implement in dsl/array.h

    template<typename UVW>
    requires(is_float_vector3_v<expr_value_t<UVW>>)
    OC_NODISCARD Array<float> sample(uint channel_num, const UVW &uvw)
        const noexcept;// implement in dsl/array.h

    template<typename UV>
    requires(is_float_vector2_v<expr_value_t<UV>>)
    OC_NODISCARD Array<float> sample(uint channel_num, const UV &uv)
        const noexcept;// implement in dsl/array.h
};

template<typename T>
struct EnableGetMemberByIndex {
    using element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
    template<int i>
    [[nodiscard]] auto get() const noexcept {
        return eval<element_type>(Function::current()->subscript(Type::of<element_type>(),
                                                                 OC_EXPR(*static_cast<const T *>(this)),
                                                                 OC_EXPR(i)));
    }
    template<int i>
    auto &get() noexcept {
        auto f = Function::current();
        const SubscriptExpr *expr = f->subscript(Type::of<element_type>(),
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
    OC_NODISCARD Var<Dest>
    cast()
        const noexcept {
        const CastExpr *expr = Function::current()->cast(Type::of<Dest>(), CastOp::STATIC,
                                                         static_cast<const T *>(this)->expression());
        return eval<Dest>(expr);
    }
};

template<typename T>
struct EnableBitwiseCast {
    template<class Dest>
    requires concepts::bitwise_convertible<Dest, expr_value_t<T>>
    OC_NODISCARD Var<Dest>
    as()
        const noexcept {
        const CastExpr *expr = Function::current()->cast(Type::of<Dest>(), CastOp::BITWISE,
                                                         static_cast<const T *>(this)->expression());
        return eval<Dest>(expr);
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
      detail::EnableSubscriptAccess<Computable<Buffer<T>>>,
      detail::BufferAsAtomicAddress<T> {
    OC_COMPUTABLE_COMMON(Computable<Buffer<T>>)
};

template<>
struct Computable<Texture>
    : detail::EnableTextureSample<Computable<Texture>>,
      detail::EnableTextureReadAndWrite<Computable<Texture>> {
    OC_COMPUTABLE_COMMON(Computable<Texture>)
};

template<size_t N>
struct Computable<Matrix<N>>
    : detail::EnableGetMemberByIndex<Computable<Matrix<N>>>,
      detail::EnableSubscriptAccess<Computable<Matrix<N>>> {
    OC_COMPUTABLE_COMMON(Computable<Matrix<N>>)
};

template<>
struct Computable<Accel> {
public:
    template<typename TRay>
    requires std::is_same_v<expr_value_t<TRay>, Ray>
    [[nodiscard]] Var<Hit> trace_closest(const TRay &ray) const noexcept;// implement in computable.inl

    template<typename TRay>
    requires std::is_same_v<expr_value_t<TRay>, Ray>
    [[nodiscard]] Var<bool> trace_any(const TRay &ray) const noexcept;// implement in computable.inl
    OC_COMPUTABLE_COMMON(Computable<Accel>)
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
    template<size_t i, typename elm_ty = ocarina::tuple_element_t<i, Tuple>>
    void set(elm_ty val) noexcept {
        auto expr = Function::current()->member(Type::of<expr_value_t<elm_ty>>(), expression(), i);
        assign(expr, val);
    }
};

}// namespace detail

template<typename T>
class ResourceArrayBuffer {
    static_assert(is_valid_buffer_element_v<T>);

private:
    const Expression *_resource_array{nullptr};
    const Expression *_index{nullptr};

public:
    ResourceArrayBuffer(const Expression *array, const Expression *index) noexcept
        : _resource_array{array}, _index{index} {}

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] Var<T> read(Index &&index) const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<T>(), CallOp::RESOURCE_ARRAY_BUFFER_READ,
                                                                 {_resource_array, _index, OC_EXPR(index)});
        return eval<T>(expr);
    }

    template<typename Size = uint>
    [[nodiscard]] Var<Size> size_in_byte() const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<T>(), CallOp::RESOURCE_ARRAY_BUFFER_SIZE,
                                                                 {_resource_array, _index});
        return eval<Size>(expr);
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<T, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<T>(), CallOp::RESOURCE_ARRAY_BUFFER_WRITE,
                                                                 {_resource_array, _index, OC_EXPR(index), OC_EXPR(elm)});
        Function::current()->expr_statement(expr);
    }
};

class ResourceArrayByteBuffer {
private:
    const Expression *_resource_array{nullptr};
    const Expression *_index{nullptr};

public:
    ResourceArrayByteBuffer(const Expression *array, const Expression *index) noexcept
        : _resource_array{array}, _index{index} {}

    template<typename T, typename Offset>
    requires concepts::integral<expr_value_t<Offset>>
    [[nodiscard]] Var<T> read(Offset &&offset) const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<T>(), CallOp::RESOURCE_ARRAY_BYTE_BUFFER_READ,
                                                                 {_resource_array, _index, OC_EXPR(offset)});
        return eval<T>(expr);
    }

    template<typename T = float, typename Size = uint>
    [[nodiscard]] Var<Size> size_in_byte() const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<T>(), CallOp::RESOURCE_ARRAY_BUFFER_SIZE,
                                                                 {_resource_array, _index});
        return eval<Size>(expr);
    }

    template<typename T, typename Offset>
    [[nodiscard]] Array<T> read_dynamic_array(uint size, Offset &&offset) const noexcept;// implement in dsl/array.h
};

class ResourceArrayTexture {
private:
    const Expression *_resource_array{nullptr};
    const Expression *_index{nullptr};

public:
    ResourceArrayTexture(const Expression *array, const Expression *index) noexcept
        : _resource_array{array}, _index{index} {}

    template<typename U, typename V, typename W>
    requires(is_all_floating_point_expr_v<U, V, W>)
    OC_NODISCARD Array<float> sample(uint channel_num, const U &u, const V &v, const W &w)
        const noexcept;// implement in dsl/array.h

    template<typename U, typename V>
    requires(is_all_floating_point_expr_v<U, V>)
    OC_NODISCARD Array<float> sample(uint channel_num, const U &u, const V &v)
        const noexcept;// implement in dsl/array.h

    template<typename UVW>
    requires(is_float_vector3_v<expr_value_t<UVW>>)
    OC_NODISCARD Array<float> sample(uint channel_num, const UVW &uvw)
        const noexcept;// implement in dsl/array.h

    template<typename UV>
    requires(is_float_vector2_v<expr_value_t<UV>>)
    OC_NODISCARD Array<float> sample(uint channel_num, const UV &uv)
        const noexcept;// implement in dsl/array.h
};

namespace detail {
template<>
struct Computable<ResourceArray> {
public:
    template<typename T, typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] ResourceArrayBuffer<T> buffer(Index &&index) const noexcept {
        return ResourceArrayBuffer<T>(expression(), OC_EXPR(index));
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] ResourceArrayTexture tex(Index &&index) const noexcept {
        return ResourceArrayTexture(expression(), OC_EXPR(index));
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] ResourceArrayByteBuffer byte_buffer(Index &&index) const noexcept {
        return ResourceArrayByteBuffer(expression(), OC_EXPR(index));
    }

    OC_COMPUTABLE_COMMON(Computable<ResourceArray>)
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
    public:                                       \
        using this_type = S;                      \
        static constexpr auto cname = #S;         \
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