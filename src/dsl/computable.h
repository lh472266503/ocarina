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
#include "core/platform.h"

namespace ocarina {

namespace detail {
template<typename Lhs, typename Rhs>
requires(!is_param_struct_v<expr_value_t<Lhs>> && !is_param_struct_v<expr_value_t<Rhs>>)
void assign(Lhs &&lhs, Rhs &&rhs) noexcept;// implement in stmt_builder.h

[[nodiscard]] Var<uint> correct_index(Var<uint> index, Var<uint> size, const string &desc,
                                      const string &tb) noexcept;// implement in env.cpp

[[nodiscard]] Var<uint> correct_index(Var<uint> index, uint size, const string &desc,
                                      const string &tb) noexcept;// implement in env.cpp

[[nodiscard]] Var<uint> divide(Var<uint> lhs, Var<uint> rhs) noexcept;// implement in env.cpp

}// namespace detail

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> eval(T &&x) noexcept;// implement in stmt_builder.h

template<typename T>
[[nodiscard]] inline Var<expr_value_t<T>> eval(const Expression *expr) noexcept;// implement in stmt_builder.h

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> make_expr(T &&x) noexcept;// implement in stmt_builder.h

template<typename T>
[[nodiscard]] inline Expr<expr_value_t<T>> make_expr(const Expression *expr) noexcept;// implement in stmt_builder.h

class Hit;
class Ray;

template<typename T, typename TBuffer>
struct SOAView;

namespace detail {

template<typename T>
[[nodiscard]] decltype(auto) extract_expression(T &&v) noexcept;// implement in dsl/expr.h

#define OC_EXPR(arg) ocarina::detail::extract_expression(OC_FORWARD(arg))

template<typename T,
         typename element_type = std::remove_cvref_t<expr_value_t<decltype(std::declval<expr_value_t<T>>()[0])>>>
struct EnableSubscriptAccess {

    [[nodiscard]] T *self() noexcept { return static_cast<T *>(this); }
    [[nodiscard]] const T *self() const noexcept { return static_cast<const T *>(this); }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto operator[](Index &&index) const noexcept {
        const SubscriptExpr *expr = Function::current()->subscript(Type::of<element_type>(),
                                                                   self()->expression(),
                                                                   OC_EXPR(index));
        expr->mark(Usage::READ);
        return eval<element_type>(expr);
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto at(Index &&index) const noexcept {
        const SubscriptExpr *expr = Function::current()->subscript(Type::of<element_type>(),
                                                                   self()->expression(),
                                                                   OC_EXPR(index));
        expr->mark(Usage::READ);
        return eval<element_type>(expr);
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto &operator[](Index &&index) noexcept {
        auto f = Function::current();
        const SubscriptExpr *expr = f->subscript(Type::of<element_type>(),
                                                 self()->expression(),
                                                 OC_EXPR(index));
        expr->mark(Usage::WRITE);
        Var<element_type> *ret = f->template create_temp_obj<Var<element_type>>(expr);
        return *ret;
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    auto &at(Index &&index) noexcept {
        auto f = Function::current();
        const SubscriptExpr *expr = f->subscript(Type::of<element_type>(),
                                                 self()->expression(),
                                                 OC_EXPR(index));
        expr->mark(Usage::WRITE);
        Var<element_type> *ret = f->template create_temp_obj<Var<element_type>>(expr);
        return *ret;
    }
};

template<typename T>
struct EnableReadAndWrite {

    [[nodiscard]] T *self() noexcept { return static_cast<T *>(this); }
    [[nodiscard]] const T *self() const noexcept { return static_cast<const T *>(this); }

    using element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
    template<typename... Index>
    requires concepts::all_integral<expr_value_t<Index>...>
    auto read_multi(Index &&...index) const noexcept {
        const SubscriptExpr *expr = Function::current()->subscript(Type::of<element_type>(),
                                                                   self()->expression(),
                                                                   {OC_EXPR(index)...});
        expr->mark(Usage::READ);
        return eval<element_type>(expr);
    }

    template<typename Index>
    requires concepts::all_integral<expr_value_t<Index>>
    auto read(Index &&index, bool check_boundary = true) const noexcept {
        Var<expr_value_t<Index>> new_index = OC_FORWARD(index);
        if (check_boundary) {
            Var<expr_value_t<Index>> size = static_cast<const T *>(this)->template size<expr_value_t<Index>>();
            new_index = correct_index(new_index, size, typeid(T).name(), traceback_string(1));
        }
        const SubscriptExpr *expr = Function::current()->subscript(Type::of<element_type>(),
                                                                   self()->expression(),
                                                                   {OC_EXPR(new_index)});
        expr->mark(Usage::READ);
        return eval<element_type>(expr);
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<element_type, expr_value_t<Val>>
    void write(Index &&index, Val &&elm, bool check_boundary = true) {
        Var<expr_value_t<Index>> new_index = OC_FORWARD(index);
        if (check_boundary) {
            Var<expr_value_t<Index>> size = static_cast<const T *>(this)->template size<expr_value_t<Index>>();
            new_index = correct_index(new_index, size, typeid(T).name(), traceback_string(1));
        }
        const SubscriptExpr *expr = Function::current()->subscript(Type::of<element_type>(),
                                                                   self()->expression(),
                                                                   OC_EXPR(new_index));
        expr->mark(Usage::WRITE);
        assign(expr, OC_FORWARD(elm));
    }

    template<typename Index, typename Val>
    requires is_integral_expr_v<Index> &&
             concepts::is_same_v<element_type, expr_value_t<Val>>
    void write_and_check(Index &&index, Val &&elm, uint size, const string &desc) {
        if constexpr (is_integral_v<Index>) {
            OC_ASSERT(index <= size);
            write(OC_FORWARD(index), OC_FORWARD(elm));
        } else {
            index = correct_index(OC_FORWARD(index), size, desc, traceback_string(1));
            write(OC_FORWARD(index), OC_FORWARD(elm));
        }
    }
};

template<typename T>
struct EnableByteLoadAndStore {

    [[nodiscard]] T *self() noexcept { return static_cast<T *>(this); }
    [[nodiscard]] const T *self() const noexcept { return static_cast<const T *>(this); }

    template<uint N = 1, typename Elm = uint, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] auto load(Offset &&offset, bool check_boundary = true) const noexcept {
        Var<expr_value_t<Offset>> new_offset = OC_FORWARD(offset);
        if (check_boundary) {
            new_offset = detail::correct_index(new_offset, self()->template size<expr_value_t<Offset>>(),
                                               typeid(*this).name(), traceback_string());
        }
        self()->expression()->mark(Usage::READ);
        if constexpr (N == 1) {
            const CallExpr *expr = Function::current()->call_builtin(Type::of<Elm>(),
                                                                     CallOp::BYTE_BUFFER_READ,
                                                                     {self()->expression(),
                                                                      OC_EXPR(new_offset)},
                                                                     {N});
            return eval<Elm>(expr);
        } else {
            const CallExpr *expr = Function::current()->call_builtin(Type::of<Vector<Elm, N>>(),
                                                                     CallOp::BYTE_BUFFER_READ,
                                                                     {self()->expression(),
                                                                      OC_EXPR(new_offset)},
                                                                     {N});
            return eval<Vector<Elm, N>>(expr);
        }
    }

    template<typename Target, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] Var<Target> load_as(Offset &&offset, bool check_boundary = true) const noexcept {
        Var<expr_value_t<Offset>> new_offset = OC_FORWARD(offset);
        if (check_boundary) {
            new_offset = detail::correct_index(new_offset, self()->template size<expr_value_t<Offset>>(),
                                               typeid(*this).name(), traceback_string());
        }
        const Type *ret_type = Type::of<Target>();
        self()->expression()->mark(Usage::READ);
        const CallExpr *expr = Function::current()->call_builtin(ret_type,
                                                                 CallOp::BYTE_BUFFER_READ,
                                                                 {self()->expression(),
                                                                  OC_EXPR(new_offset)},
                                                                 {ret_type});
        return eval<Target>(expr);
    }

    template<typename Elm = uint, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] auto load2(Offset &&offset, bool check_boundary = true) const noexcept {
        return load<2, Elm>(OC_FORWARD(offset), check_boundary);
    }

    template<typename Elm = uint, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] auto load3(Offset &&offset, bool check_boundary = true) const noexcept {
        return load<3, Elm>(OC_FORWARD(offset), check_boundary);
    }

    template<typename Elm = uint, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] auto load4(Offset &&offset, bool check_boundary = true) const noexcept {
        return load<4, Elm>(OC_FORWARD(offset), check_boundary);
    }

    template<typename Elm, typename Offset>
    requires is_integral_expr_v<Offset>
    void store(Offset &&offset, const Elm &val, bool check_boundary = true) noexcept {
        Var<expr_value_t<Offset>> new_offset = OC_FORWARD(offset);
        if (check_boundary) {
            new_offset = detail::correct_index(new_offset, self()->template size<expr_value_t<Offset>>(),
                                               typeid(*this).name(), traceback_string());
        }
        const CallExpr *expr = Function::current()->call_builtin(nullptr, CallOp::BYTE_BUFFER_WRITE,
                                                                 {self()->expression(),
                                                                  OC_EXPR(new_offset), OC_EXPR(val)});
        self()->expression()->mark(Usage::WRITE);
        Function::current()->expr_statement(expr);
    }
};

template<typename T>
struct EnableTextureReadAndWrite {

    [[nodiscard]] T *self() noexcept { return static_cast<T *>(this); }
    [[nodiscard]] const T *self() const noexcept { return static_cast<const T *>(this); }

    template<typename Output, typename X, typename Y>
    requires(is_all_integral_expr_v<X, Y>)
    OC_NODISCARD auto read(const X &x, const Y &y) const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<Output>(), CallOp::TEX_READ,
                                                                 {self()->expression(), OC_EXPR(x), OC_EXPR(y)},
                                                                 {Type::of<Output>()});
        self()->expression()->mark(Usage::READ);
        return eval<Output>(expr);
    }

    template<typename Output, typename X, typename Y, typename Z>
    requires(is_all_integral_expr_v<X, Y, Z>)
    OC_NODISCARD auto read(const X &x, const Y &y, const Z &z) const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<Output>(), CallOp::TEX_READ,
                                                                 {self()->expression(), OC_EXPR(x), OC_EXPR(y), OC_EXPR(z)},
                                                                 {Type::of<Output>()});
        self()->expression()->mark(Usage::READ);
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
    void write(const Val &elm, const X &x, const Y &y) noexcept {
        const T *texture = static_cast<const T *>(this);
        const CallExpr *expr = Function::current()->call_builtin(nullptr, CallOp::TEX_WRITE,
                                                                 {self()->expression(),
                                                                  OC_EXPR(elm), OC_EXPR(x), OC_EXPR(y)});
        self()->expression()->mark(Usage::WRITE);
        Function::current()->expr_statement(expr);
    }

    template<typename X, typename Y, typename Z, typename Val>
    requires(is_all_integral_expr_v<X, Y, Z> &&
             (is_uchar_element_expr_v<Val> ||
              is_float_element_expr_v<Val>))
    void write(const Val &elm, const X &x, const Y &y, const Z &z) noexcept {
        const CallExpr *expr = Function::current()->call_builtin(nullptr, CallOp::TEX_WRITE,
                                                                 {self()->expression(),
                                                                  OC_EXPR(elm), OC_EXPR(x), OC_EXPR(y), OC_EXPR(z)});
        self()->expression()->mark(Usage::WRITE);
        Function::current()->expr_statement(expr);
    }

    template<typename XYZ, typename Val>
    requires(is_uint_vector3_v<expr_value_t<XYZ>>)
    void write(const Val &elm, const XYZ &xyz) noexcept {
        write(elm, xyz.x, xyz.y, xyz.z);
    }

    template<typename XY, typename Val>
    requires(is_uint_vector2_v<expr_value_t<XY>>)
    void write(const Val &elm, const XY &xy) noexcept {
        write(elm, xy.x, xy.y);
    }
};

template<typename T>
struct AtomicRef {
private:
    const Expression *range_{};
    const Expression *index_{};

public:
    explicit AtomicRef(const Expression *range,
                       const Expression *index)
        : range_(range), index_(index) {}
    AtomicRef(AtomicRef &&) noexcept = delete;
    AtomicRef(const AtomicRef &) noexcept = delete;
    AtomicRef &operator=(AtomicRef &&) noexcept = delete;
    AtomicRef &operator=(const AtomicRef &) noexcept = delete;

    Var<T> exchange(Var<T> value) noexcept {
        const Expression *expr = Function::current()->call_builtin(Type::of<T>(),
                                                                   CallOp::ATOMIC_EXCH,
                                                                   {range_, index_, OC_EXPR(value)});
        range_->mark(Usage::READ_WRITE);
        return eval<T>(expr);
    }

    Var<T> fetch_add(Var<T> value) noexcept {
        const Expression *expr = Function::current()->call_builtin(Type::of<T>(),
                                                                   CallOp::ATOMIC_ADD,
                                                                   {range_, index_, OC_EXPR(value)});
        range_->mark(Usage::READ_WRITE);
        return eval<T>(expr);
    }

    Var<T> fetch_sub(Var<T> value) noexcept {
        const Expression *expr = Function::current()->call_builtin(Type::of<T>(),
                                                                   CallOp::ATOMIC_SUB,
                                                                   {range_, index_, OC_EXPR(value)});
        range_->mark(Usage::READ_WRITE);
        return eval<T>(expr);
    }
};

template<typename TBuffer, typename Elm>
struct BufferAsAtomicAddress {
    template<typename Target = Elm, typename Index>
    requires is_integral_expr_v<Index>
    [[nodiscard]] AtomicRef<Target> atomic(Index &&index) noexcept {
        static_assert(is_scalar_expr_v<Elm>);
        return AtomicRef<Target>(static_cast<TBuffer *>(this)->expression(), OC_EXPR(index));
    }
};

template<typename T>
struct EnableTextureSample {

    template<typename U, typename V>
    requires(is_all_floating_point_expr_v<U, V>)
    OC_NODISCARD DynamicArray<float> sample(uint channel_num, const U &u, const V &v)
        const noexcept;// implement in dsl/array.h

    template<typename U, typename V, typename W>
    requires(is_all_floating_point_expr_v<U, V, W>)
    OC_NODISCARD DynamicArray<float> sample(uint channel_num, const U &u, const V &v, const W &w)
        const noexcept;// implement in dsl/array.h

    template<typename UVW>
    requires(is_float_vector3_v<expr_value_t<UVW>>)
    OC_NODISCARD DynamicArray<float> sample(uint channel_num, const UVW &uvw)
        const noexcept;// implement in dsl/array.h

    template<typename UV>
    requires(is_float_vector2_v<expr_value_t<UV>>)
    OC_NODISCARD DynamicArray<float> sample(uint channel_num, const UV &uv)
        const noexcept;// implement in dsl/array.h
};

template<typename T>
struct EnableGetMemberByIndex {

    [[nodiscard]] T *self() noexcept { return static_cast<T *>(this); }
    [[nodiscard]] const T *self() const noexcept { return static_cast<const T *>(this); }

    using element_type = std::remove_cvref_t<decltype(std::declval<expr_value_t<T>>()[0])>;
    template<int i>
    [[nodiscard]] auto get() const noexcept {
        return eval<element_type>(Function::current()->subscript(Type::of<element_type>(),
                                                                 OC_EXPR(*self()),
                                                                 OC_EXPR(i)));
    }
    template<int i>
    auto &get() noexcept {
        auto f = Function::current();
        const SubscriptExpr *expr = f->subscript(Type::of<element_type>(),
                                                 OC_EXPR(*self()),
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
    const Expression *expression_{nullptr};                                             \
                                                                                        \
public:                                                                                 \
    [[nodiscard]] const Expression *expression() const noexcept { return expression_; } \
    [[nodiscard]] bool is_valid() const noexcept {                                      \
        return expression()->check_context(Function::current());                        \
    }                                                                                   \
                                                                                        \
protected:                                                                              \
    explicit Computable(const Expression *e) noexcept : expression_{e} {}               \
    Computable(Computable &&) noexcept = default;                                       \
    Computable(const Computable &) noexcept = default;

#define OC_MAKE_ASSIGNMENT_FUNC \
public:                         \
    void                        \
    set(const this_type &t) {   \
        assign(*this, t);       \
    }

template<typename T>
struct Computable
    : detail::EnableBitwiseCast<Computable<T>>,
      detail::EnableStaticCast<Computable<T>> {
    using this_type = T;
    static_assert(is_scalar_v<T>);
    OC_COMPUTABLE_COMMON(Computable<T>)
    OC_MAKE_ASSIGNMENT_FUNC
};

template<typename T>
struct Computable<Vector<T, 2>>
    : detail::EnableStaticCast<Computable<Vector<T, 2>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 2>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 2>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 2>>> {
    using this_type = Vector<T, 2>;
    OC_COMPUTABLE_COMMON(Computable<Vector<T, 2>>)
    OC_MAKE_ASSIGNMENT_FUNC

public:
    union {
        struct {
            Var<T> x;
            Var<T> y;
        };
        std::array<Var<T>, 2> arr;
    };

private:
    [[nodiscard]] bool initial() noexcept {
        Var<T> x_tmp(Function::current()->swizzle(Type::of<T>(), expression(), 0, 1));
        Var<T> y_tmp(Function::current()->swizzle(Type::of<T>(), expression(), 1, 1));
        oc_memcpy(addressof(x), addressof(x_tmp), sizeof(x));
        oc_memcpy(addressof(y), addressof(y_tmp), sizeof(y));
        return true;
    }
    bool initialed_{initial()};

public:
#include "swizzle_inl/swizzle_2.inl.h"
};

template<typename T>
struct Computable<Vector<T, 3>>
    : detail::EnableStaticCast<Computable<Vector<T, 3>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 3>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 3>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 3>>> {
    using this_type = Vector<T, 3>;
    OC_COMPUTABLE_COMMON(Computable<Vector<T, 3>>)
    OC_MAKE_ASSIGNMENT_FUNC

public:
    union {
        struct {
            Var<T> x;
            Var<T> y;
            Var<T> z;
        };
        std::array<Var<T>, 3> arr;
    };

private:
    [[nodiscard]] bool initial() noexcept {
        Var<T> x_tmp(Function::current()->swizzle(Type::of<T>(), expression(), 0, 1));
        Var<T> y_tmp(Function::current()->swizzle(Type::of<T>(), expression(), 1, 1));
        Var<T> z_tmp(Function::current()->swizzle(Type::of<T>(), expression(), 2, 1));
        oc_memcpy(addressof(x), addressof(x_tmp), sizeof(x));
        oc_memcpy(addressof(y), addressof(y_tmp), sizeof(y));
        oc_memcpy(addressof(z), addressof(z_tmp), sizeof(z));
        return true;
    }
    bool initialed_{initial()};

public:
#include "swizzle_inl/swizzle_3.inl.h"
};

template<typename T>
struct Computable<Vector<T, 4>>
    : detail::EnableStaticCast<Computable<Vector<T, 4>>>,
      detail::EnableBitwiseCast<Computable<Vector<T, 4>>>,
      detail::EnableGetMemberByIndex<Computable<Vector<T, 4>>>,
      detail::EnableSubscriptAccess<Computable<Vector<T, 4>>> {
    using this_type = Vector<T, 4>;
    OC_COMPUTABLE_COMMON(Computable<Vector<T, 4>>)
    OC_MAKE_ASSIGNMENT_FUNC

public:
    union {
        struct {
            Var<T> x;
            Var<T> y;
            Var<T> z;
            Var<T> w;
        };
        std::array<Var<T>, 4> arr;
    };

private:
    [[nodiscard]] bool initial() noexcept {
        Var<T> x_tmp(Function::current()->swizzle(Type::of<T>(), expression(), 0, 1));
        Var<T> y_tmp(Function::current()->swizzle(Type::of<T>(), expression(), 1, 1));
        Var<T> z_tmp(Function::current()->swizzle(Type::of<T>(), expression(), 2, 1));
        Var<T> w_tmp(Function::current()->swizzle(Type::of<T>(), expression(), 3, 1));
        oc_memcpy(addressof(x), addressof(x_tmp), sizeof(x));
        oc_memcpy(addressof(y), addressof(y_tmp), sizeof(y));
        oc_memcpy(addressof(z), addressof(z_tmp), sizeof(z));
        oc_memcpy(addressof(w), addressof(w_tmp), sizeof(w));
        return true;
    }
    bool initialed_{initial()};

public:
#include "swizzle_inl/swizzle_4.inl.h"
};

template<size_t N>
struct Computable<Matrix<N>>
    : detail::EnableGetMemberByIndex<Computable<Matrix<N>>>,
      detail::EnableSubscriptAccess<Computable<Matrix<N>>> {
    OC_COMPUTABLE_COMMON(Computable<Matrix<N>>)
    using this_type = Matrix<N>;
    OC_MAKE_ASSIGNMENT_FUNC
};

template<typename T, size_t N>
struct Computable<ocarina::array<T, N>>
    : detail::EnableSubscriptAccess<Computable<ocarina::array<T, N>>>,
      detail::EnableGetMemberByIndex<Computable<ocarina::array<T, N>>> {
    using this_type = ocarina::array<T, N>;
    OC_COMPUTABLE_COMMON(Computable<ocarina::array<T, N>>)
public:
    void set(const this_type &t) {
        for (int i = 0; i < N; ++i) {
            (*this)[i] = t[i];
        }
    }
    void set(const Var<array<T, N>> &t) {
        for (int i = 0; i < N; ++i) {
            (*this)[i] = t[i];
        }
    }
    void set(const Var<Vector<T, N>> &t) {
        for (int i = 0; i < N; ++i) {
            (*this)[i] = t[i];
        }
    }
    [[nodiscard]] Var<T> as_scalar() const noexcept {
        return (*this)[0];
    }
    template<size_t Dim = N>
    [[nodiscard]] auto as_vec() const noexcept {
        static_assert(Dim <= N);
        if constexpr (Dim == 1) {
            return as_scalar();
        } else {
            Var<Vector<T, Dim>> ret;
            for (int i = 0; i < Dim; ++i) {
                ret[i] = (*this)[i];
            }
            return ret;
        }
    }
    [[nodiscard]] Var<Vector<T, 2>> as_vec2() const noexcept { return as_vec<2>(); }
    [[nodiscard]] Var<Vector<T, 3>> as_vec3() const noexcept { return as_vec<3>(); }
    [[nodiscard]] Var<Vector<T, 4>> as_vec4() const noexcept { return as_vec<4>(); }
#include "swizzle_inl/array_swizzle.inl.h"
};

template<typename T, size_t N>
struct Computable<T[N]>
    : detail::EnableSubscriptAccess<Computable<T[N]>>,
      detail::EnableGetMemberByIndex<Computable<T[N]>> {
    using this_type = T[N];
    OC_COMPUTABLE_COMMON(Computable<T[N]>)
public:
    void set(const this_type &t) {
        for (int i = 0; i < N; ++i) {
            (*this)[i] = t[i];
        }
    }
};

template<typename T>
struct Computable<Buffer<T>>
    : detail::EnableReadAndWrite<Computable<Buffer<T>>>,
      detail::EnableSubscriptAccess<Computable<Buffer<T>>>,
      detail::BufferAsAtomicAddress<Computable<Buffer<T>>, T> {
    OC_COMPUTABLE_COMMON(Computable<Buffer<T>>)

public:
    void set(const BufferProxy<T> &buffer) noexcept {
        /// empty
    }
    template<typename int_type = uint64t>
    [[nodiscard]] auto size() const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<int_type>(), CallOp::BUFFER_SIZE, {expression()});
        return eval<int_type>(expr);
    }
};

template<>
struct Computable<ByteBuffer>
    : detail::EnableByteLoadAndStore<Computable<ByteBuffer>>,
      detail::BufferAsAtomicAddress<Computable<ByteBuffer>, uint> {
    OC_COMPUTABLE_COMMON(Computable<ByteBuffer>)

public:
    template<typename int_type = uint>
    [[nodiscard]] auto size_in_byte() const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<int_type>(), CallOp::BYTE_BUFFER_SIZE, {expression()});
        return eval<int_type>(expr);
    }

    template<typename int_type = uint>
    [[nodiscard]] auto size() const noexcept {
        return size_in_byte<int_type>();
    }

    template<typename Elm>
    [[nodiscard]] SOAView<Elm, Computable<ByteBuffer>> soa_view() noexcept {
        return SOAView<Elm, Computable<ByteBuffer>>(*this);
    }
};

template<>
struct Computable<Texture>
    : detail::EnableTextureSample<Computable<Texture>>,
      detail::EnableTextureReadAndWrite<Computable<Texture>> {
    OC_COMPUTABLE_COMMON(Computable<Texture>)
};

template<typename T>
struct Computable<Texture2D<T>>
    : detail::EnableTextureSample<Computable<Texture2D<T>>>,
      detail::EnableTextureReadAndWrite<Computable<Texture2D<T>>> {
    OC_COMPUTABLE_COMMON(Computable<Texture2D<T>>)
};

template<>
struct Computable<Accel> {
public:
    [[nodiscard]] inline Var<Hit> trace_closest(const Var<Ray> &ray) const noexcept;// implement in computable.inl
    [[nodiscard]] inline Var<bool> trace_any(const Var<Ray> &ray) const noexcept;   // implement in computable.inl
    OC_COMPUTABLE_COMMON(Computable<Accel>)
};

template<typename... T>
struct Computable<ocarina::tuple<T...>> {
    using Tuple = ocarina::tuple<T...>;
    OC_COMPUTABLE_COMMON(Computable<ocarina::tuple<T...>>)

private:
    template<uint i>
    requires(i >= sizeof...(T))
    void _assignment(const Tuple &t) noexcept {}
    template<uint i = 0>
    requires(i < sizeof...(T))
    void _assignment(const Tuple &t) noexcept {
        set<i>(ocarina::get<i>(t));
        _assignment<i + 1>(t);
    }

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

    void set(const Tuple &t) {
        _assignment(t);
    };
};

}// namespace detail

template<typename T>
class BindlessArrayBuffer {
    static_assert(is_valid_buffer_element_v<T>);

private:
    const Expression *_bindless_array{nullptr};
    const Expression *_index{nullptr};

public:
    BindlessArrayBuffer(const Expression *array, const Expression *index) noexcept
        : _bindless_array{array}, _index{index} {}

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] Var<T> read(Index index) const noexcept {
        if constexpr (is_dsl_v<Index>) {
            index = detail::correct_index(index, size(), typeid(*this).name(), traceback_string());
        }
        const CallExpr *expr = Function::current()->call_builtin(Type::of<T>(), CallOp::BINDLESS_ARRAY_BUFFER_READ,
                                                                 {_bindless_array, _index, OC_EXPR(index)});
        return eval<T>(expr);
    }

    template<typename Size = uint>
    [[nodiscard]] Var<Size> size_in_byte() const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<T>(), CallOp::BINDLESS_ARRAY_BUFFER_SIZE,
                                                                 {_bindless_array, _index});
        return eval<Size>(expr);
    }

    template<typename Size = uint>
    [[nodiscard]] Var<Size> size() const noexcept {
        Var<Size> ret = size_in_byte();
        return detail::divide(ret, static_cast<uint>(sizeof(T)));
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<T, expr_value_t<Val>>
    void write(Index index, Val &&elm) {
        if constexpr (is_dsl_v<Index>) {
            index = detail::correct_index(index, size(), typeid(*this).name(), traceback_string());
        }
        const CallExpr *expr = Function::current()->call_builtin(nullptr, CallOp::BINDLESS_ARRAY_BUFFER_WRITE,
                                                                 {_bindless_array, _index, OC_EXPR(index), OC_EXPR(elm)});
        Function::current()->expr_statement(expr);
    }
};

class BindlessArrayByteBuffer {
private:
    const Expression *_bindless_array{nullptr};
    const Expression *_index{nullptr};

public:
    BindlessArrayByteBuffer(const Expression *array, const Expression *index) noexcept
        : _bindless_array{array}, _index{index} {}

    template<typename T, typename Offset>
    requires concepts::integral<expr_value_t<Offset>>
    [[nodiscard]] Var<T> load_as(Offset &&offset) const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<T>(), CallOp::BINDLESS_ARRAY_BYTE_BUFFER_READ,
                                                                 {_bindless_array, _index, OC_EXPR(offset)});
        return eval<T>(expr);
    }

    template<typename Elm, typename Offset, typename Size = uint>
    requires is_integral_expr_v<Offset>
    void store(Offset &&offset, const Elm &val, bool check_boundary = true) noexcept {
        if (check_boundary) {
            offset = detail::correct_index(offset, size_in_byte<Size>(), typeid(*this).name(), traceback_string());
        }
        const CallExpr *expr = Function::current()->call_builtin(nullptr, CallOp::BINDLESS_ARRAY_BYTE_BUFFER_WRITE,
                                                                 {_bindless_array, _index,
                                                                  OC_EXPR(offset), OC_EXPR(val)});
        Function::current()->expr_statement(expr);
    }

    template<typename T = float, typename Size = uint>
    [[nodiscard]] Var<Size> size_in_byte() const noexcept {
        const CallExpr *expr = Function::current()->call_builtin(Type::of<T>(), CallOp::BINDLESS_ARRAY_BUFFER_SIZE,
                                                                 {_bindless_array, _index});
        return eval<Size>(expr);
    }

    template<typename Size = uint>
    [[nodiscard]] Var<Size> size() const noexcept {
        Var<Size> ret = size_in_byte<float, Size>();
        return detail::divide(ret, static_cast<uint>(sizeof(float)));
    }

    template<typename Elm>
    [[nodiscard]] SOAView<Elm, BindlessArrayByteBuffer> soa_view() noexcept {
        return SOAView<Elm, BindlessArrayByteBuffer>(*this);
    }

    template<typename T, typename Offset>
    [[nodiscard]] DynamicArray<T> load_dynamic_array(uint array_size, Offset &&offset)
        const noexcept;// implement in dsl/array.h
};

class BindlessArrayTexture {
private:
    const Expression *_bindless_array{nullptr};
    const Expression *_index{nullptr};

public:
    BindlessArrayTexture(const Expression *array, const Expression *index) noexcept
        : _bindless_array{array}, _index{index} {}

    template<typename U, typename V, typename W>
    requires(is_all_floating_point_expr_v<U, V, W>)
    OC_NODISCARD DynamicArray<float> sample(uint channel_num, const U &u, const V &v, const W &w)
        const noexcept;// implement in dsl/array.h

    template<typename U, typename V>
    requires(is_all_floating_point_expr_v<U, V>)
    OC_NODISCARD DynamicArray<float> sample(uint channel_num, const U &u, const V &v)
        const noexcept;// implement in dsl/array.h

    template<typename UVW>
    requires(is_float_vector3_v<expr_value_t<UVW>>)
    OC_NODISCARD DynamicArray<float> sample(uint channel_num, const UVW &uvw)
        const noexcept;// implement in dsl/array.h

    template<typename UV>
    requires(is_float_vector2_v<expr_value_t<UV>>)
    OC_NODISCARD DynamicArray<float> sample(uint channel_num, const UV &uv)
        const noexcept;// implement in dsl/array.h
};

namespace detail {
template<>
struct Computable<BindlessArray> {
    OC_COMPUTABLE_COMMON(Computable<BindlessArray>)
public:
    template<typename T, typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] BindlessArrayBuffer<T> buffer_var(Index index, const string &desc = "",
                                                    uint buffer_num = 0) const noexcept {
        if (buffer_num != 0) {
            if constexpr (is_integral_v<Index>) {
                OC_ASSERT(index <= buffer_num);
            } else {
                index = correct_index(index, buffer_num, desc, traceback_string(1));
            }
        }
        return BindlessArrayBuffer<T>(expression(), OC_EXPR(index));
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] BindlessArrayTexture tex_var(Index index, const string &desc = "",
                                               uint tex_num = 0) const noexcept {
        if (tex_num != 0) {
            if constexpr (is_integral_v<Index>) {
                OC_ASSERT(index <= tex_num);
            } else {
                index = correct_index(index, tex_num, desc, traceback_string(1));
            }
        }
        return BindlessArrayTexture(expression(), OC_EXPR(index));
    }

    template<typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] BindlessArrayByteBuffer byte_buffer_var(Index index, const string &desc = "",
                                                          uint buffer_num = 0) const noexcept {
        if (buffer_num != 0) {
            if constexpr (is_integral_v<Index>) {
                OC_ASSERT(index <= buffer_num);
            } else {
                index = correct_index(index, buffer_num, desc, traceback_string(1));
            }
        }
        return BindlessArrayByteBuffer(expression(), OC_EXPR(index));
    }

    template<typename Elm, typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] SOAView<Elm, BindlessArrayByteBuffer> soa_view(Index &&index, const string &desc = "",
                                                                 uint buffer_num = 0) noexcept {
        return byte_buffer_var(OC_FORWARD(index), desc, buffer_num).template soa_view<Elm>();
    }
};

#define OC_MAKE_STRUCT_MEMBER(m)                                                                                             \
    dsl_t<std::remove_cvref_t<decltype(this_type::m)>>(m){Function::current()->member(Type::of<this_type>()->get_member(#m), \
                                                                                      expression(),                          \
                                                                                      ocarina::struct_member_tuple<this_type>::member_index(#m))};

#define OC_MAKE_MEMBER_ASSIGNMENT(m) \
    m.set(_t_.m);

#define OC_MAKE_COMPUTABLE_BODY(S, ...)                   \
    namespace ocarina {                                   \
    namespace detail {                                    \
    template<>                                            \
    struct Computable<S> {                                \
    public:                                               \
        using this_type = S;                              \
        static constexpr auto cname = #S;                 \
        OC_COMPUTABLE_COMMON(S)                           \
    public:                                               \
        void                                              \
        set(const this_type &_t_) {                       \
            MAP(OC_MAKE_MEMBER_ASSIGNMENT, ##__VA_ARGS__) \
        }                                                 \
                                                          \
    public:                                               \
        MAP(OC_MAKE_STRUCT_MEMBER, ##__VA_ARGS__)         \
    };                                                    \
    }                                                     \
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