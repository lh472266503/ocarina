//
// Created by Zero on 02/05/2022.
//

#pragma once

#include "core/stl.h"
#include "computable.h"
#include "ast/function.h"
#include "expr.h"
#include "math/basic_types.h"
#include "type_trait.h"

namespace ocarina {

namespace detail {
struct ArgumentCreation {};
struct ReferenceArgumentCreation {};
}// namespace detail

using detail::Computable;

struct VarAccessor;

template<typename T>
struct Var : public Computable<T> {
    using this_type = T;
    using Super = Computable<T>;
    using Computable<T>::Computable;
    using dsl_type = Var<T>;
    friend class VarAccessor;
    explicit Var(const ocarina::Expression *expression) noexcept
        : ocarina::detail::Computable<this_type>(expression) {}
    Var() noexcept
        : Var(ocarina::Function::current()->local(ocarina::Type::of<this_type>())) {
        static_assert(!is_param_struct_v<T>);
        if constexpr (is_struct_v<T>) {
            Computable<T>::set(T{});
        }
    }
    Var(Var &&another) noexcept
        : Var(ocarina::detail::extract_expression(std::forward<decltype(another)>(another))) {}
    Var(const Var &another) noexcept
        : Var() { ocarina::detail::assign(*this, another); }
    template<typename Arg>
    requires ocarina::concepts::non_pointer<std::remove_cvref_t<Arg>> &&
             concepts::different<std::remove_cvref_t<Arg>, Var<this_type>> &&
             requires(ocarina::expr_value_t<this_type> a, ocarina::expr_value_t<Arg> b) { a = b; }
    Var(Arg &&arg)
        : Var() { ocarina::detail::assign(*this, std::forward<Arg>(arg)); }
    explicit Var(ocarina::detail::ArgumentCreation) noexcept
        : Var(ocarina::Function::current()->argument(ocarina::Type::of<this_type>())) {}
    explicit Var(ocarina::detail::ReferenceArgumentCreation) noexcept
        : Var(ocarina::Function::current()->reference_argument(ocarina::Type::of<this_type>())) {}
    template<typename Arg>
    requires requires(ocarina::expr_value_t<this_type> a, ocarina::expr_value_t<Arg> b) { a = b; }
    void operator=(Arg &&arg) {
        if constexpr (is_struct_v<Arg>) {
            Super::set(OC_FORWARD(arg));
        } else {
            ocarina::detail::assign(*this, std::forward<Arg>(arg));
        }
    }
    void operator=(const Var &other) { ocarina::detail::assign(*this, other); }
    OC_MAKE_GET_PROXY
private:
#define OC_MAKE_VAR_UNARY_FUNC(func, tag)                                           \
    [[nodiscard]] static auto call_##func(const dsl_type &val) noexcept {           \
        using ret_type = decltype(func(std::declval<T>()));                         \
        auto expr = Function::current()->call_builtin(Type::of<ret_type>(),         \
                                                      CallOp::tag, {OC_EXPR(val)}); \
        return eval<ret_type>(expr);                                                \
    }

    OC_MAKE_VAR_UNARY_FUNC(all, ALL)
    OC_MAKE_VAR_UNARY_FUNC(any, ANY)
    OC_MAKE_VAR_UNARY_FUNC(none, NONE)

    OC_MAKE_VAR_UNARY_FUNC(rcp, RCP)
    OC_MAKE_VAR_UNARY_FUNC(abs, ABS)
    OC_MAKE_VAR_UNARY_FUNC(sqrt, SQRT)
    OC_MAKE_VAR_UNARY_FUNC(sqr, SQR)
    OC_MAKE_VAR_UNARY_FUNC(exp, EXP)
    OC_MAKE_VAR_UNARY_FUNC(exp2, EXP2)
    OC_MAKE_VAR_UNARY_FUNC(exp10, EXP10)
    OC_MAKE_VAR_UNARY_FUNC(log, LOG)
    OC_MAKE_VAR_UNARY_FUNC(log2, LOG2)
    OC_MAKE_VAR_UNARY_FUNC(log10, LOG10)
    OC_MAKE_VAR_UNARY_FUNC(cos, COS)
    OC_MAKE_VAR_UNARY_FUNC(sin, SIN)
    OC_MAKE_VAR_UNARY_FUNC(tan, TAN)
    OC_MAKE_VAR_UNARY_FUNC(cosh, COSH)
    OC_MAKE_VAR_UNARY_FUNC(sinh, SINH)
    OC_MAKE_VAR_UNARY_FUNC(tanh, TANH)
    OC_MAKE_VAR_UNARY_FUNC(acos, ACOS)
    OC_MAKE_VAR_UNARY_FUNC(asin, ASIN)
    OC_MAKE_VAR_UNARY_FUNC(atan, ATAN)
    OC_MAKE_VAR_UNARY_FUNC(asinh, ASINH)
    OC_MAKE_VAR_UNARY_FUNC(acosh, ACOSH)
    OC_MAKE_VAR_UNARY_FUNC(atanh, ATANH)
    OC_MAKE_VAR_UNARY_FUNC(degrees, DEGREES)
    OC_MAKE_VAR_UNARY_FUNC(radians, RADIANS)
    OC_MAKE_VAR_UNARY_FUNC(ceil, CEIL)
    OC_MAKE_VAR_UNARY_FUNC(round, ROUND)
    OC_MAKE_VAR_UNARY_FUNC(floor, FLOOR)
    OC_MAKE_VAR_UNARY_FUNC(rsqrt, RSQRT)
    OC_MAKE_VAR_UNARY_FUNC(isinf, IS_INF)
    OC_MAKE_VAR_UNARY_FUNC(isnan, IS_NAN)
    OC_MAKE_VAR_UNARY_FUNC(fract, FRACT)
    OC_MAKE_VAR_UNARY_FUNC(saturate, SATURATE)
    OC_MAKE_VAR_UNARY_FUNC(sign, SIGN)
    OC_MAKE_VAR_UNARY_FUNC(normalize, NORMALIZE)
    OC_MAKE_VAR_UNARY_FUNC(length, LENGTH)
    OC_MAKE_VAR_UNARY_FUNC(length_squared, LENGTH_SQUARED)

    OC_MAKE_VAR_UNARY_FUNC(determinant, DETERMINANT)
    OC_MAKE_VAR_UNARY_FUNC(transpose, TRANSPOSE)
    OC_MAKE_VAR_UNARY_FUNC(inverse, INVERSE)

#undef OC_MAKE_VAR_LOGIC_FUNC

#define OC_MAKE_VAR_BINARY_FUNC(func, tag)                                           \
    OC_NODISCARD static auto call_##func(const dsl_type &lhs,                        \
                                         const dsl_type &rhs) noexcept {             \
        using ret_type = decltype(func(std::declval<T>(), std::declval<T>()));       \
        auto expr = Function::current()->call_builtin(Type::of<T>(),                 \
                                                      CallOp::tag,                   \
                                                      {OC_EXPR(lhs), OC_EXPR(rhs)}); \
        return eval<ret_type>(expr);                                                 \
    }

    OC_MAKE_VAR_BINARY_FUNC(max, MAX)
    OC_MAKE_VAR_BINARY_FUNC(min, MIN)
    OC_MAKE_VAR_BINARY_FUNC(pow, POW)
    OC_MAKE_VAR_BINARY_FUNC(fmod, FMOD)
    OC_MAKE_VAR_BINARY_FUNC(mod, MOD)
    OC_MAKE_VAR_BINARY_FUNC(copysign, COPYSIGN)
    OC_MAKE_VAR_BINARY_FUNC(atan2, ATAN2)

    OC_MAKE_VAR_BINARY_FUNC(cross, CROSS)
    OC_MAKE_VAR_BINARY_FUNC(dot, DOT)
    OC_MAKE_VAR_BINARY_FUNC(distance, DISTANCE)
    OC_MAKE_VAR_BINARY_FUNC(distance_squared, DISTANCE_SQUARED)

#undef OC_MAKE_VAR_BINARY_FUNC

#define OC_MAKE_VAR_TRIPLE_FUNC(func, tag)                             \
    OC_NODISCARD static auto call_##func(const dsl_type &a,            \
                                         const dsl_type &b,            \
                                         const dsl_type &c) noexcept { \
        using ret_type = decltype(func(std::declval<T>(),              \
                                       std::declval<T>(),              \
                                       std::declval<T>()));            \
        auto expr = Function::current()->call_builtin(Type::of<T>(),   \
                                                      CallOp::tag,     \
                                                      {OC_EXPR(a),     \
                                                       OC_EXPR(b),     \
                                                       OC_EXPR(c)});   \
        return eval<ret_type>(expr);                                   \
    }

    OC_MAKE_VAR_TRIPLE_FUNC(clamp, CLAMP)
    OC_MAKE_VAR_TRIPLE_FUNC(lerp, LERP)
    OC_MAKE_VAR_TRIPLE_FUNC(fma, FMA)

#undef OC_MAKE_VAR_TRIPLE_FUNC

    template<size_t N>
    requires(N == vector_dimension_v<T>)
    OC_NODISCARD static auto call_select(const Var<Vector<bool, N>> &pred,
                                         const dsl_type &t, const dsl_type &f) noexcept {
        const Expression *expr = Function::current()->call_builtin(Type::of<T>(),
                                                                   CallOp::SELECT,
                                                                   {OC_EXPR(pred),
                                                                    OC_EXPR(t),
                                                                    OC_EXPR(f)});
        return eval<T>(expr);
    }

    template<size_t N>
    requires(N == vector_dimension_v<T>)
    OC_NODISCARD static auto call_select(const Vector<bool, N> &pred,
                                         const dsl_type &t, const dsl_type &f) noexcept {
        return call_select(Var<Vector<bool, N>>(pred), t, f);
    }

    static auto call_select(const Var<bool> &pred, const dsl_type &t, const dsl_type &f) noexcept {
        const Expression *expr = Function::current()->call_builtin(Type::of<T>(),
                                                                   CallOp::SELECT,
                                                                   {OC_EXPR(pred),
                                                                    OC_EXPR(t),
                                                                    OC_EXPR(f)});
        return eval<T>(expr);
    }

    template<typename... Args>
    requires((sizeof...(Args) == 1 || sizeof...(Args) == 2) &&
             is_all_float_vector3_v<remove_device_t<Args>...>)
    static auto call_face_forward(const dsl_type &n, Args &&...args) {
        const Expression *expr = Function::current()->call_builtin(Type::of<T>(),
                                                                   CallOp::FACE_FORWARD,
                                                                   {OC_EXPR(n),
                                                                    OC_EXPR(args)...});
        return eval<T>(expr);
    }
};

struct VarAccessor {
public:
#define OC_MAKE_ACCESSOR_FUNC(func)                                     \
    template<typename T, typename... Args>                              \
    requires requires {                                                 \
        T::call_##func(std::declval<Args>()...);                        \
    }                                                                   \
    [[nodiscard]] static decltype(auto) func(Args &&...args) noexcept { \
        return T::call_##func(OC_FORWARD(args)...);                     \
    }
    /// unary functions
    OC_MAKE_ACCESSOR_FUNC(all)
    OC_MAKE_ACCESSOR_FUNC(any)
    OC_MAKE_ACCESSOR_FUNC(none)

    OC_MAKE_ACCESSOR_FUNC(rcp)
    OC_MAKE_ACCESSOR_FUNC(abs)
    OC_MAKE_ACCESSOR_FUNC(sqrt)
    OC_MAKE_ACCESSOR_FUNC(sqr)
    OC_MAKE_ACCESSOR_FUNC(exp)
    OC_MAKE_ACCESSOR_FUNC(exp2)
    OC_MAKE_ACCESSOR_FUNC(exp10)
    OC_MAKE_ACCESSOR_FUNC(log)
    OC_MAKE_ACCESSOR_FUNC(log2)
    OC_MAKE_ACCESSOR_FUNC(log10)
    OC_MAKE_ACCESSOR_FUNC(cos)
    OC_MAKE_ACCESSOR_FUNC(sin)
    OC_MAKE_ACCESSOR_FUNC(tan)
    OC_MAKE_ACCESSOR_FUNC(cosh)
    OC_MAKE_ACCESSOR_FUNC(sinh)
    OC_MAKE_ACCESSOR_FUNC(tanh)
    OC_MAKE_ACCESSOR_FUNC(acos)
    OC_MAKE_ACCESSOR_FUNC(asin)
    OC_MAKE_ACCESSOR_FUNC(atan)
    OC_MAKE_ACCESSOR_FUNC(asinh)
    OC_MAKE_ACCESSOR_FUNC(acosh)
    OC_MAKE_ACCESSOR_FUNC(atanh)
    OC_MAKE_ACCESSOR_FUNC(degrees)
    OC_MAKE_ACCESSOR_FUNC(radians)
    OC_MAKE_ACCESSOR_FUNC(ceil)
    OC_MAKE_ACCESSOR_FUNC(round)
    OC_MAKE_ACCESSOR_FUNC(floor)
    OC_MAKE_ACCESSOR_FUNC(rsqrt)
    OC_MAKE_ACCESSOR_FUNC(isinf)
    OC_MAKE_ACCESSOR_FUNC(isnan)
    OC_MAKE_ACCESSOR_FUNC(fract)
    OC_MAKE_ACCESSOR_FUNC(saturate)
    OC_MAKE_ACCESSOR_FUNC(sign)
    OC_MAKE_ACCESSOR_FUNC(normalize)
    OC_MAKE_ACCESSOR_FUNC(length)
    OC_MAKE_ACCESSOR_FUNC(length_squared)

    OC_MAKE_ACCESSOR_FUNC(determinant)
    OC_MAKE_ACCESSOR_FUNC(transpose)
    OC_MAKE_ACCESSOR_FUNC(inverse)

    /// binary functions
    OC_MAKE_ACCESSOR_FUNC(max)
    OC_MAKE_ACCESSOR_FUNC(min)
    OC_MAKE_ACCESSOR_FUNC(pow)
    OC_MAKE_ACCESSOR_FUNC(fmod)
    OC_MAKE_ACCESSOR_FUNC(mod)
    OC_MAKE_ACCESSOR_FUNC(copysign)
    OC_MAKE_ACCESSOR_FUNC(atan2)

    OC_MAKE_ACCESSOR_FUNC(cross)
    OC_MAKE_ACCESSOR_FUNC(dot)
    OC_MAKE_ACCESSOR_FUNC(distance)
    OC_MAKE_ACCESSOR_FUNC(distance_squared)

    /// triple functions
    OC_MAKE_ACCESSOR_FUNC(clamp)
    OC_MAKE_ACCESSOR_FUNC(lerp)
    OC_MAKE_ACCESSOR_FUNC(fma)
    OC_MAKE_ACCESSOR_FUNC(select)
    OC_MAKE_ACCESSOR_FUNC(face_forward)

#undef OC_MAKE_ACCESSOR_FUNC
};

template<typename T>
using BufferVar = Var<Buffer<T>>;

using ByteBufferVar = Var<ByteBuffer>;

using TextureVar = Var<Texture>;

using BindlessArrayVar = Var<BindlessArray>;

#define OC_MAKE_DSL_TYPE_IMPL(dsl_type, type, dim) \
    using dsl_type##dim = Var<type##dim>;

#define OC_MAKE_DSL_TYPE(dsl_type, type)     \
    OC_MAKE_DSL_TYPE_IMPL(dsl_type, type, )  \
    OC_MAKE_DSL_TYPE_IMPL(dsl_type, type, 2) \
    OC_MAKE_DSL_TYPE_IMPL(dsl_type, type, 3) \
    OC_MAKE_DSL_TYPE_IMPL(dsl_type, type, 4)

OC_MAKE_DSL_TYPE(Int, int)
OC_MAKE_DSL_TYPE(Uint, uint)
OC_MAKE_DSL_TYPE(Float, float)
OC_MAKE_DSL_TYPE(Uchar, uchar)
OC_MAKE_DSL_TYPE(Char, char)
OC_MAKE_DSL_TYPE(Short, short)
OC_MAKE_DSL_TYPE(Ushort, ushort)
OC_MAKE_DSL_TYPE(Bool, bool)

using Float2x2 = Var<float2x2>;
using Float3x3 = Var<float3x3>;
using Float4x4 = Var<float4x4>;

#undef OC_MAKE_DSL_TYPE
#undef OC_MAKE_DSL_TYPE_IMPL

template<typename T>
Var(T &&) -> Var<expr_value_t<T>>;

template<typename T>
Var(const Buffer<T> &) -> Var<Buffer<T>>;

}// namespace ocarina