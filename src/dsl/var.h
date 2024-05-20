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

template<typename T>
struct Var : public Computable<T> {
    using this_type = T;
    using Super = Computable<T>;
    using dsl_type = Var<T>;
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

#define OC_MAKE_VAR_UNARY_FUNC(func, tag)                                           \
    [[nodiscard]] static auto call_##func(const dsl_type &val) noexcept {           \
        auto expr = Function::current()->call_builtin(Type::of<bool>(),             \
                                                      CallOp::tag, {OC_EXPR(val)}); \
        return eval<bool>(expr);                                                    \
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
    OC_MAKE_VAR_UNARY_FUNC(copysign, COPYSIGN)
    OC_MAKE_VAR_UNARY_FUNC(normalize, NORMALIZE)
    OC_MAKE_VAR_UNARY_FUNC(length, LENGTH)
    OC_MAKE_VAR_UNARY_FUNC(length_squared, LENGTH_SQUARED)

#undef OC_MAKE_VAR_LOGIC_FUNC

//    template<typename U>
//    OC_NODISCARD static auto call_max()
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