//
// Created by Zero on 02/05/2022.
//

#pragma once

#include "core/stl.h"
#include "computable.h"
#include "ast/function.h"
#include "expr.h"
#include "core/basic_types.h"

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
    explicit Var(const ocarina::Expression *expression) noexcept
        : ocarina::detail::Computable<this_type>(expression) {}
    Var() noexcept
        : Var(ocarina::Function::current()->local(ocarina::Type::of<this_type>())) {
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
};

template<typename T>
using BufferVar = Var<Buffer<T>>;

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