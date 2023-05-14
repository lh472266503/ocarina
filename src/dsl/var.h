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

#define OC_MAKE_VAR_COMMON                                                                         \
    explicit Var(const ocarina::Expression *expression) noexcept                                   \
        : ocarina::detail::Computable<this_type>(expression) {}                                    \
    Var() noexcept : Var(ocarina::Function::current()->local(ocarina::Type::of<this_type>())) {}   \
    Var(Var &&another) noexcept : Var(OC_EXPR(another)) {}                                         \
    Var(const Var &another) noexcept : Var() { ocarina::detail::assign(*this, another); }          \
    template<typename Arg>                                                                         \
    requires ocarina::concepts::non_pointer<std::remove_cvref_t<Arg>> &&                           \
        concepts::different<std::remove_cvref_t<Arg>, Var<this_type>> &&                           \
        OC_ASSIGN_CHECK(ocarina::expr_value_t<this_type>, ocarina::expr_value_t<Arg>)              \
    Var(Arg &&arg) : Var() { ocarina::detail::assign(*this, std::forward<Arg>(arg)); }             \
    explicit Var(ocarina::detail::ArgumentCreation) noexcept                                       \
        : Var(ocarina::Function::current()->argument(ocarina::Type::of<this_type>())) {}           \
    explicit Var(ocarina::detail::ReferenceArgumentCreation) noexcept                              \
        : Var(ocarina::Function::current()->reference_argument(ocarina::Type::of<this_type>())) {} \
    template<typename Arg>                                                                         \
    requires OC_ASSIGN_CHECK(ocarina::expr_value_t<this_type>, ocarina::expr_value_t<Arg>)         \
    void operator=(Arg &&arg) {                                                                    \
        ocarina::detail::assign(*this, std::forward<Arg>(arg));                                    \
    }                                                                                              \
    void operator=(const Var &other) {                                                             \
        ocarina::detail::assign(*this, other);                                                     \
    }

template<typename T>
struct Var : public Computable<T> {
    using this_type = T;
    OC_MAKE_VAR_COMMON
    OC_MAKE_GET_PROXY
};

template<typename T>
struct Var<Vector<T, 2>> : public Computable<Vector<T, 2>> {
    using this_type = Vector<T, 2>;
    OC_MAKE_VAR_COMMON
    OC_MAKE_GET_PROXY

    explicit Var(Var<T> val) : Var() {
        detail::assign(this->x, val);
        detail::assign(this->y, val);
    }
};

template<typename T>
struct Var<Vector<T, 3>> : public Computable<Vector<T, 3>> {
    using this_type = Vector<T, 3>;
    OC_MAKE_VAR_COMMON
    OC_MAKE_GET_PROXY
    explicit Var(Var<T> val) : Var() {
        detail::assign(this->x, val);
        detail::assign(this->y, val);
        detail::assign(this->z, val);
    }
};

template<typename T>
struct Var<Vector<T, 4>> : public Computable<Vector<T, 4>> {
    using this_type = Vector<T, 4>;
    OC_MAKE_VAR_COMMON
    OC_MAKE_GET_PROXY
    explicit Var(Var<T> val) : Var() {
        detail::assign(this->x, val);
        detail::assign(this->y, val);
        detail::assign(this->z, val);
        detail::assign(this->w, val);
    }
};

template<typename T>
using BufferVar = Var<Buffer<T>>;

using TextureVar = Var<Texture>;

using ResourceArrayVar = Var<ResourceArray>;

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