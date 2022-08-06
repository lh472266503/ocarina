//
// Created by Zero on 02/05/2022.
//

#pragma once

#include "core/stl.h"
#include "computable.h"
#include "ast/function.h"
#include "core/basic_types.h"

namespace ocarina {

namespace detail {
struct ArgumentCreation {};
struct ReferenceArgumentCreation {};
}// namespace detail

using detail::Computable;

#define OC_MAKE_VAR_COMMON                                                                                      \
    explicit Var(const ocarina::Expression *expression) noexcept                                                \
        : ocarina::detail::Computable<this_type>(expression) {}                                                 \
    Var() noexcept : Var(ocarina::Function::current()->local(ocarina::Type::of<this_type>())) {}                \
    template<typename Arg>                                                                                      \
    requires ocarina::concepts::non_pointer<std::remove_cvref_t<Arg>> &&                                        \
        OC_ASSIGN_CHECK(ocarina::expr_value_t<std::remove_cvref_t<this_type>>, ocarina::expr_value_t<Arg>)      \
    Var(Arg &&arg) : Var() {                                                                                    \
        ocarina::detail::assign(*this, std::forward<Arg>(arg));                                                 \
    }                                                                                                           \
    OC_MAKE_GET_PROXY                                                                                           \
    explicit Var(ocarina::detail::ArgumentCreation) noexcept                                                    \
        : Var(ocarina::Function::current()->argument(ocarina::Type::of<this_type>())) {                         \
    }                                                                                                           \
    explicit Var(ocarina::detail::ReferenceArgumentCreation) noexcept                                           \
        : Var(ocarina::Function::current()->reference_argument(ocarina::Type::of<this_type>())) {}              \
    template<typename Arg>                                                                                      \
    requires OC_ASSIGN_CHECK(ocarina::expr_value_t<std::remove_cvref_t<this_type>>, ocarina::expr_value_t<Arg>) \
    void operator=(Arg &&arg) {                                                                                 \
        ocarina::detail::assign(*this, std::forward<Arg>(arg));                                                 \
    }

#define OC_MAKE_VAR_ARG(member) \
    const Var<std::remove_cvref_t<decltype(this_type::member)>> &member

#define OC_MAKE_VAR_ASSIGN(member) \
    ocarina::detail::assign(this->member, member);

#define OC_MAKE_INITIAL_CTOR(...)                           \
    Var(MAP_LIST(OC_MAKE_VAR_ARG, ##__VA_ARGS__)) : Var() { \
        MAP(OC_MAKE_VAR_ASSIGN, ##__VA_ARGS__)              \
    }

#define OC_MAKE_STRUCT_VAR(T, ...)                                   \
    template<>                                                       \
    struct ocarina::Var<T> : public ocarina::detail::Computable<T> { \
        using this_type = T;                                         \
        OC_MAKE_VAR_COMMON                                           \
        OC_MAKE_INITIAL_CTOR(__VA_ARGS__)                            \
    };

template<typename T>
struct Var : public Computable<T> {
    using this_type = T;
    OC_MAKE_VAR_COMMON
};

template<typename T>
using BufferVar = Var<Buffer<T>>;

using TextureVar = Var<RHITexture>;

template<typename T>
Var(T &&) -> Var<expr_value_t<T>>;

}// namespace ocarina