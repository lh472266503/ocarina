//
// Created by Zero on 02/05/2022.
//

#pragma once

#include "core/stl.h"
#include "computable.h"
#include "arg.h"
#include "ast/function.h"
#include "core/basic_types.h"

namespace ocarina {


template<typename T>
struct Var : public detail::Computable<T> {
    static_assert(std::is_trivially_destructible_v<T>);

    explicit Var(const Expression *expression) noexcept
        : detail::Computable<T>(expression) {}

    Var() noexcept : Var(Function::current()->local(Type::of<T>())) {}

    template<typename Arg>
    requires concepts::non_pointer<std::remove_cvref_t<Arg>> &&
    concepts::assign_able<expr_value_t<std::remove_cvref_t<T>>, expr_value_t<Arg>>
        Var(Arg &&arg) : Var() {
        assign(*this, std::forward<Arg>(arg));
    }

    explicit Var(detail::ArgumentCreation) noexcept
        : Var(Function::current()->argument(Type::of<T>())) {
    }
    explicit Var(detail::ReferenceArgumentCreation) noexcept
        : Var(Function::current()->reference_argument(Type::of<T>())) {}

    template<typename Arg>
    requires concepts::assign_able<expr_value_t<std::remove_cvref_t<T>>, expr_value_t<Arg>>
    void operator=(Arg &&arg) {
        assign(*this, std::forward<Arg>(arg));
    }
};

#define OC_MAKE_VAR_BODY(S, ...)                                                                \
    template<>                                                                                  \
    struct Var<S> : public detail::Computable<S> {                                              \
        MAP(OC_MAKE_STRUCT_MEMBER, ##__VA_ARGS__)                                               \
        explicit Var(const Expression *expression) noexcept                                     \
            : detail::Computable<S>(expression) {}                                              \
                                                                                                \
        Var() noexcept : Var(Function::current()->local(Type::of<S>())) {}                      \
                                                                                                \
        template<typename Arg>                                                                  \
        requires concepts::non_pointer<std::remove_cvref_t<Arg>> &&                             \
            concepts::assign_able<expr_value_t<std::remove_cvref_t<S>>, expr_value_t<Arg>>      \
            Var(Arg &&arg) : Var() {                                                            \
            assign(*this, std::forward<Arg>(arg));                                              \
        }                                                                                       \
        explicit Var(detail::ArgumentCreation) noexcept                                         \
            : Var(Function::current()->argument(Type::of<S>())) {                               \
        }                                                                                       \
        explicit Var(detail::ReferenceArgumentCreation) noexcept                                \
            : Var(Function::current()->reference_argument(Type::of<S>())) {}                    \
        template<typename Arg>                                                                  \
        requires concepts::assign_able<expr_value_t<std::remove_cvref_t<S>>, expr_value_t<Arg>> \
        void operator=(Arg &&arg) {                                                             \
            assign(*this, std::forward<Arg>(arg));                                              \
        }                                                                                       \
    };

template<typename T>
Var(T &&) -> Var<expr_value_t<T>>;

}// namespace ocarina