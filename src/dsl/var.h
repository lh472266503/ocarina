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

template<typename Lhs, typename Rhs>
inline void assign(Lhs &&lhs, Rhs &&rhs) noexcept;// implement in stmt.h

template<typename T>
struct Var : public Computable<T> {
    static_assert(std::is_trivially_destructible_v<T>);

    explicit Var(const Expression * expression) noexcept
        : Computable<T>(expression) {}

    Var() noexcept : Var(Function::current()->local(Type::of<T>())) {}

    template<typename Arg>
    requires concepts::non_pointer<std::remove_cvref_t<Arg>> &&
        concepts::different<Var<T>, std::remove_cvref_t<Arg>>
        Var(Arg &&arg) : Var() {
        assign(*this, std::forward<Arg>(arg));
    }

    explicit Var(detail::ArgumentCreation) noexcept
        : Var(Function::current()->argument(Type::of<T>())) {
    }
    explicit Var(detail::ReferenceArgumentCreation) noexcept
        : Var(Function::current()->reference_argument(Type::of<T>())) {}

    Var(Var &&) noexcept = default;

    Var(const Var &) noexcept = default;

    void operator=(Var &&rhs) &noexcept {
        assign(*this, std::forward<Var>(rhs));
    }

    void operator=(const Var &rhs) &noexcept {
        assign(*this, rhs);
    }
};

template<typename T>
Var(T &&) -> Var<expr_value_t<T>>;

}// namespace ocarina