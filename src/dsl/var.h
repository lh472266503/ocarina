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
}// namespace ocarina::detail

using detail::Computable;

template<typename T>
struct Var : public Computable<T> {

    using Computable<T>::Computable;

    explicit Var(const Expression *expression) noexcept
        : Computable<T>(expression) {}

    Var() noexcept : Var(Function::current()->local(Type::of<T>())) {}

    template<typename Arg>
    requires concepts::non_pointer<std::remove_cvref_t<Arg>> &&
        OC_ASSIGN_CHECK(expr_value_t<std::remove_cvref_t<T>>, expr_value_t<Arg>)
        Var(Arg &&arg) : Var() {
        detail::assign(*this, std::forward<Arg>(arg));
    }
    auto operator->() noexcept { return reinterpret_cast<Proxy<T> *>(this); }
    auto operator->() const noexcept { return reinterpret_cast<const Proxy<T> *>(this); }

    explicit Var(detail::ArgumentCreation) noexcept
        : Var(Function::current()->argument(Type::of<T>())) {
    }
    explicit Var(detail::ReferenceArgumentCreation) noexcept
        : Var(Function::current()->reference_argument(Type::of<T>())) {}

    template<typename Arg>
    requires OC_ASSIGN_CHECK(expr_value_t<std::remove_cvref_t<T>>, expr_value_t<Arg>)
    void operator=(Arg &&arg) {
        detail::assign(*this, std::forward<Arg>(arg));
    }
};

template<typename T>
Var(T &&) -> Var<expr_value_t<T>>;

}// namespace ocarina