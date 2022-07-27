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

#define MAKE_VAR_COMMON

template<typename T>
struct Var : public Computable<T> {

    using real_ty = T;

    explicit Var(const Expression *expression) noexcept
        : Computable<real_ty>(expression) {}

    Var() noexcept : Var(Function::current()->local(Type::of<real_ty>())) {}

    template<typename Arg>
    requires concepts::non_pointer<std::remove_cvref_t<Arg>> &&
        OC_ASSIGN_CHECK(expr_value_t<std::remove_cvref_t<real_ty>>, expr_value_t<Arg>)
    Var(Arg &&arg) : Var() {
        detail::assign(*this, std::forward<Arg>(arg));
    }

    OC_MAKE_GET_PROXY

    explicit Var(detail::ArgumentCreation) noexcept
        : Var(Function::current()->argument(Type::of<real_ty>())) {
    }
    explicit Var(detail::ReferenceArgumentCreation) noexcept
        : Var(Function::current()->reference_argument(Type::of<real_ty>())) {}

    template<typename Arg>
    requires OC_ASSIGN_CHECK(expr_value_t<std::remove_cvref_t<real_ty>>, expr_value_t<Arg>)
    void operator=(Arg &&arg) {
        detail::assign(*this, std::forward<Arg>(arg));
    }
};

//template<typename T>
//struct Var<Vector<T, 3>> : Computable<Vector<T, 3>> {
//    using vector_ty = Vector<T, 3>;
//    explicit Var(const Expression *expression) noexcept
//        : Computable<vector_ty>(expression) {}
//
//    Var() noexcept : Var(Function::current()->local(Type::of<vector_ty>())) {}
//
//    template<typename Arg>
//    requires concepts::non_pointer<std::remove_cvref_t<Arg>> &&
//        OC_ASSIGN_CHECK(expr_value_t<std::remove_cvref_t<vector_ty>>, expr_value_t<Arg>)
//    Var(Arg &&arg) : Var() {
//        detail::assign(*this, std::forward<Arg>(arg));
//    }
//
//    Var(const Var<T> &x_, const Var<T> &y_, const Var<T> &z_)
//        : Var() {
//        //        detail::assign(this->x, x_);
//        //        this->y = y_;
//        //        this->z = z_;
//    }
//
//    explicit Var(detail::ArgumentCreation) noexcept
//        : Var(Function::current()->argument(Type::of<T>())) {
//    }
//    explicit Var(detail::ReferenceArgumentCreation) noexcept
//        : Var(Function::current()->reference_argument(Type::of<T>())) {}
//
//    template<typename Arg>
//    requires OC_ASSIGN_CHECK(expr_value_t<std::remove_cvref_t<vector_ty>>, expr_value_t<Arg>)
//    void operator=(Arg &&arg) {
//        detail::assign(*this, std::forward<Arg>(arg));
//    }
//};

template<typename T>
Var(T &&) -> Var<expr_value_t<T>>;

}// namespace ocarina