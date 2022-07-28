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

#define MAKE_VAR_COMMON                                                                       \
    explicit Var(const Expression *expression) noexcept                                       \
        : Computable<this_type>(expression) {}                                                \
    Var() noexcept : Var(Function::current()->local(Type::of<this_type>())) {}                \
    template<typename Arg>                                                                    \
    requires concepts::non_pointer<std::remove_cvref_t<Arg>> &&                               \
             OC_ASSIGN_CHECK(expr_value_t<std::remove_cvref_t<this_type>>, expr_value_t<Arg>) \
    Var(Arg &&arg) : Var() {                                                                  \
        detail::assign(*this, std::forward<Arg>(arg));                                        \
    }                                                                                         \
    OC_MAKE_GET_PROXY                                                                         \
    explicit Var(detail::ArgumentCreation) noexcept                                           \
        : Var(Function::current()->argument(Type::of<this_type>())) {                         \
    }                                                                                         \
    explicit Var(detail::ReferenceArgumentCreation) noexcept                                  \
        : Var(Function::current()->reference_argument(Type::of<this_type>())) {}              \
    template<typename Arg>                                                                    \
    requires OC_ASSIGN_CHECK(expr_value_t<std::remove_cvref_t<this_type>>, expr_value_t<Arg>) \
    void operator=(Arg &&arg) {                                                               \
        detail::assign(*this, std::forward<Arg>(arg));                                        \
    }

template<typename T>
struct Var : public Computable<T> {
    using this_type = T;
    MAKE_VAR_COMMON
};

template<typename T>
struct Var<Vector<T, 2>> : Computable<Vector<T, 2>> {
    using this_type = Vector<T, 2>;
    MAKE_VAR_COMMON
    Var(const Var<T> &x_, const Var<T> &y_)
        : Var() {
        detail::assign(this->x, x_);
        detail::assign(this->y, y_);
    }
};

template<typename T>
struct Var<Vector<T, 3>> : Computable<Vector<T, 3>> {
    using this_type = Vector<T, 3>;
    MAKE_VAR_COMMON
    Var(const Var<T> &x_, const Var<T> &y_, const Var<T> &z_)
        : Var() {
        detail::assign(this->x, x_);
        detail::assign(this->y, y_);
        detail::assign(this->z, z_);
    }
};

template<typename T>
struct Var<Vector<T, 4>> : Computable<Vector<T, 4>> {
    using this_type = Vector<T, 4>;
    MAKE_VAR_COMMON
    Var(const Var<T> &x_, const Var<T> &y_, const Var<T> &z_, const Var<T> &w_)
        : Var() {
        detail::assign(this->x, x_);
        detail::assign(this->y, y_);
        detail::assign(this->z, z_);
        detail::assign(this->w, w_);
    }
};

template<typename T>
Var(T &&) -> Var<expr_value_t<T>>;

}// namespace ocarina