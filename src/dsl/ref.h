//
// Created by Zero on 12/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "core/basic_traits.h"
#include "expr_traits.h"

namespace katana::dsl {

template<typename Lhs, typename Rhs>
void assign(Lhs &&lhs, Rhs &&rhs) noexcept;

namespace detail {

template<typename T>
struct ExprEnableStaticCast {

    template<typename Dest>
    requires concepts::static_convertible<expr_value_t<T>, expr_value_t<Dest>>
    KTN_NODISCARD auto cast() const noexcept {

    }
};

template<typename T>
struct ExprEnableBitwiseCast {

};

template<typename T>
class Ref {
};

}// namespace detail

}// namespace katana::dsl