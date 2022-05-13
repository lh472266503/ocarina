//
// Created by Zero on 12/05/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "core/basic_traits.h"
#include "expr_traits.h"
#include "expr.h"
#include "ast/function_builder.h"

namespace katana {

template<typename Lhs, typename Rhs>
void assign(Lhs &&lhs, Rhs &&rhs) noexcept;

namespace detail {

template<typename T>
class Ref {
};

}// namespace detail

}// namespace katana::dsl