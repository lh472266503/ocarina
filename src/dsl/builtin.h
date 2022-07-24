//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "dsl/var.h"
#include "dsl/expr.h"
#include "ast/expression.h"

namespace ocarina {

inline Var<uint3> dispatch_idx() noexcept {
    return eval<uint3>(Function::current()->dispatch_idx());
}

inline Var<uint3> block_idx() noexcept {
    return eval<uint3>(Function::current()->block_idx());
}

inline Var<uint> thread_id() noexcept {
    return eval<uint>(Function::current()->thread_id());
}

inline Var<uint> dispatch_id() noexcept {
    return eval<uint>(Function::current()->dispatch_id());
}

inline Var<uint3> thread_idx() noexcept {
    return eval<uint3>(Function::current()->thread_idx());
}

inline Var<uint3> dispatch_dim() noexcept {
    return eval<uint3>(Function::current()->dispatch_dim());
}

}// namespace ocarina