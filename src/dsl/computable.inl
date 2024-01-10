//
// Created by Zero on 14/01/2023.
//

#include "rtx_type.h"
#include "stmt_builder.h"
#include "printer.h"

namespace ocarina::detail {

Var<bool> Computable<Accel>::trace_any(const Var<Ray> &ray)
const noexcept {
    const CallExpr *expr = Function::current()->call_builtin(Type::of<bool>(),
                                                             CallOp::TRACE_ANY,
                                                             {expression(), OC_EXPR(ray)});
    return eval<bool>(expr);
}

Var<Hit> Computable<Accel>::trace_closest(const Var<Ray> &ray)
const noexcept {
    const CallExpr *expr = Function::current()->call_builtin(Type::of<Hit>(),
                                                             CallOp::TRACE_CLOSEST,
                                                             {expression(), OC_EXPR(ray)});
    return eval<Hit>(expr);
}

}// namespace ocarina::detail