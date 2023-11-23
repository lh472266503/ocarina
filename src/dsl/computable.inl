//
// Created by Zero on 14/01/2023.
//

#include "rtx_type.h"
#include "stmt_builder.h"
#include "printer.h"

namespace ocarina::detail {

template<typename TRay>
requires std::is_same_v<expr_value_t<TRay>, Ray>
Var<bool> Computable<Accel>::trace_any(const TRay &ray)
const noexcept {
    const CallExpr *expr = Function::current()->call_builtin(Type::of<bool>(),
                                                             CallOp::TRACE_ANY,
                                                             {expression(), OC_EXPR(ray)});
    return eval<bool>(expr);
}

template<typename TRay>
requires std::is_same_v<expr_value_t<TRay>, Ray>
Var<Hit> Computable<Accel>::trace_closest(const TRay &ray)
const noexcept {
    const CallExpr *expr = Function::current()->call_builtin(Type::of<bool>(),
                                                             CallOp::TRACE_CLOSEST,
                                                             {expression(), OC_EXPR(ray)});
    return eval<Hit>(expr);
}

template<typename Index, typename Size>
inline Var<bool> over_boundary(Index &&index, Size &&size, const string &desc, const string &tb) noexcept {
    Bool ret = false;
    if_(index > size, [&] {
        ret = true;
    });
    return ret;
}

}// namespace ocarina::detail