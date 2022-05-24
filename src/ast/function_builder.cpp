//
// Created by Zero on 03/05/2022.
//

#include "function_builder.h"

namespace katana {
katana::vector<FunctionBuilder *> &FunctionBuilder::_function_stack() noexcept {
    static thread_local katana::vector<FunctionBuilder *> stack;
    return stack;
}
FunctionBuilder *FunctionBuilder::current() noexcept {
    return _function_stack().back();
}
void FunctionBuilder::mark_variable_usage(uint uid, Usage usage) noexcept {
    auto old_usage = to_underlying(_variable_usages[uid]);
    auto u = static_cast<Usage>(old_usage | to_underlying(usage));
    _variable_usages[uid] = u;
}
const RefExpr *FunctionBuilder::argument(const Type *type) noexcept {
    return nullptr;
}

}// namespace katana