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

uint FunctionBuilder::_next_variable_uid() noexcept {
    auto uid = static_cast<uint32_t>(_variable_usages.size());
    _variable_usages.emplace_back(Usage::NONE);
    return uid;
}

const RefExpr *FunctionBuilder::reference_argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::REFERENCE, _next_variable_uid());
    _arguments.push_back(variable);
    return nullptr;
}

const RefExpr *FunctionBuilder::argument(const Type *type) noexcept {
    return nullptr;
}

}// namespace katana