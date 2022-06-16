//
// Created by Zero on 21/04/2022.
//

#include "expression.h"
#include "function.h"

namespace ocarina {
uint64_t Expression::hash() const noexcept {
    if (!_hash_computed) {
        OC_USING_SV
        _hash = hash64(_tag, hash64(_compute_hash(), hash64("__hash_expression")));
        if (_type != nullptr) { _hash = hash64(_type->hash(), _hash); }
        _hash_computed = true;
    }
    return _hash;
}

void RefExpr::_mark(Usage usage) const noexcept {
    Function::current()->mark_variable_usage(
        _variable.uid(), usage);
}

uint64_t LiteralExpr::_compute_hash() const noexcept {
    return ocarina::visit([&](auto &&arg) { return hash64(std::forward<decltype(arg)>(arg)); }, _value);
}
}// namespace ocarina