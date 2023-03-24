//
// Created by Zero on 21/04/2022.
//

#include "expression.h"
#include "function.h"

namespace ocarina {

void RefExpr::_mark(Usage usage) const noexcept {
    Function::current()->mark_variable_usage(
        _variable.uid(), usage);
}

uint64_t LiteralExpr::_compute_hash() const noexcept {
    return ocarina::visit([&](auto &&arg) { return hash64(std::forward<decltype(arg)>(arg)); }, _value);
}
uint64_t AccessExpr::_compute_hash() const noexcept {
    uint64_t ret = _range->hash();
    for_each_index([&](const Expression *index) {
        ret = hash64(index->hash(), ret);
    });
    return ret;
}
uint64_t UnaryExpr::_compute_hash() const noexcept {
    return hash64(_op, _operand->hash());
}

uint64_t BinaryExpr::_compute_hash() const noexcept {
    auto ret = _lhs->hash();
    ret = hash64(_op, ret);
    ret = hash64(ret, _rhs->hash());
    return ret;
}

int MemberExpr::swizzle_index(int idx) const noexcept {
    int shift = (_swizzle_size - 1 - idx) * 4;
    auto org = 0xf << (_swizzle_size - 1) * 4;
    auto mask = org >> (idx * 4);
    auto ret = (mask & _member_index) >> shift;
    return ret;
}

uint64_t MemberExpr::_compute_hash() const noexcept {
    return hash64(hash64(_member_index, _swizzle_size), _parent->hash());
}

uint64_t CallExpr::_compute_hash() const noexcept {
    uint64_t ret = _function ? _function->hash() : Hash64::default_seed;
    ret = hash64(_call_op, ret);
    for (int i = 0; i < _template_args.size(); ++i) {
        ret = hash64(ret, _template_args[i]);
    }
    for (const auto &arg : _arguments) {
        ret = hash64(ret, arg->hash());
    }
    return ret;
}

}// namespace ocarina