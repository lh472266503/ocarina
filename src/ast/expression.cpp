//
// Created by Zero on 21/04/2022.
//

#include "expression.h"
#include "function.h"

namespace ocarina {

void RefExpr::_mark(Usage usage) const noexcept {
    const_cast<Function *>(context())->mark_variable_usage(_variable.uid(), usage);
}

uint64_t RefExpr::_compute_hash() const noexcept {
    return hash64(_variable.hash(), to_underlying(usage()));
}

Usage RefExpr::usage() const noexcept {
    return context()->variable_usage(_variable.uid());
}

uint64_t LiteralExpr::_compute_hash() const noexcept {
    uint64_t ret = ocarina::visit(
        [&](auto &&arg) {
            return hash64(OC_FORWARD(arg));
        },
        _value);
    ret = hash64(ret, _value.index());
    return ret;
}

uint64_t SubscriptExpr::_compute_hash() const noexcept {
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

uint64_t ConditionalExpr::_compute_hash() const noexcept {
    auto ret = _pred->hash();
    ret = hash64(_true, ret);
    ret = hash64(ret, _false->hash());
    return ret;
}

MemberExpr::MemberExpr(const Type *type, const Expression *parent,
                       uint16_t index, uint16_t swizzle_size, Variable variable)
    : Expression(Tag::MEMBER, type), _parent(parent),
      _member_index(index), _swizzle_size(swizzle_size),
      _variable(ocarina::move(variable)) {}

void MemberExpr::_mark(ocarina::Usage usage) const noexcept {
    const_cast<Function *>(context())->mark_variable_usage(_variable.uid(), usage);
}

Usage MemberExpr::usage() const noexcept {
    return const_cast<Function *>(context())->variable_usage(_variable.uid());
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

CallExpr::CallExpr(const Type *type, const Function *func,
                   vector<const Expression *> &&args)
    : Expression(Tag::CALL, type),
      _function(func),
      _arguments(std::move(args)) {
    const_cast<Function *>(_function)->set_call_expression(this);
}

vector<const Function *> CallExpr::call_chain() const noexcept {
    vector<const Function *> ret;
    const Function *func = context();
    while (func) {
        ret.push_back(func);
        const CallExpr *call_expr = func->call_expr();
        func = call_expr ? call_expr->context() : nullptr;
    }
    return ret;
}

void CallExpr::append_argument(const Expression *expression) noexcept {
    _arguments.push_back(expression);
}

uint64_t CallExpr::_compute_hash() const noexcept {
    uint64_t ret = _function ? _function->hash() : Hash64::default_seed;
    ret = hash64(_call_op, ret);
    ret = hash64(_function_name, ret);
    for (auto _template_arg : _template_args) {
        ret = ocarina::visit(
            [&](auto &&arg) {
                return hash64(OC_FORWARD(arg));
            },
            _template_arg);
        ret = hash64(ret, _template_arg.index());
    }
    for (const auto &arg : _arguments) {
        ret = hash64(ret, arg->hash());
    }
    return ret;
}

}// namespace ocarina