//
// Created by Zero on 21/04/2022.
//

#include "statement.h"
#include "expression.h"

namespace ocarina {

uint64_t ScopeStmt::_compute_hash() const noexcept {
    auto h = Hash64::default_seed;
    for (auto &v : _local_vars) { h = hash64(v.hash(), h); }
    for (auto &&s : _statements) { h = hash64(s->hash(), h); }
    return h;
}
uint64_t ReturnStmt::_compute_hash() const noexcept {
    return hash64(_expression == nullptr ? 0ull : _expression->hash());
}
uint64_t ExprStmt::_compute_hash() const noexcept {
    return hash64(_expression == nullptr ? 0ull : _expression->hash());
}

uint64_t AssignStmt::_compute_hash() const noexcept {
    auto hl = _lhs->hash();
    auto hr = _rhs->hash();
    return hash64(hl, hr);
}
uint64_t IfStmt::_compute_hash() const noexcept {
    auto ret = _condition->hash();
    ret = hash64(ret, true_branch()->hash());
    return hash64(ret, false_branch()->hash());
}
uint64_t CommentStmt::_compute_hash() const noexcept {
    return hash64(_string);
}

uint64_t SwitchStmt::_compute_hash() const noexcept {
    auto ret = _expression->hash();
    return hash64(ret, _body.hash());
}

SwitchCaseStmt::SwitchCaseStmt(const ocarina::Expression *expression)
    : Statement(Tag::SWITCH_CASE),
      _expr(dynamic_cast<const LiteralExpr *>(expression)) {
    OC_ASSERT(_expr != nullptr);
}

uint64_t SwitchCaseStmt::_compute_hash() const noexcept {
    auto ret = _expr->hash();
    return hash64(ret, _body.hash());
}

uint64_t SwitchDefaultStmt::_compute_hash() const noexcept {
    return _body.hash();
}

uint64_t ForStmt::_compute_hash() const noexcept {
    auto ret = _var->hash();
    ret = hash64(ret, _condition->hash());
    ret = hash64(ret, _step->hash());
    return hash64(ret, _body.hash());
}

uint64_t LoopStmt::_compute_hash() const noexcept {
    return _body.hash();
}

uint64_t PrintStmt::_compute_hash() const noexcept {
    uint64_t ret = hash64(_fmt);
    for (const Expression *expr : _args) {
        ret = hash64(ret, expr->hash());
    }
    return ret;
}

}// namespace ocarina
