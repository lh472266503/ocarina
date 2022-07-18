//
// Created by Zero on 21/04/2022.
//

#include "statement.h"
#include "expression.h"

namespace ocarina {

uint64_t Statement::hash() const noexcept {
    if (!_hash_computed) {
        OC_USING_SV
        uint64_t h = _compute_hash();
        _hash = hash64(_tag, hash64(h, hash64("__hash_statement"sv)));
        _hash_computed = true;
    }
    return _hash;
}

bool ScopeStmt::is_reference(const Expression *expr) const noexcept {
    bool ret = false;
    for (const auto &stmt : statements()) {
        ret = ret || stmt->is_reference(expr);
    }
    return ret;
}
uint64_t ScopeStmt::_compute_hash() const noexcept {
    auto h = Hash64::default_seed;
    for (auto &v : _local_vars) { h = hash64(v.hash(), h); }
    for (auto &&s : _statements) { h = hash64(s->hash(), h); }
    return h;
}
bool ReturnStmt::is_reference(const Expression *expr) const noexcept {
    return expr == _expression;
}
uint64_t ReturnStmt::_compute_hash() const noexcept {
    return hash64(_expression == nullptr ? 0ull : _expression->hash());
}
bool ExprStmt::is_reference(const Expression *expr) const noexcept {
    return expr == _expression;
}
uint64_t ExprStmt::_compute_hash() const noexcept {
    return hash64(_expression == nullptr ? 0ull : _expression->hash());
}
bool AssignStmt::is_reference(const Expression *expr) const noexcept {
    return expr == _lhs || expr == _rhs;
}
uint64_t AssignStmt::_compute_hash() const noexcept {
    auto hl = _lhs->hash();
    auto hr = _rhs->hash();
    return hash64(hl, hr);
}
uint64_t IfStmt::_compute_hash() const noexcept {
    auto ret = _condition->hash();
    ret = hash64(ret, true_branch()->hash());
    return hash64(ret, false_branch()->hash());;
}
uint64_t CommentStmt::_compute_hash() const noexcept {
    return hash64(_string);
}

uint64_t SwitchStmt::_compute_hash() const noexcept {
    auto ret = _expression->hash();
    return hash64(ret, _body.hash());
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
    uint64_t ret = Hash64::default_seed;
    for (const Expression *expr : _args) {
        ret = hash64(ret, expr->hash());
    }
    return ret;
}

}// namespace ocarina
