//
// Created by Zero on 21/04/2022.
//

#include "statement.h"
#include "expression.h"

namespace ocarina {

bool ScopeStmt::is_reference(const Expression *expr) const noexcept {
    bool ret = false;
    for (const auto &stmt : statements()) {
        ret = ret || stmt->is_reference(expr);
    }
    return ret;
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
}// namespace ocarina
