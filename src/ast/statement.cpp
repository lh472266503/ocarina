//
// Created by Zero on 21/04/2022.
//

#include "statement.h"

namespace ocarina {

bool ScopeStmt::is_reference(const Expression *expr) const noexcept {
    bool ret = false;
    for (const auto &stmt : statements()) {
        ret = ret || stmt->is_reference(expr);
    }
    return ret;
}
bool ReturnStmt::is_reference(const Expression *expr) const noexcept {
    return expr == _expression.get();
}
bool ExprStmt::is_reference(const Expression *expr) const noexcept {
    return expr == _expression.get();
}
bool AssignStmt::is_reference(const Expression *expr) const noexcept {
    return expr == _lhs.get() || expr == _rhs.get();
}
}// namespace ocarina
