//
// Created by Zero on 2023/12/3.
//

#include "function_corrector.h"

namespace ocarina {

void FunctionCorrector::traverse(Function &function) noexcept {
    visit(function.body());
}

void FunctionCorrector::apply() noexcept {
    traverse(*_function);
}

void FunctionCorrector::visit(const ScopeStmt *scope) {
    for (const Statement *stmt : scope->statements()) {
        stmt->accept(*this);
    }
}

void FunctionCorrector::visit(const AssignStmt *stmt) {
    stmt->lhs()->accept(*this);
}
void FunctionCorrector::visit(const BinaryExpr *stmt) {
}
void FunctionCorrector::visit(const ExprStmt *stmt) {
}
void FunctionCorrector::visit(const ForStmt *stmt) {
}
void FunctionCorrector::visit(const IfStmt *stmt) {
}
void FunctionCorrector::visit(const LoopStmt *stmt) {
}
void FunctionCorrector::visit(const ReturnStmt *stmt) {
}
void FunctionCorrector::visit(const SwitchCaseStmt *stmt) {
}
void FunctionCorrector::visit(const SwitchStmt *stmt) {
}
void FunctionCorrector::visit(const SwitchDefaultStmt *stmt) {
}

}// namespace ocarina