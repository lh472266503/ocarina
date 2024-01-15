//
// Created by Zero on 2023/12/3.
//

#include "function_corrector.h"

namespace ocarina {

void FunctionCorrector::traverse(Function &function) noexcept {
    visit(function.body());
}

void FunctionCorrector::apply(Function *function) noexcept {
    _function_tack.push_back(function);
//    traverse(*cur_func());
    _function_tack.pop_back();
}

void FunctionCorrector::process_ref_expr(const Expression *&expression) noexcept {
    if (expression->context() == cur_func()) {
        return;
    } else if (is_from_exterior(expression)) {
        capture_exterior(expression);
    } else {
        leak_from_interior(expression);
    }
}

bool FunctionCorrector::is_from_exterior(const Expression *expression) noexcept {
    return std::find(_function_tack.begin(), _function_tack.end(),
                     expression->context()) != _function_tack.end();
}

void FunctionCorrector::capture_exterior(const Expression *&expression) noexcept {
    expression = cur_func()->replace_exterior_expression(expression);
}

void FunctionCorrector::leak_from_interior(const Expression *&expression) noexcept {
}

void FunctionCorrector::visit_expr(const Expression *const &expression) noexcept {
    if (expression->is_ref()) {
        process_ref_expr((const Expression *&)expression);
    } else if(expression->is_member()) {
        process_ref_expr((const Expression *&)expression);
    } else {
        expression->accept(*this);
    }
}

void FunctionCorrector::visit(const ScopeStmt *scope) {
    for (const Statement *stmt : scope->statements()) {
        stmt->accept(*this);
    }
}

void FunctionCorrector::visit(const AssignStmt *stmt) {
    visit_expr(stmt->_lhs);
    visit_expr(stmt->_rhs);
}

void FunctionCorrector::visit(const ExprStmt *stmt) {
    visit_expr(stmt->_expression);
}

void FunctionCorrector::visit(const ForStmt *stmt) {
    visit_expr(stmt->_var);
    visit_expr(stmt->_condition);
    visit_expr(stmt->_step);
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const IfStmt *stmt) {
    visit_expr(stmt->_condition);
    stmt->true_branch()->accept(*this);
    stmt->false_branch()->accept(*this);
}

void FunctionCorrector::visit(const LoopStmt *stmt) {
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const ReturnStmt *stmt) {
    visit_expr(stmt->_expression);
}

void FunctionCorrector::visit(const SwitchCaseStmt *stmt) {
    visit_expr(stmt->_expr);
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const SwitchStmt *stmt) {
    visit_expr(stmt->_expression);
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const SwitchDefaultStmt *stmt) {
    stmt->body()->accept(*this);
}

void FunctionCorrector::visit(const BinaryExpr *expr) {
    visit_expr(expr->_lhs);
    visit_expr(expr->_rhs);
}

void FunctionCorrector::visit(const CallExpr *expr) {
    for (const Expression *const &arg : expr->_arguments) {
        visit_expr(arg);
    }
    if (expr->_function) {
        apply(const_cast<Function *>(expr->_function));
        CallExpr *call_expr = const_cast<CallExpr*>(expr);
        call_expr->_function->for_each_exterior_expr([&](const Expression *expression) {
            auto e = expression;
            if (expression->context() != cur_func()) {
                e = cur_func()->replace_exterior_expression(expression);
            }
            call_expr->_arguments.push_back(e);
        });
        if (expr->_function->description() == "Geometry::compute_surface_interaction") {
            int i = 0;
        }
        int i = 0;
    }
}

void FunctionCorrector::visit(const CastExpr *expr) {
    visit_expr(expr->_expression);
}

void FunctionCorrector::visit(const ConditionalExpr *expr) {
    visit_expr(expr->_pred);
    visit_expr(expr->_true);
    visit_expr(expr->_false);
}

void FunctionCorrector::visit(const MemberExpr *expr) {
    visit_expr(expr->_parent);
}

void FunctionCorrector::visit(const SubscriptExpr *expr) {
    visit_expr(expr->_range);
    for (const Expression *const &index : expr->_indexes) {
        visit_expr(index);
    }
}

void FunctionCorrector::visit(const UnaryExpr *expr) {
    visit_expr(expr->_operand);
}

}// namespace ocarina