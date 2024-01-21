//
// Created by Zero on 2023/12/3.
//

#include "function_corrector.h"

namespace ocarina {

namespace detail {
bool DFS_traverse(const Function *function, const Function *target,
                  vector<const Function *> *path) noexcept {
    path->push_back(function);
    auto used_func = function->used_custom_func();
    if (function == target) {
        return true;
    }
    for (const auto &f : used_func) {
        if (DFS_traverse(f.get(), target, path)) {
            return true;
        }
    }
    path->pop_back();
    return false;
}

vector<const Function *> find_invoke_path(Function *function,
                                          const Function *target) noexcept {
    vector<const Function *> path;
    detail::DFS_traverse(function, target, &path);
    return path;
}
}// namespace detail

void FunctionCorrector::traverse(Function &function) noexcept {
    visit(function.body());
}

void FunctionCorrector::apply(Function *function) noexcept {
    _function_stack.push_back(function);
    traverse(*cur_func());
    _function_stack.pop_back();
}

bool FunctionCorrector::is_from_exterior(const Expression *expression) noexcept {
    return std::find(_function_stack.begin(), _function_stack.end(),
                     expression->context()) != _function_stack.end();
}

void FunctionCorrector::process_ref_expr(const Expression *&expression, Function *func) noexcept {
    if (expression->context() == cur_func()) {
        return;
    } else if (is_from_exterior(expression)) {
        capture_from_invoker(expression, func);
    } else {
        output_from_invoked(expression, func);
    }
}

void FunctionCorrector::visit_expr(const Expression *const &expression, Function *func) noexcept {
    func = func == nullptr ? cur_func() : func;
    if (expression == nullptr) {
        return;
    }
    if (expression->is_ref()) {
        process_ref_expr(const_cast<const Expression *&>(expression), func);
    } else if (expression->is_member()) {
        process_ref_expr(const_cast<const Expression *&>(expression), func);
    } else {
        expression->accept(*this);
    }
}

void FunctionCorrector::capture_from_invoker(const Expression *&expression, Function *func) noexcept {
    bool contain;
    const Expression *old_expr = expression;
    expression = func->mapping_captured_argument(expression, &contain);
    if (contain) {
        return;
    }
    CallExpr *call_expr = const_cast<CallExpr *>(func->call_expr());
}

void FunctionCorrector::visit(const CallExpr *const_expr) {
    CallExpr *expr = const_cast<CallExpr *>(const_expr);
    for (const Expression *const &arg : expr->_arguments) {
        visit_expr(arg);
    }
    if (expr->_function) {
        apply(const_cast<Function *>(expr->_function));
        expr->_function->for_each_invoker_expr([&](const Expression *expression) {
            visit_expr(expression);
            expr->append_argument(expression);
        });
    }
}

void FunctionCorrector::output_from_invoked(const Expression *&expression, Function *func) noexcept {
    auto context = const_cast<Function *>(expression->context());
    Function *invoked = context;
    vector<const Function *> path = detail::find_invoke_path(cur_func(), context);

    context->append_output_argument(expression);

    while (true) {

        CallExpr *call_expr = const_cast<CallExpr *>(invoked->call_expr());
        Function *invoker = const_cast<Function *>(call_expr->context());

        const RefExpr *ref_expr = nullptr;
        if (invoker == kernel()) {
            // add a local variable
            ref_expr = invoker->mapping_local_variable(expression, call_expr);
            expression = ref_expr;
            break;
        } else if (invoker == cur_func()) {
            int i = 0;
        } else {
            // add a reference output argument
            ref_expr = invoker->mapping_output_argument(expression, call_expr);
        }
        invoked = invoker;
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