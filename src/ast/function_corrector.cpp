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
    OC_WARNING_FORMAT(" FunctionCorrector  {} ", function->description().c_str());
    traverse(*current_function());
    function->check_context();
    _function_stack.pop_back();
}

bool FunctionCorrector::is_from_exterior(const Expression *expression) noexcept {
    return std::find(_function_stack.begin(), _function_stack.end(),
                     expression->context()) != _function_stack.end();
}

void FunctionCorrector::process_ref_expr(const Expression *&expression, Function *cur_func) noexcept {
    if (expression->context() == cur_func) {
        return;
    } else if (is_from_exterior(expression)) {
        capture_from_invoker(expression, cur_func);
    } else {
        output_from_invoked(expression, cur_func);
    }
}

void FunctionCorrector::visit_expr(const Expression *const &expression, Function *cur_func) noexcept {
    cur_func = cur_func == nullptr ? current_function() : cur_func;
    if (expression == nullptr) {
        return;
    }
    if (expression->is_ref()) {
        process_ref_expr(const_cast<const Expression *&>(expression), cur_func);
    } else if (expression->is_member()) {
        process_ref_expr(const_cast<const Expression *&>(expression), cur_func);
    } else {
        expression->accept(*this);
    }
}

void FunctionCorrector::capture_from_invoker(const Expression *&expression, Function *cur_func) noexcept {
    bool contain;
    const Expression *old_expr = expression;
    expression = cur_func->mapping_captured_argument(expression, &contain);
    if (contain) {
        return;
    }
    CallExpr *call_expr = const_cast<CallExpr *>(cur_func->call_expr());
    visit_expr(old_expr, const_cast<Function *>(call_expr->context()));
    call_expr->append_argument(old_expr);
}

void FunctionCorrector::visit(const CallExpr *const_expr) {
    CallExpr *expr = const_cast<CallExpr *>(const_expr);
    for (const Expression *const &arg : expr->_arguments) {
        visit_expr(arg);
    }
    if (expr->_function) {
        apply(const_cast<Function *>(expr->_function));
    }
}

void FunctionCorrector::output_from_invoked(const Expression *&expression, Function *cur_func) noexcept {
    auto context = const_cast<Function *>(expression->context());
    Function *invoked = context;
    vector<const Function *> path = detail::find_invoke_path(cur_func, context);

    bool ctx_contain;
    context->append_output_argument(expression, &ctx_contain);

    const RefExpr *kernel_local = kernel()->outer_to_local(expression);

    if (kernel_local) {
        expression = kernel_local;
        visit_expr(expression, cur_func);
        return;
    }

    const Expression *org_expr = expression;

    while (true) {

        bool contain;
        const RefExpr *ref_expr = nullptr;

        CallExpr *call_expr = const_cast<CallExpr *>(invoked->call_expr());
        Function *invoker = const_cast<Function *>(call_expr->context());

        if (invoker == kernel()) {
            /// add local variable
            ref_expr = invoker->mapping_local_variable(org_expr, &contain);
            if (!contain) {
                call_expr->append_argument(ref_expr);
            }
            if (invoker == cur_func) {
                /// using local variable
                expression = ref_expr;
            }
            break;
        } else {
            /// add passthrough argument
            ref_expr = invoker->mapping_output_argument(org_expr, &contain);
            if (invoker == cur_func) {
                /// using output argument
                expression = ref_expr;
            }
            if (!contain) {
                call_expr->append_argument(ref_expr);
            }
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