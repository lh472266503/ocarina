//
// Created by Zero on 2023/12/3.
//

#include "function_corrector.h"

namespace ocarina {

void FunctionCorrector::traverse(Function &function) noexcept {
    visit(function.body());
    function.correct_used_structures();
    bool valid = function.check_context();
    OC_ERROR_IF_NOT(valid, "FunctionCorrector error: invalid function ", function.description().c_str());
}

void FunctionCorrector::apply(Function *function) noexcept {
    _function_stack.push_back(function);
    traverse(*current_function());
    if (current_function()->is_kernel()) {
        /// Split parameter structure into separate elements
        _stage = SplitParamStruct;
        current_function()->splitting_arguments();
        traverse(*current_function());
    }
    _function_stack.pop_back();
}

bool FunctionCorrector::is_from_invoker(const Expression *expression) noexcept {
    return std::find(_function_stack.begin(), _function_stack.end(),
                     expression->context()) != _function_stack.end();
}

void FunctionCorrector::process_capture(const Expression *&expression, Function *cur_func) noexcept {
    if (expression->context() == cur_func) {
        return;
    }
    auto bit_or = [](Usage lhs, Usage rhs) {
        return Usage(to_underlying(lhs) | to_underlying(rhs));
    };
    const Expression *old_expr = expression;
    if (is_from_invoker(expression)) {
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
        process_capture(const_cast<const Expression *&>(expression), cur_func);
    } else if (expression->is_member()) {
        switch (_stage) {
            case ProcessCapture:
                process_capture(const_cast<const Expression *&>(expression), cur_func);
                break;
            case SplitParamStruct:
                process_member_expr(const_cast<const Expression *&>(expression));
                break;
        }
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

namespace detail {

void correct_usage(const CallExpr *expr) noexcept {
    vector<Variable> formal_arguments = expr->function()->all_arguments();
    OC_ERROR_IF(expr->arguments().size() != formal_arguments.size());

    auto bit_or = [](Usage lhs, Usage rhs) {
        return Usage(to_underlying(lhs) | to_underlying(rhs));
    };

    for (int i = 0; i < formal_arguments.size(); ++i) {
        Variable formal_arg = formal_arguments[i];
        Usage &formal_arg_usage = const_cast<Usage &>(expr->function()->variable_usage(formal_arg.uid()));

        const Expression *act_arg = expr->arguments()[i];
        Usage act_arg_usage = act_arg->usage();

        Usage combined = bit_or(formal_arg_usage, act_arg_usage);
        if (act_arg->type()->is_resource()) {
            formal_arg_usage = combined;
            act_arg->mark(combined);
        } else {
            act_arg->mark(combined);
        }
    }
}

}// namespace detail

void FunctionCorrector::visit(const CallExpr *const_expr) {
    CallExpr *expr = const_cast<CallExpr *>(const_expr);
    for (const Expression *const &arg : expr->_arguments) {
        visit_expr(arg);
    }
    if (!expr->_function) {
        return;
    }
    apply(const_cast<Function *>(expr->_function));
    detail::correct_usage(expr);
}

void FunctionCorrector::output_from_invoked(const Expression *&expression, Function *cur_func) noexcept {
    auto context = const_cast<Function *>(expression->context());
    Function *invoked = context;
    Function *invoker = nullptr;
    bool in_path = false;
    context->append_output_argument(expression, nullptr);
    const Expression *org_expr = expression;

    /// foreach invoke path,output target variable layer by layer to the kernel (top-level function)
    while (true) {
        CallExpr *call_expr = const_cast<CallExpr *>(invoked->call_expr());
        invoker = const_cast<Function *>(call_expr->context());
        if (invoked == cur_func || invoker == cur_func) {
            in_path = true;
        }
        bool contain;
        const RefExpr *ref_expr;
        if (invoker == kernel()) {
            ref_expr = invoker->mapping_local_variable(org_expr, &contain);
            if (!contain) {
                call_expr->append_argument(ref_expr);
            }
            break;
        } else {
            ref_expr = invoker->mapping_output_argument(org_expr, &contain);
            if (!contain) {
                call_expr->append_argument(ref_expr);
            }
        }
        invoked = invoker;
    }
    if (in_path) {
        if (cur_func == kernel()) {
            expression = cur_func->outer_to_local(org_expr);
        } else {
            expression = cur_func->outer_to_argument(org_expr);
        }
    } else {
        const RefExpr *kernel_expr = kernel()->outer_to_local(expression);
        expression = kernel_expr;
        capture_from_invoker(expression, cur_func);
    }
}

void FunctionCorrector::visit(const ScopeStmt *scope) {
    for (const Statement *stmt : scope->statements()) {
        stmt->accept(*this);
    }
}

void FunctionCorrector::visit(const AssignStmt *stmt) {
    visit_expr(stmt->_lhs);
    OC_ERROR_IF(stmt->lhs()->type()->is_resource());
    stmt->lhs()->mark(Usage::WRITE);
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
    OC_ERROR_IF(_stage == ProcessCapture);
}

void FunctionCorrector::process_member_expr(const Expression *&expression) noexcept {
    auto member_expr = dynamic_cast<const MemberExpr *>(expression);
    if (member_expr->parent()->type()->is_param_struct()) {
        process_param_struct(expression);
    } else {
        visit_expr(member_expr->_parent);
    }
}

void FunctionCorrector::process_param_struct(const Expression *&expression) noexcept {
    const MemberExpr *member_expr = dynamic_cast<const MemberExpr *>(expression);
    const Expression *parent = member_expr->parent();
    vector<int> path;
    path.push_back(member_expr->member_index());
    do {
        const MemberExpr *member_parent = dynamic_cast<const MemberExpr *>(parent);
        if (member_parent) {
            parent = member_parent->parent();
            path.push_back(member_parent->member_index());
        } else if (parent->tag() == Expression::Tag::SUBSCRIPT) {
            const SubscriptExpr *subscript_expr = dynamic_cast<const SubscriptExpr *>(parent);
            parent = subscript_expr->range();
            member_parent = dynamic_cast<const MemberExpr *>(parent);
            path.push_back(member_parent->member_index());
        }
    } while (parent->is_member());
    switch (parent->tag()) {
        case Expression::Tag::REF: {
            const RefExpr *ref_expr = dynamic_cast<const RefExpr *>(parent);
            path.push_back(ref_expr->variable().uid());
            std::reverse(path.begin(), path.end());
            break;
        }
        case Expression::Tag::SUBSCRIPT: {
            const SubscriptExpr *subscript_expr = dynamic_cast<const SubscriptExpr *>(parent);
        }
        default:
            break;
    }
    kernel()->replace_param_struct_member(path, expression);
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