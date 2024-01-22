//
// Created by Zero on 2023/12/3.
//

#pragma once

#include "function.h"

namespace ocarina {

class FunctionCorrector : public ExprVisitor, public StmtVisitor {
private:
    ocarina::deque<Function *> _function_stack;

private:
    void visit(const AssignStmt *stmt) override;
    void visit(const BreakStmt *stmt) override {}
    void visit(const CommentStmt *stmt) override {}
    void visit(const ContinueStmt *stmt) override {}
    void visit(const ExprStmt *stmt) override;
    void visit(const ForStmt *stmt) override;
    void visit(const IfStmt *stmt) override;
    void visit(const LoopStmt *stmt) override;
    void visit(const ReturnStmt *stmt) override;
    void visit(const ScopeStmt *stmt) override;
    void visit(const SwitchCaseStmt *stmt) override;
    void visit(const SwitchStmt *stmt) override;
    void visit(const SwitchDefaultStmt *stmt) override;
    void visit(const PrintStmt *stmt) override {}

    void visit(const BinaryExpr *expr) override;
    void visit(const CallExpr *expr) override;
    void visit(const CastExpr *expr) override;
    void visit(const ConditionalExpr *expr) override;
    void visit(const LiteralExpr *expr) override {}
    void visit(const MemberExpr *expr) override;
    void visit(const RefExpr *expr) override {}
    void visit(const SubscriptExpr *expr) override;
    void visit(const UnaryExpr *expr) override;

    [[nodiscard]] Function *current_function() noexcept { return _function_stack.back(); }
    [[nodiscard]] Function *kernel() noexcept { return _function_stack.front(); }

    void traverse(Function &function) noexcept;
    void process_ref_expr(const Expression *&expression, Function *cur_func) noexcept;
    void visit_expr(const Expression *const &expression, Function *cur_func = nullptr) noexcept;

    [[nodiscard]] bool is_from_exterior(const Expression *expression) noexcept;
    void capture_from_invoker(const Expression *&expression, Function *cur_func) noexcept;
    void output_from_invoked(const Expression *&expression, Function *cur_func) noexcept;

public:
    explicit FunctionCorrector() = default;
    void apply(Function *function) noexcept;
};

}// namespace ocarina