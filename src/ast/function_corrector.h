//
// Created by Zero on 2023/12/3.
//

#pragma once

#include "function.h"

namespace ocarina {

class FunctionCorrector : public ExprVisitor, public StmtVisitor {
private:
    ocarina::stack<Function *> _function_tack;

    /// key: old expression, value: new expression
    std::map<const Expression *, const Expression *> _expr_map;

private:
    [[nodiscard]] Function *top() noexcept { return _function_tack.top(); }
    template<typename Arg>
    void push(Arg &&arg) noexcept { _function_tack.push(OC_FORWARD(arg)); }

public:
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

    explicit FunctionCorrector() = default;
    void process_ref_expr(const Expression *&expression) noexcept;
    void visit_expr(const ocarina::Expression *const &expression) noexcept;
    void apply(Function *function) noexcept;
    void traverse(Function &function) noexcept;
};

}// namespace ocarina