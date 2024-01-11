//
// Created by Zero on 2023/12/3.
//

#pragma once

#include "function.h"

namespace ocarina {

class FunctionCorrector : public ExprVisitor, public StmtVisitor {
private:
    Function *_function{};

    /// key: old expression, value: new expression
    std::map<const Expression *, const Expression *> _expr_map;

public:
    void visit(const AssignStmt *stmt) override;
    void visit(const BinaryExpr *stmt) override;
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

    void visit(const CallExpr *expr) override {}
    void visit(const CastExpr *expr) override {}
    void visit(const ConditionalExpr *expr) override {}
    void visit(const LiteralExpr *expr) override {}
    void visit(const MemberExpr *expr) override {}
    void visit(const RefExpr *expr) override {}
    void visit(const SubscriptExpr *expr) override {}
    void visit(const UnaryExpr *expr) override {}

    explicit FunctionCorrector(Function *func) : _function(func) {}
    void apply() noexcept;
    void traverse(Function &function) noexcept;
};

}// namespace ocarina