//
// Created by Zero on 08/06/2022.
//

#pragma once

#include "codegen.h"
#include "ast/expression.h"
#include "ast/statement.h"
#include "ast/type.h"

namespace ocarina {

class CppCodegen : public Codegen, private ExprVisitor, private StmtVisitor, private TypeVisitor {
protected:
    void visit(const BreakStmt *stmt) noexcept override;
    void visit(const ContinueStmt *stmt) noexcept override;
    void visit(const ReturnStmt *stmt) noexcept override;
    void visit(const ScopeStmt *stmt) noexcept override;
    void visit(const IfStmt *stmt) noexcept override;
    void visit(const CommentStmt *stmt) noexcept override;
    void visit(const LoopStmt *stmt) noexcept override;
    void visit(const ExprStmt *stmt) noexcept override;
    void visit(const SwitchStmt *stmt) noexcept override;
    void visit(const SwitchCaseStmt *stmt) noexcept override;
    void visit(const SwitchDefaultStmt *stmt) noexcept override;
    void visit(const AssignStmt *stmt) noexcept override;
    void visit(const ForStmt *stmt) noexcept override;

    void visit(const UnaryExpr *expr) noexcept override;
    void visit(const BinaryExpr *expr) noexcept override;
    void visit(const MemberExpr *expr) noexcept override;
    void visit(const AccessExpr *expr) noexcept override;
    void visit(const LiteralExpr *expr) noexcept override;
    void visit(const RefExpr *expr) noexcept override;
    void visit(const CallExpr *expr) noexcept override;
    void visit(const CastExpr *expr) noexcept override;

    void visit(const Type *type) noexcept override;

    virtual void _emit_types_define() noexcept;
    virtual void _emit_variable_define(Variable v) noexcept;
    virtual void _emit_type_name(const Type *type) noexcept;
    virtual void _emit_function(const Function &f) noexcept;
    virtual void _emit_arguments(const Function &f) noexcept;
    virtual void _emit_body(const Function &f) noexcept;
    virtual void _emit_local_var_define(const ScopeStmt *scope) noexcept;
    virtual void _emit_variable_name(Variable v) noexcept;
    virtual void _emit_statements(ocarina::span<const Statement *const> stmts) noexcept;

public:
    void emit(const Function &func) noexcept override;

};

}// namespace ocarina