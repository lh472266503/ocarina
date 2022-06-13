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
    void visit(const BreakStmt *) noexcept override;
    void visit(const ContinueStmt *) noexcept override;
    void visit(const ReturnStmt *) noexcept override;
    void visit(const ScopeStmt *) noexcept override;
    void visit(const IfStmt *) noexcept override;
    void visit(const LoopStmt *) noexcept override;
    void visit(const ExprStmt *) noexcept override;
    void visit(const SwitchStmt *) noexcept override;
    void visit(const SwitchCaseStmt *) noexcept override;
    void visit(const SwitchDefaultStmt *) noexcept override;
    void visit(const AssignStmt *) noexcept override;
    void visit(const ForStmt *) noexcept override;

    void visit(const UnaryExpr *) noexcept override;
    void visit(const BinaryExpr *) noexcept override;
    void visit(const MemberExpr *) noexcept override;
    void visit(const AccessExpr *) noexcept override;
    void visit(const LiteralExpr *) noexcept override;
    void visit(const RefExpr *) noexcept override;
    void visit(const ConstantExpr *) noexcept override;
    void visit(const CallExpr *) noexcept override;
    void visit(const CastExpr *) noexcept override;

    void visit(const Type *) noexcept override;

    virtual void _emit_type_decl() noexcept;
    virtual void _emit_variable_decl(Variable v) noexcept;
    virtual void _emit_type_name(const Type *type) noexcept;
    virtual void _emit_function(Function f) noexcept;
    virtual void _emit_arguments(Function f) noexcept;
    virtual void _emit_body(Function f) noexcept;
    virtual void _emit_variable_name(Variable v) noexcept;
    virtual void _emit_indent() noexcept;
    virtual void _emit_statements(ocarina::span<const Statement *const> stmts) noexcept;

public:
    void emit(Function func) noexcept override;

};

}// namespace ocarina