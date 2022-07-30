//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include "core/hash.h"
#include "variable.h"

namespace ocarina {

class ScopeStmt;
class BreakStmt;
class ContinueStmt;
class ReturnStmt;
class ExprStmt;
class AssignStmt;
class IfStmt;
class CommentStmt;
class SwitchStmt;
class SwitchCaseStmt;
class SwitchDefaultStmt;
class LoopStmt;
class ForStmt;
class PrintStmt;

class Expression;
class LiteralExpr;

struct StmtVisitor {
    virtual void visit(const BreakStmt *) = 0;
    virtual void visit(const ContinueStmt *) = 0;
    virtual void visit(const ReturnStmt *) = 0;
    virtual void visit(const ScopeStmt *) = 0;
    virtual void visit(const IfStmt *) = 0;
    virtual void visit(const LoopStmt *) = 0;
    virtual void visit(const ExprStmt *) = 0;
    virtual void visit(const SwitchStmt *) = 0;
    virtual void visit(const SwitchCaseStmt *) = 0;
    virtual void visit(const SwitchDefaultStmt *) = 0;
    virtual void visit(const AssignStmt *) = 0;
    virtual void visit(const ForStmt *) = 0;
    virtual void visit(const CommentStmt *) = 0;
    virtual void visit(const PrintStmt *) = 0;
};

#define OC_MAKE_STATEMENT_ACCEPT_VISITOR \
    void accept(StmtVisitor &visitor) const override { visitor.visit(this); }

class OC_AST_API Statement : public concepts::Noncopyable , public Hashable {
public:
    enum struct Tag : uint32_t {
        SCOPE,
        BREAK,
        CONTINUE,
        RETURN,
        IF,
        LOOP,
        EXPR,
        SWITCH,
        SWITCH_CASE,
        SWITCH_DEFAULT,
        ASSIGN,
        COMMENT,
        FOR,
        PRINT,
    };

private:
    const Tag _tag;

public:
    explicit Statement(Tag tag) noexcept : _tag{tag} {}
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    virtual void accept(StmtVisitor &) const = 0;
    virtual ~Statement() noexcept = default;
    [[nodiscard]] virtual bool is_reference(const Expression *expr) const noexcept { return false; }
};

class OC_AST_API ScopeStmt : public Statement {
private:
    ocarina::vector<Variable> _local_vars;
    ocarina::vector<const Statement *> _statements;
    bool _is_func_body{};

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit ScopeStmt(bool is_func_body = false) noexcept
        : _is_func_body(is_func_body), Statement(Tag::SCOPE) {}
    [[nodiscard]] ocarina::span<const Variable> local_vars() const noexcept { return _local_vars; }
    [[nodiscard]] bool is_func_body() const noexcept { return _is_func_body; }
    [[nodiscard]] ocarina::span<const Statement *const> statements() const noexcept { return _statements; }
    [[nodiscard]] bool is_reference(const Expression *expr) const noexcept override;
    [[nodiscard]] bool empty() const noexcept { return _statements.empty(); }
    [[nodiscard]] auto size() const noexcept { return _statements.size(); }
    void add_stmt(const Statement *stmt) noexcept { _statements.push_back(stmt); }
    void add_var(const Variable &variable) noexcept { _local_vars.push_back(variable); }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API BreakStmt : public Statement {
private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override {
        return Hash64::default_seed;
    }

public:
    BreakStmt() noexcept : Statement{Tag::BREAK} {}
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API ContinueStmt : public Statement {
private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override {
        return Hash64::default_seed;
    }

public:
    ContinueStmt() noexcept : Statement(Tag::CONTINUE) {}
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API ReturnStmt : public Statement {
private:
    const Expression *_expression{nullptr};

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit ReturnStmt(const Expression *expr = nullptr) noexcept
        : Statement(Tag::RETURN), _expression(expr) {}
    [[nodiscard]] const Expression *expression() const noexcept { return _expression; }
    [[nodiscard]] bool is_reference(const Expression *expr) const noexcept override;
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API ExprStmt : public Statement {
private:
    const Expression *_expression{nullptr};

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit ExprStmt(const Expression *expr = nullptr) noexcept
        : Statement(Tag::EXPR), _expression(expr) {}
    [[nodiscard]] bool is_reference(const Expression *expr) const noexcept override;
    const Expression *expression() const { return _expression; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API AssignStmt : public Statement {
private:
    const Expression *_lhs{nullptr};
    const Expression *_rhs{nullptr};

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit AssignStmt(const Expression *lhs, const Expression *rhs)
        : Statement(Tag::ASSIGN), _lhs(lhs), _rhs(rhs) {}
    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] bool is_reference(const Expression *expr) const noexcept override;
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API IfStmt : public Statement {
private:
    const Expression *_condition{nullptr};
    ScopeStmt _true_branch{};
    ScopeStmt _false_branch{};

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit IfStmt(const Expression *condition) : Statement(Tag::IF), _condition(condition) {}
    [[nodiscard]] const Expression *condition() const noexcept { return _condition; }
    [[nodiscard]] const ScopeStmt *true_branch() const noexcept { return &_true_branch; }
    [[nodiscard]] const ScopeStmt *false_branch() const noexcept { return &_false_branch; }
    [[nodiscard]] ScopeStmt *true_branch() noexcept { return &_true_branch; }
    [[nodiscard]] ScopeStmt *false_branch() noexcept { return &_false_branch; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API CommentStmt : public Statement {
private:
    std::string_view _string;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit CommentStmt(std::string_view str)
        : Statement(Tag::COMMENT), _string(str) {}
    [[nodiscard]] std::string_view string() const noexcept { return _string; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API SwitchStmt : public Statement {
private:
    const Expression *_expression;
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit SwitchStmt(const Expression *expr)
        : Statement(Tag::SWITCH), _expression(expr) {}
    [[nodiscard]] auto expression() const noexcept { return _expression; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API SwitchCaseStmt : public Statement {
private:
    const Expression *_expr;
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit SwitchCaseStmt(const Expression *expression)
        : Statement(Tag::SWITCH_CASE), _expr(expression) {}
    [[nodiscard]] auto expression() const noexcept { return _expr; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API SwitchDefaultStmt : public Statement {
private:
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    SwitchDefaultStmt() : Statement(Tag::SWITCH_DEFAULT) {}
    [[nodiscard]] auto body() const noexcept { return &_body; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API ForStmt : public Statement {
private:
    const Expression *_var{};
    const Expression *_condition{};
    const Expression *_step{};
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    ForStmt(const Expression *var, const Expression *cond, const Expression *step)
        : Statement(Tag::FOR), _var(var), _condition(cond), _step(step) {}
    [[nodiscard]] auto var() const noexcept { return _var; }
    [[nodiscard]] auto condition() const noexcept { return _condition; }
    [[nodiscard]] auto step() const noexcept { return _step; }
    [[nodiscard]] auto body() const noexcept { return &_body; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API LoopStmt : public Statement {
private:
    ScopeStmt _body;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    LoopStmt() : Statement(Tag::LOOP) {}
    [[nodiscard]] auto body() const noexcept { return &_body; }
    [[nodiscard]] auto body() noexcept { return &_body; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class OC_AST_API PrintStmt : public Statement {
private:
    ocarina::string_view _fmt;
    ocarina::vector<const Expression *> _args;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    explicit PrintStmt(string_view fmt, const vector<const Expression *> &args)
        : Statement(Tag::PRINT), _fmt(fmt), _args(args) {}
    [[nodiscard]] ocarina::string_view fmt() const noexcept { return _fmt; }
    [[nodiscard]] span<const Expression *const> args() const noexcept { return _args; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

}// namespace ocarina