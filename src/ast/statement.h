//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include "core/hash.h"
#include "expression.h"

namespace ocarina {

class ScopeStmt;
class BreakStmt;
class ContinueStmt;
class ReturnStmt;
class ExprStmt;
class AssignStmt;
class IfStmt;
class LoopStmt;
class SwitchStmt;
class SwitchCaseStmt;
class SwitchDefaultStmt;
class ForStmt;

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
};

#define OC_MAKE_STATEMENT_ACCEPT_VISITOR \
    void accept(StmtVisitor &visitor) const override { visitor.visit(this); }

class OC_AST_API Statement : public concepts::Noncopyable {
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
        FOR
    };

private:
    mutable uint64_t _hash{0u};
    mutable bool _hash_computed{false};
    Tag _tag;

private:
    [[nodiscard]] virtual uint64_t _compute_hash() const noexcept = 0;

public:
    explicit Statement(Tag tag) noexcept : _tag{tag} {}
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    virtual void accept(StmtVisitor &) const = 0;
    virtual ~Statement() noexcept = default;
    [[nodiscard]] uint64_t hash() const noexcept {
        if (!_hash_computed) {
            OC_USING_SV
            uint64_t h = _compute_hash();
            _hash = hash64(_tag, hash64(h, hash64("__hash_statement"sv)));
            _hash_computed = true;
        }
        return _hash;
    }
};

class OC_AST_API ScopeStmt : public Statement {
private:
    ocarina::vector<const Statement *> _statements;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override {
        auto h = Hash64::default_seed;
        for (auto &&s : _statements) { h = hash64(s->hash(), h); }
        return h;
    }

public:
    ScopeStmt() noexcept : Statement(Tag::SCOPE) {}
    [[nodiscard]] auto statements() const noexcept { return ocarina::span(_statements); }
    void append(const Statement *stmt) noexcept { _statements.push_back(stmt); }
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
    [[nodiscard]] uint64_t _compute_hash() const noexcept override {
        return hash64(_expression == nullptr ? 0ull : _expression->hash());
    }

public:
    explicit ReturnStmt(const Expression *expr = nullptr) noexcept
        : Statement(Tag::RETURN), _expression(expr) {}
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class ExprStmt : public Statement {
private:
    const Expression *_expression{nullptr};

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override {
        return hash64(_expression == nullptr ? 0ull : _expression->hash());
    }

public:
    explicit ExprStmt(const Expression *expr = nullptr) noexcept
        : Statement(Tag::EXPR), _expression(expr) {}
    const Expression *expression() const { return _expression; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

class AssignStmt : public Statement {
private:
    const Expression *_lhs{nullptr};
    const Expression *_rhs{nullptr};

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override {
        auto hl = _lhs->hash();
        auto hr = _rhs->hash();
        return hash64(hl, hr);
    }

public:
    explicit AssignStmt(const Expression *lhs, const Expression *rhs)
        : Statement(Tag::ASSIGN), _lhs(lhs), _rhs(rhs) {}
    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    OC_MAKE_STATEMENT_ACCEPT_VISITOR
};

}// namespace ocarina