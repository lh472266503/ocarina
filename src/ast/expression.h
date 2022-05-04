//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include "core/concepts.h"
#include "usage.h"
#include "op.h"

namespace sycamore::ast {

class UnaryExpr;
class BinaryExpr;
class MemberExpr;
class AccessExpr;
class LiteralExpr;
class RefExpr;
class ConstantExpr;
class CallExpr;
class CastExpr;

struct ExprVisitor {
    virtual void visit(const UnaryExpr *) = 0;
    virtual void visit(const BinaryExpr *) = 0;
    virtual void visit(const MemberExpr *) = 0;
    virtual void visit(const AccessExpr *) = 0;
    virtual void visit(const LiteralExpr *) = 0;
    virtual void visit(const RefExpr *) = 0;
    virtual void visit(const ConstantExpr *) = 0;
    virtual void visit(const CallExpr *) = 0;
    virtual void visit(const CastExpr *) = 0;
};

class Expression : public concepts::Noncopyable {
public:
    enum struct Tag : uint32_t {
        UNARY,
        BINARY,
        MEMBER,
        ACCESS,
        LITERAL,
        REF,
        CONSTANT,
        CALL,
        CAST
    };

private:
    const Type *_type;
    mutable uint64_t _hash{0u};
    mutable bool _hash_computed{false};
    Tag _tag;

protected:
    SCM_NODISCARD virtual uint64_t _compute_hash() const noexcept = 0;
    mutable Usage _usage{Usage::NONE};
    virtual void _mark(Usage usage) const noexcept = 0;

public:
    explicit Expression(Tag tag, const Type *type) noexcept : _type{type}, _tag{tag} {}
    virtual ~Expression() noexcept = default;
    SCM_NODISCARD uint64_t hash() const noexcept;
    SCM_NODISCARD auto type() const noexcept { return _type; }
    SCM_NODISCARD auto usage() const noexcept { return _usage; }
    SCM_NODISCARD auto tag() const noexcept { return _tag; }
    virtual void accept(ExprVisitor &) const = 0;
    void mark(Usage usage) const noexcept;
};

#define SCM_MAKE_EXPRESSION_ACCEPT_VISITOR \
    void accept(ExprVisitor &visitor) const override { visitor.visit(this); }

class UnaryExpr : public Expression {
private:
    const Expression *_operand;
    UnaryOp _op;

public:
    UnaryExpr(const Type *type, UnaryOp op, const Expression *expression)
        : Expression(Tag::UNARY, type), _op(op), _operand(expression) {}
    SCM_NODISCARD auto operand() const noexcept { return _operand; }
    SCM_NODISCARD auto op() const noexcept { return _op; }
    SCM_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class BinaryExpr : public Expression {
private:
    const Expression *lhs;
    const Expression *rhs;
    BinaryOp _op;
};

}// namespace sycamore::ast