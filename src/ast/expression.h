//
// Created by Zero on 21/04/2022.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include "core/concepts.h"
#include "usage.h"
#include "variable.h"
#include "op.h"

namespace ocarina {

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

enum struct CastOp : uint32_t {
    STATIC,
    BITWISE
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
    [[nodiscard]] virtual uint64_t _compute_hash() const noexcept = 0;
    mutable Usage _usage{Usage::NONE};
    virtual void _mark(Usage usage) const noexcept {};

public:
    explicit Expression(Tag tag, const Type *type) noexcept : _type{type}, _tag{tag} {}
    virtual ~Expression() noexcept = default;
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto usage() const noexcept { return _usage; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] bool is_ref() const noexcept { return tag() == Tag::REF; }
    [[nodiscard]] bool is_arithmetic() const noexcept {
        return tag() == Tag::BINARY || tag() == Tag::UNARY;
    }
    virtual void accept(ExprVisitor &) const = 0;
    void mark(Usage usage) const noexcept {
        if (auto a = to_underlying(_usage), u = a | to_underlying(usage); a != u) {
            _usage = static_cast<Usage>(u);
            _mark(usage);
        }
    }
};

using ExprPtr = Expression *;
using ConstExprPtr  = const Expression *;

#define OC_MAKE_EXPRESSION_ACCEPT_VISITOR \
    void accept(ExprVisitor &visitor) const override { visitor.visit(this); }

class UnaryExpr : public Expression {
private:
    ConstExprPtr _operand;
    UnaryOp _op;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override {
        return hash64(_op, _operand->hash());
    }
public:
    UnaryExpr(const Type *type, UnaryOp op, ConstExprPtr expression)
        : Expression(Tag::UNARY, type), _op(op), _operand(expression) {}
    [[nodiscard]] auto operand() const noexcept { return _operand; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class BinaryExpr : public Expression {
private:
    ConstExprPtr _lhs;
    ConstExprPtr _rhs;
    BinaryOp _op;

public:
    BinaryExpr(const Type *type, BinaryOp op, ConstExprPtr lhs, ConstExprPtr rhs) noexcept
        : Expression{Tag::BINARY, type}, _lhs{lhs}, _rhs{rhs}, _op{op} {
        _lhs->mark(Usage::READ);
        _rhs->mark(Usage::READ);
    }
    [[nodiscard]] uint64_t _compute_hash() const noexcept override {
        return 0;
    }
    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class AccessExpr : public Expression {
private:
    ConstExprPtr _range;
    ConstExprPtr _index;

public:
    AccessExpr(const Type *type, ConstExprPtr range, ConstExprPtr index)
        : Expression(Tag::ACCESS, type), _range(range), _index(index) {
        _range->mark(Usage::READ);
        _index->mark(Usage::READ);
    }

    [[nodiscard]] ConstExprPtr range() const noexcept { return _range; }
    [[nodiscard]] ConstExprPtr index() const noexcept { return _index; }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class LiteralExpr : public Expression {
public:
    using value_type = basic_literal_t;

private:
    value_type _value;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    LiteralExpr(const Type *type, value_type value)
        : Expression(Tag::LITERAL, type), _value(value) {}
    [[nodiscard]] decltype(auto) value() const noexcept { return _value; }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class RefExpr : public Expression {
private:
    Variable _variable;

protected:
    void _mark(Usage usage) const noexcept override;
    uint64_t _compute_hash() const noexcept override {
        return hash64(_variable.hash());
    }

public:
    explicit RefExpr(Variable v) noexcept
        : Expression(Tag::REF, v.type()), _variable(v) {}
    [[nodiscard]] auto variable() const noexcept { return _variable; }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};



class CastExpr : public Expression {
private:
    CastOp _cast_op;
    ConstExprPtr _expression;

protected:
    void _mark(Usage) const noexcept override {}
    uint64_t _compute_hash() const noexcept override {
        return hash64(_cast_op, _expression->hash());
    }

public:
    CastExpr(const Type *type, CastOp op, ConstExprPtr expression) noexcept
        : Expression(Tag::CAST, type), _cast_op(op), _expression(expression) {
        _expression->mark(Usage::READ);
    }
    [[nodiscard]] CastOp cast_op() const noexcept {
        return _cast_op;
    }
    [[nodiscard]] ConstExprPtr expression() const noexcept {
        return _expression;
    }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};

}// namespace ocarina