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

#define OC_MAKE_EXPRESSION_ACCEPT_VISITOR \
    void accept(ExprVisitor &visitor) const override { visitor.visit(this); }

class UnaryExpr : public Expression {
private:
    const Expression *_operand;
    UnaryOp _op;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    UnaryExpr(const Type *type, UnaryOp op, const Expression *expression)
        : Expression(Tag::UNARY, type), _op(op), _operand(expression) {}
    [[nodiscard]] auto operand() const noexcept { return _operand; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class BinaryExpr : public Expression {
private:
    const Expression *_lhs;
    const Expression *_rhs;
    BinaryOp _op;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    BinaryExpr(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept
        : Expression{Tag::BINARY, type}, _lhs{lhs}, _rhs{rhs}, _op{op} {
        _lhs->mark(Usage::READ);
        _rhs->mark(Usage::READ);
    }
    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class AccessExpr : public Expression {
private:
    const Expression *_range;
    const Expression *_index;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    AccessExpr(const Type *type, const Expression *range, const Expression *index)
        : Expression(Tag::ACCESS, type), _range(range), _index(index) {
        _range->mark(Usage::READ);
        _index->mark(Usage::READ);
    }

    [[nodiscard]] const Expression *range() const noexcept { return _range; }
    [[nodiscard]] const Expression *index() const noexcept { return _index; }
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

private:
    void _mark(Usage usage) const noexcept override;
    uint64_t _compute_hash() const noexcept override {
        return _variable.hash();
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
    const Expression *_expression;

private:
    void _mark(Usage) const noexcept override {}
    uint64_t _compute_hash() const noexcept override {
        return hash64(_cast_op, _expression->hash());
    }

public:
    CastExpr(const Type *type, CastOp op, const Expression *expression) noexcept
        : Expression(Tag::CAST, type), _cast_op(op), _expression(expression) {
        _expression->mark(Usage::READ);
    }
    [[nodiscard]] CastOp cast_op() const noexcept {
        return _cast_op;
    }
    [[nodiscard]] const Expression *expression() const noexcept {
        return _expression;
    }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class MemberExpr : public Expression {
private:
    const Expression *_parent{nullptr};
    uint16_t _member_index{0};
    uint16_t _swizzle_size{0};
    ocarina::string_view _field_name;

private:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    MemberExpr(const Type *type, const Expression *parent, std::string_view name)
        : Expression(Tag::MEMBER, type), _parent(parent), _field_name(name) {}
    MemberExpr(const Type *type, const Expression *parent, uint16_t mask, uint16_t swizzle_size)
        : Expression(Tag::MEMBER, type), _parent(parent), _member_index(mask), _swizzle_size(swizzle_size) {}
    [[nodiscard]] auto member_index() const noexcept { return _member_index; }
    [[nodiscard]] bool is_swizzle() const noexcept { return _swizzle_size != 0; }
    [[nodiscard]] int swizzle_size() const noexcept { return _swizzle_size; }
    [[nodiscard]] int swizzle_index(int idx) const noexcept;
    [[nodiscard]] const Expression *parent() const noexcept { return _parent; }
    [[nodiscard]] ocarina::string_view field_name() const noexcept { return _field_name; }
    OC_MAKE_EXPRESSION_ACCEPT_VISITOR
};

}// namespace ocarina