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

namespace katana {

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
    virtual void _mark(Usage usage) const noexcept = 0;

public:
    explicit Expression(Tag tag, const Type *type) noexcept : _type{type}, _tag{tag} {}
    virtual ~Expression() noexcept = default;
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto usage() const noexcept { return _usage; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    virtual void accept(ExprVisitor &) const = 0;
    void mark(Usage usage) const noexcept;
};

#define KTN_MAKE_EXPRESSION_ACCEPT_VISITOR \
    void accept(ExprVisitor &visitor) const override { visitor.visit(this); }

class UnaryExpr : public Expression {
private:
    const Expression *_operand;
    UnaryOp _op;

public:
    UnaryExpr(const Type *type, UnaryOp op, const Expression *expression)
        : Expression(Tag::UNARY, type), _op(op), _operand(expression) {}
    [[nodiscard]] auto operand() const noexcept { return _operand; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    KTN_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class BinaryExpr : public Expression {
private:
    const Expression *_lhs;
    const Expression *_rhs;
    BinaryOp _op;

public:
    BinaryExpr(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept
        : Expression{Tag::BINARY, type}, _lhs{lhs}, _rhs{rhs}, _op{op} {
        _lhs->mark(Usage::READ);
        _rhs->mark(Usage::READ);
    }
    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    KTN_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class AccessExpr : public Expression {
private:
    const Expression *_range;
    const Expression *_index;

public:
    AccessExpr(const Type *type, const Expression *range, const Expression *index)
        : Expression(Tag::ACCESS, type), _range(range), _index(index) {
        _range->mark(Usage::READ);
        _index->mark(Usage::READ);
    }

    [[nodiscard]] const Expression *range() const noexcept { return _range; }
    [[nodiscard]] const Expression *index() const noexcept { return _index; }
    KTN_MAKE_EXPRESSION_ACCEPT_VISITOR
};

namespace detail {
template<typename T>
struct literal_value {
    static_assert(always_false_v<T>);
};

template<typename... T>
struct literal_value<std::tuple<T...>> {
    using type = katana::variant<T...>;
};
}// namespace detail

template<typename T>
using literal_value_t = typename detail::literal_value<T>::type;

class LiteralExpr : public Expression {
public:
    using value_type = literal_value_t<basic_types>;

private:
    value_type _value;

public:
    LiteralExpr(const Type *type, value_type value)
        : Expression(Tag::LITERAL, type), _value(value) {}
    [[nodiscard]] decltype(auto) value() const noexcept { return _value; }
    KTN_MAKE_EXPRESSION_ACCEPT_VISITOR
};

class RefExpr : public Expression {
private:
    Variable _variable;

public:
};

}// namespace katana