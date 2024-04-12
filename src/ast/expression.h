//
// Created by Zero on 21/04/2022.
//

#pragma once

#include <utility>

#include "ast_node.h"

namespace ocarina {

class UnaryExpr;
class BinaryExpr;
class MemberExpr;
class SubscriptExpr;
class LiteralExpr;
class RefExpr;
class CallExpr;
class CastExpr;
class ConditionalExpr;

class Function;

struct OC_AST_API ExprVisitor {
    virtual void visit(const UnaryExpr *) = 0;
    virtual void visit(const BinaryExpr *) = 0;
    virtual void visit(const MemberExpr *) = 0;
    virtual void visit(const SubscriptExpr *) = 0;
    virtual void visit(const ConditionalExpr *) = 0;
    virtual void visit(const LiteralExpr *) = 0;
    virtual void visit(const RefExpr *) = 0;
    virtual void visit(const CallExpr *) = 0;
    virtual void visit(const CastExpr *) = 0;
};

class OC_AST_API Expression : public ASTNode, public concepts::Noncopyable, public Hashable {
public:
    enum struct Tag : uint32_t {
        UNARY,
        BINARY,
        MEMBER,
        SUBSCRIPT,
        LITERAL,
        REF,
        CONSTANT,
        CALL,
        CAST,
        CONDITIONAL,
        SAMPLE
    };

private:
    const Type *_type;
    Tag _tag;

protected:
    virtual void _mark(Usage usage) const noexcept {};

public:
    explicit Expression(Tag tag, const Type *type) noexcept : _type{type}, _tag{tag} {}
    virtual ~Expression() noexcept = default;
    [[nodiscard]] auto type() const noexcept { return _type; }
    [[nodiscard]] auto tag() const noexcept { return _tag; }
    [[nodiscard]] bool is_ref() const noexcept { return tag() == Tag::REF; }
    [[nodiscard]] bool is_member() const noexcept { return tag() == Tag::MEMBER; }
    [[nodiscard]] bool is_arithmetic() const noexcept {
        return tag() == Tag::BINARY || tag() == Tag::UNARY;
    }
    virtual void accept(ExprVisitor &) const = 0;
    [[nodiscard]] virtual Usage usage() const noexcept { return Usage::NONE; }
    void mark(Usage usage) const noexcept {
        _mark(usage);
    }
};

#define OC_MAKE_EXPRESSION_COMMON   \
    friend class FunctionCorrector; \
    void accept(ExprVisitor &visitor) const override { visitor.visit(this); }

class OC_AST_API UnaryExpr : public Expression {
private:
    const Expression *_operand;
    UnaryOp _op;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    UnaryExpr(const Type *type, UnaryOp op, const Expression *expression)
        : Expression(Tag::UNARY, type), _op(op), _operand(expression) {}
    OC_MAKE_CHECK_CONTEXT(Expression, _operand)
    [[nodiscard]] auto operand() const noexcept { return _operand; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    [[nodiscard]] Usage usage() const noexcept override { return _operand->usage(); }
    OC_MAKE_EXPRESSION_COMMON
};

class OC_AST_API BinaryExpr : public Expression {
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
    OC_MAKE_CHECK_CONTEXT(Expression, _lhs, _rhs)
    [[nodiscard]] auto lhs() const noexcept { return _lhs; }
    [[nodiscard]] auto rhs() const noexcept { return _rhs; }
    [[nodiscard]] auto op() const noexcept { return _op; }
    OC_MAKE_EXPRESSION_COMMON
};

class OC_AST_API ConditionalExpr : public Expression {
private:
    const Expression *_pred{};
    const Expression *_true{};
    const Expression *_false{};

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    ConditionalExpr(const Type *type, const Expression *pred,
                    const Expression *t, const Expression *f)
        : Expression(Tag::CONDITIONAL, type),
          _pred(pred), _true(t), _false(f) {
        _pred->mark(Usage::READ);
        _true->mark(Usage::READ);
        _false->mark(Usage::READ);
    }
    OC_MAKE_CHECK_CONTEXT(Expression, _pred, _true, _false)
    [[nodiscard]] const Expression *pred() const noexcept { return _pred; }
    [[nodiscard]] const Expression *true_() const noexcept { return _true; }
    [[nodiscard]] const Expression *false_() const noexcept { return _false; }
    OC_MAKE_EXPRESSION_COMMON
};

class OC_AST_API SubscriptExpr : public Expression {
private:
    using IndexVector = vector<const Expression *>;

private:
    const Expression *_range{};
    IndexVector _indexes;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;
    void _mark(ocarina::Usage usage) const noexcept override {
        _range->mark(usage);
    }

public:
    SubscriptExpr(const Type *type, const Expression *range, const Expression *index)
        : Expression(Tag::SUBSCRIPT, type), _range(range) {
        _indexes.push_back(index);
    }
    OC_MAKE_CHECK_CONTEXT(Expression, _range, _indexes)
    SubscriptExpr(const Type *type, const Expression *range, IndexVector indexes)
        : Expression(Tag::SUBSCRIPT, type), _range(range), _indexes(ocarina::move(indexes)) {
    }

    template<typename Func>
    void for_each_index(Func &&func) const noexcept {
        for (const Expression *index : _indexes) {
            func(index);
        }
    }
    [[nodiscard]] Usage usage() const noexcept override { return _range->usage(); }
    [[nodiscard]] const Expression *range() const noexcept { return _range; }
    [[nodiscard]] const Expression *index(int i) const noexcept { return _indexes.at(i); }
    OC_MAKE_EXPRESSION_COMMON
};

class OC_AST_API LiteralExpr : public Expression {
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
    bool check_context(const Function *ctx) const noexcept override { return true; }
    OC_MAKE_EXPRESSION_COMMON
};

class OC_AST_API RefExpr : public Expression {
private:
    Variable _variable;

private:
    void _mark(Usage usage) const noexcept override;
    uint64_t _compute_hash() const noexcept override;

public:
    explicit RefExpr(const Variable &v) noexcept
        : Expression(Tag::REF, v.type()), _variable(v) {}
    [[nodiscard]] auto variable() const noexcept { return _variable; }
    [[nodiscard]] Usage usage() const noexcept override;
    OC_MAKE_EXPRESSION_COMMON
};

class OC_AST_API CastExpr : public Expression {
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
    OC_MAKE_CHECK_CONTEXT(Expression, _expression)
    [[nodiscard]] CastOp cast_op() const noexcept { return _cast_op; }
    [[nodiscard]] const Expression *expression() const noexcept { return _expression; }
    [[nodiscard]] Usage usage() const noexcept override { return _expression->usage(); }
    OC_MAKE_EXPRESSION_COMMON
};

class OC_AST_API MemberExpr : public Expression {
private:
    const Expression *_parent{nullptr};
    uint16_t _member_index{0};
    uint16_t _swizzle_size{0};
    /// used for store usage
    Variable _variable;

private:
    void _mark(Usage usage) const noexcept override;
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    MemberExpr(const Type *type, const Expression *parent, uint16_t index, uint16_t swizzle_size, Variable variable = {});
    OC_MAKE_CHECK_CONTEXT(Expression, _parent)
    [[nodiscard]] auto member_index() const noexcept { return _member_index; }
    [[nodiscard]] bool is_swizzle() const noexcept { return _swizzle_size != 0; }
    [[nodiscard]] Variable variable() const noexcept { return _variable; }
    [[nodiscard]] int swizzle_size() const noexcept { return _swizzle_size; }
    [[nodiscard]] int swizzle_index(int idx) const noexcept;
    [[nodiscard]] const Expression *parent() const noexcept { return _parent; }
    [[nodiscard]] Usage usage() const noexcept override { return parent()->usage(); }
    OC_MAKE_EXPRESSION_COMMON
};

class OC_AST_API CallExpr : public Expression {
public:
    using Template = std::variant<const Type *, uint>;

private:
    ocarina::vector<const Expression *> _arguments;
    const Function *_function{};
    CallOp _call_op{CallOp::CUSTOM};
    string_view _function_name{};
    ocarina::vector<Template> _template_args;

private:
    void _mark(Usage) const noexcept override {}
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;

public:
    CallExpr(const Type *type, const Function *func,
             ocarina::vector<const Expression *> &&args);
    CallExpr(const Type *type, CallOp op,
             ocarina::vector<const Expression *> &&args,
             ocarina::vector<Template> &&t_args = {})
        : Expression(Tag::CALL, type), _call_op(op),
          _arguments(std::move(args)), _template_args(ocarina::move(t_args)) {}
    CallExpr(const Type *type, string_view func_name,
             ocarina::vector<const Expression *> &&args)
        : Expression(Tag::CALL, type), _function_name(ocarina::move(func_name)),
          _arguments(ocarina::move(args)) {}
    OC_MAKE_CHECK_CONTEXT(Expression, _arguments)
    [[nodiscard]] ocarina::span<const Expression *const> arguments() const noexcept { return _arguments; }
    [[nodiscard]] ocarina::span<const Template> template_args() const noexcept { return _template_args; }
    OC_MAKE_MEMBER_GETTER(function_name, &)
    void append_argument(const Expression *expression) noexcept;
    [[nodiscard]] vector<const Function *> call_chain() const noexcept;
    [[nodiscard]] auto call_op() const noexcept { return _call_op; }
    [[nodiscard]] auto function() const noexcept { return _function; }
    OC_MAKE_EXPRESSION_COMMON
};

#undef OC_MAKE_EXPRESSION_COMMON

}// namespace ocarina