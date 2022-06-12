//
// Created by Zero on 30/04/2022.
//

#include "function.h"
#include "function_builder.h"

namespace ocarina {

class Function::Impl {
private:
    const Type *_ret{nullptr};
    ocarina::vector<ocarina::unique_ptr<Expression>> _expressions;
    ocarina::vector<ocarina::unique_ptr<Statement>> _statements;
    ocarina::vector<Variable> _arguments;
    ocarina::vector<Usage> _variable_usages;
    Tag _tag{Tag::CALLABLE};

private:
    template<typename Expr, typename... Args>
    [[nodiscard]] const Expr *_create_expression(Args &&...args) {
        auto expr = ocarina::make_unique<Expr>(std::forward<Args>(args)...);
        auto ret = expr.get();
        _expressions.push_back(std::move(expr));
        return ret;
    }
    template<typename Stmt, typename... Args>
    const Stmt *_create_statement(Args &&...args) {
        auto stmt = ocarina::make_unique<Stmt>(std::forward<Args>(args)...);
        auto ret = stmt.get();
        _statements.push_back(std::move(stmt));
        return ret;
    }
    [[nodiscard]] const RefExpr *_ref(Variable variable) noexcept {
        return _create_expression<RefExpr>(variable);
    }
    [[nodiscard]] uint _next_variable_uid() noexcept {
        auto ret = _variable_usages.size();
        _variable_usages.push_back(Usage::NONE);
        return ret;
    }

public:
    explicit Impl(Tag tag = Tag::CALLABLE) : _tag(tag) {}
    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] bool is_callable() const noexcept { return _tag == Tag::CALLABLE; }
    [[nodiscard]] bool is_kernel() const noexcept { return _tag == Tag::KERNEL; }
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept {
        Variable variable(type, Variable::Tag::LOCAL, _next_variable_uid());
        return _ref(variable);
    }
    [[nodiscard]] const RefExpr *reference_argument(const Type *type) noexcept {
        Variable variable(type, Variable::Tag::REFERENCE, _next_variable_uid());
        return _ref(variable);
    }
    [[nodiscard]] const BinaryExpr *binary(const Type *type, BinaryOp op,
                                           const Expression *lhs,
                                           const Expression *rhs) noexcept {
        return _create_expression<BinaryExpr>(type, op, lhs, rhs);
    }
    [[nodiscard]] const LiteralExpr *literal(const Type *type, LiteralExpr::value_type value) noexcept {
        return _create_expression<LiteralExpr>(type, value);
    }
    [[nodiscard]] const RefExpr *local(const Type *type) noexcept {
        return _create_expression<RefExpr>(Variable(type, Variable::Tag::LOCAL, _next_variable_uid()));
    }
    void return_(const Expression *expression) noexcept {
        if (expression) {
            _ret = expression->type();
        }
        _create_statement<ReturnStmt>(expression);
    }
    void assign(const Expression *lhs, const Expression *rhs) noexcept {
        _create_statement<AssignStmt>(lhs, rhs);
    }
};

ocarina::span<const Variable> Function::arguments() const noexcept {
    return _builder->arguments();
}

const Type *Function::return_type() const noexcept {
    return _builder->return_type();
}

Function::Tag Function::tag() const noexcept {
    return _builder->tag();
}

bool Function::is_callable() const noexcept {
    return _builder->is_callable();
}

bool Function::is_kernel() const noexcept {
    return _builder->is_kernel();
}
}// namespace ocarina