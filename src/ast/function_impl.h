//
// Created by Zero on 14/06/2022.
//

#pragma once

#include "ast/statement.h"

namespace ocarina {

class Function::Impl : public concepts::Noncopyable {
private:
    const Type *_ret{nullptr};
    ocarina::vector<ocarina::unique_ptr<Expression>> _all_expressions;
    ocarina::vector<ocarina::unique_ptr<Statement>> _all_statements;
    ocarina::vector<Variable> _arguments;
    ocarina::vector<Usage> _variable_usages;
    ocarina::vector<ScopeStmt *> _scope_stack;
    mutable uint64_t _hash{0};
    mutable bool _hash_computed{false};
    Tag _tag{Tag::CALLABLE};
    friend class Function;

private:
    [[nodiscard]] const RefExpr *_ref(Variable variable) noexcept {
        return create_expression<RefExpr>(variable);
    }
    [[nodiscard]] uint64_t _compute_hash() const noexcept {
        return 6545689;
    }

public:
    explicit Impl(Tag tag = Tag::CALLABLE) : _tag(tag) {
        push_scope();
    }
    [[nodiscard]] uint next_variable_uid() noexcept {
        auto ret = _variable_usages.size();
        _variable_usages.push_back(Usage::NONE);
        return ret;
    }
    template<typename Expr, typename... Args>
    [[nodiscard]] const Expr *create_expression(Args &&...args) {
        auto expr = ocarina::make_unique<Expr>(std::forward<Args>(args)...);
        auto ret = expr.get();
        _all_expressions.push_back(std::move(expr));
        return ret;
    }
    template<typename Stmt, typename... Args>
    const Stmt *_create_statement(Args &&...args) {
        auto stmt = ocarina::make_unique<Stmt>(std::forward<Args>(args)...);
        auto ret = stmt.get();
        _all_statements.push_back(std::move(stmt));
        _scope_stack.back()->append(ret);
        return ret;
    }
    void push_scope() {
        auto scope = ocarina::make_unique<ScopeStmt>();
        _scope_stack.push_back(scope.get());
        _all_statements.push_back(std::move(scope));
    }
    void pop_scope() {
        _scope_stack.pop_back();
    }
    [[nodiscard]] const ScopeStmt *current_scope() const noexcept {
        return _scope_stack.back();
    }
    void mark_variable_usage(uint uid, Usage usage) noexcept {
        _variable_usages[uid] = usage;
    }
    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] bool is_callable() const noexcept { return _tag == Tag::CALLABLE; }
    [[nodiscard]] bool is_kernel() const noexcept { return _tag == Tag::KERNEL; }
    [[nodiscard]] const Type *return_type() const noexcept { return _ret; }
    [[nodiscard]] ocarina::span<const Variable> arguments() const noexcept {
        return _arguments;
    }
    [[nodiscard]] uint64_t hash() const noexcept {
        if (!_hash_computed) {
            _hash = _compute_hash();
            _hash_computed = true;
        }
        return _hash;
    }
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept {
        Variable variable(type, Variable::Tag::LOCAL, next_variable_uid());
        _arguments.push_back(variable);
        return _ref(variable);
    }
    [[nodiscard]] const RefExpr *reference_argument(const Type *type) noexcept {
        Variable variable(type, Variable::Tag::REFERENCE, next_variable_uid());
        _arguments.push_back(variable);
        return _ref(variable);
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
}// namespace ocarina