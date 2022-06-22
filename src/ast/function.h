//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/basic_types.h"
#include "core/stl.h"
#include "core/header.h"
#include "type.h"
#include "variable.h"
#include "expression.h"
#include "statement.h"

namespace ocarina {

class Statement;
class ScopeStmt;
class IfStmt;

class OC_AST_API Function: public concepts::Noncopyable {
public:
    enum struct Tag : uint {
        KERNEL,
        CALLABLE,
    };

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

private:
    static ocarina::vector<Function *> &_function_stack() noexcept;
    static void _push(Function *f) {
        _function_stack().push_back(f);
    }
    static void _pop(Function *f) {
        OC_ASSERT(f == _function_stack().back());
        _function_stack().pop_back();
    }
    template<typename Func>
    static auto _define(Function::Tag tag, Func &&func) noexcept {
        auto ret = Function(tag);
        _push(&ret);
        func();
        _pop(&ret);
        ret.postprocess();
        return ret;
    }
    template<typename Expr, typename... Args>
    [[nodiscard]] auto create_expression(Args &&...args) {
        auto expr = ocarina::make_unique<Expr>(std::forward<Args>(args)...);
        auto ret = expr.get();
        _all_expressions.push_back(std::move(expr));
        return ret;
    }
    [[nodiscard]] ConstExprPtr _ref(Variable variable) noexcept {
        return create_expression<RefExpr>(variable);
    }
    [[nodiscard]] uint64_t _compute_hash() const noexcept {
        return 6545689;
    }
public:
    [[nodiscard]] static Function *current() noexcept {
        return _function_stack().back();
    }
    template<typename Func>
    static auto define_callable(Func &&func) noexcept {
        return _define(Tag::CALLABLE, std::forward<Func>(func));
    }
    [[nodiscard]] const ScopeStmt *current_scope() const noexcept {
        return _scope_stack.back();
    }
    [[nodiscard]] ScopeStmt *current_scope() noexcept {
        return _scope_stack.back();
    }

    template<typename Stmt, typename... Args>
    auto create_statement(Args &&...args) {
        auto stmt = ocarina::make_unique<Stmt>(std::forward<Args>(args)...);
        auto ret = stmt.get();
        _all_statements.push_back(std::move(stmt));
        current_scope()->add_stmt(ret);
        return ret;
    }
    void push_scope() {
        auto scope = ocarina::make_unique<ScopeStmt>();
        _scope_stack.push_back(scope.get());
        _all_statements.push_back(std::move(scope));
    }
    [[nodiscard]] uint next_variable_uid() noexcept {
        auto ret = _variable_usages.size();
        _variable_usages.push_back(Usage::NONE);
        return ret;
    }
    void pop_scope() {
        _scope_stack.pop_back();
    }
    void mark_variable_usage(uint uid, Usage usage) noexcept {
        _variable_usages[uid] = usage;
    }
    template<typename Func>
    static auto define_kernel(Func &&func) noexcept {
        return _define(Tag::KERNEL, std::forward<Func>(func));
    }
    Function() noexcept = default;
    explicit Function(Tag tag) noexcept;
    void assign(ConstExprPtr lhs, ConstExprPtr rhs) noexcept;
    void return_(ConstExprPtr expression) noexcept;
    [[nodiscard]] ConstExprPtr argument(const Type *type) noexcept;
    [[nodiscard]] ConstExprPtr reference_argument(const Type *type) noexcept;
    [[nodiscard]] ConstExprPtr local(const Type *type) noexcept;
    [[nodiscard]] ConstExprPtr literal(const Type *type, LiteralExpr::value_type value) noexcept;
    [[nodiscard]] ConstExprPtr binary(const Type *type, BinaryOp op, ConstExprPtr lhs, ConstExprPtr rhs) noexcept;
    [[nodiscard]] ConstExprPtr unary(const Type *type, UnaryOp op, ConstExprPtr expression) noexcept;
    [[nodiscard]] IfStmt *if_(ConstExprPtr expr) noexcept;
    [[nodiscard]] const ScopeStmt *body() const noexcept;
    [[nodiscard]] ScopeStmt *body() noexcept;
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] ocarina::span<const Variable> arguments() const noexcept;
    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] bool is_callable() const noexcept { return _tag == Tag::CALLABLE; }
    [[nodiscard]] bool is_kernel() const noexcept { return _tag == Tag::KERNEL; }
    [[nodiscard]] const Type *return_type() const noexcept;
    void postprocess() noexcept;
};

}// namespace ocarina

