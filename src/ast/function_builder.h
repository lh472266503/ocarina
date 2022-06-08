//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include "core/concepts.h"
#include "function.h"
#include "expression.h"
#include "statement.h"
#include "variable.h"
#include "core/logging.h"

namespace ocarina {

class FunctionBuilder : public ocarina::enable_shared_from_this<FunctionBuilder>,
                        public concepts::Noncopyable {
public:
    using Tag = Function::Tag;
    using Constant = Function::Constant;

private:
    const Type *_ret{nullptr};
    ocarina::vector<ocarina::unique_ptr<Expression>> _all_expressions;
    ocarina::vector<ocarina::unique_ptr<Statement>> _all_statements;
    ocarina::vector<ScopeStmt *> _scope_stack;//
    ocarina::vector<Variable> _builtin_variables;
    ocarina::vector<Variable> _arguments;
    ocarina::vector<Usage> _variable_usages;
    ocarina::vector<ocarina::shared_ptr<const FunctionBuilder>> _used_custom_callables;
    Tag _tag{};

protected:
    [[nodiscard]] static ocarina::vector<FunctionBuilder *> &_function_stack() noexcept;
    [[nodiscard]] const RefExpr *_builtin(Variable::Tag tag) noexcept;
    void _void_expr(const Expression *expr) noexcept;
    [[nodiscard]] uint _next_variable_uid() noexcept;
    [[nodiscard]] const RefExpr *_ref(Variable variable) noexcept;

    template<typename Stmt, typename... Args>
    const Stmt *_create_statement(Args &&...args) noexcept {
        auto statement = ocarina::make_unique<Stmt>(std::forward<Args>(args)...);
        auto ret = statement.get();
        _all_statements.push_back(std::move(statement));
        return ret;
    }

    template<typename Expr, typename... Args>
    const Expr *_create_expression(Args &&...args) noexcept {
        auto expression = ocarina::make_unique<Expr>(std::forward<Args>(args)...);
        auto ret = expression.get();
        _all_expressions.push_back(std::move(expression));
        return ret;
    }

private:
    template<typename Func>
    static auto _define(Function::Tag tag, Func &&func) noexcept {
        auto builder = ocarina::make_shared<FunctionBuilder>(tag);
        push(builder.get());
        func();
        pop(builder.get());
        return ocarina::const_pointer_cast<const FunctionBuilder>(builder);
    }

public:
    template<typename Func>
    static auto define_callable(Func &&func) noexcept {
        return _define(Tag::CALLABLE, std::forward<Func>(func));
    }

    template<typename Func>
    static auto define_kernel(Func &&func) noexcept {
        return _define(Tag::KERNEL, std::forward<Func>(func));
    }

    explicit FunctionBuilder(Tag tag = Tag::CALLABLE) : _tag(tag) {}
    FunctionBuilder(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder(const FunctionBuilder &) noexcept = delete;
    FunctionBuilder &operator=(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder &operator=(const FunctionBuilder &) noexcept = delete;
    ~FunctionBuilder() noexcept {
        OC_DEBUG("function builder was destructed");
    }
    [[nodiscard]] static FunctionBuilder *current() noexcept;
    static void push(FunctionBuilder *builder) noexcept;
    static void pop(FunctionBuilder *builder) noexcept;
    void mark_variable_usage(uint uid, Usage usage) noexcept;
    [[nodiscard]] const CastExpr *cast(const Type *type, CastOp cast_op, const Expression *expression) noexcept;
    [[nodiscard]] const UnaryExpr *unary(const Type *type, UnaryOp op, const Expression *expression) noexcept;
    [[nodiscard]] const BinaryExpr *binary(const Type *type, const Expression *lhs, const Expression *rhs, BinaryOp op) noexcept;
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *reference_argument(const Type *type) noexcept;
    [[nodiscard]] const LiteralExpr *literal(const Type *type, LiteralExpr::value_type value) noexcept;
    [[nodiscard]] const AccessExpr *access(const Type *type, const Expression *range, const Expression *index) noexcept;
    [[nodiscard]] const MemberExpr *member(const Type *type, const Expression *obj, size_t index) noexcept;
    [[nodiscard]] const RefExpr *local(const Type *type) noexcept;
    void break_() noexcept;
    void continue_() noexcept;
    void return_(const Expression *expression = nullptr) noexcept;
    void assign(const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] IfStmt *if_(const Expression *condition) noexcept;
    [[nodiscard]] SwitchStmt *switch_(const Expression *expression) noexcept;
    [[nodiscard]] SwitchCaseStmt *case_(const Statement *statement) noexcept;
    [[nodiscard]] SwitchDefaultStmt *default_() noexcept;
    [[nodiscard]] ForStmt *for_(const RefExpr *var, const Expression *condition, const Expression *update) noexcept;
};

}// namespace ocarina