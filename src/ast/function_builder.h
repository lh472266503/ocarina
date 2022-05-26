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

namespace katana {

class FunctionBuilder : public katana::enable_shared_from_this<FunctionBuilder>,
                        public concepts::Noncopyable {
public:
    using Tag = Function::Tag;
    using Constant = Function::Constant;

private:
    const Type *_ret{nullptr};
    katana::vector<katana::unique_ptr<Expression>> _all_expressions;
    katana::vector<katana::unique_ptr<Statement>> _all_statements;
    katana::vector<ScopeStmt *> _scope_stack;//
    katana::vector<Variable> _builtin_variables;
    katana::vector<Variable> _arguments;
    katana::vector<Usage> _variable_usages;
    katana::vector<katana::shared_ptr<const FunctionBuilder>> _used_custom_callables;
    Tag _tag{};

protected:
    [[nodiscard]] static katana::vector<FunctionBuilder *> &_function_stack() noexcept;
    void _add_statement(const Statement *statement) noexcept;
    [[nodiscard]] const RefExpr *_builtin(Variable::Tag tag) noexcept;
    [[nodiscard]] const RefExpr *_ref(Variable::Tag tag) noexcept;
    void _void_expr(const Expression *expr) noexcept;
    [[nodiscard]] uint _next_variable_uid() noexcept;
    template<typename Expr, typename... Args>
    [[nodiscard]] const Expr *_create_expression(Args &&...args) noexcept {
        auto expression = katana::make_unique<Expr>(std::forward<Args>(args)...);
        auto ret = expression.get();
        _all_expressions.push_back(std::move(expression));
        return ret;
    }

private:
    template<typename Func>
    static auto _define(Function::Tag tag, Func &&func) noexcept {
        auto builder = katana::make_shared<FunctionBuilder>(tag);
        func();
        return katana::const_pointer_cast<const FunctionBuilder>(builder);
    }

public:
    explicit FunctionBuilder(Tag tag = Tag::CALLABLE) : _tag(tag) {}
    FunctionBuilder(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder(const FunctionBuilder &) noexcept = delete;
    FunctionBuilder &operator=(FunctionBuilder &&) noexcept = delete;
    FunctionBuilder &operator=(const FunctionBuilder &) noexcept = delete;
    ~FunctionBuilder() noexcept {
        KTN_DEBUG("function builder was destructed");
    }

    [[nodiscard]] static FunctionBuilder *current() noexcept;

    template<typename Func>
    static auto define_callable(Func &&func) noexcept {
        return _define(Tag::CALLABLE, std::forward<Func>(func));
    }

    template<typename Func>
    static auto define_kernel(Func &&func) noexcept {
        return _define(Tag::KERNEL, std::forward<Func>(func));
    }

    void mark_variable_usage(uint uid, Usage usage) noexcept;

    [[nodiscard]] const CastExpr *cast(const Type *type, CastOp cast_op, const Expression *expression) noexcept;

    [[nodiscard]] const UnaryExpr *unary(const Type *type, UnaryOp op, const Expression *expression) noexcept;

    [[nodiscard]] const BinaryExpr *binary(const Type *type, const Expression *lhs, const Expression *rhs, BinaryOp op) noexcept;

    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept;

    [[nodiscard]] const RefExpr *reference_argument(const Type *type) noexcept;

    [[nodiscard]] const LiteralExpr *literal(const Type *type, LiteralExpr *literal_expr) noexcept;

    [[nodiscard]] const AccessExpr *access(const Type *type, const Expression *range, const Expression *index) noexcept;

    [[nodiscard]] const MemberExpr *member(const Type *type, const Expression *obj, size_t index) noexcept;

    [[nodiscard]] const RefExpr *local(const Type *) noexcept;

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

}// namespace katana