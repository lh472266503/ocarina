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

namespace katana {

class FunctionBuilder : public katana::enable_shared_from_this<FunctionBuilder>,
                        public concepts::Noncopyable {
private:
    const Type *_ret{nullptr};
    katana::vector<katana::unique_ptr<Expression>> _all_expressions;
    katana::vector<katana::unique_ptr<Statement>> _all_statements;
    katana::vector<ScopeStmt *> _scope_stack;//
    katana::vector<Variable> _builtin_variables;
    katana::vector<Variable> _arguments;
    katana::vector<katana::shared_ptr<const FunctionBuilder>> _used_custom_callables;

public:
    using Tag = Function::Tag;
    using Constant = Function::Constant;

protected:
    KTN_NODISCARD static katana::vector<FunctionBuilder *> &_function_stack() noexcept;
    void _add_statement(const Statement *statement) noexcept;
    KTN_NODISCARD const RefExpr *_builtin(Variable::Tag tag) noexcept;
    KTN_NODISCARD const RefExpr *_ref(Variable::Tag tag) noexcept;
    void _void_expr(const Expression *expr) noexcept;

private:
    template<typename Func>
    static auto _define(Function::Tag tag, Func &&func) noexcept {
        auto builder = katana::make_shared<FunctionBuilder>(tag);

        return katana::const_pointer_cast<const FunctionBuilder>(builder);
    }

public:
    KTN_NODISCARD static FunctionBuilder *current() noexcept;

    template<typename Func>
    static auto define_callable(Func &&func) noexcept {
        return _define(Tag::CALLABLE, std::forward<Func>(func));
    }

    template<typename Func>
    static auto define_kernel(Func &&func) noexcept {
        return _define(Tag::KERNEL, std::forward<Func>(func));
    }

    KTN_NODISCARD const CastExpr *cast(const Type *type, CastOp cast_op, const Expression *expression) noexcept;

    KTN_NODISCARD const UnaryExpr *unary(const Type *type, UnaryOp op, const Expression *expression) noexcept;

    KTN_NODISCARD const BinaryExpr *binary(const Type *type, const Expression *lhs, const Expression *rhs, BinaryOp op) noexcept;

    KTN_NODISCARD const RefExpr *argument(const Type *type) noexcept;

    KTN_NODISCARD const RefExpr *reference(const Type *type) noexcept;

    KTN_NODISCARD const LiteralExpr *literal(const Type *type, LiteralExpr *literal_expr) noexcept;

    KTN_NODISCARD const AccessExpr *access(const Type *type, const Expression *range, const Expression *index) noexcept;

    void break_() noexcept;

    void continue_() noexcept;

    void return_(const Expression *expression = nullptr) noexcept;

    void assign(const Expression *lhs, const Expression *rhs) noexcept;

    KTN_NODISCARD IfStmt *if_(const Expression *condition) noexcept;

    KTN_NODISCARD SwitchStmt *switch_(const Expression *expression) noexcept;

    KTN_NODISCARD SwitchCaseStmt *case_(const Statement *statement) noexcept;

    KTN_NODISCARD SwitchDefaultStmt *default_() noexcept;

    KTN_NODISCARD ForStmt *for_(const RefExpr *var, const Expression *condition, const Expression *update) noexcept;
};

}// namespace katana