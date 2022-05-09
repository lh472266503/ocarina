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

namespace katana::ast {

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


public:
    KTN_NODISCARD static FunctionBuilder *current() noexcept;


};

}// namespace katana::ast