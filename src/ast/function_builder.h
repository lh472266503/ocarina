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

namespace sycamore::ast {

class FunctionBuilder : public sycamore::enable_shared_from_this<FunctionBuilder>,
                        public concepts::Noncopyable {
private:
    const Type *_ret{nullptr};
    sycamore::vector<sycamore::unique_ptr<Expression>> _all_expressions;
    sycamore::vector<sycamore::unique_ptr<Statement>> _all_statements;
    sycamore::vector<ScopeStmt *> _scope_stack;//
    sycamore::vector<Variable> _builtin_variables;
    sycamore::vector<Variable> _arguments;

public:
    using Tag = Function::Tag;
    using Constant = Function::Constant;

protected:
    SCM_NODISCARD static sycamore::vector<FunctionBuilder *> &_function_stack() noexcept;

public:
    SCM_NODISCARD static FunctionBuilder *current() noexcept;
};

}// namespace sycamore::ast