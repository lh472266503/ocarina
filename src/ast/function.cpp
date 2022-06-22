//
// Created by Zero on 30/04/2022.
//

#include "function.h"

#ifdef NDEBUG

#include "function_impl.h"

#endif

namespace ocarina {

ocarina::vector<Function *> &Function::_function_stack() noexcept {
    static ocarina::vector<Function *> ret;
    return ret;
}

void Function::return_(ConstExprPtr expression) noexcept {
    if (expression) {
        _ret = expression->type();
    }
    create_statement<ReturnStmt>(expression);
}

Function::Function(Function::Tag tag) noexcept
    : _tag(tag) {
    push_scope();
}

const ScopeStmt *Function::body() const noexcept {
    return _scope_stack.front();
}

ScopeStmt *Function::body() noexcept {
    return _scope_stack.front();
}

ConstExprPtr Function::argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::LOCAL, next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}

ConstExprPtr Function::reference_argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::REFERENCE, next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}

ConstExprPtr Function::local(const Type *type) noexcept {
    auto ret = create_expression<RefExpr>(Variable(type, Variable::Tag::LOCAL,
                                                          next_variable_uid()));
    body()->add_var(ret->variable());
    return ret;
}

ConstExprPtr Function::literal(const Type *type, LiteralExpr::value_type value) noexcept {
    return create_expression<LiteralExpr>(type, value);
}

ConstExprPtr Function::binary(const Type *type, BinaryOp op, ConstExprPtr lhs, ConstExprPtr rhs) noexcept {
    return create_expression<BinaryExpr>(type, op, lhs, rhs);
}

ConstExprPtr Function::unary(const Type *type, UnaryOp op, ConstExprPtr expression) noexcept {
    return create_expression<UnaryExpr>(type, op, expression);
}

IfStmt *Function::if_(ConstExprPtr expr) noexcept {
    return create_statement<IfStmt>(expr);
}

ocarina::span<const Variable> Function::arguments() const noexcept {
    return _arguments;
}

const Type *Function::return_type() const noexcept {
    return _ret;
}

void Function::assign(ConstExprPtr lhs, ConstExprPtr rhs) noexcept {
    create_statement<AssignStmt>(lhs, rhs);
}
uint64_t Function::hash() const noexcept {
    if (!_hash_computed) {
        _hash = _compute_hash();
        _hash_computed = true;
    }
    return _hash;
}
void Function::postprocess() noexcept {
    
}

}// namespace ocarina