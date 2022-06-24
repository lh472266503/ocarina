//
// Created by Zero on 30/04/2022.
//

#include "function.h"

namespace ocarina {

ocarina::vector<Function *> &Function::_function_stack() noexcept {
    static ocarina::vector<Function *> ret;
    return ret;
}

void Function::return_(const Expression *expression) noexcept {
    if (expression) {
        _ret = expression->type();
    }
    _create_statement<ReturnStmt>(expression);
}

Function::Function(Function::Tag tag) noexcept
    : _tag(tag) {}

const ScopeStmt *Function::body() const noexcept {
    return &_body;
}

ScopeStmt *Function::body() noexcept {
    return &_body;
}

const Expression *Function::argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::LOCAL, next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}

const Expression *Function::reference_argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::REFERENCE, next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}

const Expression *Function::local(const Type *type) noexcept {
    auto ret = _create_expression<RefExpr>(Variable(type, Variable::Tag::LOCAL,
                                                    next_variable_uid()));
    _scope_stack.back()->add_var(ret->variable());
    return ret;
}

const Expression *Function::literal(const Type *type, LiteralExpr::value_type value) noexcept {
    return _create_expression<LiteralExpr>(type, value);
}

const Expression *Function::binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
    return _create_expression<BinaryExpr>(type, op, lhs, rhs);
}

const Expression *Function::unary(const Type *type, UnaryOp op, const Expression *expression) noexcept {
    return _create_expression<UnaryExpr>(type, op, expression);
}

IfStmt *Function::if_(const Expression *expr) noexcept {
    return _create_statement<IfStmt>(expr);
}

SwitchStmt *Function::switch_(const Expression *expr) noexcept {
    return _create_statement<SwitchStmt>(expr);
}

SwitchCaseStmt *Function::switch_case(const Expression *expr) noexcept {
    return _create_statement<SwitchCaseStmt>(expr);
}

SwitchDefaultStmt *Function::switch_default() noexcept {
    return _create_statement<SwitchDefaultStmt>();
}

CommentStmt *Function::comment(ocarina::string_view string) noexcept {
    return _create_statement<CommentStmt>(string);
}

ocarina::span<const Variable> Function::arguments() const noexcept {
    return _arguments;
}

void Function::assign(const Expression *lhs, const Expression *rhs) noexcept {
    _create_statement<AssignStmt>(lhs, rhs);
}

uint64_t Function::_compute_hash() const noexcept {
    return _body.hash();
}

uint64_t Function::hash() const noexcept {
    if (!_hash_computed) {
        _hash = _compute_hash();
        _hash = hash64("__hash_function", _hash);
        _hash_computed = true;
    }
    return _hash;
}
}// namespace ocarina