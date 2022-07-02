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

void Function::add_used_function(const Function *func) noexcept {
    _used_custom_func.emplace(func);
}

const RefExpr *Function::argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::LOCAL, _next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}

const RefExpr *Function::reference_argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::REFERENCE, _next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}

const RefExpr *Function::local(const Type *type) noexcept {
    auto ret = _create_expression<RefExpr>(Variable(type, Variable::Tag::LOCAL,
                                                    _next_variable_uid()));
    current_scope()->add_var(ret->variable());
    return ret;
}

const LiteralExpr *Function::literal(const Type *type, LiteralExpr::value_type value) noexcept {
    return _create_expression<LiteralExpr>(type, value);
}

const BinaryExpr *Function::binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
    return _create_expression<BinaryExpr>(type, op, lhs, rhs);
}

const UnaryExpr *Function::unary(const Type *type, UnaryOp op, const Expression *expression) noexcept {
    return _create_expression<UnaryExpr>(type, op, expression);
}

const CastExpr *Function::cast(const Type *type, CastOp op, const Expression *expression) noexcept {
    return _create_expression<CastExpr>(type, op, expression);
}

const AccessExpr *Function::access(const Type *type, const Expression *range, const Expression *index) noexcept {
    return _create_expression<AccessExpr>(type, range, index);
}

const MemberExpr *Function::swizzle(const Type *type, const Expression *obj, uint16_t mask, uint16_t swizzle_size) noexcept {
    return _create_expression<MemberExpr>(type, obj, mask, swizzle_size);
}

const MemberExpr *Function::member(const Type *type, const Expression *obj, ocarina::string_view field_name) noexcept {
    return _create_expression<MemberExpr>(type, obj, field_name);
}

const CallExpr *Function::call(const Type *type, const Function *func,
                               ocarina::vector<const Expression *> args) noexcept {
    add_used_function(func);
    return _create_expression<CallExpr>(type, func, std::move(args));
}

const CallExpr *Function::call_builtin(const Type *type, CallOp op,
                                       ocarina::vector<const Expression *> args) noexcept {
    return _create_expression<CallExpr>(type, op, std::move(args));
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

const ExprStmt *Function::expr_statement(const Expression *expr) noexcept {
    return _create_statement<ExprStmt>(expr);
}

SwitchDefaultStmt *Function::switch_default() noexcept {
    return _create_statement<SwitchDefaultStmt>();
}

LoopStmt *Function::loop() noexcept {
    return _create_statement<LoopStmt>();
}

ForStmt *Function::for_(const Expression *init, const Expression *cond, const Expression *step) noexcept {
    return _create_statement<ForStmt>(init, cond, step);
}

void Function::continue_() noexcept {
    _create_statement<ContinueStmt>();
}

void Function::break_() noexcept {
    _create_statement<BreakStmt>();
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
    auto ret = _ret ? _ret->hash() : 0;
    for (const Variable &v : _arguments) {
        ret = hash64(ret, v.hash());
    }
    ret = hash64(ret, _body.hash());
    return ret;
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