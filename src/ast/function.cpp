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

void Function::return_(const Expression *expression) noexcept {
    _impl->return_(expression);
}

Function::Function(Function::Tag tag) noexcept
    : _impl(ocarina::make_shared<Impl>(tag)) {
}

const ScopeStmt *Function::body() const noexcept {
    return _impl->_scope_stack.back();
}

const RefExpr *Function::argument(const Type *type) noexcept {
    return _impl->argument(type);
}

const RefExpr *Function::reference_argument(const Type *type) noexcept {
    return _impl->reference_argument(type);
}

const Expression *Function::local(const Type *type) noexcept {
    auto ret = _impl->create_expression<RefExpr>(Variable(type, Variable::Tag::LOCAL,
                                                          _impl->next_variable_uid()));
    _impl->_local_variables.push_back(ret->variable());
    return ret;
}

const LiteralExpr *Function::literal(const Type *type, LiteralExpr::value_type value) noexcept {
    return _impl->create_expression<LiteralExpr>(type, value);
}

const BinaryExpr *Function::binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
    return _impl->create_expression<BinaryExpr>(type, op, lhs, rhs);
}

const UnaryExpr *Function::unary(const Type *type, UnaryOp op, const Expression *expression) noexcept {
    return _impl->create_expression<UnaryExpr>(type, op, expression);
}

void Function::mark_variable_usage(uint uid, Usage usage) noexcept {
    _impl->mark_variable_usage(uid, usage);
}

ocarina::span<const Variable> Function::arguments() const noexcept {
    return _impl->arguments();
}

const Type *Function::return_type() const noexcept {
    return _impl->return_type();
}

Function::Tag Function::tag() const noexcept {
    return _impl->tag();
}

bool Function::is_callable() const noexcept {
    return _impl->is_callable();
}

bool Function::is_kernel() const noexcept {
    return _impl->is_kernel();
}
void Function::assign(const Expression *lhs, const Expression *rhs) noexcept {
    _impl->assign(lhs, rhs);
}
uint64_t Function::hash() const noexcept {
    return _impl->hash();
}
}// namespace ocarina