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
    _impl->return_(expression);
}

Function::Function(Function::Tag tag) noexcept
    : _impl(ocarina::make_unique<Impl>(tag)) {
}

const ScopeStmt *Function::body() const noexcept {
    return _impl->_scope_stack.front();
}

ScopeStmt *Function::body() noexcept {
    return _impl->_scope_stack.front();
}

ConstExprPtr Function::argument(const Type *type) noexcept {
    return _impl->argument(type);
}

ConstExprPtr Function::reference_argument(const Type *type) noexcept {
    return _impl->reference_argument(type);
}

ConstExprPtr Function::local(const Type *type) noexcept {
    auto ret = _impl->create_expression<RefExpr>(Variable(type, Variable::Tag::LOCAL,
                                                          _impl->next_variable_uid()));
    body()->add_var(ret->variable());
    return ret;
}

ConstExprPtr Function::literal(const Type *type, LiteralExpr::value_type value) noexcept {
    return _impl->create_expression<LiteralExpr>(type, value);
}

ConstExprPtr Function::binary(const Type *type, BinaryOp op, ConstExprPtr lhs, ConstExprPtr rhs) noexcept {
    return _impl->create_expression<BinaryExpr>(type, op, lhs, rhs);
}

ConstExprPtr Function::unary(const Type *type, UnaryOp op, ConstExprPtr expression) noexcept {
    return _impl->create_expression<UnaryExpr>(type, op, expression);
}

void Function::mark_variable_usage(uint uid, Usage usage) noexcept {
    _impl->mark_variable_usage(uid, usage);
}

ocarina::span<const Variable> Function::arguments() const noexcept {
    return _impl->arguments();
}

const ScopeStmt *Function::current_scope() const noexcept {
    return _impl->current_scope();
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
void Function::assign(ConstExprPtr lhs, ConstExprPtr rhs) noexcept {
    _impl->assign(lhs, rhs);
}
uint64_t Function::hash() const noexcept {
    return _impl->hash();
}
void Function::postprocess() noexcept {
    
}

}// namespace ocarina