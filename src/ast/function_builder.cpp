//
// Created by Zero on 03/05/2022.
//

#include "function_builder.h"

namespace katana {

void FunctionBuilder::push(FunctionBuilder *builder) noexcept {
    _function_stack().push_back(builder);
}

void FunctionBuilder::pop(FunctionBuilder *builder) noexcept {
    _function_stack().pop_back();
}

katana::vector<FunctionBuilder *> &FunctionBuilder::_function_stack() noexcept {
    static thread_local katana::vector<FunctionBuilder *> stack;
    return stack;
}

FunctionBuilder *FunctionBuilder::current() noexcept {
    return _function_stack().back();
}

void FunctionBuilder::mark_variable_usage(uint uid, Usage usage) noexcept {
    auto old_usage = to_underlying(_variable_usages[uid]);
    auto u = static_cast<Usage>(old_usage | to_underlying(usage));
    _variable_usages[uid] = u;
}

uint FunctionBuilder::_next_variable_uid() noexcept {
    auto uid = static_cast<uint32_t>(_variable_usages.size());
    _variable_usages.emplace_back(Usage::NONE);
    return uid;
}

const RefExpr *FunctionBuilder::_ref(Variable variable) noexcept {
    return _create_expression<RefExpr>(variable);
}

const RefExpr *FunctionBuilder::reference_argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::REFERENCE, _next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}

const RefExpr *FunctionBuilder::argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::LOCAL, _next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}
const BinaryExpr *FunctionBuilder::binary(const Type *type, const Expression *lhs, const Expression *rhs, BinaryOp op) noexcept {
    return _create_expression<BinaryExpr>(type, op, lhs, rhs);
}
const LiteralExpr *FunctionBuilder::literal(const Type *type, LiteralExpr::value_type value) noexcept {
    return _create_expression<LiteralExpr>(type, value);
}
const RefExpr *FunctionBuilder::local(const Type *type) noexcept {
    return _create_expression<RefExpr>(Variable(type, Variable::Tag::LOCAL, _next_variable_uid()));
}

void FunctionBuilder::return_(const Expression *expression) noexcept {
    _create_statement<ReturnStmt>(expression);
}
void FunctionBuilder::assign(const RefExpr *lhs, const Expression *rhs) noexcept {

}

}// namespace katana