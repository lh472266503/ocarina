//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/basic_types.h"
#include "core/stl.h"
#include "core/header.h"
#include "type.h"
#include "variable.h"
#include "expression.h"

namespace ocarina {

class Statement;
class ScopeStmt;

class OC_AST_API Function {
public:
    enum struct Tag : uint {
        KERNEL,
        CALLABLE,
    };

private:
    class Impl;
    ocarina::unique_ptr<Impl> _impl{};

private:
    static ocarina::vector<Function *> &_function_stack() noexcept;
    static void _push(Function *f) {
        _function_stack().push_back(f);
    }
    static void _pop(Function *f) {
        OC_ASSERT(f == _function_stack().back());
        _function_stack().pop_back();
    }
    template<typename Func>
    static auto _define(Function::Tag tag, Func &&func) noexcept {
        auto ret = Function(tag);
        _push(&ret);
        func();
        _pop(&ret);
        return ret;
    }

public:
    [[nodiscard]] static Function *current() noexcept {
        return _function_stack().back();
    }
    template<typename Func>
    static auto define_callable(Func &&func) noexcept {
        return _define(Tag::CALLABLE, std::forward<Func>(func));
    }
    template<typename Func>
    static auto define_kernel(Func &&func) noexcept {
        return _define(Tag::KERNEL, std::forward<Func>(func));
    }
    Function() noexcept = default;
    explicit Function(Tag tag) noexcept;
    void assign(const Expression *lhs, const Expression *rhs) noexcept;
    void mark_variable_usage(uint uid, Usage usage) noexcept;
    void return_(const Expression *expression) noexcept;
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *reference_argument(const Type *type) noexcept;
    [[nodiscard]] const Expression *local(const Type *type) noexcept;
    [[nodiscard]] const LiteralExpr *literal(const Type *type, LiteralExpr::value_type value) noexcept;
    [[nodiscard]] const BinaryExpr *binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] const UnaryExpr *unary(const Type *type, UnaryOp op, const Expression *expression) noexcept;
    [[nodiscard]] const ScopeStmt *body() const noexcept;
    [[nodiscard]] ScopeStmt *body() noexcept;
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] ocarina::span<const Variable> arguments() const noexcept;
    [[nodiscard]] const ScopeStmt *current_scope() const noexcept;
    [[nodiscard]] Tag tag() const noexcept;
    [[nodiscard]] bool is_callable() const noexcept;
    [[nodiscard]] bool is_kernel() const noexcept;
    [[nodiscard]] const Type *return_type() const noexcept;
};

}// namespace ocarina

#ifndef NDEBUG

#include "function_impl.h"

#endif