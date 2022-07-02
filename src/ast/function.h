//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "core/basic_types.h"
#include "core/stl.h"
#include "core/header.h"
#include "type.h"
#include "expression.h"
#include "statement.h"
#include "usage.h"
#include "op.h"

namespace ocarina {

class Statement;
class ScopeStmt;
class RefExpr;
class IfStmt;

class OC_AST_API Function : public concepts::Noncopyable {
public:
    enum struct Tag : uint {
        KERNEL,
        CALLABLE,
    };

private:
    const Type *_ret{nullptr};
    ocarina::vector<ocarina::unique_ptr<Expression>> _all_expressions;
    ocarina::vector<ocarina::unique_ptr<Statement>> _all_statements;
    ocarina::vector<Variable> _arguments;
    ocarina::vector<Usage> _variable_usages;
    ocarina::vector<ScopeStmt *> _scope_stack;
    /// use for assignment subscript access
    ocarina::vector<ocarina::pair<std::byte *, size_t>> _temp_memory;
    ScopeStmt _body;
    mutable uint64_t _hash{0};
    mutable bool _hash_computed{false};
    Tag _tag{Tag::CALLABLE};
    ocarina::set<const Function *> _used_custom_func;

private:
    static ocarina::vector<Function *> &_function_stack() noexcept;
    static void _push(Function *f) {
        _function_stack().push_back(f);
    }
    static void _pop(Function *f) {
        OC_ASSERT(f == _function_stack().back());
        _function_stack().pop_back();
    }

    [[nodiscard]] uint _next_variable_uid() noexcept {
        auto ret = _variable_usages.size();
        _variable_usages.push_back(Usage::NONE);
        return ret;
    }

    template<typename Func>
    static auto _define(Function::Tag tag, Func &&func) noexcept {
        auto ret = ocarina::make_unique<Function>(tag);
        _push(ret.get());
        ret->with(ret->body(), std::forward<Func>(func));
        _pop(ret.get());
        return ret;
    }

    template<typename Expr, typename... Args>
    [[nodiscard]] auto _create_expression(Args &&...args) {
        auto expr = ocarina::make_unique<Expr>(std::forward<Args>(args)...);
        auto ret = expr.get();
        _all_expressions.push_back(std::move(expr));
        return ret;
    }
    [[nodiscard]] const RefExpr *_ref(Variable variable) noexcept {
        return _create_expression<RefExpr>(variable);
    }

    void add_used_function(const Function *func) noexcept;

    template<typename Stmt, typename... Args>
    auto _create_statement(Args &&...args) {
        auto stmt = ocarina::make_unique<Stmt>(std::forward<Args>(args)...);
        auto ret = stmt.get();
        _all_statements.push_back(std::move(stmt));
        current_scope()->add_stmt(ret);
        return ret;
    }
    [[nodiscard]] uint64_t _compute_hash() const noexcept;
    class ScopeGuard {
    private:
        ocarina::vector<ScopeStmt *> &_scope_stack;
        ScopeStmt *_scope;

    public:
        ScopeGuard(ocarina::vector<ScopeStmt *> &stack, ScopeStmt *scope)
            : _scope_stack(stack), _scope(scope) {
            _scope_stack.push_back(scope);
        }
        ~ScopeGuard() {
            _scope_stack.pop_back();
        }
    };

public:
    [[nodiscard]] static Function *current() noexcept {
        return _function_stack().back();
    }
    template<typename Func>
    static auto define_callable(Func &&func) noexcept {
        return _define(Tag::CALLABLE, std::forward<Func>(func));
    }
    [[nodiscard]] const ScopeStmt *current_scope() const noexcept {
        return _scope_stack.back();
    }
    [[nodiscard]] ScopeStmt *current_scope() noexcept {
        return _scope_stack.back();
    }
    template<typename Func>
    decltype(auto) with(ScopeStmt *scope, Func&& func) noexcept {
        ScopeGuard guard(_scope_stack, scope);
        return func();
    }

    void mark_variable_usage(uint uid, Usage usage) noexcept {
        _variable_usages[uid] = usage;
    }
    template<typename Func>
    static auto define_kernel(Func &&func) noexcept {
        return _define(Tag::KERNEL, std::forward<Func>(func));
    }
    Function() noexcept = default;
    explicit Function(Tag tag) noexcept;
    ~Function() noexcept {
        for (auto &mem : _temp_memory) {
            delete_with_allocator(mem.first);
        }
    }
    template<typename T, typename... Args>
    T *create_temp_obj(Args &&...args) noexcept {
        T *ptr = new_with_allocator<T>(std::forward<Args>(args)...);
        _temp_memory.emplace_back(reinterpret_cast<std::byte*>(ptr), sizeof(T));
        return ptr;
    }

    [[nodiscard]] auto used_custom_func() const noexcept {
        return _used_custom_func;
    }

    void assign(const Expression *lhs, const Expression *rhs) noexcept;
    void return_(const Expression *expression) noexcept;
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *reference_argument(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *local(const Type *type) noexcept;
    [[nodiscard]] const LiteralExpr *literal(const Type *type, basic_literal_t value) noexcept;
    [[nodiscard]] const BinaryExpr *binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] const UnaryExpr *unary(const Type *type, UnaryOp op, const Expression *expression) noexcept;
    [[nodiscard]] const CastExpr *cast(const Type *type, CastOp op, const Expression *expression) noexcept;
    [[nodiscard]] const AccessExpr *access(const Type *type, const Expression *range, const Expression *index) noexcept;
    [[nodiscard]] const MemberExpr *swizzle(const Type *type, const Expression *obj, uint16_t mask, uint16_t swizzle_size) noexcept;
    [[nodiscard]] const MemberExpr *member(const Type *type, const Expression *obj, ocarina::string_view field_name) noexcept;
    const CallExpr *call(const Type *type, const Function*func, ocarina::vector<const Expression *> args) noexcept;
    const CallExpr *call_builtin(const Type *type, CallOp op, ocarina::vector<const Expression *> args) noexcept;
    [[nodiscard]] IfStmt *if_(const Expression *expr) noexcept;
    [[nodiscard]] SwitchStmt *switch_(const Expression *expr) noexcept;
    [[nodiscard]] SwitchCaseStmt *switch_case(const Expression *expr) noexcept;
    const ExprStmt * expr_statement(const Expression *expr) noexcept;
    void break_() noexcept;
    [[nodiscard]] SwitchDefaultStmt *switch_default() noexcept;
    [[nodiscard]] LoopStmt *loop() noexcept;
    [[nodiscard]] ForStmt *for_(const Expression *init, const Expression *cond, const Expression *step) noexcept;
    void continue_() noexcept;
    CommentStmt *comment(ocarina::string_view string) noexcept;
    [[nodiscard]] const ScopeStmt *body() const noexcept;
    [[nodiscard]] ScopeStmt *body() noexcept;
    [[nodiscard]] uint64_t hash() const noexcept;
    [[nodiscard]] ocarina::span<const Variable> arguments() const noexcept;
    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] bool is_callable() const noexcept { return _tag == Tag::CALLABLE; }
    [[nodiscard]] bool is_kernel() const noexcept { return _tag == Tag::KERNEL; }
    [[nodiscard]] const Type *return_type() const noexcept { return _ret; }
};

}// namespace ocarina

