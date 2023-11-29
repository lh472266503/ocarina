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

class ArgumentBinding : public Hashable {
private:
    const Type *_type;
    const RefExpr *_expr{nullptr};
    MemoryBlock _block;

private:
    [[nodiscard]] uint64_t _compute_hash() const noexcept final {
        return hash64(type()->hash(), _expr->hash());
    }

public:
    ArgumentBinding(const RefExpr *expr, const Type *type, MemoryBlock block)
        : _type(type), _block(block), _expr(expr) {}

    [[nodiscard]] const Type *type() const noexcept { return _type; }
    [[nodiscard]] const void *handle_ptr() const noexcept {
        return _block.address;
    }
    [[nodiscard]] const MemoryBlock &block() const noexcept {
        return _block;
    }
    [[nodiscard]] size_t block_size() const noexcept { return _block.size; }
    [[nodiscard]] size_t block_alignment() const noexcept { return _block.alignment; }
    [[nodiscard]] const RefExpr *expression() const noexcept { return _expr; }
};

class OC_AST_API Function : public concepts::Noncopyable, public Hashable {
public:
    enum Tag : uint8_t {
        KERNEL,
        CALLABLE,
    };

    struct StructureSet {
        ocarina::map<uint64_t, const Type *> struct_map;
        vector<const Type *> struct_lst;
        void add(const Type *type) noexcept;
    };

private:
    mutable string _description{};
    const Type *_ret{nullptr};
    ocarina::vector<ocarina::unique_ptr<Expression>> _all_expressions;
    ocarina::vector<ocarina::unique_ptr<Statement>> _all_statements;
    ocarina::vector<Variable> _arguments;
    ocarina::vector<ArgumentBinding> _captured_vars;
    ocarina::vector<Variable> _builtin_vars;
    ocarina::vector<Usage> _variable_usages;
    ocarina::vector<ScopeStmt *> _scope_stack;
    /// use for assignment subscript access
    ocarina::vector<ocarina::pair<std::byte *, size_t>> _temp_memory;
    ScopeStmt _body{true};
    Tag _tag{Tag::CALLABLE};
    ocarina::map<uint64_t, const Function *> _used_custom_func;
    StructureSet _used_struct;
    mutable bool _raytracing{false};
    mutable uint3 _block_dim{make_uint3(0)};
    mutable uint3 _grid_dim{make_uint3(0)};

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
    [[nodiscard]] const RefExpr *_ref(const Variable &variable) noexcept {
        return _create_expression<RefExpr>(variable);
    }

    [[nodiscard]] const RefExpr *_builtin(Variable::Tag tag, const Type *type) noexcept;

    void add_used_function(const Function *func) noexcept;

    template<typename Stmt, typename... Args>
    auto _create_statement(Args &&...args) {
        auto stmt = ocarina::make_unique<Stmt>(std::forward<Args>(args)...);
        auto ret = stmt.get();
        _all_statements.push_back(std::move(stmt));
        current_scope()->add_stmt(ret);
        return ret;
    }
    [[nodiscard]] uint64_t _compute_hash() const noexcept override;
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
    void set_description(string desc) const noexcept { _description = ocarina::move(desc); }
    [[nodiscard]] string &description() const noexcept { return _description; }
    [[nodiscard]] auto used_custom_func() const noexcept { return _used_custom_func; }
    template<typename Visitor>
    void for_each_custom_func(Visitor &&visitor) const noexcept {
        for (const auto &[_, f] : _used_custom_func) {
            visitor(f);
        }
    }
    [[nodiscard]] auto used_structure() const noexcept { return _used_struct; }
    template<typename Visitor>
    void for_each_structure(Visitor &&visitor) const noexcept {
        for (const auto &type : _used_struct.struct_lst) {
            visitor(type);
        }
    }
    void add_used_structure(const Type *type) noexcept {
        _used_struct.add(type);
    }
    const ArgumentBinding &get_captured_var(const Type *type, Variable::Tag tag, MemoryBlock block) noexcept;
    [[nodiscard]] auto &captured_vars() const noexcept { return _captured_vars; }
    template<typename Visitor>
    void for_each_captured_var(Visitor &&visitor) const noexcept {
        for (const ArgumentBinding &var : _captured_vars) {
            visitor(var);
        }
    }
    [[nodiscard]] static Function *current() noexcept {
        if (_function_stack().empty()) {
            return nullptr;
        }
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
    decltype(auto) with(ScopeStmt *scope, Func &&func) noexcept {
        ScopeGuard guard(_scope_stack, scope);
        return func();
    }

    void mark_variable_usage(uint uid, Usage usage) noexcept {
        _variable_usages[uid] = usage;
    }
    template<typename Func>
    static auto define_kernel(Func &&func) noexcept {
        auto function = _define(Tag::KERNEL, std::forward<Func>(func));
        return function;
    }
    explicit Function(Tag tag) noexcept;
    ~Function() noexcept {
        for (auto &mem : _temp_memory) {
            delete_with_allocator(mem.first);
        }
    }
    template<typename T, typename... Args>
    T *create_temp_obj(Args &&...args) noexcept {
        T *ptr = new_with_allocator<T>(std::forward<Args>(args)...);
        _temp_memory.emplace_back(reinterpret_cast<std::byte *>(ptr), sizeof(T));
        return ptr;
    }

    void set_block_dim(uint x, uint y = 1, uint z = 1) const noexcept { _block_dim = make_uint3(x, y, z); }
    void set_block_dim(uint2 size) const noexcept { _block_dim = make_uint3(size, 1); }
    void set_block_dim(uint3 size) const noexcept { _block_dim = size; }
    [[nodiscard]] uint3 block_dim() const noexcept { return _block_dim; }

    void set_grid_dim(uint x, uint y = 1, uint z = 1) const noexcept { _grid_dim = make_uint3(x, y, z); }
    void set_grid_dim(uint2 size) const noexcept { _grid_dim = make_uint3(size, 1); }
    void set_grid_dim(uint3 size) const noexcept { _grid_dim = size; }
    [[nodiscard]] uint3 grid_dim() const noexcept { return _grid_dim; }

    void configure(uint3 grid_dim, uint3 block_dim) const noexcept {
        _grid_dim = grid_dim;
        _block_dim = block_dim;
    }

    [[nodiscard]] bool has_configure() const noexcept { return all(block_dim() != 0u) || all(grid_dim() != 0u); }

    [[nodiscard]] ocarina::string func_name() const noexcept;
    void assign(const Expression *lhs, const Expression *rhs) noexcept;
    void return_(const Expression *expression) noexcept;
    [[nodiscard]] const RefExpr *block_idx() noexcept;
    [[nodiscard]] const RefExpr *thread_idx() noexcept;
    [[nodiscard]] const RefExpr *thread_id() noexcept;
    [[nodiscard]] const RefExpr *dispatch_idx() noexcept;
    [[nodiscard]] const RefExpr *dispatch_id() noexcept;
    [[nodiscard]] const RefExpr *dispatch_dim() noexcept;
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *reference_argument(const Type *type) noexcept;
    [[nodiscard]] const RefExpr *local(const Type *type) noexcept;
    [[nodiscard]] const LiteralExpr *literal(const Type *type, basic_literal_t value) noexcept;
    [[nodiscard]] const UnaryExpr *unary(const Type *type, UnaryOp op, const Expression *expression) noexcept;
    [[nodiscard]] const BinaryExpr *binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept;
    [[nodiscard]] const ConditionalExpr *conditional(const Type *type, const Expression *pred, const Expression *t, const Expression *f) noexcept;
    [[nodiscard]] const CastExpr *cast(const Type *type, CastOp op, const Expression *expression) noexcept;
    [[nodiscard]] const SubscriptExpr *subscript(const Type *type, const Expression *range, const Expression *index) noexcept;
    [[nodiscard]] const SubscriptExpr *subscript(const Type *type, const Expression *range,
                                                 vector<const Expression *> indexes) noexcept;
    [[nodiscard]] const MemberExpr *swizzle(const Type *type, const Expression *obj, uint16_t mask, uint16_t swizzle_size) noexcept;
    [[nodiscard]] const MemberExpr *member(const Type *type, const Expression *obj, int index) noexcept;
    const CallExpr *call(const Type *type, const Function *func, ocarina::vector<const Expression *> args) noexcept;
    const CallExpr *call_builtin(const Type *type, CallOp op, ocarina::vector<const Expression *> args,
                                 ocarina::vector<CallExpr::Template> t_args = {}) noexcept;
    [[nodiscard]] ScopeStmt *scope() noexcept;
    [[nodiscard]] IfStmt *if_(const Expression *expr) noexcept;
    [[nodiscard]] SwitchStmt *switch_(const Expression *expr) noexcept;
    [[nodiscard]] SwitchCaseStmt *switch_case(const Expression *expr) noexcept;
    const ExprStmt *expr_statement(const Expression *expr) noexcept;
    void break_() noexcept;
    [[nodiscard]] SwitchDefaultStmt *switch_default() noexcept;
    [[nodiscard]] LoopStmt *loop() noexcept;
    [[nodiscard]] ForStmt *for_(const Expression *init, const Expression *cond, const Expression *step) noexcept;
    void continue_() noexcept;
    void comment(const ocarina::string &string) noexcept;
    void print(string fmt, const vector<const Expression *> &args) noexcept;
    [[nodiscard]] const ScopeStmt *body() const noexcept;
    [[nodiscard]] ScopeStmt *body() noexcept;
    [[nodiscard]] ocarina::span<const Variable> arguments() const noexcept;
    [[nodiscard]] ocarina::span<const Variable> builtin_vars() const noexcept;
    [[nodiscard]] constexpr Tag tag() const noexcept { return _tag; }
    [[nodiscard]] constexpr bool is_callable() const noexcept { return _tag == Tag::CALLABLE; }
    [[nodiscard]] constexpr bool is_kernel() const noexcept { return _tag == Tag::KERNEL; }
    [[nodiscard]] constexpr bool is_raytracing_kernel() const noexcept { return is_raytracing() && is_kernel(); }
    [[nodiscard]] constexpr bool is_general_kernel() const noexcept { return !is_raytracing() && is_kernel(); }
    [[nodiscard]] constexpr bool is_raytracing() const noexcept { return _raytracing; }
    [[nodiscard]] constexpr const Type *return_type() const noexcept { return _ret; }
    constexpr void set_raytracing(bool val) const noexcept { _raytracing = val; }
};

}// namespace ocarina
