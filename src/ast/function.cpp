//
// Created by Zero on 30/04/2022.
//

#include "function.h"
#include "type_registry.h"
#include "generator/codegen.h"

namespace ocarina {

void Function::StructureSet::add(const ocarina::Type *type) noexcept {
    if (struct_map.contains(type->hash()) || !type->is_structure() ||
        type->description() == TypeDesc<Hit>::description() ||
        type->description() == TypeDesc<Ray>::description()) {
        return;
    }
    for (const Type *m : type->members()) {
        add(m);
    }
    struct_map.insert(make_pair(type->hash(), type));
    struct_lst.push_back(type);
}

ocarina::vector<Function *> &Function::_function_stack() noexcept {
    static ocarina::vector<Function *> ret;
    return ret;
}

Function::~Function() {
    for (auto &mem : _temp_memory) {
        delete_with_allocator(mem.first);
    }
}

void Function::_push(ocarina::Function *f) {
    _function_stack().push_back(f);
}

void Function::_pop(ocarina::Function *f) {
    OC_ASSERT(f == _function_stack().back());
    _function_stack().pop_back();
}

const RefExpr *Function::_ref(const ocarina::Variable &variable) noexcept {
    return create_expression<RefExpr>(variable);
}

uint Function::_next_variable_uid() noexcept {
    auto ret = _variable_usages.size();
    _variable_usages.push_back(Usage::NONE);
    return ret;
}

void Function::return_(const Expression *expression) noexcept {
    if (expression) {
        _ret = expression->type();
    }
    create_statement<ReturnStmt>(expression);
}

Function::Function(Function::Tag tag) noexcept
    : _tag(tag) {}

const ScopeStmt *Function::body() const noexcept {
    return &_body;
}

ScopeStmt *Function::body() noexcept {
    return &_body;
}

const RefExpr *Function::_builtin(Variable::Tag tag, const Type *type) noexcept {
    Variable variable(type, tag, _next_variable_uid());
    if (auto iter = std::find_if(_builtin_vars.begin(),
                                 _builtin_vars.end(),
                                 [&](auto v) {
                                     return v.tag() == tag;
                                 });
        iter != _builtin_vars.end()) {
        return _ref(*iter);
    }
    _builtin_vars.push_back(variable);
    return _ref(variable);
}

void Function::add_used_function(SP<const Function> func) noexcept {
    _used_custom_func.insert(make_pair(func->hash(), func));
}

const RefExpr *Function::block_idx() noexcept {
    return _builtin(Variable::Tag::BLOCK_IDX, Type::of<uint3>());
}

const RefExpr *Function::thread_idx() noexcept {
    return _builtin(Variable::Tag::THREAD_IDX, Type::of<uint3>());
}

const RefExpr *Function::dispatch_dim() noexcept {
    return _builtin(Variable::Tag::DISPATCH_DIM, Type::of<uint3>());
}

const RefExpr *Function::dispatch_id() noexcept {
    return _builtin(Variable::Tag::DISPATCH_ID, Type::of<uint>());
}

const RefExpr *Function::thread_id() noexcept {
    return _builtin(Variable::Tag::THREAD_ID, Type::of<uint>());
}

const RefExpr *Function::dispatch_idx() noexcept {
    return _builtin(Variable::Tag::DISPATCH_IDX, Type::of<uint3>());
}

const RefExpr *Function::argument(const Type *type) noexcept {
    Variable::Tag tag;
    switch (type->tag()) {
        case Type::Tag::BUFFER:
            tag = Variable::Tag::BUFFER;
            break;
        case Type::Tag::TEXTURE:
            tag = Variable::Tag::TEXTURE;
            break;
        case Type::Tag::ACCEL:
            tag = Variable::Tag::ACCEL;
            break;
        default:
            tag = Variable::Tag::LOCAL;
            break;
    }
    Variable variable(type, tag, _next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}

const RefExpr *Function::reference_argument(const Type *type) noexcept {
    Variable variable(type, Variable::Tag::REFERENCE, _next_variable_uid());
    _arguments.push_back(variable);
    return _ref(variable);
}

const RefExpr *Function::local(const Type *type) noexcept {
    auto ret = create_expression<RefExpr>(Variable(type, Variable::Tag::LOCAL,
                                                   _next_variable_uid()));
    body()->add_var(ret->variable());
    return ret;
}

const LiteralExpr *Function::literal(const Type *type, LiteralExpr::value_type value) noexcept {
    return create_expression<LiteralExpr>(type, value);
}

const UnaryExpr *Function::unary(const Type *type, UnaryOp op,
                                 const Expression *expression) noexcept {
    return create_expression<UnaryExpr>(type, op, expression);
}

const BinaryExpr *Function::binary(const Type *type, BinaryOp op, const Expression *lhs,
                                   const Expression *rhs) noexcept {
    return create_expression<BinaryExpr>(type, op, lhs, rhs);
}

const ConditionalExpr *Function::conditional(const Type *type, const Expression *pred,
                                             const Expression *t,
                                             const Expression *f) noexcept {
    return create_expression<ConditionalExpr>(type, pred, t, f);
}

const CastExpr *Function::cast(const Type *type, CastOp op, const Expression *expression) noexcept {
    return create_expression<CastExpr>(type, op, expression);
}

const SubscriptExpr *Function::subscript(const Type *type, const Expression *range,
                                         const Expression *index) noexcept {
    return create_expression<SubscriptExpr>(type, range, index);
}

const SubscriptExpr *Function::subscript(const Type *type, const Expression *range,
                                         vector<const Expression *> indexes) noexcept {
    return create_expression<SubscriptExpr>(type, range, indexes);
}

const MemberExpr *Function::swizzle(const Type *type, const Expression *obj, uint16_t mask,
                                    uint16_t swizzle_size) noexcept {
    return create_expression<MemberExpr>(type, obj, mask, swizzle_size);
}

const MemberExpr *Function::member(const Type *type, const Expression *obj, int index) noexcept {
    return create_expression<MemberExpr>(type, obj, index, 0);
}

const CallExpr *Function::call(const Type *type, SP<const Function> func,
                               ocarina::vector<const Expression *> args) noexcept {
    add_used_function(func);
    return create_expression<CallExpr>(type, func.get(), std::move(args));
}

const CallExpr *Function::call_builtin(const Type *type, CallOp op,
                                       ocarina::vector<const Expression *> args,
                                       ocarina::vector<CallExpr::Template> t_args) noexcept {
    if (to_underlying(op) >= to_underlying(CallOp::MAKE_RAY)) {
        set_raytracing(true);
    }
    return create_expression<CallExpr>(type, op, std::move(args), std::move(t_args));
}

const CapturedVar &Function::get_captured_var(const Type *type, Variable::Tag tag, MemoryBlock block) noexcept {
    if (auto iter = std::find_if(_captured_vars.begin(),
                                 _captured_vars.end(),
                                 [&](auto v) {
                                     return v.handle_ptr() == block.address;
                                 });
        iter != _captured_vars.end()) {
        return *iter;
    }
    const RefExpr *expr = _ref(Variable(type, tag, _next_variable_uid()));
    _captured_vars.emplace_back(expr, type, block);
    return _captured_vars.back();
}

const CapturedVar &Function::add_captured_var(const Type *type, Variable::Tag tag, MemoryBlock block) noexcept {
    const RefExpr *expr = _ref(Variable(type, tag, _next_variable_uid()));
    _captured_vars.emplace_back(expr, type, block);
    return _captured_vars.back();
}

bool Function::has_captured_var(const void *handle) const noexcept {
    bool ret = std::find_if(_captured_vars.begin(),
                            _captured_vars.end(),
                            [&](auto v) {
                                return v.handle_ptr() == handle;
                            }) != _captured_vars.end();
    return ret;
}

void Function::update_captured_vars(const Function *func) noexcept {
    func->for_each_captured_var([&](const CapturedVar &var) {
        if (!has_captured_var(var.handle_ptr())) {
            add_captured_var(var.type(), var.expression()->variable().tag(), var.block());
        }
    });
}

const CapturedVar *Function::get_captured_var_by_handle(const void *handle) const noexcept {
    const CapturedVar *var = nullptr;
    for (const CapturedVar &v : _captured_vars) {
        if (v.handle_ptr() == handle) {
            var = &v;
            break;
        }
    }
    return var;
}

void Function::_add_exterior_expression(const ocarina::Expression *expression) noexcept {
    if (std::find(_exterior_expressions.begin(), _exterior_expressions.end(), expression) == _exterior_expressions.end()) {
        _exterior_expressions.push_back(expression);
    }
}

bool Function::is_exterior(const ocarina::Expression *expression) const noexcept {
    return expression && (expression->context() != this);
}

void Function::remedy_ast_nodes() noexcept {
    for (const Expression *expression : _exterior_expressions) {

        int i = 0;
    }
}

Function *Function::current() noexcept {
    if (_function_stack().empty()) {
        return nullptr;
    }
    return _function_stack().back();
}

ScopeStmt *Function::scope() noexcept {
    return create_statement<ScopeStmt>(false);
}

IfStmt *Function::if_(const Expression *expr) noexcept {
    return create_statement<IfStmt>(expr);
}

SwitchStmt *Function::switch_(const Expression *expr) noexcept {
    return create_statement<SwitchStmt>(expr);
}

SwitchCaseStmt *Function::switch_case(const Expression *expr) noexcept {
    return create_statement<SwitchCaseStmt>(expr);
}

const ExprStmt *Function::expr_statement(const Expression *expr) noexcept {
    return create_statement<ExprStmt>(expr);
}

SwitchDefaultStmt *Function::switch_default() noexcept {
    return create_statement<SwitchDefaultStmt>();
}

LoopStmt *Function::loop() noexcept {
    return create_statement<LoopStmt>();
}

ForStmt *Function::for_(const Expression *init, const Expression *cond, const Expression *step) noexcept {
    return create_statement<ForStmt>(init, cond, step);
}

void Function::continue_() noexcept {
    create_statement<ContinueStmt>();
}

void Function::break_() noexcept {
    create_statement<BreakStmt>();
}

void Function::comment(const ocarina::string &string) noexcept {
    create_statement<CommentStmt>(string);
}

void Function::print(string fmt, const vector<const Expression *> &args) noexcept {
    create_statement<PrintStmt>(fmt, args);
}

ocarina::span<const Variable> Function::arguments() const noexcept {
    return _arguments;
}

ocarina::span<const Variable> Function::builtin_vars() const noexcept {
    return _builtin_vars;
}

ocarina::string Function::func_name() const noexcept {
    if (is_kernel()) {
        if (is_raytracing()) {
            return detail::raygen_name(hash());
        } else {
            return detail::kernel_name(hash());
        }
    } else {
        return detail::func_name(hash());
    }
}

void Function::assign(const Expression *lhs, const Expression *rhs) noexcept {
    create_statement<AssignStmt>(lhs, rhs);
}

uint64_t Function::_compute_hash() const noexcept {
    auto ret = _ret ? _ret->hash() : 0;
    ret = hash64(tag(), ret);
    for (const Variable &v : _arguments) {
        ret = hash64(ret, v.hash());
    }
    for (const Variable &v : _builtin_vars) {
        ret = hash64(ret, v.hash());
    }
    for (const CapturedVar &v : _captured_vars) {
        ret = hash64(ret, v.hash());
    }
    ret = hash64(ret, _body.hash());
    return ret;
}

}// namespace ocarina