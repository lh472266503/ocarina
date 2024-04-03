//
// Created by Zero on 30/04/2022.
//

#include "function.h"
#include "type_registry.h"
#include "generator/codegen.h"
#include "function_corrector.h"

namespace ocarina {

void Function::StructureSet::add(const ocarina::Type *type) noexcept {
    for (const Type *m : type->members()) {
        add(m);
    }
    if (struct_map.contains(type->hash()) || !type->is_structure()) {
        return;
    }
    struct_map.insert(make_pair(type->hash(), type));
    struct_lst.push_back(type);
}

ocarina::stack<Function *> &Function::_function_stack() noexcept {
    static ocarina::stack<Function *> ret;
    return ret;
}

void Function::correct() noexcept {
    TIMER(FunctionCorrect)
    FunctionCorrector().apply(this);
}

void Function::mark_variable_usage(ocarina::uint uid, ocarina::Usage usage) noexcept {
    OC_ASSERT(uid < _variable_usages.size());
    auto old_usage = to_underlying(_variable_usages[uid]);
    auto new_usage = to_underlying(usage);
    auto final_usage = old_usage | new_usage;
    if (final_usage != old_usage) {
        _variable_usages[uid] = static_cast<Usage>(final_usage);
    }
}

const RefExpr *Function::mapping_captured_argument(const Expression *outer_expr, bool *contain) noexcept {
    *contain = _expr_to_argument_index.contains(outer_expr);
    if (!*contain) {
        uint arg_index = static_cast<uint>(_appended_arguments.size());
        _expr_to_argument_index.insert(make_pair(outer_expr, arg_index));
        Variable variable(outer_expr->type(), Variable::Tag::REFERENCE, _next_variable_uid(), "", "outer");
        _appended_arguments.push_back(variable);
    }
    uint arg_index = _expr_to_argument_index.at(outer_expr);
    const RefExpr *ret = _ref(_appended_arguments.at(arg_index));
    return ret;
}

const RefExpr *Function::mapping_local_variable(const Expression *invoked_func_expr, bool *contain) noexcept {
    if (contain) {
        *contain = _outer_to_local.contains(invoked_func_expr);
    }
    if (!_outer_to_local.contains(invoked_func_expr)) {
        const RefExpr *ref_expr = local(invoked_func_expr->type());
        _outer_to_local.insert(make_pair(invoked_func_expr, ref_expr));
    }
    const RefExpr *ret = _outer_to_local.at(invoked_func_expr);
    return ret;
}

const RefExpr *Function::outer_to_local(const Expression *invoked_func_expr) noexcept {
    if (!_outer_to_local.contains(invoked_func_expr)) {
        return nullptr;
    }
    const RefExpr *ret = _outer_to_local.at(invoked_func_expr);
    return ret;
}

const RefExpr *Function::outer_to_argument(const Expression *invoked_func_expr) noexcept {
    OC_ASSERT(_expr_to_argument_index.contains(invoked_func_expr));
    uint arg_index = _expr_to_argument_index.at(invoked_func_expr);
    const RefExpr *ret = _ref(_appended_arguments.at(arg_index));
    return ret;
}

const RefExpr *Function::mapping_output_argument(const Expression *invoked_func_expr, bool *contain) noexcept {
    if (contain) {
        *contain = _expr_to_argument_index.contains(invoked_func_expr);
    }
    if (!_expr_to_argument_index.contains(invoked_func_expr)) {
        uint arg_index = static_cast<uint>(_appended_arguments.size());
        Variable variable(invoked_func_expr->type(), Variable::Tag::REFERENCE, _next_variable_uid(), "", "pass");
        _expr_to_argument_index.insert(make_pair(invoked_func_expr, arg_index));
        _appended_arguments.push_back(variable);
    }
    return outer_to_argument(invoked_func_expr);
}

void Function::append_output_argument(const Expression *expression, bool *contain) noexcept {
    if (contain) {
        *contain = _expr_to_argument_index.contains(expression);
    }
    if (_expr_to_argument_index.contains(expression)) {
        return;
    }
    _expr_to_argument_index.insert(make_pair(expression, static_cast<uint>(_appended_arguments.size())));
    Variable variable(expression->type(), Variable::Tag::REFERENCE, _next_variable_uid(), "", "output");
    _appended_arguments.push_back(variable);
    const RefExpr *ref_expr = _ref(variable);
    OC_ERROR_IF(expression->type()->is_resource());
    with(body(), [&] {
        assign(ref_expr, expression);
    });
}

namespace detail {

[[nodiscard]] string path_key(const Variable &arg, const vector<int> &path) noexcept {
    string ret = ocarina::format("{}", arg.uid());
    for (int i : path) {
        ret += "_" + to_string(i);
    }
    return ret;
}

}// namespace detail

void Function::process_param_struct_member(const ocarina::Variable &arg, const Type *type,
                                           vector<int> &path) noexcept {
    if (type->is_param_struct()) {
        splitting_param_struct(arg, type, path);
    } else {
        string key = detail::path_key(arg, path);
        Variable v(type, Variable::Tag::LOCAL, _next_variable_uid(), "", ocarina::format(""));
        _argument_map.insert(make_pair(key, v));
        _splitted_arguments.push_back(v);
    }
}

void Function::splitting_param_struct(const ocarina::Variable &arg, const Type *type,
                                      vector<int> &path) noexcept {
    for (int i = 0; i < type->members().size(); ++i) {
        path.push_back(i);
        process_param_struct_member(arg, type->members()[i], path);
        path.pop_back();
    }
}

void Function::splitting_arguments() noexcept {
    for (const Variable &arg : _arguments) {
        if (arg.type()->is_param_struct()) {
            vector<int> path;
            splitting_param_struct(arg, arg.type(), path);
        } else {
            _splitted_arguments.push_back(arg);
        }
    }
}

Function::~Function() {
    for (auto &mem : _temp_memory) {
        delete_with_allocator(mem.first);
    }
}

void Function::correct_used_structures() noexcept {
    for (const Variable &arg : _arguments) {
        add_used_structure(arg.type());
    }
    for (const Variable &arg : _appended_arguments) {
        add_used_structure(arg.type());
    }
    for (const CapturedResource &resource : _captured_resources) {
        add_used_structure(resource.type());
    }
}

void Function::_push(ocarina::Function *f) {
    _function_stack().push(f);
}

void Function::_pop(ocarina::Function *f) {
    OC_ASSERT(f == _function_stack().top());
    _function_stack().pop();
}

const RefExpr *Function::_ref(const ocarina::Variable &variable) noexcept {
    return create_expression<RefExpr>(variable);
}

uint Function::_next_variable_uid() noexcept {
    auto ret = _variable_usages.size();
    _variable_usages.push_back(Usage::NONE);
    return ret;
}

const Usage &Function::variable_usage(uint uid) const noexcept {
    OC_ASSERT(uid < _variable_usages.size());
    return _variable_usages[uid];
}

Usage &Function::variable_usage(uint uid) noexcept {
    OC_ASSERT(uid < _variable_usages.size());
    return _variable_usages[uid];
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

const Function *Function::add_used_function(SP<const Function> func) noexcept {
    _used_custom_func.push_back(func);
    return func.get();
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
        case Type::Tag::BYTE_BUFFER:
            tag = Variable::Tag::BYTE_BUFFER;
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
    Variable variable(type, Variable::Tag::REFERENCE, _next_variable_uid(), "", "ref");
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
    const Function *ptr = add_used_function(func);
    return create_expression<CallExpr>(type, ptr, std::move(args));
}

const CallExpr *Function::call(const ocarina::Type *type, string_view func_name,
                               ocarina::vector<const Expression *> args) noexcept {
    return create_expression<CallExpr>(type, func_name, ocarina::move(args));
}

const CallExpr *Function::call_builtin(const Type *type, CallOp op,
                                       ocarina::vector<const Expression *> args,
                                       ocarina::vector<CallExpr::Template> t_args) noexcept {
    if (to_underlying(op) >= to_underlying(CallOp::MAKE_RAY)) {
        set_raytracing(true);
    }
    return create_expression<CallExpr>(type, op, std::move(args), std::move(t_args));
}

const CapturedResource &Function::get_captured_resource(const Type *type, Variable::Tag tag, MemoryBlock block) noexcept {
    if (auto iter = std::find_if(_captured_resources.begin(),
                                 _captured_resources.end(),
                                 [&](auto v) {
                                     return v.handle_ptr() == block.address;
                                 });
        iter != _captured_resources.end()) {
        return *iter;
    }
    const RefExpr *expr = _ref(Variable(type, tag, _next_variable_uid()));
    _captured_resources.emplace_back(expr, type, block);
    return _captured_resources.back();
}

const CapturedResource &Function::add_captured_resource(const Type *type, Variable::Tag tag, MemoryBlock block) noexcept {
    const RefExpr *expr = _ref(Variable(type, tag, _next_variable_uid(), "", "cap_res"));
    _captured_resources.emplace_back(expr, type, block);
    return _captured_resources.back();
}

bool Function::has_captured_resource(const void *handle) const noexcept {
    bool ret = std::find_if(_captured_resources.begin(),
                            _captured_resources.end(),
                            [&](auto v) {
                                return v.handle_ptr() == handle;
                            }) != _captured_resources.end();
    return ret;
}

void Function::update_captured_resources(const Function *func) noexcept {
    func->for_each_captured_resource([&](const CapturedResource &var) {
        if (!has_captured_resource(var.handle_ptr())) {
            add_captured_resource(var.type(), var.expression()->variable().tag(), var.block());
        }
    });
}

vector<Variable> Function::all_arguments() const noexcept {
    vector<Variable> ret;
    for (const auto &v : arguments()) {
        ret.push_back(v);
    }
    for (const auto &res : captured_resources()) {
        ret.push_back(res.expression()->variable());
    }
    for (const auto &v : appended_arguments()) {
        ret.push_back(v);
    }
    return ret;
}

const CapturedResource *Function::get_captured_resource_by_handle(const void *handle) const noexcept {
    const CapturedResource *var = nullptr;
    for (const CapturedResource &v : _captured_resources) {
        if (v.handle_ptr() == handle) {
            var = &v;
            break;
        }
    }
    return var;
}

Function *Function::current() noexcept {
    if (_function_stack().empty()) {
        return nullptr;
    }
    return _function_stack().top();
}

ScopeStmt *Function::scope() noexcept {
    return create_statement<ScopeStmt>(false);
}

void Function::add_header(std::string_view fn) noexcept {
    _headers.push_back(fn);
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

ocarina::span<const Variable> Function::appended_arguments() const noexcept {
    return _appended_arguments;
}

ocarina::span<const Variable> Function::builtin_vars() const noexcept {
    return _builtin_vars;
}

ocarina::string Function::func_name(uint64_t ext_hash, string ext_name) const noexcept {
    uint64_t final_hash = ext_hash == 0 ? hash() : hash64(hash(), ext_hash);
    if (is_kernel()) {
        if (is_raytracing()) {
            return detail::raygen_name(final_hash, ocarina::move(ext_name));
        } else {
            return detail::kernel_name(final_hash, ocarina::move(ext_name));
        }
    } else {
        return detail::func_name(final_hash);
    }
}

void Function::assign(const Expression *lhs, const Expression *rhs) noexcept {
    create_statement<AssignStmt>(lhs, rhs);
}

uint64_t Function::_compute_hash() const noexcept {
    auto ret = _ret ? _ret->hash() : 0;
    for_each_header([&](string_view fn) {
        ret = hash64(ret, fn);
    });
    ret = hash64(tag(), ret);
    for (const Variable &v : _arguments) {
        ret = hash64(ret, v.hash());
    }
    for (const Variable &v : _appended_arguments) {
        ret = hash64(ret, v.hash());
    }
    for (const Variable &v : _builtin_vars) {
        ret = hash64(ret, v.hash());
    }
    for (const CapturedResource &v : _captured_resources) {
        ret = hash64(ret, v.hash());
    }
    ret = hash64(ret, _body.hash());
    return ret;
}

}// namespace ocarina