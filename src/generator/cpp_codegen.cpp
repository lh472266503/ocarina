//
// Created by Zero on 08/06/2022.
//

#include "cpp_codegen.h"
#include "ast/type_registry.h"

namespace ocarina {

namespace detail {
struct LiteralPrinter {
    using Scratch = Codegen::Scratch;
    Scratch &scratch;
    explicit LiteralPrinter(Scratch &scratch) : scratch(scratch) {}

    template<typename T>
    requires(is_scalar_v<T> || is_vector_v<T>)
    void operator()(T v) {
        if constexpr (ocarina::is_scalar_v<T>) {
            if constexpr (ocarina::is_floating_point_v<T>) {
                if (std::isnan(v)) [[unlikely]] {
                    OC_ERROR("nan error!");
                } else if (std::isinf(v)) {
                    scratch << (v < 0.f ? "(1.f/-0.f)" : "1.f/+0.f");
                } else {
                    scratch << v;
                }
            } else {
                scratch << v;
            }
        } else {
            static constexpr auto dim = ocarina::vector_dimension_v<T>;
            auto printer = *this;
            scratch << TYPE_PREFIX << Type::of<T>()->name() << "(";
            for (int i = 0; i < dim; ++i) {
                const char *token = i == dim - 1 ? ")" : ",";
                printer(v[i]);
                scratch << token;
            }
        }
    }
    template<size_t N>
    void operator()(Matrix<N> m) {
        auto printer = *this;
        scratch << TYPE_PREFIX << Type::of<Matrix<N>>()->name() << "(";
        for (int i = 0; i < N; ++i) {
            const char *token = i == N - 1 ? ")" : ",";
            printer(m[i]);
            scratch << token;
        }
    }
};
}// namespace detail

void CppCodegen::visit(const BreakStmt *stmt) noexcept {
    current_scratch() << "break";
}
void CppCodegen::visit(const ContinueStmt *stmt) noexcept {
    current_scratch() << "continue";
}
void CppCodegen::visit(const ReturnStmt *stmt) noexcept {
    current_scratch() << "return ";
    if (stmt->expression()) {
        stmt->expression()->accept(*this);
    }
}
void CppCodegen::visit(const ScopeStmt *stmt) noexcept {
    current_scratch() << "{\n";
    indent_inc();
    if (stmt->is_func_body()) {
        _emit_builtin_vars_define(current_function());
    }
    _emit_local_var_define(stmt);
    _emit_statements(stmt->statements());
    indent_dec();
    _emit_indent();
    current_scratch() << "}";
}
void CppCodegen::visit(const IfStmt *stmt) noexcept {
    current_scratch() << "if (";
    stmt->condition()->accept(*this);
    current_scratch() << ") ";
    stmt->true_branch()->accept(*this);
    auto false_branch = stmt->false_branch();
    if (false_branch->empty()) {
        return;
    }
    current_scratch() << " else ";
    if (false_branch->size() == 1 && false_branch->statements()[0]->tag() == Statement::Tag::IF) {
        false_branch->statements()[0]->accept(*this);
    } else {
        false_branch->accept(*this);
    }
}

void CppCodegen::visit(const CommentStmt *stmt) noexcept {
    current_scratch() << "/* " << stmt->string() << " */";
}

void CppCodegen::visit(const LoopStmt *stmt) noexcept {
    current_scratch() << "while (1) ";
    stmt->body()->accept(*this);
}
void CppCodegen::visit(const ExprStmt *stmt) noexcept {
    stmt->expression()->accept(*this);
}
void CppCodegen::visit(const SwitchStmt *stmt) noexcept {
    current_scratch() << "switch (";
    stmt->expression()->accept(*this);
    current_scratch() << ") ";
    stmt->body()->accept(*this);
}
void CppCodegen::visit(const SwitchCaseStmt *stmt) noexcept {
    current_scratch() << "case ";
    stmt->expression()->accept(*this);
    current_scratch() << ":";
    stmt->body()->accept(*this);
}
void CppCodegen::visit(const SwitchDefaultStmt *stmt) noexcept {
    current_scratch() << "default:";
    stmt->body()->accept(*this);
}
void CppCodegen::visit(const AssignStmt *stmt) noexcept {
    stmt->lhs()->accept(*this);
    current_scratch() << " = ";
    stmt->rhs()->accept(*this);
}

void CppCodegen::visit(const ForStmt *stmt) noexcept {
    current_scratch() << "for (;";
    stmt->condition()->accept(*this);
    current_scratch() << "; ";
    stmt->var()->accept(*this);
    current_scratch() << " += ";
    stmt->step()->accept(*this);
    current_scratch() << ")";
    stmt->body()->accept(*this);
}

void CppCodegen::visit(const PrintStmt *stmt) noexcept {
    span<const Expression *const> args = stmt->args();
    current_scratch() << "printf(";
    Scratch format_scratch("\"");
    format_scratch << stmt->fmt() << "\\n\"";
    Scratch args_scratch;

    for (const Expression *expr : args) {
        switch (expr->type()->tag()) {
            case Type::Tag::USHORT:
            case Type::Tag::UINT: {
                format_scratch.replace("{}", "%u");
                break;
            }
            case Type::Tag::BOOL: {
                format_scratch.replace("{}", "%d");
                break;
            }
            case Type::Tag::FLOAT: {
                format_scratch.replace("{}", "%f");
                break;
            }
            case Type::Tag::SHORT:
            case Type::Tag::INT: {
                format_scratch.replace("{}", "%d");
                break;
            }
            case Type::Tag::UCHAR: {
                format_scratch.replace("{}", "%u");
                break;
            }
            case Type::Tag::CHAR: {
                format_scratch.replace("{}", "%d");
                break;
            }
            default: break;
        }
        SCRATCH_GUARD(args_scratch);
        current_scratch() << ",";
        expr->accept(*this);
    }
    current_scratch() << format_scratch
                      << args_scratch
                      << ")";
}

void CppCodegen::visit(const UnaryExpr *expr) noexcept {
    switch (expr->op()) {
        case UnaryOp::POSITIVE: current_scratch() << "+"; break;
        case UnaryOp::NEGATIVE: current_scratch() << "-"; break;
        case UnaryOp::NOT: current_scratch() << "!"; break;
        case UnaryOp::BIT_NOT: current_scratch() << "~"; break;
    }
    current_scratch() << "(";
    expr->operand()->accept(*this);
    current_scratch() << ")";
}
void CppCodegen::visit(const BinaryExpr *expr) noexcept {
    current_scratch() << "(";
    expr->lhs()->accept(*this);
    switch (expr->op()) {
        case BinaryOp::ADD: current_scratch() << " + "; break;
        case BinaryOp::SUB: current_scratch() << " - "; break;
        case BinaryOp::MUL: current_scratch() << " * "; break;
        case BinaryOp::DIV: current_scratch() << " / "; break;
        case BinaryOp::MOD: current_scratch() << " % "; break;
        case BinaryOp::BIT_AND: current_scratch() << " & "; break;
        case BinaryOp::BIT_OR: current_scratch() << " | "; break;
        case BinaryOp::BIT_XOR: current_scratch() << " ^ "; break;
        case BinaryOp::SHL: current_scratch() << " << "; break;
        case BinaryOp::SHR: current_scratch() << " >> "; break;
        case BinaryOp::AND: current_scratch() << " && "; break;
        case BinaryOp::OR: current_scratch() << " || "; break;
        case BinaryOp::LESS: current_scratch() << " < "; break;
        case BinaryOp::GREATER: current_scratch() << " > "; break;
        case BinaryOp::LESS_EQUAL: current_scratch() << " <= "; break;
        case BinaryOp::GREATER_EQUAL: current_scratch() << " >= "; break;
        case BinaryOp::EQUAL: current_scratch() << " == "; break;
        case BinaryOp::NOT_EQUAL: current_scratch() << " != "; break;
    }
    expr->rhs()->accept(*this);
    current_scratch() << ")";
}
void CppCodegen::visit(const MemberExpr *expr) noexcept {
    expr->parent()->accept(*this);
    if (expr->is_swizzle()) {
        static constexpr std::string_view xyzw[] = {"x", "y", "z", "w"};
        current_scratch() << ".";
        for (int i = 0; i < expr->swizzle_size(); ++i) {
            current_scratch() << xyzw[expr->swizzle_index(i)];
        }
    } else {
        current_scratch() << ".";
        _emit_member_name(expr->member_index());
    }
}
void CppCodegen::visit(const SubscriptExpr *expr) noexcept {
    const Type *type = expr->range()->type();
    if (type->has_multi_dims()) {
        current_scratch() << "(*reinterpret_cast<";
        _emit_type_name(type->element());
        current_scratch() << "(*)";
        for (int d : type->dims()) {
            current_scratch() << "[" << d << "]";
        }
        current_scratch() << ">(";
        expr->range()->accept(*this);
        current_scratch() << "))";
    } else {
        expr->range()->accept(*this);
    }
    expr->for_each_index([&](const Expression *index) {
        current_scratch() << "[";
        index->accept(*this);
        current_scratch() << "]";
    });
}
void CppCodegen::visit(const LiteralExpr *expr) noexcept {
    ocarina::visit(
        detail::LiteralPrinter(current_scratch()),
        expr->value());
}
void CppCodegen::visit(const RefExpr *expr) noexcept {
    _emit_variable_name(expr->variable());
}
void CppCodegen::visit(const CallExpr *expr) noexcept {
    switch (expr->call_op()) {
        case CallOp::CUSTOM: {
            _emit_func_name(*expr->function());
            current_scratch() << "(";
            for (const auto &arg : expr->arguments()) {
                arg->accept(*this);
                current_scratch() << ",";
            }
            if (!expr->arguments().empty()) {
                current_scratch().pop_back();
            }
            current_scratch() << ")";
            break;
        }
    }
}
void CppCodegen::visit(const CastExpr *expr) noexcept {
    switch (expr->cast_op()) {
        case CastOp::STATIC: current_scratch() << "static_cast<"; break;
        case CastOp::BITWISE: current_scratch() << "bit_cast<"; break;
    }
    _emit_type_name(expr->type());
    current_scratch() << ">(";
    expr->expression()->accept(*this);
    current_scratch() << ")";
}

void CppCodegen::visit(const Type *type) noexcept {
    if (!type->is_structure() || has_generated(type)) { return; }
    current_scratch() << "struct ";
    current_scratch() << "alignas(";
    current_scratch() << type->alignment();
    current_scratch() << ") ";
    _emit_struct_name(type);
    current_scratch() << " {\n";
    indent_inc();
    for (int i = 0; i < type->members().size(); ++i) {
        const Type *member = type->members()[i];
        _emit_indent();
        _emit_type_name(member);
        _emit_space();
        _emit_member_name(i);
        current_scratch() << "{};\n";
    }
    indent_dec();
    current_scratch() << "};\n";

    add_generated(type);
}

void CppCodegen::_emit_types_define() noexcept {
    Type::for_each(this);
}

bool CppCodegen::has_generated(const Type *type) const noexcept {
    return _generated_struct.contains(type);
}

void CppCodegen::add_generated(const Type *type) noexcept {
    _generated_struct.emplace(type);
}

bool CppCodegen::has_generated(const Function *func) const noexcept {
    return _generated_func.contains(func);
}

void CppCodegen::add_generated(const Function *func) noexcept {
    _generated_func.emplace(func);
}

void CppCodegen::_emit_uniform_var(const ArgumentBinding &uniform) noexcept {
}

void CppCodegen::_emit_variable_define(const Variable &v) noexcept {
    if (v.type()->is_buffer()) {
        _emit_type_name(v.type());
        _emit_space();
        _emit_variable_name(v);
    } else {
        _emit_type_name(v.type());
        _emit_space();
        switch (v.tag()) {
            case Variable::Tag::REFERENCE: current_scratch() << "&"; break;
            default: break;
        }
        _emit_variable_name(v);
    }
}

void CppCodegen::_emit_local_var_define(const ScopeStmt *scope) noexcept {
    for (const auto &var : scope->local_vars()) {
        _emit_indent();
        _emit_variable_define(var);
        current_scratch() << "{};\n";
    }
}

void CppCodegen::_emit_builtin_vars_define(const Function &f) noexcept {
    for (const Variable &var : f.builtin_vars()) {
        _emit_indent();
        _emit_builtin_var(var);
        current_scratch() << ";\n";
    }
}

void CppCodegen::_emit_type_name(const Type *type) noexcept {
    if (type == nullptr) {
        current_scratch() << "void";
    } else {
        switch (type->tag()) {
            case Type::Tag::BOOL: current_scratch() << "bool"; break;
            case Type::Tag::FLOAT: current_scratch() << "float"; break;
            case Type::Tag::INT: current_scratch() << "int"; break;
            case Type::Tag::UINT: current_scratch() << "uint"; break;
            case Type::Tag::UCHAR: current_scratch() << "uchar"; break;
            case Type::Tag::CHAR: current_scratch() << "char"; break;
            case Type::Tag::VECTOR:
                _emit_type_name(type->element());
                current_scratch() << type->dimension();
                break;
            case Type::Tag::ARRAY:
                _emit_type_name(type->element());
                current_scratch() << "[";
                current_scratch() << type->dimension();
                current_scratch() << "]";
                break;
            case Type::Tag::MATRIX: {
                auto d = type->dimension();
                current_scratch() << "float" << d << "x" << d;
                break;
            }
            case Type::Tag::STRUCTURE:
                _emit_struct_name(type);
                break;
            case Type::Tag::BUFFER:
                _emit_type_name(type->element());
                current_scratch() << "*";
                break;
            case Type::Tag::TEXTURE: break;
            case Type::Tag::RESOURCE_ARRAY: break;
            case Type::Tag::ACCEL: break;
            case Type::Tag::NONE: break;
        }
    }
}
void CppCodegen::_emit_function(const Function &f) noexcept {
    if (has_generated(&f)) {
        return;
    }
    _emit_type_name(f.return_type());
    _emit_space();
    _emit_func_name(f);
    _emit_arguments(f);
    _emit_body(f);
    add_generated(&f);
}
void CppCodegen::_emit_variable_name(Variable v) noexcept {
    current_scratch() << v.name();
}
void CppCodegen::_emit_statements(ocarina::span<const Statement *const> stmts) noexcept {
    for (const Statement *stmt : stmts) {
        _emit_indent();
        stmt->accept(*this);
        current_scratch() << ";";
        _emit_newline();
    }
}
void CppCodegen::_emit_body(const Function &f) noexcept {
    f.body()->accept(*this);
}
void CppCodegen::_emit_arguments(const Function &f) noexcept {
    current_scratch() << "(";
    for (const auto &v : f.arguments()) {
        _emit_variable_define(v);
        current_scratch() << ",";
    }
    for (const auto &uniform : f.uniform_vars()) {
        _emit_variable_define(uniform.expression()->variable());
        current_scratch() << ",";
    }
    if (f.is_kernel() && !f.is_raytracing()) {
        Variable dispatch_dim(Type::of<uint3>(), Variable::Tag::LOCAL, -1, "d_dim");
        _emit_variable_define(dispatch_dim);
    } else {
        if (f.arguments().size() + f.uniform_vars().size() > 0) {
            current_scratch().pop_back();
        }
    }
    current_scratch() << ")";
}
void CppCodegen::emit(const Function &func) noexcept {
    FUNCTION_GUARD(func)
    if (!func.description().empty()) {
        current_scratch() << "/* " << func.description() << " */\n";
    }
    TIMER_TAG(codegen, "function " + func.func_name() + " generated");
    func.for_each_structure([&](const Type *type) {
        visit(type);
    });
    if (func.is_raytracing_kernel()) {
        _emit_raytracing_param(func);
    }
    func.for_each_custom_func([&](const Function *f) {
        emit(*f);
    });
    _emit_function(func);
    _emit_newline();
}
}// namespace ocarina