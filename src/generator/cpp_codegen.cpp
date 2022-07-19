//
// Created by Zero on 08/06/2022.
//

#include "cpp_codegen.h"

namespace ocarina {

namespace detail {
struct LiteralPrinter {
    using Scratch = Codegen::Scratch;
    Scratch &scratch;
    explicit LiteralPrinter(Scratch &scratch) : scratch(scratch) {}
    template<typename T>
    void operator()(T v) {
        if constexpr (ocarina::is_scalar_v<T>) {
            scratch << v;
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
    current_scratch() << "// " << stmt->string();
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
    format_scratch << stmt->fmt() << "\"";
    Scratch args_scratch;

    for (int i = 0; i < args.size(); ++i) {
        const Expression *expr = args[i];
        switch (expr->type()->tag()) {
            case Type::Tag::UINT: {
                format_scratch.replace(i, "{}", "%u");
                break;
            }
            case Type::Tag::BOOL: {
                format_scratch.replace(i, "{}", "%d");
                break;
            }
            case Type::Tag::FLOAT: {
                format_scratch.replace(i, "{}", "%f");
                break;
            }
            case Type::Tag::INT: {
                format_scratch.replace(0, "{}", "%d");
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
void CppCodegen::visit(const AccessExpr *expr) noexcept {
    expr->range()->accept(*this);
    current_scratch() << "[";
    expr->index()->accept(*this);
    current_scratch() << "]";
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
    _emit_func_name(expr->function()->hash());
    current_scratch() << "(";
    for (const auto &arg : expr->arguments()) {
        arg->accept(*this);
        current_scratch() << ",";
    }
    if (!expr->arguments().empty()) {
        current_scratch().pop_back();
    }
    current_scratch() << ")";
}
void CppCodegen::visit(const CastExpr *expr) noexcept {
    switch (expr->cast_op()) {
        case CastOp::STATIC: current_scratch() << "static_cast<"; break;
        case CastOp::BITWISE: current_scratch() << "reinterpret_cast<"; break;
    }
    _emit_type_name(expr->type());
    current_scratch() << ">(";
    expr->expression()->accept(*this);
    current_scratch() << ")";
}
void CppCodegen::visit(const Type *type) noexcept {
    if (!type->is_structure() || type->has_defined()) { return; }
    current_scratch() << "struct ";
    current_scratch() << "alignas(";
    current_scratch() << type->alignment();
    current_scratch() << ") ";
    _emit_struct_name(type->hash());
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

    type->define();
}

void CppCodegen::_emit_types_define() noexcept {
    Type::for_each(this);
}

void CppCodegen::_emit_variable_define(Variable v) noexcept {
    if (!v.type()->is_array()) {
        _emit_type_name(v.type());
        _emit_space();
        switch (v.tag()) {
            case Variable::Tag::REFERENCE: current_scratch() << "&"; break;
            default: break;
        }
        _emit_variable_name(v);
    } else {
        _emit_type_name(v.type()->element());
        _emit_space();
        _emit_variable_name(v);
        current_scratch() << "[";
        current_scratch() << v.type()->dimension();
        current_scratch() << "]";
    }
}

void CppCodegen::_emit_local_var_define(const ScopeStmt *scope) noexcept {
    for (const auto &var : scope->local_vars()) {
        _emit_indent();
        _emit_variable_define(var);
        current_scratch() << "{};\n";
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
                _emit_struct_name(type->hash());
                break;
            case Type::Tag::BUFFER: break;
            case Type::Tag::TEXTURE: break;
            case Type::Tag::BINDLESS_ARRAY: break;
            case Type::Tag::ACCEL: break;
            case Type::Tag::NONE: break;
        }
    }
}
void CppCodegen::_emit_function(const Function &f) noexcept {
    if (f.has_defined()) {
        return;
    }
    _emit_type_name(f.return_type());
    _emit_space();
    _emit_func_name(f.hash());
    _emit_arguments(f);
    _emit_body(f);
    f.define();
}
void CppCodegen::_emit_variable_name(Variable v) noexcept {
    current_scratch() << "v" << v.uid();
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
    if (!f.arguments().empty()) {
        current_scratch().pop_back();
    }
    current_scratch() << ")";
}
void CppCodegen::emit(const Function &func) noexcept {
    for (const auto &f : func.used_custom_func()) {
        emit(*f);
    }
    _emit_types_define();
    _emit_function(func);
    _emit_newline();
}
}// namespace ocarina