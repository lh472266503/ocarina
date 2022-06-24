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
    _scratch << "break";
}
void CppCodegen::visit(const ContinueStmt *stmt) noexcept {
}
void CppCodegen::visit(const ReturnStmt *stmt) noexcept {
    _scratch << "return ";
    if (stmt->expression()) {
        stmt->expression()->accept(*this);
    }
}
void CppCodegen::visit(const ScopeStmt *stmt) noexcept {
    _scratch << "{\n";
    _indent += 1;
    _emit_local_var_decl(stmt);
    _emit_statements(stmt->statements());
    _indent -= 1;
    _emit_indent();
    _scratch << "}";
}
void CppCodegen::visit(const IfStmt *stmt) noexcept {
    _scratch << "if (";
    stmt->condition()->accept(*this);
    _scratch << ") ";
    stmt->true_branch()->accept(*this);
    if (!stmt->false_branch()->empty()) {
        _scratch << " else ";
        stmt->false_branch()->accept(*this);
    }
}

void CppCodegen::visit(const CommentStmt *stmt) noexcept {
    _scratch << "// " << stmt->string();
}

void CppCodegen::visit(const LoopStmt *stmt) noexcept {
}
void CppCodegen::visit(const ExprStmt *stmt) noexcept {
}
void CppCodegen::visit(const SwitchStmt *stmt) noexcept {
    _scratch << "switch (";
    stmt->expression()->accept(*this);
    _scratch<< ") ";
    stmt->body()->accept(*this);
}
void CppCodegen::visit(const SwitchCaseStmt *stmt) noexcept {
    _scratch << "case ";
    stmt->expression()->accept(*this);
    _scratch << ":";
    stmt->body()->accept(*this);
}
void CppCodegen::visit(const SwitchDefaultStmt *stmt) noexcept {
}
void CppCodegen::visit(const AssignStmt *stmt) noexcept {
    stmt->lhs()->accept(*this);
    _scratch << " = ";
    stmt->rhs()->accept(*this);
}
void CppCodegen::visit(const ForStmt *stmt) noexcept {
}
void CppCodegen::visit(const UnaryExpr *expr) noexcept {
}
void CppCodegen::visit(const BinaryExpr *expr) noexcept {
    _scratch << "(";
    expr->lhs()->accept(*this);
    switch (expr->op()) {
        case BinaryOp::ADD: _scratch << " + "; break;
        case BinaryOp::SUB: _scratch << " - "; break;
        case BinaryOp::MUL: _scratch << " * "; break;
        case BinaryOp::DIV: _scratch << " / "; break;
        case BinaryOp::MOD: _scratch << " % "; break;
        case BinaryOp::BIT_AND: _scratch << " & "; break;
        case BinaryOp::BIT_OR: _scratch << " | "; break;
        case BinaryOp::BIT_XOR: _scratch << " ^ "; break;
        case BinaryOp::SHL: _scratch << " << "; break;
        case BinaryOp::SHR: _scratch << " >> "; break;
        case BinaryOp::AND: _scratch << " && "; break;
        case BinaryOp::OR: _scratch << " || "; break;
        case BinaryOp::LESS: _scratch << " < "; break;
        case BinaryOp::GREATER: _scratch << " > "; break;
        case BinaryOp::LESS_EQUAL: _scratch << " <= "; break;
        case BinaryOp::GREATER_EQUAL: _scratch << " >= "; break;
        case BinaryOp::EQUAL: _scratch << " == "; break;
        case BinaryOp::NOT_EQUAL: _scratch << " != "; break;
    }
    expr->rhs()->accept(*this);
    _scratch << ")";
}
void CppCodegen::visit(const MemberExpr *expr) noexcept {
}
void CppCodegen::visit(const AccessExpr *expr) noexcept {
}
void CppCodegen::visit(const LiteralExpr *expr) noexcept {
    ocarina::visit(
        detail::LiteralPrinter(_scratch),
        expr->value());
}
void CppCodegen::visit(const RefExpr *expr) noexcept {
    _emit_variable_name(expr->variable());
}
void CppCodegen::visit(const ConstantExpr *expr) noexcept {
}
void CppCodegen::visit(const CallExpr *expr) noexcept {
}
void CppCodegen::visit(const CastExpr *expr) noexcept {
}
void CppCodegen::visit(const Type *type) noexcept {
}
void CppCodegen::_emit_type_decl() noexcept {
    Type::for_each(this);
}
void CppCodegen::_emit_variable_decl(Variable v) noexcept {
    _emit_type_name(v.type());
    _emit_variable_name(v);
}

void CppCodegen::_emit_local_var_decl(const ScopeStmt *scope) noexcept {
    for (const auto &var : scope->local_vars()) {
        _emit_indent();
        _emit_variable_decl(var);
        _scratch << ";\n";
    }
}

void CppCodegen::_emit_type_name(const Type *type) noexcept {
    if (type == nullptr) {
        _scratch << "void";
    } else {
        switch (type->tag()) {
            case Type::Tag::BOOL: _scratch << "bool"; break;
            case Type::Tag::FLOAT: _scratch << "float"; break;
            case Type::Tag::INT: _scratch << "int"; break;
            case Type::Tag::UINT: _scratch << "uint"; break;
            case Type::Tag::VECTOR:
                _emit_type_name(type->element());
                _scratch << type->dimension();
                break;
            case Type::Tag::MATRIX: break;
            case Type::Tag::ARRAY: break;
            case Type::Tag::STRUCTURE: break;
            case Type::Tag::BUFFER: break;
            case Type::Tag::TEXTURE: break;
            case Type::Tag::BINDLESS_ARRAY: break;
            case Type::Tag::ACCEL: break;
            case Type::Tag::NONE: break;
        }
    }
    _emit_space();
}
void CppCodegen::_emit_function(const Function &f) noexcept {
    if (f.is_callable()) {
        _scratch << "__device__";
    }
    _emit_space();
    _emit_type_name(f.return_type());
    _scratch << "function_" << f.hash();
    _emit_arguments(f);
    _emit_body(f);
}
void CppCodegen::_emit_variable_name(Variable v) noexcept {
    switch (v.tag()) {
        case Variable::Tag::REFERENCE: _scratch << "&"; break;
        default:break;
    }
    _scratch << "v" << v.uid();
}
void CppCodegen::_emit_statements(ocarina::span<const Statement *const> stmts) noexcept {

    for (const Statement *stmt : stmts) {
        _emit_indent();
        stmt->accept(*this);
        _scratch << ";";
        _emit_newline();
    }

}
void CppCodegen::_emit_body(const Function &f) noexcept {
    f.body()->accept(*this);
}
void CppCodegen::_emit_arguments(const Function &f) noexcept {
    _scratch << "(";
    for (const auto &v : f.arguments()) {
        _emit_variable_decl(v);
        _scratch << ",";
    }
    if (!f.arguments().empty()) {
        _scratch.pop_back();
    }
    _scratch << ")";
}
void CppCodegen::emit(const Function &func) noexcept {
    _emit_type_decl();
    _emit_function(func);
    _emit_newline();
}
}