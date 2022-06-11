//
// Created by Zero on 08/06/2022.
//

#include "cpp_codegen.h"

namespace ocarina {

void CppCodegen::visit(const BreakStmt *) noexcept {
}
void CppCodegen::visit(const ContinueStmt *) noexcept {
}
void CppCodegen::visit(const ReturnStmt *) noexcept {
}
void CppCodegen::visit(const ScopeStmt *) noexcept {
}
void CppCodegen::visit(const IfStmt *) noexcept {
}
void CppCodegen::visit(const LoopStmt *) noexcept {
}
void CppCodegen::visit(const ExprStmt *) noexcept {
}
void CppCodegen::visit(const SwitchStmt *) noexcept {
}
void CppCodegen::visit(const SwitchCaseStmt *) noexcept {
}
void CppCodegen::visit(const SwitchDefaultStmt *) noexcept {
}
void CppCodegen::visit(const AssignStmt *) noexcept {
}
void CppCodegen::visit(const ForStmt *) noexcept {
}
void CppCodegen::visit(const UnaryExpr *) noexcept {
}
void CppCodegen::visit(const BinaryExpr *) noexcept {
}
void CppCodegen::visit(const MemberExpr *) noexcept {
}
void CppCodegen::visit(const AccessExpr *) noexcept {
}
void CppCodegen::visit(const LiteralExpr *) noexcept {
}
void CppCodegen::visit(const RefExpr *) noexcept {
}
void CppCodegen::visit(const ConstantExpr *) noexcept {
}
void CppCodegen::visit(const CallExpr *) noexcept {
}
void CppCodegen::visit(const CastExpr *) noexcept {
}
void CppCodegen::visit(const Type *) noexcept {
}
void CppCodegen::_emit_type_decl() noexcept {
    Type::for_each(this);
}
void CppCodegen::_emit_variable_decl(Variable v) noexcept {
}
void CppCodegen::_emit_type_name(const Type *type) noexcept {
}
void CppCodegen::_emit_function(Function f) noexcept {
}
void CppCodegen::_emit_variable_name(Variable v) noexcept {
}
void CppCodegen::_emit_indent() noexcept {
}
void CppCodegen::_emit_statements(ocarina::span<const Statement *const> stmts) noexcept {
}
void CppCodegen::emit(Function func) noexcept {
    _emit_type_decl();
    _emit_function(func);
}
}