//
// Created by Zero on 2024/11/8.
//

#include "pyexporter/ocapi.h"
#include "math/base.h"

namespace py = pybind11;
using namespace ocarina;

void export_expressions(PythonExporter &exporter) {
    py::class_<ASTNode>(exporter.module, "ASTNode");
    py::class_<Expression, ASTNode, concepts::Noncopyable, Hashable>(exporter.module, "Expression");

    using ExpressionTag = Expression::Tag;
    OC_EXPORT_ENUM(exporter.module, ExpressionTag, UNARY,
                   BINARY, MEMBER, SUBSCRIPT,
                   LITERAL, REF, CONSTANT, CALL,
                   CAST, CONDITIONAL);
    {
        auto mt = py::class_<UnaryExpr, Expression>(exporter.module, "UnaryExpr");
    }
}