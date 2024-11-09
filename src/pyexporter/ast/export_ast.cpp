//
// Created by Zero on 2024/11/3.
//

#include "pyexporter/ocapi.h"
#include "math/base.h"

namespace py = pybind11;
using namespace ocarina;

void export_type(PythonExporter &exporter);
void export_expressions(PythonExporter &exporter);

void export_function(PythonExporter &exporter) {
    using FunctionTag = Function::Tag;
    OC_EXPORT_ENUM(exporter.module, FunctionTag, KERNEL, CALLABLE)

    auto mt = py::class_<Function, Hashable, concepts::Noncopyable>(exporter.module, "Function");
    mt.def_static("push", [](FunctionTag tag) {
        auto ret = make_shared<Function>(tag);
        Function::push(ret);
        return ret;
    });

    mt.def_static("pop", [] {
        auto ret = Function::current();
        Function::pop(nullptr);
        return ret;
    });
}

void export_ast(PythonExporter &exporter) {
    export_type(exporter);
    export_expressions(exporter);
    export_function(exporter);
}