//
// Created by ling.zhu on 2024/10/28.
//

#include "ocapi.h"

void export_math(PythonExporter &exporter);
void export_ast(PythonExporter &exporter);
void export_rhi(PythonExporter &exporter);
void export_rtx(PythonExporter &exporter);
void export_window(PythonExporter &exporter);
void export_image(PythonExporter &exporter);

Context *Context::s_context = nullptr;

Context &Context::instance() noexcept {
    if (s_context == nullptr) {
        s_context = new Context{};
    }
    return *s_context;
}

void Context::destroy_instance() noexcept {
    delete s_context;
    s_context = nullptr;
}

void export_base_type(PythonExporter &exporter) {
    auto &m = exporter.module;
    py::class_<concepts::Noncopyable>(m, "concepts_Noncopyable");
    py::class_<RTTI>(m, "RTTI");
    py::class_<Hashable, RTTI>(m, "Hashable");
}

PYBIND11_MODULE(ocapi, m) {
    auto &context = Context::instance();
    PythonExporter python_exporter;
    python_exporter.module = m;
    export_base_type(python_exporter);
    export_image(python_exporter);
    export_ast(python_exporter);
    export_rhi(python_exporter);
    export_math(python_exporter);
    export_window(python_exporter);
    export_rtx(python_exporter);
}