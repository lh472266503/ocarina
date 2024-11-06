//
// Created by ling.zhu on 2024/10/28.
//

#include "ocapi.h"

void export_math(PythonExporter &exporter);
void export_ast(PythonExporter &exporter);
void export_rhi(PythonExporter &exporter);

Context &Context::instance() noexcept {
    static Context context;
    return context;
}

PYBIND11_MODULE(ocapi, m) {
    auto &context = Context::instance();
    PythonExporter python_exporter;
    python_exporter.module = m;
    export_ast(python_exporter);
    export_rhi(python_exporter);
    export_math(python_exporter);
    export_struct<TriangleHit>(python_exporter);
}