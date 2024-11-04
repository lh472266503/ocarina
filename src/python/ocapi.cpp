//
// Created by ling.zhu on 2024/10/28.
//


#include "exporter.h"

void export_math(PythonExporter &exporter);
void export_ast(PythonExporter &exporter);
void export_rhi(PythonExporter &exporter);

PYBIND11_MODULE(ocapi, m) {
    PythonExporter python_exporter;
    python_exporter.module = m;
    export_ast(python_exporter);
    export_rhi(python_exporter);
    export_math(python_exporter);
}