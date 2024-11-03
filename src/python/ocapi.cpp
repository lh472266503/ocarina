//
// Created by ling.zhu on 2024/10/28.
//

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "core/stl.h"

namespace py = pybind11;
using namespace ocarina;

struct PythonExporter {
    py::module module;
};

void export_math(py::module &m);
void export_ast(py::module &m);
void export_rhi(py::module &m);

PYBIND11_MODULE(ocapi, m) {
    PythonExporter python_exporter;
    python_exporter.module = m;
    export_ast(python_exporter.module);
    export_math(python_exporter.module);
    export_rhi(python_exporter.module);
}