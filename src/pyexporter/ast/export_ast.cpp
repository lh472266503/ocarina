//
// Created by Zero on 2024/11/3.
//

#include "pyexporter/ocapi.h"
#include "math/base.h"

namespace py = pybind11;
using namespace ocarina;

void export_type(PythonExporter &exporter);
void export_expressions(PythonExporter &exporter);

void export_ast(PythonExporter &exporter) {
    export_type(exporter);
    export_expressions(exporter);
}