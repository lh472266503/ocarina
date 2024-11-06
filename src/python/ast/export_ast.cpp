//
// Created by Zero on 2024/11/3.
//

#include "python/ocapi.h"
#include "math/base.h"

namespace py = pybind11;
using namespace ocarina;

void export_type(PythonExporter &exporter);
void export_base_type(PythonExporter &exporter);

void export_ast(PythonExporter &exporter) {
    export_base_type(exporter);
    export_type(exporter);
}