//
// Created by Zero on 2024/11/3.
//

#include "python/exporter.h"
#include "math/basic_types.h"

namespace py = pybind11;
using namespace ocarina;

void export_resource(PythonExporter &exporter);

void export_device(PythonExporter &exporter);

void export_rhi(PythonExporter &exporter) {
    export_resource(exporter);
    export_device(exporter);
}