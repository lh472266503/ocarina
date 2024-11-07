//
// Created by ling.zhu on 2024/11/7.
//

#include "python/ocapi.h"

namespace py = pybind11;
using namespace ocarina;

void export_rtx(PythonExporter &exporter) {
    export_struct<TriangleHit>(exporter);
    export_struct<Ray>(exporter)
        .def(py::init<float3, float3, float>())
        .def(py::init<float3, float3>())
        .def_property_readonly("origin", &Ray::origin)
        .def_property_readonly("direction", &Ray::direction);
}
