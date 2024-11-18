//
// Created by ling.zhu on 2024/11/7.
//

#include "pyexporter/ocapi.h"

namespace py = pybind11;
using namespace ocarina;

void export_rtx(PythonExporter &exporter) {
    export_struct<TriangleHit>(exporter)
        .def("is_miss", &TriangleHit::is_miss);
    export_struct<Ray>(exporter)
        .def(py::init<float3, float3, float>())
        .def(py::init<float3, float3>())
        .def_property("origin", &Ray::origin,
                      [](Ray &self, const float3 &val) { self.org_min.xyz() = val; })
        .def_property("direction", &Ray::direction,
                      [](Ray &self, const float3 &val) { self.dir_max.xyz() = val; })
        .def("at", &Ray::at)
        .def("t_min", &Ray::t_min)
        .def("t_max", &Ray::t_max);
}
