//
// Created by ling.zhu on 2024/10/31.
//

#include "pyexporter/ocapi.h"
#include "math/base.h"

namespace py = pybind11;
using namespace ocarina;


void export_type(PythonExporter &exporter) {
    auto &m = exporter.module;
    auto m_type = py::class_<Type, Hashable, concepts::Noncopyable>(m, "Type");
    m_type.def_static("from_desc", [](const string &desc) { return Type::from(desc); }, py::return_value_policy::reference);
    m_type.def("description", [](const Type &self) {
        return self.description();
    });
    m_type.def("name", [](const Type &self) {
        return self.name();
    });
}