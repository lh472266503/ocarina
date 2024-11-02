//
// Created by ling.zhu on 2024/10/31.
//

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "math/base.h"

namespace py = pybind11;
using namespace ocarina;

void export_base_type(py::module &m) {
    py::class_<RTTI>(m, "RTTI");
    py::class_<Hashable, RTTI>(m, "Hashable");
}

void export_type(py::module &m) {
    export_base_type(m);

}