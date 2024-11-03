//
// Created by Zero on 2024/11/3.
//

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "math/basic_types.h"

namespace py = pybind11;
using namespace ocarina;

void export_resource(py::module &m);

void export_device(py::module &m);

void export_rhi(py::module &m) {
    export_resource(m);
    export_device(m);
}