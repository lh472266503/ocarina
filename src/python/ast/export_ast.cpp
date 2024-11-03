//
// Created by Zero on 2024/11/3.
//

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "math/base.h"

namespace py = pybind11;
using namespace ocarina;

void export_type(py::module &m);
void export_base_type(py::module &m);

void export_ast(py::module &m) {
    export_base_type(m);
    export_type(m);
}