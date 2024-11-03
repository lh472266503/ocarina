//
// Created by ling.zhu on 2024/10/28.
//

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "core/stl.h"

namespace py = pybind11;
using namespace ocarina;

void export_math(py::module &m);
void export_ast(py::module &m);
void export_rhi(py::module &m);

PYBIND11_MODULE(ocapi, m) {
    export_ast(m);
    export_math(m);
    export_rhi(m);
}