//
// Created by ling.zhu on 2024/10/28.
//


#include "export_vector_func.h"

namespace py = pybind11;
using namespace ocarina;

void export_vector2(py::module &m);
void export_vector3(py::module &m);
void export_vector4(py::module &m);

PYBIND11_MODULE(ocapi, m) {

    m.def("add", [](int a, int b) { return a + b; }, "A function that adds two numbers");
    m.def("sub", [](int a, int b) { return a - b;}, "func");
    export_vector2(m);
    export_vector3(m);
    export_vector4(m);
}