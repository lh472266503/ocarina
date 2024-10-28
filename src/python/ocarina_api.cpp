//
// Created by ling.zhu on 2024/10/28.
//


#include "export_vector.h"

namespace py = pybind11;
using namespace ocarina;

void export_vector2(py::module &m);
void export_vector3(py::module &m);
void export_vector4(py::module &m);

PYBIND11_MODULE(ocarina_api, m) {
    export_vector2(m);
    export_vector3(m);
    export_vector4(m);
}