//
// Created by ling.zhu on 2024/10/28.
//


#include "export_vector.h"

namespace py = pybind11;
using namespace ocarina;

void export_vector2(py::module &m);
void export_vector3(py::module &m);
void export_vector4(py::module &m);

PYBIND11_MODULE(ocapi, m) {

    m.doc() = "pybind11 example plugin"; // 可选的模块文档字符串

    // 绑定函数或类
    m.def("add", [](int a, int b) { return a + b; }, "A function that adds two numbers");
    m.def("sub", [](int a, int b) { return a - b;}, "func");
//    export_vector2(m);
//    export_vector3(m);
//    export_vector4(m);
}