//
// Created by Zero on 2024/10/27.
//

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(test_pybind, m) {
    m.doc() = "pybind11 example plugin"; // 可选的模块文档字符串

    // 绑定函数或类
    m.def("add", [](int a, int b) { return a + b; }, "A function that adds two numbers");
}