//
// Created by ling.zhu on 2024/10/28.
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include "core/stl.h"
#include "math/basic_types.h"
#include "core/string_util.h"

namespace py = pybind11;
using namespace ocarina;

#define OC_EXPORT_ARITHMETIC_OP(type, dim)                                                                                     \
    m = m.def("__add__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a + b; }, py::is_operator());     \
    m = m.def("__add__", [](const Vector<type, dim> &a, const type &b) { return a + b; }, py::is_operator());                  \
    m = m.def("__radd__", [](const Vector<type, dim> &a, const type &b) { return a + b; }, py::is_operator());                 \
    m = m.def("__sub__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a - b; }, py::is_operator());     \
    m = m.def("__sub__", [](const Vector<type, dim> &a, const type &b) { return a - b; }, py::is_operator());                  \
    m = m.def("__rsub__", [](const Vector<type, dim> &a, const type &b) { return b - a; }, py::is_operator());                 \
    m = m.def("__mul__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a * b; }, py::is_operator());     \
    m = m.def("__mul__", [](const Vector<type, dim> &a, const type &b) { return a * b; }, py::is_operator());                  \
    m = m.def("__rmul__", [](const Vector<type, dim> &a, const type &b) { return a * b; }, py::is_operator());                 \
    m = m.def("__truediv__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a / b; }, py::is_operator()); \
    m = m.def("__truediv__", [](const Vector<type, dim> &a, const type &b) { return a / b; }, py::is_operator());              \
    m = m.def("__rtruediv__", [](const Vector<type, dim> &a, const type &b) { return b / a; }, py::is_operator());             \
    m = m.def("__gt__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a > b; }, py::is_operator());      \
    m = m.def("__ge__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a >= b; }, py::is_operator());     \
    m = m.def("__lt__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a < b; }, py::is_operator());      \
    m = m.def("__le__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a <= b; }, py::is_operator());     \
    m = m.def("__eq__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a == b; }, py::is_operator());     \
    m = m.def("__ne__", [](const Vector<type, dim> &a, const Vector<type, dim> &b) { return a != b; }, py::is_operator());\
