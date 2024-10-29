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

template<typename T, size_t N, typename M>
void export_vector_op(M &m) {
    if constexpr (ocarina::is_number_v<T>) {
        m = m.def("__add__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a + b; }, py::is_operator());
        m = m.def("__add__", [](const Vector<T, N> &a, const T &b) { return a + b; }, py::is_operator());
        m = m.def("__radd__", [](const Vector<T, N> &a, const T &b) { return a + b; }, py::is_operator());
        m = m.def("__sub__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a - b; }, py::is_operator());
        m = m.def("__sub__", [](const Vector<T, N> &a, const T &b) { return a - b; }, py::is_operator());
        m = m.def("__rsub__", [](const Vector<T, N> &a, const T &b) { return b - a; }, py::is_operator());
        m = m.def("__mul__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a * b; }, py::is_operator());
        m = m.def("__mul__", [](const Vector<T, N> &a, const T &b) { return a * b; }, py::is_operator());
        m = m.def("__rmul__", [](const Vector<T, N> &a, const T &b) { return a * b; }, py::is_operator());
        m = m.def("__truediv__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a / b; }, py::is_operator());
        m = m.def("__truediv__", [](const Vector<T, N> &a, const T &b) { return a / b; }, py::is_operator());
        m = m.def("__rtruediv__", [](const Vector<T, N> &a, const T &b) { return b / a; }, py::is_operator());
        m = m.def("__gt__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a > b; }, py::is_operator());
        m = m.def("__ge__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a >= b; }, py::is_operator());
        m = m.def("__lt__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a < b; }, py::is_operator());
        m = m.def("__le__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a <= b; }, py::is_operator());
    }
    if constexpr (ocarina::is_integral_v<T>) {
        m = m.def("__mod__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a % b; }, py::is_operator());
        m = m.def("__mod__", [](const Vector<T, N> &a, const T &b) { return a % b; }, py::is_operator());
        m = m.def("__rmod__", [](const Vector<T, N> &a, const T &b) { return b % a; }, py::is_operator());
        m = m.def("__shl__", [](const Vector<T, N> &a, const T &b) { return a << b; }, py::is_operator());
        m = m.def("__shr__", [](const Vector<T, N> &a, const T &b) { return a >> b; }, py::is_operator());
        m = m.def("__xor__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a ^ b; }, py::is_operator());
        m = m.def("__and__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a & b; }, py::is_operator());
        m = m.def("__or__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a | b; }, py::is_operator());
    }
    m = m.def("__eq__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a == b; }, py::is_operator());
    m = m.def("__ne__", [](const Vector<T, N> &a, const Vector<T, N> &b) { return a != b; }, py::is_operator());
}