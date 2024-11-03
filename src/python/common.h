//
// Created by Zero on 2024/10/31.
//

#pragma once

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "ext/pybind11/include/pybind11/operators.h"
#include "core/stl.h"
#include "math/basic_types.h"
#include "math/base.h"
#include "core/string_util.h"
#include "ast/type_registry.h"

namespace py = pybind11;
using namespace ocarina;

template<typename T, typename... Base>
requires (is_basic_v<T> || is_struct_v<T>)
auto export_type(py::module &m, const char *name) {
    auto mt = py::class_<T, Base...>(m, name);
    mt.def_static("desc", []() {
        return TypeDesc<T>::description();
    });
    mt.def("clone", [](const T &self) {
        return T{self};
    });
    mt.def_static("sizeof", []() {
        return sizeof(T);
    });
    mt.def_static("alignof", []() {
        return alignof(T);
    });
    mt.def(py::init<>());
    return mt;
}