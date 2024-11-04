//
// Created by Zero on 2024/10/31.
//

#pragma once

#include "python/exporter.h"
#include "ext/pybind11/include/pybind11/operators.h"
#include "ast/type_desc.h"
#include "rhi/resources/buffer.h"

namespace py = pybind11;
using namespace ocarina;

template<typename T>
void export_buffer(PythonExporter &exporter) {
    string buffer_name = ocarina::format("buffer_{}", string(TypeDesc<T>::name()));
    auto m_buffer = py::class_<Buffer<T>, RHIResource>(exporter.module, buffer_name.c_str());
    m_buffer.def("size", [](const Buffer<T> &self) {
        return self.size();
    });

    exporter.m_device->def(("create_" + buffer_name).c_str(), [](const Device &self, uint size) {
        return self.create_buffer<T>(size);
    }, py::return_value_policy::move);
}

template<typename T, typename... Base>
requires(is_basic_v<T> || is_struct_v<T>)
auto export_pod_type(PythonExporter &exporter, const char *name) {
    auto &m = exporter.module;
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
    export_buffer<T>(exporter);
    return mt;
}