//
// Created by Zero on 2024/10/31.
//

#pragma once

#include "python/ocapi.h"
#include "ext/pybind11/include/pybind11/operators.h"
#include "ast/type_desc.h"
#include "rhi/common.h"
#include "numpy.h"

namespace py = pybind11;
using namespace ocarina;

template<typename T>
void export_buffer(PythonExporter &exporter) {
    string buffer_name = ocarina::format("buffer_{}", string(TypeDesc<T>::name()));
    auto m_buffer = py::class_<Buffer<T>, RHIResource>(exporter.module, buffer_name.c_str());
    m_buffer.def_static("create", [](uint size) { return Context::instance().device->create_buffer<T>(size); }, ret_policy::move);
    m_buffer.def("size", [](const Buffer<T> &self) { return self.size(); });
    m_buffer.def("upload", [](const Buffer<T> &self, const vector<T> &lst) { self.upload_immediately(lst.data()); });
    m_buffer.def("download", [](const Buffer<T> &self, py::buffer &lst) { self.download_immediately(lst.request().ptr); });
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
    if constexpr (!std::is_same_v<vector_element_t<T>, bool>) {
        export_buffer<T>(exporter);
    }
    return mt;
}