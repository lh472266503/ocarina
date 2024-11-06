//
// Created by Zero on 2024/11/3.
//

#pragma once

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "ext/pybind11/include/pybind11/operators.h"
#include "core/stl.h"
#include "rhi/common.h"
#include "ast/type_desc.h"
#include "numpy.h"

namespace py = pybind11;
using namespace ocarina;

struct PythonExporter {
    py::module module;
    UP<py::class_<Device, concepts::Noncopyable>> m_device;
};

using ret_policy = py::return_value_policy;

struct Context {
    UP<Device> device;
    UP<Stream> stream;
    [[nodiscard]] static Context &instance() noexcept;
    Context() {
        OC_INFO("ocapi load!");
    }
    ~Context() {
        OC_INFO("ocapi unload!");
    }
};

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
