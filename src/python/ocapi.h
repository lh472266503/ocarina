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

namespace ocarina::python {
template<typename T>
class Array : public vector<T> {
public:
    using Super = vector<T>;
    using Super::Super;
};
}// namespace ocarina::python

template<typename T>
void export_array(PythonExporter &exporter, const char *name = nullptr) {
    using array_t = python::Array<T>;
    name = name ? name : TypeDesc<T>::name().data();
    static string class_name = ocarina::format("Array{}", name);
    auto mt = py::class_<array_t>(exporter.module, class_name.c_str());
    mt.def_static("from_list", [](const py::list &lst) {
        array_t arr;
        arr.reserve(lst.size());
        for (int i = 0; i < lst.size(); ++i) {
            py::object item = lst[i];
            arr.push_back(item.template cast<T>());
        }
        return arr;
    });
    mt.def(py::init<>());
    mt.def("push_back", [](array_t &self, const T &t) { self.push_back(t); });
    mt.def("pop_back", [](array_t &self) { self.pop_back(); });
    mt.def("size", [](array_t &self) { return self.size(); });
    mt.def("__getitem__", [](array_t &self, size_t i) { return self[i]; });
    mt.def("__setitem__", [](array_t &self, size_t i, const T &t) { self[i] = t; });
    mt.def("resize", [](array_t &self, const uint &t) { self.resize(t); });
    mt.def("clear", [](array_t &self) { self.clear(); });
    mt.def("__repr__", [&](array_t &self) {
        string ret = class_name + "[";
        for (int i = 0; i < self.size(); ++i) {
            ret += to_str(self[i]) + ",";
        }
        return ret + "]";
    });
}

template<typename T>
void export_buffer(PythonExporter &exporter, const char *name = nullptr) {
    name = name ? name : TypeDesc<T>::name().data();
    string buffer_name = ocarina::format("Buffer{}", name);
    auto m_buffer = py::class_<Buffer<T>, RHIResource>(exporter.module, buffer_name.c_str());
    m_buffer.def_static("create", [](uint size) { return Context::instance().device->create_buffer<T>(size); }, ret_policy::move);
    m_buffer.def("size", [](const Buffer<T> &self) { return self.size(); });
    m_buffer.def("upload", [](const Buffer<T> &self, const vector<T> &lst) { self.upload_immediately(lst.data()); });
    m_buffer.def("download", [](const Buffer<T> &self, py::buffer &lst) { self.download_immediately(lst.request().ptr); });
}

template<typename T>
void export_container(PythonExporter &exporter, const char *name = nullptr) {
    export_buffer<T>(exporter, name);
    export_array<T>(exporter, name);
}

template<typename T, typename... Base>
requires(is_basic_v<T> || is_struct_v<T>)
auto export_pod_type(PythonExporter &exporter, const char *name = nullptr) {
    auto &m = exporter.module;
    name = name ? name : TypeDesc<T>::name().data();
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
    mt.def("__repr__", [](const T &self) {
        return to_str(self);
    });
    mt.def(py::init<>());
    if constexpr (!std::is_same_v<vector_element_t<T>, bool>) {
        export_container<T>(exporter, name);
    }
    return mt;
}

template<typename Elm, typename T>
decltype(auto) get_member_by_offset(T &t, uint offset) {
    return *reinterpret_cast<Elm *>(reinterpret_cast<std::byte *>(&t) + offset);
}

template<typename Elm, typename T>
decltype(auto) get_member_by_offset(const T &t, uint offset) {
    return *reinterpret_cast<const Elm *>(reinterpret_cast<const std::byte *>(&t) + offset);
}

template<typename T, typename... Base>
requires(is_struct_v<T>)
auto export_struct(PythonExporter &exporter) {
    string_view cname = TypeDesc<T>::name();
    static string_view simple_name = cname.substr(cname.find_last_of("::") + 1);
    auto mt = export_pod_type<T, Base...>(exporter, simple_name.data());
    traverse_tuple(struct_member_tuple_t<T>{}, [&]<typename Elm>(const Elm &_, uint index) {
        auto getter = [index = index](const T &self) {
            auto ofs = struct_member_tuple<T>::offset_array[index];
            return get_member_by_offset<Elm>(self, ofs);
        };
        auto setter = [index = index](T &self, const Elm &val) {
            auto ofs = struct_member_tuple<T>::offset_array[index];
            get_member_by_offset<Elm>(self, ofs) = val;
        };
        mt.def_property(struct_member_tuple<T>::members[index].data(), getter, setter);
    });
    return mt;
}