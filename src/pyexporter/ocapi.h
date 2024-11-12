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

#define OC_EXPORT_ENUM_VALUE(v, enum_type) \
    .value(#v, enum_type::v)

#define OC_EXPORT_ENUM(m, Enum, ...) \
    py::enum_<Enum>(m, #Enum)        \
        MAP_UD(OC_EXPORT_ENUM_VALUE, Enum, ##__VA_ARGS__);

#define OC_EXPORT_STRUCT_MEMBER(member, S) \
    .def_readwrite(#member, &S::member)

#define OC_EXPORT_STRUCT(m, S, ...)      \
    py::class_<S>(m, #S).def(py::init()) \
        MAP_UD(OC_EXPORT_STRUCT_MEMBER, S, ##__VA_ARGS__);

struct PythonExporter {
    py::module module;
    UP<py::class_<Device, concepts::Noncopyable>> m_device;
};

using ret_policy = py::return_value_policy;

struct Context {
    static Context *s_context;
    UP<Device> device;
    UP<Stream> stream;
    BindlessArray bindless_array;
    [[nodiscard]] static Context &instance() noexcept;
    static void destroy_instance() noexcept;
    Context() {
        OC_INFO("ocapi load!");
    }
    ~Context() {
        OC_INFO("ocapi unload!");
    }
};

template<typename T>
class StructArray : public vector<T> {
public:
    using Super = vector<T>;
    using Super::Super;

public:
    void push_back(const py::array_t<T> &arr) {
        for (int i = 0; i < arr.size(); ++i) {
            Super::push_back(arr.at(i));
        }
    }
    void pop_back(size_t num) {
        for (int i = 0; i < num; ++i) {
            Super::pop_back();
        }
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
    mt.def("resize", [](array_t &self, const uint &t, const T &elm) { self.resize(t, elm); });
    mt.def("resize", [](array_t &self, const uint &t) { self.resize(t); });
    mt.def("as_float_array_t", [](array_t &self) {
        using type = float;
        auto size = self.size();
        return py::array_t<type>(size, reinterpret_cast<type *>((void *)(self.data())), py::none());
    });
    mt.def("clear", [](array_t &self) { self.clear(); });
    mt.def("__repr__", [&](array_t &self) {
        string ret = class_name + "[";
        for (int i = 0; i < self.size(); ++i) {
            ret += to_str(self[i]) + ",";
            if (i > 100) {
                ret += "......" + to_str(self[self.size() - 1]);
                break;
            }
        }
        return ret + "]";
    });
}

template<typename T>
void export_buffer(PythonExporter &exporter, const char *name = nullptr) {
    using array_t = python::Array<T>;
    name = name ? name : TypeDesc<T>::name().data();
    string buffer_name = ocarina::format("Buffer{}", name);
    auto mt = py::class_<Buffer<T>, RHIResource>(exporter.module, buffer_name.c_str());
    mt.def(py::init([](uint size) { return Context::instance().device->create_buffer<T>(size); }), ret_policy::move);
    mt.def("size", [](const Buffer<T> &self) { return self.size(); });
    mt.def("handle", [](const Buffer<T> &self) { return self.handle(); });
    mt.def("upload_immediately", [](const Buffer<T> &self, const vector<T> &lst) { self.upload_immediately(lst.data()); });
    mt.def("download_immediately", [](const Buffer<T> &self, py::buffer &lst) { self.download_immediately(lst.request().ptr); });
}

template<typename T>
void export_container(PythonExporter &exporter, const char *name = nullptr) {
    export_array<T>(exporter, name);
    export_buffer<T>(exporter, name);
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
    exporter.module.def("hash64", [](const T &arg) {
        return hash64(arg);
    });
    mt.def_static("alignof", []() {
        return alignof(T);
    });
    mt.def("to_bytes", [](T &self) {
        py::array_t<uchar, alignof(T)> ret{sizeof(T)};
        oc_memcpy(ret.request().ptr, addressof(self), ret.size());
        return ret;
    });
    mt.def_static("from_bytes", [](const py::array_t<uchar> &arr) {
        T ret;
        oc_memcpy(addressof(ret), arr.request().ptr, sizeof(T));
        return ret;
    });
    mt.def("to_floats", [](T &self) {
        py::array_t<float, alignof(T)> ret{sizeof(T) / sizeof(float)};
        oc_memcpy(ret.request().ptr, addressof(self), sizeof(T));
        return ret;
    });
    mt.def_static("from_floats", [](const py::array_t<float> &arr) {
        T ret;
        oc_memcpy(addressof(ret), arr.request().ptr, sizeof(T));
        return ret;
    });
    mt.def("__repr__", [](const T &self) {
        return to_str(self);
    });
    mt.def(py::init<>());
    if constexpr (sizeof(T) >= sizeof(float) && is_scalar_v<T>) {
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