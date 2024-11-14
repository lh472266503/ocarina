//
// Created by ling.zhu on 2024/10/28.
//

#include "ocapi.h"

void export_math(PythonExporter &exporter);
void export_ast(PythonExporter &exporter);
void export_rhi(PythonExporter &exporter);
void export_rtx(PythonExporter &exporter);
void export_window(PythonExporter &exporter);
void export_image(PythonExporter &exporter);

Context *Context::s_context = nullptr;

Context &Context::instance() noexcept {
    if (s_context == nullptr) {
        s_context = new Context{};
    }
    return *s_context;
}

void Context::destroy_instance() noexcept {
    delete s_context;
    s_context = nullptr;
}

void export_struct_array(PythonExporter &exporter) {
    auto mt = py::class_<StructDynamicArray<float>>(exporter.module, "StructDynamicArrayImpl");
    mt.def(py::init());
    mt.def("push_back_", [](StructDynamicArray<float> &self, const py::array_t<float> &arr) {
        self.push_back(arr);
    });
    mt.def("pop_back_", [](StructDynamicArray<float> &self, size_t size_in_byte) {
        self.pop_back(size_in_byte / sizeof(float));
    });
    mt.def("load", [](StructDynamicArray<float> &self, size_t ofs, size_t size_in_byte) {
        py::array_t<float> ret{static_cast<ssize_t>(size_in_byte / sizeof(float))};
        oc_memcpy(ret.request().ptr,
                  reinterpret_cast<const std::byte *>(self.data()) + ofs,
                  size_in_byte);
        return ret;
    });
    mt.def("as_float_array_t", [](StructDynamicArray<float> &self) {
        using type = float;
        auto size = self.size();
        return py::array_t<type>(size, self.data(), py::none());
    });
    mt.def("store", [](StructDynamicArray<float> &self, size_t ofs, const py::array_t<float> &arr) {
        auto index = ofs / sizeof(float);
        for (int i = 0; i < arr.size(); ++i) {
            self[i + index] = arr.at(i);
        }
    });
    mt.def("clear", [](StructDynamicArray<float> &self) {
        self.clear();
    });
    mt.def("size_in_byte", [](StructDynamicArray<float> &self) {
        return self.size() * sizeof(float);
    });
    mt.def("resize_", [](StructDynamicArray<float> &self, size_t size_in_byte) {
        self.resize(size_in_byte / sizeof(float));
    });
}

void export_base_type(PythonExporter &exporter) {
    auto &m = exporter.module;
    py::class_<concepts::Noncopyable>(m, "concepts_Noncopyable");
    py::class_<RTTI>(m, "RTTI");
    py::class_<Hashable, RTTI>(m, "Hashable");
    py::class_<RHIResource, concepts::Noncopyable>(m, "RHIResource");
    exporter.module.def("hash64", [](const string &str) {
        return hash64(str);
    });
}

void export_byte_func(PythonExporter &exporter) {
    exporter.module.def("to_bytes", [](const basic_literal_t &v) {
        return ocarina::visit([&]<typename T>(const T &t) {
            return py::bytes(reinterpret_cast<const char *>(addressof(v)), sizeof(T));
        },
                              v);
    });
    exporter.module.def("bytes2int", [](const py::bytes &bt) {
        int ret;
        oc_memcpy(addressof(ret), py::buffer(bt).request().ptr, sizeof(int));
        return ret;
    });
    exporter.module.def("bytes2float", [](const py::bytes &bt) {
        float ret;
        oc_memcpy(addressof(ret), py::buffer(bt).request().ptr, sizeof(float));
        return ret;
    });
}

PYBIND11_MODULE(ocapi, m) {
    auto &context = Context::instance();
    PythonExporter python_exporter;
    python_exporter.module = m;
    export_base_type(python_exporter);
    export_math(python_exporter);
    export_struct_array(python_exporter);
    export_image(python_exporter);
    export_ast(python_exporter);
    export_rhi(python_exporter);
    export_window(python_exporter);
    export_rtx(python_exporter);
    export_byte_func(python_exporter);
}