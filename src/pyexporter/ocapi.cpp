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
    auto mt = py::class_<StructArray<float>>(exporter.module, "StructArrayImpl");
    mt.def(py::init());
    mt.def("push_back_", [](StructArray<float> &self, const py::array_t<float> &arr) {
        self.push_back(arr);
    });
    mt.def("push_back_", [](StructArray<float> &self, float arg) {
        self.Super::push_back(arg);
    });
    mt.def("pop_back_", [](StructArray<float> &self, size_t size_in_byte) {
        self.pop_back(size_in_byte / sizeof(float));
    });
    mt.def("load", [](StructArray<float> &self, size_t ofs, size_t size_in_byte) {
        py::array_t<float> ret{static_cast<ssize_t>(size_in_byte / sizeof(float))};
        oc_memcpy(ret.request().ptr,
                  reinterpret_cast<const std::byte *>(self.data()) + ofs,
                  size_in_byte);
        return ret;
    });
    mt.def("as_float_array_t", [](StructArray<float> &self) {
        using type = float;
        auto size = self.size();
        return py::array_t<type>(size, self.data(), py::none());
    });
    mt.def("store", [](StructArray<float> &self, size_t ofs, const py::array_t<float> &arr) {
        auto index = ofs / sizeof(float);
        for (int i = 0; i < arr.size(); ++i) {
            self[i + index] = arr.at(i);
        }
    });
    mt.def("clear", [](StructArray<float> &self) {
        self.clear();
    });
    mt.def("size_in_byte", [](StructArray<float> &self) {
        return self.size() * sizeof(float);
    });
    mt.def("resize_", [](StructArray<float> &self, size_t size_in_byte) {
        self.resize(size_in_byte / sizeof(float));
    });
}

void export_base_type(PythonExporter &exporter) {
    auto &m = exporter.module;
    py::class_<concepts::Noncopyable>(m, "concepts_Noncopyable");
    py::class_<RTTI>(m, "RTTI");
    py::class_<Hashable, RTTI>(m, "Hashable");
    exporter.module.def("hash64", [](const string &str) {
        return hash64(str);
    });
}

PYBIND11_MODULE(ocapi, m) {
    auto &context = Context::instance();
    PythonExporter python_exporter;
    python_exporter.module = m;
    export_base_type(python_exporter);
    export_struct_array(python_exporter);
    export_image(python_exporter);
    export_ast(python_exporter);
    export_rhi(python_exporter);
    export_math(python_exporter);
    export_window(python_exporter);
    export_rtx(python_exporter);
}