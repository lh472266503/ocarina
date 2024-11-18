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
        using T = int;
        T ret;
        oc_memcpy(addressof(ret), py::buffer(bt).request().ptr, sizeof(T));
        return ret;
    });
    exporter.module.def("bytes2uint", [](const py::bytes &bt) {
        using T = uint;
        T ret;
        oc_memcpy(addressof(ret), py::buffer(bt).request().ptr, sizeof(T));
        return ret;
    });
    exporter.module.def("bytes2float", [](const py::bytes &bt) {
        using T = float;
        T ret;
        oc_memcpy(addressof(ret), py::buffer(bt).request().ptr, sizeof(T));
        return ret;
    });
}

PYBIND11_MODULE(ocapi, m) {
    auto &context = Context::instance();
    PythonExporter python_exporter;
    python_exporter.module = m;
    export_base_type(python_exporter);
    export_math(python_exporter);
    export_image(python_exporter);
    export_ast(python_exporter);
    export_rhi(python_exporter);
    export_window(python_exporter);
    export_rtx(python_exporter);
    export_byte_func(python_exporter);
}