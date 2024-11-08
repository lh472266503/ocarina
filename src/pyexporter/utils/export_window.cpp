//
// Created by ling.zhu on 2024/11/7.
//

#include "pyexporter/ocapi.h"
#include "GUI/window.h"
#include "util/file_manager.h"

namespace py = pybind11;
using namespace ocarina;

void export_window(PythonExporter &exporter) {
    dependency_window();
    auto mt = py::class_<Window>(exporter.module, "Window");
    mt.def_static("create", [](uint2 res) {
        static auto ret = FileManager::instance().create_window("Python", res, "imGui");
        ret->init_widgets();
        return ret.get();
    }, ret_policy::reference);

    mt.def("set_clear_color", &Window::set_clear_color);
    mt.def("set_should_close", &Window::set_should_close);
    mt.def("set_background", [](Window &self, const py::array_t<float> &buffer) {
        self.set_background(reinterpret_cast<float4 *>(buffer.request().ptr));
    });

    mt.def("set_background", [](Window &self, const py::array_t<uchar> &buffer) {
        self.set_background(reinterpret_cast<uchar4 *>(buffer.request().ptr));
    });

    mt.def("set_background", [](Window &self, const python::Array<float4> &arr) {
        self.set_background(arr.data());
    });

#define OC_EXPORT_WINDOW_CB(func_name)                              \
    mt.def(#func_name, [](Window &self, const py::function &func) { \
        self.func_name([=]<typename... Args>(Args &&...args) {      \
            func(OC_FORWARD(args)...);                              \
        });                                                         \
    });

    OC_EXPORT_WINDOW_CB(set_mouse_callback)
    OC_EXPORT_WINDOW_CB(set_cursor_position_callback)
    OC_EXPORT_WINDOW_CB(set_key_callback)
    OC_EXPORT_WINDOW_CB(set_scroll_callback)
    OC_EXPORT_WINDOW_CB(run)
#undef OC_EXPORT_WINDOW_CB
}