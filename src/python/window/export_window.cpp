//
// Created by ling.zhu on 2024/11/7.
//


#include "python/ocapi.h"
#include "GUI/window.h"
#include "util/file_manager.h"

namespace py = pybind11;
using namespace ocarina;

void export_window(PythonExporter &exporter) {
    dependency_window();
    auto mt = py::class_<Window>(exporter.module, "Window");
    mt.def_static("create", [](uint width, uint height) {
        static auto ret = FileManager::instance().create_window("Python", make_uint2(width, height), "imGui");
        return ret.get();
    }, ret_policy::reference);
}