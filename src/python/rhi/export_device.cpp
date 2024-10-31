//
// Created by ling.zhu on 2024/10/31.
//

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "ext/pybind11/include/pybind11/operators.h"
#include "core/stl.h"
#include "math/basic_types.h"
#include "math/base.h"
#include "rhi/device.h"
#include "python/common.h"
#include "rhi/common.h"
#include "util/file_manager.h"

namespace py = pybind11;
using namespace ocarina;

void export_device(py::module &m) {
    py::class_<concepts::Noncopyable>(m, "concepts_Noncopyable");
    auto mt = py::class_<Device, concepts::Noncopyable>(m, "Device");
    mt.def_static("create", [&](const string &name, const string &path) {
        DynamicModule::add_search_path(path);
        return FileManager::instance().create_device(name);
    });
}