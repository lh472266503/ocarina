//
// Created by ling.zhu on 2024/10/31.
//

#include "ext/pybind11/include/pybind11/pybind11.h"
#include "ext/pybind11/include/pybind11/stl.h"
#include "math/basic_types.h"
#include "rhi/device.h"
#include "rhi/common.h"
#include "dsl/dsl.h"
#include "util/file_manager.h"
#include "generator/cpp_codegen.h"

namespace py = pybind11;
using namespace ocarina;

void export_resource(py::module &m) {
    py::class_<RHIResource>(m, "RHIResource");
}

void export_device(py::module &m) {

    auto m_accel = py::class_<Accel, RHIResource>(m, "Accel");

    auto m_device = py::class_<Device, concepts::Noncopyable>(m, "Device");

    m_device.def_static("create", [](const string &name, const string &path) {
        DynamicModule::add_search_path(path);
        return FileManager::instance().create_device(name);
    });

    m_device.def("create_accel", [](const Device &device) { return device.create_accel(); }, py::return_value_policy::move);

    auto func = [] {
        auto &_ = Env::printer();
        CppCodegen a(false);
    };
    func();

}