//
// Created by ling.zhu on 2024/10/31.
//

#include "python/ocapi.h"
#include "math/basic_types.h"
#include "rhi/common.h"
#include "dsl/dsl.h"
#include "util/file_manager.h"
#include "generator/cpp_codegen.h"

namespace py = pybind11;
using namespace ocarina;

void export_resource(PythonExporter &exporter) {
    auto &m = exporter.module;
    py::class_<RHIResource>(m, "RHIResource");
}

void export_mesh(PythonExporter &exporter) {
    auto m_mesh = py::class_<RHIMesh, RHIResource>(exporter.module, "RHIMesh");
}

void export_accel(PythonExporter &exporter) {
    auto m_accel = py::class_<Accel, RHIResource>(exporter.module, "Accel");
}

void export_device(PythonExporter &exporter) {
    auto &m = exporter.module;
    auto m_device = py::class_<Device, concepts::Noncopyable>(m, "Device");

    m_device.def("create_accel", [](const Device &device) { return device.create_accel(); }, ret_policy::move);

    auto func = [] {
        auto &_ = Env::printer();
        CppCodegen a(false);
    };
    func();
    exporter.m_device = std::make_unique<py::class_<Device, concepts::Noncopyable>>(m_device);

    export_mesh(exporter);
    export_accel(exporter);
}