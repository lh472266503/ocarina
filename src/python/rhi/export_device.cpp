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

auto export_mesh(PythonExporter &exporter) {
    auto mt = py::class_<RHIMesh, RHIResource>(exporter.module, "RHIMesh");
}

auto export_accel(PythonExporter &exporter) {
    auto mt = py::class_<Accel, RHIResource>(exporter.module, "Accel");
}

void export_stream(PythonExporter &exporter) {
    auto mt = py::class_<Stream, RHIResource>(exporter.module, "Stream");
    mt.def("add", [](Stream &self, Command *cmd) -> Stream & { return self << cmd; }, ret_policy ::reference);
    mt.def("commit", [&](Stream &self) { self.commit(Commit{}); });
}

void export_bindless_array(PythonExporter &exporter) {
    auto mt = py::class_<BindlessArray, RHIResource>(exporter.module, "BindlessArray");
}

void export_device(PythonExporter &exporter) {
    export_mesh(exporter);
    export_accel(exporter);

    auto &m = exporter.module;
    auto m_device = py::class_<Device, concepts::Noncopyable>(m, "Device");

    m_device.def("create_accel", [](const Device &device) { return device.create_accel(); }, ret_policy::move);

    auto func = [] {
        auto &_ = Env::printer();
        CppCodegen a(false);
    };
    func();
    exporter.m_device = std::make_unique<py::class_<Device, concepts::Noncopyable>>(m_device);

    export_stream(exporter);
    export_bindless_array(exporter);
}