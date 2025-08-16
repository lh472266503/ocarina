//
// Created by Zero on 2024/11/3.
//

#include "pyexporter/ocapi.h"
#include "math/basic_types.h"
#include "ocarina/src/rhi/file_manager.h"

namespace py = pybind11;
using namespace ocarina;

void export_device(PythonExporter &exporter);

void export_commands(PythonExporter &exporter);

void export_init(PythonExporter &exporter) noexcept {
    exporter.module.def("init_context", [](const string &name,
                                           const string &path) {
        DynamicModule::add_search_path(path);
        auto device = FileManager::instance().create_device(name);
        Context::instance().stream = std::make_unique<Stream>(device.create_stream());
        Context::instance().bindless_array = device.create_bindless_array();
        Context::instance().device = std::make_unique<Device>(std::move(device));
    });

    exporter.module.def("exit", []() {
        Context::destroy_instance();
    });

    exporter.module.def("device", []() {
        return Context::instance().device.get();
    }, ret_policy::reference);
    exporter.module.def("stream", []() {
        return Context::instance().stream.get();
    }, ret_policy::reference);
    exporter.module.def("bindless_array", []() -> BindlessArray & {
        return Context::instance().bindless_array;
    }, ret_policy::reference);
}

void export_rhi(PythonExporter &exporter) {
    export_commands(exporter);
    export_device(exporter);
    export_init(exporter);
}