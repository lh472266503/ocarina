//
// Created by Zero on 2024/11/3.
//

#include "python/ocapi.h"
#include "math/basic_types.h"
#include "util/file_manager.h"

namespace py = pybind11;
using namespace ocarina;

void export_resource(PythonExporter &exporter);

void export_device(PythonExporter &exporter);

void export_init(PythonExporter &exporter) noexcept {
    exporter.module.def("init_context", [](const string &name,
                                           const string &path) {
        DynamicModule::add_search_path(path);
        Context::instance().device = std::make_unique<Device>(FileManager::instance().create_device(name));
    });

    exporter.module.def("device", []() {
        return Context::instance().device.get();
    });
}

void export_rhi(PythonExporter &exporter) {
    export_resource(exporter);
    export_device(exporter);
    export_init(exporter);
}