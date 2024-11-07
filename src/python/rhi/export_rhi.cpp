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
        auto device = FileManager::instance().create_device(name);
        Context::instance().stream = std::make_unique<Stream>(device.create_stream());
        Context::instance().bindless_array = device.create_bindless_array();
        Context::instance().device = std::make_unique<Device>(std::move(device));
    });

    exporter.module.def("device", []() {
        return Context::instance().device.get();
    });
//    exporter.module.def("stream", []() {
//        return Context::instance().stream.get();
//    });
//    exporter.module.def("bindless_array", []() -> BindlessArray & {
//        return Context::instance().bindless_array;
//    });
}

void export_rhi(PythonExporter &exporter) {
    export_resource(exporter);
    export_device(exporter);
    export_init(exporter);
}