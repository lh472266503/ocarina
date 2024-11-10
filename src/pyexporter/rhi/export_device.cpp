//
// Created by ling.zhu on 2024/10/31.
//

#include "pyexporter/ocapi.h"
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

auto export_byte_buffer(PythonExporter &exporter) {
    auto mt = py::class_<ByteBuffer, RHIResource>(exporter.module, "ByteBuffer");
    mt.def_static("create", [](uint size) { return Context::instance().device->create_byte_buffer(size); }, ret_policy::move);
    mt.def("size_in_byte", [](ByteBuffer &self) {
        return self.size_in_byte();
    });
    mt.def("upload", [](ByteBuffer &self, const StructArray<float> &arr) {
        self.upload_immediately(arr.data());
    });
    mt.def("download", [](ByteBuffer &self, StructArray<float> &arr) {
        self.download_immediately(arr.data());
    });
    mt.def("download", [](ByteBuffer &self) {
                StructArray<float> ret;
                ret.resize(self.size_in_byte() / sizeof(float));
                self.download_immediately(ret.data());
                return ret; }, ret_policy::move);
}

auto export_mesh(PythonExporter &exporter) {
    auto mt = py::class_<RHIMesh, RHIResource>(exporter.module, "RHIMesh");
}

auto export_accel(PythonExporter &exporter) {
    auto mt = py::class_<Accel, RHIResource>(exporter.module, "Accel");
    mt.def_static("create", []() { return Context::instance().device->create_accel(); }, ret_policy::move);
}

void export_stream(PythonExporter &exporter) {
    auto mt = py::class_<Stream, RHIResource>(exporter.module, "Stream");
    mt.def("add", [](Stream &self, Command *cmd) -> Stream & { return self << cmd; }, ret_policy ::reference);
    mt.def("commit", [&](Stream &self) { self.commit(Commit{}); });
}

void export_bindless_array(PythonExporter &exporter) {
    auto mt = py::class_<BindlessArray, RHIResource>(exporter.module, "BindlessArray");
}

void export_texture(PythonExporter &exporter) {
    auto mt = py::class_<Texture, RHIResource>(exporter.module, "Texture");
    mt.def_static("create", [&](uint2 res, PixelStorage storage) {
        return Context::instance().device->create_texture(res, storage);
    });
    mt.def_property_readonly("resolution", [](Texture &self) -> uint2 {
        return self.resolution().xy();
    });
    mt.def_property_readonly("pixel_storage", [](Texture &self) {
        return self->pixel_storage();
    });
    mt.def_property_readonly("pixel_num", &Texture::pixel_num);
    mt.def("upload", [](Texture &self, const py::buffer &buffer) {
        self.upload_immediately(buffer.request().ptr);
    });
    mt.def("download", [](Texture &self, py::buffer &buffer) {
        self.download_immediately(buffer.request().ptr);
    });
}

void export_device(PythonExporter &exporter) {
    export_mesh(exporter);
    export_texture(exporter);
    export_byte_buffer(exporter);
    export_accel(exporter);

    auto &m = exporter.module;
    auto m_device = py::class_<Device, concepts::Noncopyable>(m, "Device");

    auto func = [] {
        auto &_ = Env::printer();
        CppCodegen a(false);
    };
    
    func();
    exporter.m_device = std::make_unique<py::class_<Device, concepts::Noncopyable>>(m_device);

    export_stream(exporter);
    export_bindless_array(exporter);
}