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

auto export_byte_buffer(PythonExporter &exporter) {
    auto mt = py::class_<ByteBuffer, RHIResource>(exporter.module, "ByteBuffer");
    mt.def(
        py::init([](size_t size) {
            return Context::instance().device->create_byte_buffer(size);
        }),
        ret_policy::move);
    mt.def("size_in_byte", [](ByteBuffer &self) {
        return self.size_in_byte();
    });
    mt.def("handle", [](ByteBuffer &self) { return self.handle(); });
    mt.def("upload_immediately", [](ByteBuffer &self, const StructArray<float> &arr) {
        self.upload_immediately(arr.data());
    });
    mt.def("upload_immediately", [](ByteBuffer &self, const py::buffer &arr) {
        self.upload_immediately(arr.request().ptr);
    });
    mt.def("upload", [](ByteBuffer &self, const StructArray<float> &arr) { return self.upload(arr.data()); }, ret_policy::reference);
    mt.def("upload", [](ByteBuffer &self, const py::buffer &arr) { return self.upload(arr.request().ptr); }, ret_policy::reference);
    mt.def("download_immediately", [](ByteBuffer &self, StructArray<float> &arr) {
        self.download_immediately(arr.data());
    });
    mt.def("download_immediately", [](const ByteBuffer &self, py::buffer &lst) {
        self.download_immediately(lst.request().ptr);
    });
    mt.def("download", [](ByteBuffer &self, StructArray<float> &arr) { return self.download(arr.data()); }, ret_policy::reference);
    mt.def("download", [](ByteBuffer &self, py::buffer &arr) { return self.download(arr.request().ptr); }, ret_policy::reference);
}

auto export_mesh(PythonExporter &exporter) {
    OC_EXPORT_ENUM(exporter.module, AccelUsageTag, FAST_BUILD,
                   FAST_UPDATE,
                   FAST_TRACE)
    OC_EXPORT_ENUM(exporter.module, AccelGeomTag, NONE,
                   DISABLE_ANYHIT,
                   SINGLE_ANYHIT_CALL,
                   DISABLE_FACE_CULLING)
    OC_EXPORT_STRUCT(exporter.module, MeshParams, vert_handle,
                     vert_stride, vert_num, tri_handle, tri_stride,
                     tri_num, usage_tag, geom_tag)

    auto mt = py::class_<RHIMesh, RHIResource>(exporter.module, "RHIMesh");
    mt.def(py::init([](MeshParams params) {
               return Context::instance().device->create<RHIMesh>(params);
           }),
           ret_policy ::move);
    mt.def("build_bvh", [](RHIMesh &self) { return self.build_bvh(); }, ret_policy::reference);
}

auto export_accel(PythonExporter &exporter) {
    auto mt = py::class_<Accel, RHIResource>(exporter.module, "Accel");
    mt.def(py::init([]() { return Context::instance().device->create_accel(); }), ret_policy::move);
    mt.def("build_bvh", [](Accel &self) { return self.build_bvh(); }, ret_policy::reference);
}

void export_stream(PythonExporter &exporter) {
    auto mt = py::class_<Stream, RHIResource>(exporter.module, "Stream");
    mt.def("add", [](Stream &self, Command *cmd) -> Stream & { return self << cmd; }, ret_policy ::reference);
    mt.def("sync", [](Stream &self) -> Stream & { return self << synchronize(); }, ret_policy ::reference);
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
    mt.def("upload_immediately", [](Texture &self, const py::buffer &buffer) {
        self.upload_immediately(buffer.request().ptr);
    });
    mt.def("download_immediately", [](Texture &self, py::buffer &buffer) {
        self.download_immediately(buffer.request().ptr);
    });
    mt.def("upload", [](Texture &self, const py::buffer &buffer) { return self.upload(buffer.request().ptr); }, ret_policy::reference);
    mt.def("download", [](Texture &self, py::buffer &buffer) { return self.download(buffer.request().ptr); }, ret_policy::reference);
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