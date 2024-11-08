//
// Created by Zero on 2024/11/8.
//

#include "python/ocapi.h"
#include "GUI/window.h"
#include "util/file_manager.h"

namespace py = pybind11;
using namespace ocarina;

void export_image_load(PythonExporter &exporter) {
}

void export_image_enum(PythonExporter &exporter) {
    OC_EXPORT_ENUM(exporter.module, PixelStorage,
                   BYTE1, BYTE2, BYTE4,
                   UINT1, UINT2, UINT4,
                   FLOAT1, FLOAT2, FLOAT4, UNKNOWN)

    OC_EXPORT_ENUM(exporter.module, ImageWrap,
                   Repeat, Black, Clamp)

    OC_EXPORT_ENUM(exporter.module, ColorSpace,
                   LINEAR, SRGB)
}

void export_image_class(PythonExporter &exporter) {
    auto _ = py::class_<ImageBase, concepts::Noncopyable>(exporter.module, "ImageBase");
    auto mt = py::class_<Image, ImageBase>(exporter.module, "Image");
    mt.def_static("load", [](const string &fn, ColorSpace color_space, float3 scale) {
        return Image::load(fn, color_space, scale);
    });
    mt.def_static("load", [](const string &fn, ColorSpace color_space) {
        return Image::load(fn, color_space);
    });
    mt.def_property_readonly("resolution", &Image::resolution);
    mt.def_property_readonly("pixel_storage", &Image::pixel_storage);
    mt.def("as_float_array_t", [](Image &self) {
        auto size = self.size_in_bytes() / sizeof(float);
        return py::array_t<float>(size - 0, self.pixel_ptr<float>(), py::none());
    });
}

void export_image(PythonExporter &exporter) {
    export_image_enum(exporter);
    export_image_class(exporter);
    export_image_load(exporter);
}