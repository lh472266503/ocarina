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
}

void export_image_class(PythonExporter &exporter) {
    
}

void export_image(PythonExporter &exporter) {
    export_image_enum(exporter);
    export_image_class(exporter);
    export_image_load(exporter);
}