//
// Created by Zero on 2024/11/8.
//

#include "python/ocapi.h"
#include "GUI/window.h"
#include "util/file_manager.h"

namespace py = pybind11;
using namespace ocarina;

void export_pixel_storage(PythonExporter &exporter) {
    OC_EXPORT_ENUM(exporter.module, PixelStorage,
                   BYTE1, BYTE2, BYTE4,
                   UINT1, UINT2, UINT4,
                   FLOAT1, FLOAT2, FLOAT4, UNKNOWN)
    
}

void export_image(PythonExporter &exporter) {
    export_pixel_storage(exporter);
}