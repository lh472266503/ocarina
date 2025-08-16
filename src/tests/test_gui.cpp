//
// Created by Zero on 2022/8/16.
//


#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/context.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "core/image.h"
#include "dsl/dsl.h"
#include "GUI_impl/imGui/window.h"
#include "core/image.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    RHIContext &file_manager = RHIContext::instance();

    auto window = file_manager.create_window("display", make_uint2(500), "gl");
    auto image_io = Image::pure_color(make_float4(1,0,0,1), ColorSpace::LINEAR, make_uint2(500));
    window->run([&](double d){
        window->set_background(image_io.pixel_ptr<float4>(), make_uint2(500));
    });
}