//
// Created by Zero on 2022/8/16.
//


#include "core/stl.h"
#include "dsl/common.h"
#include "rhi/context.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "util/image_io.h"
#include "dsl/common.h"
#include "windows/gl.h"
#include "util/image_io.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    Context context(path.parent_path());

    auto window = context.create_window("display", make_uint2(500), "gl");
    auto image_io = ImageIO::pure_color(make_float4(1,0,0,1), ColorSpace::LINEAR, make_uint2(500));
    window->run([&](double d){
        window->set_background(image_io.pixel_ptr<float4>(), make_uint2(500));
    });
}