//
// Created by Zero on 05/08/2022.
//

#include "util/image_io.h"
#include "core/stl.h"
#include "dsl/common.h"
#include "rhi/context.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "math/geometry.h"
#include "util/image_io.h"
#include "dsl/common.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    Context context(path.parent_path());
    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    auto path1 = R"(E:/work/compile/ocarina/res/test.png)";
    auto path2 = R"(E:/work/compile/ocarina/res/test.jpg)";
    auto image_io = ImageIO::load(path1, LINEAR);

    auto image = device.create_image<uchar4>(image_io.resolution());

    stream << image.upload_sync(image_io.pixel_ptr());

    Kernel kernel = [&](ImageVar<uchar4> img) {
        img.write(make_uint2(dispatch_idx()), make_float4(0.5f));
        Var v = img.read<float4>(make_uint2(dispatch_idx()));
        print("-{}--{}--{}", v.x, v.y, v.z);
    };

    auto shader = device.compile(kernel);
    stream << shader(image).dispatch(50,50);
    stream << synchronize()
           << image.download_sync(image_io.pixel_ptr())
           << commit();
    image_io.save(path2);



    return 0;
}