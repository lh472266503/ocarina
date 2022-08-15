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
    auto image_out = device.create_image<uchar4>(image_io.resolution());
    stream << image.upload_sync(image_io.pixel_ptr());

    Kernel kernel = [&](ImageVar<uchar4> img, ImageVar<uchar4> img_out) {
        Var v = img.read(make_uint2(dispatch_idx()));

        

        img_out.write(make_uint2(dispatch_idx()), v);
        //        print("-{}--{}--{}", v.x, v.y, v.z);
    };

    auto shader = device.compile(kernel);
    stream << shader(image, image_out).dispatch(image_io.resolution());
    stream << synchronize()
           << image_out.download_sync(image_io.pixel_ptr())
           << commit();
    image_io.save(path2);

    return 0;
}