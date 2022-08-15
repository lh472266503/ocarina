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
        Var v = image.sample(0.f, 0.5f);
//
        uint2 res = image_io.resolution();
        int r = 5;
        Var<uint> min_x = max(0u, dispatch_idx().x - r);
        Var<uint> max_x = min(res.x - 1, dispatch_idx().x + r);
        Var<uint> min_y = max(0u, dispatch_idx().y - r);
        Var<uint> max_y = min(res.y - 1, dispatch_idx().y + r);

        Var<float4> var = make_float4(0.f);
        Var<uint> count = 0u;
        $for(x, min_x, max_x) {
            $for(y, min_y, max_y) {
                Var v = img.read<float4>(x, y);
                var += v;
                count += 1;
            };
        };
        var /= count.cast<float>();
        img_out.write(make_uint2(dispatch_idx()), var);
    };

    auto shader = device.compile(kernel);
    stream << shader(image, image_out).dispatch(image_io.resolution());
    stream << synchronize()
           << image_out.download_sync(image_io.pixel_ptr())
           << commit();
    image_io.save(path2);

    return 0;
}