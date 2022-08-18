//
// Created by Zero on 2022/8/18.
//

#include "util/image_io.h"
#include "core/stl.h"
#include "dsl/common.h"
#include "rhi/context.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "math/geometry.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    Context context(path.parent_path());
    context.clear_cache();
    Device device = context.create_device("cuda");
    device.init_rtx();

    Kernel kernel = [&](Int a) {
//        Float3 pos;
//        Float4 p4;
//        Var<Ray> ray;
//        Var<Ray> r = make_ray(float3(0), float3());
    };
    cout << detail::TypeDesc<Ray>::description();
    auto shader = device.compile(kernel);

    return 0;
}