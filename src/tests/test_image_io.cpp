//
// Created by Zero on 05/08/2022.
//

#include "util/image.h"
#include "core/stl.h"
#include "dsl/common.h"
#include "rhi/context.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "math/geometry.h"
#include "util/image.h"
#include "dsl/common.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    Context context(path.parent_path());
    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    auto path1 = R"(E:/work/compile/ocarina/res/test.png)";
    auto path2 = R"(E:/work/compile/ocarina/res/test.tga)";
    auto image = Image::load(path1, LINEAR);
    image.save(path2);
    return 0;
}