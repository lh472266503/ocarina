//
// Created by Zero on 2022/8/18.
//

#include "rhi/device.h"
#include "rhi/context.h"
#include "dsl/common.h"
#include "math/rt.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    Context context(path.parent_path());
    context.clear_cache();
    Device device = context.create_device("cuda");
    device.init_rtx();

    Kernel kernel = [&]() {
        Float3 pos;
        Float4 p4;
        Var<Ray> ray;
        cout << typeid(ray.org_min).name() << endl;
        cout << typeid(p4).name() << endl;
//        pos = make_float3(p4);
    };

    return 0;
}