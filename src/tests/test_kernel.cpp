//
// Created by Zero on 26/04/2022.
//

#include "core/stl.h"
#include "dsl/common.h"
#include "runtime/context.h"
#include "generator/cpp_codegen.h"
#include "runtime/common.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    Callable add = [&](Var<float> a, Var<float> b) {
        return a + b;
    };

    fs::path path(argv[0]);
    Context context(path.parent_path());
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();

    Buffer<float> f_buffer = device.create_buffer<float>(10);

    ocarina::vector<float> v;
    for (int i = 0; i < 10; ++i) {
        v.push_back(i);
    }
    stream << f_buffer.upload(v.data()) << synchronize();
    stream << commit([&](void*) -> int{
        return 0;
    });

    return 0;
}
