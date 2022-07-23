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
    ocarina::vector<float> v;
    const int count = 10;
    for (int i = 0; i < count; ++i) {
        v.push_back(9);
    }

    Callable add = [&](Var<float> a, Var<float> b) {
        return a + b;
    };

    Kernel kn = [&](Var<float> a, Var<float> b) {
        print("{}, {}---------{}--\\n", a, b, b);
        a = add(a, b);
    };

    fs::path path(argv[0]);
    Context context(path.parent_path());
    context.clear_cache();
    return 0;
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    auto shader = device.compile(kn);
    Buffer<float> f_buffer = device.create_buffer<float>(count);
    stream << f_buffer.upload(v.data());
    stream << synchronize();
    stream << shader(1.f,6.f).dispatch(5);
    stream << synchronize();
    stream << commit();

    cout << f_buffer.handle() << endl;

    return 0;
}
