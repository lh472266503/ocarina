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
//        print("{}, {}------------\\n",a, b);
        return a + b;
    };

    Kernel kn = [&](Var<float> a, Var<float> b, Var<float> c) {
        print("{}, {}----------{}--\\n",a, b, c);
        a = add(a , b);
    };

    fs::path path(argv[0]);
    Context context(path.parent_path());
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();

    auto shader = device.compile(kn);

    stream << shader(1.f,6.f, 9.f).dispatch(5);

//    shader(1.f, 1.f);
//    Buffer<float> f_buffer = device.create_buffer<float>(10);

//    ocarina::vector<float> v;

//    stream << f_buffer.upload(v.data());

    stream << synchronize();

    stream << commit();
//    stream << f_buffer.download(v.data()) << synchronize();
//    stream << commit();

//    for (int i = 0; i < 10; ++i) {
//        cout << v[i] << endl;
//    }

    return 0;
}
