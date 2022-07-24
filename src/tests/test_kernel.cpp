//
// Created by Zero on 26/04/2022.
//

#include "core/stl.h"
#include "dsl/common.h"
#include "runtime/context.h"
#include "generator/cpp_codegen.h"
#include "runtime/common.h"
#include <windows.h>
#include "backends/cuda/cuda_math_lib_embed.h"

using namespace ocarina;

int main(int argc, char *argv[]) {

//    cout << std::string (cuda_math_lib);
//    return 0;
    ocarina::vector<float> v;
    const int count = 10;
    for (int i = 0; i < count; ++i) {
        v.push_back(i);
    }

    Callable add = [&](Var<float> a, Var<float> b) {
        return a + b;
    };

    Kernel kn = [&](Var<float> a, Var<float> b, Var<Buffer<float>> c) {
//        configure_block(1,2,1);
        Var<int3> vec;
        Var<int2> vec2 = vec.xy();
        print("{}, {}---------{}--\\n", a, b, c.read(6));
        c.write(a.cast<int>(), b);
        a = add(a, b);
        $return();
    };

    fs::path path(argv[0]);
    Context context(path.parent_path());
    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    auto shader = device.compile(kn);
//    shader.compute_fit_size();
//    Buffer<float> f_buffer = device.create_buffer<float>(count);
//    stream << f_buffer.upload_sync(v.data());
//    stream << shader(1.f, 6.f, f_buffer).dispatch(5);
//    stream << synchronize();
//    stream << f_buffer.download_sync(v.data());
//    stream << commit();
//
//    for (int i = 0; i < count; ++i) {
//        cout << v[i] << endl;
//    }

    return 0;
}
