//
// Created by Zero on 26/04/2022.
//

#include "core/stl.h"
#include "dsl/common.h"
#include "rhi/context.h"
#include "generator/cpp_codegen.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "backends/cuda/cuda_builtin_embed.h"
#include "backends/cuda/cuda_math_embed.h"

using namespace ocarina;

int main(int argc, char *argv[]) {

    ocarina::vector<float> v;
    const int count = 10;
    for (int i = 0; i < count; ++i) {
        v.push_back(i);
    }

    Callable add = [&](Var<float> a, Var<float> b) {
        return a + b;
    };
    fs::path path(argv[0]);
    Context context(path.parent_path());
    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();

    auto texture = device.create_texture(make_uint2(20), PixelStorage::BYTE4);

    Buffer<float> f_buffer = device.create_buffer<float>(count);
    Kernel kn = [&](Var<float> a, Var<float> b, BufferVar<float> c) {
        //        configure_block(1,2,1);
        Var<int3> vec{1, 2, 3};
        Var<int> ii = 2;
        Var<int2> v22 = make_int2(ii, ii);
        Var<bool2> bv;
//        Var vf = vec.cast<float3>();
        Var vv = all(bv);
        Var f = 0.5f;
        f = fma(f, a, b);
        Var<int2> vec2 = vec.xy();
        vec2 = -vec2;
        //        Var<bool3> pred = vec > make_int3(5);
        vec = select(vec > make_int3(5), vec, -vec);
        print("{}, {}---------{}--", sqr(a), f, f_buffer.read(5));
        f_buffer.write(thread_id(), f_buffer.read(thread_id()) * 2);
        c.write(thread_id(), c.read(thread_id()) * 2);
        a = add(a, b);
        $return();
    };

    auto shader = device.compile(kn);
    //    shader.compute_fit_size();

    stream << f_buffer.upload_sync(v.data());
    stream << shader(1.f, 6.f, f_buffer).dispatch(10);
    stream << synchronize();
    stream << f_buffer.download_sync(v.data());
    stream << commit();
    for (int i = 0; i < count; ++i) {
        cout << v[i] << endl;
    }

    stream << shader(3.f, 6.f, f_buffer).dispatch(10);
    stream << synchronize();
    stream << f_buffer.download_sync(v.data());
    stream << commit();
    //
        for (int i = 0; i < count; ++i) {
            cout << v[i] << endl;
        }

    return 0;
}
