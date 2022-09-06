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
#include "util/image_io.h"
#include "dsl/common.h"

using namespace ocarina;

int main(int argc, char *argv[]) {

    float3 v1 = make_float3(2), v2, v3;
    coordinate_system(v1,v2, v3);

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

    auto path2 = R"(E:/work/compile/ocarina/res/test.png)";
    auto image = ImageIO::load(path2, LINEAR);

    auto texture = device.create_image(image.resolution(), image.pixel_storage());
    stream << texture.upload_sync(image.pixel_ptr());

    Buffer<float> f_buffer = device.create_buffer<float>(count);
    Kernel kn = [&](Var<float> a, Var<float> b, BufferVar<float> c, ImageVar tex) {

        Var<float3> v1 = make_float3(a), v2,v3;
//        v1 = normalize(v1);
        coordinate_system(v1,v2,v3);
        print("{},{},{}",v2.x,v2.y,v2.z);
//                configure_block(1,2,1);
        Array<uint> ua(10);
        ua[5] = 1u;
        Var uuu = ua[5];
        Var<int3> vec = make_int3(1,2, a.cast<int>());
        Var<int> ii = 2;
        Var<int2> v22 = make_int2(ii, ii);
        Var<bool2> bv;

        Var tex_v = tex.sample<float4>(a, b);
        Var tv2 = texture.sample<float4>(a, b);

        Var vv = all(bv);
        Var f = 0.5f;
        f = fma(f, a, b);
        Var mat = make_float4x4(1.f);
        Var<int2> vec2 = make_int2(1);
        vec2 = -vec2;
        //        Var<bool3> pred = vec > make_int3(5);
        vec = select(vec > make_int3(5), vec, -vec);
        Var tr = tex.read<float4>(20u,20u);
        print("{}, {}---------{}--{}>>>>>{}", sqr(a), tv2.x, f_buffer.read(5), tex_v.x, tr.x);
        f_buffer.write(thread_id(), f_buffer.read(thread_id()) * 2);
//        c.write(thread_id(), c.read(thread_id()) * 2);
        c[thread_id()] *= 2;
        a = add(a, b);
    };

    auto shader = device.compile(kn);
    //    shader.compute_fit_size();
//    return 0;
    stream << f_buffer.upload_sync(v.data());
    stream << shader(0.1f, 0.9f, f_buffer, texture).dispatch(10);
    stream << synchronize();
    stream << f_buffer.download_sync(v.data());
    stream << commit();
//    for (int i = 0; i < count; ++i) {
//        cout << v[i] << endl;
//    }

    stream << shader(0.95f, 0.55f, f_buffer, texture).dispatch(10);
    stream << synchronize();
    stream << f_buffer.download_sync(v.data());
    stream << commit();
    //
//        for (int i = 0; i < count; ++i) {
//            cout << v[i] << endl;
//        }

    return 0;
}
