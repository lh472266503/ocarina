//
// Created by Zero on 21/09/2022.
//

#include <utility>
#include "core/stl.h"
#include "dsl/common.h"
#include "dsl/polymorphic.h"
#include "dsl/common.h"
#include "rhi/common.h"

using namespace ocarina;

struct Data : public ISerializable<float>{
    Serialize<float> f;
    Serialize<float4> f4;

    OC_SERIALIZABLE_FUNC(float,f, f4)
};

struct Test : public ISerializable<float>{
    Serialize<float2> a;
    Serialize<int3> b;
    Serialize<float> c;
    Serialize<int> d;
    Serialize<vector<float>> e;
    Serialize<float3x3> f;
    ManagedWrapper<float> mw;
    Data data;
    OC_SERIALIZABLE_FUNC(float,a, b, c, d, e, f,mw, data)
};

int main(int argc, char *argv[]) {
    log_level_debug();

    fs::path path(argv[0]);
    Context context(path.parent_path());
    //    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Printer::instance().init(device);


    Test t;
    t.a = make_float2(1,2);
    t.a = [&]() {
        return make_float2(1.1,2.2);
    };
    t.b = make_int3(3,4,5);
    t.c = 6.8f;
    t.d = 100;
    for (int i = 0; i < 3; ++i) {
        t.e.hv().push_back(i);
    }
    t.f = make_float3x3(56.1f);
    t.data.f = 106;
    t.data.f4 = make_float4(199.f);
    ResourceArray ra = device.create_resource_array();
    t.mw.init(ra);
    t.mw.register_self();
    t.mw.push_back(9.98);
    t.mw.push_back(9.98);
    ManagedWrapper<float> vv(ra);


    t.encode(vv);
    vv.reset_device_buffer(device);
    vv.upload_immediately();
    vv.register_self();
    ra.prepare_slotSOA(device);
    stream << ra->upload_buffer_handles() << synchronize();



    Kernel kernel = [&](Float a) {
        DataAccessor<float> da{0u, vv};
        t.decode(&da);
        Printer::instance().info("a = {} {}", t.a.dv());
        Printer::instance().info("b = {} {} {}", t.b.dv());
        Printer::instance().info("c = {}", t.c.dv());
        Printer::instance().info("d = {}", t.d.dv());
        Printer::instance().info("e = {} {} {}", t.e.dv().as_vec3());
        Printer::instance().info("f0 = {} {} {}", t.f.dv()[0]);
        Printer::instance().info("f1 = {} {} {}", t.f.dv()[1]);
        Printer::instance().info("f2 = {} {} {}", t.f.dv()[2]);
        Printer::instance().info("data.f = {}", t.data.f.dv());
        Printer::instance().info("data.f4 = {} {} {} {}", *t.data.f4);
    };
    auto shader = device.compile(kernel);
    stream << shader(1.5f).dispatch(1);
    stream << synchronize() << commit();
    Printer::instance().retrieve_immediately();
    //    Float a{nullptr};
    //    Float a{};

    //    cout << is_vector_v<float2>;
    //    cout << (!is_dsl_v<Float3>) && is_vector_v<Float3> ;

    return 0;

    //    cout << typeid(scalar_t<int3>).name() << endl;
    //    cout << typeid(scalar_t<float>).name() << endl;
    //    cout << typeid(scalar_t<float4x4>).name() << endl;
    //    cout << typeid(scalar_t<Float3>).name() << endl;
    //    cout << typeid(scalar_t<Float>).name() << endl;
    //    cout << typeid(scalar_t<Float4x4>).name() << endl;
    //
    //    cout << endl;
    //
    //    cout << typeid(vec_t<float3, 2>).name() << endl;
    //    cout << typeid(vec_t<float, 2>).name() << endl;
    //    cout << typeid(vec_t<float4x4, 2>).name() << endl;
    //    cout << typeid(vec_t<Float3, 2>).name() << endl;
    //    cout << typeid(vec_t<Float, 2>).name() << endl;
    //    cout << typeid(vec_t<Float4x4, 2>).name() << endl;
    //
    //    cout << endl;
    //
    //    cout << typeid(matrix_t<float3, 2>).name() << endl;
    //    cout << typeid(matrix_t<float, 2>).name() << endl;
    //    cout << typeid(matrix_t<float4x4, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float3, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float4x4, 2>).name() << endl;
    //
    //    cout << typeid(boolean_t<Float>).name() << endl;
    //    cout << typeid(boolean_t<float>).name() << endl;

    return 0;
}