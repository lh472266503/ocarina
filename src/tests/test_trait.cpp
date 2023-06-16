//
// Created by Zero on 21/09/2022.
//

#include <utility>
#include "core/stl.h"
#include "dsl/common.h"
#include "dsl/polymorphic.h"
#include "dsl/common.h"
#include "rhi/common.h"
#include <type_traits>

using namespace ocarina;

struct Data : public Serializable<float>{
    Serial<float> f;
    Serial<float4> f4;

    OC_SERIALIZABLE_FUNC(Serializable<float>,f, f4)
};

struct Data2 : public Data {
    Serial<float3> f3;
    OC_SERIALIZABLE_FUNC(Serializable<float>, f3)
};

struct Test : public Serializable<float>{
    Serial<float2> a;
    Serial<int3> b;
    Serial<float> c;
    Serial<int> d;
    Serial<vector<float>> e;
    Serial<float3x3> f;
    RegistrableManaged<float> mw;
    Data2 data;
    OC_SERIALIZABLE_FUNC(Serializable<float>,a, b, c, d, e, f,mw, data)
};

union oc_scalar{
    int i;
    uint u;
    float f;
};

int main(int argc, char *argv[]) {
    log_level_debug();

    fs::path path(argv[0]);
    Context context(path.parent_path());
    //    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Printer::instance().init(device);

//    auto yy = std::is_de;

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
    RegistrableManaged<float> vv(ra);

    oc_scalar os{.f = 2.3f};
    os.f = 2.f;


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