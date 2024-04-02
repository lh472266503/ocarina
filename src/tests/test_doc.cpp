//
// Created by Zero on 2023/11/23.
//

#include "util/image_io.h"
#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "math/base.h"
#include "base/scattering/interaction.h"
#include "core/platform.h"
#include "util/file_manager.h"

using namespace ocarina;

struct Triple {
    uint i{50}, j{}, k{};
    Hit h{};
    Triple(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triple() = default;
};

/// register a DSL struct, if you need upload a struct to device, be sure to register
OC_STRUCT(Triple, i, j, k, h){
    [[nodiscard]] Uint sum() const noexcept {
        return i + j + k;
}
}
;

struct Pair {
    uint i{50};
    BufferProxy<float3> b;
    BufferProxy<Triple> t;
    Pair() = default;
};

/// register a DSL struct, if you need upload a struct to device, be sure to register
OC_PARAM_STRUCT(Pair, i, b, t){

};

struct Param {
    uint i{50};
    BufferProxy<float3> b;
    BufferProxy<Triple> t;
    Pair pa;
    Param() = default;
};

/// register a DSL struct, if you need upload a struct to device, be sure to register
OC_PARAM_STRUCT(Param, i, b, t,pa){

};

auto get_cube(float x = 1, float y = 1, float z = 1) {
    x = x / 2.f;
    y = y / 2.f;
    z = z / 2.f;
    auto vertices = vector<float3>{
        float3(-x, -y, z), float3(x, -y, z), float3(-x, y, z), float3(x, y, z),    // +z
        float3(-x, y, -z), float3(x, y, -z), float3(-x, -y, -z), float3(x, -y, -z),// -z
        float3(-x, y, z), float3(x, y, z), float3(-x, y, -z), float3(x, y, -z),    // +y
        float3(-x, -y, z), float3(x, -y, z), float3(-x, -y, -z), float3(x, -y, -z),// -y
        float3(x, -y, z), float3(x, y, z), float3(x, y, -z), float3(x, -y, -z),    // +x
        float3(-x, -y, z), float3(-x, y, z), float3(-x, y, -z), float3(-x, -y, -z),// -x
    };
    auto triangles = vector<Triple>{
        Triple(0, 1, 3),
        Triple(0, 3, 2),
        Triple(6, 5, 7),
        Triple(4, 5, 6),
        Triple(10, 9, 11),
        Triple(8, 9, 10),
        Triple(13, 14, 15),
        Triple(13, 12, 14),
        Triple(18, 17, 19),
        Triple(17, 16, 19),
        Triple(21, 22, 23),
        Triple(20, 21, 23),
    };

    return ocarina::make_pair(vertices, triangles);
}

void test_compute_shader(Device &device, Stream &stream) {
    auto [vertices, triangles] = get_cube();

    Buffer<float3> vert = device.create_buffer<float3>(vertices.size());
    Buffer tri = device.create_buffer<Triple>(triangles.size());

    /// used for store the handle of texture or buffer
    BindlessArray bindless_array = device.create_bindless_array();
    //    uint v_idx = bindless_array.emplace(vert);
    //    uint t_idx = bindless_array.emplace(tri);

    using Elm = float4x4;
    uint len = 10;
    auto byte_buffer = device.create_byte_buffer(sizeof(Elm) * len, "");
    vector<Elm> byte_vec;

    byte_vec.resize(len);

    stream << byte_buffer.upload(byte_vec.data(), false);

    //    uint byte_handle = bindless_array.emplace(byte_buffer);

    /// upload buffer and texture handle to device memory
    //    stream << bindless_array->upload_buffer_handles(true) << synchronize();
    //    stream << bindless_array->upload_texture_handles(true) << synchronize();

    stream << vert.upload(vertices.data())
           << tri.upload(triangles.data());

    //    BufferView<float4> f4v = byte_buffer.view_as<float4>();

    Callable add = [&](Float a, Float b) {
        return a + b;
    };
    Pair pa;

    pa.b = vert.proxy();

    //    Type::of<Hit>();

    //    auto tt = Type::of<Triple>();

    //    Kernel k = [] {
    //        Var<Triple> a;
    //    };

    std::tuple<int, float> tp;

    traverse_tuple(tp, [&](auto elm) {
        int i = 0;
    });

    Kernel kernel = [&](Var<Pair> p, BufferVar<Triple> triangle,
                        ByteBufferVar byte_buffer_var, BufferVar<float3> vert_buffer) {
        $info("{}   ", p.i);
        Float3 ver = p.b.read(dispatch_id());
        $info("{}  {}  {}   {}", ver, p.b.size());
        HitVar hit;

        Float l = 1.f;
        Float r = 2.f;

        Float c = add(l, r);

        /// Note the usage and implementation of DSL struct member function, e.g sum()
        Var t = triangle.read(dispatch_id());
        $info("triple  index {} : i = {}, j = {}, k = {},  sum: {} ", dispatch_id(), t.i, t.j, t.k, t->sum());

        $info("vert from capture {} {} {}", vert.read(dispatch_id()));
        $info("vert_buffer  {} {} {}", vert_buffer.read(dispatch_id()));

        $switch(dispatch_id()) {
            $case(1) {
                $info("dispatch_idx is {} {} {}", dispatch_idx());
            };
            $default {
                $info("switch default  dispatch_idx is {} {} {}", dispatch_idx());
            };
        };

        $if(dispatch_id() == 1) {
            $info("if branch dispatch_idx is {} {} {}", dispatch_idx());
        }
        $elif(dispatch_id() == 2) {
            $info("if else branch dispatch_idx is {} {} {}", dispatch_idx());
        }
        $else {
            $info("else branch dispatch_idx is {} {} {}", dispatch_idx());
        };

        Uint count = 2;

        $for(i, count) {
            $info("count for statement dispatch_idx is {} {} {}, i = {} ", dispatch_idx(), i);
        };

        Uint begin = 2;
        Uint end = 10;
        $for(i, begin, end) {
            $info("begin end for statement dispatch_idx is {} {} {}, i = {} ", dispatch_idx(), i);
        };

        Uint step = 2;

        $for(i, begin, end, step) {
            $info("begin end step for statement dispatch_idx is {} {} {}, i = {} ", dispatch_idx(), i);
        };

        SOAView soa = byte_buffer_var.soa_view<Elm>();
        soa.write(dispatch_id(), make_float4x4(1.f * dispatch_id()));
        Var a = soa.read(dispatch_id());

        Uint2 aa = make_uint2(1);
        Float2 bb = make_float2(1.5f);

        bb += bb + aa;

        $info("\n {} {} {} {}  \n"
              "{} {} {} {}  \n"
              "{} {} {} {}  \n"
              "{} {} {} {}  \n",
              a[0], a[1], a[2], a[3]);

        $info("{} {}   ", bb);
    };
    Triple triple1{1, 2, 3};

    /// set debug range
    Env::debugger().set_lower(make_uint2(0));
    Env::debugger().set_upper(make_uint2(1));
    auto shader = device.compile(kernel, "test desc");
    stream << Env::debugger().upload();
    stream << shader(pa, tri, byte_buffer.view(), vert).dispatch(len)
           /// explict retrieve log
           << byte_buffer.download(byte_vec.data(), 0)
           << Env::printer().retrieve()
           << synchronize() << commit();

    int iii = 0;
}

struct Test {
    Uint a;
};

void test_lambda(Device &device, Stream &stream) {
    auto [vertices, triangles] = get_cube();

    Buffer<float3> vert = device.create_buffer<float3>(vertices.size());
    Buffer tri = device.create_buffer<Triple>(triangles.size());

    BindlessArray bindless_array = device.create_bindless_array();
    uint v_idx = bindless_array.emplace(vert);
    stream << bindless_array->upload_buffer_handles(true) << synchronize();
    stream << vert.upload(vertices.data())
           << tri.upload(triangles.data());

    Kernel kernel = [&](Uint i) {
        Float *p;
        HitVar *hit;
        Float b;
        $outline {

            Var<Triple> ttt;
            Var<Hit> hit;

            //            auto sn = hit.expression()->type()->simple_cname();

            $outline {
                b = 123;

                $outline {
                    p = new Float(15);
                    //                    hit = new HitVar{};
                    //                    *p = b;
                    //                    b = 19;
                };
            };
            //            $info("{}   i  ---   ", *p);
        };
        $outline {
            //            Float a = *p;
            //            //        Float bb = $outline {
            //            //            return (*hit).inst_id;
            //            //        };
            ////            b = 10;

            $info("{}    {}  ---   ", min(b, *p), call<float>("oc_max", b, *p));
        };
    };
    Shader shader = device.compile(kernel);

    stream << shader(1).dispatch(1)
           << Env::printer().retrieve()
           << synchronize() << commit();
}

struct Base {
    int a = 1;
    int b = 3;
    virtual ~Base() = default;
    virtual Base &operator=(const Base &base) noexcept = default;
};

struct Derive1 : public Base {
    int d = 9;
    SP<int> sp;
    Derive1(int arg) : d(arg) {}
    virtual ~Derive1() = default;
};

struct Derive : public Base {
    int c = 9;
    SP<int> sp;
    Derive(int arg) : c(arg) {}

    Derive &operator=(const Base &other) noexcept override {
        //        Derive *ptr = dynamic_cast<decltype(this) >(const_cast<Base *>(&other));
        *this = *dynamic_cast<decltype(this)>(const_cast<Base *>(&other));

        return *this;
    }

    Derive &operator=(const Derive &other) noexcept {

        Base::operator=(other);
        *sp = *other.sp;
        c = other.c;
        //        *this = dynamic_cast<decltype(*this) &>(const_cast<Base &>(other));

        return *this;
    }
};

void func(deep_copy_shared_ptr<Derive> d) {
}

void func(deep_copy_unique_ptr<Derive> d) {
}

void test_poly() {
    deep_copy_shared_ptr<Base> p1 = make_deep_copy_shared<Derive>(1);
    deep_copy_shared_ptr<Derive> p2 = make_deep_copy_shared<Derive>(2);

    deep_copy_unique_ptr<Base> p3 = make_deep_copy_unique<Derive>(1);
    deep_copy_unique_ptr<Derive> p4 = make_deep_copy_unique<Derive>(2);
    //    p0 = p1;
    //    p0 = p1;
    func(p2);
    p1->a = 10;
    //    deep_copy_shared_ptr<Base> p2 = make_deep_copy_shared<Derive>(2);
    p2->a = 8;

    dynamic_cast<Derive *>(p1.get())->sp = std::make_shared<int>(123);
    dynamic_cast<Derive *>(p2.get())->sp = std::make_shared<int>(456);
    //
    cout << "before p1->a = " << p1->a << ", p1->c = " << dynamic_cast<Derive *>(p1.get())->c << endl;
    cout << "before p2->a = " << p2->a << ", p2->c = " << dynamic_cast<Derive *>(p2.get())->c << endl;
    p1 = p2;
    cout << "after p1->a = " << p1->a << ", p1->c = " << dynamic_cast<Derive *>(p1.get())->c << endl;
    cout << "after p2->a = " << p2->a << ", p2->c = " << dynamic_cast<Derive *>(p2.get())->c << endl;

    //    auto p3 = make_unique<Derive1>(1);
    //
    //    unique_ptr<Derive> derive = dynamic_unique_pointer_cast<Derive>(move(p2));

    return;
}

void test_parameter_struct(Device &device, Stream &stream) {
    auto [vertices, triangles] = get_cube();

    Buffer<float3> vert = device.create_buffer<float3>(vertices.size());
    Buffer tri = device.create_buffer<Triple>(triangles.size());

    stream << vert.upload(vertices.data());
    stream << tri.upload(triangles.data());

    Param p;
    p.b = vert.proxy();
    p.t = tri.proxy();
    p.pa.b = vert.proxy();

    Kernel kernel = [&](Var<Param> pp) {
        auto v = pp.b.read(dispatch_id());
        $info("{} {} {} ", v);
        v = pp.pa.b.read(dispatch_id());
        $info("{} {} {}  -- ", v);
    };
    auto shader = device.compile(kernel, "param struct");

    stream << shader(p).dispatch(2);
    stream << Env::printer().retrieve();
    stream << synchronize() << commit();
}

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    FileManager file_manager(path.parent_path());

    /**
     * Conventional scheme
     * create device and stream
     * stream used for process some command,e.g buffer upload and download, dispatch shader
     * default is asynchronous operation
     */
    Device device = file_manager.create_device("cuda");
    Stream stream = device.create_stream();
    Env::printer().init(device);
    Env::debugger().init(device);

    //    Env::set_code_obfuscation(true);
    Env::set_valid_check(true);

    /// create rtx file_manager if need
    device.init_rtx();

    float3 a;
    int3 b;

    a = a + a;
    a = a + b;

    //    test_compute_shader(device, stream);
    test_parameter_struct(device, stream);
    //        test_lambda(device, stream);

    //    test_poly();
    return 0;
}