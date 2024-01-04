//
// Created by Zero on 2023/11/23.
//

#include "util/image_io.h"
#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "math/base.h"
#include "core/platform.h"

using namespace ocarina;

struct Triple {
    uint i{5}, j{}, k{};
    Triple(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triple() = default;
};

/// register a DSL struct, if you need upload a struct to device, be sure to register
OC_STRUCT(Triple, i, j, k){
    [[nodiscard]] Uint sum() const noexcept {
        return i + j + k;
    }
};

struct TTT {
    Triple triple;
};
OC_STRUCT(TTT, triple){};

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
    uint v_idx = bindless_array.emplace(vert);
    uint t_idx = bindless_array.emplace(tri);

    /// upload buffer and texture handle to device memory
    stream << bindless_array->upload_buffer_handles() << synchronize();
    stream << bindless_array->upload_texture_handles() << synchronize();

    stream << vert.upload(vertices.data())
           << tri.upload(triangles.data());

    Kernel kernel = [&](Var<Triple> triple, BufferVar<Triple> triangle, Var<BindlessArray> ra) {
        $info("triple   {} {} {}", triple.i, triple.j, triple.k);

        Var t = triangle.read(dispatch_id());

        /// Note the usage and implementation of DSL struct member function, e.g sum()
        $info("triple  index {} : i = {}, j = {}, k = {},  sum: {} ", dispatch_id(), t.i, t.j, t.k, t->sum());

        $info("vert from capture {} {} {}", vert.read(dispatch_id()));

        vert.write(dispatch_id(), vert.read(dispatch_id()));
        $info("vert from capture resource array {} {} {}", bindless_array.buffer<float3>(0).read(Var(10000)));
        $info("vert from ra {} {} {}", ra.buffer<float3>(v_idx).read(dispatch_id()));

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

        $debug_if(dispatch_id() == 0, "{} ", step);
        /// execute if thread idx in debug range
        $condition_execute {
            Float f = 2.f;
            Float a = 6.f;
            $warn_with_location("this thread idx is in debug range {} {} {},  f * a = {} ",
                                vert.read(dispatch_id()), ra.buffer<Triple>(t_idx).size_in_byte() / 12);
        };
    };
    Triple triple1{1, 2, 3};

    /// set debug range
    Env::debugger().set_lower(make_uint2(0));
    Env::debugger().set_upper(make_uint2(1));
    auto shader = device.compile(kernel, "test desc");
    stream << Env::debugger().upload();
    stream << shader(triple1, tri, bindless_array).dispatch(1)
           /// explict retrieve log
           << Env::printer().retrieve()
           << synchronize() << commit();
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
    stream << bindless_array->upload_buffer_handles() << synchronize();
    stream << vert.upload(vertices.data())
           << tri.upload(triangles.data());

    Kernel kernel = [&](Uint i) {
        OCRay hit;

        Uint begin = 2;
        Uint end = 10;
        //        Uint end1 = 10;
        //        Uint end2 = 10;
        //        Uint end3 = 10;
        //        Uint end4 = 10;
        //        Uint end6 = 10;

        //        Lambda func = [&](int i, float z, Float a) {
        //            $info("{}    {}   {}     ---------", i,z, a);
        ////            return a;
        //        };
        //
        //       func(1,2.f, Float(7.f));
        Var<TTT> triple{};
        Float3 a;
        float3 b;
        a = b;
//        triple = TTT();
        Var aa = $outline {
            Var aa = $outline {
                return begin * begin;
            }
            +end + end;
            return aa;
        };

        //        auto ld = Lambda([&] {
        //            Test test;
        //            test.a = begin * 2;
        //            return test;
        //        });

        //        auto aa = ld();
        //        Uint b = aa + 2;

        //        $outline {
        //                $info("{} {} {} --===", begin,begin,begin);
        //        $info("{} === --===", end);
        $info("{} {} --===", aa, triple.triple.i);
        //        $info("{}  --===", begin);
        //        };
        //        Callable cb = [&](Float a) {
        //            a = a + end6;
        //            auto c = end6;
        //
        //            Callable cb2 = [&](Uint& b) {
        //                b = begin + b;
        //                Var v = vert.read(1);
        //                $info("{}         ---------", v.x);
        //                return b;
        //            };
        //
        //            cb2(end6);
        //
        //            int i = 0;
        //        };
        //
        //        cb(Float(1.f));
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

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    Context context(path.parent_path());

    /**
     * Conventional scheme
     * create device and stream
     * stream used for process some command,e.g buffer upload and download, dispatch shader
     * default is asynchronous operation
     */
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Env::printer().init(device);
    Env::debugger().init(device);

    //    Env::set_code_obfuscation(true);
    //    Env::set_valid_check(false);

    /// create rtx context if need
    device.init_rtx();
        test_compute_shader(device, stream);
//    test_lambda(device, stream);

    //    test_poly();
    return 0;
}