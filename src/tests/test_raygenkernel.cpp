//
// Created by Zero on 2022/8/18.
//

#include "util/image_io.h"
#include "core/stl.h"
#include "dsl/common.h"
#include "rhi/common.h"
#include "math/base.h"
#include "tiny_obj_loader.h"
#include "cornell_box.h"

using namespace ocarina;

struct Triangle {
public:
    uint i, j, k;
    Triangle(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triangle() = default;
};
OC_STRUCT(Triangle, i, j, k){};

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
    auto triangles = vector<Triangle>{
        Triangle(0, 1, 3),
        Triangle(0, 3, 2),
        Triangle(6, 5, 7),
        Triangle(4, 5, 6),
        Triangle(10, 9, 11),
        Triangle(8, 9, 10),
        Triangle(13, 14, 15),
        Triangle(13, 12, 14),
        Triangle(18, 17, 19),
        Triangle(17, 16, 19),
        Triangle(21, 22, 23),
        Triangle(20, 21, 23),
    };

    return ocarina::make_pair(vertices, triangles);
}

auto operator-(Array<float> arr) {
    const Expression *expr = Function::current()->unary(arr.type(), UnaryOp::NEGATIVE, arr.expression());
    return Array<float>(arr.size(), expr);
}

//template<typename... Args>
//void func(Args... args) {
//    auto tp = args_to_tuple(args...);
//
//    int i = 0;
//}

int main(int argc, char *argv[]) {

//    func(1, 2, 3, 4, 5, 6, 7, 8, 18);
//
//    return 0;
    log_level_debug();

    fs::path path(argv[0]);
    Context context(path.parent_path());
    //    context.clear_cache();
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Printer::instance().init(device);
    auto path1 = R"(E:/work/test_dir/D5.png)";
    auto path2 = R"(E:/work/test_dir/D5.jpg)";
    auto image_io = ImageIO::load(path1, LINEAR);
    auto image = device.create_texture(image_io.resolution(), image_io.pixel_storage());
    stream << image.upload_sync(image_io.pixel_ptr());
    device.init_rtx();

    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    auto [vertices, triangle] = get_cube();

    auto vert = device.create_managed<float3>(vertices.size());

    vert.host() = std::move(vertices);

    Buffer v_buffer = device.create_buffer<float3>(vert.host().size());
    Buffer t_buffer = device.create_buffer<Triangle>(triangle.size());

    Managed managed = device.create_managed<float>(100);
    for (int i = 0; i < 100; ++i) {
        managed.push_back(i);
    }
    managed.upload_immediately();
    Mesh cube = device.create_mesh(vert.device(), t_buffer);

    ResourceArray bindless_array = device.create_resource_array();

    auto r1 = bindless_array.emplace(v_buffer.view(1));
    bindless_array.emplace(image);
    auto r2 = bindless_array.emplace(v_buffer);

    uint index = bindless_array.emplace(managed.device());

    stream << vert.upload_sync();
    stream << v_buffer.upload_sync(vert.host().data());
    stream << t_buffer.upload_sync(triangle.data());

    bindless_array.prepare_slotSOA(device);
    stream << bindless_array->upload_buffer_handles() << synchronize();
    stream << bindless_array->upload_texture_handles() << synchronize();

    stream << cube.build_bvh();

    Accel accel = device.create_accel();
    accel.add_mesh(cube, make_float4x4(1.f));
    stream << accel.build_bvh();
    stream << synchronize() << commit();

    Callable cb = [&]() {
        return Var<std::array<float, 10>>();
    };

    Kernel kernel = [&](const BufferVar<Triangle> t_buffer,
                        const Var<Accel> acc,
                        const TextureVar img,
                        Var<Triangle> tri,
                        ResourceArrayVar ba) {
        //        t_buffer.atomic()
        //        managed.device().atomic(1).fetch_sub(2);
                Var<Ray> r = make_ray(Var(float3(0, 0.1, -5)), float3(1.6f, 0, 1));
                Var hit = accel.trace_closest(r);
        Int3 f = make_int3(ba.byte_buffer(index).read<float>(19 * 4).cast<int>(), 6, 9);
        auto arr = bindless_array.byte_buffer(index).read_dynamic_array<float>(3, 19 * 4);
        Printer::instance().warn_with_location("{} {} {} {} {} ", f, arr.sub(1,3).as_vec2());
//      Int a = 1, b = 2, c = 3;
//      printer.log_debug("--{} {} {}", a, b, c);
        //        prints("++{} {} {}", f);
        //        printer.log_debug("--------{} {} {}", 1.5f,f.x,1.11f);
        //        print("sdfasdasdfasdsdafsdafasdfsda{} {} {}", 1.5f,f.x,1.9f);
        //        print("------adfasdfsdafasdasasfasdfasdfasdf--{} {} {}", 1.5f,f.x,1.11f);
        //        Array<float> arr = Array<float>::create(1.f, 2.f, 3.f, 4.f);
        //        arr *= arr;
        //        prints("{} {} {} {}", arr.wzyx().as_vec4());
        //        Float3 pos = r->direction();
        //        Float4 pix = img.read<float4>(200, 150);
        //        Float2 uv = make_float2(0.7f);
        //        Float4 pix2 = img.sample(4, uv).as_vec4();
        //        Float3 p = vert.read(1);
        //        Var f2 = make_float2(Var(7.f));
        //        auto t = bindless_array.buffer<array<float3, 1>>(0).read(0);
        //        print("{},{}----------{} {}", hit.prim_id, hit.inst_id, hit->bary.x, hit.bary.y);
        //        print("{}  {}  {}  {} {}", tri.i, f2.x, f2.y, p.x, p.y);
        //        //        prints("{} {} {}", t);
        //        prints("{} {} {} {}", pix2);
        //        prints("{} {} {} {}", ba.tex(0).sample(4, uv).as_vec4());
        //        prints("{} {} {} {}", bindless_array.tex(0).sample(4, uv).as_vec4());
    };
    auto shader = device.compile(kernel);
    stream << shader(t_buffer, accel, image, triangle[0], bindless_array).dispatch(3);
    stream << synchronize() << commit();

    float tf = bit_cast<float>(19);
    OC_WARNING_FORMAT("{}", tf);
    Printer::instance().retrieve_immediately();
//    cout << "sdafasdf" << endl;
    Printer::instance().retrieve_immediately();
    Printer::destroy_instance();
    return 0;
}