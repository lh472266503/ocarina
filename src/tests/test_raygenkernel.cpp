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

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    Context context(path.parent_path());
    context.clear_cache();
    Device device = context.create_device("cuda");
    device.init_rtx();
    Stream stream = device.create_stream();
    tinyobj::ObjReaderConfig obj_reader_config;
    obj_reader_config.triangulate = true;
    obj_reader_config.vertex_color = false;
    auto [vertices, triangle] = get_cube();

    auto vert = device.create_managed<float3>(vertices.size());

    vert.host() = std::move(vertices);

    Buffer v_buffer = device.create_buffer<float3>(vert.host().size());
    Buffer t_buffer = device.create_buffer<Triangle>(triangle.size());

    Mesh cube = device.create_mesh(vert.device(), t_buffer);

    stream << vert.upload_sync();
    stream << v_buffer.upload_sync(vert.host().data());
    stream << t_buffer.upload_sync(triangle.data());

    auto path1 = R"(E:/work/compile/ocarina/res/test.png)";
    auto path2 = R"(E:/work/compile/ocarina/res/test.jpg)";
    auto image_io = ImageIO::load(path1, LINEAR);

    auto image = device.create_texture(image_io.resolution(), image_io.pixel_storage());
    stream << image.upload_sync(image_io.pixel_ptr());

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
                        const ImageVar img,
                        Var<Triangle> tri) {
        Var<Ray> r = make_ray(Var(float3(0,0.1, -5)), float3(0,0,1));
        Var hit= accel.trace_closest(r);
        //        Float3 org = r->origin();
        Float3 pos = r->direction();
//        Var<Triangle> tri = t_buffer.read(3);
        Float4 pix = img.read<float4>(200,150);
        Float4 pix2 = img.read<float4>(200,150);
        Float3 p = vert.read(1);
        Var f2 = make_float2(Var(7.f));
        print("{},{}----------{} {}", hit.prim_id, hit.inst_id, hit->bary.x, hit.bary.y);
        print("{}  {}  {}  {} {}", tri.i, f2.x, f2.y, p.x, p.y);
    };
    auto shader = device.compile(kernel);
    stream << shader(t_buffer, accel, image,triangle[0]).dispatch(1);
    stream << synchronize() << commit();

    cout << vert[0].x << endl;

    return 0;
}