//
// Created by Zero on 2022/8/18.
//

#include "util/image_io.h"
#include "core/stl.h"
#include "dsl/common.h"
#include "rhi/common.h"
#include "math/base.h"
#include "math/geometry.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include "cornell_box.h"

using namespace ocarina;

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
    tinyobj::ObjReader obj_reader;
    if (!obj_reader.ParseFromString(obj_string, "", obj_reader_config)) {
        std::string_view error_message = "unknown error.";
        if (auto &&e = obj_reader.Error(); !e.empty()) { error_message = e; }
        OC_ERROR_FORMAT("Failed to load OBJ file: {}", error_message);
    }
    if (auto &&e = obj_reader.Warning(); !e.empty()) {
        OC_ERROR_FORMAT("{}", e);
    }

    auto &&p = obj_reader.GetAttrib().vertices;
    std::vector<float3> vertices;

    vertices.reserve(p.size() / 3u);
    for (auto i = 0u; i < p.size(); i += 3u) {
        vertices.emplace_back(float3{p[i + 0u], p[i + 1u], p[i + 2u]});
    }

    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.upload_sync(vertices.data());
    std::vector<Mesh> meshes;
    std::vector<Buffer<Triangle>> triangle_buffers;
    for (auto &&shape : obj_reader.GetShapes()) {
        auto index = static_cast<uint>(meshes.size());
        auto &&t = shape.mesh.indices;
        auto triangle_count = t.size() / 3u;
        OC_INFO_FORMAT(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, index, triangle_count);
        std::vector<uint> indices;
        indices.reserve(t.size());

        for (auto i : t) { indices.emplace_back(i.vertex_index); }
        auto &&triangle_buffer = triangle_buffers.emplace_back(device.create_buffer<Triangle>(triangle_count));
        auto &&mesh = meshes.emplace_back(device.create_mesh(vertex_buffer, triangle_buffer));
    }

    auto accel = device.create_accel();
    //    auto [vertices, triangle] = get_cube();
    //
    //    Buffer v_buffer = device.create_buffer<float3>(vertices.size());
    //    Buffer t_buffer = device.create_buffer<Triangle>(triangle.size());
    //
    //    Mesh cube = device.create_mesh(v_buffer.view(), t_buffer.view());
    //
    //    stream << v_buffer.upload_sync(vertices.data());
    //    stream << t_buffer.upload_sync(triangle.data());
    //
    //    stream << cube.build_bvh();
    //
    //    Accel accel = device.create_accel();
    //    accel.add_mesh(cube, make_float4x4(1.f));
    //    stream << accel.build_bvh();
    //    stream << synchronize() << commit();
    //
    //    Callable cb = [&]() {
    //        return Var<std::array<float, 10>>();
    //    };

    //    Kernel kernel = [&]() {
    //        Var<Ray> r = make_ray(float3(0,0.1, -5), float3(0,0,1));
    //        Var hit= accel.trace_closest(r);
    ////        Float3 org = r->origin();
    //        Float3 pos = r->direction();
    //        Var<Triangle> tri = t_buffer.read(0);
    //        print("{},{}----------{} {}", hit.prim_id, hit.inst_id, hit->bary.x, hit.bary.y);
    //        print("{}  {}  {}  {}", pos.x, pos.y, pos.z, r->t_max());
    //    };
    //    auto shader = device.compile(kernel);
    //    stream << shader().dispatch(1);
    //    stream << synchronize() << commit();

    return 0;
}