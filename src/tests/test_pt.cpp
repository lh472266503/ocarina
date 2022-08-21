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
#include "gui/window.h"

using namespace ocarina;

struct Material {
    float3 albedo;
    float3 emission;
};

struct Onb {
    float3 tangent;
    float3 binormal;
    float3 normal;
};

// clang-format off
OC_STRUCT(Material, albedo, emission) {};
OC_STRUCT(Onb, tangent, binormal, normal) {
    [[nodiscard]] auto to_world(Var<float3> v) const noexcept {
        return v.x * tangent + v.y * binormal + v.z * normal;
    }
};
// clang-format on

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
    uint2 res = make_uint2(768, 768);

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
        vertices.emplace_back(float3{
            p[i + 0u],
            p[i + 1u],
            p[i + 2u]});
    }
    OC_INFO_FORMAT(
        "Loaded mesh with {} shape(s) and {} vertices.",
        obj_reader.GetShapes().size(), vertices.size());
    auto vertex_buffer = device.create_buffer<float3>(vertices.size());
    stream << vertex_buffer.upload_sync(vertices.data())
           << commit();
    std::vector<Mesh> meshes;
    size_t num = 0;
    std::vector<Triangle> triangles;
    for (auto &&shape : obj_reader.GetShapes()) {
        auto &&t = shape.mesh.indices;
        auto triangle_count = t.size() / 3u;
        OC_INFO_FORMAT(
            "Processing shape '{}' at index {} with {} triangle(s).",
            shape.name, num++, triangle_count);
        for (int i = 0; i < t.size(); i += 3) {
            triangles.emplace_back(t[i].vertex_index, t[i + 1].vertex_index, t[i + 2].vertex_index);
        }
    }
    auto triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    stream << triangle_buffer.upload_sync(triangles.data());
    stream << synchronize() << commit();

    std::vector<Material> materials;
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// floor
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// ceiling
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// back wall
    materials.emplace_back(Material{make_float3(0.14f, 0.45f, 0.091f), make_float3(0.0f)});// right wall
    materials.emplace_back(Material{make_float3(0.63f, 0.065f, 0.05f), make_float3(0.0f)});// left wall
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// short box
    materials.emplace_back(Material{make_float3(0.725f, 0.71f, 0.68f), make_float3(0.0f)});// tall box
    materials.emplace_back(Material{make_float3(0.0f), make_float3(17.0f, 12.0f, 4.0f)});  // light

    auto material_buffer = device.create_buffer<Material>(materials.size());
    stream << material_buffer.upload_sync(materials.data());

    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    stream << mesh.build_bvh() << synchronize() << commit();
    auto accel = device.create_accel();
    accel.add_mesh(mesh, make_float4x4(1.f));
    stream << accel.build_bvh() << synchronize() << commit();

    auto image = ImageIO::pure_color(make_float4(0, 0, 0, 1), ColorSpace::LINEAR, res);

    auto window = context.create_window("display", res);

    window->run([&](double t) {
        window->set_background(image.pixel_ptr<float4>(), res);
    });

    return 0;
}