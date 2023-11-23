//
// Created by Zero on 2023/11/23.
//

#include "util/image_io.h"
#include "core/stl.h"
#include "dsl/common.h"
#include "rhi/common.h"
#include "math/base.h"

using namespace ocarina;

struct Triple {
    uint i{}, j{}, k{};
    Triple(uint i, uint j, uint k) : i(i), j(j), k(k) {}
    Triple() = default;
};
OC_STRUCT(Triple, i, j, k){};

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

    stream << vert.upload(vertices.data())
           << tri.upload(triangles.data());

    Kernel kernel = [&](Var<Triple> triple, BufferVar<Triple> triangle) {
        $info("triple   {} {} {}", triple.i, triple.j, triple.k);

        Var t = triangle.read(dispatch_id());
        $info("triple  index {} :  {} {} {}",dispatch_id(), t.i, t.j, t.k);

        $info("vert {} {} {}", vert.read(dispatch_id()));

    };
    Triple triple1{1,2,3};
    auto shader = device.compile(kernel, "test desc");
    stream << shader(triple1, tri).dispatch(6)
           << Printer::instance().retrieve()
           << synchronize() << commit();
}

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    Context context(path.parent_path());
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();
    Printer::instance().init(device);

    /// create rtx context if need
    device.init_rtx();
    test_compute_shader(device, stream);

    return 0;
}