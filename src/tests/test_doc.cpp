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
OC_STRUCT(Triple, i, j, k){

    [[nodiscard]] Uint sum() const noexcept {
        return i + j + k;
    }
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
    ResourceArray resource_array = device.create_resource_array();
    uint v_idx = resource_array.emplace(vert);
    uint t_idx = resource_array.emplace(tri);

    /// upload buffer and texture handle to device memory
    stream << resource_array->upload_buffer_handles() << synchronize();
    stream << resource_array->upload_texture_handles() << synchronize();

    stream << vert.upload(vertices.data())
           << tri.upload(triangles.data());

    Kernel kernel = [&](Var<Triple> triple, BufferVar<Triple> triangle, Var<ResourceArray> ra) {
        $info("triple   {} {} {}", triple.i, triple.j, triple.k);

        Var t = triangle.read(dispatch_id());
        $info("triple  index {} : i = {}, j = {}, k = {},  sum: {} ",dispatch_id(), t.i, t.j, t.k, t->sum());

        $info("vert from capture {} {} {}", vert.read(dispatch_id()));
        $info("vert from capture resource array {} {} {}", resource_array.buffer<float3>(v_idx).read(dispatch_id()));
        $info("vert from ra {} {} {}", ra.buffer<float3>(v_idx).read(dispatch_id()));

        /// execute if thread idx in debug range
        $debugger_execute {
            $warn_with_location("this thread idx is in debug range {} {} {}", vert.read(dispatch_id()));
        };
    };
    Triple triple1{1,2,3};

    /// set debug range
    Debugger::instance().set_lower(make_uint2(0));
    Debugger::instance().set_upper(make_uint2(1));
    auto shader = device.compile(kernel, "test desc");
    stream << Debugger::instance().upload();
    stream << shader(triple1, tri, resource_array).dispatch(2,3)
           /// explict retrieve log
           << Printer::instance().retrieve()
           << synchronize() << commit();
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
    Printer::instance().init(device);
    Debugger::instance().init(device);

    /// create rtx context if need
    device.init_rtx();
    test_compute_shader(device, stream);

    return 0;
}