//
// Created by Zero on 2022/8/16.
//


#include "core/stl.h"
#include "dsl/dsl.h"
#include "util/file_manager.h"
#include "rhi/common.h"
#include <windows.h>
#include "math/base.h"
#include "util/image.h"
#include "rhi/vertex_buffer.h"
#include "rhi/index_buffer.h"
#include "rhi/resources/buffer.h"
#include "GUI/window.h"
#include "framework/renderer.h"
#include "framework/primitive.h"
#include "rhi/descriptor_set.h"

using namespace ocarina;

class TestA
{
public:
    TestA() = default;
    explicit TestA(const char *str) {
        int length = strlen(str);
        _a = new char[length + 1];
        _a[length] = 0;
        memcpy(_a, str, length);
    }
    ~TestA()
    {
        if (_a)
        {
            delete[] _a;
            _a = nullptr;
        }
    }

    TestA(const TestA& other) {
        if (_a != nullptr)
        {
            delete[] _a;
        }

        if (other._a != nullptr)
        {
            int length = strlen(other._a);
            _a = new char[length + 1];
            _a[length] = 0;
            memcpy(_a, other._a, length);
        }
        std::cout << "Copy constuctor." << std::endl;
    }
    TestA(TestA&& rvalue) noexcept
    {
        std::cout << "Move constuctor." << std::endl;
        _a = std::move(rvalue._a);
        rvalue._a = nullptr;
    }
    TestA& operator=(const TestA& other)
    {
        if (_a != nullptr) {
            delete[] _a;
        }

        if (other._a != nullptr) {
            int length = strlen(other._a);
            _a = new char[length + 1];
            _a[length] = 0;
            memcpy(_a, other._a, length);
        }
        std::cout << "Copy assignment constuctor." << std::endl;
        return *this;
    }

    TestA &operator=(TestA &&rvalue) noexcept
    {
        _a = std::move(rvalue._a);
        rvalue._a = nullptr;
        std::cout << "Move assignment constuctor." << std::endl;
        return *this;
    }

public: 
    char* _a = nullptr;
};

struct ExampleStruct {
    char a;
    int b;
    short c;
    float d;
};

class TestRef
{
public:
    TestRef() { ranges.resize(1); }
    std::vector<int> ranges;
};

struct TestContainer
{
    TestContainer()
    {
        m_ref = new TestRef();
    }

    ~TestContainer()
    {
        if (m_ref)
        {
            delete m_ref;
        }
    }

    TestRef *m_ref = nullptr;
};

class TestB
{
public:
    TestB(const TestRef &ref) : m_Ref(ref) {}

    const TestRef& GetTestRef()
    {
        return m_Ref;
    }
private:
    const TestRef &m_Ref;
};

struct PerframeUniformBuffer
{
    float4x4 view_matrix;
    float4x4 projection_matrix;
};

struct PerframeUBOWriter
{
    PerframeUniformBuffer per_frame_uniform_buffer = {};
    std::unique_ptr<Buffer<std::byte>> per_frame_buffer_ = nullptr;
    std::unique_ptr<DescriptorSet> per_frame_descriptor_set_ = nullptr;
};

int main(int argc, char *argv[]) {
    TestContainer *container = new TestContainer();

    TestB testB(*(container->m_ref));

    const TestRef &ref = testB.GetTestRef();

    for (size_t i = 0; i < ref.ranges.size(); ++i)
    {
        std::cout << ref.ranges[i] << std::endl;
    }

    delete container;
    container = nullptr;

    for (size_t i = 0; i < ref.ranges.size(); ++i) {
        std::cout << ref.ranges[i] << std::endl;
    }

    fs::path path(argv[0]);
    FileManager &file_manager = FileManager::instance();

    auto window = file_manager.create_window("display", make_uint2(800, 600), WindowLibrary::SDL3);

    InstanceCreation instanceCreation = {};
    //instanceCreation.instanceExtentions =
    instanceCreation.windowHandle = window->get_window_handle();
    Device device = Device::create_device("vulkan", instanceCreation);

    //Shader
    std::set<string> options;
    handle_ty vertex_shader = device.create_shader_from_file("D:\\github\\Vision\\src\\ocarina\\src\\backends\\vulkan\\builtin\\triangle.vert", ShaderType::VertexShader, options);
    handle_ty pixel_shader = device.create_shader_from_file("D:\\github\\Vision\\src\\ocarina\\src\\backends\\vulkan\\builtin\\triangle.frag", ShaderType::PixelShader, options);


    Primitive triangle;
    //std::vector<Primitive> opaques;
    PipelineState pipeline_state;
    pipeline_state.shaders[0] = vertex_shader;
    pipeline_state.shaders[1] = pixel_shader;
    pipeline_state.blend_state = BlendState::Opaque();
    pipeline_state.raster_state = RasterState::Default();
    pipeline_state.depth_stencil_state = DepthStencilState::Default();

    auto setup_triangle = [&](Primitive& triangle) {
        triangle.set_vertex_shader(vertex_shader);
        triangle.set_pixel_shader(pixel_shader);
        
        VertexBuffer* vertex_buffer = device.create_vertex_buffer();
        float3 positions[3] = { {  1.0f,  1.0f, 0.0f }, { -1.0f,  1.0f, 0.0f }, {  0.0f, -1.0f, 0.0f } };
        vertex_buffer->add_vertex_stream(VertexAttributeType::Enum::Position, 3, sizeof(float3), (const void*)&positions[0]);
        float4 colors[3] = {{1.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f}};
        vertex_buffer->add_vertex_stream(VertexAttributeType::Enum::Color0, 3, sizeof(float4), (const void*)&colors[0]);

        triangle.set_vertex_buffer(vertex_buffer);
        pipeline_state.vertex_buffer = vertex_buffer;

        // Setup indices
        std::vector<uint32_t> indices{ 0, 1, 2 };
        uint32_t indices_count = static_cast<uint32_t>(indices.size());
        uint32_t indices_bytes = indices_count * sizeof(uint32_t);
        IndexBuffer* index_buffer = device.create_index_buffer(indices.data(), indices_bytes);
        triangle.set_index_buffer(index_buffer);
        triangle.set_pipeline_state(pipeline_state);
        //opaques.push_back(triangle);
    };

    auto setup_renderer = [&]() {
        // Setup renderer if needed
        };

    auto release_renderer = [&]() {
    };

    triangle.set_geometry_data_setup(setup_triangle);

    Renderer renderer(&device);

    RenderPassCreation render_pass_creation;
    RenderPass* render_pass = device.create_render_pass(render_pass_creation);
    render_pass->add_draw_call(triangle.get_draw_call_item(&device));
    renderer.add_render_pass(render_pass);

    auto image_io = Image::pure_color(make_float4(1, 0, 0, 1), ColorSpace::LINEAR, make_uint2(500));
    window->set_background(image_io.pixel_ptr<float4>(), make_uint2(800, 600));
    window->run([&](double d) {
        while (!window->should_close())
        {
            Window::WindowLoop win_loop(window.get());
            renderer.render_frame();
        }
    });

    
}