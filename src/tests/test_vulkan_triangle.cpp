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
#include "dsl/dsl.h"
#include "GUI/window.h"
#include "util/image.h"

using namespace ocarina;

class TestA
{
public:
    TestA() : _a(0) {}
    TestA(int v) : _a(v) {}
    TestA(const TestA& other) {
        _a = other._a;
        std::cout << "Copy constuctor." << std::endl;
    }
    TestA(const TestA&& rvalue)
    {
        std::cout << "Move constuctor." << std::endl;
        _a = std::move(rvalue)._a;
    }
    TestA& operator=(const TestA& other)
    {
        _a = other._a;
        std::cout << "Copy assignment constuctor." << std::endl;
        return *this;
    }

    TestA &operator=(const TestA &&rvalue) {
        _a = std::move(rvalue)._a;
        std::cout << "Move assignment constuctor." << std::endl;
        return *this;
    }

private: 
    int _a;

public:
    static TestA PlusTestAMove(const TestA& left, const TestA& right)
    {
        TestA ret(left._a + right._a);
        return std::move(ret);
    }
};


int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    FileManager &file_manager = FileManager::instance();

    auto window = file_manager.create_window("display", make_uint2(800, 600), WindowLibrary::SDL3);

    InstanceCreation instanceCreation = {};
    //instanceCreation.instanceExtentions =
    instanceCreation.windowHandle = window->get_window_handle();
    Device device = Device::create_device("vulkan", instanceCreation);

    //Shader
    handle_ty vertex_shader = device.create_shader_from_file("D:\\github\\Vision\\src\\ocarina\\src\\backends\\vulkan\\builtin\\triangle.vert", ShaderType::VertexShader);

    auto image_io = Image::pure_color(make_float4(1, 0, 0, 1), ColorSpace::LINEAR, make_uint2(500));
    window->run([&](double d) {
        window->set_background(image_io.pixel_ptr<float4>(), make_uint2(800, 600));
    });
}