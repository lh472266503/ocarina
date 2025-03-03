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


int main(int argc, char *argv[]) {
    size_t structSize = sizeof(ExampleStruct);

    std::vector<TestA> test;
    {
        TestA a("test right value");
        //TestA a = "test right value";   compile error because we declare the constructor as explicit
        test.emplace_back(std::move(a));
    }

    fs::path path(argv[0]);
    FileManager &file_manager = FileManager::instance();

    auto window = file_manager.create_window("display", make_uint2(800, 600), WindowLibrary::SDL3);

    InstanceCreation instanceCreation = {};
    //instanceCreation.instanceExtentions =
    instanceCreation.windowHandle = window->get_window_handle();
    Device device = Device::create_device("vulkan", instanceCreation);

    //Shader
    handle_ty vertex_shader = device.create_shader_from_file("D:\\github\\Vision\\src\\ocarina\\src\\backends\\vulkan\\builtin\\triangle.vert", ShaderType::VertexShader);
    handle_ty pixel_shader = device.create_shader_from_file("D:\\github\\Vision\\src\\ocarina\\src\\backends\\vulkan\\builtin\\triangle.frag", ShaderType::PixelShader);

    auto image_io = Image::pure_color(make_float4(1, 0, 0, 1), ColorSpace::LINEAR, make_uint2(500));
    window->run([&](double d) {
        window->set_background(image_io.pixel_ptr<float4>(), make_uint2(800, 600));
    });
}