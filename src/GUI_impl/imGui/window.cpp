//
// Created by Zero on 2022/8/16.
//

#include "window.h"
#include "ext/imgui/imgui_impl_opengl3.h"
#include <ext/imgui/imgui_impl_glfw.h>
#include "core/logging.h"
#include "gl_helper.h"

namespace ocarina {

namespace detail {
[[nodiscard]] auto gl_error_string(GLenum error) noexcept {
    OC_USING_SV;
    switch (error) {
        case GL_INVALID_ENUM: return "invalid enum"sv;
        case GL_INVALID_VALUE: return "invalid value"sv;
        case GL_INVALID_OPERATION: return "invalid operation"sv;
        case GL_OUT_OF_MEMORY: return "out of memory"sv;
        default: return "unknown error"sv;
    }
}
}// namespace detail

class GLFWContext {

public:
    GLFWContext() noexcept {
        glfwSetErrorCallback([](int error, const char *message) noexcept {
            OC_WARNING_FORMAT("GLFW error (code = {}): {}", error, message);
        });
        if (!glfwInit()) { OC_ERROR("Failed to initialize GLFW."); }
    }
    GLFWContext(GLFWContext &&) noexcept = delete;
    GLFWContext(const GLFWContext &) noexcept = delete;
    GLFWContext &operator=(GLFWContext &&) noexcept = delete;
    GLFWContext &operator=(const GLFWContext &) noexcept = delete;
    ~GLFWContext() noexcept { glfwTerminate(); }
    [[nodiscard]] static auto retain() noexcept {
        static std::weak_ptr<GLFWContext> instance;
        if (auto p = instance.lock()) { return p; }
        auto p = ocarina::make_shared<GLFWContext>();
        instance = p;
        return p;
    }
};

#define CHECK_GL(...)                                          \
    [&] {                                                      \
        __VA_ARGS__;                                           \
        if (auto error = glGetError(); error != GL_NO_ERROR) { \
            OC_ERROR_FORMAT(                                   \
                "OpenGL error: {}.",                           \
                ::ocarina::detail::gl_error_string(error));    \
        }                                                      \
    }()

class GLTexture {

private:
    uint32_t _handle{0u};
    bool _is_float4{false};
    uint2 _size{};

public:
    explicit GLTexture() noexcept {
        CHECK_GL(glGenTextures(1, &_handle));
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _handle));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        CHECK_GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    }

    GLTexture(GLTexture &&) noexcept = delete;
    GLTexture(const GLTexture &) noexcept = delete;
    GLTexture &operator=(GLTexture &&) noexcept = delete;
    GLTexture &operator=(const GLTexture &) noexcept = delete;

    ~GLTexture() noexcept { CHECK_GL(glDeleteTextures(1, &_handle)); }

    [[nodiscard]] auto handle() const noexcept { return _handle; }
    [[nodiscard]] auto size() const noexcept { return _size; }

    void load(const uchar4 *pixels, uint2 size) noexcept {
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _handle));
        if (any(_size != size) || _is_float4) {
            _size = size;
            _is_float4 = false;
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size.x, size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }
        CHECK_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _size.x, _size.y, GL_RGBA, GL_UNSIGNED_BYTE, pixels));
    }

    void load(const float4 *pixels, uint2 size) noexcept {
        CHECK_GL(glBindTexture(GL_TEXTURE_2D, _handle));
        if (any(_size != size) || !_is_float4) {
            _size = size;
            _is_float4 = true;
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, nullptr);
        }
        CHECK_GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _size.x, _size.y, GL_RGBA, GL_FLOAT, pixels));
    }
};


void GLWindow::init(const char *name, uint2 initial_size, bool resizable) noexcept {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, resizable);

    // Create window with graphics context
    _handle = glfwCreateWindow(
        static_cast<int>(initial_size.x),
        static_cast<int>(initial_size.y),
        name, nullptr, nullptr);
    if (_handle == nullptr) {
        const char *error = nullptr;
        glfwGetError(&error);
        OC_ERROR_FORMAT("Failed to create GLFW window: {}.", error);
    }
    glfwMakeContextCurrent(_handle);
    glfwSwapInterval(0);// disable vsync

    if (!gladLoadGL()) { OC_ERROR("Failed to initialize GLAD."); }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(_handle, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    glfwSetWindowUserPointer(_handle, this);
    glfwSetMouseButtonCallback(_handle, [](GLFWwindow *window, int button, int action, int mods) noexcept {
        if (ImGui::GetIO().WantCaptureMouse) {// ImGui is handling the mouse
            ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);
        } else {
            auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
            auto x = 0.0;
            auto y = 0.0;
            glfwGetCursorPos(self->handle(), &x, &y);
            if (auto &&cb = self->_mouse_button_callback) {
                cb(button, action, make_float2(static_cast<float>(x), static_cast<float>(y)));
            }
        }
    });
    glfwSetCursorPosCallback(_handle, [](GLFWwindow *window, double x, double y) noexcept {
        auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
        if (auto &&cb = self->_cursor_position_callback) { cb(make_float2(static_cast<float>(x), static_cast<float>(y))); }
    });
    glfwSetWindowSizeCallback(_handle, [](GLFWwindow *window, int width, int height) noexcept {
        auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
        if (auto &&cb = self->_window_size_callback) { cb(make_uint2(width, height)); }
    });
    glfwSetKeyCallback(_handle, [](GLFWwindow *window, int key, int scancode, int action, int mods) noexcept {
        if (ImGui::GetIO().WantCaptureKeyboard) {// ImGui is handling the keyboard
            ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
        } else {
            auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->_key_callback) { cb(key, action); }
        }
    });
    glfwSetScrollCallback(_handle, [](GLFWwindow *window, double dx, double dy) noexcept {
        if (ImGui::GetIO().WantCaptureMouse) {// ImGui is handling the mouse
            ImGui_ImplGlfw_ScrollCallback(window, dx, dy);
        } else {
            auto self = static_cast<GLWindow *>(glfwGetWindowUserPointer(window));
            if (auto &&cb = self->_scroll_callback) {
                cb(make_float2(static_cast<float>(dx), static_cast<float>(dy)));
            }
        }
    });
    glfwSetCharCallback(_handle, ImGui_ImplGlfw_CharCallback);
}

GLWindow::GLWindow(const char *name, uint2 initial_size, bool resizable) noexcept
    : Window(resizable), _context{GLFWContext::retain()} {
    init(name, initial_size, resizable);
}

GLWindow::~GLWindow() noexcept {
    glfwMakeContextCurrent(_handle);
    _texture.reset();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(_handle);
}

uint2 GLWindow::size() const noexcept {
    auto width = 0;
    auto height = 0;
    glfwGetWindowSize(_handle, &width, &height);
    return make_uint2(
        static_cast<uint>(width),
        static_cast<uint>(height));
}

bool GLWindow::should_close() const noexcept {
    return glfwWindowShouldClose(_handle);
}

void GLWindow::set_background(const uchar4 *pixels, uint2 size) noexcept {
    if (_texture == nullptr) { _texture = ocarina::make_unique<GLTexture>(); }
    _texture->load(pixels, size);
}

void GLWindow::set_background(const uchar4 *pixels) noexcept {
    set_background(pixels, size());
}

void GLWindow::set_background(const float4 *pixels, uint2 size) noexcept {
    if (_texture == nullptr) { _texture = ocarina::make_unique<GLTexture>(); }
    _texture->load(pixels, size);
}

void GLWindow::set_background(const float4 *pixels) noexcept {
    set_background(pixels, size());
}

void GLWindow::set_should_close() noexcept {
    glfwSetWindowShouldClose(_handle, true);
}

void GLWindow::_begin_frame() noexcept {
    if (!should_close()) {
        glfwMakeContextCurrent(_handle);
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }
}

void GLWindow::_end_frame() noexcept {
    if (!should_close()) {
        // background
        if (_texture != nullptr) {
            ImVec2 background_size{
                static_cast<float>(_texture->size().x),
                static_cast<float>(_texture->size().y)};
            ImGui::GetBackgroundDrawList()->AddImage(
                reinterpret_cast<ImTextureID>(static_cast<uint64_t>(_texture->handle())), {}, background_size);
        }
        // rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(_handle, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(_clear_color.x, _clear_color.y, _clear_color.z, _clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(_handle);
    }
}

void GLWindow::set_size(uint2 size) noexcept {
    if (_resizable) {
        glfwSetWindowSize(_handle, static_cast<int>(size.x), static_cast<int>(size.y));
    } else {
        OC_WARNING("Ignoring resize on non-resizable window.");
    }
}

}// namespace ocarina

OC_EXPORT_API ocarina::GLWindow *create(const char *name, ocarina::uint2 initial_size, bool resizable) {
    return ocarina::new_with_allocator<ocarina::GLWindow>(name, initial_size, resizable);
}

OC_EXPORT_API void destroy(ocarina::GLWindow *ptr) {
    ocarina::delete_with_allocator(ptr);
}