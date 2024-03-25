//
// Created by Zero on 2022/8/16.
//

#pragma once

#include <ext/imgui/glad/glad.h>
#include <GLFW/glfw3.h>
#include "GUI/window.h"
#include "widgets.h"
#include "gl_helper.h"

namespace ocarina {

class GLTexture;
class GLFWContext;

class GLWindow : public Window {
private:
    ocarina::shared_ptr<GLFWContext> _context;
    GLFWwindow *_handle{nullptr};
    mutable ocarina::unique_ptr<GLTexture> _texture;

private:
    void _begin_frame() noexcept override;
    void _end_frame() noexcept override;

public:
    GLWindow(const char *name, uint2 initial_size, bool resizable = false) noexcept;
    void init(const char *name, uint2 initial_size, bool resizable) noexcept override;
    void init_widgets() noexcept override;
    GLWindow(GLWindow &&) noexcept = delete;
    GLWindow(const GLWindow &) noexcept = delete;
    GLWindow &operator=(GLWindow &&) noexcept = delete;
    GLWindow &operator=(const GLWindow &) noexcept = delete;
    ~GLWindow() noexcept override;
    [[nodiscard]] uint2 size() const noexcept override;
    [[nodiscard]] bool should_close() const noexcept override;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void set_background(const uchar4 *pixels, uint2 size) noexcept override;
    void set_background(const float4 *pixels, uint2 size) noexcept override;
    template<typename T>
    void bind_buffer(uint &buffer, uint2 res) noexcept {
        if (buffer == 0) {
            CHECK_GL(glGenBuffers(1, addressof(buffer)));
            CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, buffer));
            CHECK_GL(glBufferData(GL_ARRAY_BUFFER, sizeof(T) * res.x * res.y,
                                  nullptr, GL_STREAM_DRAW));
            CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, 0u));
        }
        CHECK_GL(glBindBuffer(GL_ARRAY_BUFFER, buffer));
    }
    void set_background(const Buffer<ocarina::float4> &buffer, ocarina::uint2 size) noexcept override;
    void set_should_close() noexcept override;
    void set_size(uint2 size) noexcept override;
};
}// namespace ocarina