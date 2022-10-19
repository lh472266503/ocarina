//
// Created by Zero on 2022/8/16.
//

#pragma once

#include <ext/imgui/glad/glad.h>
#include <GLFW/glfw3.h>
#include "core/window.h"

namespace ocarina {

class GLTexture;
class GLFWContext;

class GLWindow : public Window {
private:
    ocarina::shared_ptr<GLFWContext> _context;
    GLFWwindow *_handle{nullptr};
    mutable ocarina::unique_ptr<GLTexture> _texture;

private:
    void _begin_frame() noexcept;
    void _end_frame() noexcept;

public:
    GLWindow(const char *name, uint2 initial_size, bool resizable = false) noexcept;
    void init(const char *name, uint2 initial_size, bool resizable = false) noexcept;
    GLWindow(GLWindow &&) noexcept = delete;
    GLWindow(const GLWindow &) noexcept = delete;
    GLWindow &operator=(GLWindow &&) noexcept = delete;
    GLWindow &operator=(const GLWindow &) noexcept = delete;
    ~GLWindow() noexcept;
    [[nodiscard]] uint2 size() const noexcept override;
    [[nodiscard]] bool should_close() const noexcept override;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void set_background(const std::array<uint8_t, 4u> *pixels, uint2 size) noexcept override;
    void set_background(const float4 *pixels, uint2 size) noexcept override;
    void set_should_close() noexcept override;
    void set_size(uint2 size) noexcept override;
    void run(UpdateCallback &&draw) noexcept override;
    void run_one_frame(UpdateCallback &&draw) noexcept override;
};
}// namespace ocarina