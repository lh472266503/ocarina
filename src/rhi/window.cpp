//
// Created by Zero on 19/10/2022.
//

#include "window.h"

namespace ocarina {

Window::Window(bool resizable) noexcept
    : _resizable{resizable} {}

Window &Window::set_mouse_callback(Window::MouseButtonCallback cb) noexcept {
    _mouse_button_callback = std::move(cb);
    return *this;
}

Window &Window::set_cursor_position_callback(Window::CursorPositionCallback cb) noexcept {
    _cursor_position_callback = std::move(cb);
    return *this;
}

Window &Window::set_window_size_callback(Window::WindowSizeCallback cb) noexcept {
    _window_size_callback = std::move(cb);
    return *this;
}

Window &Window::set_key_callback(Window::KeyCallback cb) noexcept {
    _key_callback = std::move(cb);
    return *this;
}

Window &Window::set_scroll_callback(Window::ScrollCallback cb) noexcept {
    _scroll_callback = std::move(cb);
    return *this;
}

void Window::run_one_frame(Window::UpdateCallback &&draw) noexcept {
    _begin_frame();
    draw(0);
    _end_frame();
}

void Window::run(Window::UpdateCallback &&draw) noexcept {
    while (!should_close()) {
        run_one_frame(OC_FORWARD(draw));
    }
}

}// namespace ocarina