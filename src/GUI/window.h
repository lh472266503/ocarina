//
// Created by Zero on 19/10/2022.
//

#pragma once

#include "decl.h"
#include "widget_context.h"

namespace ocarina {
class Window {
public:
    using MouseButtonCallback = ocarina::function<void(int /* button */, int /* action */, float2 /* (x, y) */)>;
    using CursorPositionCallback = ocarina::function<void(float2 /* (x, y) */)>;
    using WindowSizeCallback = ocarina::function<void(uint2 /* (width, height) */)>;
    using KeyCallback = ocarina::function<void(int /* key */, int /* action */)>;
    using ScrollCallback = ocarina::function<void(float2 /* (dx, dy) */)>;
    using UpdateCallback = ocarina::function<void(double)>;
    using BeginFrame = ocarina::function<void()>;
    using EndFrame = ocarina::function<void()>;

protected:
    MouseButtonCallback _mouse_button_callback;
    CursorPositionCallback _cursor_position_callback;
    WindowSizeCallback _window_size_callback;
    KeyCallback _key_callback;
    ScrollCallback _scroll_callback;
    BeginFrame _begin_frame_callback;
    EndFrame _end_frame_callback;
    float4 _clear_color{make_float4(0, 0, 0, 0)};
    bool _resizable{false};
    Clock _clock;
    double _dt{};

protected:
    virtual void _begin_frame() noexcept;
    virtual void _end_frame() noexcept;

public:
    explicit Window(bool resizable = false) noexcept;
    virtual void init(const char *name, uint2 initial_size, bool resizable) noexcept = 0;
    Window(Window &&) noexcept = delete;
    Window(const Window &) noexcept = delete;
    Window &operator=(Window &&) noexcept = delete;
    Window &operator=(const Window &) noexcept = delete;
    virtual ~Window() noexcept = default;
    [[nodiscard]] double dt() const noexcept { return _dt; }
    [[nodiscard]] virtual uint2 size() const noexcept = 0;
    [[nodiscard]] virtual bool should_close() const noexcept = 0;
    [[nodiscard]] explicit operator bool() const noexcept { return !should_close(); }
    virtual Window &set_mouse_callback(MouseButtonCallback cb) noexcept;
    virtual Window &set_cursor_position_callback(CursorPositionCallback cb) noexcept;
    virtual Window &set_window_size_callback(WindowSizeCallback cb) noexcept;
    virtual Window &set_key_callback(KeyCallback cb) noexcept;
    virtual Window &set_scroll_callback(ScrollCallback cb) noexcept;
    virtual Window &set_begin_frame_callback(BeginFrame cb) noexcept;
    virtual Window &set_end_frame_callback(EndFrame cb) noexcept;
    virtual void set_background(const uchar4 *pixels, uint2 size) noexcept = 0;
    virtual void set_background(const uchar4 *pixels) noexcept = 0;
    void set_clear_color(float4 color) noexcept { _clear_color = color; }
    virtual void set_background(const float4 *pixels, uint2 size) noexcept = 0;
    virtual void set_background(const float4 *pixels) noexcept = 0;
    virtual void set_should_close() noexcept = 0;
    virtual void set_size(uint2 size) noexcept = 0;
    virtual void run(UpdateCallback &&draw) noexcept;
    virtual void run_one_frame(UpdateCallback &&draw, double dt) noexcept;
    virtual void run_one_frame(UpdateCallback &&draw) noexcept {
        run_one_frame(OC_FORWARD(draw), 0);
    }
};
}