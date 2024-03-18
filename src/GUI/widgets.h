//
// Created by Zero on 2024/3/16.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "core/util.h"

namespace ocarina {

enum WindowFlag {
    None = 0,
    MenuBar = 1 << 10
};

class Widgets {
public:
    Widgets() = default;

    template<typename Func>
    void use_window(const char *label, Func &&func) noexcept {
        push_window(label);
        func();
        pop_window();
    }

    virtual void push_window(const char *label) noexcept = 0;
    virtual void push_window(const char *label, WindowFlag flag) noexcept = 0;
    virtual void pop_window() noexcept = 0;

    virtual bool folding_header(const char *label) noexcept = 0;

    virtual bool begin_menu_bar() noexcept = 0;
    virtual bool begin_menu(const char *label) noexcept = 0;
    virtual bool menu_item(const char *label) noexcept = 0;
    virtual void end_menu() noexcept = 0;
    virtual void end_menu_bar() noexcept = 0;

    virtual void text(const char *format, ...) noexcept = 0;
    virtual bool check_box(const char *label, bool *val) noexcept = 0;

    virtual bool slider_float(const char *label, float *val, float min, float max) noexcept = 0;
    virtual bool slider_float2(const char *label, float2 *val, float min, float max) noexcept = 0;
    virtual bool slider_float3(const char *label, float3 *val, float min, float max) noexcept = 0;
    virtual bool slider_float4(const char *label, float4 *val, float min, float max) noexcept = 0;

    virtual bool slider_int(const char *label, int *val, int min, int max) noexcept = 0;
    virtual bool slider_int2(const char *label, int2 *val, int min, int max) noexcept = 0;
    virtual bool slider_int3(const char *label, int3 *val, int min, int max) noexcept = 0;
    virtual bool slider_int4(const char *label, int4 *val, int min, int max) noexcept = 0;

    virtual bool color_edit(const char *label, float3 *val) noexcept = 0;
    virtual bool color_edit(const char *label, float4 *val) noexcept = 0;

    virtual bool button(const char *label, uint2 size) noexcept = 0;
    virtual bool button(const char *label) noexcept = 0;

    virtual void same_line() noexcept = 0;
    virtual void new_line() noexcept = 0;

    virtual bool input_int(const char *label, int *val) noexcept = 0;
    virtual bool input_float(const char *label, float *val) noexcept = 0;

    virtual ~Widgets() = default;
};

}// namespace ocarina