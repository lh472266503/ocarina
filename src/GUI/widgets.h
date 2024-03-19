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

    virtual bool push_window(const char *label) noexcept = 0;
    virtual bool push_window(const char *label, WindowFlag flag) noexcept = 0;
    virtual void pop_window() noexcept = 0;

    template<typename Func>
    bool use_window(const char *label, WindowFlag flag, Func &&func) noexcept {
        bool show = push_window(label, flag);
        if (show) {
            func();
        }
        pop_window();
        return show;
    }

    template<typename Func>
    bool use_window(const char *label, Func &&func) noexcept {
        return use_window(label, WindowFlag::None, OC_FORWARD(func));
    }

    virtual bool tree_node(const char *label) noexcept = 0;
    virtual void tree_pop() noexcept = 0;

    template<typename Func>
    bool use_tree(const char *label, Func &&func) noexcept {
        bool show = tree_node(label);
        if (show) {
            func();
            tree_pop();
        }
        return show;
    }

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
    virtual bool input_int(const char *label, int *val, int step, int step_fast) noexcept = 0;
    template<typename ...Args>
    bool input_int_limit(const char *label, int *val, int min, int max, Args &&...args) noexcept {
        int old_value = *val;
        bool dirty = input_int(label, val, OC_FORWARD(args)...);
        *val = ocarina::clamp(*val, min, max);
        if (*val == old_value) {
            dirty = false;
        }
        return dirty;
    }
    virtual bool input_int2(const char *label, int2 *val) noexcept = 0;
    virtual bool input_int3(const char *label, int3 *val) noexcept = 0;
    virtual bool input_int4(const char *label, int4 *val) noexcept = 0;

    virtual bool input_float(const char *label, float *val) noexcept = 0;
    virtual bool input_float(const char *label, float *val, float step, float step_fast) noexcept = 0;
    template<typename ...Args>
    bool input_float_limit(const char *label, float *val, float min, float max, Args &&...args) noexcept {
        float old_value = *val;
        bool dirty = input_float(label, val, OC_FORWARD(args)...);
        *val = ocarina::clamp(*val, min, max);
        if (*val == old_value) {
            dirty = false;
        }
        return dirty;
    }
    virtual bool input_float2(const char *label, float2 *val) noexcept = 0;
    virtual bool input_float3(const char *label, float3 *val) noexcept = 0;
    virtual bool input_float4(const char *label, float4 *val) noexcept = 0;

    virtual bool drag_int(const char *label, int *val, float speed, int min, int max) noexcept = 0;
    virtual bool drag_int2(const char *label, int2 *val, float speed, int min, int max) noexcept = 0;
    virtual bool drag_int3(const char *label, int3 *val, float speed, int min, int max) noexcept = 0;
    virtual bool drag_int4(const char *label, int4 *val, float speed, int min, int max) noexcept = 0;

    virtual bool drag_float(const char *label, float *val, float speed, float min, float max) noexcept = 0;
    virtual bool drag_float2(const char *label, float2 *val, float speed, float min, float max) noexcept = 0;
    virtual bool drag_float3(const char *label, float3 *val, float speed, float min, float max) noexcept = 0;
    virtual bool drag_float4(const char *label, float4 *val, float speed, float min, float max) noexcept = 0;

    virtual ~Widgets() = default;
};

}// namespace ocarina