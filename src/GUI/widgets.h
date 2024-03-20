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

    virtual bool push_window(const string &label) noexcept = 0;
    virtual bool push_window(const string &label, WindowFlag flag) noexcept = 0;
    virtual void pop_window() noexcept = 0;

    template<typename Func>
    bool use_window(const string &label, WindowFlag flag, Func &&func) noexcept {
        bool show = push_window(label, flag);
        if (show) {
            func();
        }
        pop_window();
        return show;
    }

    template<typename Func>
    bool use_window(const string &label, Func &&func) noexcept {
        return use_window(label, WindowFlag::None, OC_FORWARD(func));
    }

    virtual bool tree_node(const string &label) noexcept = 0;
    virtual void tree_pop() noexcept = 0;

    template<typename T, typename Func>
    bool use_tree(T &&label, Func &&func) noexcept {
        bool show = tree_node(OC_FORWARD(label));
        if (show) {
            func();
            tree_pop();
        }
        return show;
    }

    virtual bool folding_header(const string &label) noexcept = 0;

    template<typename T, typename Func>
    bool use_folding_header(T &&label, Func &&func) noexcept {
        bool open = folding_header(OC_FORWARD(label));
        if (open) {
            func();
        }
        return open;
    }

    virtual bool begin_main_menu_bar() noexcept = 0;
    virtual void end_main_menu_bar() noexcept = 0;

    virtual bool begin_menu_bar() noexcept = 0;
    virtual bool begin_menu(const string &label) noexcept = 0;
    virtual bool menu_item(const string &label) noexcept = 0;
    virtual void end_menu() noexcept = 0;
    virtual void end_menu_bar() noexcept = 0;

    template<typename Func>
    bool use_main_menu_bar(Func &&func) noexcept {
        bool ret = begin_main_menu_bar();
        if (ret) {
            func();
            end_main_menu_bar();
        }
        return ret;
    }

    template<typename Func>
    bool use_menu_bar(Func &&func) noexcept {
        bool ret = begin_menu_bar();
        if (ret) {
            func();
            end_menu_bar();
        }
        return ret;
    }

    template<typename Func>
    bool use_menu(const string &label, Func &&func) noexcept {
        bool ret = begin_menu(label.c_str());
        if (ret) {
            func();
            end_menu();
        }
        return ret;
    }

    virtual void text(const char *format, ...) noexcept = 0;
    virtual bool check_box(const string &label, bool *val) noexcept = 0;

    virtual bool slider_float(const string &label, float *val, float min, float max) noexcept = 0;
    virtual bool slider_float2(const string &label, float2 *val, float min, float max) noexcept = 0;
    virtual bool slider_float3(const string &label, float3 *val, float min, float max) noexcept = 0;
    virtual bool slider_float4(const string &label, float4 *val, float min, float max) noexcept = 0;

    virtual bool slider_int(const string &label, int *val, int min, int max) noexcept = 0;
    virtual bool slider_int2(const string &label, int2 *val, int min, int max) noexcept = 0;
    virtual bool slider_int3(const string &label, int3 *val, int min, int max) noexcept = 0;
    virtual bool slider_int4(const string &label, int4 *val, int min, int max) noexcept = 0;

    virtual bool color_edit(const string &label, float3 *val) noexcept = 0;
    virtual bool color_edit(const string &label, float4 *val) noexcept = 0;

    bool colorN_edit(const string &label, float *val, uint size) noexcept {
        switch (size) {
            case 3:
                return color_edit(label, reinterpret_cast<float3 *>(val));
            case 4:
                return color_edit(label, reinterpret_cast<float4 *>(val));
            default:
                OC_ERROR("error");
                return false;
        }
    }

    virtual bool button(const string &label, uint2 size) noexcept = 0;
    virtual bool button(const string &label) noexcept = 0;

    template<typename Func>
    bool button_click(const string &label, Func &&func) noexcept {
        bool ret = button(label.c_str());
        if (ret) {
            func();
        }
        return ret;
    }

    virtual void same_line() noexcept = 0;
    virtual void new_line() noexcept = 0;

    virtual bool input_int(const string &label, int *val) noexcept = 0;
    virtual bool input_int(const string &label, int *val, int step, int step_fast) noexcept = 0;
    template<typename... Args>
    bool input_int_limit(const string &label, int *val, int min, int max, Args &&...args) noexcept {
        int old_value = *val;
        bool dirty = input_int(label, val, OC_FORWARD(args)...);
        *val = ocarina::clamp(*val, min, max);
        if (*val == old_value) {
            dirty = false;
        }
        return dirty;
    }
    virtual bool input_int2(const string &label, int2 *val) noexcept = 0;
    virtual bool input_int3(const string &label, int3 *val) noexcept = 0;
    virtual bool input_int4(const string &label, int4 *val) noexcept = 0;

    virtual bool input_uint(const string &label, uint *val) noexcept = 0;
    virtual bool input_uint(const string &label, uint *val, uint step, uint step_fast) noexcept = 0;
    template<typename... Args>
    bool input_uint_limit(const string &label, uint *val, uint min, uint max, Args &&...args) noexcept {
        uint old_value = *val;
        bool dirty = input_uint(label, val, OC_FORWARD(args)...);
        *val = ocarina::clamp(*val, min, max);
        if (*val == old_value) {
            dirty = false;
        }
        return dirty;
    }
    virtual bool input_uint2(const string &label, uint2 *val) noexcept = 0;
    virtual bool input_uint3(const string &label, uint3 *val) noexcept = 0;
    virtual bool input_uint4(const string &label, uint4 *val) noexcept = 0;

    virtual bool input_float(const string &label, float *val) noexcept = 0;
    virtual bool input_float(const string &label, float *val, float step, float step_fast) noexcept = 0;
    template<typename... Args>
    bool input_float_limit(const string &label, float *val, float min, float max, Args &&...args) noexcept {
        float old_value = *val;
        bool dirty = input_float(label, val, OC_FORWARD(args)...);
        *val = ocarina::clamp(*val, min, max);
        if (*val == old_value) {
            dirty = false;
        }
        return dirty;
    }
    virtual bool input_float2(const string &label, float2 *val) noexcept = 0;
    virtual bool input_float3(const string &label, float3 *val) noexcept = 0;
    virtual bool input_float4(const string &label, float4 *val) noexcept = 0;

    bool input_floatN(const string &label, float *val, uint size) noexcept {
        switch (size) {
            case 1:
                return input_float(label, val);
            case 2:
                return input_float2(label, reinterpret_cast<float2 *>(val));
            case 3:
                return input_float3(label, reinterpret_cast<float3 *>(val));
            case 4:
                return input_float4(label, reinterpret_cast<float4 *>(val));
            default:
                OC_ERROR("error");
                break;
        }
        return false;
    }

    virtual bool drag_int(const string &label, int *val, float speed, int min, int max) noexcept = 0;
    virtual bool drag_int2(const string &label, int2 *val, float speed, int min, int max) noexcept = 0;
    virtual bool drag_int3(const string &label, int3 *val, float speed, int min, int max) noexcept = 0;
    virtual bool drag_int4(const string &label, int4 *val, float speed, int min, int max) noexcept = 0;

    virtual bool drag_uint(const string &label, uint *val, float speed, uint min, uint max) noexcept = 0;
    virtual bool drag_uint2(const string &label, uint2 *val, float speed, uint min, uint max) noexcept = 0;
    virtual bool drag_uint3(const string &label, uint3 *val, float speed, uint min, uint max) noexcept = 0;
    virtual bool drag_uint4(const string &label, uint4 *val, float speed, uint min, uint max) noexcept = 0;

    virtual bool drag_float(const string &label, float *val, float speed, float min, float max) noexcept = 0;
    virtual bool drag_float2(const string &label, float2 *val, float speed, float min, float max) noexcept = 0;
    virtual bool drag_float3(const string &label, float3 *val, float speed, float min, float max) noexcept = 0;
    virtual bool drag_float4(const string &label, float4 *val, float speed, float min, float max) noexcept = 0;

    virtual ~Widgets() = default;
};

}// namespace ocarina