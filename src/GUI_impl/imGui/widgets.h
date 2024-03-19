//
// Created by Zero on 2024/3/16.
//

#pragma once

#include "GUI/widgets.h"
#include "ext/imgui/imgui_impl_opengl3.h"
#include <ext/imgui/imgui_impl_glfw.h>

namespace ocarina {

class ImGuiWidgets : public Widgets {
public:
    ImGuiWidgets();
    bool push_window(const char *label) noexcept override;
    bool push_window(const char *label, WindowFlag flag) noexcept override;
    void pop_window() noexcept override;

    bool tree_node(const char *label) noexcept override;
    void tree_pop() noexcept override;

    bool folding_header(const char *label) noexcept override;

    bool begin_menu_bar() noexcept override;
    bool begin_menu(const char *label) noexcept override;
    bool menu_item(const char *label) noexcept override;
    void end_menu() noexcept override;
    void end_menu_bar() noexcept override;

    void text(const char *format, ...) noexcept override;
    bool check_box(const char *label, bool *val) noexcept override;

    bool slider_float(const char *label, float *val, float min, float max) noexcept override;
    bool slider_float2(const char *label, float2 *val, float min, float max) noexcept override;
    bool slider_float3(const char *label, float3 *val, float min, float max) noexcept override;
    bool slider_float4(const char *label, float4 *val, float min, float max) noexcept override;

    bool slider_int(const char *label, int *val, int min, int max) noexcept override;
    bool slider_int2(const char *label, int2 *val, int min, int max) noexcept override;
    bool slider_int3(const char *label, int3 *val, int min, int max) noexcept override;
    bool slider_int4(const char *label, int4 *val, int min, int max) noexcept override;

    bool color_edit(const char *label, float3 *val) noexcept override;
    bool color_edit(const char *label, float4 *val) noexcept override;

    bool button(const char *label, uint2 size) noexcept override;
    bool button(const char *label) noexcept override;

    void same_line() noexcept override;
    void new_line() noexcept override;

    bool input_int(const char *label, int *val) noexcept override;
    bool input_int(const char *label, int *val, int step, int step_fast) noexcept override;
    bool input_int2(const char *label, ocarina::int2 *val) noexcept override;
    bool input_int3(const char *label, ocarina::int3 *val) noexcept override;
    bool input_int4(const char *label, ocarina::int4 *val) noexcept override;

    bool input_uint(const char *label, uint *val) noexcept override;
    bool input_uint(const char *label, uint *val, uint step, uint step_fast) noexcept override;
    bool input_uint2(const char *label, ocarina::uint2 *val) noexcept override;
    bool input_uint3(const char *label, ocarina::uint3 *val) noexcept override;
    bool input_uint4(const char *label, ocarina::uint4 *val) noexcept override;

    bool input_float(const char *label, float *val) noexcept override;
    bool input_float(const char *label, float *val, float step, float step_fast) noexcept override;
    bool input_float2(const char *label, ocarina::float2 *val) noexcept override;
    bool input_float3(const char *label, ocarina::float3 *val) noexcept override;
    bool input_float4(const char *label, ocarina::float4 *val) noexcept override;

    bool drag_int(const char *label, int *val, float speed, int min, int max) noexcept override;
    bool drag_int2(const char *label, ocarina::int2 *val, float speed, int min, int max) noexcept override;
    bool drag_int3(const char *label, ocarina::int3 *val, float speed, int min, int max) noexcept override;
    bool drag_int4(const char *label, ocarina::int4 *val, float speed, int min, int max) noexcept override;

    bool drag_uint(const char *label, ocarina::uint *val, float speed, ocarina::uint min, ocarina::uint max) noexcept override;
    bool drag_uint2(const char *label, ocarina::uint2 *val, float speed, ocarina::uint min, ocarina::uint max) noexcept override;
    bool drag_uint3(const char *label, ocarina::uint3 *val, float speed, ocarina::uint min, ocarina::uint max) noexcept override;
    bool drag_uint4(const char *label, ocarina::uint4 *val, float speed, ocarina::uint min, ocarina::uint max) noexcept override;

    bool drag_float(const char *label, float *val, float speed, float min, float max) noexcept override;
    bool drag_float2(const char *label, ocarina::float2 *val, float speed, float min, float max) noexcept override;
    bool drag_float3(const char *label, ocarina::float3 *val, float speed, float min, float max) noexcept override;
    bool drag_float4(const char *label, ocarina::float4 *val, float speed, float min, float max) noexcept override;
};

}// namespace ocarina