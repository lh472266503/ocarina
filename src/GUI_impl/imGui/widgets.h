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
    void push_window(const char *label) noexcept override;
    void pop_window() noexcept override;

    bool folding_header(const char *label) noexcept override;

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
    bool input_float(const char *label, float *val) noexcept override;
};

}// namespace ocarina