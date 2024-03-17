//
// Created by Zero on 2024/3/16.
//

#include "widgets.h"

namespace ocarina {

void ImGuiWidgets::init() noexcept {
}
void ImGuiWidgets::push_window(const char *label) noexcept {
    ImGui::Begin(label);
}
void ImGuiWidgets::pop_window() noexcept {
    ImGui::End();
}
void ImGuiWidgets::text(const char *format, ...) noexcept {
    va_list args;
    va_start(args, format);
    ImGui::TextV(format, args);
    va_end(args);
}
bool ImGuiWidgets::check_box(const char *label, bool *val) noexcept {
    return ImGui::Checkbox(label, val);
}
bool ImGuiWidgets::slider_float(const char *label, float *val, float min, float max) noexcept {
    return ImGui::SliderFloat(label, val, min, max);
}
bool ImGuiWidgets::slider_float2(const char *label, float2 *val, float min, float max) noexcept {
    return ImGui::SliderFloat2(label, reinterpret_cast<float *>(val), min, max);
}
bool ImGuiWidgets::slider_float3(const char *label, float3 *val, float min, float max) noexcept {
    return false;
}
bool ImGuiWidgets::slider_float4(const char *label, float4 *val, float min, float max) noexcept {
    return false;
}
bool ImGuiWidgets::slider_int(const char *label, int *val, int min, int max) noexcept {
    return false;
}
bool ImGuiWidgets::slider_int2(const char *label, int2 *val, int min, int max) noexcept {
    return false;
}
bool ImGuiWidgets::slider_int3(const char *label, int3 *val, int min, int max) noexcept {
    return false;
}
bool ImGuiWidgets::slider_int4(const char *label, int4 *val, int min, int max) noexcept {
    return false;
}
bool ImGuiWidgets::color_edit(const char *label, float3 *val) noexcept {
    return false;
}
bool ImGuiWidgets::color_edit(const char *label, float4 *val) noexcept {
    return false;
}
bool ImGuiWidgets::button(const char *label, uint2 size) noexcept {
    return false;
}
bool ImGuiWidgets::button(const char *label) noexcept {
    return false;
}
void ImGuiWidgets::same_line() noexcept {
}
void ImGuiWidgets::new_line() noexcept {
}
bool ImGuiWidgets::input_int(const char *label, int *val) noexcept {
    return false;
}
bool ImGuiWidgets::input_float(const char *label, float *val) noexcept {
    return false;
}

}// namespace ocarina