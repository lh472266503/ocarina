//
// Created by Zero on 2024/3/16.
//

#include "widgets.h"

namespace ocarina {

ImGuiWidgets::ImGuiWidgets()
    : Widgets() {
    ImGui::CreateContext();
}

bool ImGuiWidgets::push_window(const char *label) noexcept {
    return ImGui::Begin(label);
}

bool ImGuiWidgets::push_window(const char *label, ocarina::WindowFlag flag) noexcept {
    return ImGui::Begin(label, nullptr, flag);
}

void ImGuiWidgets::pop_window() noexcept {
    ImGui::End();
}

bool ImGuiWidgets::tree_node(const char *label) noexcept {
    return ImGui::TreeNode(label);
}

void ImGuiWidgets::tree_pop() noexcept {
    ImGui::TreePop();
}

bool ImGuiWidgets::folding_header(const char *label) noexcept {
    return ImGui::CollapsingHeader(label);
}

bool ImGuiWidgets::begin_menu_bar() noexcept {
    return ImGui::BeginMenuBar();
}

bool ImGuiWidgets::begin_menu(const char *label) noexcept {
    return ImGui::BeginMenu(label);
}

bool ImGuiWidgets::menu_item(const char *label) noexcept {
    return ImGui::MenuItem(label);
}

void ImGuiWidgets::end_menu() noexcept {
    ImGui::EndMenu();
}

void ImGuiWidgets::end_menu_bar() noexcept {
    ImGui::EndMenuBar();
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
    return ImGui::SliderFloat3(label, reinterpret_cast<float *>(val), min, max);
}

bool ImGuiWidgets::slider_float4(const char *label, float4 *val, float min, float max) noexcept {
    return ImGui::SliderFloat4(label, reinterpret_cast<float *>(val), min, max);
}

bool ImGuiWidgets::slider_int(const char *label, int *val, int min, int max) noexcept {
    return ImGui::SliderInt(label, val, min, max);
}

bool ImGuiWidgets::slider_int2(const char *label, int2 *val, int min, int max) noexcept {
    return ImGui::SliderInt2(label, reinterpret_cast<int *>(val), min, max);
}

bool ImGuiWidgets::slider_int3(const char *label, int3 *val, int min, int max) noexcept {
    return ImGui::SliderInt3(label, reinterpret_cast<int *>(val), min, max);
}

bool ImGuiWidgets::slider_int4(const char *label, int4 *val, int min, int max) noexcept {
    return ImGui::SliderInt4(label, reinterpret_cast<int *>(val), min, max);
}

bool ImGuiWidgets::color_edit(const char *label, float3 *val) noexcept {
    return ImGui::ColorEdit3(label, reinterpret_cast<float *>(val));
}

bool ImGuiWidgets::color_edit(const char *label, float4 *val) noexcept {
    return ImGui::ColorEdit4(label, reinterpret_cast<float *>(val));
}

bool ImGuiWidgets::button(const char *label, uint2 size) noexcept {
    return ImGui::Button(label, ImVec2(size.x, size.y));
}

bool ImGuiWidgets::button(const char *label) noexcept {
    return ImGui::Button(label);
}

void ImGuiWidgets::same_line() noexcept {
    ImGui::SameLine();
}

void ImGuiWidgets::new_line() noexcept {
    ImGui::NewLine();
}

bool ImGuiWidgets::input_int(const char *label, int *val) noexcept {
    return ImGui::InputInt(label, val);
}

bool ImGuiWidgets::input_int(const char *label, int *val, int step, int step_fast) noexcept {
    return ImGui::InputInt(label, val, step, step_fast);
}

bool ImGuiWidgets::input_int2(const char *label, ocarina::int2 *val) noexcept {
    return ImGui::InputInt2(label, reinterpret_cast<int *>(val));
}

bool ImGuiWidgets::input_int3(const char *label, ocarina::int3 *val) noexcept {
    return ImGui::InputInt3(label, reinterpret_cast<int *>(val));
}

bool ImGuiWidgets::input_int4(const char *label, ocarina::int4 *val) noexcept {
    return ImGui::InputInt4(label, reinterpret_cast<int *>(val));
}

bool ImGuiWidgets::input_float(const char *label, float *val) noexcept {
    return ImGui::InputFloat(label, val);
}

bool ImGuiWidgets::input_float(const char *label, float *val, float step, float step_fast) noexcept {
    return ImGui::InputFloat(label, val, step, step_fast);
}

bool ImGuiWidgets::input_float2(const char *label, ocarina::float2 *val) noexcept {
    return ImGui::InputFloat2(label, reinterpret_cast<float *>(val));
}

bool ImGuiWidgets::input_float3(const char *label, ocarina::float3 *val) noexcept {
    return ImGui::InputFloat3(label, reinterpret_cast<float *>(val));
}

bool ImGuiWidgets::input_float4(const char *label, ocarina::float4 *val) noexcept {
    return ImGui::InputFloat4(label, reinterpret_cast<float *>(val));
}

}// namespace ocarina