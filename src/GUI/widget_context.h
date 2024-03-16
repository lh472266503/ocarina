//
// Created by Zero on 2024/3/16.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "core/util.h"

namespace ocarina {

class WidgetContext {
public:
    virtual void init() noexcept = 0;
    virtual void push_window(string_view label) noexcept = 0;
    virtual void pop_window() noexcept = 0;
    virtual void text(string_view format, ...) noexcept = 0;
    [[nodiscard]] virtual bool check_box(string_view label) noexcept = 0;

    [[nodiscard]] virtual float slider_float(string_view label, float min, float max) noexcept = 0;
    [[nodiscard]] virtual float2 slider_float2(string_view label, float2 min, float2 max) noexcept = 0;
    [[nodiscard]] virtual float3 slider_float3(string_view label, float3 min, float3 max) noexcept = 0;
    [[nodiscard]] virtual float4 slider_float4(string_view label, float4 min, float4 max) noexcept = 0;

    [[nodiscard]] virtual int slider_int(string_view label, int min, int max) noexcept = 0;
    [[nodiscard]] virtual int2 slider_int2(string_view label, int2 min, int2 max) noexcept = 0;
    [[nodiscard]] virtual int3 slider_int3(string_view label, int3 min, int3 max) noexcept = 0;
    [[nodiscard]] virtual int4 slider_int4(string_view label, int4 min, int4 max) noexcept = 0;

    [[nodiscard]] virtual float3 color_edit(string_view label) noexcept = 0;

    [[nodiscard]] virtual bool button(string_view label, uint2 size) noexcept = 0;
    [[nodiscard]] virtual bool button(string_view label) noexcept = 0;

    virtual void same_line() noexcept = 0;
    virtual void new_line() noexcept = 0;

    virtual int input_int(string_view label) noexcept = 0;
};

}// namespace ocarina