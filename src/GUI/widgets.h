//
// Created by Zero on 2024/3/16.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "core/util.h"

namespace ocarina {

class Widgets {
public:
    virtual void init() noexcept = 0;
    virtual void push_window(string_view label) noexcept = 0;
    virtual void pop_window() noexcept = 0;

    virtual void text(string_view format, ...) noexcept = 0;
    virtual void check_box(string_view label, bool *val) noexcept = 0;

    virtual void slider_float(string_view label, float *val, float min, float max) noexcept = 0;
    virtual void slider_float2(string_view label, float2 *val, float2 min, float2 max) noexcept = 0;
    virtual void slider_float3(string_view label, float3 *val, float3 min, float3 max) noexcept = 0;
    virtual void slider_float4(string_view label, float4 *val, float4 min, float4 max) noexcept = 0;

    virtual void slider_int(string_view label, int *val, int min, int max) noexcept = 0;
    virtual void slider_int2(string_view label, int2 *val, int2 min, int2 max) noexcept = 0;
    virtual void slider_int3(string_view label, int3 *val, int3 min, int3 max) noexcept = 0;
    virtual void slider_int4(string_view label, int4 *val, int4 min, int4 max) noexcept = 0;

    virtual void color_edit(string_view label, float3 *val) noexcept = 0;
    virtual void color_edit(string_view label, float4 *val) noexcept = 0;

    virtual bool button(string_view label, uint2 size) noexcept = 0;
    virtual bool button(string_view label) noexcept = 0;

    virtual void same_line() noexcept = 0;
    virtual void new_line() noexcept = 0;

    virtual void input_int(string_view label, int *val) noexcept = 0;
    virtual void input_float(string_view label, float *val) noexcept = 0;
};

}// namespace ocarina