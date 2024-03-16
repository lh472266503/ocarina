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
    virtual void push_window(const char *label) noexcept = 0;
    virtual void pop_window() noexcept = 0;

    virtual void text(const char *format, ...) noexcept = 0;
    virtual void check_box(const char *label, bool *val) noexcept = 0;

    virtual void slider_float(const char *label, float *val, float min, float max) noexcept = 0;
    virtual void slider_float2(const char *label, float2 *val, float2 min, float2 max) noexcept = 0;
    virtual void slider_float3(const char *label, float3 *val, float3 min, float3 max) noexcept = 0;
    virtual void slider_float4(const char *label, float4 *val, float4 min, float4 max) noexcept = 0;

    virtual void slider_int(const char *label, int *val, int min, int max) noexcept = 0;
    virtual void slider_int2(const char *label, int2 *val, int2 min, int2 max) noexcept = 0;
    virtual void slider_int3(const char *label, int3 *val, int3 min, int3 max) noexcept = 0;
    virtual void slider_int4(const char *label, int4 *val, int4 min, int4 max) noexcept = 0;

    virtual void color_edit(const char *label, float3 *val) noexcept = 0;
    virtual void color_edit(const char *label, float4 *val) noexcept = 0;

    virtual bool button(const char *label, uint2 size) noexcept = 0;
    virtual bool button(const char *label) noexcept = 0;

    virtual void same_line() noexcept = 0;
    virtual void new_line() noexcept = 0;

    virtual void input_int(const char *label, int *val) noexcept = 0;
    virtual void input_float(const char *label, float *val) noexcept = 0;
};

}// namespace ocarina