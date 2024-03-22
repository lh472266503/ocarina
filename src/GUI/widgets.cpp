//
// Created by Zero on 2024/3/22.
//

#include "widgets.h"

namespace ocarina {

bool Widgets::slider_floatN(const std::string &label, float *val, ocarina::uint size, float min, float max) noexcept {
    switch (size) {
        case 1:
            return slider_float(label, val, min, max);
        case 2:
            return slider_float2(label, reinterpret_cast<float2 *>(val), min, max);
        case 3:
            return slider_float3(label, reinterpret_cast<float3 *>(val), min, max);
        case 4:
            return slider_float4(label, reinterpret_cast<float4 *>(val), min, max);
        default:
            OC_ERROR("error");
            break;
    }
    return false;
}

bool Widgets::colorN_edit(const std::string &label, float *val, ocarina::uint size) noexcept {
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

bool Widgets::input_floatN(const std::string &label, float *val, ocarina::uint size) noexcept {
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
}// namespace ocarina