//
// Created by Zero on 06/06/2022.
//

#pragma once
#include "core/logging.h"
#include <vulkan/vulkan.h>

static std::string errorString(VkResult errorCode) {
    switch (errorCode) {
#define STR(r) \
    case VK_##r: return #r
        STR(NOT_READY);
        STR(TIMEOUT);
        STR(EVENT_SET);
        STR(EVENT_RESET);
        STR(INCOMPLETE);
        STR(ERROR_OUT_OF_HOST_MEMORY);
        STR(ERROR_OUT_OF_DEVICE_MEMORY);
        STR(ERROR_INITIALIZATION_FAILED);
        STR(ERROR_DEVICE_LOST);
        STR(ERROR_MEMORY_MAP_FAILED);
        STR(ERROR_LAYER_NOT_PRESENT);
        STR(ERROR_EXTENSION_NOT_PRESENT);
        STR(ERROR_FEATURE_NOT_PRESENT);
        STR(ERROR_INCOMPATIBLE_DRIVER);
        STR(ERROR_TOO_MANY_OBJECTS);
        STR(ERROR_FORMAT_NOT_SUPPORTED);
        STR(ERROR_SURFACE_LOST_KHR);
        STR(ERROR_NATIVE_WINDOW_IN_USE_KHR);
        STR(SUBOPTIMAL_KHR);
        STR(ERROR_OUT_OF_DATE_KHR);
        STR(ERROR_INCOMPATIBLE_DISPLAY_KHR);
        STR(ERROR_VALIDATION_FAILED_EXT);
        STR(ERROR_INVALID_SHADER_NV);
        STR(ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
#undef STR
        default:
            return "UNKNOWN_ERROR";
    }
}

#define VK_CHECK_RESULT(f)                                                                                                                    \
    {                                                                                                                                         \
        VkResult res = (f);                                                                                                                   \
        if (res != VK_SUCCESS) {                                                                                                              \
            OC_ERROR_FORMAT("Vulkan rhi error: {} at {}:{}", errorString(res), __FILE__, __LINE__);                                    \
            assert(res == VK_SUCCESS);                                                                                                        \
        }                                                                                                                                     \
    }


namespace ocarina {
    static VkFormat get_vulkan_format(PixelStorage format, bool srgb)
    {
    switch (format) {
        case ocarina::PixelStorage::BYTE1:
            return srgb ? VK_FORMAT_R8_SRGB : VK_FORMAT_A8_UNORM_KHR;
        case ocarina::PixelStorage::BYTE2:
            return srgb ? VK_FORMAT_R8G8_SRGB : VK_FORMAT_R8G8_UNORM;
            break;
        case ocarina::PixelStorage::BYTE4:
            return srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            break;
        case ocarina::PixelStorage::UINT1:
            return VK_FORMAT_R32_UINT;
            break;
        case ocarina::PixelStorage::UINT2:
            return VK_FORMAT_R32G32_UINT;
            break;
        case ocarina::PixelStorage::UINT4:
            return VK_FORMAT_R32G32B32A32_UINT;
            break;
        case ocarina::PixelStorage::FLOAT1:
            return VK_FORMAT_R32_SFLOAT;
            break;
        case ocarina::PixelStorage::FLOAT2:
            return VK_FORMAT_R32G32_SFLOAT;
            break;
        case ocarina::PixelStorage::FLOAT4:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
            break;
        case ocarina::PixelStorage::UNKNOWN:
            break;
        default:
            break;
            return VK_FORMAT_R8G8B8A8_UNORM;
    }
    }

    static VkColorSpaceKHR colorspace_vulkan(const ColorSpace colorSpace) {
        static const VkColorSpaceKHR colorSpaceToVulkan[] =
            {
                VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,         // sRGB
                VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT,// scRGB
                VK_COLOR_SPACE_HDR10_ST2084_EXT,           // BT2020
            };

        return colorSpaceToVulkan[uint32_t(colorSpace)];
    }
}
