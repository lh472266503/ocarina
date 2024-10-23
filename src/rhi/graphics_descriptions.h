//
// Created by Zero on 2022/8/10.
//

#pragma once

#include "core/stl.h"
#include "core/image_base.h"

namespace ocarina {

struct InstanceCreation {
    const char *applicationName;
    bool validation = false;
    std::vector<const char *> instanceExtentions;
    uint32_t windowHandle = -1;
};

struct SwapChainCreation {
    PixelStorage format = PixelStorage::BYTE4; ///< Back buffers format.
    ColorSpace colorSpace = ColorSpace::SRGB;   ///< Back buffers color space.
    uint32_t bufferCount = 1;         ///< Back buffers count.
    uint32_t width = 800;          ///< Back buffers width.
    uint32_t height = 600;              ///< Back buffers height.
    uint32_t refreshRate = 60;    ///< Refresh rate in Hz.
    bool vsync = false;///< 0 - The presentation occurs immediately, there is no synchronization. 1 through 4 - Synchronize presentation after the nth vertical blank.
    bool isWindowed = true;   //.
};

enum QueueType
{
    Graphics,///< Graphics queue
    Compute, ///< Compute queue
    Copy,    ///< Copy queue
    NumQueueType,
};

//enum CommandListType {
//    Graphics,///< Graphics command list
//    Compute, ///< Compute command list
//    Copy     ///< Copy command list
//};

}// namespace ocarina