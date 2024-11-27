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
    uint64_t windowHandle = InvalidUI64;
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

enum class GraphicBufferBindFlags {
    None = (0 << 0),                    ///< None
    VertexBuffer = (1 << 0),            ///< Can bind as a vertex buffer
    IndexBuffer = (1 << 1),             ///< Can bind as an index buffer
    ConstantBuffer = (1 << 2),          ///< Can bind as a constant buffer.
    StructuredBuffer = (1 << 3),        ///< Can bind as a structured buffer.
    ByteAddressBuffer = (1 << 4),       ///< Can bind as a byte address buffer.
    FormattedBuffer = (1 << 5),         ///< Can bind as a formatted buffer.
    CopySrc = (1 << 6),                 ///< Can bind as a copy source.
    CopyDst = (1 << 7),                 ///< Can bind as a copy destination.
    ShaderReadWrite = (1 << 8),         ///< Can bind for usage as ReadWrite buffer for any shader stage (UAV on D3D12).
    PredicationExt = (1 << 9),          ///< Can use the buffer for SetPredicationEx calls (must have AdapterFeatures::Predication requested feature enabled)
    AccelerationStructureExt = (1 << 10)///< Can bind as a ray tracing acceleration structure (require AdapterFeatures::RayTracing)
};

enum QueueType
{
    Graphics,///< Graphics queue
    Compute, ///< Compute queue
    Copy,    ///< Copy queue
    NumQueueType,
};

struct BufferCreation {
    //AccessFlags AccessFlags;   ///< Access flags.
    GraphicBufferBindFlags BindFlags;///< Buffer bind flags.
    //BufferFlags Flags;         ///< Buffer flags.
    uint32_t ElementSize;           ///< Element size in bytes.
    uint32_t NumElements;      ///< Number of elements in the resource.
    //Format Format;             ///< Data format.
    //ResourceState InitialState;///< Initial resource state.
    //Extension Extensions;      ///< Extensions
};

//enum CommandListType {
//    Graphics,///< Graphics command list
//    Compute, ///< Compute command list
//    Copy     ///< Copy command list
//};

}// namespace ocarina