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

enum class ShaderType {
    VertexShader,           ///< Vertex shader.
    PixelShader,            ///< Pixel shader.
    GeometryShader,         ///< Geometry shader.
    ComputeShader,          ///< Compute shader.
    MeshShader,             ///< Mesh shader.
    NumShaderType,          ///< Number of shader types.
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

//! blending operator function
enum class BlendOperator : uint8_t {
    ADD,             //!< the fragment is added to the color buffer
    SUBTRACT,        //!< the fragment is subtracted from the color buffer
    REVERSE_SUBTRACT,//!< the color buffer is subtracted from the fragment
    MIN,             //!< the min between the fragment and color buffer
    MAX              //!< the max between the fragment and color buffer
};

//! blending function
enum class BlendFunction : uint8_t {
    ZERO,               //!< f(src, dst) = 0
    ONE,                //!< f(src, dst) = 1
    SRC_COLOR,          //!< f(src, dst) = src
    ONE_MINUS_SRC_COLOR,//!< f(src, dst) = 1-src
    DST_COLOR,          //!< f(src, dst) = dst
    ONE_MINUS_DST_COLOR,//!< f(src, dst) = 1-dst
    SRC_ALPHA,          //!< f(src, dst) = src.a
    ONE_MINUS_SRC_ALPHA,//!< f(src, dst) = 1-src.a
    DST_ALPHA,          //!< f(src, dst) = dst.a
    ONE_MINUS_DST_ALPHA,//!< f(src, dst) = 1-dst.a
    SRC_ALPHA_SATURATE  //!< f(src, dst) = (1,1,1) * min(src.a, 1 - dst.a), 1
};

//! stencil operation
enum class StencilOperation : uint8_t {
    KEEP,     //!< Keeps the current value.
    ZERO,     //!< Sets the value to 0.
    REPLACE,  //!< Sets the value to the stencil reference value.
    INCR,     //!< Increments the current value. Clamps to the maximum representable unsigned value.
    INCR_WRAP,//!< Increments the current value. Wraps value to zero when incrementing the maximum representable unsigned value.
    DECR,     //!< Decrements the current value. Clamps to 0.
    DECR_WRAP,//!< Decrements the current value. Wraps value to the maximum representable unsigned value when decrementing a value of zero.
    INVERT,   //!< Bitwise inverts the current value.
};

//! comparison function for the depth / stencil sampler
enum class SamplerCompareFunc : uint8_t {
    // don't change the enums values
    LE = 0,//!< Less or equal
    GE,    //!< Greater or equal
    L,     //!< Strictly less than
    G,     //!< Strictly greater than
    E,     //!< Equal
    NE,    //!< Not equal
    A,     //!< Always. Depth / stencil testing is deactivated.
    N      //!< Never. The depth / stencil test always fails.
};

enum class PrimitiveType : uint8_t
{
    POINTS = 0,       //!< points
    LINES = 1,        //!< lines
    LINE_STRIP = 3,   //!< line strip
    TRIANGLES = 4,    //!< triangles
    TRIANGLE_STRIP = 5//!< triangle strip
};

//enum CommandListType {
//    Graphics,///< Graphics command list
//    Compute, ///< Compute command list
//    Copy     ///< Copy command list
//};

}// namespace ocarina