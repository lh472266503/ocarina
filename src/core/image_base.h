//
// Created by Zero on 24/07/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "core/concepts.h"
#include "math/optics.h"

namespace ocarina {

enum struct PixelStorage : uint32_t {
    BYTE1,
    BYTE2,
    BYTE4,

    FLOAT1,
    FLOAT2,
    FLOAT4,

    UNKNOWN
};

enum struct PixelFormat : uint8_t {
    R8U,
    RG8U,
    RGBA8U,
    R32F,
    RG32F,
    RGBA32F,
    UNKNOWN
};

enum struct ImageWrap : uint8_t {
    Repeat,
    Black,
    Clamp
};

namespace detail {

template<typename T>
struct PixelFormatImpl {

    template<typename U>
    static constexpr auto always_false = false;

    static_assert(always_false<T>, "Unsupported type for pixel format.");
};

#define MAKE_PIXEL_FORMAT_OF_TYPE(Type, f)               \
    template<>                                           \
    struct PixelFormatImpl<Type> {                       \
        static constexpr auto format = PixelFormat::f;   \
        static constexpr auto pixel_size = sizeof(Type); \
    };

MAKE_PIXEL_FORMAT_OF_TYPE(uchar, R8U)
MAKE_PIXEL_FORMAT_OF_TYPE(uchar2, RG8U)
MAKE_PIXEL_FORMAT_OF_TYPE(uchar4, RGBA8U)
MAKE_PIXEL_FORMAT_OF_TYPE(float, R32F)
MAKE_PIXEL_FORMAT_OF_TYPE(float2, RG32F)
MAKE_PIXEL_FORMAT_OF_TYPE(float4, RGBA32F)

#undef MAKE_PIXEL_FORMAT_OF_TYPE
}// namespace detail

template<typename T>
using PixelFormatImpl = detail::PixelFormatImpl<T>;

OC_NDSC_INLINE PixelStorage format_to_storage(PixelFormat pixel_format) noexcept {
    switch (pixel_format) {
        case PixelFormat::R8U: return PixelStorage::BYTE1;
        case PixelFormat::RG8U: return PixelStorage::BYTE2;
        case PixelFormat::RGBA8U: return PixelStorage::BYTE4;
        case PixelFormat::R32F: return PixelStorage::FLOAT1;
        case PixelFormat::RG32F: return PixelStorage::FLOAT2;
        case PixelFormat::RGBA32F: return PixelStorage::FLOAT4;
        default: OC_ASSERT(0); return PixelStorage::UNKNOWN;
    }
}

OC_NDSC_INLINE size_t pixel_size(PixelStorage pixel_storage) noexcept {
    switch (pixel_storage) {
        case PixelStorage::BYTE1: return sizeof(uchar);
        case PixelStorage::BYTE2: return sizeof(uchar2);
        case PixelStorage::BYTE4: return sizeof(uchar4);
        case PixelStorage::FLOAT1: return sizeof(float);
        case PixelStorage::FLOAT2: return sizeof(float2);
        case PixelStorage::FLOAT4: return sizeof(float4);
        case PixelStorage::UNKNOWN: break;
    }
    OC_ASSERT(0);
    return 0;
}

OC_NDSC_INLINE size_t pixel_size(PixelFormat pixel_format) noexcept {
    return pixel_size(format_to_storage(pixel_format));
}

OC_NDSC_INLINE bool is_8bit(PixelFormat pixel_format) noexcept {
    return pixel_format == PixelFormat::R8U || pixel_format == PixelFormat::RG8U || pixel_format == PixelFormat::RGBA8U;
}

OC_NDSC_INLINE bool is_32bit(PixelFormat pixel_format) noexcept {
    return pixel_format == PixelFormat::R32F || pixel_format == PixelFormat::RG32F || pixel_format == PixelFormat::RGBA32F;
}

OC_NDSC_INLINE size_t channel_num(PixelStorage pixel_storage) {
    if (pixel_storage == PixelStorage::BYTE1 || pixel_storage == PixelStorage::FLOAT1) { return 1u; }
    if (pixel_storage == PixelStorage::BYTE2 || pixel_storage == PixelStorage::FLOAT2) { return 2u; }
    return 4u;
}

OC_NDSC_INLINE size_t channel_num(PixelFormat pixel_format) noexcept {
    return channel_num(format_to_storage(pixel_format));
}

class ImageBase : public concepts::Noncopyable {
protected:
    PixelFormat _pixel_format{PixelFormat::UNKNOWN};
    uint2 _resolution{};

public:
    ImageBase(PixelFormat pixel_format, uint2 resolution)
        : _pixel_format(pixel_format),
          _resolution(resolution) {}
    ImageBase(ImageBase &&other) noexcept {
        _pixel_format = other._pixel_format;
        _resolution = other._resolution;
    }
    ImageBase() = default;
    ImageBase &operator=(ImageBase &&) = default;
    [[nodiscard]] int channel_num() const { return ::ocarina::channel_num(_pixel_format); }
    [[nodiscard]] uint2 resolution() const { return _resolution; }
    [[nodiscard]] uint width() const { return _resolution.x; }
    [[nodiscard]] uint height() const { return _resolution.y; }
    [[nodiscard]] PixelFormat pixel_format() const { return _pixel_format; }
    [[nodiscard]] size_t pitch_byte_size() const { return _resolution.x * pixel_size(_pixel_format); }
    [[nodiscard]] size_t pixel_num() const { return _resolution.x * _resolution.y; }
    [[nodiscard]] size_t size_in_bytes() const {
        return pixel_size(_pixel_format) * pixel_num() * channel_num();
    }
};

}// namespace ocarina