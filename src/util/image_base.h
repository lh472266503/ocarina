//
// Created by Zero on 24/07/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "core/concepts.h"

namespace ocarina {
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

enum ColorSpace : uint8_t {
    LINEAR,
    SRGB
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

OC_NDSC_INLINE size_t pixel_size(PixelFormat pixel_format) {
    switch (pixel_format) {
        case PixelFormat::R8U:
            return sizeof(uchar);
        case PixelFormat::RG8U:
            return sizeof(uchar2);
        case PixelFormat::RGBA8U:
            return sizeof(uchar4);
        case PixelFormat::R32F:
            return sizeof(float);
        case PixelFormat::RG32F:
            return sizeof(float2);
        case PixelFormat::RGBA32F:
            return sizeof(float4);
        default:
            return 0;
    }
}

OC_NDSC_INLINE bool is_8bit(PixelFormat pixel_format) {
    return pixel_format == PixelFormat::R8U || pixel_format == PixelFormat::RG8U || pixel_format == PixelFormat::RGBA8U;
}

OC_NDSC_INLINE bool is_32bit(PixelFormat pixel_format) {
    return pixel_format == PixelFormat::R32F || pixel_format == PixelFormat::RG32F || pixel_format == PixelFormat::RGBA32F;
}

OC_NDSC_INLINE size_t channel_num(PixelFormat pixel_format) {
    if (pixel_format == PixelFormat::R8U || pixel_format == PixelFormat::R32F) { return 1u; }
    if (pixel_format == PixelFormat::RG8U || pixel_format == PixelFormat::RG32F) { return 2u; }
    return 4u;
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