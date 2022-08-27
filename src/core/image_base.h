//
// Created by Zero on 24/07/2022.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "core/concepts.h"

namespace ocarina {

enum ColorSpace {
    LINEAR,
    SRGB
};

enum EToneMap {
    Gamma,
    Filmic,
    Reinhard,
    Linear
};

enum struct PixelStorage : uint {
    BYTE1,
    BYTE2,
    BYTE4,

    UINT1,
    UINT2,
    UINT4,

    FLOAT1,
    FLOAT2,
    FLOAT4,

    UNKNOWN
};

enum struct ImageWrap : uint8_t {
    Repeat,
    Black,
    Clamp
};

namespace detail {

template<typename T>
struct PixelStorageImpl {

    template<typename U>
    static constexpr auto always_false = false;

    static_assert(always_false<T>, "Unsupported type for pixel format.");
};

#define MAKE_PIXEL_FORMAT_OF_TYPE(Type, f)               \
    template<>                                           \
    struct PixelStorageImpl<Type> {                      \
        static constexpr auto storage = PixelStorage::f; \
        static constexpr auto pixel_size = sizeof(Type); \
    };

MAKE_PIXEL_FORMAT_OF_TYPE(uchar, BYTE1)
MAKE_PIXEL_FORMAT_OF_TYPE(uchar2, BYTE2)
MAKE_PIXEL_FORMAT_OF_TYPE(uchar4, BYTE4)
MAKE_PIXEL_FORMAT_OF_TYPE(float, FLOAT1)
MAKE_PIXEL_FORMAT_OF_TYPE(float2, FLOAT2)
MAKE_PIXEL_FORMAT_OF_TYPE(float4, FLOAT4)

#undef MAKE_PIXEL_FORMAT_OF_TYPE
}// namespace detail

template<typename T>
using PixelStorageImpl = detail::PixelStorageImpl<T>;

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

OC_NDSC_INLINE bool is_8bit(PixelStorage pixel_format) noexcept {
    return pixel_format == PixelStorage::BYTE1 || pixel_format == PixelStorage::BYTE2 || pixel_format == PixelStorage::BYTE4;
}

OC_NDSC_INLINE bool is_32bit(PixelStorage pixel_format) noexcept {
    return pixel_format == PixelStorage::FLOAT1 || pixel_format == PixelStorage::FLOAT2 || pixel_format == PixelStorage::FLOAT4;
}

OC_NDSC_INLINE size_t channel_num(PixelStorage pixel_storage) {
    if (pixel_storage == PixelStorage::BYTE1 || pixel_storage == PixelStorage::FLOAT1) { return 1u; }
    if (pixel_storage == PixelStorage::BYTE2 || pixel_storage == PixelStorage::FLOAT2) { return 2u; }
    return 4u;
}

struct ImageData {
    handle_ty texture{};
    handle_ty surface{};
    PixelStorage pixel_storage{};
};

class ImageBase : public concepts::Noncopyable {
protected:
    PixelStorage _pixel_storage{PixelStorage::UNKNOWN};
    uint2 _resolution{};

public:
    ImageBase(PixelStorage pixel_format, uint2 resolution)
        : _pixel_storage(pixel_format),
          _resolution(resolution) {}
    ImageBase(ImageBase &&other) noexcept {
        _pixel_storage = other._pixel_storage;
        _resolution = other._resolution;
    }
    ImageBase() = default;
    ImageBase &operator=(ImageBase &&) = default;
    [[nodiscard]] int channel_num() const { return ::ocarina::channel_num(_pixel_storage); }
    [[nodiscard]] uint2 resolution() const { return _resolution; }
    [[nodiscard]] uint width() const { return _resolution.x; }
    [[nodiscard]] uint height() const { return _resolution.y; }
    [[nodiscard]] PixelStorage pixel_storage() const { return _pixel_storage; }
    [[nodiscard]] size_t pitch_byte_size() const { return _resolution.x * pixel_size(_pixel_storage); }
    [[nodiscard]] size_t pixel_num() const { return _resolution.x * _resolution.y; }
    [[nodiscard]] size_t size_in_bytes() const {
        return pixel_size(_pixel_storage) * pixel_num() * channel_num();
    }
};

}// namespace ocarina