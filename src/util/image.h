//
// Created by Zero on 26/07/2022.
//

#pragma once

#include "image_base.h"

namespace ocarina {
class Image : public ImageBase {
private:
    fs::path _path;
    std::unique_ptr<const std::byte[]> _pixel;

public:
    Image() = default;
    Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res, const fs::path &path);
    Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res);
    Image(Image &&other) noexcept;
    Image(const Image &other) = delete;
    Image &operator=(const Image &other) = delete;
    Image &operator=(Image &&other) noexcept;
    template<typename T = std::byte>
    const T *pixel_ptr() const { return reinterpret_cast<const T *>(_pixel.get()); }
    template<typename T = std::byte>
    T *pixel_ptr() { return reinterpret_cast<T *>(const_cast<std::byte *>(_pixel.get())); }
    [[nodiscard]] static Image pure_color(float4 color, ColorSpace color_space, uint2 res = make_uint2(1u));
    [[nodiscard]] static Image load(const fs::path &fn, ColorSpace color_space, float3 scale = make_float3(1.f));
    [[nodiscard]] static Image load_hdr(const fs::path &fn, ColorSpace color_space, float3 scale = make_float3(1.f));
    [[nodiscard]] static Image load_exr(const fs::path &fn, ColorSpace color_space, float3 scale = make_float3(1.f));
    [[nodiscard]] static Image create_empty(PixelFormat pixel_format, uint2 res) {
        size_t size_in_bytes = pixel_size(pixel_format) * res.x * res.y;
        auto pixel = allocate<std::byte>(size_in_bytes);
        return {pixel_format, pixel, res};
    }
};
}// namespace ocarina