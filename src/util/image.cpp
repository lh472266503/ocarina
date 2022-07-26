//
// Created by Zero on 26/07/2022.
//

#include "image.h"

namespace ocarina {

Image::Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res, const fs::path &path)
    : ImageBase(pixel_format, res),
      _path(path) {
    _pixel.reset(pixel);
}

Image::Image(PixelFormat pixel_format, const std::byte *pixel, uint2 res)
    : ImageBase(pixel_format, res) {
    _pixel.reset(pixel);
}

Image::Image(Image &&other) noexcept
    : ImageBase(other._pixel_format, other._resolution) {
    _pixel = move(other._pixel);
}

Image &Image::operator=(Image &&rhs) noexcept {
    (*(ImageBase *)this) = std::forward<ImageBase>(rhs);
    std::swap(this->_pixel, rhs._pixel);
    return *this;
}

Image Image::pure_color(float4 color, ColorSpace color_space, uint2 res) {
    auto pixel_count = res.x * res.y;
    auto pixel_size = PixelFormatImpl<float4>::pixel_size * pixel_count;
    auto pixel = allocate<std::byte>(pixel_size);
    auto dest = (float4 *)pixel;
    if (color_space == ColorSpace::LINEAR) {
        for (auto i = 0; i < pixel_count; ++i) {
            dest[i] = color;
        }
    } else {
        for (auto i = 0; i < pixel_count; ++i) {
            //            dest[i] = Spectrum::srgb_to_linear(color);
        }
    }
    return {PixelFormat::RGBA32F, pixel, res};
}
}// namespace ocarina