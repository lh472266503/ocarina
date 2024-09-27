//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "rhi/resources/texture.h"
#include <vulkan/vulkan.h>

namespace ocarina {

    class VulkanDevice;

class VulkanTexture : public Texture::Impl {
private:
    VkImageType imageType_;
    VkImage image_;
    PixelStorage pixel_storage_;
    uint3 res_{};
    VkDevice device_;

public:
    VulkanTexture(VulkanDevice *device, VkImage image, VkImageType imageType, uint3 res, PixelStorage pixel_storage, uint level_num);
    ~VulkanTexture() override;
    void init();
    [[nodiscard]] uint3 resolution() const noexcept override { return res_; }
    [[nodiscard]] handle_ty array_handle() const noexcept override {
        return reinterpret_cast<handle_ty>(image_);
    }
    [[nodiscard]] const handle_ty *array_handle_ptr() const noexcept override {
        return reinterpret_cast<handle_ty *>(image_);
    }
    [[nodiscard]] handle_ty tex_handle() const noexcept override {
        return (handle_ty)image_;
    }
    [[nodiscard]] const void *handle_ptr() const noexcept override {
        return &image_;
    }
    [[nodiscard]] size_t data_size() const noexcept override;
    [[nodiscard]] size_t data_alignment() const noexcept override;
    [[nodiscard]] size_t max_member_size() const noexcept override;
    [[nodiscard]] PixelStorage pixel_storage() const noexcept override { return pixel_storage_; }
};
}// namespace ocarina