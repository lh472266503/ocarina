//
// Created by Zero on 06/08/2022.
//

#include "vulkan_texture.h"
#include "util.h"
#include "vulkan_device.h"

namespace ocarina {

VulkanTexture::VulkanTexture(VulkanDevice *device, VkImage image, VkImageType imageType, uint3 res, PixelStorage pixel_storage, uint level_num)
    : device_(device->logicalDevice()), image_(image), imageType_(imageType), res_(res) {
    init();
}

void VulkanTexture::init() {
    
}

VulkanTexture::~VulkanTexture() {
    
}
size_t VulkanTexture::data_size() const noexcept { return 0; }
size_t VulkanTexture::data_alignment() const noexcept { return 0; }
size_t VulkanTexture::max_member_size() const noexcept { return sizeof(handle_ty); }

}// namespace ocarina