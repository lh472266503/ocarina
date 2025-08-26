//
// Created by Zero on 06/08/2022.
//

#include "vulkan_texture.h"
#include "util.h"
#include "vulkan_device.h"
#include "vulkan_driver.h"
#include "core/image.h"
#include "vulkan_buffer.h"

namespace ocarina {

VulkanTexture::VulkanTexture(VulkanDevice *device, Image *image, VkMemoryPropertyFlags memory_property_flags)
    : device_(device), image_resource_(image) {
    init(memory_property_flags);
}

void VulkanTexture::init(VkMemoryPropertyFlags memory_property_flags) {
    res_.x = image_resource_->resolution().x;
    res_.y = image_resource_->resolution().y;
    image_format_ = get_vulkan_format(image_resource_->pixel_storage(), false);

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = image_format_;
    image_info.mipLevels = mip_levels_;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.extent = {static_cast<uint32_t>(res_.x), static_cast<uint32_t>(res_.y), 1};
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    VK_CHECK_RESULT(vkCreateImage(device_->logicalDevice(), &image_info, nullptr, &image_));

    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device_->logicalDevice(), image_, &mem_requirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = mem_requirements.size;
    allocInfo.memoryTypeIndex = device_->get_memory_type(mem_requirements.memoryTypeBits, memory_property_flags);

    VK_CHECK_RESULT(vkAllocateMemory(device_->logicalDevice(), &allocInfo, nullptr, &image_memory_));

    VK_CHECK_RESULT(vkBindImageMemory(device_->logicalDevice(), image_, image_memory_, 0));

    if (image_resource_)
    {
        load_cpu_data(image_resource_);
    }
}

void VulkanTexture::load_cpu_data(Image *image) {
    VulkanBuffer staging_buffer(device_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                image->size_in_bytes(), image->pixel_ptr());

    VulkanDriver::instance().copy_image(&staging_buffer, this);
}

VulkanTexture::~VulkanTexture() {
    if (image_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_->logicalDevice(), image_memory_, nullptr);
    }
    if (image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device_->logicalDevice(), image_, nullptr);
    }
}
size_t VulkanTexture::data_size() const noexcept { return 0; }
size_t VulkanTexture::data_alignment() const noexcept { return 0; }
size_t VulkanTexture::max_member_size() const noexcept { return sizeof(handle_ty); }

}// namespace ocarina