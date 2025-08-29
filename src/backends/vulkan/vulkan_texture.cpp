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

VulkanTexture::VulkanTexture(VulkanDevice *device, Image *image, const TextureViewCreation &texture_view)
    : device_(device) {
    init(image, texture_view);
}

void VulkanTexture::init(Image *image, const TextureViewCreation &texture_view) {
    res_.x = image->resolution().x;
    res_.y = image->resolution().y;
    image_format_ = get_vulkan_format(image->pixel_storage(), false);

    uint32_t max_mip_levels = static_cast<uint32_t>(std::floor(std::log2(std::max(res_.x, res_.y)))) + 1;
    mip_levels_ = texture_view.mip_level_count == 0 ? max_mip_levels : std::min(max_mip_levels, texture_view.mip_level_count);

    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (mip_levels_ > 1)
    {
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = image_format_;
    image_info.mipLevels = mip_levels_;
    image_info.arrayLayers = texture_view.array_layer_count;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image_info.extent = {static_cast<uint32_t>(res_.x), static_cast<uint32_t>(res_.y), 1};
    image_info.usage = usage;

    VK_CHECK_RESULT(vkCreateImage(device_->logicalDevice(), &image_info, nullptr, &image_));

    VkMemoryRequirements mem_requirements;
    vkGetImageMemoryRequirements(device_->logicalDevice(), image_, &mem_requirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = mem_requirements.size;
    allocInfo.memoryTypeIndex = device_->get_memory_type(mem_requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VK_CHECK_RESULT(vkAllocateMemory(device_->logicalDevice(), &allocInfo, nullptr, &image_memory_));

    VK_CHECK_RESULT(vkBindImageMemory(device_->logicalDevice(), image_, image_memory_, 0));

    load_cpu_data(image);

    if (mip_levels_ > 1) {
        generate_mipmaps();
    }
}

void VulkanTexture::load_cpu_data(Image *image) {
    VulkanBuffer staging_buffer(device_, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                image->size_in_bytes(), image->pixel_ptr());

    VulkanDriver::instance().copy_image(&staging_buffer, this);
}

void VulkanTexture::generate_mipmaps() {
    VkFormatProperties format_properties;
    vkGetPhysicalDeviceFormatProperties(device_->physicalDevice(), image_format_, &format_properties);

    if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
        throw std::runtime_error("texture image format does not support linear blitting!");
    }

    VkCommandBuffer command_buffer = VulkanDriver::instance().begin_one_time_command_buffer();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image_;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    int32_t mip_width = res_.x;
    int32_t mip_height = res_.y;

    for (uint32_t i = 1; i < mip_levels_; i++) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mip_width, mip_height, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mip_width > 1 ? mip_width / 2 : 1, mip_height > 1 ? mip_height / 2 : 1, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(command_buffer, image_, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        if (mip_width > 1) mip_width /= 2;
        if (mip_height > 1) mip_height /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mip_levels_ - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    VulkanDriver::instance().end_one_time_command_buffer(command_buffer);
}

VulkanTexture::~VulkanTexture() {
    if (image_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_->logicalDevice(), image_memory_, nullptr);
    }
    if (image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device_->logicalDevice(), image_, nullptr);
    }
    if (image_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_->logicalDevice(), image_view_, nullptr);
    }
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_->logicalDevice(), sampler_, nullptr);
    }
}
size_t VulkanTexture::data_size() const noexcept { return 0; }
size_t VulkanTexture::data_alignment() const noexcept { return 0; }
size_t VulkanTexture::max_member_size() const noexcept { return sizeof(handle_ty); }

}// namespace ocarina