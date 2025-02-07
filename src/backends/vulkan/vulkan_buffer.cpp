#include "util.h"
#include "vulkan_device.h"
#include "vulkan_buffer.h"

namespace ocarina {

VulkanBuffer::VulkanBuffer(VulkanDevice *device) : device_(device) {
    
}

VulkanBuffer::~VulkanBuffer() {
    if (vulkan_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_->logicalDevice(), vulkan_buffer_, nullptr);
    }
}

void VulkanBuffer::load_from_cpu(VkCommandBuffer cmdbuf, const void* cpuData, uint32_t byteOffset,
    uint32_t numBytes)
{

}

}// namespace ocarina


