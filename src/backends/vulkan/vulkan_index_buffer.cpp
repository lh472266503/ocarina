#pragma once
#include "vulkan_index_buffer.h"
#include "vulkan_device.h"
namespace ocarina {

VulkanIndexBuffer::VulkanIndexBuffer(VulkanDevice *device, uint32_t size) : device_(device) {
    vulkan_buffer_ = device_->create_vulkan_buffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, size);
}

VulkanIndexBuffer::~VulkanIndexBuffer() {
    //release_index_buffer(device_, this);
}

void VulkanIndexBuffer::load_from_cpu(const void* cpuData, uint32_t byteOffset, uint32_t numBytes)
{
    vulkan_buffer_->load_from_cpu(cpuData, byteOffset, numBytes);
}

}// namespace ocarina