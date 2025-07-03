#pragma once
#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include "core/singleton.h"
#include "rhi/resources/buffer.h"
#include <vulkan/vulkan.h>
namespace ocarina {
class VulkanDevice;


class VulkanBuffer : public Buffer<> {
public:
    VulkanBuffer(VulkanDevice *device, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_property_flags, VkDeviceSize size, const void *data = nullptr);
    ~VulkanBuffer();
    void load_from_cpu(const void *cpuData, uint32_t byteOffset,
                     uint32_t numBytes);

    VkBuffer buffer_handle() const
    {
        return vulkan_buffer_;
    }

    //VkDeviceSize size() const
    //{
    //    return size_;
    //}

    VkResult bind(VkDeviceSize offset = 0);

    VkDescriptorBufferInfo get_descriptor_info(VkDeviceSize offset = 0) const
    {
        VkDescriptorBufferInfo info{};
        info.buffer = vulkan_buffer_;
        info.offset = offset;
        info.range = size_;
        return info;
    }
private:
    VulkanDevice *device_ = nullptr;
    VkBuffer vulkan_buffer_ = VK_NULL_HANDLE;
    VkBufferUsageFlags usage_ = {};
    VkMemoryPropertyFlags memory_property_flags_ = {};
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void *mapped_ = nullptr;
    //VkDeviceSize size_; //buffer size in bytes
    VkResult flush(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
};

}// namespace ocarina