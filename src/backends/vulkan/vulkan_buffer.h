#pragma once
#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include "core/singleton.h"
#include <vulkan/vulkan.h>
namespace ocarina {
class VulkanDevice;


class VulkanBuffer {
public:
    VulkanBuffer(VulkanDevice *device, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_property_flags, VkDeviceSize size, const void *data = nullptr);
    ~VulkanBuffer();
    void load_from_cpu(const void *cpuData, uint32_t byteOffset,
                     uint32_t numBytes);

    VkBuffer buffer_handle() const
    {
        return vulkan_buffer_;
    }
private:
    VulkanDevice *device_ = nullptr;
    VkBuffer vulkan_buffer_ = VK_NULL_HANDLE;
    VkBufferUsageFlags usage_ = {};
    VkMemoryPropertyFlags memory_property_flags_ = {};
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void *mapped_ = nullptr;

    VkResult flush(VkDeviceSize size = VK_WHOLE_SIZE, VkDeviceSize offset = 0);
};

class VulkanBufferManager : public Singleton<VulkanBufferManager>
{
public:
    friend class Singleton<VulkanBufferManager>;
    VulkanBuffer *create_vulkan_buffer(VulkanDevice *device, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags memory_property_flags, VkDeviceSize size, const void *initial_data);
    VulkanBuffer *get_vulkan_buffer(VkBuffer handle);
    void clear();

private:
    std::unordered_map<VkBuffer, VulkanBuffer*> vulkan_buffers;
};
}// namespace ocarina